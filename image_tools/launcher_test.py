#!/usr/bin/env python3
"""Behavioural tests for ``vllm-image launch``.

These cover what the launcher itself decides: how a group's ranks differ, what
the leader publishes, which role owns which precondition, and where the process
hands over to the engine. Policy resolution is covered against upstream by
``launcher_parity_test.py``, and recipe contents by ``launcher_data_test.py``.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from image_tools.launcher import cache_runtime, launch, probe_server, resolver
from image_tools.launcher.probes import ProbeConfig, write_config
from image_tools.upstream_runtime import (
    REPOSITORY_ROOT,
    UPSTREAM_RUNTIME,
    UpstreamTree,
)

# The launcher reads its policy data from VLLM_IMAGE_DATA_ROOT, whose default
# /opt/vllm-image/runtime exists only inside the image, so every case that
# resolves something points the resolver at the pinned upstream checkout
# instead. LIL_RUNTIME_CHECKOUT says where that checkout is, and where the lane
# declares it mandatory -- LIL_RUNTIME_REQUIRED, as the image_tools host lane of
# .github/workflows/recipes-pages.yaml does -- an absent tree fails rather than
# skipping. Both rules come from image_tools/upstream_runtime.py.
TREE = UpstreamTree("options.yaml")
upstream_only = TREE.gate

BASE_VALUES = {
    "port": 8888,
    "served-model-name": "qwen38-flash-next",
    "replicas": 1,
}

LEADER_ADDRESS = "qwen38-0.qwen38.svc"
LWS_ENV = {
    "LWS_WORKER_INDEX": "0",
    "LWS_GROUP_SIZE": "2",
    "LWS_LEADER_ADDRESS": LEADER_ADDRESS,
    "POD_IP": "10.0.0.7",
}

KV = launch.KvEvents("zmq", "tcp://*:5556", "tcp://*:5559")
TOPOLOGY_SINGLE = {
    "kind": "single",
    "nodes": 1,
    "rendezvous_port": None,
    "kv_events": None,
    "replica_port_base": None,
}


def resolved_plan(argv, values, environment=None, service=None):
    environment = environment or {}
    return resolver.ResolvedPlan(
        profile="qwen38-flash-next",
        hardware="gb10-roce",
        preset=None,
        recipe="qwen38-flash-next-gb10-tp2",
        values=values,
        origins={key: "recipe:x" for key in values},
        environment=environment,
        environment_origins={key: "recipe:x" for key in environment},
        argv=argv,
        passthrough=[],
        warnings=[],
        cache_service=service,
        draft_subfolder=None,
    )


def lws(kv_events=KV, nodes=2):
    return launch.Topology(launch.TOPOLOGY_LWS, nodes, 25000, kv_events, None)


def leader_payload(**overrides) -> str:
    config = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": "tcp://*:5556",
        "replay_endpoint": "tcp://*:5559",
        "topic": "kv@10.0.0.7:8888@qwen38-flash-next",
    }
    config.update(overrides)
    return json.dumps(config, separators=(",", ":"))


class RankTests(unittest.TestCase):
    def test_read_rank_requires_the_controller_environment(self):
        for missing in ("LWS_WORKER_INDEX", "LWS_GROUP_SIZE", "LWS_LEADER_ADDRESS"):
            environment = {key: value for key, value in LWS_ENV.items() if key != missing}
            with self.subTest(missing=missing), self.assertRaises(launch.ConfigError) as caught:
                launch.read_rank(lws(), environment)
            self.assertIn(missing, str(caught.exception))
            self.assertEqual(caught.exception.code, launch.EXIT_CONFIG)

    def test_read_rank_refuses_a_group_that_disagrees_with_the_recipe(self):
        with self.assertRaises(launch.ConfigError) as caught:
            launch.read_rank(lws(), dict(LWS_ENV, LWS_GROUP_SIZE="4"))
        self.assertIn("validated for 2", str(caught.exception))

    def test_read_rank_rejects_a_non_numeric_rank(self):
        with self.assertRaises(launch.ConfigError):
            launch.read_rank(lws(), dict(LWS_ENV, LWS_WORKER_INDEX="leader"))

    def test_read_rank_of_a_single_topology_needs_nothing(self):
        single = launch.Topology(launch.TOPOLOGY_SINGLE, 1, 25000, None, None)
        self.assertEqual(launch.read_rank(single, {}), launch.Rank(0, 1, None))

    def test_worker_rank_follows_the_leader(self):
        leader = launch.read_rank(lws(), LWS_ENV)
        worker = launch.read_rank(lws(), dict(LWS_ENV, LWS_WORKER_INDEX="1"))
        self.assertTrue(leader.is_leader)
        self.assertFalse(worker.is_leader)
        self.assertEqual(worker.leader, LEADER_ADDRESS)

    def test_leader_and_worker_argument_lists(self):
        leader = launch.topology_args(launch.Rank(0, 2, LEADER_ADDRESS), BASE_VALUES, KV, LWS_ENV)
        worker = launch.topology_args(launch.Rank(1, 2, LEADER_ADDRESS), BASE_VALUES, KV, LWS_ENV)
        self.assertEqual(worker, [
            "--nnodes", "2", "--node-rank", "1", "--master-addr", LEADER_ADDRESS, "--headless",
        ])
        self.assertEqual(leader[:6], [
            "--nnodes", "2", "--node-rank", "0", "--master-addr", "127.0.0.1",
        ])
        self.assertEqual(leader[6:8], ["--enable-scale-out", "--kv-events-config"])
        self.assertEqual(leader[8], leader_payload())
        self.assertNotIn("--headless", leader)
        # Publishing belongs to the leader alone: a second publisher would index
        # KV blocks under an address no router lookup can produce.
        self.assertNotIn("--kv-events-config", worker)

    def test_single_node_emits_no_rank_flags(self):
        self.assertEqual(
            launch.topology_args(launch.Rank(0, 1, None), BASE_VALUES, None, {}), []
        )
        published = launch.topology_args(launch.Rank(0, 1, None), BASE_VALUES, KV, LWS_ENV)
        self.assertEqual(published[:2], ["--enable-scale-out", "--kv-events-config"])


class KvEventTests(unittest.TestCase):
    def published(self, environment=LWS_ENV, **values) -> dict:
        args = launch.kv_events_args(
            KV, launch.Rank(0, 2, LEADER_ADDRESS), dict(BASE_VALUES, **values), environment
        )
        self.assertEqual(args[0], "--enable-scale-out")
        self.assertEqual(args[1], "--kv-events-config")
        return json.loads(args[2])

    def test_topic_names_the_pod_address_port_and_served_model(self):
        self.assertEqual(self.published(), json.loads(leader_payload()))

    def test_topic_follows_a_changed_port_and_model_name(self):
        published = self.published(**{"port": 8000, "served-model-name": "other"})
        self.assertEqual(published["topic"], "kv@10.0.0.7:8000@other")

    def test_topic_follows_the_pod_address(self):
        self.assertEqual(
            self.published(dict(LWS_ENV, POD_IP="10.0.0.9"))["topic"],
            "kv@10.0.0.9:8888@qwen38-flash-next",
        )

    def test_leader_without_pod_ip_is_a_configuration_failure(self):
        for environment in ({}, {"POD_IP": ""}):
            with self.subTest(environment=environment), self.assertRaises(launch.ConfigError) as caught:
                launch.kv_events_args(KV, launch.Rank(0, 2, LEADER_ADDRESS), BASE_VALUES, environment)
            self.assertIn("POD_IP", str(caught.exception))
            self.assertEqual(caught.exception.code, launch.EXIT_CONFIG)

    def test_worker_publishes_nothing_even_without_pod_ip(self):
        self.assertEqual(
            launch.kv_events_args(KV, launch.Rank(1, 2, LEADER_ADDRESS), BASE_VALUES, {}), []
        )

    def test_recipe_without_kv_events_publishes_nothing(self):
        self.assertEqual(
            launch.kv_events_args(None, launch.Rank(0, 2, LEADER_ADDRESS), BASE_VALUES, LWS_ENV), []
        )


class RecipeTests(unittest.TestCase):
    """The recipe document is one file per deployment, and the launcher reads it whole.

    ``launch`` is the launcher's own section; the checkpoint facts come from
    ``model`` and ``deployment``, so a published path is stated once. Whether a
    recipe syncs a checkpoint is the presence of ``deployment.model_path``, which is
    what separates the two TP=2 groups from the engine-download single-node server.
    """

    MODEL = {"model_id": "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "revision": "6" * 40}
    DEPLOYMENT = {
        "storage_root": "/models",
        "model_path": "/models/qwen38-flash-next",
        "storage_min_free_gib": 130,
        "download_workers": 8,
        "ignore_patterns": [],
        "required_files": ["tokenizer_config.json"],
    }

    def launch_section(self, **overrides):
        base = {
            "profile": "qwen38-flash-next",
            "hardware": "gb10-roce",
            "preset": None,
            "options": {"tensor-parallel-size": 2},
            "environment": {"HF_HOME": "/models"},
            "topology": {
                "kind": "lws",
                "nodes": 2,
                "rendezvous_port": 25000,
                "kv_events": {"publisher": "zmq", "endpoint": "tcp://*:5556", "replay_endpoint": "tcp://*:5559"},
                "replica_port_base": None,
            },
            "probe_port": 8890,
        }
        base.update(overrides)
        return base

    def recipe(self, *, model=None, deployment=None, section=None, **overrides):
        return {
            "model": dict(self.MODEL if model is None else model),
            "deployment": dict(self.DEPLOYMENT if deployment is None else deployment),
            "launch": self.launch_section(**overrides) if section is None else section,
        }

    def test_load_accepts_a_complete_recipe(self):
        recipe = launch.Recipe.from_dict("x", self.recipe())
        self.assertEqual(recipe.topology.kind, launch.TOPOLOGY_LWS)
        self.assertEqual(recipe.model_sync.min_free_gib, 130)
        self.assertEqual(recipe.model_sync.repo, self.MODEL["model_id"])
        self.assertEqual(recipe.model_sync.publish, Path("/models/qwen38-flash-next"))
        self.assertEqual(recipe.topology.kv_events.publisher, "zmq")

    def test_no_published_path_means_the_engine_downloads(self):
        deployment = {name: value for name, value in self.DEPLOYMENT.items() if name != "model_path"}
        recipe = launch.Recipe.from_dict("x", self.recipe(deployment=deployment))
        self.assertIsNone(recipe.model_sync)

    def test_load_recipe_reads_a_file_and_refuses_a_bad_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "x.yaml").write_text(yaml.safe_dump(self.recipe(), indent=2))
            self.assertEqual(launch.load_recipe("x", root).profile, "qwen38-flash-next")
            (root / "bare.yaml").write_text(yaml.safe_dump(self.MODEL, indent=2))
            with self.assertRaises(launch.ConfigError):
                launch.load_recipe("bare", root)
            with self.assertRaises(launch.ConfigError) as caught:
                launch.load_recipe("absent", root)
            self.assertIn("absent", str(caught.exception))

    def test_unknown_or_missing_keys_are_refused_with_the_key_named(self):
        for key in ("profile", "hardware", "probe_port", "topology"):
            broken = {name: value for name, value in self.launch_section().items() if name != key}
            with self.subTest(missing=key), self.assertRaises(launch.ConfigError) as caught:
                launch.Recipe.from_dict("x", self.recipe(section=broken))
            self.assertIn(key, str(caught.exception))
        with self.assertRaises(launch.ConfigError) as caught:
            launch.Recipe.from_dict("x", self.recipe(section={**self.launch_section(), "extra": 1}))
        self.assertIn("extra", str(caught.exception))

    def test_revision_must_be_a_full_commit_oid(self):
        for revision in ("main", "6" * 39, "6" * 41, "F" * 40):
            with self.subTest(revision=revision), self.assertRaises(launch.ConfigError):
                launch.Recipe.from_dict("x", self.recipe(model=dict(self.MODEL, revision=revision)))

    def test_single_topology_needs_no_leader_flags(self):
        recipe = launch.Recipe.from_dict("x", self.recipe(section=self.launch_section(topology=TOPOLOGY_SINGLE)))
        self.assertEqual(launch.read_rank(recipe.topology, {}), launch.Rank(0, 1, None))

    def test_worker_topology_below_two_nodes_is_refused(self):
        broken = dict(self.launch_section()["topology"], nodes=1)
        with self.assertRaises(launch.ConfigError):
            launch.Recipe.from_dict("x", self.recipe(section=self.launch_section(topology=broken)))

    def test_probe_port_must_be_a_port(self):
        for probe_port in (0, 70000, "8890"):
            with self.subTest(probe_port=probe_port), self.assertRaises(launch.ConfigError):
                launch.Recipe.from_dict("x", self.recipe(section=self.launch_section(probe_port=probe_port)))

    def test_environment_values_must_be_strings(self):
        with self.assertRaises(launch.ConfigError):
            launch.Recipe.from_dict("x", self.recipe(section=self.launch_section(environment={"OMP_NUM_THREADS": 2})))


class CacheRoleTests(unittest.TestCase):
    service = {
        "argv": ["/opt/venv/bin/lmcache", "server", "--port", "18000"],
        "environment": {"LMCACHE_PROFILE": "1"},
        "health_url": "http://127.0.0.1:18000/healthcheck",
        "shm_name": "lmcache_l1_pool_x",
        "shm_bytes": 4096,
        "startup_timeout": 5,
        "directories": ["/cache/lmcache/namespace"],
        "identity_required": False,
        "namespace": "/cache/lmcache/namespace",
        "stop_grace": 10,
        "prune_stale_tiers": False,
    }

    def test_roles_agree_on_the_cache_geometry(self):
        """The engine and the cache must resolve the same ports and pool."""
        recorded = launch.cache_record(dict(self.service), {"instance_id": 42})
        self.assertEqual(recorded["shm_name"], self.service["shm_name"])
        self.assertEqual(recorded["shm_bytes"], self.service["shm_bytes"])
        self.assertEqual(recorded["tier_dirs"], self.service["directories"])
        self.assertEqual(recorded["health_url"], self.service["health_url"])
        self.assertEqual(recorded["identity"], {"instance_id": 42})
        self.assertEqual(recorded["shm_root"], str(cache_runtime.SHM_ROOT))

    def test_validate_cache_waits_then_measures_the_arena(self):
        with (
            mock.patch.object(cache_runtime, "wait_ready") as wait,
            mock.patch.object(cache_runtime, "validate_arena") as arena,
        ):
            recorded = launch.validate_cache(dict(self.service), read=lambda url: {"instance_id": 42})
        wait.assert_called_once_with(self.service["health_url"], 5.0)
        arena.assert_called_once_with(self.service["shm_name"], 4096)
        self.assertEqual(recorded["identity"], {"instance_id": 42})

    def test_validate_cache_propagates_an_arena_refusal(self):
        def refuse(name, size, **_kwargs):
            raise launch.ConfigError(f"arena {name} is {size} bytes short; no pickle fallback")

        with (
            mock.patch.object(cache_runtime, "wait_ready"),
            mock.patch.object(cache_runtime, "validate_arena", side_effect=refuse),
        ):
            with self.assertRaises(launch.ConfigError) as caught:
                launch.validate_cache(dict(self.service), read=lambda url: {})
        self.assertIn("no pickle fallback", str(caught.exception))

    def test_child_environment_strips_managed_aliases_then_applies_resolved_values(self):
        resolved = resolved_plan(["x"], BASE_VALUES, {"NCCL_NET": "IB", "OMP_NUM_THREADS": "2"})
        aliases = frozenset({"TP", "TP_SIZE", "OMP_NUM_THREADS"})
        with (
            mock.patch.dict(os.environ, {"TP": "4", "OMP_NUM_THREADS": "1"}),
            mock.patch.object(launch.policy, "managed_environment_names", return_value=aliases),
        ):
            environment = launch.child_environment(resolved, {"EXTRA": "1"})
        self.assertEqual(environment["NCCL_NET"], "IB")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")
        self.assertEqual(environment["EXTRA"], "1")
        self.assertNotIn("TP", environment)
        self.assertIn("site-packages", environment["PYTHONPATH"])
        self.assertIn("/opt/nccl/lib", environment["LD_LIBRARY_PATH"])

    def test_exec_command_uses_the_real_interpreter(self):
        with mock.patch.object(launch.os, "execve") as execve:
            with self.assertRaises(launch.ConfigError):
                launch.exec_command([launch.VENV_PYTHON, "-c", "pass"], {"A": "1"})
        self.assertEqual(execve.call_args.args[0], launch.INTERPRETER)
        self.assertEqual(execve.call_args.args[2], {"A": "1"})

    def test_exec_command_leaves_a_console_script_alone(self):
        with mock.patch.object(launch.os, "execve") as execve:
            with self.assertRaises(launch.ConfigError):
                launch.exec_command(["/opt/venv/bin/lmcache", "server"], {})
        self.assertEqual(execve.call_args.args[0], "/opt/venv/bin/lmcache")


class SyncTests(unittest.TestCase):
    def recipe(self, publish: Path | None) -> launch.Recipe:
        deployment = {
            "storage_root": "/models",
            "storage_min_free_gib": 130,
            "download_workers": 8,
            "ignore_patterns": ["assets/*"],
            "required_files": ["tokenizer_config.json"],
        }
        if publish is not None:
            deployment["model_path"] = str(publish)
        return launch.Recipe.from_dict("x", {
            "model": {"model_id": "org/model", "revision": "6" * 40},
            "deployment": deployment,
            "launch": {
                "profile": "qwen38-flash-next",
                "hardware": "gb10-roce",
                "preset": None,
                "options": {},
                "environment": {},
                "topology": TOPOLOGY_SINGLE,
                "probe_port": 8890,
            },
        })

    def test_sync_refuses_to_exec_without_a_published_index(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe = self.recipe(Path(directory) / "model")
            with mock.patch("image_tools.vllm_image.sync_model", return_value=1) as sync:
                with self.assertRaises(launch.ConfigError) as caught:
                    launch.publish_model(recipe)
            self.assertIn("model.safetensors.index.json", str(caught.exception))
            sync.assert_called_once()
            self.assertEqual(sync.call_args.kwargs["ignore"], ["assets/*"])

    def test_sync_passes_the_recipe_checkpoint_to_the_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            publish = Path(directory) / "model"
            publish.mkdir()
            (publish / "model.safetensors.index.json").write_text("{}")
            with mock.patch("image_tools.vllm_image.sync_model", return_value=3) as sync:
                launch.publish_model(self.recipe(publish))
            self.assertEqual(sync.call_args.kwargs["repo"], "org/model")
            self.assertEqual(sync.call_args.kwargs["min_free_bytes"], 130 * 1024**3)

    def test_sync_without_a_published_path_is_refused(self):
        with self.assertRaises(launch.ConfigError) as caught:
            launch.publish_model(self.recipe(None))
        self.assertIn("deployment.model_path", str(caught.exception))
        self.assertEqual(caught.exception.code, launch.EXIT_MODEL)


class EntrypointTests(unittest.TestCase):
    def invoke(self, *args):
        environment = dict(
            os.environ,
            PYTHONPATH=str(REPOSITORY_ROOT),
            VLLM_IMAGE_DATA_ROOT=str(UPSTREAM_RUNTIME),
        )
        return subprocess.run(
            [sys.executable, "-m", "image_tools.vllm_image", *args],
            capture_output=True,
            text=True,
            env=environment,
        )

    @upstream_only
    def test_launcher_options_select_the_launcher(self):
        result = self.invoke(
            "launch", "--profile", "qwen38-flash-next", "--hardware", "native", "--print-config"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["role"], "server")
        self.assertEqual(record["selection"]["profile"], "qwen38-flash-next")
        self.assertEqual(record["topology"]["kind"], "single")

    def test_recipe_flags_cannot_be_mixed_with_a_profile_selection(self):
        result = self.invoke("launch", "--recipe", "missing", "--hardware", "native", "--print-config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--hardware", result.stderr)

    def test_profile_selection_requires_hardware(self):
        result = self.invoke("launch", "--profile", "qwen38-flash-next", "--print-config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--hardware", result.stderr)

    def test_entrypoint_passes_a_vllm_command_through(self):
        with mock.patch.object(launch.os, "execve") as execve:
            launch.entrypoint(["serve", "model", "--port", "8888"])
        interpreter, argv, environment = execve.call_args.args
        self.assertEqual(interpreter, launch.INTERPRETER)
        self.assertEqual(argv[:3], [launch.INTERPRETER, "-m", "vllm.entrypoints.cli.main"])
        self.assertEqual(argv[3:], ["serve", "model", "--port", "8888"])
        # The entrypoint names the interpreter directly, so it must supply what the
        # shell wrapper used to export.
        self.assertIn("site-packages", environment["PYTHONPATH"])
        self.assertIn("/opt/nccl/lib", environment["LD_LIBRARY_PATH"])

    def test_entrypoint_dispatches_launcher_options_to_the_launcher(self):
        with mock.patch.object(launch, "run", return_value=0) as run:
            self.assertEqual(launch.entrypoint(["--recipe", "x", "--print-config"]), 0)
        run.assert_called_once_with(["--recipe", "x", "--print-config"])

    def test_entrypoint_accepts_the_explicit_launch_command(self):
        with mock.patch.object(launch, "run", return_value=0) as run:
            self.assertEqual(launch.entrypoint(["launch", "--recipe", "x"]), 0)
        run.assert_called_once_with(["--recipe", "x"])

    def test_entrypoint_without_arguments_explains_the_choice(self):
        result = self.invoke("entrypoint")
        self.assertEqual(result.returncode, launch.EXIT_CONFIG)
        self.assertIn("--recipe", result.stderr)
        self.assertIn("--profile", result.stderr)

    @upstream_only
    def test_probe_role_serves_the_endpoints(self):
        # In-process, so the data root has to be pointed at the pinned policy tree
        # the way `invoke` points the subprocess at it: the default is the image's
        # /opt/vllm-image/runtime, which exists only inside the image.
        with (
            mock.patch.dict(os.environ, {"VLLM_IMAGE_DATA_ROOT": str(UPSTREAM_RUNTIME)}),
            mock.patch.object(probe_server, "serve", return_value=0) as serve,
        ):
            self.assertEqual(
                launch.run(
                    ["--profile", "qwen38-flash-next", "--hardware", "native", "--role", "probe"]
                ),
                0,
            )
        self.assertEqual(serve.call_args.args, (launch.DEFAULT_PROBE_PORT,))

    def test_probe_record_round_trips_for_the_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "launch.json"
            write_config(ProbeConfig(1, LEADER_ADDRESS, 8888, "m", None, {"image": "x"}), record)
            loaded = ProbeConfig.from_record(json.loads(record.read_text()))
            self.assertEqual(loaded.rank, 1)
            self.assertEqual(loaded.leader, LEADER_ADDRESS)
            self.assertEqual(
                launch.topology_args(launch.Rank(1, 2, LEADER_ADDRESS), BASE_VALUES, None, LWS_ENV)[-1],
                "--headless",
            )

    @upstream_only
    def test_cache_role_without_a_cache_is_refused(self):
        result = self.invoke(
            "launch", "--profile", "qwen38-flash-next", "--hardware", "native", "--role", "cache",
        )
        self.assertEqual(result.returncode, launch.EXIT_CACHE)
        self.assertIn("--cache-mode", result.stderr)


@upstream_only
class ResolutionTests(unittest.TestCase):
    """The launcher must add rank and probe wiring without disturbing policy."""

    def resolve(self, extra=()):
        return resolver.resolve(
            "qwen38-flash-next",
            "native",
            argv=list(extra),
            env={},
            cli_env={},
            runtime_identity="deadbeef",
            root=UPSTREAM_RUNTIME,
        )

    def test_launcher_arguments_are_appended_after_the_resolved_argv(self):
        resolved = self.resolve()
        argv = list(resolved.argv) + launch.topology_args(
            launch.Rank(1, 2, LEADER_ADDRESS), resolved.values, KV, LWS_ENV
        ) + ["--middleware", launch.PROBE_MIDDLEWARE]
        self.assertEqual(argv[: len(resolved.argv)], resolved.argv)
        self.assertEqual(argv[-2:], ["--middleware", launch.PROBE_MIDDLEWARE])

    def test_native_arguments_reach_vllm_unvalidated(self):
        resolved = self.resolve(("--scheduler-preemption-mode", "recompute"))
        self.assertEqual(resolved.argv[-2:], ["--scheduler-preemption-mode", "recompute"])
        self.assertTrue(any("Unmanaged native options" in warning for warning in resolved.warnings))

    def test_recipe_layer_overrides_the_profile_and_reports_its_source(self):
        resolved = resolver.resolve(
            "qwen38-flash-next",
            "native",
            recipe_layer={
                "options": {"max-num-seqs": 8},
                "environment": {"HF_HOME": "/models"},
                "source": "recipe:test",
            },
            env={},
            cli_env={},
            runtime_identity="deadbeef",
            root=UPSTREAM_RUNTIME,
        )
        self.assertEqual(resolved.values["max-num-seqs"], 8)
        self.assertEqual(resolved.origins["max-num-seqs"], "recipe:test")
        self.assertEqual(resolved.environment["HF_HOME"], "/models")
        self.assertEqual(resolved.environment_origins["HF_HOME"], "recipe:test")


if __name__ == "__main__":
    unittest.main()
