#!/usr/bin/env python3
"""Parity tests for the in-image policy resolver against the pinned launcher.

``image_tools/launcher/resolver.py`` is a port of ``runtime/launcher.py`` and
the policy half of ``runtime/cache.py`` from
``local-inference-lab/blackwell-llm-docker`` at the commit the vllmb12x profile
pins in ``profiles/vllmb12x/profile.json``. This module is the gate that keeps
it a port: for identical inputs both resolvers must agree on the command, the
settings with their sources, the environment with its sources, the warnings
and the cache service. A refresh of the upstream pin that changes policy fails
here until the port follows.

Both resolvers run in this interpreter, so upstream's ``installed_source`` and
``installed_b12x_mxfp8_moe`` answer identically on both sides, and no test
relies on the ambient process environment: ``env`` is always passed explicitly.

The pinned tree is read from ``LIL_RUNTIME_CHECKOUT``, which defaults to the
developer clone, and the commit it is at is checked against the profile: a
checkout that is not at the pin fails, because comparing against some other
tree measures nothing. An *absent* checkout fails too wherever the lane
declares the pin mandatory -- ``LIL_RUNTIME_REQUIRED=1``, which is what the
launcher-tests job of ``.github/workflows/recipes-pages.yaml`` sets -- since a
gate that has only ever passed by skipping never caught the drift it exists to
catch. A workstation run without that switch is the one case allowed to skip,
with a message naming the path and the commit to clone. Every comparison set
also asserts that the cases it planned ran, and ``tearDownModule`` prints the
number that was actually compared.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from image_tools.launcher import ConfigError
from image_tools.launcher import resolver

DEFAULT_CHECKOUT = Path(
    "/home/naadir/go/src/github.com/local-inference-lab/blackwell-llm-docker"
)
UPSTREAM_CHECKOUT = Path(os.environ.get("LIL_RUNTIME_CHECKOUT") or DEFAULT_CHECKOUT)
UPSTREAM_RUNTIME = UPSTREAM_CHECKOUT / "runtime"
UPSTREAM_LAUNCHER = UPSTREAM_RUNTIME / "launcher.py"
# The pin belongs to the profile, not to this file:
# scripts/refresh-vllmb12x.py moves it together with the vLLM and B12X pins,
# and a duplicate literal here would quietly compare the port against a tree
# that nothing else claims is pinned.
PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "vllmb12x" / "profile.json"
LIL_RUNTIME = json.loads(PROFILE.read_text())["sources"]["lil_runtime"]
UPSTREAM_REMOTE = LIL_RUNTIME["remote"]
UPSTREAM_COMMIT = LIL_RUNTIME["commit"]
# Where the absence of the tree is a defect rather than a convenience: the CI
# lane is the only place a pinned upstream can drift out from under the port.
REQUIRED = os.environ.get("LIL_RUNTIME_REQUIRED", "").strip().lower() in ("1", "true", "yes")


def _revision() -> str | None:
    """HEAD of the checkout, or None when it cannot be read."""
    result = subprocess.run(
        ["git", "-C", str(UPSTREAM_CHECKOUT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


# Only a tree that is there is asked for its commit: an absent checkout is
# reported by its own reason, not as a mismatch. A tree whose commit cannot be
# read is unverified, and an unverified tree is what this gate exists to rule
# out, so a missing ``git`` is a failure rather than a pass.
UPSTREAM_PRESENT = UPSTREAM_LAUNCHER.is_file()
UPSTREAM_HEAD = _revision() if UPSTREAM_PRESENT else None
UPSTREAM_MISMATCH = UPSTREAM_PRESENT and UPSTREAM_HEAD != UPSTREAM_COMMIT
UPSTREAM_AVAILABLE = UPSTREAM_PRESENT and not UPSTREAM_MISMATCH

SKIP_REASON = (
    f"pinned upstream checkout is absent: {UPSTREAM_CHECKOUT} @ {UPSTREAM_COMMIT} "
    f"(clone {UPSTREAM_REMOTE} at that commit, or point LIL_RUNTIME_CHECKOUT at it)"
)
REQUIRED_REASON = (
    f"{SKIP_REASON}; LIL_RUNTIME_REQUIRED says this lane must have it, so an "
    "absent checkout is a broken gate and not a passing one"
)
MISMATCH_REASON = (
    f"{UPSTREAM_CHECKOUT} is at "
    f"{UPSTREAM_HEAD or 'an unreadable commit'}, not the pinned {UPSTREAM_COMMIT} "
    "from profiles/vllmb12x/profile.json: comparing the port against a tree that "
    "is not the pin proves nothing"
)

# What the gate actually compared, so a run that compared nothing cannot print
# the same OK as a run that compared the whole matrix.
COMPARISONS = {"count": 0}


def upstream_barrier() -> BaseException:
    """What the gate raises when the pinned tree is not usable.

    An ``AssertionError`` wherever the checkout is at the wrong commit or the
    lane declared it mandatory -- both mean the comparison silently stopped
    happening -- and a ``SkipTest`` only for the developer who genuinely has not
    cloned upstream yet.
    """
    if UPSTREAM_MISMATCH:
        return AssertionError(MISMATCH_REASON)
    if REQUIRED:
        return AssertionError(REQUIRED_REASON)
    return unittest.SkipTest(SKIP_REASON)


def upstream_only(item):
    """Gate one case, or a whole class of them, on the pinned checkout."""
    if UPSTREAM_AVAILABLE:
        return item
    if isinstance(item, type):

        def setUpClass(cls):
            raise upstream_barrier()

        item.setUpClass = classmethod(setUpClass)
        return item

    @functools.wraps(item)
    def gate(self, *args, **kwargs):
        raise upstream_barrier()

    return gate


if UPSTREAM_AVAILABLE:  # depends on the pinned checkout, not on this repository
    if str(UPSTREAM_CHECKOUT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_CHECKOUT))
    from runtime import ConfigError as UpstreamConfigError
    from runtime import launcher as upstream

# The keys upstream's LaunchPlan.public() produces. Our record adds `preset`
# and `recipe`, which upstream has no equivalent for, so they are not compared.
SHARED_KEYS = (
    "profile",
    "hardware",
    "argv",
    "settings",
    "environment",
    "warnings",
    "cache_service",
)
CACHE_SERVICE_KEYS = {
    "argv",
    "environment",
    "health_url",
    "shm_name",
    "shm_bytes",
    "startup_timeout",
    "directories",
    "identity_required",
    "namespace",
    "stop_grace",
    "prune_stale_tiers",
}
EDITED_CASES = (
    ("max-num-seqs", ["--max-num-seqs", "16"]),
    ("tensor-parallel-size", ["--tensor-parallel-size", "2"]),
    ("mode off", ["--mode", "off"]),
    ("cache-mode", ["--cache-mode", "lmcache"]),
    ("max-model-len", ["--max-model-len", "524288"]),
    ("draft-tokens", ["--draft-tokens", "7"]),
    ("unmanaged native", ["--disable-log-requests", "always"]),
    ("dotted json field", ["--additional-config.layered_forward_pass_params=3"]),
    ("tp aliases", {"config": {"environment": {"TP": "2", "TP_SIZE": "2"}}}),
    (
        "tp alias conflict",
        {"config": {"environment": {"TP": "2", "TP_SIZE": "4"}}},
    ),
)


def inventory():
    """The real upstream inventory, read from the pinned data files."""
    profiles = [
        data["id"]
        for data in (
            resolver.read_yaml(path)
            for path in sorted((UPSTREAM_RUNTIME / "profiles").glob("*.yaml"))
        )
        if data["kind"] == "model"
    ]
    hardware = [
        data["id"]
        for data in (
            resolver.read_yaml(path)
            for path in sorted((UPSTREAM_RUNTIME / "hardware").glob("*.yaml"))
        )
    ]
    presets = sorted(resolver.deployment_presets(root=UPSTREAM_RUNTIME))
    return profiles, hardware, presets


class ParityCase:
    """One comparison: the same arguments into both resolvers, same record out."""

    def __init__(self, label, profile, hardware, **kwargs):
        self.label = label
        self.profile = profile
        self.hardware = hardware
        self.kwargs = kwargs

    def __str__(self):
        return self.label


class ParityTest(unittest.TestCase):
    """Shared harness; every subclass runs against the pinned checkout only."""

    @classmethod
    def setUpClass(cls):
        if not UPSTREAM_AVAILABLE:
            raise upstream_barrier()
        cls.profiles, cls.hardware, cls.presets = inventory()

    def setUp(self):
        self.compared = 0

    def theirs(self, case):
        kwargs = {"env": {}, "vllm_environment": frozenset(), **case.kwargs}
        try:
            plan = upstream.resolve(case.profile, case.hardware, **kwargs)
        except UpstreamConfigError as error:
            return "error", str(error)
        return "public", plan.public()

    def ours(self, case):
        kwargs = {"env": {}, "vllm_environment": frozenset(), **case.kwargs}
        try:
            plan = resolver.resolve(
                case.profile, case.hardware, root=UPSTREAM_RUNTIME, **kwargs
            )
        except ConfigError as error:
            return "error", str(error)
        return "public", resolver.public(plan)

    def assert_parity(self, case):
        self.compared += 1
        COMPARISONS["count"] += 1
        expected, actual = self.theirs(case), self.ours(case)
        if expected[0] == "error" or actual[0] == "error":
            self.assertEqual(
                expected, actual, f"{case.label}: the resolvers disagree on a refusal"
            )
            return
        self.assertEqual(
            "public", actual[0], f"{case.label}: ours refused where upstream did not"
        )
        for key in SHARED_KEYS:
            self.assertEqual(
                expected[1][key],
                actual[1][key],
                f"{case.label}: {key} differs from the pinned upstream resolver",
            )
        # Our record is upstream's plus the two data layers it cannot name.
        self.assertEqual(
            {"preset", "recipe"}, set(actual[1]) - set(expected[1]), case.label
        )

    def assert_ran(self, planned):
        self.assertEqual(
            len(planned),
            self.compared,
            "every planned comparison must actually execute",
        )


class MatrixParityTest(ParityTest):
    def profile_hardware_cases(self):
        return [
            ParityCase(f"{profile}/{kind}", profile, kind)
            for profile in self.profiles
            for kind in self.hardware
        ]

    def preset_cases(self, argv=None):
        cases = []
        for name in self.presets:
            deployment = upstream.deployment_preset(name)
            cases.append(
                ParityCase(
                    f"preset:{name}" + (" edited" if argv else ""),
                    deployment["profile"],
                    deployment["hardware"],
                    preset=name,
                    **({"argv": argv} if argv else {}),
                )
            )
        return cases

    def edited_cases(self):
        return [
            ParityCase(f"{profile} {label}", profile, "native", **extra)
            if isinstance(extra, dict)
            else ParityCase(
                f"{profile} {label}", profile, "native", argv=extra
            )
            for profile in self.profiles
            for label, extra in EDITED_CASES
        ]

    def identity_cases(self):
        return [
            ParityCase(
                f"{profile} runtime identity",
                profile,
                "native",
                runtime_identity="0" * 64,
            )
            for profile in self.profiles
        ]

    def test_every_profile_and_hardware_pair(self):
        cases = self.profile_hardware_cases()
        self.assertEqual(len(self.profiles) * len(self.hardware), len(cases))
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_every_preset_with_its_own_hardware(self):
        cases = self.preset_cases()
        self.assertEqual(len(self.presets), len(cases))
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_every_preset_with_an_edited_option(self):
        cases = self.preset_cases(argv=["--max-num-seqs", "16"])
        self.assertEqual(len(self.presets), len(cases))
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_edited_options_and_environment_aliases(self):
        cases = self.edited_cases()
        self.assertEqual(len(self.profiles) * len(EDITED_CASES), len(cases))
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_explicit_runtime_identity(self):
        cases = self.identity_cases()
        self.assertEqual(len(self.profiles), len(cases))
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_the_matrix_is_not_vacuous(self):
        planned = (
            self.profile_hardware_cases()
            + self.preset_cases()
            + self.preset_cases(argv=["--max-num-seqs", "16"])
            + self.edited_cases()
            + self.identity_cases()
        )
        self.assertGreaterEqual(len(planned), 30)

    def test_public_record_keeps_upstream_key_names(self):
        plan = resolver.resolve(
            "glm53-flash", root=UPSTREAM_RUNTIME, env={}, vllm_environment=frozenset()
        )
        record = resolver.public(plan)
        self.assertEqual(1, record["schema_version"])
        self.assertEqual("implemented", record["status"])
        self.assertEqual(
            "CPU configuration tests only; no GPU performance claim",
            record["qualification"],
        )
        self.assertEqual("glm53-flash", record["profile"])
        self.assertEqual("native", record["hardware"])
        self.assertIsNone(record["cache_service"])
        self.assertIsNone(record["preset"])
        self.assertIsNone(record["recipe"])
        cached = resolver.public(
            resolver.resolve(
                "glm53-flash",
                root=UPSTREAM_RUNTIME,
                env={"CACHE_MODE": "lmcache"},
                vllm_environment=frozenset(),
            )
        )
        self.assertEqual(CACHE_SERVICE_KEYS, set(cached["cache_service"]))

    def test_secrets_are_redacted_identically(self):
        token = "hf_topsecretvalue"
        url = "https://user:pa55word@huggingface.co/private"
        case = ParityCase(
            "redaction",
            "glm53-flash",
            "native",
            argv=[
                "--api-key",
                token,
                "--served-model-name",
                url,
                "--compilation-config",
                '{"api_key":"' + token + '","level":3}',
            ],
        )
        self.assert_parity(case)
        record = self.ours(case)[1]
        serialized = json.dumps(record)
        self.assertNotIn(token, serialized)
        self.assertNotIn("pa55word", serialized)
        # The value following a secret flag, the credentials inside a URL and a
        # secret key inside a JSON argument all disappear, exactly as upstream
        # removes them.
        self.assertIn("<redacted>", record["argv"])
        self.assertEqual(
            {"api_key": "<redacted>", "level": 3},
            json.loads(
                record["argv"][record["argv"].index("--compilation-config") + 1]
            ),
        )
        self.assertEqual(
            "<redacted>",
            record["settings"]["compilation-config"]["value"]["api_key"],
        )
        self.assertEqual(
            "https://<redacted>@huggingface.co/private",
            record["settings"]["served-model-name"]["value"],
        )


class FallbackParityTest(ParityTest):
    def fallback_preset(self):
        for name in self.presets:
            fallback = resolver.deployment_presets(root=UPSTREAM_RUNTIME)[name].get(
                "vllm_fallback"
            )
            if fallback:
                return name, fallback
        self.fail("the pinned presets must include a vllm_fallback path")

    def test_installed_vllm_decides_the_fallback_both_ways(self):
        name, fallback = self.fallback_preset()
        deployment = upstream.deployment_preset(name)
        required = fallback["requires_environment"]
        cases = [
            ParityCase(
                f"{name} with {label}",
                deployment["profile"],
                deployment["hardware"],
                preset=name,
                vllm_environment=names,
            )
            for label, names in (
                ("every required name", frozenset(required)),
                ("none of the required names", frozenset()),
                ("only the first required name", frozenset(required[:1])),
            )
        ]
        for case in cases:
            self.assert_parity(case)
        self.assert_ran(cases)

    def test_fallback_reports_the_missing_names_in_the_source(self):
        name, fallback = self.fallback_preset()
        deployment = upstream.deployment_preset(name)
        required = fallback["requires_environment"]
        plan = resolver.resolve(
            deployment["profile"],
            deployment["hardware"],
            root=UPSTREAM_RUNTIME,
            preset=name,
            env={},
            vllm_environment=frozenset(required[1:]),
        )
        sources = {
            item["source"] for item in resolver.public(plan)["settings"].values()
        }
        reported = [
            source for source in sources if "(fallback: installed vLLM lacks " in source
        ]
        self.assertTrue(
            reported, f"a preset missing vLLM features must say so: {sources}"
        )
        self.assertIn(fallback["requires_environment"][0], reported[0])
        self.assertNotIn(required[1], reported[0])
        satisfied = resolver.public(
            resolver.resolve(
                deployment["profile"],
                deployment["hardware"],
                root=UPSTREAM_RUNTIME,
                preset=name,
                env={},
                vllm_environment=frozenset(required),
            )
        )
        self.assertFalse(
            any(
                "(fallback:" in item["source"]
                for item in satisfied["settings"].values()
            ),
            "an installed vLLM that satisfies the preset must not report a fallback",
        )


class RefusalParityTest(ParityTest):
    """One negative case per validation rule class, with upstream's own message."""

    def assert_refused(self, expected, case):
        self.compared += 1
        COMPARISONS["count"] += 1
        actual = self.ours(case)
        self.assertEqual(("error", expected), actual, case.label)
        self.assertEqual(self.theirs(case), actual, f"{case.label}: upstream diverged")

    def test_positive_integers(self):
        self.assert_refused(
            "max-num-seqs must be positive",
            ParityCase(
                "max-num-seqs", "glm53-flash", "native", argv=["--max-num-seqs", "0"]
            ),
        )

    def test_dcp_must_divide_tp(self):
        self.assert_refused(
            "DCP must divide TP",
            ParityCase(
                "dcp",
                "qwen38-flash-next",
                "native",
                config={"environment": {"TP": "4", "DCP": "3"}},
            ),
        )

    def test_kv_cache_dtype_set(self):
        self.assert_refused(
            "glm53-flash supports target KV settings ['fp8', 'fp8_e4m3', "
            "'nvfp4_ds_mla']; other precisions require separate qualification",
            ParityCase(
                "kv dtype", "glm53-flash", "native", argv=["--kv-cache-dtype", "auto"]
            ),
        )

    def test_capture_size_monotonicity(self):
        self.assert_refused(
            "Capture sizes must be positive, strictly increasing and within the "
            "capture cap",
            ParityCase(
                "capture sizes",
                "ds41-flash",
                "native",
                argv=[
                    "--max-cudagraph-capture-size",
                    "64",
                    "--cudagraph-capture-sizes",
                    "8",
                    "4",
                ],
            ),
        )

    def test_glm_page_divisibility(self):
        self.assert_refused(
            "Recurrent page spacing must divide, or be a multiple of, the target page",
            ParityCase(
                "glm pages",
                "glm53-flash",
                "native",
                argv=["--target-page-size", "128", "--recurrent-page-size", "192"],
            ),
        )

    def test_pipeline_parallel_size(self):
        self.assert_refused(
            "These profiles support single-node tensor parallelism, not pipeline "
            "parallelism",
            ParityCase(
                "pp", "ds4-flash", "native", argv=["--pipeline-parallel-size", "2"]
            ),
        )

    def test_extra_vllm_args(self):
        self.assert_refused(
            "EXTRA_VLLM_ARGS is ambiguous shell text; pass native CLI arguments "
            "after --",
            ParityCase(
                "extra args",
                "ds4-vision",
                "native",
                config={"environment": {"EXTRA_VLLM_ARGS": "--foo"}},
            ),
        )

    def test_obsolete_aliases(self):
        for obsolete in (
            "BACKEND",
            "MODE",
            "SPEC_MODE",
            "DS4_OMP_NUM_THREADS",
            "DS4_MAX_CUDAGRAPH_CAPTURE_SIZE",
            "DS4_CUDAGRAPH_CAPTURE_SIZES",
        ):
            with self.subTest(alias=obsolete):
                self.assert_refused(
                    f"{obsolete} belongs to a compatibility wrapper; use the "
                    "documented native option or canonical environment variable",
                    ParityCase(
                        f"obsolete {obsolete}",
                        "mimo26-flash",
                        "native",
                        config={"environment": {obsolete: "b12x"}},
                    ),
                )

    def test_fairness_engine(self):
        self.assert_refused(
            "FAIRNESS_ENGINE must be none or compute_share; micro_slicing is "
            "unsupported",
            ParityCase(
                "fairness",
                "ds41-flash",
                "native",
                config={"environment": {"FAIRNESS_ENGINE": "micro_slicing"}},
            ),
        )

    def test_managed_option_bypass_is_refused(self):
        self.assert_refused(
            "--kv-transfer-config requires the external-cache/config-file "
            "integration; it cannot bypass profile validation",
            ParityCase(
                "bypass",
                "glm53-flash",
                "native",
                argv=["--kv-transfer-config", "{}"],
            ),
        )


@upstream_only
class RecipeLayerTest(unittest.TestCase):
    """The one layer this port adds: our accepted values and environment."""

    @classmethod
    def setUpClass(cls):
        presets = resolver.deployment_presets(root=UPSTREAM_RUNTIME)
        # A vLLM that declares every name any preset needs, so no preset takes
        # its vllm_fallback path and the values under test are the declared ones.
        cls.satisfied_vllm = frozenset(
            name for preset in presets.values() for name in preset["environment"]
        )

    def resolve(self, profile="qwen38-flash-next", hardware="native", **kwargs):
        return resolver.resolve(
            profile,
            hardware,
            root=UPSTREAM_RUNTIME,
            env={},
            vllm_environment=self.satisfied_vllm,
            **kwargs,
        )

    def deployment(self, name):
        return resolver.deployment_preset(name, root=UPSTREAM_RUNTIME)

    def test_recipe_outranks_the_preset_it_names(self):
        name = "glm53-spark-tp2"
        plan = self.resolve(
            profile="glm53-flash",
            hardware="rtx-pro-6000-pcie",
            preset=name,
            recipe_layer={
                "source": "recipe:glm53-spark-tp2-site",
                "options": {"max-num-seqs": 24},
                "environment": {"NCCL_MIN_NCHANNELS": "4"},
            },
        )
        record = resolver.public(plan)
        self.assertEqual(name, record["preset"])
        self.assertEqual("glm53-spark-tp2-site", record["recipe"])
        self.assertEqual(
            {"value": 24, "source": "recipe:glm53-spark-tp2-site"},
            record["settings"]["max-num-seqs"],
        )
        self.assertEqual(
            {"value": "4", "source": "recipe:glm53-spark-tp2-site"},
            record["environment"]["NCCL_MIN_NCHANNELS"],
        )
        self.assertEqual("24", plan.argv[plan.argv.index("--max-num-seqs") + 1])

    def test_operator_layers_outrank_the_recipe(self):
        plan = self.resolve(
            recipe_layer={
                "source": "recipe:qwen38-flash-next-gb10-tp2",
                "options": {
                    "max-num-seqs": 24,
                    "tensor-parallel-size": 2,
                    "draft-model": "someone/recipe-drafter",
                },
            },
            config={"options": {"tensor-parallel-size": 4}},
            cli_env={"DRAFT_MODEL": "someone/cli-drafter"},
            argv=["--max-num-seqs", "2"],
        )
        record = resolver.public(plan)
        self.assertEqual(
            {"value": 2, "source": "cli"}, record["settings"]["max-num-seqs"]
        )
        self.assertEqual(
            {"value": 4, "source": "settings:options"},
            record["settings"]["tensor-parallel-size"],
        )
        self.assertEqual(
            "cli:environment:DRAFT_MODEL", record["settings"]["draft-model"]["source"]
        )

    def test_recipe_pinned_kv_bytes_shrinks_like_a_preset(self):
        name = "glm53-spark-tp2"
        deployment = self.deployment(name)
        qualified = deployment["options"]["kv-cache-memory-bytes"]
        slots = deployment["options"]["max-num-seqs"]
        recipe = self.resolve(
            profile=deployment["profile"],
            hardware=deployment["hardware"],
            preset=name,
            recipe_layer={
                "source": f"recipe:{name}-gb10",
                "options": {
                    "kv-cache-memory-bytes": qualified,
                    "max-num-seqs": slots + 2,
                    "cache-mode": "lmcache",
                },
            },
        )
        self.assertEqual(
            {
                "value": qualified
                - 2 * deployment["kv_bytes_per_extra_slot"]
                - deployment["kv_bytes_for_external_cache"],
                "source": "derived:2 request slots above the preset; external cache buffers",
            },
            resolver.public(recipe)["settings"]["kv-cache-memory-bytes"],
        )
        # The same request-slot count chosen by an operator is left alone.
        operator = self.resolve(
            profile=deployment["profile"],
            hardware=deployment["hardware"],
            preset=name,
            argv=[
                "--kv-cache-memory-bytes",
                str(qualified),
                "--max-num-seqs",
                str(slots + 2),
            ],
        )
        self.assertEqual(
            qualified,
            resolver.public(operator)["settings"]["kv-cache-memory-bytes"]["value"],
        )

    def test_recipe_qsa_interleave_survives_the_derivation(self):
        shared = {
            "tensor-parallel-size": 4,
            "decode-context-parallel-size": 2,
        }
        derived = self.resolve(
            recipe_layer={
                "source": "recipe:qwen38-flash-next-gb10-tp2",
                "options": shared,
            }
        )
        self.assertEqual(4, derived.values["cp-kv-cache-interleave-size"])
        self.assertEqual(
            "derived:QSA DCP interleave",
            derived.origins["cp-kv-cache-interleave-size"],
        )
        chosen = self.resolve(
            recipe_layer={
                "source": "recipe:qwen38-flash-next-gb10-tp2",
                "options": {
                    **shared,
                    "cp-kv-cache-interleave-size": 8,
                    "dcp-kv-cache-interleave-size": 8,
                },
            }
        )
        self.assertEqual(8, chosen.values["cp-kv-cache-interleave-size"])
        self.assertEqual(
            "recipe:qwen38-flash-next-gb10-tp2",
            chosen.origins["cp-kv-cache-interleave-size"],
        )

    def test_recipe_capture_sizes_are_extended_like_a_preset_list(self):
        plan = self.resolve(
            recipe_layer={
                "source": "recipe:qwen38-flash-next-gb10-tp2",
                "options": {
                    "max-num-seqs": 16,
                    "cudagraph-capture-sizes": [1, 2, 4, 8],
                },
            }
        )
        sizes = plan.values["cudagraph-capture-sizes"]
        self.assertEqual(sorted(set(sizes)), sizes)
        self.assertEqual(
            "derived:capture-size cap override",
            plan.origins["cudagraph-capture-sizes"],
        )
        self.assertGreater(max(sizes), 8)

    def test_malformed_recipe_layer_is_refused(self):
        for layer, message in (
            (
                {"source": "recipe:x", "modes": {}},
                "Recipe layer accepts only options, environment and source",
            ),
            ({"source": "recipe:x", "options": []}, "Recipe options must be a mapping"),
            (
                {"source": "recipe:x", "environment": []},
                "Recipe environment must be a mapping",
            ),
            (
                {"options": {}},
                "Recipe layer requires a source prefixed with recipe:",
            ),
            (
                {"source": "preset:x"},
                "Recipe layer requires a source prefixed with recipe:",
            ),
            (
                {"source": "recipe:Bad Name"},
                "Recipe names must contain only lowercase letters, digits and hyphens",
            ),
            (
                {"source": "recipe:x", "environment": {"not a name": "1"}},
                "Recipe environment requires valid names and string values",
            ),
            (
                {"source": "recipe:x", "options": {"nope": "1"}},
                "Unknown managed setting: nope",
            ),
            (
                {"source": "recipe:x", "options": {"max-num-seqs": "many"}},
                "Invalid value for max-num-seqs; expected integer",
            ),
        ):
            with self.subTest(layer=layer):
                with self.assertRaises(ConfigError) as caught:
                    self.resolve(recipe_layer=layer)
                self.assertEqual(message, str(caught.exception))
                self.assertEqual(2, caught.exception.code)

    def test_absent_recipe_layer_changes_nothing(self):
        self.assertEqual(
            resolver.public(self.resolve()),
            resolver.public(self.resolve(recipe_layer=None)),
        )


class PurityTest(unittest.TestCase):
    """The resolver runs inside the image and in the browser, and nowhere else."""

    ALLOWED = {
        "__future__",
        "ast",
        "copy",
        "dataclasses",
        "hashlib",
        "image_tools.launcher",
        "importlib",
        "importlib.util",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "sys",
        "typing",
        "yaml",
    }

    def source_tree(self):
        return ast.parse(Path(resolver.__file__).read_text())

    def test_only_the_ordinary_library_and_pyyaml_are_imported(self):
        imported = set()
        for node in ast.walk(self.source_tree()):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertEqual(set(), imported - self.ALLOWED)

    def test_no_process_or_shell_entry_point(self):
        for node in ast.walk(self.source_tree()):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                self.assertFalse(
                    node.attr.startswith(("exec", "spawn", "system", "popen", "fork")),
                    f"os.{node.attr} at line {node.lineno} would leave the process",
                )

    @upstream_only
    def test_managed_environment_names_are_the_options_yaml_aliases(self):
        names = resolver.managed_environment_names(UPSTREAM_RUNTIME)
        expected = {
            alias
            for spec in upstream.read_yaml(UPSTREAM_RUNTIME / "options.yaml").values()
            for alias in spec["env"]
        }
        self.assertEqual(expected, names)
        # The launcher strips exactly this set from the child environment, so a
        # managed alias never reaches a second resolver; profile environment
        # keys, which are policy rather than an alias, must not be in it.
        for alias in ("LMCACHE_MODE", "TP_SIZE"):
            self.assertIn(alias, names)
        self.assertNotIn("VLLM_PLE_CPU_OFFLOAD", names)

    @upstream_only
    def test_profile_schema_rules_are_enforced_without_jsonschema(self):
        schema = json.loads((UPSTREAM_RUNTIME / "schema.json").read_text())
        for document, fragment in (
            ({"kind": "model"}, "required property"),
            ({"kind": "typo", "id": "x"}, "is not one of"),
            ({"kind": "model", "id": "Bad Name"}, "does not match"),
            ({"kind": "model", "id": "x", "unexpected": 1}, "not an allowed property"),
            (
                {"kind": "model", "id": "x", "modes": {"mtp": {"draft_tokens": 0}}},
                "less than the minimum",
            ),
            (
                {"kind": "model", "id": "x", "environment": {"OK": 1}},
                "is not of type string",
            ),
        ):
            with self.subTest(document=document):
                document.setdefault("schema_version", 1)
                document.setdefault("status", "implemented")
                document.setdefault("defaults", {})
                document.setdefault("environment", {})
                with self.assertRaises(ConfigError) as caught:
                    resolver._validate_schema(
                        document, schema, schema["$defs"], ""
                    )
                self.assertIn(fragment, str(caught.exception))

    @upstream_only
    def test_real_profiles_pass_the_pure_validator(self):
        profiles, hardware, _ = inventory()
        for identifier in profiles:
            self.assertEqual(
                identifier,
                resolver.profile("model", identifier, root=UPSTREAM_RUNTIME)["id"],
            )
        for kind in hardware:
            self.assertEqual(
                kind,
                resolver.profile("hardware", kind, root=UPSTREAM_RUNTIME)["id"],
            )
        with self.assertRaises(ConfigError) as caught:
            resolver.profile("model", "no-such-profile", root=UPSTREAM_RUNTIME)
        self.assertIn("Cannot read configuration", str(caught.exception))

    def test_qsa_atomic_transfer_is_decided_from_source_text(self):
        declared = (
            "from vllm.v1.attention.backend import AttentionBackend\n"
            "class Qwen4ExpQSABackend(AttentionBackend):\n"
            "    @classmethod\n"
            "    def supports_kv_connector(cls) -> bool:\n"
            "        return True\n"
        )
        self.assertTrue(resolver.qsa_atomic_transfer_supported(declared))
        self.assertFalse(
            resolver.qsa_atomic_transfer_supported(
                declared.replace("        return True", "        return False")
            )
        )
        self.assertFalse(
            resolver.qsa_atomic_transfer_supported(
                declared.replace(
                    "        return True",
                    "        if envs.VLLM_QSA_ATOMIC_CHECKPOINTS:\n"
                    "            return True\n"
                    "        return False",
                )
            )
        )
        self.assertFalse(resolver.qsa_atomic_transfer_supported(None))
        self.assertFalse(resolver.qsa_atomic_transfer_supported("class Other: pass"))
        self.assertFalse(
            resolver.qsa_atomic_transfer_supported("class Qwen4ExpQSABackend:\n    pass")
        )
        self.assertFalse(
            resolver.qsa_atomic_transfer_supported("this is not python ((")
        )

    def test_installed_source_probe_paths_are_the_upstream_ones(self):
        self.assertEqual(
            "models/qwen4_exp/nvidia/b12x_qsa.py",
            resolver.QSA_ATOMIC_TRANSFER_SOURCE,
        )



def tearDownModule():
    """Report what the gate compared, in the lane's own log.

    The number is the evidence that the gate ran: an OK that compared nothing
    is indistinguishable in the summary from one that compared the matrix,
    which is how this module passed CI for a month with no upstream checkout.
    """
    print(
        f"resolver parity gate: {COMPARISONS['count']} combinations compared "
        f"against {UPSTREAM_CHECKOUT} @ {UPSTREAM_HEAD or UPSTREAM_COMMIT}",
        file=sys.stderr,
    )
    if UPSTREAM_AVAILABLE and not COMPARISONS["count"]:
        raise AssertionError(
            "the pinned checkout was available and nothing was compared: the "
            "parity gate has gone vacuous"
        )

if __name__ == "__main__":
    unittest.main()
