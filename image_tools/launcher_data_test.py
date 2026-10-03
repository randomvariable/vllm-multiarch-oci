#!/usr/bin/env python3
"""Data-layer tests for the launcher's GB10 hardware profile and its recipes.

``image_tools/data/`` is the data half of ``vllm-image launch``: one hardware
profile for two RoCE-joined GB10 nodes, and three recipes that pair an upstream
model policy with it. This module is the gate that keeps that data honest.

What it proves:

* ``gb10-roce`` validates against the pinned upstream ``runtime/schema.json``
  through ``resolver.profile``, carries every hardware policy value the
  deployment needs, and carries none of the site values that belong to the
  manifest.
* Each recipe loads through ``launch.load_recipe`` -- which validates the
  ``launch`` section's keys exactly -- and resolves through
  ``resolver.resolve`` to the bytes stored in ``image_tools/data/golden``.
* Each recipe still carries its source of truth's whole command and
  environment: no accepted flag or variable is silently dropped.
* No recipe restates a setting or an environment variable that the common
  layer, an upstream model profile, the hardware profile or a preset already
  sets to the same value.
* The two LWS recipes differ between rank zero and rank one only in the
  arguments ``launch.topology_args`` and ``launch.kv_events_args`` build.

The accepted command is read from the file that recorded it: the two
``recipes/**.yaml`` manifests' ``runtime.base_args`` and ``runtime.base_env``,
or, for the checkpoint with no upstream profile and no recipe manifest, the
flag and variable names of the live definition, listed in ``ACCEPTED`` below.
The places where the resolved record deliberately differs from that command are
listed in ``DRIFT``, one line each; every entry must be a value the recipe does
not own, so the list cannot rot into fiction.

The golden files are generated, not hand-written::

    python3 image_tools/launcher_data_test.py --update

and committing the regenerated JSON with the data change is expected.

The upstream policy tree is read from the pinned checkout of
``local-inference-lab/blackwell-llm-docker`` at commit
``353efc679f631206e0b001e67047dea80ee6d76e``. When that checkout is absent the
resolution and hardware cases skip with an explicit message instead of passing
vacuously; the recipe-schema cases still run, because they need only this
repository.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

IMAGE_TOOLS = Path(__file__).resolve().parent
REPO = IMAGE_TOOLS.parent
if str(REPO) not in sys.path:
    # python3 image_tools/launcher_data_test.py, as --update is invoked.
    sys.path.insert(0, str(REPO))

from image_tools.launcher import ConfigError, launch, resolver

UPSTREAM_CHECKOUT = Path(
    "/home/naadir/go/src/github.com/local-inference-lab/blackwell-llm-docker"
)
UPSTREAM_RUNTIME = UPSTREAM_CHECKOUT / "runtime"
UPSTREAM_COMMIT = "353efc679f631206e0b001e67047dea80ee6d76e"
UPSTREAM_AVAILABLE = (UPSTREAM_RUNTIME / "schema.json").is_file()
SKIP_REASON = (
    f"pinned upstream checkout is absent: {UPSTREAM_CHECKOUT} @ {UPSTREAM_COMMIT} "
    "(clone it to run the launcher data gates)"
)

DATA = IMAGE_TOOLS / "data"
RECIPE_DIR = DATA / "recipes"
GOLDEN_DIR = DATA / "golden"
HARDWARE = "gb10-roce"
PROBE_PORT = 8890

# The three keys the golden records carry. resolver.public() also reports
# schema_version, status, qualification, warnings and the cache service, which
# belong to the resolver's own tests, not to this data.
RECORD_KEYS = ("argv", "settings", "environment")

# ``--update`` regenerates the golden files from the resolved data instead of
# comparing against them. It is read here, before any class decorator needs it.
UPDATE = "--update" in sys.argv

# Rank-independent launch arguments that runtime/options.yaml does not describe.
# The resolver cannot take them as recipe options -- set_value refuses an
# unmanaged key -- so a deployment passes them after `--`, exactly as
# launch.run forwards request.native. They are recorded in the recipe file's
# trailing comment and the goldens are generated with them.
NATIVE_ARGS: dict[str, list[str]] = {
    "qwen38-flash-next-gb10-tp2": [
        "--distributed-executor-backend", "mp",
        "--master-port", "25000",
        "--gdn-prefill-backend", "b12x",
        "--disable-access-log-for-endpoints", "/metrics,/v1/models",
    ],
    "deepseek-v4-flash-vision-gb10-tp2": [
        "--distributed-executor-backend", "mp",
        "--master-port", "25000",
        "--dcp-comm-backend", "a2a",
        "--reasoning-config",
        '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}',
        "--disable-access-log-for-endpoints", "/metrics,/v1/models",
    ],
    # The QAD checkpoint routes dense and MoE GEMMs through b12x with the
    # FlashInfer autotuner off; --kernel-config is not a managed option
    # (internal.randomvariable.co.uk qwen38-27b.yaml:171-179, 208).
    "qwen38-27b": [
        "--kernel-config",
        '{"enable_flashinfer_autotune": false, "linear_backend": "b12x"}',
    ],
}

# Which file recorded each recipe's accepted command. The two TP=2 deployments
# carry it in this repository's own recipe manifests; the single-node checkpoint
# has no manifest, so its accepted names are listed inline below, quoted from
# the live definition.
ACCEPTED_MANIFEST: dict[str, Path] = {
    "qwen38-flash-next-gb10-tp2": REPO / "recipes/local-inference-lab/Qwen3.8-Flash-Next-NVFP4.yaml",
    "deepseek-v4-flash-vision-gb10-tp2": REPO / "recipes/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp.yaml",
}

# The single-node checkpoint has no manifest in this repository, so its
# accepted command is the live definition's own argv tail, quoted from
# internal.randomvariable.co.uk src/rv/k8s/clusters/home/llm-d/qwen38-27b.yaml
# lines 183-209 with the three shell variables the init script interpolated
# replaced by the values the manifest gives them at lines 221-228, and its
# accepted environment is lines 210-374 minus the variables the manifest
# injects at run time (POD_IP, HF_TOKEN) and the rank arguments the launcher
# appends itself.
ACCEPTED_ARGUMENTS: dict[str, list[str]] = {
    "qwen38-27b": [
        "local-inference-lab/Qwen3.8-27B-NVFP4-QAD",
        "--revision", "f40a31cd813a6746067e7d6446ff2cb708dbb779",
        "--served-model-name", "qwen3.8-27b",
        "--host", "0.0.0.0",
        "--port", "8888",
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--quantization", "modelopt_mixed",
        "--hf-overrides",
        '{"text_config":{"max_position_embeddings":1048576,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}',
        "--max-model-len", "1048576",
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "32",
        "--max-cudagraph-capture-size", "128",
        "--max-num-batched-tokens", "8192",
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching",
        "--mamba-cache-mode", "align",
        "--prefix-match-unit", "16",
        "--enable-chunked-prefill",
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,2,3],[3,8,0]]}',
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_xml",
        "--enable-auto-tool-choice",
        "--default-chat-template-kwargs", '{"preserve_thinking": true}',
        "--override-generation-config",
        '{"temperature":0.7,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}',
        "--kernel-config",
        '{"enable_flashinfer_autotune": false, "linear_backend": "b12x"}',
        "--limit-mm-per-prompt", '{"image":8}',
    ],
}

ACCEPTED_ENVIRONMENT: dict[str, frozenset[str]] = {
    "qwen38-27b": frozenset(
        {
            "HOME", "HF_HOME", "HF_HUB_OFFLINE", "HF_XET_HIGH_PERFORMANCE",
            "HF_XET_CACHE", "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST",
            "CUTE_DSL_ARCH", "SAFETENSORS_FAST_GPU", "PYTORCH_CUDA_ALLOC_CONF",
            "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
            "FLASHINFER_WORKSPACE_BASE", "VLLM_CACHE_ROOT",
            "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_DISABLED_KERNELS",
            "B12X_COMPILE_CACHE_DIR", "VLLM_USE_RUST_FRONTEND",
            "VLLM_WORKER_MULTIPROC_METHOD", "OMP_NUM_THREADS",
            "NCCL_CUMEM_ENABLE",
        }
    ),
}

# Shell indirection the accepted deployment's init or serve script interpolated
# into ``exec vllm serve``. There is no shell in the launcher, so each of these
# is a recipe option now: the variable must be gone from the resolved
# environment and its value must reach the command line instead. (flag, field,
# expected) compares inside a JSON object option when field is not None; the
# pseudo-flag ``model`` is the checkpoint the engine loads.
RETIRED_SHELL_INDIRECTION: dict[str, dict[str, tuple[str, str | None, object]]] = {
    "deepseek-v4-flash-vision-gb10-tp2": {
        "DSV4_MODEL": ("model", None, "/models/deepseek-v4-flash-vision"),
        "SERVED_MODEL_NAME": ("served-model-name", None, "deepseek-v4-flash-vision"),
        "LOAD_FORMAT": ("load-format", None, "instanttensor"),
        "DSPARK_TOKENS": ("speculative-config", "num_speculative_tokens", 3),
        "GPU_MEMORY_UTILIZATION": ("gpu-memory-utilization", None, "0.81"),
        "MM_IMAGE_LIMIT": ("limit-mm-per-prompt", "image", 1),
        "MAX_MODEL_LEN": ("max-model-len", None, "1000000"),
        "MAX_NUM_SEQS": ("max-num-seqs", None, "4"),
        "MAX_NUM_BATCHED_TOKENS": ("max-num-batched-tokens", None, "8192"),
    },
    "qwen38-27b": {
        "MODEL_REPO": ("model", None, "local-inference-lab/Qwen3.8-27B-NVFP4-QAD"),
        "MODEL_REV": ("revision", None, "f40a31cd813a6746067e7d6446ff2cb708dbb779"),
        "SERVED_MODEL_NAME": ("served-model-name", None, "qwen3.8-27b"),
    },
}

RECIPES = tuple(sorted(NATIVE_ARGS))

# Every difference between an accepted command and its golden, one line each.
# A key may appear in both and still be drift when its value changed.
DRIFT: dict[str, dict[str, str]] = {
    "qwen38-flash-next-gb10-tp2": {
        "setting:speculative-config": (
            "the profile's mtp mode adds draft_sample_method, "
            "rejection_sample_method and moe_backend to the accepted "
            '{"method":"mtp","num_speculative_tokens":3}.'
        ),
        "setting:hf-overrides": (
            "resolver.resolve derives the 4x YaRN block from max-model-len, so "
            "it emits factor as the integer 4 where the accepted command wrote "
            "4.0; mrope_section, mrope_interleaved, partial_rotary_factor and "
            "rope_theta match."
        ),
        "setting:mm-processor-cache-gb": (
            "the profile's number 0 renders as 0.0; the same value to vLLM."
        ),
        "setting:async-scheduling": "adopted from the upstream model profile.",
        "setting:decode-context-parallel-size": "adopted from the common layer.",
        "setting:language-model-only": "adopted from the upstream model profile.",
        "setting:mamba-ssm-cache-dtype": "adopted from the upstream model profile.",
        "environment:TORCH_CUDA_ARCH_LIST": (
            "gb10-roce names the single arch this device is; the accepted Qwen "
            "deployment left the image's multi-arch default in place."
        ),
        "environment:FLASHINFER_CUDA_ARCH_LIST": (
            "gb10-roce policy; the accepted Qwen deployment did not set it."
        ),
        "environment:VLLM_USE_FASTOKENS": (
            "the common layer's Rust tokenizer backend, which the accepted "
            "deployment did not export."
        ),
        "environment:NCCL_NET_PLUGIN": (
            "platform:foundation, which the accepted deployment inherited from "
            "the image environment rather than from the resolver."
        ),
    },
    "deepseek-v4-flash-vision-gb10-tp2": {
        "setting:pipeline-parallel-size": "adopted from the common layer.",
        "setting:speculative-config": (
            "the profile's dspark mode adds rejection_sample_method and the "
            "self-draft model path to the accepted "
            '{"method":"dspark","num_speculative_tokens":3,'
            '"draft_sample_method":"probabilistic"}.'
        ),
        "setting:moe-backend": (
            "the profile selects b12x for the experts; the accepted command "
            "named it only for the dense projections."
        ),
        "setting:prefix-cache-retention-interval": "adopted from the model profile.",
        "setting:enable-force-include-usage": "adopted from the model profile.",
        "setting:enable-request-id-headers": "adopted from the model profile.",
        "setting:override-generation-config": "adopted from the model profile.",
        "environment:VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": (
            "the profile's 1 replaces the accepted 0. This knob feeds the "
            "memory profiler that produced the measured 0.81 utilization, so "
            "re-check 0.81 on the next live GB10 run."
        ),
        "environment:VLLM_ROCE_ALLREDUCE_MAX_SIZE": (
            "gb10-roce policy; the accepted DeepSeek deployment relied on the "
            "image default."
        ),
        "environment:B12X_ROCE_SPIN_LIMIT": (
            "gb10-roce policy; the accepted DeepSeek deployment did not set it."
        ),
        "environment:VLLM_ENABLE_PCIE_ALLREDUCE": (
            "gb10-roce disables the crossover all-reduce outright; the accepted "
            "deployment simply never enabled it."
        ),
        "environment:VLLM_WORKER_MULTIPROC_METHOD": (
            "gb10-roce restates the common layer's value for this topology; the "
            "accepted deployment exported it too."
        ),
        "environment:VLLM_USE_FASTOKENS": (
            "the common layer's Rust tokenizer backend, which the accepted "
            "deployment did not export."
        ),
        "environment:VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": (
            "adopted from the model profile."
        ),
        "environment:VLLM_B12X_MOE_FP4_FORCE_A16": "adopted from the model profile.",
        "environment:NCCL_NET_PLUGIN": (
            "platform:foundation, which the accepted deployment inherited from "
            "the image environment rather than from the resolver."
        ),
    },
    "qwen38-27b": {
        "setting:pipeline-parallel-size": "adopted from the common layer.",
        "setting:block-size": (
            "the borrowed qwen38-flash-next profile's 64-token block policy "
            "reaches a checkpoint the live definition left at the engine default."
        ),
        "setting:compilation-config": "adopted from the borrowed model profile.",
        "setting:load-format": "adopted from the common layer.",
        "setting:enable-prompt-tokens-details": "adopted from the common layer.",
        "setting:async-scheduling": "adopted from the borrowed model profile.",
        "setting:decode-context-parallel-size": "adopted from the common layer.",
        "setting:language-model-only": "adopted from the borrowed model profile.",
        "setting:mamba-ssm-cache-dtype": "adopted from the borrowed model profile.",
        "setting:mm-encoder-tp-mode": "adopted from the borrowed model profile.",
        "setting:mm-processor-cache-gb": "adopted from the borrowed model profile.",
        "setting:gdn-decode-kernel": "adopted from the borrowed model profile.",
        "setting:linear-backend": "adopted from the borrowed model profile.",
        "setting:moe-backend": "adopted from the borrowed model profile.",
        "setting:enable-flashinfer-autotune": (
            "the profile's false renders as --no-enable-flashinfer-autotune; "
            "the live definition reached the same result through --kernel-config."
        ),
        "environment:VLLM_PLE_CPU_OFFLOAD": (
            "the whole qwen38-flash-next environment block is inherited by a "
            "checkpoint that has no PLE n-gram tables; the recipe borrows the "
            "profile because none exists for this checkpoint."
        ),
        "environment:VLLM_SSM_CONV_STATE_LAYOUT": "adopted from the borrowed profile.",
        "environment:VLLM_ENABLE_ROCE_ALLREDUCE": (
            "gb10-roce policy on a single-node deployment: at TP=1 there is no "
            "cross-node collective for it to carry, and the live definition "
            "leaves it at the image default."
        ),
        "environment:XDG_CACHE_HOME": (
            "the resolver namespaces the JIT caches from the image's source-lock "
            "digest; the live definition left it unset and derived them from ~."
        ),
        "environment:VLLM_WORKER_MULTIPROC_METHOD": (
            "gb10-roce restates the common layer's value for this topology."
        ),
        "environment:VLLM_USE_V2_MODEL_RUNNER": (
            "the common layer exports it; the live definition did not."
        ),
        "environment:NCCL_NET_PLUGIN": (
            "platform:foundation, which the live definition inherited from the "
            "image environment rather than from the resolver."
        ),
    },
}

# The hardware policy gb10-roce must own.
HARDWARE_ENVIRONMENT = (
    "NCCL_NET", "NCCL_IB_DISABLE", "NCCL_CROSS_NIC", "NCCL_CUMEM_ENABLE",
    "NCCL_IGNORE_CPU_AFFINITY", "NCCL_NVLS_ENABLE", "NCCL_DEBUG",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING", "VLLM_ENABLE_ROCE_ALLREDUCE",
    "VLLM_ROCE_ALLREDUCE_MAX_SIZE", "B12X_ROCE_SPIN_LIMIT",
    "VLLM_ENABLE_PCIE_ALLREDUCE", "CUTE_DSL_ARCH", "TORCH_CUDA_ARCH_LIST",
    "FLASHINFER_CUDA_ARCH_LIST", "VLLM_WORKER_MULTIPROC_METHOD",
    "PYTHONFAULTHANDLER",
)

# The environment the manifest supplies and the data layer must not: HCA and
# GID selection, socket interfaces, and the pod addresses.
SITE_ENVIRONMENT = (
    "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP", "POD_IP",
)


# A fixed stand-in for the image's source-lock digest, so the derived JIT cache
# namespace is stable in the goldens instead of naming this machine.
RUNTIME_IDENTITY = "0" * 64


def canonical(flag: str) -> str:
    """Compare flags on their option name, not their shell spelling.

    ``--no-foo`` is ``foo`` set false, and ``--root.field=value`` refines the
    object ``--root`` already carries, so both spellings resolve to the option
    runtime/options.yaml names.
    """
    name = flag[2:] if flag.startswith("--") else flag
    name = name.split(".", 1)[0]
    return name[3:] if name.startswith("no-") else name


def fold_arguments(tokens: list[str]) -> tuple[dict, list]:
    """An argv tail as ``{option: value}``, plus its dotted JSON refinements.

    A token with no value is a boolean flag: ``None`` when positive, the string
    ``"false"`` when spelled ``--no-...``, so the polarity survives the fold. The
    positional checkpoint and anything else that is not a flag is skipped.
    """
    values: dict[str, str | None] = {}
    dotted: list[tuple[str, str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            continue
        flag, _, inline = token[2:].partition("=")
        name = canonical(flag)
        if "." in flag:
            dotted.append((name, flag.split(".", 1)[1], inline))
            values.setdefault(name, None)
            continue
        negated = flag.startswith("no-")
        if inline:
            values[name] = inline
        elif index < len(tokens) and not tokens[index].startswith("--"):
            values[name] = tokens[index]
            index += 1
        else:
            values[name] = "false" if negated else None
    return values, dotted


def accepted_arguments(name: str) -> tuple[dict, list]:
    """The command the source of truth recorded for this recipe."""
    manifest = ACCEPTED_MANIFEST.get(name)
    if manifest is None:
        return fold_arguments(ACCEPTED_ARGUMENTS[name])
    document = resolver.read_yaml(manifest)
    tokens = list(document["runtime"]["base_args"])
    deployment = document.get("deployment", {})
    if deployment.get("tensor_parallel_size"):
        # The live manifest passes it from LWS_GROUP_SIZE, so it is part of the
        # accepted command without appearing in base_args.
        tokens += ["--tensor-parallel-size", str(deployment["tensor_parallel_size"])]
    return fold_arguments(tokens)


def resolved_arguments(record_: dict) -> dict:
    return fold_arguments(record_["argv"])[0]


def accepted_environment(name: str) -> frozenset[str]:
    """The runtime variables the source of truth exported.

    Two kinds of entry in an accepted ``base_env`` are not runtime variables and
    must not appear in the resolved environment: the option aliases
    runtime/options.yaml declares, which a recipe states as ``launch.options``
    because setting an alias as environment exports it without ever reaching the
    setting, and the shell variables the old init or serve script interpolated
    into its own command line, listed in RETIRED_SHELL_INDIRECTION.
    """
    manifest = ACCEPTED_MANIFEST.get(name)
    keys = (
        ACCEPTED_ENVIRONMENT[name]
        if manifest is None
        else resolver.read_yaml(manifest)["runtime"]["base_env"]
    )
    aliases = resolver.managed_environment_names(UPSTREAM_RUNTIME)
    retired = set(RETIRED_SHELL_INDIRECTION.get(name, {}))
    return frozenset(key for key in keys if key not in aliases and key not in retired)


def same_value(left: str | None, right: str | None) -> bool:
    """Two argv values, compared the way vLLM reads them.

    JSON is compared structurally, so ``{"a": 1}`` and ``{"a":1}`` are one
    value; anything else is compared as written, so ``0`` and ``0.0`` are not.
    """
    if left is None or right is None:
        return left == right
    try:
        parsed = (json.loads(left), json.loads(right))
    except ValueError:
        return left == right
    if any(isinstance(item, (dict, list)) for item in parsed):
        return json.dumps(parsed[0], sort_keys=True) == json.dumps(
            parsed[1], sort_keys=True
        )
    return left == right


def scratch_parent() -> Path:
    """Where the merged data root is built.

    /tmp is tmpfs on this host, so tempfile is pointed at durable disk rather
    than left to the default. Override with VLLM_IMAGE_TEST_SCRATCH.
    """
    parent = Path(
        os.environ.get(
            "VLLM_IMAGE_TEST_SCRATCH",
            Path.home() / ".cache" / "vllm-image-launcher" / "test-tmp",
        )
    )
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def build_data_root(parent: Path) -> Path:
    """The pinned upstream tree with our hardware files landed beside its own."""
    root = parent / "runtime"
    root.mkdir(parents=True)
    for entry in sorted(UPSTREAM_RUNTIME.iterdir()):
        (root / entry.name).symlink_to(entry)
    # hardware/ is copied, not symlinked: gb10-roce.yaml has to sit next to
    # upstream's native.yaml and rtx-pro-6000-pcie.yaml in one readable
    # directory, and a symlinked directory would hide it behind the pin.
    (root / "hardware").unlink()
    shutil.copytree(UPSTREAM_RUNTIME / "hardware", root / "hardware")
    for path in sorted((DATA / "hardware").glob("*.yaml")):
        shutil.copy2(path, root / "hardware" / path.name)
    return root


def recipe_layer(recipe: launch.Recipe) -> dict:
    return {
        "options": dict(recipe.options),
        "environment": dict(recipe.environment),
        "source": f"recipe:{recipe.name}",
    }


def resolve_recipe(
    recipe: launch.Recipe, *, root: Path, env: dict[str, str] | None = None
) -> resolver.ResolvedPlan:
    """One recipe through the resolver, exactly as launch.resolved_plan does."""
    environment = dict(env or {})
    return resolver.resolve(
        recipe.profile,
        recipe.hardware,
        preset=recipe.preset,
        recipe_layer=recipe_layer(recipe),
        env=environment,
        argv=list(NATIVE_ARGS[recipe.name]),
        cli_env=environment,
        runtime_identity=RUNTIME_IDENTITY,
        vllm_environment=frozenset(),
        root=root,
    )


def record(plan: resolver.ResolvedPlan) -> dict:
    public = resolver.public(plan)
    return {key: public[key] for key in RECORD_KEYS}


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def dump(record_: dict) -> str:
    return json.dumps(record_, sort_keys=True, indent=2) + "\n"


UPDATE = "--update" in sys.argv


class LauncherDataBase(unittest.TestCase):
    """Shared merged data root, published through the launcher's own roots."""

    @classmethod
    def setUpClass(cls) -> None:
        if not UPSTREAM_AVAILABLE:
            raise unittest.SkipTest(SKIP_REASON)
        cls._directory = tempfile.TemporaryDirectory(
            dir=str(scratch_parent()), prefix="launcher-data-"
        )
        cls.root = build_data_root(Path(cls._directory.name))
        cls._saved = {
            name: os.environ.get(name)
            for name in ("VLLM_IMAGE_DATA_ROOT", "VLLM_IMAGE_RECIPE_ROOT")
        }
        # The launcher reads both from the environment, so the whole process
        # points at the merged tree for the duration of the class.
        os.environ["VLLM_IMAGE_DATA_ROOT"] = str(cls.root)
        os.environ["VLLM_IMAGE_RECIPE_ROOT"] = str(RECIPE_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        for name, value in cls._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls._directory.cleanup()

    def recipe(self, name: str) -> launch.Recipe:
        # No root argument: this must resolve the way the launcher resolves it,
        # from VLLM_IMAGE_RECIPE_ROOT.
        return launch.load_recipe(name)


@unittest.skipUnless(UPSTREAM_AVAILABLE, SKIP_REASON)
class HardwareProfileTest(LauncherDataBase):
    def test_gb10_roce_validates_against_the_pinned_schema(self) -> None:
        profile = resolver.profile("hardware", HARDWARE, root=self.root)
        self.assertEqual(profile["kind"], "hardware")
        self.assertEqual(profile["id"], HARDWARE)
        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["status"], "implemented")
        self.assertEqual(profile["defaults"], {}, "this hardware carries no settings")

    def test_gb10_roce_carries_the_hardware_policy(self) -> None:
        environment = resolver.profile("hardware", HARDWARE, root=self.root)["environment"]
        self.assertEqual(set(environment), set(HARDWARE_ENVIRONMENT))
        self.assertEqual(environment["CUTE_DSL_ARCH"], "sm_121a")
        self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], "12.1a")
        self.assertEqual(environment["FLASHINFER_CUDA_ARCH_LIST"], "12.1a")
        self.assertEqual(environment["NCCL_NET"], "IB")
        self.assertEqual(environment["NCCL_IB_DISABLE"], "0")
        self.assertEqual(environment["VLLM_ENABLE_PCIE_ALLREDUCE"], "0")

    def test_gb10_roce_holds_no_site_values(self) -> None:
        environment = resolver.profile("hardware", HARDWARE, root=self.root)["environment"]
        for name in SITE_ENVIRONMENT:
            self.assertNotIn(name, environment, f"{name} belongs to the manifest")

    def test_gb10_roce_holds_no_per_model_policy(self) -> None:
        """Every value here is true of the pair of hosts whichever model runs."""
        profile = resolver.profile("hardware", HARDWARE, root=self.root)
        self.assertEqual(profile.get("model_environment", {}), {})


class RecipeSchemaTest(unittest.TestCase):
    """The recipes must load through the launcher's own exact key validation."""

    def test_each_recipe_loads(self) -> None:
        for name in RECIPES:
            with self.subTest(recipe=name):
                recipe = launch.load_recipe(name, root=RECIPE_DIR)
                self.assertEqual(recipe.name, name)
                self.assertEqual(recipe.hardware, HARDWARE)
                self.assertEqual(recipe.probe_port, PROBE_PORT)

    def test_lws_recipes_declare_sync_and_kv_events(self) -> None:
        for name in ("qwen38-flash-next-gb10-tp2", "deepseek-v4-flash-vision-gb10-tp2"):
            recipe = launch.load_recipe(name, root=RECIPE_DIR)
            self.assertEqual(recipe.topology.kind, launch.TOPOLOGY_LWS)
            self.assertEqual(recipe.topology.nodes, 2)
            self.assertEqual(recipe.topology.rendezvous_port, 25000)
            self.assertIsNone(recipe.topology.replica_port_base)
            events = recipe.topology.kv_events
            self.assertIsNotNone(events)
            self.assertEqual(events.publisher, "zmq")
            self.assertEqual(events.endpoint, "tcp://*:5556")
            self.assertEqual(events.replay_endpoint, "tcp://*:5559")
            self.assertIsNotNone(recipe.model_sync)
            self.assertEqual(str(recipe.model_sync.storage_root), "/models")
            self.assertEqual(recipe.model_sync.workers, 8)
            self.assertEqual(
                recipe.model_sync.required_files,
                ("tokenizer_config.json", "chat_template.jinja"),
            )
            self.assertTrue(
                launch._is_commit(recipe.model_sync.revision),
                "model_sync.revision must be a full lowercase commit OID",
            )
            self.assertEqual(
                str(recipe.model_sync.publish),
                f"/models/{recipe.options['served-model-name']}",
                "the published root and the served name are one identity",
            )

    def test_pinned_revisions_match_the_accepted_manifests(self) -> None:
        expected = {
            "qwen38-flash-next-gb10-tp2": (
                "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
                "60215d26cf5e42c2db6128774032d57fc62678da",
            ),
            "deepseek-v4-flash-vision-gb10-tp2": (
                "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
                "6821d6ad3681a4b137b066b76094fa82ebd0a380",
            ),
        }
        for name, (repo, revision) in expected.items():
            sync = launch.load_recipe(name, root=RECIPE_DIR).model_sync
            with self.subTest(recipe=name):
                self.assertEqual((sync.repo, sync.revision), (repo, revision))

    def test_qwen38_27b_is_a_single_node_engine_download(self) -> None:
        recipe = launch.load_recipe("qwen38-27b", root=RECIPE_DIR)
        self.assertEqual(recipe.topology.kind, launch.TOPOLOGY_SINGLE)
        self.assertEqual(recipe.topology.nodes, 1)
        self.assertIsNone(recipe.topology.kv_events)
        # Topology.from_dict fills the default in; a single node never uses it.
        self.assertEqual(recipe.topology.rendezvous_port, launch.DEFAULT_RENDEZVOUS_PORT)
        self.assertIsNone(recipe.model_sync, "the engine downloads this checkpoint")
        self.assertEqual(recipe.environment["HF_HUB_OFFLINE"], "0")
        # A repository id, not a published snapshot path: with no model_sync
        # phase there is nothing to publish, so vLLM must resolve the hub itself.
        self.assertEqual(
            recipe.options["model"], "local-inference-lab/Qwen3.8-27B-NVFP4-QAD"
        )
        self.assertTrue(launch._is_commit(str(recipe.options["revision"])))

    def test_qwen_recipe_leaves_the_speculative_policy_to_the_profile(self) -> None:
        recipe = launch.load_recipe("qwen38-flash-next-gb10-tp2", root=RECIPE_DIR)
        for key in ("speculative-config", "mode", "draft-tokens", "hf-overrides"):
            self.assertNotIn(key, recipe.options, f"{key} is derived upstream")
        self.assertEqual(recipe.options["max-model-len"], 1048576)
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN stays unset, as the accepted recipe argues.
        self.assertNotIn("VLLM_ALLOW_LONG_MAX_MODEL_LEN", recipe.environment)

    def test_read_rank_and_the_recipes_agree_on_group_size(self) -> None:
        env = {"LWS_WORKER_INDEX": "0", "LWS_GROUP_SIZE": "2", "LWS_LEADER_ADDRESS": "l"}
        for name in ("qwen38-flash-next-gb10-tp2", "deepseek-v4-flash-vision-gb10-tp2"):
            recipe = launch.load_recipe(name, root=RECIPE_DIR)
            rank = launch.read_rank(recipe.topology, env)
            self.assertEqual((rank.index, rank.nodes), (0, 2))
            with self.assertRaises(ConfigError):
                launch.read_rank(recipe.topology, {**env, "LWS_GROUP_SIZE": "4"})


@unittest.skipIf(UPDATE, "the generation pass rewrites the goldens instead")
class RecipeGoldenTest(LauncherDataBase):
    def test_each_recipe_resolves_to_its_golden(self) -> None:
        for name in RECIPES:
            with self.subTest(recipe=name):
                plan = resolve_recipe(self.recipe(name), root=self.root)
                path = golden_path(name)
                self.assertTrue(path.is_file(), f"{path} is missing; run --update")
                self.assertEqual(
                    dump(record(plan)),
                    path.read_text(),
                    f"{path} disagrees with the resolved recipe; run "
                    "python3 image_tools/launcher_data_test.py --update",
                )


@unittest.skipUnless(UPSTREAM_AVAILABLE, SKIP_REASON)
class RecipePolicyTest(LauncherDataBase):
    """A recipe is a delta: nothing in it may restate another layer's value."""

    POLICY_PREFIXES = ("platform:", "common:", "model:", "hardware:", "preset:")

    def policy_layers(self, recipe: launch.Recipe) -> tuple[dict, dict]:
        plan = resolver.resolve(
            recipe.profile,
            recipe.hardware,
            preset=recipe.preset,
            env={},
            argv=[],
            cli_env={},
            runtime_identity=RUNTIME_IDENTITY,
            vllm_environment=frozenset(),
            root=self.root,
        )
        settings = {
            key: value
            for key, value in plan.values.items()
            if plan.origins[key].startswith(self.POLICY_PREFIXES)
        }
        environment = {
            key: value
            for key, value in plan.environment.items()
            if plan.environment_origins[key].startswith(self.POLICY_PREFIXES)
        }
        return settings, environment

    def test_no_recipe_duplicates_a_policy_layer(self) -> None:
        specs = resolver.read_yaml(self.root / "options.yaml")
        for name in RECIPES:
            recipe = self.recipe(name)
            settings, environment = self.policy_layers(recipe)
            for key, value in recipe.options.items():
                if key not in settings:
                    continue
                converted = resolver.convert(key, value, specs[key])
                with self.subTest(recipe=name, option=key):
                    self.assertNotEqual(
                        converted,
                        settings[key],
                        f"launch.options.{key} restates the value "
                        f"{settings[key]!r} another layer sets; delete it",
                    )
            for key, value in recipe.environment.items():
                if key not in environment:
                    continue
                with self.subTest(recipe=name, environment=key):
                    self.assertNotEqual(
                        value,
                        environment[key],
                        f"launch.environment.{key} restates "
                        f"{environment[key]!r}; delete it",
                    )

    def test_every_recipe_value_wins_in_the_resolved_record(self) -> None:
        """A recipe value another layer silently overrode is a value to delete."""
        for name in RECIPES:
            recipe = self.recipe(name)
            plan = resolve_recipe(recipe, root=self.root)
            for key in recipe.options:
                with self.subTest(recipe=name, option=key):
                    self.assertEqual(plan.origins[key], f"recipe:{name}")
            for key in recipe.environment:
                with self.subTest(recipe=name, environment=key):
                    self.assertEqual(plan.environment_origins[key], f"recipe:{name}")

    def test_no_recipe_carries_an_unmanaged_setting(self) -> None:
        """options keys the resolver would refuse would fail the launch, not this test."""
        specs = resolver.read_yaml(self.root / "options.yaml")
        for name in RECIPES:
            for key in self.recipe(name).options:
                with self.subTest(recipe=name, option=key):
                    self.assertIn(key, specs)


@unittest.skipUnless(UPSTREAM_AVAILABLE, SKIP_REASON)
class AcceptedCommandTest(LauncherDataBase):
    """The golden must still be the accepted command, plus a listed difference.

    Upstream-supplied defaults are *expected additions*: the common layer, the
    model profile, the hardware profile and a preset each contribute settings the
    accepted deployment inherited the same way and never restated -- ``dtype``
    and ``kv-cache-dtype`` from ``profiles/common.yaml:29-30`` are the standing
    example. A flag therefore has to be written down in ``DRIFT`` only when
    neither the accepted command nor any of those layers accounts for it, so no
    future reader pins ``dtype`` into a recipe to satisfy this test.
    """

    maxDiff = None

    def golden(self, name: str) -> dict:
        return json.loads(golden_path(name).read_text())

    def layer_defaults(self, recipe: launch.Recipe) -> set[str]:
        keys: set[str] = set()
        for kind, identifier in (
            ("common", "common"),
            ("model", recipe.profile),
            ("hardware", recipe.hardware),
        ):
            keys |= set(resolver.profile(kind, identifier, root=self.root).get("defaults", {}))
        if recipe.preset:
            keys |= set(resolver.deployment_preset(recipe.preset, root=self.root)["options"])
        return keys

    def test_every_accepted_flag_and_value_survives(self) -> None:
        for name in RECIPES:
            accepted, dotted = accepted_arguments(name)
            resolved = resolved_arguments(self.golden(name))
            declared = self.declared_settings(name)
            for flag in sorted(accepted):
                with self.subTest(recipe=name, flag=flag):
                    self.assertIn(flag, resolved, f"--{flag} dropped from the recipe")
                    if accepted[flag] is None or flag in declared:
                        continue
                    self.assertTrue(
                        same_value(accepted[flag], resolved[flag]),
                        f"--{flag}: accepted {accepted[flag]!r}, resolved "
                        f"{resolved[flag]!r}; declare it in DRIFT if intended",
                    )
            for parent, field, raw in dotted:
                with self.subTest(recipe=name, flag=f"{parent}.{field}"):
                    self.assertIn(parent, resolved)
                    document = json.loads(resolved[parent])
                    try:
                        expected = json.loads(raw)
                    except ValueError:
                        expected = raw
                    self.assertEqual(document[field], expected)
            for key in sorted(accepted_environment(name)):
                with self.subTest(recipe=name, environment=key):
                    self.assertIn(key, self.golden(name)["environment"])

    def test_retired_shell_indirection_reaches_the_command_line(self) -> None:
        """No shell sits between the launcher and argv, so these are options now."""
        for name, retired in RETIRED_SHELL_INDIRECTION.items():
            record_ = self.golden(name)
            resolved = resolved_arguments(record_)
            environment = record_["environment"]
            for variable, (flag, field, expected) in sorted(retired.items()):
                with self.subTest(recipe=name, variable=variable):
                    self.assertNotIn(variable, environment)
                    if flag == "model":
                        actual = record_["argv"][4]
                    elif field is None:
                        actual = resolved.get(flag)
                    else:
                        actual = json.loads(resolved[flag]).get(field)
                    self.assertEqual(actual, expected)

    def declared_settings(self, name: str) -> set[str]:
        return {
            key[len("setting:") :]
            for key in DRIFT[name]
            if key.startswith("setting:")
        }

    def test_the_golden_adds_no_unlisted_flag(self) -> None:
        """Anything the accepted command did not carry, and no layer supplies."""
        for name in RECIPES:
            accepted, _ = accepted_arguments(name)
            allowed = set(accepted) | self.layer_defaults(self.recipe(name))
            resolved = set(resolved_arguments(self.golden(name)))
            unexplained = resolved - allowed - self.declared_settings(name)
            missing = self.declared_settings(name) - resolved
            with self.subTest(recipe=name):
                self.assertEqual(unexplained, set(), "flag in the golden nothing accounts for")
                self.assertEqual(missing, set(), "DRIFT names a flag the golden lacks")

    def test_every_drift_entry_is_real_and_not_recipe_owned(self) -> None:
        for name in RECIPES:
            plan = resolve_recipe(self.recipe(name), root=self.root)
            for subject, reason in DRIFT[name].items():
                kind, _, key = subject.partition(":")
                with self.subTest(recipe=name, subject=subject):
                    self.assertTrue(reason.strip(), "drift needs a reason")
                    self.assertIn(kind, ("setting", "environment"))
                    if kind == "setting":
                        self.assertIn(key, plan.values)
                        self.assertNotEqual(plan.origins[key], f"recipe:{name}")
                    else:
                        self.assertIn(key, plan.environment)
                        self.assertNotEqual(
                            plan.environment_origins[key], f"recipe:{name}"
                        )


@unittest.skipUnless(UPSTREAM_AVAILABLE, SKIP_REASON)
class TopologyArgumentTest(LauncherDataBase):
    """One pod template, two ranks: only the topology arguments may differ."""

    RANK_ENVIRONMENT = {
        "LWS_GROUP_SIZE": "2",
        "LWS_LEADER_ADDRESS": "qwen38-flash-next-0.qwen38-flash-next.vllm.svc",
        "POD_IP": "10.42.7.19",
    }

    def ranked(self, name: str, index: int) -> tuple[list[str], list[str], dict]:
        recipe = self.recipe(name)
        # resolve() never sees the rank: it exports the manifest environment,
        # which is identical on every rank, and the rank enters only below.
        plan = resolve_recipe(recipe, root=self.root, env=dict(self.RANK_ENVIRONMENT))
        rank = launch.read_rank(
            recipe.topology, {**self.RANK_ENVIRONMENT, "LWS_WORKER_INDEX": str(index)}
        )
        arguments = launch.topology_args(
            rank, plan.values, recipe.topology.kv_events, dict(self.RANK_ENVIRONMENT)
        )
        return list(plan.argv), arguments, record(plan)

    def test_lws_recipes_differ_only_in_the_rank_arguments(self) -> None:
        for name in ("qwen38-flash-next-gb10-tp2", "deepseek-v4-flash-vision-gb10-tp2"):
            with self.subTest(recipe=name):
                base0, added0, record0 = self.ranked(name, 0)
                base1, added1, record1 = self.ranked(name, 1)
                recipe = self.recipe(name)
                # Everything resolve() produced is rank-independent.
                self.assertEqual(record0, record1)
                self.assertEqual(base0, base1)
                self.assertEqual(
                    added0,
                    [
                        "--nnodes", "2",
                        "--node-rank", "0",
                        "--master-addr", "127.0.0.1",
                        "--enable-scale-out",
                        "--kv-events-config",
                        json.dumps(
                            {
                                "enable_kv_cache_events": True,
                                "publisher": recipe.topology.kv_events.publisher,
                                "endpoint": recipe.topology.kv_events.endpoint,
                                "replay_endpoint": recipe.topology.kv_events.replay_endpoint,
                                "topic": (
                                    f"kv@{self.RANK_ENVIRONMENT['POD_IP']}"
                                    f":{record0['settings']['port']['value']}"
                                    f"@{record0['settings']['served-model-name']['value']}"
                                ),
                            },
                            separators=(",", ":"),
                        ),
                    ],
                )
                self.assertEqual(
                    added1,
                    [
                        "--nnodes", "2",
                        "--node-rank", "1",
                        "--master-addr", self.RANK_ENVIRONMENT["LWS_LEADER_ADDRESS"],
                        "--headless",
                    ],
                )

    def test_kv_topic_is_the_address_the_router_looks_up(self) -> None:
        recipe = self.recipe("qwen38-flash-next-gb10-tp2")
        plan = resolve_recipe(recipe, root=self.root)
        arguments = launch.kv_events_args(
            recipe.topology.kv_events,
            launch.Rank(0, 2, "leader"),
            plan.values,
            {"POD_IP": "10.42.7.19"},
        )
        self.assertEqual(arguments[:2], ["--enable-scale-out", "--kv-events-config"])
        config = json.loads(arguments[2])
        self.assertEqual(config["topic"], "kv@10.42.7.19:8888@qwen38-flash-next")
        self.assertTrue(config["enable_kv_cache_events"])
        # A follower publishes nothing: a second source for the same blocks
        # would be unroutable.
        self.assertEqual(
            launch.kv_events_args(
                recipe.topology.kv_events, launch.Rank(1, 2, "leader"), plan.values, {}
            ),
            [],
        )

    def test_missing_pod_ip_on_the_leader_is_a_configuration_error(self) -> None:
        recipe = self.recipe("deepseek-v4-flash-vision-gb10-tp2")
        plan = resolve_recipe(recipe, root=self.root)
        with self.assertRaises(ConfigError) as raised:
            launch.kv_events_args(
                recipe.topology.kv_events, launch.Rank(0, 2, "leader"), plan.values, {}
            )
        self.assertEqual(raised.exception.code, launch.EXIT_CONFIG)
        self.assertIn("POD_IP", str(raised.exception))

    def test_single_recipe_gets_no_topology_arguments(self) -> None:
        recipe = self.recipe("qwen38-27b")
        plan = resolve_recipe(recipe, root=self.root)
        rank = launch.read_rank(recipe.topology, {})
        self.assertEqual(
            launch.topology_args(rank, plan.values, recipe.topology.kv_events, {}), []
        )




@unittest.skipUnless(UPDATE, "generation pass only")
class GoldenGenerationTest(unittest.TestCase):
    """Rewrites image_tools/data/golden/<recipe>.json from the resolved data."""

    def test_write_goldens(self) -> None:
        if not UPSTREAM_AVAILABLE:
            self.skipTest(SKIP_REASON)
        directory = tempfile.TemporaryDirectory(dir=str(scratch_parent()), prefix="generate-")
        try:
            root = build_data_root(Path(directory.name))
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            for name in RECIPES:
                recipe = launch.load_recipe(name, root=RECIPE_DIR)
                golden_path(name).write_text(dump(record(resolve_recipe(recipe, root=root))))
                print(f"wrote {golden_path(name).relative_to(REPO)}")
        finally:
            directory.cleanup()


if __name__ == "__main__":
    if UPDATE:
        sys.argv.remove("--update")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GoldenGenerationTest)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    unittest.main()
