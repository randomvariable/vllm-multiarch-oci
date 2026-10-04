#!/usr/bin/env python3
"""Resolve model policy without importing vLLM, CUDA, or model checkpoints.

This module is a port of ``runtime/launcher.py`` and the policy half of
``runtime/cache.py`` from ``local-inference-lab/blackwell-llm-docker`` at
commit ``353efc679f631206e0b001e67047dea80ee6d76e``. Functions are kept in
upstream order and upstream shape so that ``launcher_parity_test`` and a
reviewer can diff the two files side by side.

The port is deliberate about three things:

* It reads the pinned upstream data files (``options.yaml``, ``schema.json``,
  ``platform-environment.json``, ``presets.yaml``, ``profiles/``,
  ``hardware/``, ``templates/``) relative to a ``root`` argument instead of
  upstream's module-directory ``ROOT``, because the image installs them under
  ``/opt/vllm-image/runtime`` while the site ships them next to a copy of this
  file.
* It never spawns a process, execs, parses a command line, or touches the
  network, so the same code answers ``--print-config`` in the image, in a CPU
  test, and in the browser under Pyodide. Where upstream reached for a
  subprocess or an optional third-party package, the effect is a parameter the
  caller supplies (``vllm_environment``, ``fetch_snapshot``, ``fetch_file``,
  the QSA source text).
* It adds one configuration layer upstream does not have: a ``recipe_layer``
  carrying our own accepted values and environment for a recipe, applied
  immediately after the deployment preset it names and before anything an
  operator supplies. The rule for every downstream refinement is *preset
  equivalence*: a recipe value must be treated exactly as upstream treats the
  same value coming from a preset. Upstream's presets.yaml describes itself as
  the overlays "owned by the container recipe", so a recipe is a preset in
  everything but its file. Concretely, ``recipe:`` joins a branch only where
  ``preset:`` already appears (linked options, the preset KV reduction, the
  capture-size extension); the branches that recognise only
  ``common:``/``model:``/``hardware:`` keep upstream's behaviour, because a
  preset-set value is not matched by them either. Every such site is marked
  ``PORT(recipe)``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from image_tools.launcher import ConfigError

# Where the image installs the pinned upstream policy data and our recipes.
DEFAULT_DATA_ROOT = Path("/opt/vllm-image/runtime")

JIT_PATHS = {
    "XDG_CACHE_HOME": "",
    "VLLM_CACHE_ROOT": "vllm",
    "VLLM_CACHE_DIR": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUTE_DSL_CACHE_DIR": "cute-dsl",
    "B12X_CUTE_COMPILE_CACHE_DIR": "b12x/cute",
    "B12X_COMPILE_CACHE_DIR": "b12x/compile",
    "SPARKINFER_COMPILE_CACHE_DIR": "b12x/compile",
    "CUDA_CACHE_PATH": "cuda",
    # These default to the home directory, which a recreated container loses.
    "TILELANG_CACHE_DIR": "tilelang",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
    "TVM_CACHE_DIR": "tvm",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "flash-attention-cute-dsl",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "NUMBA_CACHE_DIR": "numba",
    "CUPY_CACHE_DIR": "cupy",
}
NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Chat-template settings with this prefix name a file in root / "templates".
RUNTIME_TEMPLATE_PREFIX = "runtime:"
SECRET = re.compile(
    r"api[-_]?key|password|secret|authorization|access[-_]?token|hf_token",
    re.IGNORECASE,
)
# PORT: upstream imports this from runtime.replicas at call time; that module
# belongs to the launcher's replicas role, so the constant is inlined here to
# keep the resolver's import set at the standard library plus PyYAML.
PLE_SHARED_DIRECTORY = "/dev/shm/lil-ple"


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently selecting a value."""


# YAML 1.2 booleans: mode names such as "off" are strings, not boolean keys.
UniqueLoader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in rules if tag != "tag:yaml.org,2002:bool"]
    for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def _mapping(loader: UniqueLoader, node: yaml.MappingNode, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict:
    try:
        result = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    if not isinstance(result, dict):
        raise ConfigError(f"Expected a mapping in {path}")
    return result


# PORT: upstream validates profiles with jsonschema. Pyodide ships this module
# with PyYAML only, so the subset schema.json actually uses is enforced here.
# Unsupported keywords and types are refused rather than ignored, which keeps
# a schema edit from silently disabling a constraint.
_SCHEMA_TYPES = ("object", "array", "string", "integer")


def _json_type(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    raise ConfigError(f"Unsupported schema type {kind}")


# JSON Schema keywords this validator enforces. A keyword that constrains an
# instance and is not listed here is refused rather than ignored, so editing
# schema.json can never silently disable a check. Annotation keywords carry no
# constraint and are skipped.
_SCHEMA_KEYWORDS = {
    "type",
    "const",
    "enum",
    "pattern",
    "required",
    "properties",
    "additionalProperties",
    "propertyNames",
    "items",
    "minimum",
}
_SCHEMA_ANNOTATIONS = {
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "format",
}


def _validate_schema(instance: Any, schema: dict, defs: dict, trail: str) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/$defs/"):
            raise ConfigError(f"Unsupported schema reference {reference}")
        if reference[len("#/$defs/") :] not in defs:
            raise ConfigError(f"Unknown schema definition {reference}")
        schema = defs[reference[len("#/$defs/") :]]
    keywords = set(schema) - _SCHEMA_KEYWORDS - _SCHEMA_ANNOTATIONS
    if keywords:
        raise ConfigError(f"Unsupported schema keywords: {', '.join(sorted(keywords))}")
    if "type" in schema:
        kinds = [schema["type"]] if isinstance(schema["type"], str) else schema["type"]
        if not any(_json_type(instance, kind) for kind in kinds):
            raise ConfigError(f"{trail} is not of type {' or '.join(kinds)}")
    if "const" in schema and instance != schema["const"]:
        raise ConfigError(f"{trail} must be {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise ConfigError(f"{trail} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str):
        if not re.search(schema["pattern"], instance):
            raise ConfigError(f"{trail} does not match {schema['pattern']!r}")
    if "minimum" in schema and isinstance(instance, int):
        if instance < schema["minimum"]:
            raise ConfigError(f"{trail} is less than the minimum {schema['minimum']}")
    if isinstance(instance, list) and "items" in schema:
        for position, item in enumerate(instance):
            _validate_schema(item, schema["items"], defs, f"{trail}[{position}]")
    if not isinstance(instance, dict):
        return
    for required in schema.get("required", []):
        if required not in instance:
            raise ConfigError(f"{trail} is a required property: {required}")
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        child = f"{trail}.{key}" if trail else key
        if key in properties:
            _validate_schema(value, properties[key], defs, child)
        elif additional is False:
            raise ConfigError(f"{child} is not an allowed property")
        elif isinstance(additional, dict):
            _validate_schema(value, additional, defs, child)
    names = schema.get("propertyNames")
    if isinstance(names, dict):
        for key in instance:
            _validate_schema(key, names, defs, f"{trail}.{key}")


def platform_environment(*, root: Path = DEFAULT_DATA_ROOT) -> dict[str, str]:
    """Read foundation defaults below model policy and explicit user settings."""
    try:
        data = json.loads((root / "platform-environment.json").read_text())
    except (OSError, ValueError) as error:
        raise ConfigError("Cannot read platform-environment.json") from error
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "source_image", "environment"}
        or data["schema_version"] != 1
        or not isinstance(data["source_image"], str)
        or not isinstance(data["environment"], dict)
        or any(
            not NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value
            for key, value in data["environment"].items()
        )
    ):
        raise ConfigError("Invalid platform environment policy")
    return data["environment"]


def managed_environment_names(root: Path = DEFAULT_DATA_ROOT) -> frozenset[str]:
    """Every environment alias options.yaml declares as managed.

    The launcher removes these names from the child environment before it
    applies the resolved values, so a managed alias never reaches a second
    resolver; upstream does the same strip inside ``execute()``. Unmanaged
    runtime variables stay intact.
    """
    return frozenset(
        alias
        for spec in read_yaml(root / "options.yaml").values()
        for alias in spec["env"]
    )


def profile(kind: str, identifier: str, *, root: Path = DEFAULT_DATA_ROOT) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier):
        raise ConfigError(
            "Profile names must contain only lowercase letters, digits and hyphens"
        )
    path = (
        root / ("hardware" if kind == "hardware" else "profiles") / f"{identifier}.yaml"
    )
    result = read_yaml(path)
    try:
        schema = json.loads((root / "schema.json").read_text())
    except (OSError, ValueError) as error:
        raise ConfigError(f"Cannot read schema.json: {error}") from error
    try:
        _validate_schema(result, schema, schema.get("$defs", {}), "")
    except ConfigError as error:
        raise ConfigError(f"Invalid profile {identifier}: {error}") from error
    return result


def deployment_presets(*, root: Path = DEFAULT_DATA_ROOT) -> dict:
    """Read data-only deployment overlays owned by the container recipe."""
    data = read_yaml(root / "presets.yaml")
    if set(data) != {"schema_version", "presets"} or data["schema_version"] != 1:
        raise ConfigError("Unsupported deployment preset schema")
    presets = data["presets"]
    if not isinstance(presets, dict):
        raise ConfigError("Deployment presets must be a mapping")
    for name, item in presets.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or not isinstance(item, dict):
            raise ConfigError("Invalid deployment preset identity")
        required = {
            "profile",
            "hardware",
            "description",
            "options",
            "environment",
            "modes",
            "linked_options",
        }
        kv_fields = {"kv_bytes_per_extra_slot", "kv_bytes_for_external_cache"}
        optional = kv_fields | {"vllm_fallback"}
        if not required <= set(item) <= required | optional:
            raise ConfigError(f"Invalid deployment preset fields: {name}")
        fallback = item.get("vllm_fallback")
        if fallback is not None and (
            not isinstance(fallback, dict)
            or set(fallback) != {"requires_environment", "options"}
            or not isinstance(fallback["options"], dict)
            or not isinstance(fallback["requires_environment"], list)
            or not all(
                isinstance(key, str) and key in item["environment"]
                for key in fallback["requires_environment"]
            )
        ):
            raise ConfigError(f"Invalid preset vllm_fallback: {name}")
        for fields in kv_fields:
            value = item.get(fields, 0)
            if type(value) is not int or value < 0:
                raise ConfigError(f"Invalid preset {fields}: {name}")
        for key in ("profile", "hardware"):
            if not isinstance(item[key], str) or not re.fullmatch(
                r"[a-z][a-z0-9-]*", item[key]
            ):
                raise ConfigError(f"Invalid preset {key}: {name}")
        if not isinstance(item["description"], str) or not all(
            isinstance(item[key], dict)
            for key in ("options", "environment", "modes", "linked_options")
        ):
            raise ConfigError(f"Invalid deployment preset values: {name}")
    return presets


def deployment_preset(identifier: str, *, root: Path = DEFAULT_DATA_ROOT) -> dict:
    presets = deployment_presets(root=root)
    if identifier not in presets:
        raise ConfigError(f"Unknown deployment preset: {identifier}")
    return copy.deepcopy(presets[identifier])


def installed_source(package: str, relative: str) -> str | None:
    """Text of a file in an installed package, or None when it is missing.

    The launcher must not import vLLM, B12X or CUDA, so it reads source text.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        path = Path(location) / relative
        if path.is_file():
            return path.read_text()
    return None


def installed_vllm_environment() -> frozenset[str] | None:
    """Environment names the installed vLLM declares, or None without vLLM."""
    source = installed_source("vllm", "envs.py")
    if source is None:
        return None
    return frozenset(re.findall(r'^    "(VLLM_[A-Z0-9_]+)"', source, re.MULTILINE))


def installed_b12x_mxfp8_moe() -> bool:
    """Whether the installed vLLM and B12X run MXFP8 MoE experts on B12X.

    vLLM maps moe_backend b12x to its B12X_MXFP8 backend (vllm #958) and B12X
    prepares the mxfp8_e8m0_k32 source format (b12x #453).
    """
    oracle = installed_source(
        "vllm", "model_executor/layers/fused_moe/oracle/mxfp8.py"
    )
    formats = installed_source("b12x", "moe/fused_moe/source.py")
    return (
        oracle is not None
        and "Fp8MoeBackend.B12X_MXFP8" in oracle
        and formats is not None
        and '"mxfp8_e8m0_k32"' in formats
    )


def convert(name: str, value: Any, spec: dict) -> Any:
    kind = spec["type"]
    try:
        if kind == "string":
            value = str(value)
        elif kind == "integer":
            if isinstance(value, bool) or not re.fullmatch(r"[+-]?\d+", str(value)):
                raise ValueError
            value = int(value)
        elif kind == "int-or-auto":
            if str(value) not in {"auto", "None"}:
                value = convert(name, value, {"type": "integer"})
        elif kind == "number":
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        elif kind == "boolean":
            normalized = str(value).lower()
            if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ValueError
            value = normalized in {"true", "1", "yes", "on"}
        elif kind == "object":
            value = json.loads(value) if isinstance(value, str) else value
            if not isinstance(value, dict):
                raise ValueError
        elif kind == "integers":
            if isinstance(value, str):
                value = value.replace(",", " ").split()
            if not isinstance(value, list) or not value:
                raise ValueError
            value = [convert(name, item, {"type": "integer"}) for item in value]
        else:
            raise ConfigError(f"Unknown option type {kind}")
    except (ValueError, TypeError) as error:
        raise ConfigError(f"Invalid value for {name}; expected {kind}") from error
    if "enum" in spec and value not in spec["enum"]:
        raise ConfigError(f"{name} must be one of {', '.join(spec['enum'])}")
    return value


def parse_native(argv: list[str], specs: dict) -> tuple[dict, list[str]]:
    """Consume managed options; retain unknown native options without shell parsing."""
    values: dict[str, Any] = {}
    passthrough: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        index += 1
        if argument == "--":
            continue
        if not argument.startswith("--"):
            raise ConfigError(
                "Use --model for the checkpoint; native options must start with --"
            )
        name, equals, inline = argument[2:].partition("=")
        option, dot, field = name.partition(".")
        name = option.replace("_", "-") + dot + field
        negative = (
            name.startswith("no-")
            and name[3:] in specs
            and specs[name[3:]]["type"] == "boolean"
        )
        if negative:
            name = name[3:]
        root = name.split(".", 1)[0]
        if name in values:
            raise ConfigError(f"Specify --{name} only once")
        if root not in specs:
            if root in {
                "config",
                "kv-transfer-config",
                "kv-offloading-backend",
                "kv-offloading-size",
                "enable-cumem-allocator",
                "max-num-scheduled-tokens",
            }:
                raise ConfigError(
                    f"--{name} requires the external-cache/config-file integration; it cannot bypass profile validation"
                )
            passthrough.append(argument)
            while index < len(argv) and not argv[index].startswith("--"):
                passthrough.append(argv[index])
                index += 1
            continue
        if root != name and specs[root]["type"] != "object":
            raise ConfigError(f"--{root} does not accept dotted JSON fields")
        if specs[root]["type"] == "boolean":
            if negative and equals:
                raise ConfigError(f"--no-{name} does not take a value")
            value = not negative if not equals else inline
        elif equals:
            value = inline
        elif specs[root]["type"] == "integers" and root == name:
            value = []
            while index < len(argv) and not argv[index].startswith("--"):
                value.append(argv[index])
                index += 1
        else:
            if index >= len(argv) or argv[index].startswith("--"):
                raise ConfigError(f"--{name} requires a value")
            value = argv[index]
            index += 1
        if root == name:
            values[name] = convert(name, value, specs[name])
        else:
            try:
                values[name] = json.loads(value)
            except (ValueError, TypeError):
                values[name] = value
    return values, passthrough


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if SECRET.search(key) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1<redacted>@", value)
    return value


# LMCACHE_L2_CHECKPOINT_WRITES values that select a checkpoint store policy.
CHECKPOINT_STORE_POLICIES = {
    "on-reuse": "checkpoint_on_reuse",
    "on-evict": "checkpoint_on_evict",
}
# Seconds LMCache may spend writing RAM-only checkpoints to L2 at shutdown
# with on-evict, and the extra time the launcher waits before SIGKILL.
CHECKPOINT_SHUTDOWN_FLUSH_SECONDS = 30
STOP_GRACE_SECONDS = 10


@dataclass
class CacheService:
    argv: list[str]
    environment: dict[str, str]
    health_url: str
    shm_name: str
    shm_bytes: int
    startup_timeout: float
    directories: list[str] = field(default_factory=list)
    identity_required: bool = False
    namespace: str = ""
    # Seconds between SIGTERM and SIGKILL when the container stops.
    stop_grace: float = STOP_GRACE_SECONDS
    # Delete unused disk-tier namespaces of earlier images or settings.
    prune_stale_tiers: bool = False


def configure_cache(
    values, origins, environment, env_origins, identifier, runtime_identity
):
    """Derive worker/service settings once from the resolved model configuration."""

    def put(key, value):
        values[key], origins[key] = value, "derived:external-cache contract"

    def export(key, value):
        environment[key], env_origins[key] = (
            str(value),
            "derived:external-cache contract",
        )

    cache_mode = values["cache-mode"]
    if cache_mode == "vram":
        return None
    glm = identifier == "glm53-flash"
    qwen = identifier == "qwen38-flash-next"
    ds41 = identifier == "ds41-flash"
    if (
        not glm
        and not qwen
        and not ds41
        and identifier not in {"ds4-flash", "ds4-vision"}
    ):
        raise ConfigError(
            f"{identifier}: external cache is unsupported by this profile; Engram/PLE placement is independent"
        )
    if cache_mode == "native":
        if not (glm or qwen):
            raise ConfigError(
                "Native KV offload is implemented only by the GLM and Qwen profiles"
            )
        if values["cache-native-gib"] <= 0:
            raise ConfigError("cache-native-gib must be positive")
        if qwen:
            if values["decode-context-parallel-size"] != 1:
                raise ConfigError("Qwen native CPU cache requires DCP=1")
            if values.get("recurrent-checkpoint-policy", "auto") not in {
                "auto",
                "aligned",
            }:
                raise ConfigError("Qwen native CPU cache requires aligned checkpoints")
            if environment.get("VLLM_USE_SIMPLE_KV_OFFLOAD", "1") != "1":
                raise ConfigError(
                    "Qwen native CPU cache requires VLLM_USE_SIMPLE_KV_OFFLOAD=1; "
                    "the generic OffloadingConnector is not supported"
                )
            export("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
            put("recurrent-checkpoint-policy", "aligned")
        put("kv-offloading-backend", "native")
        put("kv-offloading-size", values["cache-native-gib"])
        put("enable-cumem-allocator", True)
        return None

    transfer = values["cache-transfer-mode"]
    engine = transfer == "engine_driven"
    if (qwen or ds41) and not engine:
        raise ConfigError(
            "Qwen and DS4.1 external cache require engine-driven transfer"
        )
    chunk = values["cache-object-tokens"]
    for key in (
        "cache-object-tokens",
        "cache-start-timeout",
        "cache-l1-gib",
        "cache-l1-init-gib",
        "cache-l2-gib",
        "cache-cpu-workers",
        "cache-l2-workers",
    ):
        if values[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if values["cache-l1-init-gib"] > values["cache-l1-gib"]:
        raise ConfigError("Initial L1 capacity cannot exceed its configured capacity")
    if "cache-gpu-workers" not in values:
        put("cache-gpu-workers", values["tensor-parallel-size"])
    if values["cache-gpu-workers"] < 1:
        raise ConfigError("cache-gpu-workers must be positive")

    for offset, key in (
        (10000, "cache-port"),
        (10001, "cache-http-port"),
        (10002, "cache-metrics-port"),
    ):
        if key not in values:
            put(key, values["port"] + offset)
    ports = [
        values[key]
        for key in ("port", "cache-port", "cache-http-port", "cache-metrics-port")
    ]
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != len(ports):
        raise ConfigError(
            "Model and cache ports must be distinct and in [1, 65535]; set explicit cache ports when automatic offsets overflow"
        )
    for key in ("cache-host", "cache-http-host"):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", values[key]):
            raise ConfigError(f"Invalid {key}")
    if "cache-instance" not in values:
        put("cache-instance", f"{identifier}-{values['port']}")
    if "cache-shm-name" not in values:
        put(
            "cache-shm-name",
            f"lmcache-{values['cache-instance']}-{values['cache-port']}",
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", values["cache-instance"]):
        raise ConfigError(
            "Cache instance must contain only letters, digits, dots, hyphens and underscores"
        )
    shm = values["cache-shm-name"] if engine else ""
    if engine and not re.fullmatch(r"[A-Za-z0-9._-]+", shm):
        raise ConfigError(
            "Engine-driven transfer requires a nonempty valid SHM name; no pickle fallback"
        )

    semantic = False
    if glm:
        tp = values["tensor-parallel-size"]
        if tp not in {2, 4, 8}:
            raise ConfigError("The GLM external-cache profile supports TP2, TP4 or TP8")
        if tp == 2 and not engine:
            raise ConfigError(
                "GLM TP2 requires engine-driven request-boundary cache transfer"
            )
        target_budget = values.get(
            "cache-target-tokens", values["max-num-batched-tokens"]
        )
        if target_budget != chunk:
            raise ConfigError(
                "GLM cache-object-tokens and target scheduler budget must match"
            )
        policy_key = "recurrent-checkpoint-policy"
        if origins[policy_key].startswith("model:") or values[policy_key] == "auto":
            put(policy_key, "request_boundaries" if engine else "aligned")
        semantic = values[policy_key] == "request_boundaries"
        if tp == 2 and not semantic:
            raise ConfigError(
                "GLM TP2 requires engine-driven request-boundary cache transfer"
            )
        if semantic and not engine:
            raise ConfigError(
                "Request-boundary checkpoint bundles require engine-driven transfer"
            )
        if origins["target-page-size"].startswith("model:"):
            put("target-page-size", "auto")
            export("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
        page = values["target-page-size"]
        if page != "auto" and chunk % (
            int(page) * values["decode-context-parallel-size"]
        ):
            raise ConfigError("Cache objects must contain complete DCP target pages")
        input_rows = target_budget
        if values["mode"] == "dflash2":
            input_rows += values["draft-tokens"] * values["max-num-seqs"]
        put("max-num-batched-tokens", input_rows)
        put("max-num-scheduled-tokens", target_budget)
        retention = 0 if semantic else chunk
        if (
            "prefix-cache-retention-interval" in values
            and not origins["prefix-cache-retention-interval"].startswith(
                ("model:", "common:", "derived:")
            )
            and values["prefix-cache-retention-interval"] not in {"auto", retention}
        ):
            raise ConfigError(
                "Explicit checkpoint retention conflicts with the cache object geometry"
            )
        put("prefix-cache-retention-interval", retention)
        if engine and origins["gpu-memory-utilization"].startswith("model:"):
            put("gpu-memory-utilization", 0.950)
    elif qwen:
        # GDN state must be restored with its exact attention/PLE boundary.
        # Independent aligned chunks are not a substitute for that bundle.
        policy = values.get("recurrent-checkpoint-policy", "auto")
        if policy not in {"auto", "request_boundaries"}:
            raise ConfigError(
                "Qwen external cache requires request-boundary checkpoints"
            )
        if values.get("prefix-cache-retention-interval", 0) not in {0, "auto"}:
            raise ConfigError(
                "Qwen external cache uses exact boundaries, not periodic retention"
            )
        semantic = True
        put("recurrent-checkpoint-policy", "request_boundaries")
        put("prefix-cache-retention-interval", 0)
    else:
        if chunk % (values["block-size"] * values["decode-context-parallel-size"]):
            raise ConfigError("Cache object tokens must align to DS4 DCP cache pages")
        if (
            not ds41
            and engine
            and values["tensor-parallel-size"] == 2
            and (values["max-model-len"] == -1 or values["max-model-len"] >= 1048576)
            and origins["gpu-memory-utilization"].startswith("model:")
        ):
            put("gpu-memory-utilization", 0.970)

    connector = {
        "kv_connector": "LMCacheRecurrentCheckpointConnector"
        if semantic
        else "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.recurrent_checkpoint_connector"
        if semantic
        else "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": values["cache-load-failure-policy"],
        "kv_connector_extra_config": {
            "lmcache.mp.host": values["cache-host"],
            "lmcache.mp.port": values["cache-port"],
            "lmcache.mp.mp_transfer_mode": transfer,
        },
    }
    put("kv-transfer-config", connector)
    if not engine:
        if glm:
            put("enable-cumem-allocator", True)
            interposer = "/opt/lmcache/lib/liblmcache_cumem_shareable.so"
            export(
                "LD_PRELOAD",
                interposer
                + (
                    ":" + environment["LD_PRELOAD"]
                    if environment.get("LD_PRELOAD")
                    else ""
                ),
            )
            export("LMCACHE_CUMEM_BROKER_DIR", values["cache-broker-directory"])
        else:
            allocator = environment.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"
            )
            export(
                "PYTORCH_CUDA_ALLOC_CONF",
                allocator.replace(
                    "expandable_segments:True", "expandable_segments:False"
                ),
            )

    argv = [
        "/opt/venv/bin/lmcache",
        "server",
        "--instance-id",
        values["cache-instance"],
        "--host",
        values["cache-host"],
        "--port",
        str(values["cache-port"]),
        "--http-host",
        values["cache-http-host"],
        "--http-port",
        str(values["cache-http-port"]),
        "--prometheus-port",
        str(values["cache-metrics-port"]),
        "--chunk-size",
        str(chunk),
        "--supported-transfer-mode",
        transfer,
        "--separate-object-groups",
        "--l1-size-gb",
        str(values["cache-l1-gib"]),
        "--l1-init-size-gb",
        str(values["cache-l1-init-gib"]),
        "--max-gpu-workers",
        str(values["cache-gpu-workers"]),
        "--max-cpu-workers",
        str(values["cache-cpu-workers"]),
        "--eviction-policy",
        "LRU",
        "--l2-prefetch-policy",
        values["cache-prefetch-policy"],
    ]
    argv += ["--no-l1-use-lazy", "--shm-name", shm] if engine else ["--l1-use-lazy"]
    # A request-boundary checkpoint carries the complete recurrent state and
    # the next turn of the same conversation supersedes it. on-reuse keeps
    # new checkpoints in RAM until a restore proves them useful; on-evict
    # writes the current checkpoint of a conversation when it leaves RAM and
    # at shutdown, and never writes superseded ones.
    store_policy = "default"
    checkpoint_writes = values["cache-l2-checkpoint-writes"]
    if (
        values["cache-l2-enabled"]
        and checkpoint_writes in CHECKPOINT_STORE_POLICIES
        and not semantic
        and origins["cache-l2-checkpoint-writes"].startswith(("model:", "common:"))
    ):
        # The profile default applies to request-boundary checkpoints only;
        # other checkpoint policies keep writing every cache object.
        checkpoint_writes = "always"
        values["cache-l2-checkpoint-writes"] = checkpoint_writes
        origins["cache-l2-checkpoint-writes"] = (
            "derived:no request-boundary checkpoints"
        )
    if values["cache-l2-enabled"] and checkpoint_writes in CHECKPOINT_STORE_POLICIES:
        if not semantic:
            raise ConfigError(
                f"LMCACHE_L2_CHECKPOINT_WRITES={checkpoint_writes} applies only "
                "to request-boundary checkpoints"
            )
        store_policy = CHECKPOINT_STORE_POLICIES[checkpoint_writes]
    stop_grace = STOP_GRACE_SECONDS
    if store_policy == "checkpoint_on_evict":
        argv += [
            "--checkpoint-shutdown-flush-seconds",
            str(CHECKPOINT_SHUTDOWN_FLUSH_SECONDS),
        ]
        stop_grace += CHECKPOINT_SHUTDOWN_FLUSH_SECONDS
    if glm:
        argv += ["--hash-algorithm", "blake3", "--max-workers", "8"]
        if store_policy != "default":
            argv += ["--l2-store-policy", store_policy]
    else:
        argv += [
            "--l1-write-ttl-seconds",
            "600",
            "--l1-read-ttl-seconds",
            "300",
            "--eviction-trigger-watermark",
            "0.90",
            "--eviction-ratio",
            "0.10",
            "--l2-store-policy",
            store_policy,
            "--worker-reap-timeout-seconds",
            "120",
            "--worker-registration-grace-seconds",
            "3600",
        ]
        connector["kv_connector_extra_config"].update(
            {"lmcache.mp.mq_timeout": 60.0, "lmcache.mp.heartbeat_interval": 10.0}
        )
    # Model identity must be resolved before persistent storage is opened. A
    # dry-run uses a visibly unresolved namespace and performs no Hub/disk I/O.
    layout = {
        key: values[key]
        for key in (
            "model",
            "tensor-parallel-size",
            "decode-context-parallel-size",
            "kv-cache-dtype",
            "block-size",
            "mode",
            "draft-tokens",
            "cache-object-tokens",
        )
    }
    layout.update(
        {
            key: values[key]
            for key in (
                "target-page-size",
                "recurrent-page-size",
                "dcp-ckv-gather",
                "cp-kv-cache-interleave-size",
                "recurrent-checkpoint-policy",
                "speculative-config",
            )
            if key in values
        }
    )
    if qwen or ds41:
        layout.update(
            {
                key: values[key]
                for key in (
                    "swa-block-size",
                    "mamba-cache-mode",
                    "mamba-ssm-cache-dtype",
                )
                if key in values
            }
        )
    layout["runtime"] = runtime_identity or "UNBOUND-RUNTIME"
    layout_digest = hashlib.sha256(
        json.dumps(layout, sort_keys=True).encode()
    ).hexdigest()
    path = Path(values["cache-directory"])
    if not path.is_absolute() or path == Path("/"):
        raise ConfigError("Cache directory must be an absolute, dedicated directory")
    namespace = str(path / identifier / layout_digest / "UNRESOLVED-CHECKPOINT")
    directories = []
    if values["cache-l2-enabled"]:
        l2 = {
            "type": "fs_native",
            "base_path": namespace,
            "num_workers": values["cache-l2-workers"],
            "use_odirect": values["cache-l2-odirect"],
            "max_capacity_gb": values["cache-l2-gib"],
        }
        if glm or qwen:
            l2["eviction"] = {
                "eviction_policy": "LRU",
                "trigger_watermark": 0.8,
                "eviction_ratio": 0.2,
            }
        argv += ["--l2-adapter", json.dumps(l2, separators=(",", ":"))]
        if glm and values["cache-prefetch-policy"] == "retain":
            argv += ["--emergency-evict-for-prefetch"]
        directories += [namespace]
        if semantic:
            argv += [
                "--checkpoint-index-path",
                str(Path(namespace) / "semantic-directory.sqlite3"),
            ]
    if glm and not engine:
        directories += [values["cache-broker-directory"]]
    probe = values["cache-http-host"]
    probe = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(probe, probe)
    if ":" in probe:
        probe = f"[{probe}]"
    return CacheService(
        argv,
        {"CUDA_VISIBLE_DEVICES": "", "CUDA_MODULE_LOADING": "LAZY"} if engine else {},
        f"http://{probe}:{values['cache-http-port']}/healthcheck",
        # LMCache's --shm-name is a suffix; the allocator owns this OS name.
        f"lmcache_l1_pool_{shm}" if engine else "",
        int(values["cache-l1-gib"] * 1024**3) if engine else 0,
        values["cache-start-timeout"],
        directories,
        semantic or values["cache-l2-enabled"],
        namespace,
        stop_grace,
        bool(values.get("cache-l2-prune-stale")),
    )


# Asked in a GPU-free child upstream; here decided from source text, because
# the resolver never spawns a process. The probe upstream runs is
#   from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSABackend as B
#   raise SystemExit(0 if B.supports_kv_connector() else 3)
# and the module it imports, relative to the installed vllm package:
QSA_ATOMIC_TRANSFER_SOURCE = "models/qwen4_exp/nvidia/b12x_qsa.py"
_QSA_ATOMIC_TRANSFER_CLASS = "Qwen4ExpQSABackend"
_QSA_ATOMIC_TRANSFER_METHOD = "supports_kv_connector"


def qsa_atomic_transfer_supported(vllm_qsa_source: str | None) -> bool:
    """Whether the installed vLLM keeps QSA checkpoints atomic (vllm #864).

    ``vllm_qsa_source`` is the text of ``QSA_ATOMIC_TRANSFER_SOURCE`` in the
    installed vLLM, which the caller reads with :func:`installed_source`; the
    launcher asks for it so this module never spawns a process. Absent source,
    an absent backend class, an absent capability, and a capability whose
    answer is not the literal ``True`` all count as unsupported, which is the
    same refusal upstream reaches by exiting non-zero. A vLLM that computed the
    answer at call time would have to be asked in a child process; nothing in
    the pinned lineage does.
    """
    if vllm_qsa_source is None:
        return False
    try:
        tree = ast.parse(vllm_qsa_source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.ClassDef)
            or node.name != _QSA_ATOMIC_TRANSFER_CLASS
        ):
            continue
        for member in node.body:
            if (
                isinstance(member, ast.FunctionDef)
                and member.name == _QSA_ATOMIC_TRANSFER_METHOD
            ):
                # Only a bare `return True` is a declaration. Parsing, never
                # executing, the installed module is what keeps this pure; a
                # conditional or computed answer cannot be decided here.
                return (
                    len(member.body) == 1
                    and isinstance(member.body[0], ast.Return)
                    and isinstance(member.body[0].value, ast.Constant)
                    and member.body[0].value.value is True
                )
        return False
    return False


@dataclass
class ResolvedPlan:
    profile: str
    hardware: str
    preset: str | None
    recipe: str | None
    values: dict
    origins: dict[str, str]
    environment: dict[str, str]
    environment_origins: dict[str, str]
    argv: list[str]
    passthrough: list[str]
    warnings: list[str]
    cache_service: dict | None = None
    # Drafter shipped inside the target checkpoint; resolved to a local path
    # at launch, so printing a configuration never downloads anything.
    draft_subfolder: str | None = None


def public(plan: ResolvedPlan) -> dict:
    # Build public argv from redacted values; never dump the process environment.
    public_argv = list(plan.argv)
    redact_next = False
    for index, argument in enumerate(public_argv):
        if argument.startswith("--"):
            redact_next = False
        if redact_next:
            public_argv[index] = "<redacted>"
        elif argument.startswith("--") and SECRET.search(argument.split("=", 1)[0]):
            redact_next = True
            if "=" in argument:
                public_argv[index] = argument.split("=", 1)[0] + "=<redacted>"
        else:
            try:
                decoded = json.loads(argument)
            except (ValueError, TypeError):
                public_argv[index] = _redact(argument)
            else:
                if isinstance(decoded, (dict, list)):
                    public_argv[index] = json.dumps(
                        _redact(decoded), separators=(",", ":")
                    )
    result = {
        "schema_version": 1,
        "status": "implemented",
        "qualification": "CPU configuration tests only; no GPU performance claim",
        "profile": plan.profile,
        "hardware": plan.hardware,
        "argv": public_argv,
        "settings": {
            key: {"value": _redact(value), "source": plan.origins[key]}
            for key, value in plan.values.items()
        },
        "environment": {
            key: {
                "value": "<redacted>" if SECRET.search(key) else _redact(value),
                "source": plan.environment_origins[key],
            }
            for key, value in sorted(plan.environment.items())
        },
        "warnings": plan.warnings,
        "cache_service": _redact(plan.cache_service) if plan.cache_service else None,
    }
    # Our two extra keys name the data layers upstream does not have; the
    # parity harness compares only the keys upstream's public() produces.
    result["preset"] = plan.preset
    result["recipe"] = plan.recipe
    return result


def _recipe(recipe_layer: dict | None) -> tuple[str | None, dict, dict, str]:
    """Validate the recipe overlay and return its name, options, env and source.

    A recipe layer is the shape ``{"options": {...}, "environment": {...},
    "source": "recipe:<name>"}``: our accepted native arguments and environment
    for one deployment, applied at the deployment-preset precedence slot.
    """
    if recipe_layer is None:
        return None, {}, {}, ""
    if not isinstance(recipe_layer, dict):
        raise ConfigError("Recipe layer must be a mapping")
    if set(recipe_layer) - {"options", "environment", "source"}:
        raise ConfigError("Recipe layer accepts only options, environment and source")
    for key in ("options", "environment"):
        if not isinstance(recipe_layer.get(key, {}), dict):
            raise ConfigError(f"Recipe {key} must be a mapping")
    source = recipe_layer.get("source")
    if not isinstance(source, str) or not source.startswith("recipe:"):
        raise ConfigError("Recipe layer requires a source prefixed with recipe:")
    name = source[len("recipe:") :]
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ConfigError(
            "Recipe names must contain only lowercase letters, digits and hyphens"
        )
    for key, value in recipe_layer.get("environment", {}).items():
        if not NAME.fullmatch(key) or not isinstance(value, str):
            raise ConfigError(
                "Recipe environment requires valid names and string values"
            )
    return (
        name,
        recipe_layer.get("options", {}),
        recipe_layer.get("environment", {}),
        source,
    )


def resolve(
    identifier: str,
    hardware: str = "native",
    *,
    preset: str | None = None,
    recipe_layer: dict | None = None,
    env: dict[str, str] | None = None,
    config: dict | None = None,
    argv: list[str] | None = None,
    cli_env: dict[str, str] | None = None,
    runtime_identity: str | None = None,
    vllm_environment: frozenset[str] | None = None,
    root: Path = DEFAULT_DATA_ROOT,
) -> ResolvedPlan:
    incoming = dict(os.environ if env is None else env)
    config = config or {}
    if set(config) - {"options", "environment"}:
        raise ConfigError("Settings file accepts only options and environment mappings")
    for key in ("options", "environment"):
        if not isinstance(config.get(key, {}), dict):
            raise ConfigError(f"Settings {key} must be a mapping")
    for key, value in config.get("environment", {}).items():
        if not NAME.fullmatch(key) or not isinstance(value, str):
            raise ConfigError(
                "Settings environment requires valid names and string values"
            )
    recipe_name, recipe_options, recipe_environment, recipe_source = _recipe(
        recipe_layer
    )
    specs = read_yaml(root / "options.yaml")
    common, model, hw = (
        profile("common", "common", root=root),
        profile("model", identifier, root=root),
        profile("hardware", hardware, root=root),
    )
    deployment = deployment_preset(preset, root=root) if preset else None
    if deployment and deployment["profile"] != identifier:
        raise ConfigError("The deployment preset belongs to a different model profile")
    fallback_reason = None
    if deployment and deployment.get("vllm_fallback") and vllm_environment is not None:
        # A preset that relies on vLLM features newer than the installed vLLM
        # (for example the same recipe in a channel with an older vLLM) keeps
        # its previously qualified values instead.
        fallback = deployment["vllm_fallback"]
        missing = sorted(set(fallback["requires_environment"]) - vllm_environment)
        if missing:
            deployment["options"].update(fallback["options"])
            for name in fallback["requires_environment"]:
                deployment["environment"].pop(name, None)
            fallback_reason = "installed vLLM lacks " + ", ".join(missing)
    if deployment:
        model = copy.deepcopy(model)
        for mode_name, overrides in deployment["modes"].items():
            if (
                mode_name not in model["modes"]
                or not isinstance(overrides, dict)
                or set(overrides) - {"draft_tokens", "config"}
            ):
                raise ConfigError(f"Invalid preset mode: {mode_name}")
            if "draft_tokens" in overrides:
                model["modes"][mode_name]["draft_tokens"] = overrides["draft_tokens"]
            if "config" in overrides:
                if not isinstance(overrides["config"], dict):
                    raise ConfigError("Preset mode config must be a mapping")
                model["modes"][mode_name].setdefault("config", {}).update(
                    overrides["config"]
                )
    cli, passthrough = parse_native(argv or [], specs)
    values, origins = {}, {}
    environment, env_origins = {}, {}

    def set_value(key, value, source):
        if key not in specs:
            raise ConfigError(f"Unknown managed setting: {key}")
        values[key] = convert(key, value, specs[key])
        origins[key] = source

    def set_env(key, value, source):
        if not NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
            raise ConfigError(
                "Environment requires valid names and NUL-free string values"
            )
        environment[key] = value
        env_origins[key] = source

    platform = platform_environment(root=root)
    for key, value in platform.items():
        set_env(key, value, "platform:foundation")
    for layer in (common, model, hw):
        source = f"{layer['kind']}:{layer['id']}"
        for key, value in layer["defaults"].items():
            set_value(key, value, source)
        for key, value in layer["environment"].items():
            set_env(key, value, source)
    for key, value in hw.get("model_environment", {}).get(identifier, {}).items():
        set_env(key, value, f"hardware:{hardware}/{identifier}")
    if deployment:
        source = f"preset:{preset}"
        if fallback_reason:
            source += f" (fallback: {fallback_reason})"
        for key, value in deployment["options"].items():
            set_value(key, value, source)
        for key, value in deployment["environment"].items():
            set_env(key, value, f"preset:{preset}")
    # PORT(recipe): our overlay sits in the same precedence slot as the preset,
    # after it and before anything an operator supplies, so a recipe wins over
    # the preset it names and loses to --settings, the environment and the CLI.
    if recipe_name:
        for key, value in recipe_options.items():
            set_value(key, value, recipe_source)
        for key, value in recipe_environment.items():
            set_env(key, value, recipe_source)

    explicit_env = {**incoming, **config.get("environment", {}), **(cli_env or {})}
    for key, value in config.get("options", {}).items():
        if key not in cli:
            set_value(key, value, "settings:options")
    for key, spec in specs.items():
        if key in cli:
            continue
        candidates = [
            (alias, (cli_env or {})[alias])
            for alias in spec["env"]
            if alias in (cli_env or {})
        ]
        candidate_source = "cli:environment"
        if not candidates:
            if key in config.get("options", {}):
                continue
            candidates = [
                (alias, config.get("environment", {})[alias])
                for alias in spec["env"]
                if alias in config.get("environment", {})
            ]
            candidate_source = "settings:environment"
        if not candidates:
            candidates = [
                (alias, incoming[alias]) for alias in spec["env"] if alias in incoming
            ]
            candidate_source = "environment"
        normalized = []
        for alias, raw in candidates:
            if alias == "LMCACHE_MODE":
                cache_modes = {
                    "off": ("vram", False),
                    "0": ("vram", False),
                    "ram": ("lmcache", False),
                    "memory": ("lmcache", False),
                    "1": ("lmcache", False),
                    "disk": ("lmcache", True),
                    "ram-disk": ("lmcache", True),
                    "memory-disk": ("lmcache", True),
                }
                if str(raw).lower() not in cache_modes:
                    raise ConfigError("LMCACHE_MODE must select off, ram or disk")
                mode, l2 = cache_modes[str(raw).lower()]
                raw = mode if key == "cache-mode" else l2
            if alias == "LMCACHE_ENABLED":
                if raw not in {"0", "1"}:
                    raise ConfigError("LMCACHE_ENABLED must be 0 or 1")
                raw = "lmcache" if raw == "1" else "vram"
            if key == "kv-cache-dtype" and raw == "fp8_ds_mla":
                raw = "fp8"
            if key == "mode" and raw in {"dflash", "none"}:
                raw = {"dflash": "dflash2", "none": "off"}[raw]
            normalized.append((alias, convert(key, raw, spec)))
        if normalized:
            if any(value != normalized[0][1] for _, value in normalized[1:]):
                raise ConfigError(
                    f"Conflicting environment aliases for {key}: {', '.join(alias for alias, _ in normalized)}"
                )
            set_value(
                key,
                normalized[0][1],
                candidate_source + ":" + ",".join(alias for alias, _ in normalized),
            )
    for key, value in cli.items():
        if "." not in key:
            set_value(key, value, "cli")
    if "generation-config" in cli and origins.get(
        "override-generation-config", ""
    ).startswith(("model:", "common:")):
        # Selecting a native generation-config file owns the sampling defaults.
        values.pop("override-generation-config", None)
        origins.pop("override-generation-config", None)

    # Dotted JSON CLI fields refine an explicitly supplied root or its resolved default.
    for key, value in cli.items():
        if "." not in key:
            continue
        root_key, *parts = key.split(".")
        target = values.setdefault(root_key, {})
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ConfigError(f"Cannot refine non-object parent in --{key}")
        if not isinstance(target, dict):
            raise ConfigError(f"Cannot refine non-object --{root_key}")
        target[parts[-1]] = value
        origins[root_key] = "cli:json-fields"

    # Environment values are explicit only in a model-neutral image. Its build
    # contract is checked before execution; value-equality origin guessing is forbidden.
    consumed = {alias for spec in specs.values() for alias in spec["env"]}
    for key, value in explicit_env.items():
        if key in consumed:
            continue
        if (
            key in environment
            or key.startswith(
                (
                    "VLLM_",
                    "B12X_",
                    "NCCL_",
                    "CUTE_",
                    "TRITON_",
                    "TORCHINDUCTOR_",
                    "CUDA_",
                    "SPARKINFER_",
                    "TILELANG_",
                    "TVM_",
                    "TORCH_EXTENSIONS_",
                    "FLASHINFER_",
                    "FLASH_ATTENTION_",
                    "NUMBA_",
                    "CUPY_",
                    "INSTANTTENSOR_",
                    "SAFETENSORS_",
                )
            )
            or key
            in {
                "OMP_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
                "INSTANTTENSOR_BACKEND",
                "XDG_CACHE_HOME",
                "LD_PRELOAD",
            }
        ):
            source = (
                "cli:environment"
                if key in (cli_env or {})
                else "settings:environment"
                if key in config.get("environment", {})
                else "environment"
            )
            set_env(key, value, source)
    for key, value in config.get("environment", {}).items():
        if key not in consumed:
            set_env(key, value, "settings:environment")
    for key, value in (cli_env or {}).items():
        if key not in consumed:
            set_env(key, value, "cli:environment")

    def derive(key, value, reason):
        values[key] = value
        origins[key] = f"derived:{reason}"

    if deployment:
        for target, source in deployment["linked_options"].items():
            if target not in specs or source not in values:
                raise ConfigError("Preset option dependency is not declared")
            # PORT(recipe): a recipe occupies the deployment preset's
            # precedence slot, so a declared link refines a value the recipe
            # set just as it refines one the preset set.
            if origins.get(target, "").startswith(
                ("preset:", "model:", "common:", "recipe:")
            ):
                set_value(target, values[source], f"derived:preset link to {source}")
        # A preset's fixed KV size is qualified at its own request-slot count.
        # Each additional slot needs working memory (CUDA graphs up to the
        # larger verifier-row count, sampler and state buffers), so the KV
        # allocation shrinks unless the operator set it explicitly.
        # The external cache connector keeps its own GPU buffers.
        # PORT(recipe): a KV allocation the recipe pinned is a data-layer value
        # in the same sense a preset's is, so it still absorbs the per-slot and
        # external-cache cost; only an operator value is left untouched.
        preset_kv = origins.get("kv-cache-memory-bytes", "").startswith(
            ("preset:", "recipe:")
        )
        per_slot = deployment.get("kv_bytes_per_extra_slot", 0)
        base_slots = deployment["options"].get("max-num-seqs")
        reductions = []
        if per_slot and base_slots and values.get("max-num-seqs", 0) > base_slots:
            extra = values["max-num-seqs"] - base_slots
            reductions.append(
                (extra * per_slot, f"{extra} request slots above the preset")
            )
        external = deployment.get("kv_bytes_for_external_cache", 0)
        if external and values.get("cache-mode", "vram") != "vram":
            reductions.append((external, "external cache buffers"))
        if preset_kv and reductions:
            derive(
                "kv-cache-memory-bytes",
                values["kv-cache-memory-bytes"] - sum(size for size, _ in reductions),
                "; ".join(reason for _, reason in reductions),
            )

    # A repository-specific code revision must not leak to an operator's model.
    if (
        "revision" not in values
        and values["model"] == model["defaults"]["model"]
        and model.get("checkpoint_revision")
    ):
        derive(
            "revision", model["checkpoint_revision"], "profile checkpoint/code revision"
        )
    if (
        values.get("trust-remote-code")
        and "revision" in values
        and "code-revision" not in values
    ):
        derive(
            "code-revision",
            values["revision"],
            "remote code follows selected checkpoint",
        )

    if identifier == "mimo26-flash" and "max-num-scheduled-tokens" not in values:
        # Qualified MiMo split: half of each step's rows for target tokens,
        # the rest for DFlash verification rows (vllm #881: 4096 / 2048).
        derive(
            "max-num-scheduled-tokens",
            values["max-num-batched-tokens"] // 2,
            "MiMo target share of the step",
        )

    if identifier == "qwen38-flash-next":
        # B12X QSA shards compressed KV across DCP ranks in groups of four
        # tokens and refuses vLLM's default interleave of one, so every
        # DCP > 1 deployment needs this unless the operator chose a value.
        if values["decode-context-parallel-size"] > 1:
            for key in ("cp-kv-cache-interleave-size", "dcp-kv-cache-interleave-size"):
                if key not in values or origins[key].startswith(
                    ("common:", "model:", "hardware:")
                ):
                    derive(key, 4, "QSA DCP interleave")
        context_length = values["max-model-len"]
        base_context_length = 262144
        if context_length > base_context_length and "hf-overrides" not in values:
            if context_length > 1048576:
                raise ConfigError(
                    "Qwen3.8 Flash Next contexts above 1048576 tokens require "
                    "an explicit HF_OVERRIDES configuration"
                )
            # The checkpoint advertises 262144 positions. Extend its text
            # config before vLLM constructs both target and MTP draft models.
            factor = 2 if context_length <= 524288 else 4
            derive(
                "hf-overrides",
                {
                    "text_config": {
                        "max_position_embeddings": context_length,
                        "rope_parameters": {
                            "mrope_interleaved": True,
                            "mrope_section": [11, 11, 10],
                            "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000,
                            "rope_type": "yarn",
                            "factor": factor,
                            "original_max_position_embeddings": base_context_length,
                        },
                    }
                },
                f"Qwen3.8 YaRN for {context_length} tokens",
            )

    replicas = values["replicas"]
    ple_copy_per_replica = False
    if replicas < 1:
        raise ConfigError("replicas must be at least 1")
    if replicas > 1:
        for key in (
            "tensor-parallel-size",
            "pipeline-parallel-size",
            "decode-context-parallel-size",
            "data-parallel-size",
        ):
            if values.get(key, 1) != 1:
                raise ConfigError(
                    f"replicas runs independent single-GPU servers; {key} must be 1, "
                    f"got {values[key]}"
                )
        # One host-RAM PLE table serves every replica instead of one each; a
        # vLLM without shared tables keeps a copy per replica.
        if (
            identifier == "qwen38-flash-next"
            and environment.get("VLLM_PLE_CPU_OFFLOAD") == "1"
            and "VLLM_PLE_TABLE_MEMORY" not in explicit_env
        ):
            ple_copy_per_replica = (
                vllm_environment is not None
                and "VLLM_PLE_SHARED_TABLE_DIR" not in vllm_environment
            )
            if not ple_copy_per_replica:
                set_env("VLLM_PLE_TABLE_MEMORY", "shared", "derived:replicas")
                if "VLLM_PLE_SHARED_TABLE_DIR" not in explicit_env:
                    set_env(
                        "VLLM_PLE_SHARED_TABLE_DIR",
                        PLE_SHARED_DIRECTORY,
                        "derived:replicas",
                    )

    if explicit_env.get("EXTRA_VLLM_ARGS"):
        raise ConfigError(
            "EXTRA_VLLM_ARGS is ambiguous shell text; pass native CLI arguments after --"
        )
    for obsolete in (
        "BACKEND",
        "MODE",
        "SPEC_MODE",
        "DS4_OMP_NUM_THREADS",
        "DS4_MAX_CUDAGRAPH_CAPTURE_SIZE",
        "DS4_CUDAGRAPH_CAPTURE_SIZES",
    ):
        if obsolete in explicit_env:
            raise ConfigError(
                f"{obsolete} belongs to a compatibility wrapper; use the documented native option or canonical environment variable"
            )
    fairness = explicit_env.get("FAIRNESS_ENGINE", "compute_share")
    if fairness not in {"none", "compute_share"}:
        raise ConfigError(
            "FAIRNESS_ENGINE must be none or compute_share; micro_slicing is unsupported"
        )
    if fairness == "none" and "prefill-compute-share" not in cli:
        values.pop("prefill-compute-share", None)
        origins.pop("prefill-compute-share", None)

    if (
        "draft-tokens" in values
        and values["draft-tokens"] > 0
        and values["mode"] == "off"
        and origins["mode"].startswith("model:")
    ):
        derive("mode", "mtp", "positive MTP depth")
    mode = values["mode"]
    draft_subfolder = None
    if mode not in model["modes"]:
        raise ConfigError(f"{identifier} does not define mode {mode}")
    if "speculative-config" not in values:
        tokens = values.get("draft-tokens", model["modes"][mode].get("draft_tokens", 0))
        if tokens < 0:
            raise ConfigError("draft-tokens must be nonnegative")
        if tokens == 0:
            mode = "off"
            derive("mode", "off", "zero draft tokens")
        elif mode == "off":
            raise ConfigError("mode off conflicts with positive draft-tokens")
        if mode != "off":
            spec = copy.deepcopy(model["modes"][mode]["config"])
            spec["num_speculative_tokens"] = tokens
            if identifier == "ds41-flash":
                spec.update(
                    draft_tensor_parallel_size=values["tensor-parallel-size"],
                    enable_adaptive_verification=values["adaptive-verification"],
                    adaptive_verification_cost_scale=values[
                        "adaptive-verification-cost-scale"
                    ],
                )
            elif identifier == "mimo26-flash":
                spec["draft_tensor_parallel_size"] = values["tensor-parallel-size"]
            elif identifier.startswith("ds4-") and mode == "dspark":
                spec["model"] = values["model"]
            if "draft-model" in values:
                spec["model"] = values["draft-model"]
            elif model["modes"][mode].get("draft_subfolder"):
                draft_subfolder = model["modes"][mode]["draft_subfolder"]
                spec["model"] = f"{values['model']}/{draft_subfolder}"
            if "draft-revision" in values:
                spec["revision"] = values["draft-revision"]
            elif mode in {"mtp", "dspark"} and "revision" in values:
                spec["revision"] = values["revision"]
            derive(
                "speculative-config", spec, f"{mode} policy and resolved draft settings"
            )
    else:
        method = values["speculative-config"].get("method")
        mode = "dflash2" if method == "dflash" else method
        if mode not in model["modes"] or mode == "off":
            raise ConfigError(
                "Explicit speculative-config method is not defined by this model profile"
            )
        derive("mode", mode, "explicit speculative-config")
    derive(
        "draft-tokens",
        values.get("speculative-config", {}).get("num_speculative_tokens", 0),
        "effective proposal width",
    )

    if identifier == "glm53-flash":
        for name, key in (
            ("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "target-page-size"),
            ("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "recurrent-page-size"),
        ):
            set_env(name, str(values[key]), origins[key])
        gather = values["dcp-ckv-gather"]
        set_env(
            "VLLM_B12X_MLA_CKV_GATHER",
            str(int(values["decode-context-parallel-size"] > 1))
            if gather == "auto"
            else gather,
            "derived:DCP CKV policy",
        )
        if mode == "off" and not any(
            name in explicit_env
            for name in (
                "VLLM_B12X_DENSE_ACTIVATION_MODE",
                "VLLM_B12X_NVFP4_ACTIVATION_MODE",
            )
        ):
            set_env(
                "VLLM_B12X_NVFP4_ACTIVATION_MODE",
                "quantized",
                "derived:GLM non-speculative activation policy",
            )
        if (
            values["recurrent-checkpoint-policy"] == "aligned"
            and "prefix-cache-retention-interval" not in values
        ):
            derive(
                "prefix-cache-retention-interval", "None", "GPU-local aligned retention"
            )
        if hardware != "native" and "VLLM_PCIE_DMA_MIN_BYTES" not in environment:
            set_env(
                "VLLM_PCIE_DMA_MIN_BYTES",
                "off" if values["decode-context-parallel-size"] > 1 else "6MB",
                "derived:GLM PCIe DCP policy",
            )
    if identifier == "ds41-flash" and "engram-config" not in values:
        derive(
            "engram-config",
            {
                "cpu_offload": False,
                "table_memory": values["engram-table-memory"],
                "disk_resident_scales": values["engram-disk-resident-scales"],
                "projection_tp": values["engram-projection-tp"],
            },
            "Engram table placement, not generic CPU offload",
        )
    width = values.get("speculative-config", {}).get("num_speculative_tokens", 0) + 1
    if identifier.startswith("ds4-") and mode != "dspark":
        width = 4 if mode == "off" else 8
    if values.get("max-cudagraph-capture-size") == "auto":
        derive(
            "max-cudagraph-capture-size",
            max(6, values["max-num-seqs"] * width),
            "bounded verifier rows",
        )
    if "cudagraph-capture-sizes" in values and origins.get(
        "cudagraph-capture-sizes", ""
    ).startswith(
        # PORT(recipe): a recipe-listed capture-size set is data, so it is
        # extended to a raised cap exactly like a model or preset list.
        ("model:", "preset:", "recipe:")
    ):
        cap = values["max-cudagraph-capture-size"]
        # PORT(recipe): a recipe is a data layer in the same sense a preset is;
        # either one may raise the cap away from the profile default.
        if (
            cap != model["defaults"].get("max-cudagraph-capture-size")
            or deployment
            or recipe_name
        ):
            sizes = {n for n in values["cudagraph-capture-sizes"] if n <= cap}
            # A raised cap (for example more request slots) continues the
            # listed sizes in steps of one request's verifier rows, so every
            # running-request count up to the cap still replays a graph.
            step = width if width > 1 else 8
            top = max(sizes, default=0)
            sizes |= {n for n in range(step, cap, step) if n > top}
            derive(
                "cudagraph-capture-sizes",
                sorted(sizes | {cap}),
                "capture-size cap override",
            )

    if values.get("disable-custom-all-reduce"):
        set_env("VLLM_ENABLE_PCIE_ALLREDUCE", "0", "cli:disable-custom-all-reduce")
    if environment.get("NCCL_GRAPH_FILE") == "":
        environment.pop("NCCL_GRAPH_FILE")
        env_origins.pop("NCCL_GRAPH_FILE")
    policy_digest = hashlib.sha256(
        json.dumps(
            [platform, common, model, hw, *([deployment] if deployment else [])],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    jit_root = environment.get(
        "XDG_CACHE_HOME",
        f"/cache/jit/{runtime_identity or 'UNBOUND-RUNTIME'}/{identifier}-{policy_digest}",
    )
    cache_paths = {
        name: f"{jit_root}/{suffix}" if suffix else jit_root
        for name, suffix in JIT_PATHS.items()
    }
    for name, path in cache_paths.items():
        if name not in environment:
            set_env(name, path, "derived:runtime-lock and profile identity")
    warnings = [
        "Model parameters preserve recipe intent; changing installed vLLM/B12X requires independent qualification."
    ]
    if identifier.startswith("ds4-") and "linear-backend" not in values:
        warnings.append(
            "DS4 b12x-a8-dglin policy leaves dense selection to native vLLM; confirm DeepGEMM dispatch in serving logs."
        )
    if passthrough:
        warnings.append(
            "Unmanaged native options are forwarded to vLLM; their values are not validated by the profile schema."
        )
    if replicas > 1 and identifier == "qwen38-flash-next" and ple_copy_per_replica:
        warnings.append(
            "The installed vLLM cannot share the PLE table, so every replica keeps "
            "its own host-RAM copy (about 27 GiB each)."
        )

    if (
        values["cache-mode"] != "vram"
        and model.get("cache", {}).get("external") != "implemented"
    ):
        raise ConfigError(
            f"{identifier}: external cache is unsupported by this image's profile"
        )
    cache_service = configure_cache(
        values, origins, environment, env_origins, identifier, runtime_identity
    )
    if values.get("kv-transfer-config", {}).get(
        "kv_connector"
    ) == "LMCacheRecurrentCheckpointConnector" and not values.get(
        "language-model-only", False
    ):
        warnings.append(
            "External recurrent checkpoints support text requests only. "
            "Image/video requests can use native GPU prefix caching but do not "
            "restore their recurrent state from CPU or disk."
        )
    validate(values, environment, identifier)
    command = make_argv(values, passthrough, root=root)
    return ResolvedPlan(
        profile=identifier,
        hardware=hardware,
        preset=preset,
        recipe=recipe_name,
        values=values,
        origins=origins,
        environment=environment,
        environment_origins=env_origins,
        argv=command,
        passthrough=passthrough,
        warnings=warnings,
        cache_service=asdict(cache_service) if cache_service else None,
        draft_subfolder=draft_subfolder,
    )


def resolve_draft_subfolder(
    plan: ResolvedPlan,
    fetch_snapshot: Callable[..., Path | None],
    *,
    root: Path = DEFAULT_DATA_ROOT,
) -> None:
    """Point the speculative config at the drafter inside the target checkpoint.

    ``fetch_snapshot(repo, *, revision, allow_patterns, local_files_only)``
    replaces ``huggingface_hub.snapshot_download``: it returns the snapshot
    directory, or ``None`` when that snapshot is not available (the launcher's
    offline case). The resolver never reaches the network itself.
    """
    if not plan.draft_subfolder or "speculative-config" not in plan.values:
        return
    target = plan.values["model"]
    if Path(target).is_dir():
        base = Path(target)
    else:
        # Fetches only the drafter's files; vLLM downloads the target itself.
        # A cached snapshot is used as is, which also works offline.
        options = {
            "revision": plan.values.get("revision"),
            "allow_patterns": [f"{plan.draft_subfolder}/*"],
        }
        base = fetch_snapshot(target, local_files_only=True, **options)
        if base is None or not (
            Path(base) / plan.draft_subfolder / "config.json"
        ).is_file():
            base = fetch_snapshot(target, local_files_only=False, **options)
        if base is None:
            raise ConfigError(
                f"Drafter config not found: {Path(target) / plan.draft_subfolder}/config.json"
            )
    draft = Path(base) / plan.draft_subfolder
    if not (draft / "config.json").is_file():
        raise ConfigError(f"Drafter config not found: {draft}/config.json")
    plan.values["speculative-config"]["model"] = str(draft)
    plan.argv = make_argv(plan.values, plan.passthrough, root=root)


def mtp_expert_formats(
    model: str, revision: str | None, fetch_file: Callable[..., Path | None]
) -> set[str]:
    """Quantization formats of the MTP experts named by a ModelOpt checkpoint.

    Empty when the checkpoint has no hf_quant_config.json, it cannot be read,
    or it does not list the MTP experts. ``fetch_file(repo, filename, *,
    revision, local_files_only)`` replaces ``huggingface_hub.hf_hub_download``
    and returns ``None`` when the file is not available; a local directory is
    read without it.
    """
    path: Path | None = Path(model) / "hf_quant_config.json"
    if not Path(model).is_dir():
        # One small file; a cached snapshot is used as is, also offline.
        path = fetch_file(
            model, "hf_quant_config.json", revision=revision, local_files_only=True
        )
        if path is None:
            path = fetch_file(
                model, "hf_quant_config.json", revision=revision, local_files_only=False
            )
        if path is None:
            return set()
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return set()
    # Local Inference Lab exports list quantized_layers at the top level;
    # ModelOpt nests them under "quantization".
    layers = (
        config.get("quantized_layers")
        or (config.get("quantization") or {}).get("quantized_layers")
        or {}
    )
    return {
        str(entry.get("quant_algo"))
        for name, entry in layers.items()
        if name.startswith("mtp.") and name.endswith(".experts")
    }


def resolve_draft_moe_backend(
    plan: ResolvedPlan,
    fetch_file: Callable[..., Path | None],
    *,
    root: Path = DEFAULT_DATA_ROOT,
) -> None:
    """Let vLLM choose the MoE backend of a profile's MTP drafter when b12x cannot run it.

    Profiles run MTP drafter experts on b12x, which takes NVFP4 and MXFP4
    experts, and MXFP8 experts (the Qwen3.8 QAD exports) when the installed
    vLLM and B12X have the MXFP8 path. A checkpoint revision with other MTP
    experts would fail to load. The drafter then gets "auto"; without a
    backend it would inherit the target's --moe-backend b12x. vLLM's choice
    (Marlin for MXFP8) runs them. An explicit speculative-config is kept.
    """
    spec = plan.values.get("speculative-config")
    if (
        not spec
        or spec.get("method") != "mtp"
        or spec.get("moe_backend") != "b12x"
        or not plan.origins["speculative-config"].startswith("derived:")
    ):
        return
    # The drafter's own checkpoint when one is set (--draft-model), else the target.
    drafter = spec.get("model") or plan.values["model"]
    formats = mtp_expert_formats(drafter, spec.get("revision"), fetch_file)
    supported = ("NVFP4", "MXFP4")
    if any("MXFP8" in name for name in formats) and installed_b12x_mxfp8_moe():
        supported += ("MXFP8",)
    unsupported = sorted(
        name for name in formats if not any(kind in name for kind in supported)
    )
    if not unsupported:
        return
    spec["moe_backend"] = "auto"
    plan.argv = make_argv(plan.values, plan.passthrough, root=root)
    print(
        f"MTP drafter experts are {', '.join(unsupported)}, which b12x does not "
        "run; the drafter's MoE backend is auto",
        file=sys.stderr,
    )


def chat_template_path(value: str, *, root: Path = DEFAULT_DATA_ROOT) -> str | None:
    """Resolve a chat-template setting to vLLM's --chat-template value.

    ``checkpoint`` keeps the template shipped with the model (no option), and
    ``runtime:NAME`` names a template installed with these profiles. Any other
    value is a path or template text for vLLM.
    """
    if value == "checkpoint":
        return None
    if not value.startswith(RUNTIME_TEMPLATE_PREFIX):
        return value
    name = value[len(RUNTIME_TEMPLATE_PREFIX) :]
    path = root / name
    if (
        Path(name).is_absolute()
        or path.resolve().parent != (root / "templates").resolve()
        or not path.is_file()
    ):
        raise ConfigError(f"Unknown runtime chat template: {value}")
    return str(path)


def make_argv(
    values: dict, passthrough: list[str], *, root: Path = DEFAULT_DATA_ROOT
) -> list[str]:
    specs = read_yaml(root / "options.yaml")
    command = [
        "/opt/venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        values["model"],
    ]
    for key, value in sorted(values.items()):
        if key == "model" or specs.get(key, {}).get("control"):
            continue
        if key == "chat-template":
            value = chat_template_path(value, root=root)
            if value is None:
                continue
        if isinstance(value, bool):
            command.append("--" + ("" if value else "no-") + key)
        elif isinstance(value, dict):
            command.extend(
                ["--" + key, json.dumps(value, separators=(",", ":"), allow_nan=False)]
            )
        elif isinstance(value, list):
            command.extend(["--" + key, *map(str, value)])
        else:
            command.extend(["--" + key, str(value)])
    command.extend(passthrough)
    return command


def validate(values: dict, environment: dict, identifier: str) -> None:
    try:
        json.dumps(values, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ConfigError("Configuration must contain finite JSON values") from error
    for key in (
        "tensor-parallel-size",
        "decode-context-parallel-size",
        "pipeline-parallel-size",
        "max-num-seqs",
        "max-num-batched-tokens",
        "block-size",
        "max-cudagraph-capture-size",
    ):
        if not isinstance(values[key], int) or values[key] <= 0:
            raise ConfigError(f"{key} must be positive")
    if values["pipeline-parallel-size"] != 1:
        raise ConfigError(
            "These profiles support single-node tensor parallelism, not pipeline parallelism"
        )
    supported_kv = {"fp8", "fp8_e4m3"}
    if identifier == "glm53-flash":
        supported_kv.add("nvfp4_ds_mla")
    if identifier == "mimo26-flash":
        # vllm #882 qualified the exact BF16 cache (half the FP8 capacity).
        supported_kv.add("bfloat16")
    if values["kv-cache-dtype"] not in supported_kv:
        raise ConfigError(
            f"{identifier} supports target KV settings {sorted(supported_kv)}; other precisions require separate qualification"
        )
    if values["tensor-parallel-size"] % values["decode-context-parallel-size"]:
        raise ConfigError("DCP must divide TP")
    if not 0 < values["gpu-memory-utilization"] <= 1:
        raise ConfigError("gpu-memory-utilization must be in (0, 1]")
    if "kv-cache-memory-bytes" in values and values["kv-cache-memory-bytes"] <= 0:
        raise ConfigError("kv-cache-memory-bytes must be positive")
    if not 1 <= values["port"] <= 65535:
        raise ConfigError("port must be in [1, 65535]")
    share = values.get("prefill-compute-share")
    if share is not None:
        if share != "auto":
            try:
                numeric_share = float(share)
            except ValueError as error:
                raise ConfigError(
                    "prefill-compute-share must be auto or a value in (0, 1)"
                ) from error
            if not math.isfinite(numeric_share) or not 0 < numeric_share < 1:
                raise ConfigError(
                    "prefill-compute-share must be auto or a value in (0, 1)"
                )
        if values.get("prefill-schedule-interval", 1) != 1:
            raise ConfigError("Compute sharing requires prefill-schedule-interval=1")
    if "prefill-compute-half-life" in values:
        if share != "auto":
            raise ConfigError(
                "prefill-compute-half-life requires automatic compute share"
            )
        half_life = values["prefill-compute-half-life"]
        if half_life not in {"smooth", "responsive"}:
            try:
                seconds = float(half_life)
            except ValueError as error:
                raise ConfigError(
                    "half-life must be smooth, responsive or positive seconds"
                ) from error
            if not math.isfinite(seconds) or seconds <= 0:
                raise ConfigError("half-life must be positive and finite")
    prefill_step_tokens = values.get("max-num-prefill-tokens-per-step")
    if prefill_step_tokens is not None:
        if not isinstance(prefill_step_tokens, int) or prefill_step_tokens < 0:
            raise ConfigError("max-num-prefill-tokens-per-step must be nonnegative")
        if prefill_step_tokens > values["max-num-batched-tokens"]:
            raise ConfigError(
                "max-num-prefill-tokens-per-step cannot exceed max-num-batched-tokens"
            )
        if prefill_step_tokens > 0 and share is None:
            raise ConfigError(
                "max-num-prefill-tokens-per-step requires prefill-compute-share"
            )
    for key in ("max-parallel-prefills", "decode-refill-target"):
        if (
            key in values
            and values[key] != "auto"
            and (not isinstance(values[key], int) or values[key] < 1)
        ):
            raise ConfigError(f"{key} must be auto or a positive integer")
    if (
        values.get("max-parallel-prefills", 1) != 1
        and not values["enable-chunked-prefill"]
    ):
        raise ConfigError("Parallel prefills require chunked prefill")
    if (
        values.get("max-parallel-prefills", 1) == 1
        and values.get("prefill-policy", "round-robin") != "round-robin"
    ):
        raise ConfigError(
            "decode-aware prefill requires max-parallel-prefills > 1 or auto"
        )
    if (
        values.get("prefill-policy", "round-robin") != "decode-aware"
        and values.get("decode-refill-target", "auto") != "auto"
    ):
        raise ConfigError("decode-refill-target requires decode-aware prefill")
    captures = values.get("cudagraph-capture-sizes", [])
    if captures and (
        captures != sorted(set(captures))
        or min(captures) < 1
        or max(captures) > values["max-cudagraph-capture-size"]
    ):
        raise ConfigError(
            "Capture sizes must be positive, strictly increasing and within the capture cap"
        )
    if identifier == "glm53-flash":
        for key in ("target-page-size", "recurrent-page-size"):
            page = values[key]
            if page != "auto" and (
                not page.isdigit() or int(page) <= 0 or int(page) % 64
            ):
                raise ConfigError(f"{key} must be auto or a positive multiple of 64")
        target, recurrent = values["target-page-size"], values["recurrent-page-size"]
        if (
            target != "auto"
            and recurrent != "auto"
            and int(target) % int(recurrent)
            and int(recurrent) % int(target)
        ):
            raise ConfigError(
                "Recurrent page spacing must divide, or be a multiple of, the target page"
            )
        if values.get("cp-kv-cache-interleave-size") != values.get(
            "dcp-kv-cache-interleave-size"
        ):
            raise ConfigError(
                "GLM target and draft require matching CP and DCP interleave sizes"
            )
    if "speculative-config" in values:
        depth = values["speculative-config"].get("num_speculative_tokens")
        if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
            raise ConfigError("Speculative token count must be positive")
    if identifier == "ds41-flash":
        if "engram-config" in values:
            engram = values["engram-config"]
            if (
                engram.get("table_memory") not in {"ram", "disk"}
                or engram.get("cpu_offload") is not False
            ):
                raise ConfigError(
                    "The DS4.1 profile covers RAM/disk Engram with cpu_offload=false"
                )
        if values["adaptive-verification-cost-scale"] <= 0:
            raise ConfigError("adaptive-verification-cost-scale must be positive")
    if "OMP_NUM_THREADS" in environment:
        omp = environment["OMP_NUM_THREADS"]
        if not omp.isdigit() or int(omp) <= 0:
            raise ConfigError("OMP_NUM_THREADS must be positive")
