"""Start one container's process from a recipe or an upstream model profile.

The launcher is the only thing between an operator's command line and vLLM. It
replaces the shell that used to live in Kubernetes init containers and in
``bash -c`` entry arguments: checkpoint publication, rank selection, rendezvous,
leader-only arguments, cache validation and the final ``exec`` all happen here,
so a deployment manifest contains a command and arguments and nothing else.

Four roles, one process each. Every role resolves the same configuration from
the same arguments, so a cache server and its engine cannot disagree about a
port, a shared-memory pool name or a chunk size:

``server``
    Publish the checkpoint, wait for the leader when this is a worker, validate
    the cache, write the probe record and exec vLLM.
``cache``
    Claim the shared-memory arena and the disk tiers, then exec ``lmcache server``.
``proxy``
    Run the conversation-affinity proxy in front of several single-GPU replicas.
``probe``
    Serve ``/livez``, ``/readyz``, ``/healthz`` and ``/statusz`` for a rank that
    starts no API server of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn

from image_tools.launcher import EXIT_CACHE, EXIT_CONFIG, EXIT_MODEL, ConfigError
from image_tools.launcher import cache_runtime, proxy
from image_tools.launcher import resolver as policy
from image_tools.launcher.probes import ProbeConfig, write_config

RECIPE_ROOT = Path("/opt/vllm-image/recipes")
DATA_ROOT = Path("/opt/vllm-image/runtime")
SOURCES_LOCK = Path("/opt/vllmb12x/sources.lock.json")


def recipe_root() -> Path:
    """Where the recipe YAML files live.

    One directory per owner in the repository (``recipes/<owner>/<name>.yaml``),
    installed flat below ``/opt/vllm-image/recipes`` in the image, which is why
    :func:`load_recipe` searches the tree by name rather than by path. The
    environment override lets the Pages build and the tests point at a checkout.
    """
    return Path(os.environ.get("VLLM_IMAGE_RECIPE_ROOT", RECIPE_ROOT))


def data_root() -> Path:
    """Where the pinned policy data lives: profiles, hardware, options, presets."""
    return Path(os.environ.get("VLLM_IMAGE_DATA_ROOT", DATA_ROOT))


def sources_lock() -> Path:
    """The image's source lock, whose digest keys the JIT cache namespace."""
    return Path(os.environ.get("VLLM_IMAGE_SOURCES_LOCK", SOURCES_LOCK))


# Upstream's resolver names the interpreter as ``/opt/venv/bin/python``, which in
# this image is a shell wrapper. The launcher execs the real interpreter and
# supplies the two variables that wrapper used to export, so no shell sits
# between a container's PID 1 and the engine.
VENV_PYTHON = "/opt/venv/bin/python"
INTERPRETER = "/opt/python/bin/python"
SITE_PACKAGES = "/opt/venv/lib/python3.12/site-packages"
NATIVE_LIBRARY_PATH = "/opt/nccl/lib:/usr/local/cuda/lib64"

# vLLM loads this class into its own API app, which puts the probe endpoints on
# the serving port wherever an API server exists.
PROBE_MIDDLEWARE = "image_tools.launcher.probe_server:ProbeMiddleware"

TOPOLOGY_SINGLE = "single"
TOPOLOGY_LWS = "lws"
ROLES = ("server", "cache", "proxy", "probe")
DEFAULT_RENDEZVOUS_PORT = 25000
DEFAULT_PROBE_PORT = 8890

# Fields of a cache health response that identify the instance and change when it
# restarts. Counters are excluded on purpose: a field that moves during normal
# serving would make every readiness check fail.
CACHE_IDENTITY_KEYS = ("instance_id", "instance_uuid", "pid", "boot_id", "start_time", "started_at")

_RECIPE_KEYS = {
    "profile",
    "hardware",
    "preset",
    "options",
    "environment",
    "topology",
    "probe_port",
}
# The checkpoint facts a sync-mode deployment must state, grouped by how each is
# validated and named by the section that carries it: ``model`` says which
# checkpoint, ``deployment`` says where it lands on the node. A launcher-side
# block restating these same seven values is what this table replaces.
_SYNC_STRINGS = (("model", "model_id"), ("deployment", "storage_root"), ("deployment", "model_path"))
_SYNC_POSITIVE_INTEGERS = (
    ("deployment", "storage_min_free_gib"),
    ("deployment", "download_workers"),
)
_SYNC_STRING_LISTS = (("deployment", "ignore_patterns"), ("deployment", "required_files"))
# The two sections the launcher still owns outright: how a model is spread across
# pods, and how rank zero publishes its KV events. Neither is a checkpoint fact, so
# neither has a home outside ``launch``.
_TOPOLOGY_KEYS = {"kind", "nodes", "rendezvous_port", "kv_events", "replica_port_base"}
_KV_EVENT_KEYS = {"publisher", "endpoint", "replay_endpoint"}


def _field(source: dict[str, Any], section: str, name: str, where: str) -> Any:
    """One required recipe field, refused by name rather than raising a KeyError."""
    if name not in source:
        raise ConfigError(f"{where}: {section}.{name} is required")
    return source[name]


def _require_mapping(data: Any, where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping")
    return data


def _require_keys(data: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{where} must be a mapping")
    missing, extra = allowed - set(data), set(data) - allowed
    if missing or extra:
        raise ConfigError(
            f"{where} keys must be exactly {sorted(allowed)}; "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}"
        )
    return data


def _is_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ModelSync:
    """Where the checkpoint is published before the engine loads it."""

    repo: str
    revision: str
    storage_root: Path
    publish: Path
    min_free_gib: int
    workers: int
    ignore_patterns: tuple[str, ...]
    required_files: tuple[str, ...]

    @classmethod
    def from_dict(cls, model: Any, deployment: Any, where: str) -> ModelSync:
        """Read the checkpoint facts from the two sections that carry them.

        ``where`` names the recipe, so a refusal points at the file and the field
        rather than at a block the file no longer has. The types checked here are
        the ones the launcher-side block checked; only their spelling moved.
        """
        sections = {
            "model": _require_mapping(model, f"{where}: model"),
            "deployment": _require_mapping(deployment, f"{where}: deployment"),
        }
        values: dict[str, Any] = {}
        for section, name in _SYNC_STRINGS:
            value = _field(sections[section], section, name, where)
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{where}: {section}.{name} must be a non-empty string")
            values[name] = value
        revision = _field(sections["model"], "model", "revision", where)
        if not isinstance(revision, str) or not _is_commit(revision):
            raise ConfigError(
                f"{where}: model.revision must be a full lowercase 40-hex Hugging Face commit OID"
            )
        for section, name in _SYNC_POSITIVE_INTEGERS:
            value = _field(sections[section], section, name, where)
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{where}: {section}.{name} must be a positive integer")
            values[name] = value
        for section, name in _SYNC_STRING_LISTS:
            value = _field(sections[section], section, name, where)
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ConfigError(f"{where}: {section}.{name} must be a list of non-empty strings")
            values[name] = tuple(value)
        return cls(
            repo=values["model_id"],
            revision=revision,
            storage_root=Path(values["storage_root"]),
            publish=Path(values["model_path"]),
            min_free_gib=values["storage_min_free_gib"],
            workers=values["download_workers"],
            ignore_patterns=values["ignore_patterns"],
            required_files=values["required_files"],
        )


@dataclass(frozen=True)
class KvEvents:
    """Prefix-cache event publishing, which only the group's rank zero may own."""

    publisher: str
    endpoint: str
    replay_endpoint: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], where: str) -> KvEvents:
        for name in ("publisher", "endpoint", "replay_endpoint"):
            if not isinstance(data[name], str) or not data[name]:
                raise ConfigError(f"{where}.{name} must be a non-empty string")
        return cls(data["publisher"], data["endpoint"], data["replay_endpoint"])


@dataclass(frozen=True)
class Topology:
    """How this deployment spreads one model across pods."""

    kind: str
    nodes: int
    rendezvous_port: int
    kv_events: KvEvents | None
    replica_port_base: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any], where: str) -> Topology:
        if data["kind"] not in {TOPOLOGY_SINGLE, TOPOLOGY_LWS}:
            raise ConfigError(f"{where}.kind must be {TOPOLOGY_SINGLE!r} or {TOPOLOGY_LWS!r}")
        nodes = data["nodes"]
        if type(nodes) is not int or nodes < 1:
            raise ConfigError(f"{where}.nodes must be a positive integer")
        if data["kind"] == TOPOLOGY_LWS and nodes < 2:
            raise ConfigError(f"{where}: an {TOPOLOGY_LWS} topology needs at least two nodes")
        for name in ("rendezvous_port", "replica_port_base"):
            value = data[name]
            if value is not None and (type(value) is not int or not 1 <= value <= 65535):
                raise ConfigError(f"{where}.{name} must be a port or null")
        events = data["kv_events"]
        if events is not None:
            events = KvEvents.from_dict(
                _require_keys(events, _KV_EVENT_KEYS, f"{where}.kv_events"), f"{where}.kv_events"
            )
        return cls(
            kind=data["kind"],
            nodes=nodes,
            rendezvous_port=(
                DEFAULT_RENDEZVOUS_PORT if data["rendezvous_port"] is None else data["rendezvous_port"]
            ),
            kv_events=events,
            replica_port_base=data["replica_port_base"],
        )


@dataclass(frozen=True)
class Recipe:
    """One upstream profile plus this image's accepted overrides."""

    name: str
    profile: str
    hardware: str
    preset: str | None
    options: dict[str, Any]
    environment: dict[str, str]
    model_sync: ModelSync | None
    topology: Topology
    probe_port: int

    @classmethod
    def from_dict(cls, name: str, document: Any) -> Recipe:
        """Build a recipe from one whole recipe document.

        ``launch`` is the launcher's own section and its keys are exact. The
        checkpoint facts come out of ``model`` and ``deployment``, the sections
        that describe where the weights live, so a deployment is stated once.
        """
        where = f"recipe {name}"
        document = _require_mapping(document, f"{where}")
        launch = _require_keys(document.get("launch"), _RECIPE_KEYS, f"{where}: launch")
        for key in ("profile", "hardware"):
            if not isinstance(launch[key], str) or not launch[key]:
                raise ConfigError(f"{where}: launch.{key} must be a non-empty string")
        preset = launch["preset"]
        if preset is not None and not isinstance(preset, str):
            raise ConfigError(f"{where}: launch.preset must be a string or null")
        for key in ("options", "environment"):
            if not isinstance(launch[key], dict):
                raise ConfigError(f"{where}: launch.{key} must be a mapping")
        if not all(isinstance(value, str) for value in launch["environment"].values()):
            raise ConfigError(f"{where}: launch.environment values must be strings")
        probe_port = launch["probe_port"]
        if type(probe_port) is not int or not 1 <= probe_port <= 65535:
            raise ConfigError(f"{where}: launch.probe_port must be a port")
        model, deployment = document.get("model"), document.get("deployment")
        # A published root is what makes this a sync-mode deployment: with no path
        # to publish to, the launcher fetches nothing and the engine resolves the
        # repository id itself, which is the engine-download mode qwen38-27b runs.
        sync = (
            None
            if not isinstance(deployment, dict) or deployment.get("model_path") is None
            else ModelSync.from_dict(model, deployment, where)
        )
        return cls(
            name=name,
            profile=launch["profile"],
            hardware=launch["hardware"],
            preset=preset,
            options=dict(launch["options"]),
            environment=dict(launch["environment"]),
            model_sync=sync,
            topology=Topology.from_dict(
                _require_keys(launch["topology"], _TOPOLOGY_KEYS, f"{where}: launch.topology"),
                f"{where}: launch.topology",
            ),
            probe_port=probe_port,
        )


def recipe_path(name: str, root: Path | None = None) -> Path:
    """The one file that declares ``name``, searched across the owner directories.

    Recipes live at ``recipes/<owner>/<name>.yaml`` in the repository and are
    installed flat below ``/opt/vllm-image/recipes`` in the image, so the name is
    the identity a deployment selects and the path is an implementation detail.
    Two files with one name is an error rather than a guess: ``--recipe`` has to
    resolve to exactly one definition, and it names both candidates.
    """
    base = Path(root or recipe_root())
    found = sorted(path for path in base.rglob(f"{name}.yaml") if path.is_file())
    if not found:
        raise ConfigError(f"no recipe {name}.yaml under {base}", EXIT_CONFIG)
    if len(found) > 1:
        raise ConfigError(
            f"recipe {name} is declared twice: " + ", ".join(str(path) for path in found),
            EXIT_CONFIG,
        )
    return found[0]


def load_recipe(name: str, root: Path | None = None) -> Recipe:
    """Read the recipe named ``name``. A missing file or a missing section is an error.

    The recipe's identity is the file name, so a deployment that asks for
    ``--recipe x`` always gets ``x.yaml``, and the value recorded as the source of
    its overrides names the same thing.
    """
    path = recipe_path(name, root)
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except OSError as error:
        raise ConfigError(f"cannot read recipe {path}: {error}", EXIT_CONFIG) from error
    except ImportError as error:  # PyYAML is a locked runtime dependency
        raise ConfigError(f"cannot read recipes without PyYAML: {error}", EXIT_CONFIG) from error
    except Exception as error:  # yaml.YAMLError and anything raised while reading
        raise ConfigError(f"cannot parse recipe {path}: {error}", EXIT_CONFIG) from error
    if not isinstance(data, dict):
        raise ConfigError(f"recipe {path} must be a mapping", EXIT_CONFIG)
    for section in ("launch", "model", "deployment"):
        if section not in data:
            raise ConfigError(f"recipe {path} has no {section} section", EXIT_CONFIG)
    return Recipe.from_dict(name, data)


def runtime_identity() -> str | None:
    """The image's source-lock digest, which keys the JIT cache namespace."""
    try:
        return hashlib.sha256(sources_lock().read_bytes()).hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class Rank:
    """Where this container sits inside its tensor-parallel group."""

    index: int
    nodes: int
    leader: str | None

    @property
    def is_leader(self) -> bool:
        return self.index == 0

    @property
    def serves_api(self) -> bool:
        return self.index == 0


def read_rank(topology: Topology, env: dict[str, str]) -> Rank:
    if topology.kind == TOPOLOGY_SINGLE:
        return Rank(0, 1, None)
    for name in ("LWS_WORKER_INDEX", "LWS_GROUP_SIZE", "LWS_LEADER_ADDRESS"):
        if name not in env:
            raise ConfigError(
                f"a {TOPOLOGY_LWS} topology requires {name}, which the LeaderWorkerSet "
                "controller injects into every container",
                EXIT_CONFIG,
            )
    try:
        index, group = int(env["LWS_WORKER_INDEX"]), int(env["LWS_GROUP_SIZE"])
    except ValueError as error:
        raise ConfigError(f"LWS rank or group size is not an integer: {error}", EXIT_CONFIG) from error
    if index < 0 or group < 1:
        raise ConfigError("LWS_WORKER_INDEX and LWS_GROUP_SIZE must be non-negative", EXIT_CONFIG)
    if topology.nodes > 1 and group != topology.nodes:
        raise ConfigError(
            f"the group is {group} pods but this recipe is validated for {topology.nodes}; "
            "change the recipe, not the cluster",
            EXIT_CONFIG,
        )
    return Rank(index, group, env["LWS_LEADER_ADDRESS"] or None)


def kv_events_args(
    events: KvEvents | None, rank: Rank, values: dict[str, Any], env: dict[str, str]
) -> list[str]:
    """Build the rank-zero KV-event publishing arguments.

    The pod half of the topic must be exactly the ``<podIP>:<port>`` the router
    looks up, so it is assembled here from the injected pod address and the
    resolved serving port rather than pasted into a manifest by hand.
    """
    if events is None or not rank.is_leader:
        return []
    pod_ip = env.get("POD_IP", "")
    if not pod_ip:
        raise ConfigError(
            "this recipe publishes KV events and needs POD_IP, which the manifest must "
            "expose from status.podIP",
            EXIT_CONFIG,
        )
    for name in ("port", "served-model-name"):
        if name not in values:
            raise ConfigError(f"the KV-event topic needs the resolved --{name}", EXIT_CONFIG)
    config = json.dumps(
        {
            "enable_kv_cache_events": True,
            "publisher": events.publisher,
            "endpoint": events.endpoint,
            "replay_endpoint": events.replay_endpoint,
            "topic": f"kv@{pod_ip}:{values['port']}@{values['served-model-name']}",
        },
        separators=(",", ":"),
    )
    return ["--enable-scale-out", "--kv-events-config", config]


def topology_args(
    rank: Rank, values: dict[str, Any], events: KvEvents | None, env: dict[str, str]
) -> list[str]:
    """The arguments that differ between the ranks of one tensor-parallel group."""
    args: list[str] = []
    if rank.nodes > 1:
        args += [
            "--nnodes",
            str(rank.nodes),
            "--node-rank",
            str(rank.index),
            "--master-addr",
            "127.0.0.1" if rank.is_leader else str(rank.leader),
        ]
        if not rank.is_leader:
            args += ["--headless"]
    args += kv_events_args(events, rank, values, env)
    return args


def child_environment(
    plan: policy.ResolvedPlan, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """The environment a role's process starts with.

    Managed aliases are removed before the resolved values are written, so a
    stale alias left in a pod spec cannot be resolved a second time by a child.
    """
    environment = dict(os.environ)
    for name in policy.managed_environment_names(root=data_root()):
        environment.pop(name, None)
    if environment.get("NCCL_GRAPH_FILE") == "":
        environment.pop("NCCL_GRAPH_FILE")
    environment.update(plan.environment)
    if extra:
        environment.update(extra)
    environment["PYTHONPATH"] = _prepend_path(environment.get("PYTHONPATH", ""), SITE_PACKAGES)
    environment["LD_LIBRARY_PATH"] = _prepend_path(
        environment.get("LD_LIBRARY_PATH", ""), NATIVE_LIBRARY_PATH
    )
    return environment


def _prepend_path(existing: str, addition: str) -> str:
    parts = [part for part in existing.split(os.pathsep) if part]
    if addition not in parts:
        parts.insert(0, addition)
    return os.pathsep.join(parts)


def cache_record(service: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """The subset of the validated cache a probe needs to detect a restart."""
    return {
        "health_url": service["health_url"],
        "shm_name": service["shm_name"],
        "shm_bytes": service["shm_bytes"],
        "shm_root": str(cache_runtime.SHM_ROOT),
        "tier_dirs": list(service["directories"]),
        "identity": {key: status[key] for key in CACHE_IDENTITY_KEYS if key in status},
    }


def validate_cache(
    service: dict[str, Any], *, read: Callable[[str], dict[str, Any]] = cache_runtime.read_json
) -> dict[str, Any]:
    """Require the cache to be up, the arena to exist, and record its identity.

    ``/healthcheck`` only says the server's engine object exists, so the arena is
    measured where it lives and the instance is read from ``/status``. Doing this
    before ``exec`` means a deployment fails here, with a reason, instead of
    serving with a silently degraded cache.
    """
    cache_runtime.wait_ready(service["health_url"], float(service["startup_timeout"]))
    status = read(cache_runtime.status_url(service["health_url"]))
    cache_runtime.validate_arena(service["shm_name"], int(service["shm_bytes"]))
    return cache_record(service, status)


def exec_command(argv: list[str], environment: dict[str, str]) -> NoReturn:
    """Replace this process with ``argv``, resolving the interpreter first."""
    command = list(argv)
    if not command:
        raise ConfigError("nothing to exec", EXIT_CONFIG)
    if command[0] == VENV_PYTHON:
        command[0] = INTERPRETER
    os.execve(command[0], command, environment)
    raise ConfigError("exec returned, which should be impossible", EXIT_CONFIG)


@dataclass
class LaunchRequest:
    role: str
    recipe: str | None
    profile: str | None
    preset: str | None
    hardware: str | None
    topology: str | None
    model_sync: bool
    print_config: bool
    native: list[str] = field(default_factory=list)


LAUNCHER_OPTIONS = frozenset(
    {
        "--recipe",
        "--profile",
        "--preset",
        "--hardware",
        "--role",
        "--topology",
        "--model-sync",
        "--print-config",
    }
)


def parse(argv: list[str]) -> LaunchRequest:
    parser = argparse.ArgumentParser(prog="vllm-image launch", allow_abbrev=False)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--recipe", help="a recipe name under /opt/vllm-image/recipes")
    identity.add_argument("--profile", help="an upstream model profile id")
    parser.add_argument("--preset", help="an upstream deployment preset, with --profile")
    parser.add_argument("--hardware", help="a hardware profile id, with --profile")
    parser.add_argument("--role", choices=ROLES, default="server")
    parser.add_argument("--topology", choices=(TOPOLOGY_SINGLE, TOPOLOGY_LWS))
    parser.add_argument(
        "--model-sync", action="store_true", help="publish the checkpoint before serving"
    )
    parser.add_argument("--print-config", action="store_true", help="print the resolved plan and stop")
    parser.add_argument("native", nargs="*", help="native vLLM arguments, after --")
    arguments = parser.parse_args(argv)
    if arguments.recipe and (arguments.hardware or arguments.preset):
        raise ConfigError(
            "--hardware and --preset belong to a --profile selection; a recipe names its own",
            EXIT_CONFIG,
        )
    if arguments.profile and not arguments.hardware:
        raise ConfigError("--hardware is required with --profile", EXIT_CONFIG)
    if arguments.topology == TOPOLOGY_LWS and not arguments.recipe and "LWS_GROUP_SIZE" not in os.environ:
        raise ConfigError(
            f"--topology {TOPOLOGY_LWS} needs LWS_GROUP_SIZE, which the LeaderWorkerSet controller injects",
            EXIT_CONFIG,
        )
    native = list(arguments.native)
    if native and native[0] == "--":
        native = native[1:]
    return LaunchRequest(
        role=arguments.role,
        recipe=arguments.recipe,
        profile=arguments.profile,
        preset=arguments.preset,
        hardware=arguments.hardware,
        topology=arguments.topology,
        model_sync=arguments.model_sync,
        print_config=arguments.print_config,
        native=native,
    )


def selection(
    request: LaunchRequest, env: dict[str, str]
) -> tuple[Recipe | None, str, str | None, str, dict[str, Any] | None, Topology]:
    """Resolve what to serve: profile, preset, hardware, recipe layer and topology."""
    if request.recipe:
        recipe = load_recipe(request.recipe)
        if request.topology and request.topology != recipe.topology.kind:
            raise ConfigError(
                f"recipe {recipe.name} is a {recipe.topology.kind} deployment; "
                f"--topology {request.topology} contradicts it",
                EXIT_CONFIG,
            )
        layer = {
            "options": dict(recipe.options),
            "environment": dict(recipe.environment),
            "source": f"recipe:{recipe.name}",
        }
        return recipe, recipe.profile, recipe.preset, recipe.hardware, layer, recipe.topology
    nodes = int(env["LWS_GROUP_SIZE"]) if request.topology == TOPOLOGY_LWS else 1
    topology = Topology(
        kind=request.topology or TOPOLOGY_SINGLE,
        nodes=nodes,
        rendezvous_port=DEFAULT_RENDEZVOUS_PORT,
        kv_events=None,
        replica_port_base=None,
    )
    return None, str(request.profile), request.preset, str(request.hardware), None, topology


def resolved_plan(
    profile: str,
    hardware: str,
    preset: str | None,
    layer: dict[str, Any] | None,
    native: list[str],
    env: dict[str, str],
) -> policy.ResolvedPlan:
    return policy.resolve(
        profile,
        hardware,
        preset=preset,
        recipe_layer=layer,
        env=env,
        argv=native,
        cli_env=env,
        runtime_identity=runtime_identity(),
        vllm_environment=policy.installed_vllm_environment(),
        root=data_root(),
    )


def _fetch_snapshot(
    repo: str, *, revision: str | None = None, allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> Path | None:
    """Resolve a checkpoint subtree in the local cache, or None if unavailable."""
    from huggingface_hub import LocalEntryNotFoundError, snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=repo,
                revision=revision,
                allow_patterns=allow_patterns,
                local_files_only=local_files_only,
            )
        )
    except (LocalEntryNotFoundError, OSError):
        return None


def _fetch_file(
    repo: str, filename: str, *, revision: str | None = None, local_files_only: bool = False
) -> Path | None:
    """Resolve one checkpoint file in the local cache, or None if unavailable."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    except OSError:
        return None


def prepare_plan(plan: policy.ResolvedPlan) -> None:
    """Finish the two resolutions upstream defers to start-up.

    A drafter that ships inside the target checkpoint is named by subfolder, and
    an MTP drafter whose experts b12x cannot run falls back to vLLM's own choice.
    Both read the checkpoint, so they run at launch and never at ``--print-config``
    time: printing a configuration must not download anything.
    """
    policy.resolve_draft_subfolder(plan, _fetch_snapshot)
    policy.resolve_draft_moe_backend(plan, _fetch_file)
    if plan.cache_service:
        if not policy.qsa_atomic_transfer_supported(
            policy.installed_source("vllm", policy.QSA_ATOMIC_TRANSFER_SOURCE)
        ) and plan.values.get("kv-transfer-config", {}).get("kv_connector"):
            raise ConfigError(
                "Qwen external cache with DCP>1 requires a vLLM build with atomic QSA "
                "checkpoint transfer; use DCP1 or an image that includes it",
                EXIT_CACHE,
            )


def resolution_context() -> dict[str, Any]:
    """What the resolver was told about the machine it ran on.

    ``--print-config`` output is published by the site and re-run in the browser, so
    the two environment-dependent inputs the resolver takes have to travel with the
    record. Without them a browser, which cannot import vLLM, resolves a preset
    differently from the container that serves it -- and silently, because both
    answers are well-formed.

    ``source`` names which interpreter answered. ``image`` means the published
    container resolved for itself; ``host`` means a checkout's Python did, which is
    what a pull-request preview uses, and the site says so instead of implying the
    image did.
    """
    vllm_environment = policy.installed_vllm_environment()
    return {
        "source": "image" if SOURCES_LOCK.is_file() else "host",
        "vllm_environment": sorted(vllm_environment) if vllm_environment is not None else None,
        "b12x_mxfp8_moe": policy.installed_b12x_mxfp8_moe(),
        "runtime_identity": runtime_identity(),
    }


def publish_model(recipe: Recipe) -> None:
    """Fetch and verify the checkpoint this container must load, in place."""
    from image_tools.vllm_image import sync_model

    if recipe.model_sync is None:
        raise ConfigError(
            f"recipe {recipe.name} requests --model-sync but declares no deployment.model_path",
            EXIT_MODEL,
        )
    sync = recipe.model_sync
    sync_model(
        repo=sync.repo,
        revision=sync.revision,
        storage_root=sync.storage_root,
        publish=sync.publish,
        min_free_bytes=sync.min_free_gib * 1024**3,
        workers=sync.workers,
        ignore=list(sync.ignore_patterns),
        require=list(sync.required_files),
    )
    index = sync.publish / "model.safetensors.index.json"
    if not index.is_file():
        raise ConfigError(f"published model root {sync.publish} has no {index.name} to load", EXIT_MODEL)


def run(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    """Run one role. Only ``--print-config``, ``--role probe`` and ``--role proxy`` return."""
    request = parse(sys.argv[1:] if argv is None else argv)
    environment = dict(os.environ if env is None else env)
    recipe, profile, preset, hardware, layer, topology = selection(request, environment)
    plan = resolved_plan(profile, hardware, preset, layer, request.native, environment)
    rank = read_rank(topology, environment)
    public = policy.public(plan)
    public["role"] = request.role
    public["selection"] = {
        "recipe": request.recipe,
        "profile": profile,
        "preset": preset,
        "hardware": hardware,
    }
    public["topology"] = {
        "kind": topology.kind,
        "nodes": topology.nodes,
        "rank": rank.index,
        "leader": rank.leader,
        "rendezvous_port": topology.rendezvous_port,
        "probe_port": recipe.probe_port if recipe else DEFAULT_PROBE_PORT,
    }
    public["resolution_context"] = resolution_context()
    if request.print_config:
        print(json.dumps(public, indent=2, allow_nan=False, sort_keys=True))
        return 0

    if request.role == "probe":
        from image_tools.launcher import probe_server

        return probe_server.serve(recipe.probe_port if recipe else DEFAULT_PROBE_PORT)

    if request.role == "cache":
        service = plan.cache_service
        if not service:
            raise ConfigError(
                "--role cache needs an external cache; select one with --cache-mode", EXIT_CACHE
            )
        cache_runtime.verify_installed_transfer(service)
        cache_runtime.preflight(service, dict(plan.environment))
        if service["shm_name"]:
            cache_runtime.claim_arena(cache_runtime.SHM_ROOT / service["shm_name"])
        for directory in service["directories"]:
            cache_runtime.claim_tier(Path(directory))
        namespace = Path(service["namespace"]) if service["namespace"] else Path("/cache")
        cache_runtime.prune_stale_tiers(namespace.parent, namespace, prune=service["prune_stale_tiers"])
        cache_runtime.fit_disk_tiers(service)
        exec_command(service["argv"], child_environment(plan, service["environment"]))

    if request.role == "proxy":
        replicas = int(plan.values["replicas"])
        if replicas < 2:
            raise ConfigError("--role proxy needs replicas above one", EXIT_CONFIG)
        base = topology.replica_port_base or (int(plan.values["port"]) + 1)
        upstreams = [f"http://127.0.0.1:{base + offset}" for offset in range(replicas)]
        return proxy.run(upstreams, host="0.0.0.0", port=int(plan.values["port"]))

    # Only the engine needs this: it reads the checkpoint to decide the drafter's
    # location and MoE backend, and the other roles never start an engine.
    prepare_plan(plan)
    if request.model_sync:
        if recipe is None:
            raise ConfigError(
                "--model-sync needs --recipe, which declares the checkpoint to publish", EXIT_MODEL
            )
        publish_model(recipe)

    if not rank.is_leader:
        from image_tools.vllm_image import wait_for_leader

        wait_for_leader(
            leader=str(rank.leader),
            port=topology.rendezvous_port,
            dns_timeout=1200.0,
            connect_timeout=300.0,
        )

    argv = list(plan.argv)
    cache = validate_cache(plan.cache_service) if plan.cache_service else None
    if cache:
        write_config(
            ProbeConfig(
                rank=rank.index,
                leader=rank.leader,
                port=int(plan.values["port"]),
                served_model_name=str(plan.values["served-model-name"]),
                cache=cache,
                status=public,
            )
        )
    argv += topology_args(rank, plan.values, topology.kv_events, environment)
    if rank.serves_api:
        argv += ["--middleware", PROBE_MIDDLEWARE]
    exec_command(argv, child_environment(plan))


def entrypoint(argv: list[str] | None = None) -> int:
    """The image entry point: launcher options start a server, anything else is vLLM."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            "usage: docker run IMAGE --recipe NAME | --profile NAME --hardware NAME | vllm arguments",
            file=sys.stderr,
        )
        return EXIT_CONFIG
    if arguments[0] == "launch":
        return run(arguments[1:])
    if arguments[0] in LAUNCHER_OPTIONS:
        return run(arguments)
    environment = dict(os.environ)
    environment.setdefault("PYTHONPATH", SITE_PACKAGES)
    environment.setdefault("LD_LIBRARY_PATH", NATIVE_LIBRARY_PATH)
    os.execve(
        INTERPRETER, [INTERPRETER, "-m", "vllm.entrypoints.cli.main", *arguments], environment
    )
