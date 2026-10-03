"""Liveness, readiness and status checks for a served model.

One evaluation core, two adapters. The core is synchronous and uses only the
standard library so that it runs identically in the probe container's HTTP
server and inside vLLM's API process as ASGI middleware.

The split between ``/livez`` and ``/readyz`` is deliberate and load bearing. A
long model operation delays vLLM's API event loop without killing the
distributed engine, so anything that asks the engine a question belongs in
``/readyz``, which removes a pod from service, and never in ``/livez``, which
restarts a container. That is why no check here calls the engine, reads a GPU,
or waits on an inference step.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from image_tools.launcher import EXIT_CACHE, EXIT_ENGINE, EXIT_HTTP, ConfigError
from image_tools.launcher import cache_runtime

# Process names the vLLM engine gives its worker and scheduler children. A live
# server process with no such child is wedged: the parent survives while the
# engine does not, which is the failure an existence-only probe cannot see.
ENGINE_COMMS = frozenset({"VLLM::EngineCor", "VLLM::Worker_TP"})
SERVER_ARGV_MARKER = "vllm.entrypoints.cli.main"

DEFAULT_PROC_ROOT = Path("/proc")
DEFAULT_RECORD_PATH = Path("/run/vllm-image/launch.json")
DEFAULT_TIMEOUT_SECONDS = 5.0

LIVEZ = "livez"
READYZ = "readyz"
HEALTHZ = "healthz"
STATUSZ = "statusz"

ENDPOINT_NAMES = (LIVEZ, READYZ, HEALTHZ, STATUSZ)


class ProbeError(Exception):
    """A probe target answered, but not with what the check needed."""


StatusFetcher = Callable[[str, float], int]
JsonFetcher = Callable[[str, float], dict[str, Any]]


@dataclass(frozen=True)
class ProbeConfig:
    """What a container must know to answer questions about one model server.

    The launcher writes this record before it execs, so the probe container and
    the middleware read the same file. It carries the resolved configuration
    rather than re-deriving it: two sources of truth would let a probe answer
    about a server it is not watching.
    """

    rank: int
    leader: str | None
    port: int
    served_model_name: str
    cache: dict[str, Any] | None
    status: dict[str, Any]

    @property
    def distributed(self) -> bool:
        return self.rank > 0

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def to_record(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "leader": self.leader,
            "port": self.port,
            "served_model_name": self.served_model_name,
            "cache": self.cache,
            "status": self.status,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProbeConfig:
        return cls(
            rank=int(record["rank"]),
            leader=record.get("leader"),
            port=int(record["port"]),
            served_model_name=str(record["served_model_name"]),
            cache=record.get("cache"),
            status=record.get("status") or {},
        )


def load_config(path: Path = DEFAULT_RECORD_PATH) -> ProbeConfig:
    """Read the launcher's record. A missing or broken record is not ready."""
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"launcher record {path} is unavailable: {error}") from error
    if not isinstance(record, dict):
        raise RuntimeError(f"launcher record {path} is not a mapping")
    try:
        return ProbeConfig.from_record(record)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"launcher record {path} is malformed: {error}") from error


def write_config(config: ProbeConfig, path: Path = DEFAULT_RECORD_PATH) -> None:
    """Atomically publish the record a probe answers from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(config.to_record(), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _open(url: str, timeout: float) -> Any:
    # Loopback probes must never be redirected through a cluster HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def fetch_status(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> int:
    """GET ``url`` and return its status code."""
    with _open(url, timeout) as response:
        response.read(1)
        return int(response.status)


def fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """GET ``url`` and return its JSON body."""
    try:
        with _open(url, timeout) as response:
            status = int(response.status)
            body = response.read()
    except urllib.error.HTTPError as error:
        status, body = int(error.code), error.read()
    if status != 200:
        raise ProbeError(f"{url} returned {status}")
    try:
        payload = json.loads(body)
    except ValueError as error:
        raise ProbeError(f"{url} returned a non-JSON body") from error
    if not isinstance(payload, dict):
        raise ProbeError(f"{url} returned a {type(payload).__name__}, expected a mapping")
    return payload


def _children(pid: int, proc_root: Path) -> list[int]:
    try:
        listing = (proc_root / str(pid) / "task" / str(pid) / "children").read_text()
    except OSError:
        return []
    children = []
    for value in listing.split():
        try:
            children.append(int(value))
        except ValueError:
            continue
    return children


def _state(pid: int, proc_root: Path) -> str | None:
    """The process state letter, or None when the process is gone."""
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
    except OSError:
        return None
    try:
        return stat.split(") ", 1)[1][0]
    except IndexError:
        return None


def _comm(pid: int, proc_root: Path) -> str:
    try:
        return (proc_root / str(pid) / "comm").read_text().rstrip("\n")
    except OSError:
        return ""


def _cmdline(pid: int, proc_root: Path) -> str:
    try:
        return (proc_root / str(pid) / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""


def find_server_pid(proc_root: Path = DEFAULT_PROC_ROOT) -> int | None:
    """The PID whose command line starts the vLLM command in this container."""
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        argv = _cmdline(pid, proc_root).split("\0")
        if SERVER_ARGV_MARKER in argv and "serve" in argv:
            return pid
    return None


def check_engine_processes(
    config: ProbeConfig,
    *,
    proc_root: Path = DEFAULT_PROC_ROOT,
) -> str | None:
    """Require one live, non-zombie engine child of the server process.

    Engine children re-parented to the pod's PID 1 mean the server died while
    the engine kept running, or the reverse: either way the group cannot make
    progress and the container still looks alive.
    """
    server_pid = find_server_pid(proc_root)
    if server_pid is None:
        return "no vLLM server process is running in this container"
    live: list[str] = []
    wedged: list[str] = []
    for pid in _children(server_pid, proc_root):
        comm = _comm(pid, proc_root)
        if comm not in ENGINE_COMMS:
            continue
        state = _state(pid, proc_root)
        if state in {None, "X", "Z"}:
            wedged.append(f"{comm} (state {state})")
        else:
            live.append(comm)
    if wedged:
        return "engine child is not runnable: " + ", ".join(sorted(wedged))
    if not live:
        return (
            f"vLLM server PID {server_pid} has no direct child with comm in "
            f"{sorted(ENGINE_COMMS)}; the engine is orphaned or gone"
        )
    return None


def check_cache_identity(
    config: ProbeConfig,
    *,
    get_json: JsonFetcher = fetch_json,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """Require the cache validated at launch to still be the live instance.

    Three signals, because each catches what the others cannot. The arena and
    disk-tier ``flock``s, read through ``/proc/locks`` so this check never takes
    the lock it tests, prove a live owner: the kernel releases them when the
    holder exits, including on an OOM kill. The ``/status`` payload then proves
    it is the same instance, and measuring the arena segment proves the transfer
    did not silently fall back to pickle. ``/healthcheck`` is not usable for
    either: at our pin it returns only ``{"status": "healthy"}``.
    """
    cache = config.cache
    if not cache:
        return None
    shm_root = Path(cache.get("shm_root") or cache_runtime.SHM_ROOT)
    reason = cache_runtime.live_owner(
        str(cache.get("shm_name") or ""),
        [Path(directory) for directory in cache.get("tier_dirs") or ()],
        shm_root=shm_root,
    )
    if reason:
        return reason
    health_url = cache.get("health_url")
    if not health_url:
        return "a cache is configured but records no health URL"
    try:
        status_url = cache_runtime.status_url(str(health_url))
    except ConfigError as error:
        return f"cache health URL is unusable: {error}"
    try:
        status = get_json(status_url, timeout)
    except (OSError, urllib.error.URLError, ProbeError) as error:
        return f"cache status endpoint {status_url} is unusable: {error}"
    for key, want in sorted((cache.get("identity") or {}).items()):
        have = status.get(key)
        if have != want:
            return f"cache {key} changed since launch: expected {want!r}, got {have!r}"
    try:
        cache_runtime.validate_arena(
            str(cache.get("shm_name") or ""), int(cache.get("shm_bytes") or 0), shm_root=shm_root
        )
    except ConfigError as error:
        return str(error)
    return None


def check_engine_health(
    config: ProbeConfig,
    *,
    get_status: StatusFetcher = fetch_status,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """Ask the API server whether its engine is alive."""
    url = f"{config.local_url}/health"
    try:
        status = get_status(url, timeout)
    except (OSError, urllib.error.URLError, ProbeError) as error:
        return f"API server is unreachable: {error}"
    if status != 200:
        return f"engine health check returned {status}"
    return None


def check_model_served(
    config: ProbeConfig,
    *,
    get_json: JsonFetcher = fetch_json,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """Require the served model name to be listed by the API server."""
    url = f"{config.local_url}/v1/models"
    try:
        payload = get_json(url, timeout)
    except (OSError, urllib.error.URLError, ProbeError) as error:
        return f"model listing is unusable: {error}"
    listed = sorted(str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict))
    if config.served_model_name not in listed:
        return f"{config.served_model_name!r} is not served; the listing shows {listed}"
    return None


def check_leader_ready(
    config: ProbeConfig,
    *,
    get_status: StatusFetcher = fetch_status,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    """A headless rank is ready only while its leader's rank zero is ready."""
    if not config.leader:
        return "this rank is a worker but no leader address was recorded"
    target = config.leader if "://" in config.leader else f"http://{config.leader}"
    url = target.rstrip("/")
    if not url.endswith(f"/{READYZ}"):
        url = f"{url}/{READYZ}"
    try:
        status = get_status(url, timeout)
    except (OSError, urllib.error.URLError, ProbeError) as error:
        return f"leader readiness endpoint {url} is unusable: {error}"
    if status != 200:
        return f"leader readiness endpoint {url} returned {status}"
    return None


# Check name -> function, per endpoint. Names are stable: they appear in
# ?verbose output and are accepted by ?exclude= and the /<endpoint>/<check> form.
CHECKS: dict[str, dict[str, Callable[..., str | None]]] = {
    LIVEZ: {"engine": check_engine_processes, "cache": check_cache_identity},
    READYZ: {
        "engine": check_engine_processes,
        "cache": check_cache_identity,
        "server": check_engine_health,
        "model": check_model_served,
        "leader": check_leader_ready,
    },
    # /healthz is the union of the other two, which is the same set /readyz runs.
    HEALTHZ: {
        "engine": check_engine_processes,
        "cache": check_cache_identity,
        "server": check_engine_health,
        "model": check_model_served,
        "leader": check_leader_ready,
    },
    STATUSZ: {},
}

# A headless rank starts no API server; rank zero and single-node servers have
# no leader to follow. Skipping by rank keeps one pod template usable for both.
_RANK_SKIPS = {
    True: frozenset({"server", "model"}),
    False: frozenset({"leader"}),
}


def evaluate(
    endpoint: str,
    config: ProbeConfig,
    *,
    get_status: StatusFetcher = fetch_status,
    get_json: JsonFetcher = fetch_json,
    proc_root: Path = DEFAULT_PROC_ROOT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    only: str | None = None,
    exclude: frozenset[str] = frozenset(),
) -> tuple[bool, dict[str, str | None]]:
    """Run an endpoint's named checks and report each result.

    ``only`` restricts the run to one check. ``exclude`` skips named checks.
    """
    if endpoint not in CHECKS:
        raise ValueError(f"unknown endpoint {endpoint}")
    checks = CHECKS[endpoint]
    skipped = _RANK_SKIPS[config.distributed] | exclude
    results: dict[str, str | None] = {}
    for name, function in checks.items():
        if only is not None and name != only:
            continue
        if name in skipped:
            continue
        if function is check_engine_processes:
            results[name] = function(config, proc_root=proc_root)
        elif "check_cache" in function.__name__ or "model" in function.__name__:
            results[name] = function(config, get_json=get_json, timeout=timeout)
        else:
            results[name] = function(config, get_status=get_status, timeout=timeout)
    return all(reason is None for reason in results.values()), results


def render_verbose(results: dict[str, str | None]) -> str:
    """Format check results the way the Kubernetes API server does."""
    return "\n".join(
        f"[+] {name} ok" if reason is None else f"[-] {name} failed: {reason}"
        for name, reason in results.items()
    )


def _status_fields(config: ProbeConfig) -> dict[str, Any]:
    fields = dict(config.status)
    fields.setdefault("rank", config.rank)
    fields.setdefault("leader", config.leader)
    fields.setdefault("port", config.port)
    fields.setdefault("served_model_name", config.served_model_name)
    return fields


def status_page(config: ProbeConfig) -> str:
    """A plain-text status page in the z-pages ``key=value`` field format."""
    fields = _status_fields(config)
    return "".join(f"{key}={value}\n" for key, value in sorted(fields.items()))


def status_json(config: ProbeConfig) -> dict[str, Any]:
    """The launcher's resolved-configuration record, for ``/statusz?format=json``."""
    return _status_fields(config)


def exit_code_for(results: dict[str, str | None]) -> int:
    """The status a command-line probe reports for failing checks."""
    if any(name == "cache" and reason for name, reason in results.items()):
        return EXIT_CACHE
    if any(name in {"engine", "leader"} and reason for name, reason in results.items()):
        return EXIT_ENGINE
    return EXIT_HTTP
