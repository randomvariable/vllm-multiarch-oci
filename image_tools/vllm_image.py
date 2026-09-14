#!/usr/bin/env python3
"""Model-agnostic Kubernetes helpers shipped with the vLLM image."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


EXIT_CONFIG = 2
EXIT_DNS = 10
EXIT_PORT = 11
EXIT_HTTP = 20
EXIT_ENGINE = 21
EXIT_MODEL = 30
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CommandError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _terminated(_signum: int, _frame: object) -> None:
    raise SystemExit(128 + signal.SIGTERM)


def _duration(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)?", value)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}")
    scale = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return float(match.group(1)) * scale[match.group(2)]


def _value_or_env(value: str | int | None, env_name: str | None, label: str) -> str:
    if value is not None:
        return str(value)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    raise CommandError(f"{label} is required (flag or {env_name or 'configured environment variable'})", EXIT_CONFIG)


def rendezvous_wait(args: argparse.Namespace) -> None:
    rank = int(_value_or_env(args.rank, args.rank_env, "rank"))
    if rank < 0:
        raise CommandError("rank must be non-negative", EXIT_CONFIG)
    if rank == 0:
        print("rendezvous: leader rank 0; no wait required", flush=True)
        return
    leader = _value_or_env(args.leader, args.leader_env, "leader")
    dns_deadline = time.monotonic() + args.dns_timeout
    addresses: list[tuple] = []
    last_dns_error = "no addresses"
    while time.monotonic() < dns_deadline:
        try:
            addresses = socket.getaddrinfo(leader, args.port, type=socket.SOCK_STREAM)
            if addresses:
                break
        except socket.gaierror as error:
            last_dns_error = str(error)
        time.sleep(min(args.dns_period, max(0, dns_deadline - time.monotonic())))
    if not addresses:
        raise CommandError(
            f"rendezvous DNS timeout: {leader} did not resolve within {args.dns_timeout:g}s ({last_dns_error})",
            EXIT_DNS,
        )

    port_deadline = time.monotonic() + args.connect_timeout
    last_connect_error = "connection refused"
    while time.monotonic() < port_deadline:
        for family, socktype, proto, _canonname, sockaddr in addresses:
            try:
                with socket.socket(family, socktype, proto) as connection:
                    connection.settimeout(args.socket_timeout)
                    connection.connect(sockaddr)
                print(f"rendezvous: rank {rank} connected to {leader}:{args.port}", flush=True)
                return
            except OSError as error:
                last_connect_error = str(error)
        time.sleep(min(args.connect_period, max(0, port_deadline - time.monotonic())))
    raise CommandError(
        f"rendezvous port timeout: {leader}:{args.port} did not accept a connection within {args.connect_timeout:g}s ({last_connect_error})",
        EXIT_PORT,
    )


_ENGINE_COMMS = {"VLLM::EngineCor", "VLLM::Worker_TP"}


def _process_state(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return (proc_root / str(pid) / "stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return None


def _engine_alive(pid: int, proc_root: Path = Path("/proc")) -> bool:
    return _process_state(pid, proc_root) not in {None, "X", "Z"}


def _engine_child(parent_pid: int, proc_root: Path = Path("/proc")) -> int:
    try:
        children = (proc_root / str(parent_pid) / "task" / str(parent_pid) / "children").read_text().split()
    except OSError as error:
        raise CommandError(f"health engine failure: cannot inspect children of PID {parent_pid}: {error}", EXIT_ENGINE) from error
    engines: list[int] = []
    for value in children:
        try:
            pid = int(value)
            comm = (proc_root / value / "comm").read_text().rstrip("\n")
        except (OSError, ValueError):
            continue
        if comm in _ENGINE_COMMS and _engine_alive(pid, proc_root):
            engines.append(pid)
    if not engines:
        raise CommandError(
            f"health engine failure: PID {parent_pid} has no live direct child with comm in {sorted(_ENGINE_COMMS)}",
            EXIT_ENGINE,
        )
    if len(engines) != 1:
        raise CommandError(
            f"health engine failure: PID {parent_pid} has ambiguous live engine children: {engines}",
            EXIT_ENGINE,
        )
    return engines[0]


def _http_ready(url: str, timeout: float) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise CommandError(f"health HTTP failure: {url} returned {response.status}", EXIT_HTTP)
            response.read(1)
    except CommandError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise CommandError(f"health HTTP failure: {url}: {error}", EXIT_HTTP) from error


def health(args: argparse.Namespace) -> None:
    rank = int(_value_or_env(args.rank, args.rank_env, "rank"))
    if rank < 0:
        raise CommandError("rank must be non-negative", EXIT_CONFIG)
    if rank == 0:
        if not args.local_url:
            raise CommandError("--local-url is required for leader rank", EXIT_CONFIG)
        url = args.local_url
    else:
        if args.engine_pid is not None:
            engine_pid = args.engine_pid
        elif args.engine_parent_pid is not None:
            engine_pid = _engine_child(args.engine_parent_pid)
        else:
            raise CommandError("--engine-pid or --engine-parent-pid is required for worker ranks", EXIT_CONFIG)
        if not _engine_alive(engine_pid):
            raise CommandError(f"health engine failure: PID {engine_pid} is not alive", EXIT_ENGINE)
        if not args.leader_url:
            raise CommandError("--leader-url is required for worker ranks", EXIT_CONFIG)
        url = args.leader_url
    _http_ready(url, args.timeout)
    print(f"health: {args.phase} rank {rank} ready", flush=True)


def _safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or name in {"", "."}:
        raise CommandError(f"invalid indexed shard path: {name!r}", EXIT_MODEL)
    return Path(*pure.parts)


def _validate_snapshot(snapshot: Path) -> int:
    config = snapshot / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise CommandError("incomplete model snapshot: config.json is missing or empty", EXIT_MODEL)
    shards: set[Path] = set()
    indexes = sorted(snapshot.glob("*.safetensors.index.json"))
    for index in indexes:
        try:
            data = json.loads(index.read_text())
            values = data["weight_map"].values()
        except (OSError, json.JSONDecodeError, KeyError, AttributeError) as error:
            raise CommandError(f"invalid safetensors index {index.name}: {error}", EXIT_MODEL) from error
        shards.update(_safe_relative(str(name)) for name in values)
    if not indexes:
        shards.update(path.relative_to(snapshot) for path in snapshot.glob("*.safetensors"))
    if not shards:
        raise CommandError("incomplete model snapshot: no safetensors weights or index found", EXIT_MODEL)
    bad = [str(path) for path in sorted(shards) if not (snapshot / path).is_file() or (snapshot / path).stat().st_size == 0]
    if bad:
        raise CommandError(f"incomplete model snapshot: missing or empty shards: {', '.join(bad)}", EXIT_MODEL)
    return len(shards)


def _marker_key(repo: str, revision: str) -> str:
    return hashlib.sha256(f"{repo}\0{revision}".encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _publish(snapshot: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        raise CommandError(f"publish path exists and is not a symlink: {destination}", EXIT_MODEL)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.symlink_to(snapshot)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def model_sync(args: argparse.Namespace) -> None:
    if not _COMMIT.fullmatch(args.revision):
        raise CommandError("revision must be a full lowercase 40-hex Hugging Face commit OID", EXIT_CONFIG)
    root = args.storage_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    minimum = args.min_free_bytes if args.min_free_bytes is not None else int(args.min_free_gib * 1024**3)
    if free < minimum:
        raise CommandError(f"insufficient storage: {free} bytes free, require {minimum}", EXIT_MODEL)
    state = root / ".vllm-image" / "model-sync"
    state.mkdir(parents=True, exist_ok=True)
    marker = state / f"{_marker_key(args.repo, args.revision)}.json"
    lock_path = state / f"{_marker_key(args.repo, args.revision)}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot: Path | None = None
        if marker.is_file():
            try:
                saved = json.loads(marker.read_text())
                if saved.get("repo") == args.repo and saved.get("revision") == args.revision:
                    snapshot = Path(saved["snapshot"])
                    _validate_snapshot(snapshot)
            except (OSError, KeyError, json.JSONDecodeError, CommandError):
                snapshot = None
        if snapshot is None:
            from huggingface_hub import snapshot_download

            token = args.token or (os.environ.get(args.token_env) if args.token_env else None)
            snapshot = Path(snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                cache_dir=str(root / "hub"),
                max_workers=args.workers,
                ignore_patterns=args.ignore or None,
                token=token,
            )).resolve()
            count = _validate_snapshot(snapshot)
            _atomic_json(marker, {"repo": args.repo, "revision": args.revision, "snapshot": str(snapshot)})
        else:
            count = _validate_snapshot(snapshot)
        _publish(snapshot, args.publish.absolute())
    print(f"model-sync: published {args.repo}@{args.revision} ({count} shards) at {args.publish}", flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="vllm-image")
    commands = root.add_subparsers(dest="command", required=True)
    rendezvous = commands.add_parser("rendezvous-wait")
    rendezvous.add_argument("--rank", type=int)
    rendezvous.add_argument("--rank-env", default="LWS_WORKER_INDEX")
    rendezvous.add_argument("--leader")
    rendezvous.add_argument("--leader-env", default="LWS_LEADER_ADDRESS")
    rendezvous.add_argument("--port", type=int, required=True)
    rendezvous.add_argument("--dns-timeout", type=_duration, default=1200)
    rendezvous.add_argument("--dns-period", type=_duration, default=10)
    rendezvous.add_argument("--connect-timeout", type=_duration, default=300)
    rendezvous.add_argument("--connect-period", type=_duration, default=5)
    rendezvous.add_argument("--socket-timeout", type=_duration, default=2)
    rendezvous.set_defaults(run=rendezvous_wait)

    sync = commands.add_parser("model-sync")
    sync.add_argument("--repo", required=True)
    sync.add_argument("--revision", required=True)
    sync.add_argument("--storage-root", type=Path, required=True)
    sync.add_argument("--publish", type=Path, required=True)
    minimum = sync.add_mutually_exclusive_group()
    minimum.add_argument("--min-free-bytes", type=int)
    minimum.add_argument("--min-free-gib", type=float, default=0)
    sync.add_argument("--workers", type=int, default=8)
    sync.add_argument("--ignore", action="append", default=[])
    token = sync.add_mutually_exclusive_group()
    token.add_argument("--token")
    token.add_argument("--token-env", default="HF_TOKEN")
    sync.set_defaults(run=model_sync)

    probe = commands.add_parser("health")
    probe.add_argument("--phase", choices=("startup", "readiness"), required=True)
    probe.add_argument("--rank", type=int)
    probe.add_argument("--rank-env", default="LWS_WORKER_INDEX")
    probe.add_argument("--local-url")
    probe.add_argument("--leader-url")
    probe.add_argument("--engine-pid", type=int)
    probe.add_argument("--engine-parent-pid", type=int)
    probe.add_argument("--timeout", type=_duration, default=8)
    probe.set_defaults(run=health)
    return root


def main() -> int:
    signal.signal(signal.SIGTERM, _terminated)
    try:
        args = parser().parse_args()
        args.run(args)
        return 0
    except CommandError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return error.code
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
