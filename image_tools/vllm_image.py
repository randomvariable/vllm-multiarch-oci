#!/usr/bin/env python3
"""Command line for the Kubernetes and Docker helpers shipped with the vLLM image.

Every subcommand here is a process the deployment manifest names directly: no
shell script, no inline Python, no ``sh -c``. ``launch`` is the server entry
point and the file lives in :mod:`image_tools.launcher.launch`; the commands
below are the pieces it reuses plus the operator-facing checks.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from image_tools.launcher import (
    EXIT_CACHE,
    EXIT_CONFIG,
    EXIT_DNS,
    EXIT_ENGINE,
    EXIT_HTTP,
    EXIT_MODEL,
    EXIT_PORT,
    ConfigError,
)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# NVFP4-CSF checkpoints ship as a container the loader reads through an
# explicit ``manifest.json`` and a ``metadata/`` directory; the conventional
# root-``config.json``-plus-root-index layout does not apply. ``_CSF_SCHEMA``
# is the one manifest schema this helper will stage, matching the pinned
# loader's ``read_csf_contract`` (third_party/vllm_nvfp4_csf_loader.patch).
_CSF_SCHEMA = "lil-nvfp4-csf-checkpoint/1"
_CSF_METADATA_DIR = "metadata"
_CSF_TENSORS_DIR = "tensors"
# The files ``read_csf_contract`` opens before any weight plus the metadata the
# serving directory is built from; the tensor shards live under ``tensors/`` and
# are enumerated from the index rather than pinned here.
_CSF_REQUIRED_FILES = (
    "manifest.json",
    "build-contract.json",
    "verification.json",
    f"{_CSF_METADATA_DIR}/config.json",
    f"{_CSF_METADATA_DIR}/tokenizer_config.json",
    f"{_CSF_METADATA_DIR}/chat_template.jinja",
    f"{_CSF_METADATA_DIR}/model.safetensors.index.json",
)

# Upstream ``prepare_csf_checkpoint`` derives the manifest-layout container's
# checkpoint identity by hashing ``manifest.json`` with this salt (it then hands
# the value to ``resolve_identity`` as ``target_identity`` so the identity step
# never hashes the generated serving directory). The staging is done here at
# sync time rather than at engine start, so the digest is computed and recorded
# in the sync marker; the exact byte recipe MUST stay in lockstep with
# ``image_tools.launcher.resolver``'s CSF contract.
_CSF_IDENTITY_SALT = b"lil-fp4-csf-v1\0"


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
    raise ConfigError(f"{label} is required (flag or {env_name or 'configured environment variable'})", EXIT_CONFIG)


def wait_for_leader(
    leader: str,
    port: int,
    *,
    dns_timeout: float = 1200.0,
    dns_period: float = 10.0,
    connect_timeout: float = 300.0,
    connect_period: float = 5.0,
    socket_timeout: float = 2.0,
) -> None:
    """Block until the leader's rendezvous port accepts a connection.

    A worker that starts its collective before the leader is listening fails in
    a way that is hard to tell apart from a network fault, so the wait is
    explicit: resolve the headless-service address, then connect.
    """
    if not leader:
        raise ConfigError("leader address is empty", EXIT_CONFIG)
    dns_deadline = time.monotonic() + dns_timeout
    addresses: list[tuple] = []
    last_dns_error = "no addresses"
    while time.monotonic() < dns_deadline:
        try:
            addresses = socket.getaddrinfo(leader, port, type=socket.SOCK_STREAM)
            if addresses:
                break
        except socket.gaierror as error:
            last_dns_error = str(error)
        time.sleep(min(dns_period, max(0, dns_deadline - time.monotonic())))
    if not addresses:
        raise ConfigError(
            f"rendezvous DNS timeout: {leader} did not resolve within {dns_timeout:g}s ({last_dns_error})",
            EXIT_DNS,
        )

    port_deadline = time.monotonic() + connect_timeout
    last_connect_error = "connection refused"
    while time.monotonic() < port_deadline:
        for family, socktype, proto, _canonname, sockaddr in addresses:
            try:
                with socket.socket(family, socktype, proto) as connection:
                    connection.settimeout(socket_timeout)
                    connection.connect(sockaddr)
                print(f"rendezvous: connected to {leader}:{port}", flush=True)
                return
            except OSError as error:
                last_connect_error = str(error)
        time.sleep(min(connect_period, max(0, port_deadline - time.monotonic())))
    raise ConfigError(
        f"rendezvous port timeout: {leader}:{port} did not accept a connection within {connect_timeout:g}s ({last_connect_error})",
        EXIT_PORT,
    )


def rendezvous_wait(args: argparse.Namespace) -> None:
    rank = int(_value_or_env(args.rank, args.rank_env, "rank"))
    if rank < 0:
        raise ConfigError("rank must be non-negative", EXIT_CONFIG)
    if rank == 0:
        print("rendezvous: leader rank 0; no wait required", flush=True)
        return
    wait_for_leader(
        _value_or_env(args.leader, args.leader_env, "leader"),
        args.port,
        dns_timeout=args.dns_timeout,
        dns_period=args.dns_period,
        connect_timeout=args.connect_timeout,
        connect_period=args.connect_period,
        socket_timeout=args.socket_timeout,
    )


def health(args: argparse.Namespace) -> None:
    """Ask a probe endpoint whether this container may serve or stay running.

    The checks themselves live in :mod:`image_tools.launcher.probes`; this is
    the command-line client the Docker ``healthcheck`` uses, so the container
    needs no shell to ask its own server a question.
    """
    try:
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
            status, body = int(response.status), response.read(4096)
    except urllib.error.HTTPError as error:
        status, body = int(error.code), error.read(4096)
    except (OSError, urllib.error.URLError) as error:
        raise ConfigError(f"probe {args.url} is unreachable: {error}", EXIT_HTTP) from error
    if status != 200:
        detail = body.decode("utf-8", "replace").strip() or f"status {status}"
        raise ConfigError(f"probe {args.url} failed: {detail}", EXIT_HTTP)
    print(f"health: {args.url} ready", flush=True)


def _safe_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or name in {"", "."}:
        raise ConfigError(f"invalid indexed shard path: {name!r}", EXIT_MODEL)
    return Path(*pure.parts)


def _default_env(name: str, value: str) -> None:
    """Supply a Hugging Face cache location unless the deployment pinned one.

    Deployments keep the Xet chunk cache on the JIT volume rather than the model
    store, so an explicit value wins.
    """
    if not os.environ.get(name):
        os.environ[name] = value


def _validate_snapshot(snapshot: Path, required: Sequence[str] = ()) -> int:
    config = snapshot / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ConfigError("incomplete model snapshot: config.json is missing or empty", EXIT_MODEL)
    shards: set[Path] = set()
    indexes = sorted(snapshot.glob("*.safetensors.index.json"))
    for index in indexes:
        try:
            data = json.loads(index.read_text())
            values = data["weight_map"].values()
        except (OSError, json.JSONDecodeError, KeyError, AttributeError) as error:
            raise ConfigError(f"invalid safetensors index {index.name}: {error}", EXIT_MODEL) from error
        shards.update(_safe_relative(str(name)) for name in values)
    if not indexes:
        shards.update(path.relative_to(snapshot) for path in snapshot.glob("*.safetensors"))
    if not shards:
        raise ConfigError("incomplete model snapshot: no safetensors weights or index found", EXIT_MODEL)
    bad = [str(path) for path in sorted(shards) if not (snapshot / path).is_file() or (snapshot / path).stat().st_size == 0]
    for name in required:
        path = _safe_relative(name)
        if not (snapshot / path).is_file() or (snapshot / path).stat().st_size == 0:
            bad.append(str(path))
    if bad:
        raise ConfigError(f"incomplete model snapshot: missing or empty files: {', '.join(bad)}", EXIT_MODEL)
    return len(shards)


def _csf_missing(snapshot: Path, paths: Sequence[str]) -> list[str]:
    """Return the snapshot-relative paths under *snapshot* that are absent or empty.

    Each name is passed through :func:`_safe_relative`, so an index or a
    deployment that reaches outside the snapshot fails closed the same way the
    conventional check does.
    """
    missing: list[str] = []
    for name in paths:
        path = _safe_relative(name)
        target = snapshot / path
        if not target.is_file() or target.stat().st_size == 0:
            missing.append(str(path))
    return missing


def _validate_csf_snapshot(snapshot: Path, family: str, required: Sequence[str] = ()) -> int:
    """Verify one NVFP4-CSF container; return the tensor-shard count.

    Detection is explicit and fails closed. The manifest must parse and name
    :data:`_CSF_SCHEMA` together with the *family* the caller declared, so a
    snapshot that is not the expected CSF container aborts instead of being
    mistaken for a conventional one. The container files and every shard the
    ``metadata/`` index references under ``tensors/`` must exist and be
    non-empty before anything is published.
    """
    manifest = snapshot / "manifest.json"
    try:
        contract = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"invalid CSF manifest {manifest.name}: {error}", EXIT_MODEL) from error
    if not isinstance(contract, dict):
        raise ConfigError(f"invalid CSF manifest {manifest.name}: not a JSON object", EXIT_MODEL)
    schema = contract.get("schema")
    actual_family = contract.get("family")
    if schema != _CSF_SCHEMA or actual_family != family:
        raise ConfigError(
            f"unexpected CSF contract {schema}/{actual_family}, "
            f"require {_CSF_SCHEMA}/{family}",
            EXIT_MODEL,
        )
    index = snapshot / _CSF_METADATA_DIR / "model.safetensors.index.json"
    try:
        document = json.loads(index.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"invalid CSF safetensors index {index.name}: {error}", EXIT_MODEL) from error
    weight_map = document.get("weight_map") if isinstance(document, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConfigError(f"invalid CSF safetensors index {index.name}: empty or missing weight_map", EXIT_MODEL)
    shards = {str(name) for name in weight_map.values()}
    missing = _csf_missing(snapshot, _CSF_REQUIRED_FILES)
    missing += _csf_missing(snapshot, [f"{_CSF_TENSORS_DIR}/{shard}" for shard in sorted(shards)])
    missing += _csf_missing(snapshot, required)
    if missing:
        raise ConfigError(f"incomplete CSF snapshot: missing or empty files: {', '.join(missing)}", EXIT_MODEL)
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
        raise ConfigError(f"publish path exists and is not a symlink: {destination}", EXIT_MODEL)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.symlink_to(snapshot)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_csf_serving(snapshot: Path, generated: Path) -> None:
    """Build the immutable ``nvfp4_csf`` serving directory at *generated*.

    vLLM cannot load the CSF container directly: there is no root ``config.json``.
    The serving directory is the snapshot's ``metadata/`` files lifted to its root
    with ``config.json``'s ``quantization_config`` replaced by the ``nvfp4_csf``
    block whose ``checkpoint_root`` points back at the snapshot -- exactly what the
    ``prepare-csf`` init container staged by hand.

    A deployment reads weights straight out of the snapshot for the whole process
    lifetime (disk-PLE), so the serving directory must NEVER be rebuilt in place:
    that would unlink files live readers hold open. The directory is derived purely
    from the immutable snapshot, written once to a temp sibling and brought into
    place with a single ``os.replace``, and left untouched on every later sync of
    the same revision. Repointing the published path is then an atomic symlink
    swap done by :func:`_publish`, which refuses to clobber a real directory.
    """
    if generated.is_dir():
        return
    metadata = snapshot / _CSF_METADATA_DIR
    staging_root = generated.parent
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f".{generated.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for source in sorted(metadata.iterdir()):
            if source.is_file():
                shutil.copy2(source, temporary / source.name)
        config = temporary / "config.json"
        try:
            document = json.loads(config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(f"invalid CSF serving config {config.name}: {error}", EXIT_MODEL) from error
        if not isinstance(document, dict):
            raise ConfigError(f"invalid CSF serving config {config.name}: not a JSON object", EXIT_MODEL)
        source_qc = document.get("quantization_config")
        if not isinstance(source_qc, dict):
            raise ConfigError("CSF serving config carries no quantization_config to wrap", EXIT_MODEL)
        document["quantization_config"] = {
            "quant_method": "nvfp4_csf",
            "format_version": 1,
            "checkpoint_root": str(snapshot),
            "source_quantization_config": source_qc,
        }
        config.write_text(json.dumps(document))
        if generated.is_symlink():
            generated.unlink()
        os.replace(temporary, generated)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def csf_manifest_identity(manifest: Path) -> str:
    """Return the manifest-layout container's checkpoint identity digest.

    Hashes ``manifest.json``'s bytes with :data:`_CSF_IDENTITY_SALT`, matching
    upstream ``prepare_csf_checkpoint``. The identity therefore describes the
    container that was validated and staged, never the generated serving
    directory, so republishing or touching the serving tree cannot move it.
    """
    return hashlib.sha256(_CSF_IDENTITY_SALT + manifest.read_bytes()).hexdigest()


def csf_serving_identity(storage_root: Path, repo: str, revision: str) -> dict[str, str] | None:
    """Return ``{"identity": <sha256 hex>, "revision": ""}`` for a staged CSF snapshot.

    The launcher sets ``ResolvedPlan.target_identity`` from this so the external
    cache resolves identity from the digest of what ``sync_model`` actually put on
    disk instead of hashing the generated serving directory -- the same value
    upstream ``runtime/cache.py`` ``resolve_identity`` prefers over
    ``resolve_checkpoint``. Returns ``None`` when nothing CSF-shaped is staged at
    this revision (conventional snapshots carry no ``manifest.json`` and no
    recorded ``identity``), so callers fall back to the ordinary checkpoint hash.
    """
    root = Path(storage_root).resolve()
    marker = root / ".vllm-image" / "model-sync" / f"{_marker_key(repo, revision)}.json"
    if not marker.is_file():
        return None
    try:
        saved = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if saved.get("repo") != repo or saved.get("revision") != revision:
        return None
    identity = saved.get("identity")
    if not identity:
        snapshot = saved.get("snapshot")
        manifest = Path(snapshot) / "manifest.json" if snapshot else None
        if manifest is not None and manifest.is_file():
            identity = csf_manifest_identity(manifest)
    if not identity:
        return None
    return {"identity": identity, "revision": ""}


def sync_model(
    *,
    repo: str,
    revision: str,
    storage_root: Path,
    publish: Path,
    min_free_bytes: int | None = None,
    min_free_gib: float = 0.0,
    workers: int = 8,
    ignore: Sequence[str] = (),
    require: Sequence[str] = (),
    token: str | None = None,
    token_env: str | None = "HF_TOKEN",
    csf: bool = False,
    csf_family: str | None = None,
) -> int:
    """Fetch, verify and publish one checkpoint revision; return the shard count.

    The launcher calls this before it execs the engine, which is why there are
    no model-download init containers. A revision that is already published and
    still complete is re-validated rather than re-downloaded, and the marker is
    written only after every index-referenced shard verified, so a rank never
    opens a half-written snapshot.

    When *csf* is set the checkpoint is an NVFP4-CSF container instead: the
    manifest schema and *csf_family* are validated (fail closed), an immutable
    ``nvfp4_csf`` serving directory is generated once under the state root, and
    ``--publish`` becomes an atomically repointed symlink to it (never a
    rebuild-in-place that would unlink files a disk-PLE reader holds open), so a
    recipe can declare the checkpoint and let the engine load it without a
    hand-written init container.
    """
    if not _COMMIT.fullmatch(revision):
        raise ConfigError("revision must be a full lowercase 40-hex Hugging Face commit OID", EXIT_CONFIG)
    if csf and not csf_family:
        raise ConfigError("CSF sync requires the expected manifest family", EXIT_CONFIG)

    def validate(target: Path) -> int:
        if csf:
            return _validate_csf_snapshot(target, csf_family, require)
        return _validate_snapshot(target, require)

    root = Path(storage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _default_env("HF_HOME", str(root))
    _default_env("HF_HUB_CACHE", str(root / "hub"))
    _default_env("HF_XET_CACHE", str(root / "xet"))
    stats = os.statvfs(root)
    free = stats.f_bavail * stats.f_frsize
    minimum = min_free_bytes if min_free_bytes is not None else int(min_free_gib * 1024**3)
    if free < minimum:
        raise ConfigError(f"insufficient storage: {free} bytes free, require {minimum}", EXIT_MODEL)
    state = root / ".vllm-image" / "model-sync"
    state.mkdir(parents=True, exist_ok=True)
    marker = state / f"{_marker_key(repo, revision)}.json"
    lock_path = state / f"{_marker_key(repo, revision)}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        snapshot: Path | None = None
        if marker.is_file():
            try:
                saved = json.loads(marker.read_text())
                if saved.get("repo") == repo and saved.get("revision") == revision:
                    snapshot = Path(saved["snapshot"])
                    validate(snapshot)
            except (OSError, KeyError, ValueError, ConfigError):
                snapshot = None
        if snapshot is None:
            from huggingface_hub import snapshot_download

            credential = token or (os.environ.get(token_env) if token_env else None)
            snapshot = Path(snapshot_download(
                repo_id=repo,
                revision=revision,
                cache_dir=str(root / "hub"),
                max_workers=workers,
                ignore_patterns=list(ignore) or None,
                token=credential,
            )).resolve()
            count = validate(snapshot)
            record = {"repo": repo, "revision": revision, "snapshot": str(snapshot)}
            if csf:
                record["identity"] = csf_manifest_identity(snapshot / "manifest.json")
            _atomic_json(marker, record)
        else:
            count = validate(snapshot)
        if csf:
            generated = state / f"{_marker_key(repo, revision)}.csf-serve"
            _stage_csf_serving(snapshot, generated)
            _publish(generated, Path(publish).absolute())
        else:
            _publish(snapshot, Path(publish).absolute())
    print(f"model-sync: published {repo}@{revision} ({count} shards) at {publish}", flush=True)
    return count


def model_sync(args: argparse.Namespace) -> None:
    sync_model(
        repo=args.repo,
        revision=args.revision,
        storage_root=args.storage_root,
        publish=args.publish,
        min_free_bytes=args.min_free_bytes,
        min_free_gib=args.min_free_gib,
        workers=args.workers,
        ignore=args.ignore,
        require=args.require,
        token=args.token,
        token_env=args.token_env,
        csf=args.csf,
        csf_family=args.csf_family,
    )


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
    # Checkpoints carry assets vLLM loads after the weights, such as the
    # tokenizer template and configuration. A deployment that depends on one
    # names it here instead of re-checking the snapshot in a separate container.
    sync.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="PATH",
        help="snapshot-relative file that must exist and be non-empty",
    )
    token = sync.add_mutually_exclusive_group()
    token.add_argument("--token")
    token.add_argument("--token-env", default="HF_TOKEN")
    # An NVFP4-CSF container has no root config.json and keeps its weights in
    # tensors/; --csf switches the sync over to that contract and the serving
    # directory the loader expects, validated against the declared family.
    sync.add_argument(
        "--csf",
        action="store_true",
        help="stage an NVFP4-CSF container instead of a conventional checkpoint",
    )
    sync.add_argument(
        "--csf-family",
        help="expected CSF manifest family (required with --csf)",
    )
    sync.set_defaults(run=model_sync)

    probe = commands.add_parser("health")
    probe.add_argument("--url", required=True)
    probe.add_argument("--timeout", type=_duration, default=8)
    probe.set_defaults(run=health)

    # The probe server imports only the standard library, so registering it here
    # costs nothing. The launcher module pulls the resolver, and the resolver needs
    # PyYAML: importing it at parser-build time would make ``--help`` and every
    # other subcommand depend on a package the wheel tests do not carry. It is
    # therefore imported where it is used, in :func:`main`.
    from image_tools.launcher import probe_server as _probe_server

    # Registered so ``--help`` lists them; :func:`main` intercepts these two
    # commands before argparse sees them. Their own options begin with ``--``,
    # which this parser would otherwise claim as unknown arguments of the
    # subcommand.
    launch = commands.add_parser("launch", help="resolve a recipe or profile and exec the engine", add_help=False)
    launch.add_argument("arguments", nargs=argparse.REMAINDER)
    entrypoint = commands.add_parser("entrypoint", help="the image entry point", add_help=False)
    entrypoint.add_argument("arguments", nargs=argparse.REMAINDER)
    _probe_server.parser(commands)
    return root


def main() -> int:
    signal.signal(signal.SIGTERM, _terminated)
    argv = list(sys.argv[1:])
    if argv and argv[0] in {"launch", "entrypoint"}:
        # Imported here, not at module scope or in :func:`parser`: the launcher
        # pulls the resolver, which needs PyYAML, and a command like ``health``
        # must work in an environment that does not carry it.
        from image_tools.launcher import launch as _launch

        # These two carry their own ``--`` options, so they parse themselves; the
        # dispatcher above only lists them in ``--help``.
        arguments = (argv[1:],)
        handler = _launch.run if argv[0] == "launch" else _launch.entrypoint
    else:
        try:
            parsed = parser().parse_args(argv)
        except SystemExit as exit_request:  # --help and usage errors
            return int(exit_request.code or 0)
        arguments, handler = (parsed,), parsed.run
    try:
        return int(handler(*arguments) or 0)
    except ConfigError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return error.code
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
