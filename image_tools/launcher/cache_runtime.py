"""Runtime support for the ``vllm-image launch --role cache`` container.

These are the pure, in-process steps the cache container performs before it
``exec``s ``lmcache server``: it claims the shared-memory arena and its disk
tier so a second live owner is refused, waits for the started cache to answer
its health URL, checks that the cache advertises the exact engine-driven SHM
pool the server role resolved, and prunes disk-tier namespaces left behind by
earlier images or cache settings.

Ported from upstream ``runtime/supervisor.py`` (pinned at
``353efc679f631206e0b001e67047dea80ee6d76e``). Supervision is intentionally
absent: there is no ``Popen``, no process tree, no signal handling and no HTTP
server here. The launcher owns starting ``lmcache server`` and the lifetime of
the descriptors returned by the claim helpers; the arena and tier locks are
held on file descriptors kept open in this module for the process lifetime so
they survive ``os.execv`` (``FD_CLOEXEC`` is never set).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from image_tools.launcher import ConfigError

# A tier written this recently may belong to an image that predates the lock.
RECENT_TIER_WRITE_SECONDS = 1800
# Each running cache holds a shared lock on its disk tier. A tier whose lock can
# be taken exclusively has no running owner that took the lock.
TIER_LOCK = ".lil-tier.lock"

# ``/dev/shm`` where ``claim_arena`` materialises ``<shm_name>.lock``; the probe
# container shares it with the cache container and can test the same arena lock.
SHM_ROOT = Path("/dev/shm")

# Arena and tier locks stay open for the lifetime of this process. The kernel
# releases an flock when its holder exits, including on SIGKILL or an OOM kill,
# so a held lock proves a running owner and a free one proves the arena is
# stale. The launcher ``exec``s the cache server, so the descriptors must NOT be
# close-on-exec; they are cleared of ``FD_CLOEXEC`` when opened.
_ARENA_LOCKS: list[int] = []
_TIER_LOCKS: list[int] = []


def read_json(url: str) -> dict:
    """Fetch ``url`` as JSON, never routing loopback through an HTTP proxy."""
    # Internal service readiness must not be redirected through an HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=1) as response:
        return json.load(response)


def _keep_open(fd: int) -> None:
    """Ensure ``fd`` is inherited across ``exec`` so the flock survives it."""
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    fcntl.fcntl(fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)


def _owner_token_path(lock_path: Path) -> Path:
    return lock_path.with_name(lock_path.name + ".owner")


def _process_start_time(pid: int) -> int | None:
    """The process start time in clock ticks from field 22 of ``/proc/<pid>/stat``.

    It distinguishes a reused PID from the original holder: a PID recycled to a
    new process has a different start time, so a token naming a gone-and-replaced
    process must not read as alive. Returns None when the process does not exist
    or ``stat`` cannot be parsed. The comm field can hold spaces and parentheses,
    so the split is taken after the last ``)``.
    """
    try:
        with open(f"/proc/{pid}/stat") as handle:
            data = handle.read()
    except OSError:
        return None
    close = data.rfind(")")
    if close < 0:
        return None
    fields = data[close + 1 :].split()
    # fields[0] is the state field (field 3); starttime is field 22.
    try:
        return int(fields[22 - 3])
    except (IndexError, ValueError):
        return None


def _write_owner_token(lock_path: Path) -> None:
    """Record which process claimed ``lock_path`` next to it.

    The ownership token lets a probe tell a fresh start (no token) from a dead
    owner (token naming a gone or replaced process) without ever touching the
    lock it watches. It carries the holder pid plus that process's start time
    (``/proc/<pid>/stat`` field 22), so a reused PID cannot impersonate the
    original owner. It is a plain write; the close-on-exec discipline that
    matters for the lock descriptor is irrelevant here.
    """
    token = _owner_token_path(lock_path)
    pid = os.getpid()
    payload = json.dumps(
        {"pid": pid, "start": _process_start_time(pid), "holder": "vllm-image"}
    )
    fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)


def _owner_token_is_live(token_path: Path) -> bool:
    """Whether ``token_path`` names a process alive with its recorded start time."""
    try:
        with open(token_path) as handle:
            token = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(token, dict):
        return False
    pid = token.get("pid")
    start = token.get("start")
    if not isinstance(pid, int) or not isinstance(start, int):
        return False
    return _process_start_time(pid) == start


def claim_arena(shm_path: Path) -> None:
    """Lock the arena name for this container or refuse if another owns it.

    Ported verbatim from upstream ``_claim_arena``. The lock is taken
    ``LOCK_EX | LOCK_NB`` on ``<shm_path>.lock``; a ``BlockingIOError`` means
    another live holder owns the name, which is refused with upstream's
    message. The descriptor is kept open in ``_ARENA_LOCKS`` (and cleared of
    ``FD_CLOEXEC``) so it is held for the process lifetime and across the
    launcher's ``exec`` of the cache server.
    """
    shm = Path(shm_path)
    fd = os.open(shm.with_name(shm.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _keep_open(fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ConfigError(
            f"Refusing to reuse a cache SHM arena owned by another running container: {shm}"
        ) from None
    _ARENA_LOCKS.append(fd)
    _write_owner_token(shm.with_name(shm.name + ".lock"))


def claim_tier(tier_dir: Path) -> None:
    """Hold a shared lock on this container's LMCache disk tier.

    Ported from the tier-claiming half of upstream ``report_stale_tiers``: a
    running cache takes ``LOCK_SH`` on ``<tier_dir>/.lil-tier.lock``. If the
    lock cannot be taken, another container is mid-deletion of the same tier
    (the exclusive side is ``prune_stale_tiers``), which is refused with
    upstream's message. The descriptor is retained for the process lifetime, so
    the ``RECENT_TIER_WRITE_SECONDS`` grace that lets a tier written by a
    lock-predating image survive ``prune_stale_tiers`` applies to every tier
    this container has claimed.
    """
    base = Path(tier_dir)
    base.mkdir(parents=True, exist_ok=True)
    fd = os.open(base / TIER_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _keep_open(fd)
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ConfigError(
            f"Another container is deleting the LMCache disk tier {base}"
        ) from None
    _TIER_LOCKS.append(fd)
    _write_owner_token(base / TIER_LOCK)


def wait_ready(health_url: str, timeout: float) -> None:
    """Block until the cache answers ``health_url`` or ``timeout`` elapses.

    Ported from the readiness loop of upstream ``_supervise``: it opens the
    cache ``/healthcheck`` URL with the proxy handler disabled, retrying while
    the request fails. With no child process to poll here, only the deadline
    branch of the upstream loop is reproduced; on expiry it raises the same
    ``ConfigError`` upstream uses ("LMCache startup timed out"). The
    exited-before-readiness branch is left to the launcher, which owns the
    cache process.
    """
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while True:
        if time.monotonic() >= deadline:
            raise ConfigError("LMCache startup timed out")
        try:
            # /healthcheck can be non-JSON; reaching it at all is the contract.
            with opener.open(health_url, timeout=1):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)


def validate_arena(
    shm_name: str, expected_bytes: int, *, shm_root: Path = SHM_ROOT
) -> None:
    """Require the cache to have created the resolved shared-memory arena.

    Upstream ``validate_pool`` proved the same thing by asking the server to
    advertise an ``engine_driven_shm_pool`` field, but that field is produced
    only by an unmerged LMCache patch; our clean pin (``9cebd405``) never emits
    it, so ported verbatim it would fail on every start of our own image. The
    arena is instead measured where it lives: ``<shm_root>/<shm_name>`` must
    exist and be at least the resolved size. A server that silently fell back to
    the pickle transport creates no POSIX segment
    (``lmcache/v1/multiprocess/transfer_context/shm.py`` opens the arena with
    ``create=False`` and only ever in SHM mode), which is exactly the case this
    refuses. Keeps upstream's "no pickle fallback" wording. The caller passes the
    normalized arena name the server uses (``lmcache/v1/multiprocess/engine_context.py``
    prefixes ``lmcache_l1_pool_`` and strips a leading slash).
    """
    arena = shm_root / shm_name
    if not arena.exists():
        raise ConfigError(
            "LMCache must advertise the requested SHM arena and capacity; no pickle fallback. "
            f"Expected arena {str(arena)!r} of at least {expected_bytes} bytes, "
            "but the segment does not exist"
        )
    size = arena.stat().st_size
    if size < expected_bytes:
        raise ConfigError(
            "LMCache must advertise the requested SHM arena and capacity; no pickle fallback. "
            f"Expected arena {str(arena)!r} of at least {expected_bytes} bytes, "
            f"received {size} bytes"
        )


def status_url(health_url: str) -> str:
    """Derive the LMCache ``/status`` URL from its ``/healthcheck`` URL.

    ``/healthcheck`` only reports ``{"status": "healthy"}`` and carries neither
    pool geometry nor instance identity; the restart-sensitive observables --
    ``engine.report_status()``, e.g. ``registered_non_cuda_instance_ids`` from
    the engine-driven module -- come from ``/status``. Replaces a trailing
    ``/healthcheck`` with ``/status`` and raises ``ConfigError`` when the URL
    does not end in ``/healthcheck``, so a misspelled URL fails loudly instead
    of comparing nothing.
    """
    suffix = "/healthcheck"
    if not health_url.endswith(suffix):
        raise ConfigError(
            f"health URL must end in {suffix!r} to derive the status endpoint: "
        )
    return health_url[: -len(suffix)] + "/status"


def _tree_bytes(path: Path) -> int:
    """Bytes allocated on disk by the files below ``path``."""
    total = 0
    stack = [path]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_blocks * 512
                except OSError:
                    continue
    return total


def _tree_newest(path: Path) -> float:
    """Newest modification time of the files below ``path``."""
    newest = 0.0
    stack = [path]
    while stack:
        try:
            entries = os.scandir(stack.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
    return newest


def _announce(message: str) -> None:
    print(f"[vllm-image] {message}", file=sys.stderr, flush=True)


def prune_stale_tiers(
    root: Path, keep_namespace: Path, *, prune: bool = False
) -> None:
    """Name -- and with ``prune``, delete -- disk-tier namespaces of earlier images.

    Ported from the stale-tier half of upstream ``report_stale_tiers``. A tier
    lives in ``<root>/<layout>/<checkpoint>``; ``keep_namespace`` is the tier
    this container claimed and is never a candidate. Under each layout every
    other directory is stale. A stale tier is deleted only when ``prune`` is
    set (upstream's ``service.prune_stale_tiers``) and its ``.lil-tier.lock``
    can be taken exclusively -- proving no running container owns it -- and it
    was not written within ``RECENT_TIER_WRITE_SECONDS`` (the grace for a tier
    written by an image that predates the lock). Otherwise it is reported as
    kept. Upstream couples this with the shared-lock claim; the launcher calls
    ``claim_tier`` for its own tier and this function for the siblings.
    """
    root = Path(root)
    keep = Path(keep_namespace)
    stale: list[Path] = []
    for layout in sorted(root.iterdir()):
        if layout.is_symlink() or not layout.is_dir():
            continue
        for tier in sorted(layout.iterdir()):
            if tier == keep or tier.is_symlink() or not tier.is_dir():
                continue
            stale.append(tier)
    if not stale:
        return
    now = time.time()
    removed, kept = [], []
    for tier in stale:
        size = _tree_bytes(tier)
        newest = _tree_newest(tier)
        lock = tier / TIER_LOCK
        owner = None
        if lock.exists():
            owner = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(owner)
                kept.append((tier, size, "in use by a running container"))
                continue
        try:
            if not prune:
                written = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))
                kept.append((tier, size, f"last written {written}"))
            elif now - newest < RECENT_TIER_WRITE_SECONDS:
                kept.append((tier, size, "written in the last 30 minutes"))
            else:
                shutil.rmtree(tier)
                removed.append((tier, size, "removed"))
        finally:
            if owner is not None:
                os.close(owner)
    for label, tiers in (("Removed", removed), ("Kept", kept)):
        if not tiers:
            continue
        total = sum(size for _, size, _ in tiers)
        listing = "; ".join(
            f"{tier} ({size / 1024**3:.0f} GiB, {why})" for tier, size, why in tiers
        )
        _announce(
            f"{label} {len(tiers)} disk-tier namespace(s) of earlier images or "
            f"cache settings: {listing}"
        )
    if kept and not prune:
        _announce(
            "Set LMCACHE_L2_PRUNE_STALE=1 to delete the ones no running "
            "container uses at startup, or delete them by hand"
        )


# ``live_owner`` deliberately does NOT read ``/proc/locks``. A naive match on
# (major, minor, inode) breaks on overlay and network filesystems: on ``/home``
# (an overlay) ``os.stat().st_dev`` reports the overlay's anonymous device
# (major 0, minor 51) while the kernel records the flock against the upper
# filesystem's device (major 0, minor 35), and ``/cache`` is a ZFS CSI volume
# with the same class of mismatch, so a flock-observability probe reports a live
# cache as dead. Ownership is instead proven by the ``.owner`` token naming a
# process that is alive with its recorded ``/proc/<pid>/stat`` start time;
# neither ``flock`` nor ``/proc/locks`` is consulted, so the probe can never
# steal a lock it is testing nor misjudge a filesystem.


def live_owner(
    shm_name: str, tier_dirs: Sequence[Path], *, shm_root: Path = SHM_ROOT
) -> str | None:
    """Return None when a live cache owns its arena and tiers, else a probe-readable reason.

    Ownership is decided by a pair of observations, because neither alone is
    sufficient: an ownership token written by ``claim_arena``/``claim_tier`` sits
    beside the lock file, and the process the token names is alive with the
    ``/proc/<pid>/stat`` start time it recorded. The start-time match
    distinguishes a reused PID from the original holder, so a token whose owner
    exited, or whose PID was recycled, is no owner. ``live_owner`` never calls
    ``flock`` and never consults ``/proc/locks``: a probe that briefly took
    ``LOCK_EX`` could make the restarting cache it watches fail ``claim_arena``
    with ``EWOULDBLOCK`` and become a restart loop, and a ``/proc/locks`` device
    match is unreliable on the overlay/CSI filesystems this build runs on.

    The token is what makes a *fresh* start distinguishable from a dead one: the
    lock and its ``.owner`` token are created together at claim time, so a token
    whose owner is gone means the cache died, while no token at all means nothing
    claimed this resource. A missing or unreadable token is not an error; old
    images and hand-started caches have none. The reason names which observation
    failed: ``"no ownership token for the ..."`` when the token is absent, and
    ``"no live process holds the ..."`` when a token is present but its owner is
    gone or replaced. An empty ``shm_name`` (the non-engine transfer mode has no
    arena) skips the arena test. Nothing is appended to ``_ARENA_LOCKS`` or
    ``_TIER_LOCKS``, since the probe must not claim ownership.
    """
    if shm_name:
        token = _owner_token_path(shm_root / f"{shm_name}.lock")
        if not token.exists():
            return f"no ownership token for the cache arena {shm_name}"
        if not _owner_token_is_live(token):
            return f"no live process holds the cache arena {shm_name}"
    for tier in tier_dirs:
        token = _owner_token_path(Path(tier) / TIER_LOCK)
        if not token.exists():
            return f"no ownership token for the disk tier {tier}"
        if not _owner_token_is_live(token):
            return f"no live process holds the disk tier {tier}"
    return None


# Share of a filesystem's space (free plus what the tier already holds) that a
# disk tier may claim; the rest stays for the model, logs and other writers.
L2_DISK_SHARE = 0.9

# Decides whether the installed vLLM keeps QSA checkpoints atomic (vllm#864: the
# QSA backend refuses KV connectors while nothing enforces it, and DCP>1 would
# silently fall back to KV chunks without recurrent state). It is a GPU-free
# child, so ``CUDA_VISIBLE_DEVICES`` is cleared. Mirrors upstream
# ``runtime.cache._QSA_ATOMIC_TRANSFER_PROBE``.
_QSA_ATOMIC_TRANSFER_PROBE = (
    "from vllm.models.qwen4_exp.nvidia.b12x_qsa import Qwen4ExpQSABackend as B\n"
    "raise SystemExit(0 if B.supports_kv_connector() else 3)"
)


def disk_tier_paths(service: dict) -> list[Path]:
    """The filesystem disk-tier base paths named by ``service["argv"]``.

    Ported from upstream ``_disk_tier_paths``: each ``--l2-adapter`` JSON whose
    type is ``fs`` or ``fs_native`` contributes its ``base_path``.
    """
    paths = []
    argv = service["argv"]
    for index, arg in enumerate(argv[:-1]):
        if arg == "--l2-adapter":
            adapter = json.loads(argv[index + 1])
            if adapter.get("type") in {"fs", "fs_native"}:
                paths.append(Path(adapter["base_path"]))
    return paths


def fit_disk_tiers(service: dict) -> None:
    """Cap filesystem L2 tiers to what their disk can still hold.

    Ported verbatim from upstream ``fit_disk_tiers``. A tier larger than its disk
    fills the disk and fails every writer on it, including the model server, so
    an ``--l2-adapter`` whose ``max_capacity_gb`` exceeds what the disk can give
    is rewritten in place inside ``service["argv"]`` to
    ``floor((free + held) * L2_DISK_SHARE / 1GiB)``. Existing tier contents count
    as available, since the tier can evict them. Raises ``ConfigError`` when not
    even 1 GiB is left.
    """
    argv = service["argv"]
    for index, arg in enumerate(argv[:-1]):
        if arg != "--l2-adapter":
            continue
        adapter = json.loads(argv[index + 1])
        requested = adapter.get("max_capacity_gb")
        if adapter.get("type") not in {"fs", "fs_native"} or not requested:
            continue
        base = Path(adapter["base_path"])
        probe = base
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        stats = os.statvfs(probe)
        free = stats.f_bavail * stats.f_frsize
        held = _tree_bytes(base) if base.exists() else 0
        usable = math.floor((free + held) * L2_DISK_SHARE / 1024**3)
        if requested <= usable:
            continue
        if usable < 1:
            raise ConfigError(
                f"No space for the LMCache disk tier at {base}: "
                f"{free / 1024**3:.1f} GiB free. Free disk space or set "
                "LMCACHE_L2_ENABLED=0"
            )
        adapter["max_capacity_gb"] = usable
        argv[index + 1] = json.dumps(adapter, separators=(",", ":"))
        _announce(
            f"LMCache disk tier capped at {usable} GiB instead of {requested:g} GiB: "
            f"{probe} has {free / 1024**3:.0f} GiB free and the tier already holds "
            f"{held / 1024**3:.0f} GiB. Lower LMCACHE_L2_GB or free space to "
            "silence this"
        )


def preflight(service: dict, environment: dict[str, str]) -> None:
    """Refuse to start a cache whose geometry, ports or tiers are unusable.

    Ported from upstream ``runtime.supervisor.preflight`` with two deliberate
    differences required by the in-image launcher: it reads the resolved
    ``cache_service`` dict rather than a ``CacheService`` object, and it runs no
    process supervision, so nothing here waits on a child (the launcher decides
    process lifetime). Every pre-``exec`` gate upstream applied is preserved: the
    unresolved-checkpoint guard (which also catches an ``identity_required`` cache
    whose persistent identity was never resolved, since ``resolve_identity`` --
    owned by the launcher, not ported here -- is what replaces the
    ``UNRESOLVED-CHECKPOINT`` marker), the arena claim, the cuMem interposer and
    broker checks, the per-directory symlink and ownership checks, the port-bind
    check (a health response must not accidentally belong to another cache
    service), the stale-arena unlink, the ``/dev/shm`` free-space check, then the
    disk-tier claim/prune and the disk-tier cap.
    """
    argv = service["argv"]
    if any("UNRESOLVED-CHECKPOINT" in arg for arg in argv):
        raise ConfigError(
            "Persistent cache identity must be resolved before starting processes"
        )
    shm_name = service.get("shm_name") or ""
    shm_bytes = int(service.get("shm_bytes") or 0)
    shm = SHM_ROOT / shm_name if shm_bytes else None
    if shm is not None:
        claim_arena(shm)
    interposer = Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
    if (
        str(interposer) in environment.get("LD_PRELOAD", "").split(":")
        and not interposer.is_file()
    ):
        raise ConfigError(
            "LMCache-driven GLM transfer requires the packaged CUDA cuMem interposer"
        )
    broker = environment.get("LMCACHE_CUMEM_BROKER_DIR")
    for name in service.get("directories", []):
        path = Path(name)
        if path.is_symlink():
            raise ConfigError(f"Refusing a symlinked cache service directory: {path}")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if name == broker and (
            path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077
        ):
            raise ConfigError(
                "The cuMem broker directory must be owned by the serving user with mode 0700"
            )
    # A health response must not accidentally belong to another cache service.
    host = argv[argv.index("--host") + 1]
    http_host = argv[argv.index("--http-host") + 1]
    for flag, address in (
        ("--port", host),
        ("--http-port", http_host),
        ("--prometheus-port", http_host),
    ):
        port = int(argv[argv.index(flag) + 1])
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((address, port))
            except OSError as error:
                raise ConfigError(
                    f"Cache service address is unavailable: {address}:{port}"
                ) from error
    if shm is not None:
        # This container holds the arena lock and the cache ports are free, so an
        # existing arena was left by a cache that did not shut down cleanly.
        if shm.exists():
            shm.unlink()
            _announce(
                f"Removed the stale cache SHM arena {shm} left by a cache service "
                "that did not shut down cleanly"
            )
        stats = os.statvfs(SHM_ROOT)
        if stats.f_bavail * stats.f_frsize < shm_bytes:
            raise ConfigError(
                "Insufficient free /dev/shm for the complete engine-driven L1 arena"
            )
    namespace = service.get("namespace") or ""
    prune = bool(service.get("prune_stale_tiers"))
    for base in disk_tier_paths(service):
        if not namespace or base != Path(namespace):
            continue
        claim_tier(base)
        prune_stale_tiers(base.parent.parent, base, prune=prune)
    fit_disk_tiers(service)


def verify_installed_transfer(
    service: dict, *, run: Callable = subprocess.run
) -> None:
    """Refuse a cache whose vLLM build cannot keep QSA checkpoints atomic.

    Ported from upstream ``runtime.cache.verify_installed_transfer``. Upstream
    gates the probe on the plan (only ``profile == "qwen38-flash-next"`` with
    ``decode-context-parallel-size > 1``); those fields live in the plan, not in
    the ``cache_service`` dict, so this function enforces the precondition
    whenever it is called and the launcher is responsible for calling it only in
    that case. ``run`` is injectable so the test can stub the GPU-free child. An
    empty service is a no-op, matching upstream's early return with no cache
    configured.
    """
    if not service:
        return
    result = run(
        [sys.executable, "-c", _QSA_ATOMIC_TRANSFER_PROBE],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ConfigError(
            "Qwen external cache with DCP>1 requires a vLLM build with atomic QSA "
            "checkpoint transfer; use DCP1 or an image that includes it"
        )
