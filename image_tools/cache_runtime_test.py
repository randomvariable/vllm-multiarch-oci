#!/usr/bin/env python3
"""Focused behavioral tests for the cache-role runtime helpers.

These exercise the pure, unsupervised pieces the ``--role cache`` container runs
before it ``exec``s ``lmcache server``: the SHM arena claim, the disk-tier
shared lock, the cache readiness wait, and the engine-driven pool check. No
process is started and no HTTP server other than a loopback stub is used.
"""

import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from image_tools.launcher import ConfigError
from image_tools.launcher import cache_runtime


class _StubServer:
    """A loopback HTTP server whose handler is chosen per instance."""

    def __init__(self, handler):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.url()

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def url(self, path="/"):
        host, port = self.server.server_address
        return f"http://{host}:{port}{path}"


def _closed_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _free_ports(count):
    sockets = []
    ports = []
    for _ in range(count):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        sockets.append(sock)
    for sock in sockets:
        sock.close()
    return ports


def _cache_argv(host, ports):
    port, http_port, prom_port = ports
    return [
        "lmcache",
        "server",
        "--host", host,
        "--http-host", host,
        "--port", str(port),
        "--http-port", str(http_port),
        "--prometheus-port", str(prom_port),
    ]


class CacheRuntimeTests(unittest.TestCase):
    def tearDown(self):
        # Release every lock the test left open so the next test starts clean
        # and the temporary files are not pinned by a held descriptor.
        for table in (cache_runtime._ARENA_LOCKS, cache_runtime._TIER_LOCKS):
            for fd in list(table):
                os.close(fd)
            table.clear()

    # -- claim_arena -------------------------------------------------------

    def test_claim_arena_refuses_second_live_owner(self):
        with tempfile.TemporaryDirectory() as d:
            shm = Path(d) / "arena"
            cache_runtime.claim_arena(shm)
            self.assertTrue(shm.with_name("arena.lock").exists())
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.claim_arena(shm)
            self.assertEqual(ctx.exception.code, ConfigError("x").code)
            self.assertIn("owned by another running container", str(ctx.exception))

    def test_claim_arena_succeeds_after_holder_releases(self):
        with tempfile.TemporaryDirectory() as d:
            shm = Path(d) / "arena"
            cache_runtime.claim_arena(shm)
            holder = cache_runtime._ARENA_LOCKS.pop()
            os.close(holder)
            # The flock is gone with the descriptor, so a fresh claim succeeds.
            cache_runtime.claim_arena(shm)
            self.assertEqual(len(cache_runtime._ARENA_LOCKS), 1)

    def test_claim_arena_clears_close_on_exec(self):
        import fcntl

        with tempfile.TemporaryDirectory() as d:
            shm = Path(d) / "arena"
            cache_runtime.claim_arena(shm)
            fd = cache_runtime._ARENA_LOCKS[-1]
            self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC, 0)

    # -- claim_tier --------------------------------------------------------

    def test_claim_tier_allows_coexisting_shared_holders(self):
        with tempfile.TemporaryDirectory() as d:
            tier = Path(d) / "tier"
            cache_runtime.claim_tier(tier)
            # A shared lock is compatible with another shared holder.
            cache_runtime.claim_tier(tier)
            self.assertTrue((tier / cache_runtime.TIER_LOCK).exists())

    def test_claim_tier_refuses_when_deleter_holds_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            tier = Path(d) / "tier"
            tier.mkdir()
            lock = tier / cache_runtime.TIER_LOCK
            deleter = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            import fcntl

            fcntl.flock(deleter, fcntl.LOCK_EX)
            try:
                with self.assertRaises(ConfigError) as ctx:
                    cache_runtime.claim_tier(tier)
                self.assertIn("Another container is deleting", str(ctx.exception))
            finally:
                os.close(deleter)

    # -- validate_arena ----------------------------------------------------

    def test_validate_arena_accepts_segment_at_least_the_size(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "lmcache_l1_pool_x").write_bytes(b"\0" * 8192)
            cache_runtime.validate_arena("lmcache_l1_pool_x", 4096, shm_root=root)

    def test_validate_arena_refuses_smaller_segment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "arena").write_bytes(b"\0" * 1024)
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.validate_arena("arena", 4096, shm_root=root)
            self.assertIn("received 1024 bytes", str(ctx.exception))
            self.assertIn("no pickle fallback", str(ctx.exception))

    def test_validate_arena_refuses_missing_segment(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.validate_arena("absent", 4096, shm_root=Path(d))
            self.assertIn("does not exist", str(ctx.exception))
            self.assertIn("no pickle fallback", str(ctx.exception))

    # -- status_url --------------------------------------------------------

    def test_status_url_maps_healthcheck(self):
        self.assertEqual(
            cache_runtime.status_url("http://127.0.0.1:8080/healthcheck"),
            "http://127.0.0.1:8080/status",
        )

    def test_status_url_rejects_other_endpoints(self):
        for bad in (
            "http://127.0.0.1:8080/status",
            "http://127.0.0.1:8080/health",
            "http://127.0.0.1:8080",
        ):
            with self.assertRaises(ConfigError):
                cache_runtime.status_url(bad)

    # -- wait_ready --------------------------------------------------------

    def test_wait_ready_times_out(self):
        url = f"http://127.0.0.1:{_closed_port()}/healthcheck"
        with self.assertRaises(ConfigError) as ctx:
            cache_runtime.wait_ready(url, 0.5)
        self.assertIn("LMCache startup timed out", str(ctx.exception))

    def test_wait_ready_returns_on_success(self):
        class Ready(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, *_args):
                pass

        with _StubServer(Ready) as url:
            cache_runtime.wait_ready(url, 5.0)

    def test_wait_ready_ignores_http_proxy(self):
        # With proxies disabled a loopback health URL is reached even when an
        # unreachable HTTP_PROXY is exported; a default opener would hang on it
        # and hit the timeout instead.
        class Ready(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_args):
                pass

        with _StubServer(Ready) as url:
            with mock.patch.dict(
                os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}
            ):
                cache_runtime.wait_ready(url, 3.0)

    # -- read_json ---------------------------------------------------------

    def test_read_json_disables_proxies(self):
        class Json(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"engine_driven_shm_pool": {"shm_name": "a"}}')

            def log_message(self, *_args):
                pass

        with _StubServer(Json) as url:
            with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1"}):
                status = cache_runtime.read_json(url)
        self.assertEqual(status["engine_driven_shm_pool"]["shm_name"], "a")

    # -- prune_stale_tiers -------------------------------------------------

    def _make_layout(self, root, tier):
        path = root / "layout" / tier
        path.mkdir(parents=True)
        (path / "data").write_text("x")
        return path

    def test_prune_stale_tiers_reports_without_deleting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cache" / "model"
            keep = self._make_layout(root, "keep")
            other = self._make_layout(root, "other")
            cache_runtime.prune_stale_tiers(root, keep, prune=False)
            self.assertTrue(other.exists())

    def test_prune_stale_tiers_deletes_old_free_tier_when_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cache" / "model"
            keep = self._make_layout(root, "keep")
            other = self._make_layout(root, "other")
            old = cache_runtime.RECENT_TIER_WRITE_SECONDS + 120
            stamp = other.stat().st_mtime - old
            for entry in other.rglob("*"):
                os.utime(entry, (stamp, stamp))
            os.utime(other, (stamp, stamp))
            cache_runtime.prune_stale_tiers(root, keep, prune=True)
            self.assertFalse(other.exists())
            self.assertTrue(keep.exists())

    def test_prune_stale_tiers_keeps_recent_write_grace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cache" / "model"
            keep = self._make_layout(root, "keep")
            other = self._make_layout(root, "other")
            # A tier written moments ago survives even with prune enabled, as it
            # may belong to an image that predates the lock.
            cache_runtime.prune_stale_tiers(root, keep, prune=True)
            self.assertTrue(other.exists())

    def test_prune_stale_tiers_keeps_tier_in_use(self):
        import fcntl

        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cache" / "model"
            keep = self._make_layout(root, "keep")
            other = self._make_layout(root, "other")
            old = cache_runtime.RECENT_TIER_WRITE_SECONDS + 120
            stamp = other.stat().st_mtime - old
            os.utime(other, (stamp, stamp))
            holder = os.open(
                other / cache_runtime.TIER_LOCK, os.O_RDWR | os.O_CREAT, 0o600
            )
            fcntl.flock(holder, fcntl.LOCK_EX)
            try:
                cache_runtime.prune_stale_tiers(root, keep, prune=True)
                self.assertTrue(other.exists())
            finally:
                os.close(holder)

    # -- live_owner --------------------------------------------------------

    def test_live_owner_reports_reason_when_unheld(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            reason = cache_runtime.live_owner("pool", [root / "tier"], shm_root=root)
            self.assertEqual(reason, "no ownership token for the cache arena pool")

    def test_live_owner_returns_none_while_held(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shm = root / "pool"
            tier = root / "tier"
            # A distinct open file description holds the arena and tier locks,
            # exactly as the cache container's held descriptor would.
            cache_runtime.claim_arena(shm)
            cache_runtime.claim_tier(tier)
            self.assertIsNone(cache_runtime.live_owner("pool", [tier], shm_root=root))

    def test_live_owner_skips_arena_when_name_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tier = root / "tier"
            cache_runtime.claim_tier(tier)
            self.assertIsNone(cache_runtime.live_owner("", [tier], shm_root=root))

    def test_live_owner_reports_first_unlocked_tier(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shm = root / "pool"
            held = root / "held"
            free = root / "free"
            cache_runtime.claim_arena(shm)
            cache_runtime.claim_tier(held)
            reason = cache_runtime.live_owner("pool", [held, free], shm_root=root)
            self.assertEqual(reason, f"no ownership token for the disk tier {free}")

    def test_claim_writes_owner_token(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cache_runtime.claim_arena(root / "pool")
            token = root / "pool.lock.owner"
            self.assertTrue(token.exists())
            data = json.loads(token.read_text())
            self.assertEqual(data["holder"], "vllm-image")
            self.assertIsInstance(data["start"], int)
            self.assertEqual(
                data["start"], cache_runtime._process_start_time(os.getpid())
            )
            self.assertEqual(data["pid"], os.getpid())
            cache_runtime.claim_tier(root / "tier")
            self.assertTrue(
                (root / "tier" / (cache_runtime.TIER_LOCK + ".owner")).exists()
            )

    def test_live_owner_reports_lock_free_when_token_present(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "pool.lock").write_text("")
            (root / "pool.lock.owner").write_text(
                json.dumps({"pid": 1, "holder": "vllm-image"})
            )
            self.assertEqual(
                cache_runtime.live_owner("pool", [], shm_root=root),
                "no live process holds the cache arena pool",
            )

    def test_live_owner_reports_tier_observations(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dead = root / "dead"
            dead.mkdir()
            (dead / cache_runtime.TIER_LOCK).write_text("")
            (dead / (cache_runtime.TIER_LOCK + ".owner")).write_text(
                json.dumps({"pid": 1, "holder": "vllm-image"})
            )
            self.assertEqual(
                cache_runtime.live_owner("", [dead]),
                f"no live process holds the disk tier {dead}",
            )
            fresh = root / "fresh"
            fresh.mkdir()
            self.assertEqual(
                cache_runtime.live_owner("", [fresh]),
                f"no ownership token for the disk tier {fresh}",
            )

    def test_live_owner_never_leaves_a_held_lock(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            shm = root / "pool"
            # Two probe calls on an unclaimed arena must not acquire the lock, so
            # a later claim still succeeds; this is the restart-loop hazard.
            self.assertIsNotNone(cache_runtime.live_owner("pool", [], shm_root=root))
            self.assertIsNotNone(cache_runtime.live_owner("pool", [], shm_root=root))
            cache_runtime.claim_arena(shm)
            self.assertIsNone(cache_runtime.live_owner("pool", [], shm_root=root))




class LiveOwnerProcessTests(unittest.TestCase):
    """Ownership is proven by the token's process, independent of the filesystem.

    The overlay/CSI ``st_dev`` versus ``/proc/locks`` device mismatch that made
    ``live_owner`` report a live cache as dead must not matter here: every case
    drives ``live_owner`` with a token naming a real process and depends only on
    ``/proc/<pid>/stat``. These run correctly on an overlay ``TMPDIR``.
    """

    def _sleeping_child(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        return proc

    def _arena_token(self, root, pid, start):
        (root / "pool.lock").write_text("")
        (root / "pool.lock.owner").write_text(
            json.dumps({"pid": pid, "start": start, "holder": "vllm-image"})
        )

    def test_none_while_named_process_alive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc = self._sleeping_child()
            try:
                start = cache_runtime._process_start_time(proc.pid)
                self.assertIsNotNone(start)
                self._arena_token(root, proc.pid, start)
                self.assertIsNone(
                    cache_runtime.live_owner("pool", [], shm_root=root)
                )
            finally:
                proc.kill()
                proc.wait()

    def test_reports_gone_after_process_exits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc = self._sleeping_child()
            start = cache_runtime._process_start_time(proc.pid)
            self._arena_token(root, proc.pid, start)
            proc.kill()
            proc.wait()
            self.assertEqual(
                cache_runtime.live_owner("pool", [], shm_root=root),
                "no live process holds the cache arena pool",
            )

    def test_rejects_reused_pid_token(self):
        # PID 1 is alive but its real start time is not 0, so a forged token
        # that reuses it with a stale start time reads as no owner.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._arena_token(root, 1, 0)
            self.assertEqual(
                cache_runtime.live_owner("pool", [], shm_root=root),
                "no live process holds the cache arena pool",
            )

    def test_rejects_dead_pid_token(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_pid = proc.pid
            proc.wait()
            self._arena_token(root, dead_pid, 12345)
            self.assertEqual(
                cache_runtime.live_owner("pool", [], shm_root=root),
                "no live process holds the cache arena pool",
            )

    def test_reports_missing_token(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(
                cache_runtime.live_owner("pool", [], shm_root=Path(d)),
                "no ownership token for the cache arena pool",
            )

    def test_tier_uses_process_liveness(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tier = root / "tier"
            tier.mkdir()
            proc = self._sleeping_child()
            try:
                start = cache_runtime._process_start_time(proc.pid)
                (tier / cache_runtime.TIER_LOCK).write_text("")
                (tier / (cache_runtime.TIER_LOCK + ".owner")).write_text(
                    json.dumps(
                        {"pid": proc.pid, "start": start, "holder": "vllm-image"}
                    )
                )
                self.assertIsNone(
                    cache_runtime.live_owner("", [tier], shm_root=root)
                )
            finally:
                proc.kill()
                proc.wait()
            self.assertEqual(
                cache_runtime.live_owner("", [tier], shm_root=root),
                f"no live process holds the disk tier {tier}",
            )

    def test_never_appends_to_lock_tables(self):
        arenas_before = len(cache_runtime._ARENA_LOCKS)
        tiers_before = len(cache_runtime._TIER_LOCKS)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # An unheld arena and tier produce reasons, not claimed descriptors.
            cache_runtime.live_owner("pool", [root / "tier"], shm_root=root)
        self.assertEqual(len(cache_runtime._ARENA_LOCKS), arenas_before)
        self.assertEqual(len(cache_runtime._TIER_LOCKS), tiers_before)



class DiskTierPathsTests(unittest.TestCase):
    def test_reads_fs_adapters_only(self):
        service = {
            "argv": [
                "x",
                "--l2-adapter", json.dumps({"type": "fs", "base_path": "/a"}),
                "--l2-adapter", json.dumps({"type": "fs_native", "base_path": "/b"}),
                "--l2-adapter", json.dumps({"type": "redis", "base_path": "/c"}),
                "tail",
            ]
        }
        self.assertEqual(
            cache_runtime.disk_tier_paths(service), [Path("/a"), Path("/b")]
        )


class FitDiskTiersTests(unittest.TestCase):
    def test_leaves_roomy_tier_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "l2"
            base.mkdir()
            adapter = {"type": "fs", "base_path": str(base), "max_capacity_gb": 1}
            service = {"argv": ["x", "--l2-adapter", json.dumps(adapter)]}
            before = list(service["argv"])
            cache_runtime.fit_disk_tiers(service)
            self.assertEqual(service["argv"], before)

    def test_caps_oversized_tier(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "l2"
            base.mkdir()
            adapter = {"type": "fs", "base_path": str(base), "max_capacity_gb": 100}
            service = {"argv": ["x", "--l2-adapter", json.dumps(adapter)]}
            # 1_000_000 blocks of 4096 bytes free, tier holds nothing.
            fake = types.SimpleNamespace(f_bavail=1000000, f_frsize=4096)
            with mock.patch.object(
                cache_runtime.os, "statvfs", return_value=fake
            ), mock.patch.object(cache_runtime, "_tree_bytes", return_value=0):
                cache_runtime.fit_disk_tiers(service)
            rewritten = json.loads(service["argv"][2])
            self.assertEqual(rewritten["max_capacity_gb"], 3)
            self.assertEqual(rewritten["type"], "fs")
            self.assertEqual(rewritten["base_path"], str(base))

    def test_refuses_full_disk(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "l2"
            base.mkdir()
            adapter = {"type": "fs", "base_path": str(base), "max_capacity_gb": 10}
            service = {"argv": ["x", "--l2-adapter", json.dumps(adapter)]}
            fake = types.SimpleNamespace(f_bavail=0, f_frsize=4096)
            with mock.patch.object(
                cache_runtime.os, "statvfs", return_value=fake
            ), mock.patch.object(cache_runtime, "_tree_bytes", return_value=0):
                with self.assertRaises(ConfigError) as ctx:
                    cache_runtime.fit_disk_tiers(service)
            self.assertIn("No space for the LMCache disk tier", str(ctx.exception))
            self.assertIn("LMCACHE_L2_ENABLED=0", str(ctx.exception))

    def test_ignores_non_fs_adapter(self):
        adapter = {"type": "redis", "base_path": "/no", "max_capacity_gb": 10}
        service = {"argv": ["x", "--l2-adapter", json.dumps(adapter)]}
        before = list(service["argv"])
        cache_runtime.fit_disk_tiers(service)
        self.assertEqual(service["argv"], before)


class PreflightTests(unittest.TestCase):
    def tearDown(self):
        for table in (cache_runtime._ARENA_LOCKS, cache_runtime._TIER_LOCKS):
            for fd in list(table):
                os.close(fd)
            table.clear()

    def _service(self, argv, **over):
        base = {
            "argv": argv,
            "shm_name": "",
            "shm_bytes": 0,
            "directories": [],
            "namespace": "",
            "prune_stale_tiers": False,
        }
        base.update(over)
        return base

    def test_accepts_and_creates_directories(self):
        with tempfile.TemporaryDirectory() as d:
            argv = _cache_argv("127.0.0.1", _free_ports(3))
            dirs = [str(Path(d) / "a"), str(Path(d) / "b")]
            cache_runtime.preflight(self._service(argv, directories=dirs), {})
            self.assertTrue(Path(dirs[0]).is_dir())
            self.assertTrue(Path(dirs[1]).is_dir())

    def test_rejects_unresolved_checkpoint(self):
        argv = _cache_argv("127.0.0.1", _free_ports(3)) + ["--x", "UNRESOLVED-CHECKPOINT"]
        with self.assertRaises(ConfigError) as ctx:
            cache_runtime.preflight(self._service(argv), {})
        self.assertIn("Persistent cache identity", str(ctx.exception))

    def test_rejects_symlinked_directory(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "t"
            target.mkdir()
            link = Path(d) / "link"
            link.symlink_to(target)
            argv = _cache_argv("127.0.0.1", _free_ports(3))
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.preflight(
                    self._service(argv, directories=[str(link)]), {}
                )
            self.assertIn("Refusing a symlinked", str(ctx.exception))

    def test_rejects_occupied_port(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            port = busy.getsockname()[1]
            argv = _cache_argv("127.0.0.1", [port, *_free_ports(2)])
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.preflight(self._service(argv), {})
            self.assertIn("Cache service address is unavailable", str(ctx.exception))

    def test_rejects_missing_cumem_interposer(self):
        argv = _cache_argv("127.0.0.1", _free_ports(3))
        env = {"LD_PRELOAD": "/opt/lmcache/lib/liblmcache_cumem_shareable.so"}
        with mock.patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(ConfigError) as ctx:
                cache_runtime.preflight(self._service(argv), env)
        self.assertIn("cuMem interposer", str(ctx.exception))

    def test_unlinks_stale_arena(self):
        shm_name = f"lil-{os.getpid()}-stale"
        arena = cache_runtime.SHM_ROOT / shm_name
        arena.write_text("stale")
        try:
            argv = _cache_argv("127.0.0.1", _free_ports(3))
            service = self._service(argv, shm_name=shm_name, shm_bytes=1024)
            fake = types.SimpleNamespace(f_bavail=1000000, f_frsize=4096)
            with mock.patch.object(cache_runtime.os, "statvfs", return_value=fake):
                cache_runtime.preflight(service, {})
            self.assertFalse(arena.exists())
        finally:
            arena.unlink(missing_ok=True)
            (cache_runtime.SHM_ROOT / f"{shm_name}.lock").unlink(missing_ok=True)
            (cache_runtime.SHM_ROOT / f"{shm_name}.lock.owner").unlink(missing_ok=True)

    def test_rejects_insufficient_shm(self):
        shm_name = f"lil-{os.getpid()}-small"
        try:
            argv = _cache_argv("127.0.0.1", _free_ports(3))
            service = self._service(argv, shm_name=shm_name, shm_bytes=10000000000)
            fake = types.SimpleNamespace(f_bavail=1, f_frsize=4096)
            with mock.patch.object(cache_runtime.os, "statvfs", return_value=fake):
                with self.assertRaises(ConfigError) as ctx:
                    cache_runtime.preflight(service, {})
            self.assertIn("Insufficient free /dev/shm", str(ctx.exception))
        finally:
            (cache_runtime.SHM_ROOT / f"{shm_name}.lock").unlink(missing_ok=True)
            (cache_runtime.SHM_ROOT / f"{shm_name}.lock.owner").unlink(missing_ok=True)


class VerifyInstalledTransferTests(unittest.TestCase):
    def test_ok_clears_gpu_and_invokes_probe(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append((cmd, kw))
            return types.SimpleNamespace(returncode=0)

        cache_runtime.verify_installed_transfer(
            {"argv": ["lmcache", "server"]}, run=fake_run
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("-c", calls[0][0])

    def test_raises_when_probe_fails(self):
        def fake_run(cmd, **kw):
            return types.SimpleNamespace(returncode=3)

        with self.assertRaises(ConfigError) as ctx:
            cache_runtime.verify_installed_transfer({"argv": ["x"]}, run=fake_run)
        self.assertIn("atomic QSA", str(ctx.exception))

    def test_no_service_does_not_run(self):
        def boom(*a, **k):
            raise AssertionError("must not run")

        cache_runtime.verify_installed_transfer({}, run=boom)
        cache_runtime.verify_installed_transfer(None, run=boom)
if __name__ == "__main__":
    unittest.main()
