#!/usr/bin/env python3
"""Behavioural tests for the probe checks and both probe adapters."""

import asyncio
import http.server
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from image_tools.launcher import cache_runtime, probe_server, probes
from image_tools.launcher.probes import (
    HEALTHZ,
    LIVEZ,
    READYZ,
    STATUSZ,
    ProbeConfig,
    evaluate,
    render_verbose,
    status_json,
    write_config,
)

SERVER_ARGV = ["/opt/python/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "/models/m"]
LIVE_CHILD = [(23, "VLLM::Worker_TP", "S")]
ARENA = "lmcache_l1_pool_x"
ARENA_BYTES = 4096
SERVED_MODEL = "qwen38-flash-next"
INSTANCE_IDS = [7]


def proc_tree(root: Path, server_argv: list[str] | None, children: list[tuple[int, str, str]]) -> None:
    """Materialise just enough of ``/proc`` for the process checks.

    ``server_argv`` is the command line of the vLLM server process at PID 11 and
    ``children`` are ``(pid, comm, state)`` triples parented to it.
    """
    if server_argv is not None:
        server = root / "11"
        server.mkdir(parents=True)
        (server / "cmdline").write_bytes(("\0".join(server_argv) + "\0").encode())
        (server / "stat").write_text("11 (python) S 1 1 1\n")
        (server / "task" / "11").mkdir(parents=True)
        (server / "task" / "11" / "children").write_text(
            " ".join(str(pid) for pid, _comm, _state in children)
        )
    for pid, comm, state in children:
        process = root / str(pid)
        process.mkdir()
        (process / "comm").write_text(comm + "\n")
        (process / "stat").write_text(f"{pid} ({comm}) {state} 11 1 1\n")


def claim_cache(base: Path, health_url: str = "http://127.0.0.1:18000/healthcheck") -> dict:
    """Stand up the arena and disk tier a live cache owns, as the probe sees them.

    The locks are taken for real, through the same ``claim_arena`` and
    ``claim_tier`` the cache container runs, so the probe's ownership test runs
    against ``/proc/locks`` rather than against a stub.
    """
    shm_root = base / "shm"
    shm_root.mkdir(parents=True, exist_ok=True)
    (shm_root / ARENA).write_bytes(b"\0" * ARENA_BYTES)
    tiers = [base / "cache" / "lmcache" / "tier"]
    for tier in tiers:
        tier.mkdir(parents=True, exist_ok=True)
    cache_runtime.claim_arena(shm_root / ARENA)
    for tier in tiers:
        cache_runtime.claim_tier(tier)
    return {
        "health_url": health_url,
        "shm_name": ARENA,
        "shm_bytes": ARENA_BYTES,
        "shm_root": str(shm_root),
        "tier_dirs": [str(tier) for tier in tiers],
        "identity": {"registered_non_cuda_instance_ids": INSTANCE_IDS},
    }


def config(rank: int = 0, cache: dict | None = None, port: int = 8888) -> ProbeConfig:
    return ProbeConfig(
        rank=rank,
        leader="leader-0.svc" if rank else None,
        port=port,
        served_model_name=SERVED_MODEL,
        cache=cache,
        status={"image": "sha256:abc", "profile": "qwen38-flash-next"},
    )


def healthy_status(url: str, timeout: float) -> dict:
    """Answer like a ready API server and a cache instance nobody restarted."""
    if url.endswith("/v1/models"):
        return {"data": [{"id": SERVED_MODEL}]}
    return {"registered_non_cuda_instance_ids": INSTANCE_IDS}


def ok_status(url: str, timeout: float) -> int:
    return 200


class StubAPI:
    """The API server and cache endpoints a ready deployment answers on loopback."""

    def __enter__(self):
        served = {
            "/health": b"",
            "/v1/models": json.dumps({"data": [{"id": SERVED_MODEL}]}).encode(),
            "/healthcheck": json.dumps({"status": "healthy"}).encode(),
            "/status": json.dumps({"registered_non_cuda_instance_ids": INSTANCE_IDS}).encode(),
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = served.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_address[1]

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.root = self.base / "proc"
        self.root.mkdir(parents=True)
        self.addCleanup(self.directory.cleanup)
        proc_tree(self.root, SERVER_ARGV, LIVE_CHILD)
        self.cache = claim_cache(self.base)

    def live(self, **kwargs):
        return evaluate(LIVEZ, config(cache=self.cache), proc_root=self.root, **kwargs)

    def ready(self, rank: int = 0, **kwargs):
        return evaluate(READYZ, config(rank=rank, cache=self.cache), proc_root=self.root, **kwargs)

    def test_livez_never_calls_the_api_server_or_a_status_endpoint(self):
        """A long GPU step must not be able to time a liveness probe out."""

        def forbidden(*_args, **_kwargs):
            raise AssertionError("/livez must not call the API server or a status endpoint")

        ok, results = self.live(get_status=forbidden, get_json=healthy_status)
        self.assertTrue(ok, results)
        self.assertEqual(sorted(results), ["cache", "engine"])

    def test_livez_returns_while_the_engine_health_call_is_stuck(self):
        release = threading.Event()
        called = []

        def stuck(url, timeout):
            called.append(url)
            release.wait(5)
            return 200

        outcome = {}

        def run():
            outcome["result"] = evaluate(
                LIVEZ,
                config(cache=self.cache),
                proc_root=self.root,
                get_status=stuck,
                get_json=healthy_status,
            )

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(1.0)
        release.set()
        thread.join(5)
        self.assertFalse(thread.is_alive(), "liveness waited on an engine call")
        self.assertEqual(called, [], "liveness fetched the engine or the cache status")
        ok, results = outcome["result"]
        self.assertTrue(ok, results)

    def test_livez_fails_without_an_engine_child(self):
        for children, reason in (
            ([], "no direct child"),
            ([(23, "VLLM::Worker_TP", "Z")], "not runnable"),
        ):
            with self.subTest(children=children):
                empty = Path(tempfile.mkdtemp()) / "proc"
                empty.mkdir(parents=True)
                proc_tree(empty, SERVER_ARGV, children)
                ok, results = evaluate(
                    LIVEZ, config(cache=self.cache), proc_root=empty, get_json=healthy_status
                )
                self.assertFalse(ok)
                self.assertIn(reason, results["engine"])

    def test_livez_fails_when_the_cache_identity_changes(self):
        def restarted(url, timeout):
            return {"registered_non_cuda_instance_ids": [9]}

        ok, results = self.live(get_status=ok_status, get_json=restarted)
        self.assertFalse(ok)
        self.assertIn("changed since launch", results["cache"])

    def test_livez_fails_when_the_arena_is_gone(self):
        """A cache that fell back to pickle serves no shared-memory transfer."""
        (Path(self.cache["shm_root"]) / ARENA).unlink()
        ok, results = self.live(get_status=ok_status, get_json=healthy_status)
        self.assertFalse(ok)
        self.assertIn("does not exist", results["cache"])

    def test_livez_fails_when_no_process_owns_the_cache_locks(self):
        with mock.patch.object(
            cache_runtime, "live_owner", return_value="no live process holds the cache arena x"
        ):
            ok, results = self.live(get_status=ok_status, get_json=healthy_status)
        self.assertFalse(ok)
        self.assertIn("no live process holds the cache arena", results["cache"])

    def test_cache_check_is_skipped_without_a_cache(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("a deployment with no cache must not contact one")

        ok, results = evaluate(
            LIVEZ, config(), proc_root=self.root, get_status=forbidden, get_json=forbidden
        )
        self.assertTrue(ok)
        self.assertEqual(results, {"engine": None, "cache": None})

    def test_readyz_follows_the_leader_on_a_worker_rank(self):
        ok, results = self.ready(rank=1, get_status=ok_status, get_json=healthy_status)
        self.assertTrue(ok, results)
        self.assertNotIn("server", results, "a headless rank starts no API server")
        self.assertIn("leader", results)
        for failure in ("503", "connection refused"):
            def down(url, timeout, kind=failure):
                if kind == "503":
                    return 503
                raise OSError(kind)

            with self.subTest(failure=failure):
                ok, results = self.ready(rank=1, get_status=down, get_json=healthy_status)
                self.assertFalse(ok)
                self.assertIn("leader", results["leader"].lower())

    def test_readyz_fails_when_the_engine_is_dead(self):
        def server_down(url, timeout):
            return 503 if url.endswith("/health") else 200

        ok, results = self.ready(get_status=server_down, get_json=healthy_status)
        self.assertFalse(ok)
        self.assertIsNotNone(results["server"])

    def test_readyz_fails_when_the_served_model_is_missing(self):
        def wrong_model(url, timeout):
            if url.endswith("/v1/models"):
                return {"data": [{"id": "other"}]}
            return healthy_status(url, timeout)

        ok, results = self.ready(get_status=ok_status, get_json=wrong_model)
        self.assertFalse(ok)
        self.assertIn("is not served", results["model"])

    def test_healthz_is_the_union_of_the_other_two(self):
        ok, ready = self.ready(get_status=ok_status, get_json=healthy_status)
        self.assertTrue(ok, ready)
        ok, union = evaluate(
            HEALTHZ,
            config(cache=self.cache),
            proc_root=self.root,
            get_status=ok_status,
            get_json=healthy_status,
        )
        self.assertTrue(ok, union)
        self.assertEqual(sorted(union), sorted(ready))

        def unreachable(url, timeout):
            raise OSError("no answer")

        ok, union = evaluate(
            HEALTHZ,
            config(cache=self.cache),
            proc_root=self.root,
            get_status=unreachable,
            get_json=healthy_status,
        )
        self.assertFalse(ok)
        self.assertIsNotNone(union["server"])

    def test_single_check_and_exclude(self):
        def unreachable(url, timeout):
            raise OSError("no answer")

        ok, results = self.ready(get_status=unreachable, get_json=healthy_status, only="cache")
        self.assertTrue(ok)
        self.assertEqual(list(results), ["cache"])
        ok, results = self.ready(
            get_status=ok_status, get_json=healthy_status, exclude=frozenset({"cache"})
        )
        self.assertTrue(ok)
        self.assertNotIn("cache", results)

    def test_unknown_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate("nope", config(), proc_root=self.root)

    def test_verbose_rendering_matches_the_api_server_format(self):
        self.assertEqual(
            render_verbose({"engine": None, "cache": "arena missing"}),
            "[+] engine ok\n[-] cache failed: arena missing",
        )

    def test_statusz_carries_the_launcher_record(self):
        record = config(cache=self.cache)
        self.assertEqual(status_json(record)["image"], "sha256:abc")
        page = probes.status_page(record)
        self.assertIn("image=sha256:abc", page)
        self.assertIn("rank=0", page)
        self.assertEqual(probes.CHECKS[STATUSZ], {}, "a status page must never fail a probe")

    def test_record_round_trips_atomically(self):
        path = self.base / "launch.json"
        write_config(config(cache=self.cache), path)
        loaded = ProbeConfig.from_record(json.loads(path.read_text()))
        self.assertEqual(loaded.rank, 0)
        self.assertEqual(loaded.cache["shm_name"], ARENA)
        self.assertEqual(list(path.parent.glob(".launch.json.tmp-*")), [])
        with self.assertRaises(RuntimeError):
            probes.load_config(path.with_name("absent.json"))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.root = self.base / "proc"
        self.root.mkdir(parents=True)
        proc_tree(self.root, SERVER_ARGV, LIVE_CHILD)
        self.api = StubAPI()
        self.port = self.api.__enter__()
        self.addCleanup(self.api.__exit__)
        self.record = self.base / "launch.json"
        write_config(
            config(
                cache=claim_cache(self.base, f"http://127.0.0.1:{self.port}/healthcheck"),
                port=self.port,
            ),
            self.record,
        )
        self.addCleanup(self.directory.cleanup)

    def request(self, server: probe_server.ProbeServer, path: str) -> tuple[int, bytes]:
        url = f"http://127.0.0.1:{server.server_address[1]}{path}"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:
            return int(error.code), error.read()

    def start(self, record: Path, proc_root: Path | None = None) -> probe_server.ProbeServer:
        server = probe_server.ProbeServer(0, record=record, proc_root=proc_root or self.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        return server

    def test_http_server_serves_every_endpoint(self):
        server = self.start(self.record)
        for path, expected in (
            (f"/{LIVEZ}", 200),
            (f"/{READYZ}", 200),
            (f"/{HEALTHZ}", 200),
            (f"/{STATUSZ}", 200),
            (f"/{READYZ}?verbose", 200),
            (f"/{READYZ}/cache", 200),
            (f"/{READYZ}/nosuch", 404),
            ("/nope", 404),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.request(server, path)[0], expected)

    def test_http_server_statusz_formats(self):
        server = self.start(self.record)
        text_status, text_body = self.request(server, f"/{STATUSZ}")
        json_status, json_body = self.request(server, f"/{STATUSZ}?format=json")
        self.assertEqual((text_status, json_status), (200, 200))
        self.assertIn(b"image=sha256:abc", text_body)
        self.assertEqual(json.loads(json_body)["image"], "sha256:abc")

    def test_http_server_reports_a_broken_record_as_not_ready(self):
        broken = self.base / "broken.json"
        broken.write_text("{ not json")
        status, body = self.request(self.start(broken), f"/{LIVEZ}")
        self.assertEqual(status, 503)
        self.assertIn(b"launcher record", body)

    def test_http_server_reports_a_dead_engine_as_not_ready(self):
        empty = self.base / "empty-proc"
        empty.mkdir()
        status, body = self.request(self.start(self.record, empty), f"/{LIVEZ}?verbose")
        self.assertEqual(status, 503)
        self.assertIn(b"no vLLM server process", body)

    def test_middleware_answers_without_reaching_the_application(self):
        async def forbidden_app(scope, receive, send):
            raise AssertionError("a probe path must not reach vLLM")

        async def scenario():
            middleware = probe_server.ProbeMiddleware(forbidden_app, record=self.record, proc_root=self.root)
            captured = {}

            async def receive():
                return {"type": "http.request"}

            async def send(message):
                captured.update(message)

            await middleware({"type": "http", "method": "GET", "path": "/livez"}, receive, send)
            return captured

        captured = asyncio.run(scenario())
        self.assertEqual(captured["status"], 200)
        self.assertEqual(captured["body"], b"ok\n")

    def test_middleware_defers_unknown_paths(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["path"])

        async def scenario():
            middleware = probe_server.ProbeMiddleware(app, record=self.record, proc_root=self.root)
            await middleware({"type": "http", "method": "GET", "path": "/v1/models"}, None, None)

        asyncio.run(scenario())
        self.assertEqual(reached, ["/v1/models"])

    def test_parse_target_accepts_only_known_endpoints(self):
        self.assertEqual(
            probe_server.parse_target("/livez"),
            ("livez", None, {"exclude": [], "verbose": False, "json": False}),
        )
        endpoint, only, options = probe_server.parse_target("/readyz/cache?verbose&exclude=model")
        self.assertEqual((endpoint, only), ("readyz", "cache"))
        self.assertEqual(options["exclude"], ["model"])
        self.assertTrue(options["verbose"])
        self.assertTrue(probe_server.parse_target("/statusz?format=json")[2]["json"])
        for path in ("/health", "/readyz/a/b", ""):
            with self.subTest(path=path), self.assertRaises(probes.ConfigError):
                probe_server.parse_target(path)


if __name__ == "__main__":
    unittest.main()
