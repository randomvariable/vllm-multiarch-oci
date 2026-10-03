#!/usr/bin/env python3
"""Focused behavioral tests for the installed vllm-image command."""

import contextlib
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from image_tools import vllm_image


class Server:
    def __init__(self, handler) -> None:
        self.server = handler(("127.0.0.1", 0))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.server.server_address[1]

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class TcpServer:
    def __init__(self, address):
        self.socket = socket.create_server(address)
        self.socket.settimeout(0.1)
        self.server_address = self.socket.getsockname()
        self.running = True

    def serve_forever(self):
        while self.running:
            with contextlib.suppress(OSError, TimeoutError):
                connection, _ = self.socket.accept()
                connection.close()

    def shutdown(self):
        self.running = False
        self.socket.close()

    def server_close(self):
        self.socket.close()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):
        pass


def http_server(address):
    return http.server.ThreadingHTTPServer(address, Handler)


class CommandTests(unittest.TestCase):
    def invoke(self, *args, env=None):
        runfiles_root = Path(__file__).parents[1]
        return subprocess.run(
            [sys.executable, "-m", "image_tools.vllm_image", *args],
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(runfiles_root), **(env or {})},
        )

    def test_rendezvous_leader_and_worker(self):
        leader = self.invoke("rendezvous-wait", "--rank", "0", "--port", "1234")
        self.assertEqual(leader.returncode, 0, leader.stderr)
        with Server(TcpServer) as port:
            worker = self.invoke(
                "rendezvous-wait", "--rank", "1", "--leader", "localhost", "--port", str(port),
                "--dns-timeout", "1s", "--connect-timeout", "1s",
            )
        self.assertEqual(worker.returncode, 0, worker.stderr)

    def test_rendezvous_distinguishes_dns_and_port_timeouts(self):
        dns = self.invoke(
            "rendezvous-wait", "--rank", "1", "--leader", "invalid.", "--port", "1",
            "--dns-timeout", "20ms", "--dns-period", "5ms",
        )
        self.assertEqual(dns.returncode, vllm_image.EXIT_DNS)
        self.assertIn("DNS timeout", dns.stderr)
        with socket.create_server(("127.0.0.1", 0)) as reserved:
            unused = reserved.getsockname()[1]
        port = self.invoke(
            "rendezvous-wait", "--rank", "1", "--leader", "127.0.0.1", "--port", str(unused),
            "--dns-timeout", "1s", "--connect-timeout", "20ms", "--connect-period", "5ms",
        )
        self.assertEqual(port.returncode, vllm_image.EXIT_PORT)
        self.assertIn("port timeout", port.stderr)

    def test_health_accepts_a_ready_endpoint(self):
        with Server(http_server) as port:
            result = self.invoke("health", "--url", f"http://127.0.0.1:{port}/readyz")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready", result.stdout)

    def test_health_reports_failure_reason_and_status(self):
        class Unhealthy(Handler):
            def do_GET(self):
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"[-] engine failed: no child\n")

        with Server(lambda address: http.server.ThreadingHTTPServer(address, Unhealthy)) as port:
            failing = self.invoke("health", "--url", f"http://127.0.0.1:{port}/readyz")
            self.assertEqual(failing.returncode, vllm_image.EXIT_HTTP)
            self.assertIn("engine failed: no child", failing.stderr)
        unreachable = self.invoke("health", "--url", "http://127.0.0.1:1/readyz", "--timeout", "200ms")
        self.assertEqual(unreachable.returncode, vllm_image.EXIT_HTTP)
        self.assertIn("unreachable", unreachable.stderr)

    def snapshot(self, root: Path, revision: str, complete=True) -> Path:
        snapshot = root / "odd-cache-layout" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "part.safetensors"}}))
        if complete:
            (snapshot / "part.safetensors").write_bytes(b"weights")
        return snapshot

    def test_model_sync_rejects_partial_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.snapshot(root, "a" * 40, complete=False)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", "a" * 40,
                "--storage-root", str(root), "--publish", str(root / "published"),
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                with self.assertRaisesRegex(vllm_image.ConfigError, "missing or empty files: part.safetensors"):
                    vllm_image.model_sync(args)
            self.assertFalse((root / "published").exists())
            self.assertEqual(list((root / ".vllm-image/model-sync").glob("*.json")), [])

    def test_model_sync_rejects_missing_declared_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.snapshot(root, "d" * 40)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", "d" * 40,
                "--storage-root", str(root), "--publish", str(root / "published"),
                "--require", "tokenizer_config.json",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                with self.assertRaisesRegex(vllm_image.ConfigError, "missing or empty files: tokenizer_config.json"):
                    vllm_image.model_sync(args)
            self.assertFalse((root / "published").exists())
            self.assertEqual(list((root / ".vllm-image/model-sync").glob("*.json")), [])

    def test_model_sync_rejects_unsafe_required_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.snapshot(root, "e" * 40)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", "e" * 40,
                "--storage-root", str(root), "--publish", str(root / "published"),
                "--require", "../escape.json",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                with self.assertRaisesRegex(vllm_image.ConfigError, "invalid indexed shard path"):
                    vllm_image.model_sync(args)

    def test_model_sync_preserves_declared_cache_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self.snapshot(root, "f" * 40)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", "f" * 40,
                "--storage-root", str(root), "--publish", str(root / "published"),
            ])
            fake = mock.Mock(return_value=str(snapshot))
            scratch = str(root / "scratch-xet")
            with (
                mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}),
                mock.patch.dict(os.environ, {"HF_XET_CACHE": scratch}),
            ):
                for name in ("HF_HOME", "HF_HUB_CACHE"):
                    os.environ.pop(name, None)
                vllm_image.model_sync(args)
                self.assertEqual(os.environ["HF_XET_CACHE"], scratch)
                self.assertEqual(os.environ["HF_HOME"], str(root.resolve()))
                self.assertEqual(os.environ["HF_HUB_CACHE"], str(root.resolve() / "hub"))

    def test_model_sync_ignores_stale_marker_and_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "b" * 40
            snapshot = self.snapshot(root, revision)
            old = self.snapshot(root, "c" * 40)
            destination = root / "published"
            destination.symlink_to(old)
            state = root / ".vllm-image/model-sync"
            state.mkdir(parents=True)
            marker = state / f"{vllm_image._marker_key('org/model', revision)}.json"
            marker.write_text(json.dumps({"repo": "org/model", "revision": "c" * 40, "snapshot": str(old)}))
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", str(root), "--publish", str(destination), "--ignore", "examples/*",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                vllm_image.model_sync(args)
            self.assertEqual(destination.resolve(), snapshot.resolve())
            saved = json.loads(marker.read_text())
            self.assertEqual(saved["revision"], revision)
            self.assertEqual(fake.call_args.kwargs["ignore_patterns"], ["examples/*"])
            self.assertEqual(list(root.glob(".published.tmp-*")), [])

    def test_model_sync_accepts_only_hub_full_commit_oid(self):
        for revision in ("main", "v1", "abcdef0", "A" * 40, "a" * 39, "a" * 41):
            result = self.invoke(
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", "/tmp", "--publish", "/tmp/model",
            )
            self.assertEqual(result.returncode, vllm_image.EXIT_CONFIG, revision)


if __name__ == "__main__":
    unittest.main()
