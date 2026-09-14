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

    def test_health_leader_worker_and_dead_worker(self):
        with Server(http_server) as port:
            url = f"http://127.0.0.1:{port}/v1/models"
            leader = self.invoke("health", "--phase", "startup", "--rank", "0", "--local-url", url)
            worker = self.invoke(
                "health", "--phase", "readiness", "--rank", "3", "--leader-url", url,
                "--engine-pid", str(os.getpid()),
            )
            dead = self.invoke(
                "health", "--phase", "readiness", "--rank", "1", "--leader-url", url,
                "--engine-pid", "2147483647",
            )
        self.assertEqual(leader.returncode, 0, leader.stderr)
        self.assertEqual(worker.returncode, 0, worker.stderr)
        self.assertEqual(dead.returncode, vllm_image.EXIT_ENGINE)
        self.assertIn("is not alive", dead.stderr)

    def proc_tree(self, root: Path, children: list[tuple[int, str, str]]) -> None:
        parent = root / "1/task/1"
        parent.mkdir(parents=True)
        (parent / "children").write_text(" ".join(str(pid) for pid, _comm, _state in children))
        for pid, comm, state in children:
            process = root / str(pid)
            process.mkdir()
            (process / "comm").write_text(comm + "\n")
            (process / "stat").write_text(f"{pid} ({comm}) {state} 1 1 1\n")

    def test_engine_child_exact_live_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.proc_tree(root, [(22, "python", "S"), (23, "VLLM::Worker_TP", "S")])
            with mock.patch("image_tools.vllm_image.os.kill"):
                self.assertEqual(vllm_image._engine_child(1, root), 23)

    def test_engine_child_missing_dead_zombie_and_ambiguous(self):
        cases = [
            ([], "no live direct child"),
            ([(22, "VLLM::Worker_TP", "Z")], "no live direct child"),
            ([(22, "VLLM::Worker_TP", "S"), (23, "VLLM::EngineCor", "S")], "ambiguous"),
        ]
        for children, error in cases:
            with self.subTest(children=children), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.proc_tree(root, children)
                with mock.patch("image_tools.vllm_image.os.kill"), self.assertRaisesRegex(vllm_image.CommandError, error):
                    vllm_image._engine_child(1, root)

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
                with self.assertRaisesRegex(vllm_image.CommandError, "missing or empty shards"):
                    vllm_image.model_sync(args)
            self.assertFalse((root / "published").exists())
            self.assertEqual(list((root / ".vllm-image/model-sync").glob("*.json")), [])

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
