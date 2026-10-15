#!/usr/bin/env python3
"""Focused behavioral tests for the installed vllm-image command."""

import contextlib
import hashlib
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

    def csf_snapshot(self, root: Path, revision: str, family="qwen38_flash_next_nvfp4", complete=True) -> Path:
        """Lay out one NVFP4-CSF container: manifest + metadata/ + tensors/."""
        snapshot = root / "csf-cache-layout" / revision
        metadata = snapshot / "metadata"
        tensors = snapshot / "tensors"
        metadata.mkdir(parents=True)
        tensors.mkdir(parents=True)
        (snapshot / "manifest.json").write_text(json.dumps(
            {"schema": vllm_image._CSF_SCHEMA, "family": family,
             "codec": "byte-window4-fixed-stream-u24-exceptions/1"}
        ))
        (snapshot / "build-contract.json").write_text("{}")
        (snapshot / "verification.json").write_text("{}")
        original_qc = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        (metadata / "config.json").write_text(json.dumps(
            {"architectures": ["Qwen4ExpForCausalLM"], "quantization_config": original_qc}
        ))
        (metadata / "tokenizer_config.json").write_text("{}")
        (metadata / "chat_template.jinja").write_text("template")
        (metadata / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}
        ))
        (tensors / "model-00001.safetensors").write_bytes(b"weights")
        if complete:
            (tensors / "model-00002.safetensors").write_bytes(b"weights")
        return snapshot

    def test_model_sync_csf_publishes_nvfp4_csf_serving_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = self.csf_snapshot(root, revision)
            serve = root / "serve"
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", str(root), "--publish", str(serve),
                "--csf", "--csf-family", "qwen38_flash_next_nvfp4",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                # Drive the CLI wrapper the recipe's --model-sync path runs; it
                # is the consumer-visible surface. sync_model's shard count is an
                # internal return, so assert the observable publishing instead.
                vllm_image.model_sync(args)
            self.assertTrue((snapshot / "tensors/model-00002.safetensors").is_file())
            # The published path is a symlink to an immutable generated dir
            # under the state root, never the snapshot and never rebuilt in place.
            generated = (root / ".vllm-image/model-sync"
                         / f"{vllm_image._marker_key('org/model', revision)}.csf-serve")
            self.assertTrue(serve.is_symlink())
            self.assertEqual(serve.resolve(), generated.resolve())
            self.assertTrue(generated.is_dir() and not generated.is_symlink())
            # metadata/ is lifted to the serving root; tensors stay in the snapshot.
            self.assertTrue((serve / "model.safetensors.index.json").is_file())
            self.assertTrue((serve / "chat_template.jinja").is_file())
            self.assertFalse((generated / "tensors").exists())
            self.assertFalse((generated / "manifest.json").exists())
            qc = json.loads((serve / "config.json").read_text())["quantization_config"]
            self.assertEqual(qc["quant_method"], "nvfp4_csf")
            self.assertEqual(qc["format_version"], 1)
            self.assertEqual(qc["checkpoint_root"], str(snapshot.resolve()))
            self.assertEqual(qc["source_quantization_config"],
                             {"quant_method": "modelopt", "quant_algo": "NVFP4"})
            # A repeat sync of the same revision must not touch the generated
            # directory: a disk-PLE reader may hold these files open.
            sentinel = generated / ".sentinel"
            sentinel.write_text("do-not-rebuild")
            self.assertEqual(list(generated.parent.glob("*.csf-serve.tmp-*")), [])
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                vllm_image.model_sync(args)
            self.assertEqual(sentinel.read_text(), "do-not-rebuild")

    def test_model_sync_csf_identity_digests_manifest_not_serving_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = self.csf_snapshot(root, revision)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", str(root), "--publish", str(root / "serve"),
                "--csf", "--csf-family", "qwen38_flash_next_nvfp4",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                vllm_image.model_sync(args)
            expected = hashlib.sha256(
                vllm_image._CSF_IDENTITY_SALT + (snapshot / "manifest.json").read_bytes()
            ).hexdigest()
            self.assertEqual(
                vllm_image.csf_serving_identity(root, "org/model", revision),
                {"identity": expected, "revision": ""},
            )
            # The identity describes the validated container, not the generated
            # serving directory: mutating the staged tree must not move it.
            generated = (root / ".vllm-image/model-sync"
                         / f"{vllm_image._marker_key('org/model', revision)}.csf-serve")
            (generated / "stray-added-by-a-later-run").write_text("x")
            (generated / "config.json").write_text("{}")
            self.assertEqual(
                vllm_image.csf_serving_identity(root, "org/model", revision),
                {"identity": expected, "revision": ""},
            )

    def test_conventional_sync_has_no_csf_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = self.snapshot(root, revision)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", str(root), "--publish", str(root / "published"),
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                vllm_image.model_sync(args)
            self.assertIsNone(vllm_image.csf_serving_identity(root, "org/model", revision))

    def test_model_sync_csf_fails_closed_on_wrong_contract(self):
        for family, reason in (("glm53_nvfp4", "unexpected CSF contract"),
                               (None, "unexpected CSF contract")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                revision = "a" * 40
                snapshot = self.csf_snapshot(
                    root, revision, family=family or "some_other_family",
                )
                args = vllm_image.parser().parse_args([
                    "model-sync", "--repo", "org/model", "--revision", revision,
                    "--storage-root", str(root), "--publish", str(root / "serve"),
                    "--csf", "--csf-family", "qwen38_flash_next_nvfp4",
                ])
                fake = mock.Mock(return_value=str(snapshot))
                with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                    with self.assertRaisesRegex(vllm_image.ConfigError, reason):
                        vllm_image.model_sync(args)
                self.assertFalse((root / "serve").exists())
                self.assertEqual(list((root / ".vllm-image/model-sync").glob("*.json")), [])

    def test_model_sync_csf_rejects_missing_tensor_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            snapshot = self.csf_snapshot(root, revision, complete=False)
            args = vllm_image.parser().parse_args([
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", str(root), "--publish", str(root / "serve"),
                "--csf", "--csf-family", "qwen38_flash_next_nvfp4",
            ])
            fake = mock.Mock(return_value=str(snapshot))
            with mock.patch.dict(sys.modules, {"huggingface_hub": mock.Mock(snapshot_download=fake)}):
                with self.assertRaisesRegex(vllm_image.ConfigError, "incomplete CSF snapshot"):
                    vllm_image.model_sync(args)
            self.assertFalse((root / "serve").exists())

    def test_model_sync_accepts_only_hub_full_commit_oid(self):
        for revision in ("main", "v1", "abcdef0", "A" * 40, "a" * 39, "a" * 41):
            result = self.invoke(
                "model-sync", "--repo", "org/model", "--revision", revision,
                "--storage-root", "/tmp", "--publish", "/tmp/model",
            )
            self.assertEqual(result.returncode, vllm_image.EXIT_CONFIG, revision)

    def test_model_sync_csf_requires_a_family(self):
        result = self.invoke(
            "model-sync", "--repo", "org/model", "--revision", "a" * 40,
            "--storage-root", "/tmp", "--publish", "/tmp/model", "--csf",
        )
        self.assertEqual(result.returncode, vllm_image.EXIT_CONFIG)
        self.assertIn("CSF", result.stderr)


if __name__ == "__main__":
    unittest.main()
