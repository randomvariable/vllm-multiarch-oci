"""Adapters that serve the probe checks over HTTP.

Two forms, one set of checks:

``ProbeMiddleware``
    An ASGI middleware added with vLLM's ``--middleware`` option. It answers
    ``/livez``, ``/readyz``, ``/healthz`` and ``/statusz`` on the API port, so a
    plain ``docker run`` needs no extra process. It is pure ASGI: vLLM
    instantiates the class with the application only, and the endpoints answer
    before authentication middleware runs.

``serve``
    A standard-library HTTP server for ``vllm-image launch --role probe``, which
    runs as a sidecar in every pod rank. Headless ranks start no API server, so
    a pod needs one port that answers on both ranks.

Both forms offload the checks: the middleware runs them with
:func:`asyncio.to_thread` so a ``/proc`` scan or a loopback GET never occupies
vLLM's event loop, and the server handles each request in its own thread.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from image_tools.launcher import EXIT_CONFIG, ConfigError
from image_tools.launcher.probes import (
    DEFAULT_PROC_ROOT,
    DEFAULT_RECORD_PATH,
    DEFAULT_TIMEOUT_SECONDS,
    HEALTHZ,
    LIVEZ,
    READYZ,
    STATUSZ,
    ProbeConfig,
    evaluate,
    load_config,
    render_verbose,
    status_page,
    status_json,
)

ENDPOINTS = (LIVEZ, READYZ, HEALTHZ, STATUSZ)


def parse_target(path: str) -> tuple[str, str | None, dict[str, list[str]]]:
    """Split ``/<endpoint>/<check>?verbose&exclude=x`` into its parts."""
    split = urlsplit(path)
    parts = [part for part in split.path.split("/") if part]
    if not parts or parts[0] not in ENDPOINTS:
        raise ConfigError(f"unknown probe path {split.path!r}", EXIT_CONFIG)
    endpoint = parts[0]
    if len(parts) > 2:
        raise ConfigError(f"unknown probe path {split.path!r}", EXIT_CONFIG)
    query = parse_qs(split.query, keep_blank_values=True)
    exclude = frozenset(
        name for value in query.get("exclude", []) for name in value.split(",") if name
    )
    only = parts[1] if len(parts) == 2 else None
    formats = query.get("format", [])
    return endpoint, only, {
        "exclude": sorted(exclude),
        "verbose": "verbose" in query,
        "json": "json" in formats,
    }


def respond(
    config: ProbeConfig,
    endpoint: str,
    *,
    only: str | None = None,
    exclude: frozenset[str] = frozenset(),
    verbose: bool = False,
    json_format: bool = False,
    proc_root: Path = DEFAULT_PROC_ROOT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[int, str, str]:
    """Evaluate one endpoint and render its HTTP response."""
    if endpoint == STATUSZ:
        if json_format:
            return 200, "application/json", json.dumps(status_json(config), sort_keys=True) + "\n"
        return 200, "text/plain; charset=utf-8", status_page(config)
    ok, results = evaluate(
        endpoint, config, only=only, exclude=exclude, proc_root=proc_root, timeout=timeout
    )
    if not results:
        return 404, "text/plain; charset=utf-8", f"no probe check for {endpoint}\n"
    if verbose or not ok:
        body = render_verbose(results) + "\n"
    else:
        body = "ok\n"
    return (200 if ok else 503), "text/plain; charset=utf-8", body


class ProbeMiddleware:
    """Serve the probe endpoints inside vLLM's own API process."""

    def __init__(
        self,
        app: Any,
        record: Path = DEFAULT_RECORD_PATH,
        proc_root: Path = DEFAULT_PROC_ROOT,
    ) -> None:
        self.app = app
        self.record = Path(record)
        self.proc_root = Path(proc_root)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method") not in {"GET", "HEAD"}:
            await self.app(scope, receive, send)
            return
        try:
            endpoint, only, options = parse_target(scope.get("path", ""))
        except ConfigError:
            await self.app(scope, receive, send)
            return
        try:
            config = await asyncio.to_thread(load_config, self.record)
        except RuntimeError as error:
            status, content_type, body = 503, "text/plain; charset=utf-8", f"{error}\n"
        else:
            status, content_type, body = await asyncio.to_thread(
                respond,
                config,
                endpoint,
                only=only,
                exclude=frozenset(options["exclude"]),
                verbose=bool(options["verbose"]),
                json_format=bool(options["json"]),
                proc_root=self.proc_root,
            )
        payload = body.encode()
        if scope.get("method") == "HEAD":
            payload = b""
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type.encode()),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


class ProbeHandler(BaseHTTPRequestHandler):
    """Serve the probe endpoints from a sidecar container."""

    server: "ProbeServer"  # type: ignore[assignment]

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            endpoint, only, options = parse_target(self.path)
        except ConfigError as error:
            self._send(404, f"{error}\n")
            return
        try:
            config = load_config(self.server.record)
        except (RuntimeError, OSError, ValueError) as error:
            self._send(503, f"{error}\n")
            return
        status, _content_type, body = respond(
            config,
            endpoint,
            only=only,
            exclude=frozenset(options["exclude"]),
            verbose=bool(options["verbose"]),
            json_format=bool(options["json"]),
            proc_root=self.server.proc_root,
            timeout=self.server.timeout,
        )
        self._send(status, body)

    def do_HEAD(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.do_GET()

    def _send(self, status: int, body: str) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        # Probes fire every few seconds; klog-style noise would bury the log.
        return


class ProbeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        port: int,
        *,
        record: Path = DEFAULT_RECORD_PATH,
        proc_root: Path = DEFAULT_PROC_ROOT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        bind: str = "127.0.0.1",
    ) -> None:
        self.record = Path(record)
        self.proc_root = Path(proc_root)
        self.timeout = timeout
        super().__init__((bind, port), ProbeHandler)


def serve(
    port: int,
    *,
    record: Path = DEFAULT_RECORD_PATH,
    proc_root: Path = DEFAULT_PROC_ROOT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    bind: str = "127.0.0.1",
) -> int:
    """Run the probe server until it is stopped."""
    server = ProbeServer(port, record=record, proc_root=proc_root, timeout=timeout, bind=bind)
    print(f"probe: serving {', '.join('/' + name for name in ENDPOINTS)} on {bind}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def parser(roots: argparse.ArgumentParser) -> None:
    """Add the ``probe`` subcommand used by the sidecar container."""
    probe = roots.add_parser("probe", help="serve the probe endpoints")
    probe.add_argument("--port", type=int, required=True)
    probe.add_argument("--record", type=Path, default=DEFAULT_RECORD_PATH)
    probe.add_argument("--proc-root", type=Path, default=DEFAULT_PROC_ROOT)
    probe.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    probe.add_argument("--bind", default="127.0.0.1")
    probe.set_defaults(run=lambda args: serve(
        args.port,
        record=args.record,
        proc_root=args.proc_root,
        timeout=args.timeout,
        bind=args.bind,
    ))
