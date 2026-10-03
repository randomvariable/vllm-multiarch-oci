#!/usr/bin/env python3
"""Behavioral tests for the conversation-affinity replica proxy.

Two stub backends run on loopback as independent replica servers; the ported
``image_tools.launcher.proxy`` fronts them. The tests prove the routing rules
that the launcher's ``--role proxy`` container depends on: a conversation's
later turns stay on one replica, a new conversation prefers the least-loaded
replica, ``/health`` is 503 while any replica is down, and ``/metrics`` merges
the replicas with a ``replica`` label.
"""

import asyncio
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from image_tools.launcher import proxy


def _post(url, payload, headers=None):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read().decode()


class ProxyHarness:
    """Runs the two backends and the proxy in one background event loop."""

    def __init__(self, replicas=2):
        self.replicas = replicas
        self.states = []
        self.runners = []

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.port = asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(
            timeout=15
        )
        return self

    def __exit__(self, *_args):
        asyncio.run_coroutine_threadsafe(self._stop(), self.loop).result(timeout=15)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=15)
        self.loop.close()

    async def _start(self):
        from aiohttp import web

        upstreams = []
        for index in range(self.replicas):
            state = {"index": index, "healthy": True}
            self.states.append(state)
            app = web.Application()

            async def backend_post(request):
                if b"SLOW" in await request.read():
                    await asyncio.sleep(3)
                return web.json_response({"replica": request.app["state"]["index"]})

            async def backend_health(request):
                healthy = request.app["state"]["healthy"]
                return web.Response(status=200 if healthy else 503)

            async def backend_metrics(request):
                return web.Response(
                    text="# TYPE req_total counter\nreq_total 1\n",
                    content_type="text/plain",
                )

            app["state"] = state
            app.router.add_get("/health", backend_health)
            app.router.add_get("/metrics", backend_metrics)
            app.router.add_route("*", "/{tail:.*}", backend_post)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            self.runners.append(runner)
            upstreams.append(f"127.0.0.1:{port}")

        proxy_app = proxy.build_app(upstreams)
        proxy_runner = web.AppRunner(proxy_app)
        await proxy_runner.setup()
        proxy_site = web.TCPSite(proxy_runner, "127.0.0.1", 0)
        await proxy_site.start()
        self.runners.append(proxy_runner)
        return proxy_site._server.sockets[0].getsockname()[1]

    async def _stop(self):
        for runner in self.runners:
            await runner.cleanup()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def set_healthy(self, index, healthy):
        self.states[index]["healthy"] = healthy


class StickyRouterTests(unittest.TestCase):
    """Deterministic unit checks of the routing rule under load."""

    def test_fresh_conversation_prefers_least_in_flight(self):
        router = proxy.StickyRouter(2)
        router.in_flight[0] = 3
        affinity = proxy.request_affinity(
            "/v1/chat/completions",
            json.dumps(
                {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            ).encode(),
            {},
        )
        self.assertEqual(router.choose(affinity), 1)

    def test_affinity_header_pins_one_replica(self):
        router = proxy.StickyRouter(3)
        first = proxy.request_affinity("/anything", b"", {"X-LIL-Affinity": "sess"})
        replica = router.choose(first)
        again = proxy.request_affinity("/anything", b"", {"x-lil-affinity": "sess"})
        self.assertEqual(router.choose(again), replica)


class ReplicaProxyTests(unittest.TestCase):
    def test_conversation_stays_on_one_replica(self):
        with ProxyHarness(2) as server:
            chat = "/v1/chat/completions"
            first = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
            follow_up = {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "user", "content": "again"},
                ],
            }
            _, turn1 = _post(server.url(chat), first)
            _, turn2 = _post(server.url(chat), follow_up)
            self.assertEqual(turn1["replica"], turn2["replica"])

    def test_new_conversation_prefers_idle_replica(self):
        with ProxyHarness(2) as server:
            chat = "/v1/chat/completions"
            result = {}

            def warm():
                result["warm"] = _post(
                    server.url(chat),
                    {
                        "model": "m",
                        "messages": [{"role": "user", "content": "SLOW warm"}],
                    },
                    {"x-lil-affinity": "warm"},
                )

            holder = threading.Thread(target=warm)
            holder.start()
            time.sleep(0.5)  # let the slow request register as in-flight
            _, fresh = _post(
                server.url(chat),
                {"model": "m", "messages": [{"role": "user", "content": "brand new"}]},
            )
            holder.join(timeout=20)
            _, warm_result = result["warm"]
            self.assertNotEqual(fresh["replica"], warm_result["replica"])

    def test_health_ok_when_all_replicas_ready(self):
        with ProxyHarness(2) as server:
            status, _ = _get(server.url("/health"))
            self.assertEqual(status, 200)

    def test_health_503_while_a_replica_is_down(self):
        with ProxyHarness(2) as server:
            server.set_healthy(1, False)
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                _get(server.url("/health"))
            self.assertEqual(ctx.exception.code, 503)
            body = json.loads(ctx.exception.read())
            statuses = {r["replica"]: r["status"] for r in body["replicas"]}
            self.assertEqual(statuses[1], 503)
            self.assertEqual(statuses[0], 200)

    def test_metrics_carries_replica_label(self):
        with ProxyHarness(2) as server:
            status, text = _get(server.url("/metrics"))
            self.assertEqual(status, 200)
            self.assertIn('req_total{replica="0"} 1', text)
            self.assertIn('req_total{replica="1"} 1', text)

    def test_unaffinitised_request_goes_to_replica_zero(self):
        with ProxyHarness(2) as server:
            status, text = _get(server.url("/v1/models"))
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(text)["replica"], 0)


if __name__ == "__main__":
    unittest.main()
