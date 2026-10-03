import asyncio
import json
import time
import unittest

import httpx

from quant_benchmark_lab.backends.http_stream import DeadlineHTTP
from quant_benchmark_lab.config import Protocol
from quant_benchmark_lab.utils import Clock


class Body(httpx.AsyncByteStream):
    def __init__(self, mode):
        self.mode, self.closed = mode, False

    async def __aiter__(self):
        if self.mode == "silent":
            await asyncio.sleep(10)
        elif self.mode == "first_then_stall":
            yield b'data: {"content":"ok"}\n\n'
            await asyncio.sleep(10)
        else:
            for _ in range(10000):
                await asyncio.sleep(0.005)
                yield b": ping\n\n" if self.mode == "heartbeat" else b"data: {}\n\n"

    async def aclose(self):
        self.closed = True


class DeadlineTests(unittest.TestCase):
    def test_silent_heartbeat_metadata_and_after_first_content_deadlines(self):
        for mode in ("silent", "heartbeat", "metadata", "first_then_stall"):
            with self.subTest(mode=mode):
                body = Body(mode)
                protocol = Protocol(ttft_timeout=0.05, request_timeout=0.12)
                client = DeadlineHTTP(
                    "http://test",
                    protocol,
                    Clock(),
                    httpx.MockTransport(lambda r, body=body: httpx.Response(200, stream=body)),
                )
                start = time.monotonic()
                seen = []
                try:
                    with self.assertRaises(TimeoutError):
                        for frame in client.frames("/completion", {}, "sse"):
                            seen.append(frame)
                    elapsed = time.monotonic() - start
                    self.assertLess(elapsed, 0.8)
                    if mode == "first_then_stall":
                        self.assertEqual(seen[0]["content"], "ok")
                        self.assertGreaterEqual(elapsed, 0.1)
                    self.assertTrue(body.closed)
                finally:
                    client.close()

    def test_header_wait_has_absolute_deadline(self):
        async def handler(request):
            await asyncio.sleep(10)
            return httpx.Response(200)

        client = DeadlineHTTP(
            "http://test", Protocol(ttft_timeout=0.03), Clock(), httpx.MockTransport(handler)
        )
        try:
            with self.assertRaises(TimeoutError):
                list(client.frames("/completion", {}, "sse"))
        finally:
            client.close()

    def test_fragmented_unicode_both_formats_and_reusable_client(self):
        class Fragments(httpx.AsyncByteStream):
            async def __aiter__(self):
                for byte in self.data:
                    yield bytes([byte])

        for format in ("ndjson", "sse"):
            body = Fragments()
            frame = json.dumps({"content": "Việt Nam"}, ensure_ascii=False).encode()
            body.data = frame + b"\n" if format == "ndjson" else b"data: " + frame + b"\n\n"
            client = DeadlineHTTP(
                "http://test",
                Protocol(),
                Clock(),
                httpx.MockTransport(lambda r, body=body: httpx.Response(200, stream=body)),
            )
            try:
                for _ in range(2):
                    self.assertEqual(
                        list(client.frames("/completion", {}, format)), [{"content": "Việt Nam"}]
                    )
            finally:
                client.close()

    def test_early_consumer_close_closes_http_response(self):
        body = Body("first_then_stall")
        client = DeadlineHTTP(
            "http://test",
            Protocol(),
            Clock(),
            httpx.MockTransport(lambda r: httpx.Response(200, stream=body)),
        )
        stream = client.frames("/completion", {}, "sse")
        self.assertEqual(next(stream)["content"], "ok")
        stream.close()
        self.assertTrue(body.closed)
        client.close()
