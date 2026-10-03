"""Synchronous facade over cancellable HTTP I/O with absolute request deadlines."""

import asyncio
import codecs
import json

import httpx


class DeadlineHTTP:
    """One reusable async client/event loop per backend; the runner stays sequential.

    Each I/O await gets the remaining absolute budget. Heartbeats and partial frames
    cannot renew that budget. Timeout cancellation closes the response before returning.
    """

    def __init__(self, base_url, protocol, clock, transport=None):
        self.base_url, self.protocol, self.clock = base_url, protocol, clock
        self.transport = transport
        self.loop = asyncio.Runner()
        self.client = None

    async def _frames(self, endpoint, payload, format):
        p = self.protocol
        start = self.clock.now()
        first_content = False

        def remaining():
            elapsed = (self.clock.now() - start) / 1e9
            limit = p.request_timeout if first_content else min(p.request_timeout, p.ttft_timeout)
            if elapsed >= limit:
                raise TimeoutError("stream request/first-content deadline exceeded")
            return limit - elapsed

        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, trust_env=False,
                transport=self.transport, timeout=httpx.Timeout(p.request_timeout,
                connect=min(p.connect_timeout, p.ttft_timeout, p.request_timeout)))
        request = self.client.build_request("POST", endpoint, json=payload)
        async with asyncio.timeout(remaining()):
            response = await self.client.send(request, stream=True)
        try:
            response.raise_for_status()
            chunks = response.aiter_bytes().__aiter__()
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            pending, data = "", []
            eof = False
            while not eof:
                try:
                    async with asyncio.timeout(remaining()):
                        chunk = await anext(chunks)
                    remaining()  # also checks clocks used by deterministic contract tests.
                    pending += decoder.decode(chunk)
                except StopAsyncIteration:
                    eof = True
                    pending += decoder.decode(b"", final=True)
                    if pending and format == "ndjson":
                        pending += "\n"
                while "\n" in pending:
                    remaining()
                    line, pending = pending.split("\n", 1)
                    line = line.rstrip("\r")
                    content = None
                    if format == "ndjson":
                        content = line if line.strip() else None
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif line == "" and data:
                        content = "\n".join(data)
                        data.clear()
                    if content is None or content == "[DONE]":
                        continue
                    value = json.loads(content)
                    if not isinstance(value, dict):
                        raise ValueError("stream frame must be a JSON object")
                    if value.get("content") or value.get("response") or value.get("thinking"):
                        first_content = True
                    yield value
            if format == "sse" and (data or pending.startswith("data:")):
                raise ValueError("truncated SSE frame: no event delimiter")
        finally:
            await response.aclose()

    def frames(self, endpoint, payload, format):
        stream = self._frames(endpoint, payload, format)
        try:
            while True:
                try:
                    yield self.loop.run(anext(stream))
                except StopAsyncIteration:
                    return
        finally:
            self.loop.run(stream.aclose())

    def close(self):
        if self.client is not None:
            self.loop.run(self.client.aclose())
            self.client = None
        self.loop.close()
