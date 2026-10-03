"""Managed single-slot llama-server with native completion/SSE."""

import re
from pathlib import Path

import httpx

from ..config import CampaignConfig, Cell
from ..metrics import measured
from ..prompts import eviction_text
from ..schema import LoadObservation, StreamEvent
from ..utils import Clock
from .base import Request
from .http_stream import DeadlineHTTP
from .process import OwnedProcess, offload_evidence


class LlamaCppBackend:
    def __init__(
        self,
        cell: Cell,
        config: CampaignConfig,
        work: Path,
        clock: Clock,
        client: httpx.Client | None = None,
        stream_transport=None,
    ):
        self.cell, self.config, self.clock = cell, config, clock
        rt, p = config.runtime, config.protocol
        self.client = client or httpx.Client(
            base_url=f"http://127.0.0.1:{rt.llama_port}",
            trust_env=False,
            timeout=httpx.Timeout(
                p.request_timeout, connect=p.connect_timeout, read=p.ttft_timeout
            ),
        )
        self.http_stream = DeadlineHTTP(
            f"http://127.0.0.1:{rt.llama_port}", config.protocol, clock, stream_transport
        )
        self.server = OwnedProcess(work / "llamacpp.log", rt.llama_port)

    def launch_args(self) -> list[str]:
        rt = self.config.runtime
        return [
            rt.llama_binary,
            "--model",
            self.cell.model.path,
            "--host",
            "127.0.0.1",
            "--port",
            str(rt.llama_port),
            "--ctx-size",
            str(rt.context_size),
            "--threads",
            str(rt.threads),
            "--threads-batch",
            str(rt.threads),
            "--batch-size",
            str(rt.batch_size),
            "--ubatch-size",
            str(rt.microbatch_size),
            "--parallel",
            "1",
            "--gpu-layers",
            "all" if self.config.deployment == "gpu" else "0",
            "--main-gpu",
            str(rt.gpu_index),
            "--split-mode",
            "none",
            "--fit",
            "off",
            "--flash-attn",
            "off",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--load-mode",
            "mmap",
            "--lazy-mode",
            "off",
            "--no-context-shift",
            "--kv-offload" if self.config.deployment == "gpu" else "--no-kv-offload",
            "--spec-type",
            "none",
            "--cache-ram",
            "0",
            "--cache-prompt",
            "--perf",
            "--slots",
        ]

    def load(self, load_id: str) -> LoadObservation:
        self.unload()
        start = self.clock.now()
        self.server.start(self.launch_args())
        while (self.clock.now() - start) / 1e9 < self.config.protocol.activation_timeout:
            self.server.assert_alive()
            try:
                if self.client.get("/health").status_code == 200:
                    ready = self.clock.now()
                    return LoadObservation(
                        id=load_id,
                        start_ns=start,
                        ready_ns=ready,
                        wall=measured((ready - start) / 1e6, "ms", "llama-server spawn→health"),
                        native=measured(
                            None, "ms", "llamacpp", "no validated native load boundary"
                        ),
                        evidence=self.inspect(),
                    )
            except httpx.TransportError:
                pass
            self.clock.sleep(0.1)
        raise TimeoutError("llama-server readiness timeout")

    def inspect(self) -> dict:
        log = self.server.log_text()
        matches = re.findall(r"n_ctx_per_seq\s*=\s*(\d+)", log)
        actual = int(matches[-1]) if matches else None
        return {
            **offload_evidence(log),
            "resident": self.server.process is not None,
            "context_verified": actual == self.config.runtime.context_size,
            "actual_context": actual,
            "runtime_verified": False,
            "runtime_reason": "pinned flag/sampler implementation requires preflight source audit",
            "launch_args": self.launch_args(),
        }

    def count_tokens(self, text: str) -> int:
        response = self.client.post(
            "/tokenize", json={"content": text, "add_special": True, "parse_special": True}
        )
        response.raise_for_status()
        return len(response.json()["tokens"])

    def prepare_cache(self, request: Request) -> Request:
        return Request(eviction_text(request.prompt), 1, request.generation, request.stop)

    def stream(self, request: Request):
        g = request.generation
        payload = {
            "prompt": request.prompt,
            "stream": True,
            "n_predict": request.max_tokens,
            "seed": g.seed,
            "temperature": g.temperature,
            "top_k": g.top_k,
            "top_p": g.top_p,
            "min_p": g.min_p,
            "repeat_penalty": g.repeat_penalty,
            "repeat_last_n": g.repeat_last_n,
            "presence_penalty": g.presence_penalty,
            "frequency_penalty": g.frequency_penalty,
            "mirostat": g.mirostat,
            "dynatemp_range": 0.0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "samplers": ["top_k", "top_p", "min_p", "temperature"],
            "stop": request.stop,
            "cache_prompt": True,
            "id_slot": 0,
            "ignore_eos": False,
            "return_tokens": True,
        }
        for data in self.http_stream.frames("/completion", payload, "sse"):
            now = self.clock.now()
            if "error" in data:
                yield StreamEvent(receipt_ns=now, kind="error", payload=data)
            else:
                if data.get("content"):
                    yield StreamEvent(
                        receipt_ns=now, kind="text", text=data["content"], payload=data
                    )
                if data.get("stop"):
                    yield StreamEvent(receipt_ns=now, kind="final", payload=data)
                elif not data.get("content"):
                    yield StreamEvent(receipt_ns=now, kind="metadata", payload=data)

    def unload(self):
        self.server.stop(self.config.protocol.unload_timeout)

    def close(self):
        try:
            self.unload()
        finally:
            try:
                self.http_stream.close()
            finally:
                self.client.close()
