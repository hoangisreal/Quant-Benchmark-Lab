"""Dedicated Ollama daemon, local GGUF import and native NDJSON streaming."""

import subprocess
from pathlib import Path

import httpx

from ..config import CampaignConfig, Cell
from ..environment import engine_environment
from ..metrics import duration, measured
from ..prompts import eviction_text
from ..schema import LoadObservation, StreamEvent
from ..utils import Clock, file_hash
from .base import BackendError, Request
from .http_stream import DeadlineHTTP
from .process import OwnedProcess, offload_evidence


class OllamaBackend:
    def __init__(
        self,
        cell: Cell,
        config: CampaignConfig,
        work: Path,
        clock: Clock,
        client: httpx.Client | None = None,
        stream_transport=None,
    ):
        self.cell, self.config, self.clock, self.work = cell, config, clock, work
        self.base = f"http://127.0.0.1:{config.runtime.ollama_port}"
        self.client = client or httpx.Client(
            base_url=self.base,
            trust_env=False,
            timeout=httpx.Timeout(
                config.protocol.request_timeout,
                connect=config.protocol.connect_timeout,
                read=config.protocol.ttft_timeout,
            ),
        )
        self.http_stream = DeadlineHTTP(self.base, config.protocol, clock, stream_transport)
        self.server = OwnedProcess(work / "ollama.log", config.runtime.ollama_port)
        self.name = "qbl-" + (cell.model.sha256 or "unresolved")[:20]
        self.imported = False
        self.model_log_offset = 0
        self.env = {
            **engine_environment(),
            "OLLAMA_HOST": self.base,
            "OLLAMA_MODELS": str((work / "ollama-models").resolve()),
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_FLASH_ATTENTION": "0",
            "OLLAMA_KV_CACHE_TYPE": "f16",
            "OLLAMA_DEBUG": "1",
        }

    def _wait_daemon(self):
        start = self.clock.now()
        while (self.clock.now() - start) / 1e9 < self.config.protocol.activation_timeout:
            self.server.assert_alive()
            try:
                if self.client.get("/api/version").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            self.clock.sleep(0.1)
        raise TimeoutError("Ollama daemon readiness timed out")

    def _start_import(self):
        if self.server.process is None:
            self.server.start([self.config.runtime.ollama_binary, "serve"], self.env)
            self._wait_daemon()
        if not self.imported:
            modelfile = self.work / f"{self.name}.Modelfile"
            path = Path(self.cell.model.path).resolve()
            if '"' in str(path) or "\n" in str(path):
                raise ValueError("unsupported model path for Modelfile")
            modelfile.write_text(f'FROM "{path}"\n', encoding="utf-8")
            result = subprocess.run(
                [self.config.runtime.ollama_binary, "create", self.name, "-f", str(modelfile)],
                env=self.env,
                capture_output=True,
                text=True,
                timeout=self.config.protocol.activation_timeout,
            )
            (self.work / "import.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise BackendError("GGUF import failed; see import.log")
            blob = self.work / "ollama-models" / "blobs" / f"sha256-{self.cell.model.sha256}"
            if not blob.is_file() or file_hash(blob) != self.cell.model.sha256:
                raise BackendError("imported model blob does not match the GGUF SHA-256")
            self.imported = True

    def options(self, request: Request | None = None) -> dict:
        rt = self.config.runtime
        g = (request.generation if request else self.config.generation).model_dump()
        g.pop("performance_max_tokens")
        g.pop("quality_max_tokens")
        g.pop(
            "mirostat"
        )  # Not exposed by current Ollama; disabled/inapplicable verified in source audit.
        return {
            **g,
            "num_ctx": rt.context_size,
            "num_thread": rt.threads,
            "num_batch": rt.batch_size,
            "num_gpu": 999 if self.config.deployment == "gpu" else 0,
            "main_gpu": rt.gpu_index,
            "use_mmap": True,
            "draft_num_predict": 0,
            "num_predict": request.max_tokens if request else 0,
            "stop": request.stop if request else self.cell.model.stop,
        }

    def inspect(self) -> dict:
        response = self.client.get("/api/ps")
        response.raise_for_status()
        models = [
            m
            for m in response.json().get("models", [])
            if m.get("name", "").split(":")[0] == self.name
        ]
        log = self.server.log_text()[self.model_log_offset :]
        evidence = offload_evidence(log)
        ctx = models[0].get("context_length") if models else None
        # API context and layer logs are evidence; defaults/options are only requested settings.
        return {
            **evidence,
            "resident": bool(models),
            "ps": models,
            "context_verified": ctx == self.config.runtime.context_size,
            "actual_context": ctx,
            "runtime_verified": False,
            "runtime_reason": "KV/batch/flash/sampler semantics need pinned-source preflight audit",
            "requested_options": self.options(),
        }

    def load(self, load_id: str) -> LoadObservation:
        self._start_import()  # daemon startup/import are setup, excluded from model load.
        self.unload()
        self.model_log_offset = len(self.server.log_text())
        start = self.clock.now()
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.name,
                "prompt": "",
                "raw": True,
                "stream": False,
                "keep_alive": -1,
                "options": self.options(),
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("response") or payload.get("eval_count", 0) > 0:
            raise BackendError("preload unexpectedly generated content")
        ready = self.clock.now()
        evidence = self.inspect()
        if not evidence["resident"]:
            raise BackendError("preload completed but model is not resident")
        return LoadObservation(
            id=load_id,
            start_ns=start,
            ready_ns=ready,
            wall=measured((ready - start) / 1e6, "ms", "Ollama preload→complete"),
            native=duration(payload.get("load_duration"), 1e6, "ollama.load_duration/ns"),
            evidence={**evidence, "preload_payload": payload},
        )

    def count_tokens(self, text: str) -> None:
        return (
            None  # Ollama has no assumed tokenize endpoint; preflight supplies independent count.
        )

    def prepare_cache(self, request: Request) -> Request:
        return Request(eviction_text(request.prompt), 1, request.generation, request.stop)

    def stream(self, request: Request):
        payload = {
            "model": self.name,
            "prompt": request.prompt,
            "raw": True,
            "stream": True,
            "keep_alive": -1,
            "options": self.options(request),
            "truncate": False,
            "shift": False,
        }
        if self.cell.model.thinking == "disabled":
            payload["think"] = False
        for data in self.http_stream.frames("/api/generate", payload, "ndjson"):
            now = self.clock.now()
            if "error" in data:
                yield StreamEvent(receipt_ns=now, kind="error", payload=data)
                continue
            for key, kind in (("response", "text"), ("thinking", "thinking")):
                if data.get(key):
                    yield StreamEvent(receipt_ns=now, kind=kind, text=data[key], payload=data)
            if data.get("done"):
                yield StreamEvent(receipt_ns=now, kind="final", payload=data)
            elif not data.get("response") and not data.get("thinking"):
                yield StreamEvent(receipt_ns=now, kind="metadata", payload=data)

    def unload(self):
        if self.server.process is None or not self.imported:
            return
        response = self.client.post(
            "/api/generate", json={"model": self.name, "keep_alive": 0, "stream": False}
        )
        response.raise_for_status()
        start = self.clock.now()
        while (self.clock.now() - start) / 1e9 < self.config.protocol.unload_timeout:
            if not self.inspect()["resident"]:
                return
            self.clock.sleep(0.1)
        raise TimeoutError("Ollama model did not unload")

    def close(self):
        try:
            self.unload()
        finally:
            self.server.stop(self.config.protocol.unload_timeout)
            try:
                self.http_stream.close()
            finally:
                self.client.close()
