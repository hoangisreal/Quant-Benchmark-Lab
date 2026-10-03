"""Deterministic synthetic backend: never presented as model/GPU evidence."""

from ..config import Cell
from ..metrics import measured
from ..schema import LoadObservation, StreamEvent
from ..utils import FakeClock
from .base import Request


class FakeBackend:
    def __init__(self, cell: Cell, clock: FakeClock):
        self.cell, self.clock, self.loaded = cell, clock, False

    def inspect(self) -> dict:
        return {"synthetic": True, "resident": self.loaded, "full_offload": True,
                "context_verified": True, "runtime_verified": True}

    def load(self, load_id: str) -> LoadObservation:
        start = self.clock.now()
        self.clock.sleep(0.03)
        self.loaded = True
        return LoadObservation(id=load_id, start_ns=start, ready_ns=self.clock.now(),
                               wall=measured(30.0, "ms", "synthetic clock"),
                               native=measured(25.0, "ms", "synthetic fixture"),
                               evidence=self.inspect())

    def count_tokens(self, text: str) -> int:
        return len(text.split()) + 1

    def prepare_cache(self, request: Request) -> Request:
        from ..prompts import eviction_text

        return Request(eviction_text(request.prompt), 1, request.generation, request.stop)

    def stream(self, request: Request):
        if not self.loaded:
            raise RuntimeError("fake model is not loaded")
        self.clock.sleep(0.01)
        yield StreamEvent(receipt_ns=self.clock.now(), kind="metadata", payload={"synthetic": True})
        self.clock.sleep(0.02)
        yield StreamEvent(receipt_ns=self.clock.now(), kind="text", text="SYNTHETIC ")
        self.clock.sleep(0.04)
        yield StreamEvent(receipt_ns=self.clock.now(), kind="text", text="OUTPUT")
        count = self.count_tokens(request.prompt)
        if self.cell.engine == "ollama":
            payload = {"done": True, "done_reason": "stop", "prompt_eval_count": count,
                       "prompt_eval_cached_count": 0, "prompt_eval_duration": 20_000_000,
                       "eval_count": 2, "eval_duration": 40_000_000, "load_duration": 0}
        else:
            payload = {"stop": True, "stop_type": "eos", "tokens_predicted": 2,
                       "timings": {"cache_n": 0, "prompt_n": count, "prompt_ms": 20.0,
                                   "predicted_n": 2, "predicted_ms": 40.0}}
        yield StreamEvent(receipt_ns=self.clock.now(), kind="final", payload=payload)

    def unload(self) -> None:
        self.loaded = False

    def close(self) -> None:
        self.unload()
