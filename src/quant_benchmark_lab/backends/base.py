from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ..config import Cell, Generation
from ..schema import LoadObservation, StreamEvent


@dataclass(frozen=True)
class Request:
    prompt: str
    max_tokens: int
    generation: Generation
    stop: list[str]


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    cell: Cell

    def inspect(self) -> dict: ...
    def load(self, load_id: str) -> LoadObservation: ...
    def stream(self, request: Request) -> Iterator[StreamEvent]: ...
    def count_tokens(self, text: str) -> int | None: ...
    def prepare_cache(self, request: Request) -> Request: ...
    def unload(self) -> None: ...
    def close(self) -> None: ...
