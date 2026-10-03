"""Incremental UTF-8, NDJSON and SSE parsers; transport chunks are not tokens."""

import codecs
import json
from collections.abc import Iterable, Iterator


def lines(chunks: Iterable[bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    for chunk in chunks:
        pending += decoder.decode(chunk)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            yield line.rstrip("\r")
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending.rstrip("\r")


def ndjson(chunks: Iterable[bytes]) -> Iterator[dict]:
    for line in lines(chunks):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("NDJSON frame must be an object")
            yield value


def sse(chunks: Iterable[bytes]) -> Iterator[dict]:
    data = []
    for line in lines(chunks):
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line == "" and data:
            content = "\n".join(data)
            data.clear()
            if content != "[DONE]":
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ValueError("SSE data must be an object")
                yield value
    if data:
        raise ValueError("truncated SSE frame: no event delimiter")
