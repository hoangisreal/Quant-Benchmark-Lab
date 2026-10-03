"""Fixed text, IDs and hashes. Answer keys are never rendered."""

import hashlib
import json
import re
from pathlib import Path

from .config import ModelSpec
from .utils import digest


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(model: ModelSpec, text: str) -> str:
    return model.template.replace("{prompt}", text)


def read_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items or len({i["id"] for i in items}) != len(items):
        raise ValueError(f"empty dataset or duplicate item IDs: {path}")
    for item in items:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]):
            raise ValueError("item ID must be safe for artifact paths")
        if set(item) - {"id", "text", "category", "scorer", "length_bucket"}:
            raise ValueError("prompt item contains unknown fields (including answers)")
        if not isinstance(item.get("text"), str) or not item["text"]:
            raise ValueError("prompt must be nonempty text")
    return items


def workload_hash(performance: list[dict], quality: list[dict]) -> str:
    return digest({"performance": performance, "quality": quality})


def eviction_text(prompt: str, variant: int = 0) -> str:
    # Raw, deliberately outside the chat template. Token length is verified in preflight.
    prefix = ("Zebra ", "Quartz ", "Violet ")[variant % 3]
    return prefix + (" unrelated calibration input." * max(64, len(prompt) // 8))
