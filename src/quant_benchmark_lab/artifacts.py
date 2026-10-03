"""Validate pinned files and source lineage, independently of serving engines."""

import hashlib
import json
import re
from pathlib import Path

from .config import CampaignConfig, ModelSpec
from .utils import file_hash


def verify_model(model: ModelSpec) -> dict:
    if not model.sha256 or not model.revision or not model.license or not model.provenance_path:
        raise ValueError(f"{model.id}/{model.quant}: unresolved artifact/source/license/provenance")
    path = Path(model.path)
    if file_hash(path) != model.sha256:
        raise ValueError(f"artifact checksum mismatch: {path}")
    manifest = json.loads(Path(model.provenance_path).read_text())
    for key, expected in {"sha256": model.sha256, "revision": model.revision,
                          "source_repo": model.source_repo, "quant": model.quant}.items():
        if manifest.get(key) != expected:
            raise ValueError(f"provenance mismatch: {key}")
    if manifest.get("template_sha256") != hashlib.sha256(model.template.encode()).hexdigest():
        raise ValueError("template is not the one pinned in artifact provenance")
    if not manifest.get("tokenizer_sha256") or not manifest.get("llamacpp_commit"):
        raise ValueError("missing tokenizer/toolchain provenance")
    if model.quant != "F16" and not manifest.get("parent_f16_sha256"):
        raise ValueError("quant must descend directly from a pinned F16 artifact")
    return manifest


def verify_toolchain(config: CampaignConfig) -> dict:
    if config.synthetic:
        return {"synthetic": True}
    lock = json.loads(Path(config.runtime.toolchain_lock).read_text())
    if lock.get("state") != "pinned":
        raise ValueError("toolchain lock is unresolved; bootstrap/pin before running real engines")
    if not re.fullmatch(r"[0-9a-f]{40}", lock.get("llamacpp_commit", "")):
        raise ValueError("llama.cpp commit must be a full SHA")
    from .environment import executable_identity

    for engine, binary in (("ollama", config.runtime.ollama_binary),
                           ("llamacpp", config.runtime.llama_binary)):
        if not any(c.engine == engine for c in config.cells):
            continue
        actual = executable_identity(binary)
        if actual.get("sha256") != lock.get(engine, {}).get("sha256") or not actual.get("sha256"):
            raise ValueError(f"{engine} binary does not match the toolchain lock")
    return lock


def verify_artifacts(config: CampaignConfig) -> dict:
    if config.synthetic:
        return {"synthetic": True}
    lock = verify_toolchain(config)
    manifests = {c.id: verify_model(c.model) for c in config.cells}
    for manifest in manifests.values():
        if manifest["llamacpp_commit"] != lock["llamacpp_commit"]:
            raise ValueError("conversion and serving llama.cpp commits differ")
    if config.experiment == "quantization":
        parents = {v.get("parent_f16_sha256") or v["sha256"] for v in manifests.values()}
        if len(parents) != 1:
            raise ValueError("quant artifacts have different F16 ancestors")
    return manifests
