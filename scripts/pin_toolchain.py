"""Capture local binary/source identities; this does not approve sampling semantics."""

import argparse
import json
import re
from pathlib import Path

from quant_benchmark_lab.environment import executable_identity, payload_identity
from quant_benchmark_lab.utils import atomic_json, command, file_hash


def source_identity(source: Path) -> str:
    revision = command(["git", "-C", str(source), "rev-parse", "HEAD"])
    sha = revision.get("stdout", "").strip()
    if revision.get("returncode") != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("source must be a Git checkout at a full commit")
    dirty = command(["git", "-C", str(source), "status", "--porcelain"])
    if dirty.get("returncode") != 0 or dirty.get("stdout"):
        raise ValueError("source checkout must be clean")
    return sha


def ollama_source_identity(source: Path, release: str) -> dict:
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?", release):
        raise ValueError("Ollama release must be an explicit version tag")
    sha = source_identity(source)
    tag = command(["git", "-C", str(source), "rev-parse", f"refs/tags/{release}^{{commit}}"])
    if tag.get("returncode") != 0 or tag.get("stdout", "").strip() != sha:
        raise ValueError("Ollama source HEAD must match the requested release tag")
    bundled = source / "LLAMA_CPP_VERSION"
    return {
        "source_commit": sha,
        "release": release,
        "source_reference": f"https://github.com/ollama/ollama/tree/{sha}",
        "release_reference": f"https://github.com/ollama/ollama/releases/tag/{release}",
        "bundled_llamacpp_ref": bundled.read_text().strip() if bundled.is_file() else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("vendor/llama.cpp"))
    p.add_argument("--ollama", default="ollama")
    p.add_argument(
        "--ollama-source",
        type=Path,
        required=True,
        help="clean official source checkout at the installed release tag",
    )
    p.add_argument("--ollama-release", required=True, help="explicit installed version tag")
    p.add_argument(
        "--ollama-payload",
        type=Path,
        help="native lib/ollama directory; defaults to sibling of bin/ollama",
    )
    p.add_argument("--output", type=Path, default=Path("locks/toolchain.json"))
    p.add_argument("--audit", type=Path, help="reviewed source-audit JSON, bound to binary SHA-256")
    args = p.parse_args()
    try:
        sha = source_identity(args.source)
        ollama_source = ollama_source_identity(args.ollama_source, args.ollama_release)
    except ValueError as exc:
        p.error(str(exc))
    binary = args.source / "build/bin/llama-server"
    llama, ollama = executable_identity(str(binary)), executable_identity(args.ollama)
    if not llama.get("sha256") or not ollama.get("sha256"):
        p.error("both engines must be installed before pinning the complete toolchain")
    ollama_payload = args.ollama_payload or Path(ollama["path"]).parent.parent / "lib/ollama"
    if not (ollama_payload / "llama-server").is_file():
        p.error("native Ollama llama-server payload missing; supply --ollama-payload")
    try:
        llama["payload"] = payload_identity(binary.parent)
        ollama["payload"] = payload_identity(ollama_payload)
    except ValueError as exc:
        p.error(str(exc))
    quant = args.source / "build/bin/llama-quantize"
    if not quant.is_file():
        p.error("llama-quantize is missing")
    cache = args.source / "build/CMakeCache.txt"
    if not cache.is_file():
        p.error("CMake build provenance missing")
    audits = json.loads(args.audit.read_text()) if args.audit else {}
    value = {
        "state": "pinned",
        "llamacpp_commit": sha,
        "source": str(args.source.resolve()),
        "llamacpp": llama,
        "ollama": {**ollama, **ollama_source},
        "quantizer_sha256": file_hash(quant),
        "cmake_cache": cache.read_text(),
        "compiler": command(["c++", "--version"]),
        "cuda_toolkit": command(["nvcc", "--version"]),
        "settings_audit": audits,
        "llama_help": command([str(binary), "--help"]),
    }
    atomic_json(args.output, value)
    print(args.output)


if __name__ == "__main__":
    main()
