"""Capture local binary/source identities; this does not approve sampling semantics."""

import argparse
import json
import re
from pathlib import Path

from quant_benchmark_lab.environment import executable_identity
from quant_benchmark_lab.utils import atomic_json, command, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("vendor/llama.cpp"))
    p.add_argument("--ollama", default="ollama")
    p.add_argument("--output", type=Path, default=Path("locks/toolchain.json"))
    p.add_argument("--audit", type=Path, help="reviewed source-audit JSON, bound to binary SHA-256")
    args = p.parse_args()
    revision = command(["git", "-C", str(args.source), "rev-parse", "HEAD"])
    sha = revision.get("stdout", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        p.error("source must be a Git checkout at a full commit")
    dirty = command(["git", "-C", str(args.source), "status", "--porcelain"])
    if dirty.get("returncode") != 0 or dirty.get("stdout"):
        p.error("source checkout must be clean")
    binary = args.source / "build/bin/llama-server"
    llama, ollama = executable_identity(str(binary)), executable_identity(args.ollama)
    if not llama.get("sha256") or not ollama.get("sha256"):
        p.error("both engines must be installed before pinning the complete toolchain")
    quant = args.source / "build/bin/llama-quantize"
    if not quant.is_file():
        p.error("llama-quantize is missing")
    cache = args.source / "build/CMakeCache.txt"
    if not cache.is_file():
        p.error("CMake build provenance missing")
    audits = json.loads(args.audit.read_text()) if args.audit else {}
    value = {"state": "pinned", "llamacpp_commit": sha, "source": str(args.source.resolve()),
             "llamacpp": llama, "ollama": ollama, "quantizer_sha256": file_hash(quant),
             "cmake_cache": cache.read_text(), "compiler": command(["c++", "--version"]),
             "cuda_toolkit": command(["nvcc", "--version"]), "settings_audit": audits,
             "llama_help": command([str(binary), "--help"])}
    atomic_json(args.output, value)
    print(args.output)


if __name__ == "__main__":
    main()
