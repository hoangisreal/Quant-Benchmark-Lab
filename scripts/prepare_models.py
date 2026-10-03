"""Convert pinned source weights and quantize directly from F16, outside benchmark timing."""

import argparse
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import psutil
import yaml

from quant_benchmark_lab.prompts import text_hash
from quant_benchmark_lab.utils import atomic_json, digest, file_hash


def run(args, log: Path):
    with log.open("w") as f:
        subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--toolchain", type=Path, default=Path("locks/toolchain.json"))
    p.add_argument("--model", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quants", nargs="+", choices=["F16", "Q8_0", "Q4_K_M"],
                   default=["F16", "Q8_0", "Q4_K_M"])
    args = p.parse_args()
    catalog = yaml.safe_load(args.catalog.read_text())
    spec = catalog[args.model]
    lock = json.loads(args.toolchain.read_text())
    if lock.get("state") != "pinned":
        p.error("pin the local toolchain first")
    source = Path(lock["source"])
    quantizer = source / "build/bin/llama-quantize"
    converter = source / "convert_hf_to_gguf.py"
    if file_hash(quantizer) != lock["quantizer_sha256"]:
        p.error("quantizer changed since pinning")
    if args.dry_run:
        print(json.dumps({"source_repo": spec["source_repo"], "requested_revision": spec["revision"],
                          "llamacpp_commit": lock["llamacpp_commit"],
                          "quants": args.quants}, indent=2))
        return
    from gguf import GGUFReader
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    info = HfApi().model_info(spec["source_repo"], revision=spec["revision"] or "main")
    revision = info.sha  # resolve once, then fetch immutable revision, never fetch floating weights.
    license_value = getattr(info.card_data, "license", None)
    if not license_value:
        p.error("model license absent; inspect the source card before proceeding")
    destination = (args.catalog.parent / spec["artifacts"]["F16"]["path"]).resolve().parent
    destination.mkdir(parents=True, exist_ok=True)
    match = re.search(r"(\d+(?:\.\d+)?)B", spec["source_repo"])
    estimated_parameters = float(match.group(1)) * 1e9 if match else None
    free_disk = shutil.disk_usage(destination).free
    ram_available = psutil.virtual_memory().available
    if ram_available < 512 * 1024**2:
        p.error("less than 512 MiB available RAM; conversion cannot start reliably")
    if estimated_parameters is not None and free_disk < estimated_parameters * 10:
        p.error("insufficient conservative disk budget for source + F16 + quant + partial artifacts")
    weights = Path(snapshot_download(spec["source_repo"], revision=revision,
        cache_dir=str(destination.parent.parent / "source-cache"),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md"]))
    tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=False)
    template = tokenizer.apply_chat_template([{"role": "user", "content": "{prompt}"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer_files = {f.name: file_hash(f) for f in weights.iterdir()
                       if f.is_file() and ("tokenizer" in f.name or "template" in f.name)}
    spec.update(revision=revision, license=str(license_value), template=template)
    f16_sha = None
    manifests = {}
    selected = [q for q in ("F16", "Q8_0", "Q4_K_M") if q == "F16" or q in args.quants]
    for quant in selected:
        artifact = spec["artifacts"][quant]
        target = (args.catalog.parent / artifact["path"]).resolve()
        manifest_path = (args.catalog.parent / artifact["provenance_path"]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        f16_path = (args.catalog.parent / spec["artifacts"]["F16"]["path"]).resolve()
        cmd = ([sys.executable, str(converter), str(weights), "--outfile", str(partial), "--outtype", "f16"]
               if quant == "F16" else [str(quantizer), str(f16_path), str(partial), quant])
        if target.exists():
            if not manifest_path.exists():
                p.error(f"existing artifact lacks provenance: {target}")
            existing = json.loads(manifest_path.read_text())
            expected = {"revision": revision, "llamacpp_commit": lock["llamacpp_commit"],
                        "sha256": file_hash(target), "template_sha256": text_hash(template),
                        "source_repo": spec["source_repo"], "quant": quant,
                        "license": str(license_value), "tokenizer_sha256": digest(tokenizer_files),
                        "converter_sha256": file_hash(converter),
                        "quantizer_sha256": file_hash(quantizer)}
            if quant != "F16":
                expected["parent_f16_sha256"] = f16_sha
            if (any(existing.get(k) != v for k, v in expected.items())
                    or not existing.get("prepare_environment")
                    or existing.get("size_bytes") != target.stat().st_size):
                p.error(f"existing artifact/provenance differs; use a new artifact directory: {target}")
            # A verification is not a conversion. Preserve original creation provenance byte-for-byte.
            if quant == "F16":
                f16_sha = existing["sha256"]
            artifact["sha256"] = existing["sha256"]
            manifests[quant] = existing
            continue
        else:
            if partial.exists():
                p.error(f"partial artifact exists; inspect/remove explicitly: {partial}")
            run(cmd, target.with_suffix(".log"))
            partial.replace(target)
        checksum = file_hash(target)
        reader = GGUFReader(str(target))
        tensor_types = {}
        parameters = 0
        for tensor in reader.tensors:
            key = str(tensor.tensor_type)
            tensor_types[key] = tensor_types.get(key, 0) + 1
            parameters += int(tensor.n_elements)
        manifest = {"sha256": checksum, "quant": quant, "source_repo": spec["source_repo"],
                    "revision": revision, "license": str(license_value), "llamacpp_commit": lock["llamacpp_commit"],
                    "converter_sha256": file_hash(converter), "quantizer_sha256": file_hash(quantizer),
                    "tokenizer_sha256": digest(tokenizer_files), "tokenizer_files": tokenizer_files,
                    "template_sha256": text_hash(template), "size_bytes": target.stat().st_size,
                    "tensor_types": tensor_types, "parameters": parameters,
                    "bytes_per_parameter": target.stat().st_size / parameters, "command": cmd}
        manifest["prepare_environment"] = {"python": sys.version, "free_disk_bytes_before_download": free_disk,
                                            "available_ram_bytes_before_download": ram_available}
        manifest["prepare_environment"]["packages"] = {
            name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "gguf", "huggingface-hub")}
        if quant == "F16":
            f16_sha = checksum
        else:
            manifest["parent_f16_sha256"] = f16_sha
        atomic_json(manifest_path, manifest)
        artifact["sha256"] = checksum
        manifests[quant] = manifest
    # Catalog changes only after all requested conversions are valid.
    args.catalog.write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True))
    index = args.toolchain.parent / "models.json"
    prior = json.loads(index.read_text()) if index.exists() else {"models": {}}
    existing = prior["models"].get(args.model, {})
    if any(m.get("revision") != revision for m in existing.values()):
        existing = {}
    prior["models"][args.model] = {**existing, **manifests}
    prior["state"] = "prepared"
    atomic_json(index, prior)


if __name__ == "__main__":
    main()
