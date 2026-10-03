import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from quant_benchmark_lab.artifacts import verify_model
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.preflight import monitor_overhead
from quant_benchmark_lab.prompts import text_hash
from quant_benchmark_lab.protocol import make_schedule
from quant_benchmark_lab.utils import file_hash

ROOT = Path(__file__).resolve().parents[2]


class AuditValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/demo.yaml")

    def test_inline_and_external_runtime_resolve_at_defining_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = self.config.runtime.model_dump()
            runtime.update(
                llama_binary="llama-server",
                ollama_binary="./bin/ollama",
                toolchain_lock="./toolchain.json",
            )
            for external in (False, True):
                with self.subTest(external=external):
                    raw = self.config.model_dump()
                    parent = root
                    if external:
                        parent = root / "shared"
                        parent.mkdir()
                        (parent / "runtime.yaml").write_text(yaml.safe_dump(runtime))
                        raw["runtime"] = "shared/runtime.yaml"
                    else:
                        raw["runtime"] = runtime
                    source = root / "campaign.yaml"
                    source.write_text(yaml.safe_dump(raw))
                    config = load_config(source)
                    self.assertEqual(config.runtime.llama_binary, "llama-server")
                    self.assertEqual(config.runtime.ollama_binary, str(parent / "bin/ollama"))
                    self.assertEqual(config.runtime.toolchain_lock, str(parent / "toolchain.json"))

    def test_missing_extra_and_mismatched_answers_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            answers = [
                json.loads(line) for line in Path(self.config.answers_path).read_text().splitlines()
            ]
            config = self.config.model_copy(update={"answers_path": str(path)})
            for mode in ("missing", "extra", "scorer"):
                changed = copy.deepcopy(answers)
                if mode == "missing":
                    changed.pop()
                elif mode == "extra":
                    changed.append({"id": "extra", "scorer": "exact", "expected": "x"})
                else:
                    changed[0] = {"id": changed[0]["id"], "scorer": "json", "expected": {}}
                path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "IDs|scorer"):
                    make_schedule(config)

    def test_model_provenance_binds_license_and_actual_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.gguf"
            artifact.write_bytes(b"unit-test artifact")
            provenance = root / "model.json"
            model = self.config.cells[0].model.model_copy(
                update={
                    "path": str(artifact),
                    "provenance_path": str(provenance),
                    "sha256": file_hash(artifact),
                    "revision": "a" * 40,
                    "license": "apache-2.0",
                }
            )
            manifest = {
                key: getattr(model, key)
                for key in ("sha256", "revision", "source_repo", "quant", "license")
            }
            manifest.update(
                size_bytes=artifact.stat().st_size,
                template_sha256=text_hash(model.template),
                tokenizer_sha256="b" * 64,
                llamacpp_commit="a" * 40,
                parent_f16_sha256="c" * 64,
            )
            provenance.write_text(json.dumps(manifest))
            verify_model(model)
            for key, value in (("license", "other"), ("size_bytes", 1)):
                provenance.write_text(json.dumps({**manifest, key: value}))
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                    verify_model(model)

    def test_monitor_overhead_checks_every_cell_and_preserves_failure(self):
        def result(config, counts, work, cell):
            return {
                "passed": cell.id != "llamacpp",
                "overhead_fraction": 0.04 if cell.id == "llamacpp" else 0.01,
            }

        with patch(
            "quant_benchmark_lab.preflight._monitor_overhead_cell", side_effect=result
        ) as probe:
            evidence = monitor_overhead(self.config, {}, Path("fixture-only"))
        self.assertFalse(evidence["passed"])
        self.assertEqual(evidence["overhead_fraction"], 0.04)
        self.assertEqual(set(evidence["cells"]), {cell.id for cell in self.config.cells})
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            [Path("fixture-only") / cell.id for cell in self.config.cells],
        )

    def test_workload_manifest_matches_all_public_datasets(self):
        manifest = json.loads((ROOT / "locks/workloads.json").read_text())
        self.assertEqual(
            set(manifest["files"]),
            {
                "data/performance/prompts.jsonl",
                "data/quality/items.jsonl",
                "data/quality/answers.jsonl",
            },
        )
        for path, expected in manifest["files"].items():
            self.assertEqual(file_hash(ROOT / path), expected)
