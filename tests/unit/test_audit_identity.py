import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_benchmark_lab.backends.llamacpp import LlamaCppBackend
from quant_benchmark_lab.backends.ollama import OllamaBackend
from quant_benchmark_lab.backends.process import OwnedProcess
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.environment import engine_environment
from quant_benchmark_lab.prompts import eviction_text, render, text_hash
from quant_benchmark_lab.protocol import (
    dataset_identity,
    make_schedule,
    measurement_identity,
    validate_certificate,
)
from quant_benchmark_lab.runner import BenchmarkRunner
from quant_benchmark_lab.utils import FakeClock

ROOT = Path(__file__).resolve().parents[2]


class IdentityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/demo.yaml")
        for key in ("performance_path", "quality_path", "answers_path"):
            dest = self.root / (key + ".jsonl")
            shutil.copyfile(getattr(self.config, key), dest)
            setattr(self.config, key, str(dest))

    def certificate(self):
        config = self.config.model_copy(update={"synthetic": False, "stage": "official", "deployment": "gpu",
            "protocol": self.config.protocol.model_copy(update={"state": "frozen"})})
        # A mock certificate is used only to exercise validation; no hardware evidence is emitted.
        evidence = {"passed": True, "synthetic": False,
                    "measurement_hash": measurement_identity(config),
                    "dataset_hashes": dataset_identity(config), "input_counts": {}, "cache_counts": {}}
        schedule = make_schedule(config)
        cells = {c.id: c for c in config.cells}
        for session in schedule["sessions"]:
            cell = cells[session["cell_id"]]
            counts = evidence["input_counts"].setdefault(cell.model.sha256, {})
            caches = evidence["cache_counts"].setdefault(cell.id, {}).setdefault(session["mode"], {})
            for trial in session["trials"]:
                text = render(cell.model, trial["item"]["text"])
                counts[text_hash(text)] = 10
                counts[text_hash(eviction_text(text))] = 20
                caches[text_hash(text)] = 0
        return config, evidence, schedule

    def test_each_workload_content_change_invalidates_certificate(self):
        for key in ("performance_path", "quality_path", "answers_path"):
            with self.subTest(file=key):
                config, evidence, _ = self.certificate()
                path = Path(getattr(config, key))
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                self.assertNotEqual(measurement_identity(config), evidence["measurement_hash"])
                with self.assertRaisesRegex(ValueError, "dataset mismatch"):
                    validate_certificate(config, evidence)
                path.write_bytes(original)

    def test_official_gate_rejects_missing_input_cache_or_quality_coverage(self):
        config, evidence, schedule = self.certificate()
        validate_certificate(config, evidence, schedule)
        for mode in ("cold", "warm", "quality"):
            copy = json.loads(json.dumps(evidence))
            # JSON stringifies None only for fixture model keys; remove coverage by cell instead.
            copy["input_counts"] = evidence["input_counts"]
            del copy["cache_counts"][config.cells[0].id][mode]
            with self.assertRaisesRegex(ValueError, "cache evidence"):
                validate_certificate(config, copy, schedule)
        evidence["input_counts"] = {}
        with self.assertRaisesRegex(ValueError, "reference tokens"):
            validate_certificate(config, evidence, schedule)
        runner = BenchmarkRunner(schedule, self.root / "official", evidence=evidence)
        runner.env["gpu"] = {"available": True, "devices": [{"index": 0, "name": "RTX 3050", "total_bytes": 4 * 1024**3}]}
        with self.assertRaisesRegex(ValueError, "reference tokens"):
            runner._hardware_gate()  # fails before artifact verification or any backend load

    def test_engine_environment_filters_ambient_overrides_at_launch(self):
        with patch.dict(os.environ, {"LLAMA_ARG_KV_OFFLOAD": "false", "LLAMA_ARG_SPEC_TYPE": "ngram-simple",
                                     "OLLAMA_NUM_PARALLEL": "99", "LD_PRELOAD": "/not-loaded.so",
                                     "GGML_CUDA_FORCE_MMQ": "1", "CUDA_VISIBLE_DEVICES": "0"}):
            env = engine_environment()
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            for key in ("LLAMA_ARG_KV_OFFLOAD", "LLAMA_ARG_SPEC_TYPE", "OLLAMA_NUM_PARALLEL", "LD_PRELOAD", "GGML_CUDA_FORCE_MMQ"):
                self.assertNotIn(key, env)
            process = OwnedProcess(self.root / "process.log", 12345)
            with patch("quant_benchmark_lab.backends.process.socket.socket") as socket, patch("quant_benchmark_lab.backends.process.subprocess.Popen") as popen:
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                process.start(["mock-engine"])
                self.assertEqual(popen.call_args.kwargs["env"], env)
                process.handle.close()
            backend = OllamaBackend(self.config.cells[0], self.config, self.root, FakeClock())
            self.assertEqual(backend.env["OLLAMA_NUM_PARALLEL"], "1")
            self.assertNotIn("LD_PRELOAD", backend.env)
            backend.close()
        llama = LlamaCppBackend(self.config.cells[1], self.config, self.root, FakeClock())
        self.assertIn("--no-kv-offload", llama.launch_args())
        self.assertEqual(llama.launch_args()[llama.launch_args().index("--spec-type") + 1], "none")
        llama.close()
