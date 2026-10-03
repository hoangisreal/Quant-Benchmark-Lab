"""Opt-in real integration checks. No model downloads in default CI."""

import os
import tempfile
import unittest
from pathlib import Path

from quant_benchmark_lab.artifacts import verify_artifacts
from quant_benchmark_lab.backends.base import Request
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.metrics import native_metrics
from quant_benchmark_lab.preflight import preflight, reference_counts
from quant_benchmark_lab.prompts import read_items, render, text_hash
from quant_benchmark_lab.runner import make_backend
from quant_benchmark_lab.utils import Clock


class RealEngineTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("QBL_CPU_CONFIG"), "set QBL_CPU_CONFIG to a pinned tiny-model engine config")
    def test_cpu_engine_parity(self):
        config = load_config(Path(os.environ["QBL_CPU_CONFIG"]))
        self.assertEqual(config.deployment, "cpu")
        self.assertFalse(config.synthetic)
        verify_artifacts(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            counts = reference_counts(config, root / "reference")
            for cell in config.cells:
                backend = make_backend(cell, config, root / cell.id, Clock())
                try:
                    backend.load("cpu-smoke")
                    item = read_items(Path(config.performance_path))[0]
                    prompt = render(cell.model, item["text"])
                    request = Request(prompt, 8, config.generation, cell.model.stop)
                    for _ in range(2):
                        list(backend.stream(request))
                    list(backend.stream(backend.prepare_cache(request)))
                    events = list(backend.stream(request))
                    finals = [e for e in events if e.kind == "final"]
                    self.assertEqual(len(finals), 1)
                    usage, _ = native_metrics(cell.engine, finals[0].payload)
                    self.assertEqual(usage.input_total, counts[cell.model.sha256][text_hash(prompt)])
                    self.assertLessEqual(usage.input_cached, config.protocol.allowed_bos_cache_tokens)
                finally:
                    backend.close()

    @unittest.skipUnless(os.environ.get("QBL_GPU_CONFIG"), "set QBL_GPU_CONFIG on the RTX 3050 lab host")
    def test_target_gpu_preflight(self):
        config = load_config(Path(os.environ["QBL_GPU_CONFIG"]))
        with tempfile.TemporaryDirectory() as directory:
            result = preflight(config, Path(directory) / "pilot")
            self.assertTrue(result["passed"], result["failures"])
