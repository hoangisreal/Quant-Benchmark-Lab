"""Opt-in real integration checks. No model downloads in default CI."""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import psutil

from quant_benchmark_lab.artifacts import verify_artifacts
from quant_benchmark_lab.backends.base import Request
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.metrics import native_metrics
from quant_benchmark_lab.preflight import preflight, reference_counts
from quant_benchmark_lab.prompts import read_items, render, text_hash
from quant_benchmark_lab.runner import make_backend
from quant_benchmark_lab.utils import Clock, atomic_json


class RealEngineTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("QBL_CPU_CONFIG"), "set QBL_CPU_CONFIG to a pinned tiny-model engine config"
    )
    def test_cpu_engine_parity(self):
        config = load_config(Path(os.environ["QBL_CPU_CONFIG"]))
        self.assertEqual(config.deployment, "cpu")
        self.assertFalse(config.synthetic)
        verify_artifacts(config)
        output = os.environ.get("QBL_CPU_OUTPUT")
        with (
            contextlib.nullcontext(output) if output else tempfile.TemporaryDirectory()
        ) as directory:
            root = Path(directory)
            if output:
                root.mkdir(parents=True, exist_ok=False)
                atomic_json(root / "config.json", config.model_dump())
                atomic_json(
                    root / "toolchain.json",
                    json.loads(Path(config.runtime.toolchain_lock).read_text()),
                )
            counts = reference_counts(config, root / "reference")
            for cell in config.cells:
                backend = make_backend(cell, config, root / cell.id, Clock())
                try:
                    load = backend.load("cpu-smoke")
                    atomic_json(root / cell.id / "load.json", load.model_dump())
                    self.assertTrue(load.evidence["context_verified"], load.evidence)
                    process = psutil.Process(backend.server.process.pid)
                    loaded_pids = {process.pid, *(p.pid for p in process.children(recursive=True))}
                    item = read_items(Path(config.performance_path))[0]
                    prompt = render(cell.model, item["text"])
                    request = Request(prompt, 8, config.generation, cell.model.stop)
                    for _ in range(2):
                        list(backend.stream(request))
                    list(backend.stream(backend.prepare_cache(request)))
                    events = list(backend.stream(request))
                    atomic_json(
                        root / cell.id / "stream.json", [event.model_dump() for event in events]
                    )
                    finals = [e for e in events if e.kind == "final"]
                    self.assertEqual(len(finals), 1)
                    self.assertTrue(any(e.kind == "text" and e.text for e in events))
                    self.assertFalse(any(e.kind == "error" for e in events))
                    usage, metrics = native_metrics(cell.engine, finals[0].payload)
                    self.assertEqual(
                        usage.input_total, counts[cell.model.sha256][text_hash(prompt)]
                    )
                    self.assertLessEqual(
                        usage.input_cached, config.protocol.allowed_bos_cache_tokens
                    )
                    self.assertGreater(usage.output, 0)
                    self.assertLessEqual(usage.output, request.max_tokens)
                    self.assertEqual(usage.decoded, max(0, usage.output - 1))
                    self.assertGreater(metrics["prefill_ms"].value, 0)
                    self.assertIsNotNone(metrics["decode_tok_s"].value)
                    if cell.engine == "llamacpp":
                        native_rate = finals[0].payload["timings"]["predicted_per_second"]
                        self.assertAlmostEqual(
                            metrics["decode_tok_s"].value,
                            native_rate,
                            delta=max(1e-6, native_rate * 1e-5),
                        )
                    self.assertEqual(
                        loaded_pids,
                        {process.pid, *(p.pid for p in process.children(recursive=True))},
                    )
                    backend.unload()
                    self.assertFalse(backend.inspect()["resident"])
                finally:
                    backend.close()

    @unittest.skipUnless(
        os.environ.get("QBL_GPU_CONFIG"), "set QBL_GPU_CONFIG on the RTX 3050 lab host"
    )
    def test_target_gpu_preflight(self):
        config = load_config(Path(os.environ["QBL_GPU_CONFIG"]))
        with tempfile.TemporaryDirectory() as directory:
            result = preflight(config, Path(directory) / "pilot")
            self.assertTrue(result["passed"], result["failures"])
