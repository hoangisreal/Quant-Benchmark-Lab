import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from quant_benchmark_lab.artifacts import verify_artifacts, verify_model
from quant_benchmark_lab.config import load_config, yaml_read
from quant_benchmark_lab.environment import capture_environment, environment_identity
from quant_benchmark_lab.monitoring.gpu import GPULock, GPUMonitor
from quant_benchmark_lab.utils import FakeClock

ROOT = Path(__file__).resolve().parents[2]


class MonitorAndIntegrityTests(unittest.TestCase):
    def test_different_campaigns_cannot_hold_same_gpu(self):
        identifier = "GPU-test-" + str(uuid.uuid4())
        first, second = GPULock(identifier), GPULock(identifier)
        first.acquire()
        try:
            with self.assertRaises(RuntimeError):
                second.acquire()
        finally:
            first.close()
        second.acquire()
        second.close()
    def test_peak_scope_and_idle_median(self):
        clock = FakeClock()
        monitor = GPUMonitor(clock, 20, synthetic=True)
        monitor.start()
        idle_start = clock.now()
        clock.sleep(2)
        self.assertEqual(monitor.memory(idle_start, clock.now(), "median").value, 100 * 1024**2)
        monitor.mark("load")
        load_start = clock.now()
        clock.sleep(1)
        self.assertEqual(monitor.memory(load_start, clock.now()).value, 350 * 1024**2)
        monitor.mark("request")
        start = clock.now()
        clock.sleep(1)
        self.assertEqual(monitor.memory(start, clock.now()).value, 400 * 1024**2)
        monitor.close()

    def test_cpu_monitor_does_not_invent_vram(self):
        clock = FakeClock()
        monitor = GPUMonitor(clock, 20, enabled=False)
        monitor.start()
        self.assertIsNone(monitor.memory(0, clock.now()).value)
        monitor.close()

    def test_monitor_error_is_fatal(self):
        monitor = GPUMonitor(FakeClock(), 20, enabled=False)
        monitor.error = "sensor lost"
        with self.assertRaises(RuntimeError):
            monitor.mark("request")

    def test_unresolved_real_artifact_cannot_run(self):
        config = load_config(ROOT / "configs/experiments/engine.yaml")
        with self.assertRaises(ValueError):
            verify_artifacts(config)
        with self.assertRaises(ValueError):
            verify_model(config.cells[0].model)

    def test_duplicate_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("seed: 42\nseed: 0\n")
            with self.assertRaises(ValueError):
                yaml_read(path)

    def test_doctor_missing_tools_are_explicit(self):
        with patch("quant_benchmark_lab.environment.command", return_value={"unavailable": "missing"}):
            env = capture_environment("no-such-ollama", "no-such-llama")
        self.assertIn("unavailable", env["ollama"])
        self.assertIsNotNone(env["harness_source_sha256"])
        self.assertEqual(environment_identity(env), environment_identity(json.loads(json.dumps(env))))
