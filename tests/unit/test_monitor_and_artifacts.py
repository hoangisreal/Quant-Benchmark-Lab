import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from quant_benchmark_lab.artifacts import verify_artifacts, verify_model, verify_payload
from quant_benchmark_lab.config import load_config, yaml_read
from quant_benchmark_lab.environment import (
    capture_environment,
    environment_identity,
    payload_identity,
)
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
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "toolchain.json"
            lock.write_text(json.dumps({"state": "unresolved"}))
            config.runtime.toolchain_lock = str(lock)
            with self.assertRaises(ValueError):
                verify_artifacts(config)
        unresolved = config.cells[0].model.model_copy(update={"sha256": None})
        with self.assertRaises(ValueError):
            verify_model(unresolved)

    def test_unchanged_launcher_does_not_hide_native_library_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_bytes(b"unchanged launcher")
            native = root / "libllama-server-impl.so"
            native.write_bytes(b"native implementation A")
            identity = payload_identity(root)
            verify_payload(identity, "llamacpp")
            native.write_bytes(b"native implementation B")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                verify_payload(identity, "llamacpp")

    def test_added_backend_and_missing_payload_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "llama-server").write_bytes(b"launcher")
            identity = payload_identity(root)
            (root / "libggml-cuda.so").write_bytes(b"new discovered backend")
            with self.assertRaisesRegex(ValueError, "payload changed"):
                verify_payload(identity, "ollama")
        with self.assertRaisesRegex(ValueError, "provenance missing"):
            verify_payload(None, "ollama")

    def test_duplicate_yaml_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("seed: 42\nseed: 0\n")
            with self.assertRaises(ValueError):
                yaml_read(path)

    def test_doctor_missing_tools_are_explicit(self):
        with patch(
            "quant_benchmark_lab.environment.command", return_value={"unavailable": "missing"}
        ):
            env = capture_environment("no-such-ollama", "no-such-llama")
        self.assertIn("unavailable", env["ollama"])
        self.assertIsNotNone(env["harness_source_sha256"])
        self.assertEqual(
            environment_identity(env), environment_identity(json.loads(json.dumps(env)))
        )
