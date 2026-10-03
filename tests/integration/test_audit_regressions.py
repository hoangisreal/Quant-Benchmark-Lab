"""Regression coverage for the nine findings in the 2026-10-03 audit."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant_benchmark_lab.backends.fake import FakeBackend
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.protocol import make_schedule
from quant_benchmark_lab.quality.evaluator import evaluate
from quant_benchmark_lab.reporting.reporter import report
from quant_benchmark_lab.runner import BenchmarkRunner
from quant_benchmark_lab.storage import JOURNALS, ResultStore, read_jsonl
from quant_benchmark_lab.utils import file_hash

ROOT = Path(__file__).resolve().parents[2]


class AuditCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "campaign"
        self.config = load_config(ROOT / "configs/demo.yaml")
        self.schedule = make_schedule(self.config)

    def run_campaign(self):
        BenchmarkRunner(self.schedule, self.root).run()
        return ResultStore(self.root)

    def test_resume_partial_warm_session_preserves_loads_and_traces(self):
        class Crash(BenchmarkRunner):
            warm = 0

            def _request(self, backend, cell, session, trial, request, phase, idle):
                if phase == "measurement" and session["mode"] == "warm":
                    self.warm += 1
                    if self.warm == 2:
                        raise SystemExit("process stopped before next trial journal")
                return super()._request(backend, cell, session, trial, request, phase, idle)

        with self.assertRaises(SystemExit):
            Crash(self.schedule, self.root).run()
        before = {str(p): file_hash(p) for p in self.root.glob("engine-logs/*/gpu-session.jsonl")}
        BenchmarkRunner(self.schedule, self.root, resume=True).run()
        loads = read_jsonl(self.root / "loads.jsonl")
        self.assertEqual(len(loads), len({r["observation"]["id"] for r in loads}))
        self.assertTrue(all(file_hash(Path(p)) == sha for p, sha in before.items()))
        report(self.root, Path(self.temp.name) / "report")

    def test_all_journals_detect_edit_and_deletion(self):
        store = self.run_campaign()
        for name in JOURNALS:
            # control can be empty on a successful campaign; add a committed event.
            if name == "control.jsonl":
                store.acquire()
                try:
                    store.append(name, {"kind": "audit-test"})
                finally:
                    store.release()
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(journal=name):
                path.write_bytes(b'{"changed":true}\n')
                with self.assertRaises(ValueError):
                    store.verify()
                path.write_bytes(original)
                path.unlink()
                with self.assertRaises(ValueError):
                    store.verify()
                path.write_bytes(original)
        store.verify()

    def test_session_trace_and_metadata_inventory_required(self):
        store = self.run_campaign()
        trace = next(self.root.glob("engine-logs/*/gpu-session.jsonl"))
        original = trace.read_bytes()
        trace.write_bytes(b"[]\n")
        with self.assertRaisesRegex(ValueError, "trace changed"):
            store.verify()
        trace.write_bytes(original)
        (self.root / "metadata_checksums.json").unlink()
        with self.assertRaisesRegex(ValueError, "inventory missing"):
            store.verify()
        with self.assertRaisesRegex(ValueError, "missing manifest"):
            ResultStore(Path(self.temp.name) / "not-a-campaign").verify()

    def test_every_journal_recovers_torn_tail_and_committed_append(self):
        store = self.run_campaign()
        for name in JOURNALS:
            original = (self.root / name).read_bytes()
            for suffix in (b'{"partial":', b'\xf0\x9f'):
                with self.subTest(journal=name, suffix=suffix):
                    (self.root / name).write_bytes(original + suffix)
                    store.acquire()
                    store.release()
                    self.assertEqual((self.root / name).read_bytes(), original)
            if original:
                # Write-ahead entry exists but its materialized append was interrupted.
                boundary = original.rfind(b"\n", 0, len(original) - 1) + 1
                (self.root / name).write_bytes(original[:boundary] + original[boundary:boundary + 3])
                store.acquire()
                store.release()
                self.assertEqual((self.root / name).read_bytes(), original)
        store.verify()
        report(self.root, Path(self.temp.name) / "recovered-report")

    def test_missing_journal_snapshot_is_detected(self):
        store = self.run_campaign()
        next((self.root / "journal" / "runs.jsonl").glob("*.json")).unlink()
        with self.assertRaises(ValueError):
            store.verify()

    def test_sensor_loss_preserves_received_response(self):
        runner = None

        class LoseSensor(FakeBackend):
            def stream(self, request):
                for event in super().stream(request):
                    if event.kind == "final":
                        runner.monitor.error = "sensor disconnected"
                    yield event

        runner = BenchmarkRunner(self.schedule, self.root,
            backend_factory=lambda c, cfg, work, clock: LoseSensor(c, clock))
        with self.assertRaises(RuntimeError):
            runner.run()
        store = ResultStore(self.root)
        record = store.records()[0]
        self.assertEqual(record.output, "SYNTHETIC OUTPUT")
        self.assertEqual(record.status, "invalid")
        self.assertIsNone(record.metrics["peak_request_vram_bytes"].value)
        self.assertTrue(read_jsonl(store.safe_path(record.paths["events.jsonl"])))
        store.verify()

    def test_report_recomputes_stale_and_modified_quality(self):
        class Stop(BenchmarkRunner):
            quality = 0

            def _session(self, session, pending):
                if session["mode"] == "quality":
                    self.quality += 1
                    if self.quality == 2:
                        raise SystemExit("stopped between cells")
                return super()._session(session, pending)

        with self.assertRaises(SystemExit):
            Stop(self.schedule, self.root).run()
        old = evaluate(self.root)
        self.assertEqual(sum(i["answered"] for i in old["items"]), 6)
        BenchmarkRunner(self.schedule, self.root, resume=True).run()
        old["summary"][0]["macro_accuracy"] = 0.987654321
        (self.root / "quality.json").write_text(json.dumps(old))
        output = Path(self.temp.name) / "fresh-report"
        report(self.root, output)
        self.assertNotIn("0.988", (output / "report.md").read_text())
        fresh = evaluate(self.root, persist=False)
        self.assertEqual(sum(i["answered"] for i in fresh["items"]), 12)
        self.assertNotEqual(old["identity"], fresh["identity"])
        with (output / "quality_items.csv").open() as f:
            self.assertEqual(sum(row["answered"] == "True" for row in csv.DictReader(f)), 12)

    def test_session_reservation_survives_crash_before_load(self):
        class FailFactory:
            def __call__(self, *args):
                raise SystemExit("before engine starts")

        with self.assertRaises(SystemExit):
            BenchmarkRunner(self.schedule, self.root, backend_factory=FailFactory()).run()
        reserved = next((self.root / "engine-logs").iterdir())
        BenchmarkRunner(self.schedule, self.root, resume=True).run()
        self.assertTrue(reserved.is_dir())
        self.assertTrue(reserved.with_name(reserved.name[:-1] + "2").is_dir())
        ResultStore(self.root).verify()
