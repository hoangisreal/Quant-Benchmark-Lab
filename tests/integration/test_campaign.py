import json
import tempfile
import unittest
from pathlib import Path

from quant_benchmark_lab.backends.fake import FakeBackend
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.preflight import preflight
from quant_benchmark_lab.protocol import make_schedule
from quant_benchmark_lab.quality.evaluator import evaluate
from quant_benchmark_lab.reporting.aggregate import aggregate
from quant_benchmark_lab.reporting.reporter import report
from quant_benchmark_lab.runner import BenchmarkRunner
from quant_benchmark_lab.schema import StreamEvent
from quant_benchmark_lab.storage import ResultStore, read_jsonl
from quant_benchmark_lab.utils import file_hash

ROOT = Path(__file__).resolve().parents[2]


class FakeCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "campaign"
        self.config = load_config(ROOT / "configs/demo.yaml")
        self.schedule = make_schedule(self.config)

    def run_campaign(self, **kwargs):
        BenchmarkRunner(self.schedule, self.root, **kwargs).run()
        return ResultStore(self.root)

    def test_end_to_end_counts_timings_and_synthetic_report(self):
        store = self.run_campaign()
        store.verify()
        records = store.records()
        measured = [r for r in records if r.phase == "measurement"]
        quality = [r for r in records if r.phase == "quality"]
        self.assertEqual(len(measured), 24)
        self.assertEqual(len(quality), 12)
        self.assertEqual(len([r for r in records if r.phase == "warmup"]), 20)
        self.assertTrue(all(r.synthetic and r.status == "ok" for r in records))
        for r in measured:
            self.assertEqual(r.metrics["ttft_stream_ms"].value, 30)
            self.assertEqual(r.metrics["e2e_request_ms"].value, 70)
            self.assertEqual(r.usage.output, 2)  # not HTTP chunk count
        loads = read_jsonl(self.root / "loads.jsonl")
        self.assertEqual(len(loads), 18)  # 8 cold + 8 warm sessions + 2 quality sessions
        quality_result = evaluate(self.root)
        self.assertTrue(quality_result["synthetic"])
        output = Path(self.temp.name) / "reports"
        report(self.root, output)
        self.assertIn("SYNTHETIC DATA", (output / "report.md").read_text())
        report_identity = json.loads((output / "manifest.json").read_text())["quality_identity"]
        self.assertEqual(report_identity, quality_result["identity"])
        self.assertTrue(list(output.glob("*.svg")))

    def test_resume_skips_successful_trials(self):
        store = self.run_campaign()
        before = (self.root / "runs.jsonl").read_bytes()
        self.run_campaign(resume=True)
        self.assertEqual((self.root / "runs.jsonl").read_bytes(), before)
        self.assertEqual(len(store.records()), 84)

    def test_report_rebuild_is_deterministic(self):
        self.run_campaign()
        evaluate(self.root)
        output = Path(self.temp.name) / "reports"
        report(self.root, output)
        first = file_hash(output / "metrics.csv"), file_hash(output / "report.md")
        report(self.root, output)
        self.assertEqual(
            first, (file_hash(output / "metrics.csv"), file_hash(output / "report.md"))
        )

    def test_raw_output_tamper_rejected(self):
        store = self.run_campaign()
        r = store.records()[0]
        store.safe_path(r.paths["output.txt"]).write_text("modified")
        with self.assertRaises(ValueError):
            store.verify()

    def test_run_metadata_tamper_rejected(self):
        store = self.run_campaign()
        path = self.root / "runs.jsonl"
        rows = read_jsonl(path)
        rows[0]["metrics"]["e2e_request_ms"]["value"] = 1.0
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(ValueError):
            store.verify()

    def test_resume_schedule_change_rejected(self):
        self.run_campaign()
        changed = dict(self.schedule)
        changed["sessions"] = changed["sessions"][::-1]
        with self.assertRaises(ValueError):
            BenchmarkRunner(changed, self.root, resume=True)

    def test_mix_synthetic_flags_rejected(self):
        store = self.run_campaign()
        rows = store.records()
        rows[0].synthetic = False
        with self.assertRaises(ValueError):
            aggregate(rows, self.schedule)

    def test_missing_final_preserved_as_invalid(self):
        class NoFinal(FakeBackend):
            def stream(self, request):
                yield StreamEvent(receipt_ns=self.clock.now(), kind="text", text="partial")

        def factory(cell, cfg, work, clock):
            return NoFinal(cell, clock)

        with self.assertRaises(ValueError):  # warmup must halt the session/campaign.
            self.run_campaign(backend_factory=factory)
        records = ResultStore(self.root).records()
        self.assertTrue(records)
        self.assertTrue(all(r.status == "invalid" for r in records))
        self.assertIn("partial", records[0].output)

    def test_torn_tail_quarantined_and_orphan_number_reserved(self):
        store = self.run_campaign()
        with (self.root / "runs.jsonl").open("ab") as f:
            f.write(b'{"unfinished":')
        orphan = self.root / "attempts" / "reserved-attempt1"
        orphan.mkdir()
        store.acquire()
        try:
            self.assertEqual(store.attempts("reserved"), 1)
            self.assertTrue(list(self.root.glob("runs.jsonl.torn-*")))
            self.assertTrue((self.root / "runs.jsonl").read_bytes().endswith(b"\n"))
        finally:
            store.release()

    def test_preflight_rejects_synthetic_evidence(self):
        result = preflight(self.config, Path(self.temp.name) / "preflight")
        self.assertFalse(result["passed"])
        self.assertTrue(result["synthetic"])

    def test_answer_key_tamper_rejected(self):
        self.run_campaign()
        (self.root / "answers.jsonl").write_text('{"id":"changed"}\n')
        with self.assertRaises(ValueError):
            evaluate(self.root)

    def test_simultaneous_campaign_writers_rejected(self):
        a, b = ResultStore(self.root), ResultStore(self.root)
        a.acquire()
        try:
            with self.assertRaises(RuntimeError):
                b.acquire()
        finally:
            a.release()

    def test_rejected_runner_does_not_mutate_locked_campaign(self):
        store = self.run_campaign()
        before = {p.name: p.read_bytes() for p in self.root.glob("*.json*")}
        store.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.run_campaign(resume=True)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.glob("*.json*")})
        finally:
            store.release()
