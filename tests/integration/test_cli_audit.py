import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from quant_benchmark_lab.cli import main
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.protocol import make_schedule
from quant_benchmark_lab.runner import BenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]


class AuditExitTests(unittest.TestCase):
    def test_incomplete_campaign_fails_audit_until_resume_completes(self):
        class StopAfterOneSession(BenchmarkRunner):
            sessions = 0

            def _session(self, session, pending):
                self.sessions += 1
                if self.sessions == 2:
                    raise SystemExit("interrupted before second session")
                return super()._session(session, pending)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            schedule = make_schedule(load_config(ROOT / "configs/demo.yaml"))
            with self.assertRaises(SystemExit):
                StopAfterOneSession(schedule, root).run()
            for expected_code, expected_completion in ((2, "incomplete"), (0, "complete")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = main(["audit", "--campaign", str(root)])
                audit = json.loads(output.getvalue())
                self.assertEqual(result, expected_code)
                self.assertEqual(audit["integrity"], "verified")
                self.assertEqual(audit["trial_completion"], expected_completion)
                self.assertEqual(audit["n_unresolved"] == 0, expected_code == 0)
                if expected_code == 2:
                    BenchmarkRunner(schedule, root, resume=True).run()
