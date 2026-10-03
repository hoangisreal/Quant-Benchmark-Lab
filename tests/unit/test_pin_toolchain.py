import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class ToolchainSourceTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "qbl_pin_test", ROOT / "scripts/pin_toolchain.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.sha = "a" * 40

    def test_ollama_release_must_match_clean_source_head(self):
        results = [
            {"returncode": 0, "stdout": self.sha},
            {"returncode": 0, "stdout": ""},
            {"returncode": 0, "stdout": "b" * 40},
        ]
        with patch.object(self.module, "command", side_effect=results):
            with self.assertRaisesRegex(ValueError, "HEAD must match"):
                self.module.ollama_source_identity(Path("source"), "v0.35.1")

    def test_dirty_source_and_floating_release_are_refused(self):
        results = [
            {"returncode": 0, "stdout": self.sha},
            {"returncode": 0, "stdout": " M llm/server.go"},
        ]
        with patch.object(self.module, "command", side_effect=results):
            with self.assertRaisesRegex(ValueError, "clean"):
                self.module.source_identity(Path("source"))
        with self.assertRaisesRegex(ValueError, "explicit version"):
            self.module.ollama_source_identity(Path("source"), "main")

    def test_pinned_release_records_immutable_source_reference(self):
        results = [
            {"returncode": 0, "stdout": self.sha},
            {"returncode": 0, "stdout": ""},
            {"returncode": 0, "stdout": self.sha},
        ]
        with patch.object(self.module, "command", side_effect=results):
            identity = self.module.ollama_source_identity(Path("missing-test-source"), "v0.35.1")
        self.assertEqual(identity["source_commit"], self.sha)
        self.assertIn(self.sha, identity["source_reference"])
        self.assertEqual(identity["release"], "v0.35.1")


if __name__ == "__main__":
    unittest.main()
