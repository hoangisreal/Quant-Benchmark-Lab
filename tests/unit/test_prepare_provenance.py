import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from quant_benchmark_lab.utils import file_hash

ROOT = Path(__file__).resolve().parents[2]


class PrepareProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "build/bin").mkdir(parents=True)
        quant = self.source / "build/bin/llama-quantize"
        quant.write_text("quantizer fixture")
        self.converter = self.source / "convert_hf_to_gguf.py"
        self.converter.write_text("converter fixture")
        self.weights = self.root / "weights"
        self.weights.mkdir()
        (self.weights / "tokenizer.json").write_text("{}")
        self.revision = "a" * 40
        self.catalog = self.root / "catalog.yaml"
        artifacts = {q: {"path": str(self.root / f"{q}.gguf"),
                         "provenance_path": str(self.root / f"{q}.json")}
                     for q in ("F16", "Q4_K_M")}
        self.catalog.write_text(yaml.safe_dump({"test": {"source_repo": "owner/Test", "revision": self.revision,
                                                        "artifacts": artifacts}}))
        self.lock = self.root / "toolchain.json"
        self.lock.write_text(json.dumps({"state": "pinned", "source": str(self.source),
                                        "quantizer_sha256": file_hash(quant), "llamacpp_commit": self.revision}))
        spec = importlib.util.spec_from_file_location("qbl_prepare_test", ROOT / "scripts/prepare_models.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        info = types.SimpleNamespace(sha=self.revision, card_data=types.SimpleNamespace(license="apache-2.0"))
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: "<user>{prompt}")
        self.stubs = {
            "huggingface_hub": types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=lambda *a, **kw: info),
                                                      snapshot_download=lambda *a, **kw: str(self.weights)),
            "transformers": types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)),
            "gguf": types.SimpleNamespace(GGUFReader=lambda *a: types.SimpleNamespace(
                tensors=[types.SimpleNamespace(tensor_type="fixture", n_elements=4)])),
        }

    def prepare(self, allow_conversion, version="original-version"):
        def convert(args, log):
            if not allow_conversion:
                self.fail("existing artifact must not be converted")
            output = args[args.index("--outfile") + 1] if "--outfile" in args else args[2]
            Path(output).write_bytes(b"artifact fixture " + str(output).encode())
        argv = ["prepare", "--catalog", str(self.catalog), "--toolchain", str(self.lock),
                "--model", "test", "--quants", "F16", "Q4_K_M"]
        with patch.dict(sys.modules, self.stubs), patch.object(sys, "argv", argv), patch.object(self.module, "run", side_effect=convert), patch.object(self.module.importlib.metadata, "version", return_value=version):
            self.module.main()

    def test_reuse_preserves_creation_bytes_across_package_change(self):
        self.prepare(True)
        before = {q: (self.root / f"{q}.json").read_bytes() for q in ("F16", "Q4_K_M")}
        self.prepare(False, "new-version")
        self.assertEqual(before, {q: (self.root / f"{q}.json").read_bytes() for q in before})

    def test_changed_converter_refused_without_rewriting_catalog_or_manifests(self):
        self.prepare(True)
        before = {p: p.read_bytes() for p in (self.catalog, self.root / "F16.json", self.root / "Q4_K_M.json", self.root / "models.json")}
        self.converter.write_text("changed converter")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.prepare(False)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_changed_f16_lineage_refused_before_catalog_write(self):
        self.prepare(True)
        path = self.root / "Q4_K_M.json"
        manifest = json.loads(path.read_text())
        manifest["parent_f16_sha256"] = "b" * 64
        path.write_text(json.dumps(manifest))
        before = self.catalog.read_bytes(), path.read_bytes(), (self.root / "models.json").read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.prepare(False)
        self.assertEqual(before, (self.catalog.read_bytes(), path.read_bytes(), (self.root / "models.json").read_bytes()))
