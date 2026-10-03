import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from quant_benchmark_lab.backends.base import Request
from quant_benchmark_lab.backends.llamacpp import LlamaCppBackend
from quant_benchmark_lab.backends.ollama import OllamaBackend
from quant_benchmark_lab.backends.process import offload_evidence
from quant_benchmark_lab.backends.streaming import ndjson, sse
from quant_benchmark_lab.config import load_config
from quant_benchmark_lab.utils import FakeClock

ROOT = Path(__file__).resolve().parents[2]


class StreamingContractTests(unittest.TestCase):
    def test_llama_context_requires_native_per_sequence_log(self):
        config = load_config(ROOT / "configs/demo.yaml")
        size = config.runtime.context_size
        cases = [
            (f"llama_context: n_ctx_seq = {size}\n", size, True),
            (f"llama_context: n_ctx_per_seq = {size}\n", size, True),
            (f"llama_context: n_ctx = {size}\n", None, False),
            (f"llama_context: n_ctx_seq = {size // 2}\n", size // 2, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            backend = LlamaCppBackend(config.cells[1], config, Path(directory), FakeClock())
            try:
                args = backend.launch_args()
                self.assertEqual(args[args.index("--log-verbosity") + 1], "4")
                for log, actual, verified in cases:
                    with (
                        self.subTest(log=log),
                        patch.object(backend.server, "log_text", return_value=log),
                    ):
                        evidence = backend.inspect()
                        self.assertEqual(evidence["actual_context"], actual)
                        self.assertIs(evidence["context_verified"], verified)
            finally:
                backend.close()

    def test_ollama_bundled_server_cannot_restore_evicted_ram_cache(self):
        # b11232 defaults to an 8192 MiB cross-request prompt cache.
        config = load_config(ROOT / "configs/demo.yaml")
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"LLAMA_ARG_CACHE_RAM": "8192", "LLAMA_ARG_FIT": "on"}):
                backend = OllamaBackend(config.cells[0], config, Path(directory), FakeClock())
                try:
                    self.assertEqual(backend.env["LLAMA_ARG_CACHE_RAM"], "0")
                    self.assertEqual(backend.env["LLAMA_ARG_FIT"], "off")
                    self.assertEqual(backend.env["LLAMA_ARG_LOAD_MODE"], "mmap")
                    self.assertEqual(backend.env["LLAMA_ARG_LAZY_MODE"], "off")
                finally:
                    backend.close()

    def test_ollama_preload_preserves_request_context_shift_policy(self):
        # v0.35.1 sched.go needsReload reloads the runner when shift changes.
        config = load_config(ROOT / "configs/demo.yaml")
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if payload.get("stream"):
                return httpx.Response(200, content=b'{"response":"x"}\n{"done":true}\n')
            return httpx.Response(200, json={"done": True, "done_reason": "load"})

        transport = httpx.MockTransport(handler)
        with tempfile.TemporaryDirectory() as directory:
            client = httpx.Client(base_url="http://test", transport=transport)
            backend = OllamaBackend(
                config.cells[0],
                config,
                Path(directory),
                FakeClock(),
                client,
                stream_transport=transport,
            )
            try:
                with (
                    patch.object(backend, "_start_import"),
                    patch.object(backend, "unload"),
                    patch.object(backend, "inspect", return_value={"resident": True}),
                ):
                    backend.load("preload")
                    list(backend.stream(Request("golden", 5, config.generation, ["END"])))
                self.assertEqual(len(seen), 2)
                for payload in seen:
                    self.assertIs(payload["shift"], False)
                    self.assertIs(payload["truncate"], False)
                self.assertEqual(seen[0]["prompt"], "")
                self.assertEqual(seen[0]["options"]["num_predict"], 0)
                self.assertEqual(seen[0]["options"]["num_ctx"], seen[1]["options"]["num_ctx"])
            finally:
                backend.http_stream.close()
                client.close()

    def test_ndjson_utf8_split_at_every_byte(self):
        data = (json.dumps({"response": "Việt Nam"}, ensure_ascii=False) + "\n").encode()
        self.assertEqual(list(ndjson([bytes([x]) for x in data]))[0]["response"], "Việt Nam")

    def test_sse_heartbeat_multiple_events_and_done(self):
        data = b': ping\n\ndata: {"content":"a"}\n\ndata: {"stop":true}\n\ndata: [DONE]\n\n'
        result = list(sse([data[:20], data[20:]]))
        self.assertEqual(result, [{"content": "a"}, {"stop": True}])

    def test_sse_truncation_rejected(self):
        with self.assertRaises(ValueError):
            list(sse([b'data: {"stop":true}\n']))

    def test_invalid_ndjson_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            list(ndjson([b"not json\n"]))

    def test_offload_not_inferred_from_gpu_presence(self):
        self.assertIsNone(offload_evidence("CUDA found")["full_offload"])
        self.assertFalse(offload_evidence("offloaded 4/10 layers")["full_offload"])
        self.assertTrue(offload_evidence("offloaded 10/10 layers")["full_offload"])

    def test_ollama_raw_options_and_final_event(self):
        config = load_config(ROOT / "configs/demo.yaml")
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=b'{"response":" "}\n{"done":true,"eval_count":1}\n')

        client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as d:
            backend = OllamaBackend(
                config.cells[0],
                config,
                Path(d),
                FakeClock(),
                client,
                stream_transport=httpx.MockTransport(handler),
            )
            events = list(backend.stream(Request("golden", 5, config.generation, ["END"])))
        self.assertTrue(seen[0]["raw"])
        self.assertEqual(seen[0]["prompt"], "golden")
        self.assertEqual(seen[0]["options"]["num_predict"], 5)
        self.assertEqual(seen[0]["options"]["stop"], ["END"])
        self.assertEqual([e.kind for e in events], ["text", "final"])
        self.assertEqual(events[0].text, " ")
        backend.http_stream.close()
        client.close()

    def test_llama_completion_settings_and_final(self):
        config = load_config(ROOT / "configs/demo.yaml")
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, content=b'data: {"content":"x"}\n\ndata: {"stop":true}\n\n')

        client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as d:
            backend = LlamaCppBackend(
                config.cells[1],
                config,
                Path(d),
                FakeClock(),
                client,
                stream_transport=httpx.MockTransport(handler),
            )
            events = list(backend.stream(Request("golden", 5, config.generation, ["END"])))
            self.assertIn("--no-context-shift", backend.launch_args())
        self.assertEqual(seen[0]["n_predict"], 5)
        self.assertEqual(seen[0]["temperature"], 0)
        self.assertTrue(seen[0]["cache_prompt"])
        self.assertEqual([e.kind for e in events], ["text", "final"])
        backend.http_stream.close()
        client.close()
