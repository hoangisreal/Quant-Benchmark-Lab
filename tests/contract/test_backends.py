import json
import tempfile
import unittest
from pathlib import Path

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
            list(ndjson([b'not json\n']))

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
            backend = OllamaBackend(config.cells[0], config, Path(d), FakeClock(), client, stream_transport=httpx.MockTransport(handler))
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
            backend = LlamaCppBackend(config.cells[1], config, Path(d), FakeClock(), client, stream_transport=httpx.MockTransport(handler))
            events = list(backend.stream(Request("golden", 5, config.generation, ["END"])))
            self.assertIn("--no-context-shift", backend.launch_args())
        self.assertEqual(seen[0]["n_predict"], 5)
        self.assertEqual(seen[0]["temperature"], 0)
        self.assertTrue(seen[0]["cache_prompt"])
        self.assertEqual([e.kind for e in events], ["text", "final"])
        backend.http_stream.close()
        client.close()
