import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from quant_benchmark_lab.config import CampaignConfig, load_config
from quant_benchmark_lab.metrics import native_metrics
from quant_benchmark_lab.prompts import read_items, render, text_hash
from quant_benchmark_lab.protocol import equivalence, make_schedule, validate_schedule
from quant_benchmark_lab.quality.scorers import score
from quant_benchmark_lab.reporting.aggregate import aggregate_loads, stats
from quant_benchmark_lab.schema import Metric, TokenUsage

ROOT = Path(__file__).resolve().parents[2]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/demo.yaml")

    def test_all_experiment_configs_validate(self):
        for name in ("quantization", "engine", "model"):
            c = load_config(ROOT / f"configs/experiments/{name}.yaml")
            self.assertFalse(c.synthetic)

    def test_unknown_field_rejected(self):
        raw = self.config.model_dump()
        raw["concurrency"] = 8
        with self.assertRaises(ValidationError):
            CampaignConfig.model_validate(raw)

    def test_real_ollama_rejects_ineffective_microbatch_setting(self):
        raw = self.config.model_dump()
        raw.update(synthetic=False, stage="pilot")
        raw["runtime"].update(batch_size=128, microbatch_size=64)
        with self.assertRaisesRegex(ValidationError, "sizes must match"):
            CampaignConfig.model_validate(raw)
        raw["experiment"] = "quantization"
        raw["cells"][0]["engine"] = "llamacpp"
        raw["cells"][0]["model"]["quant"] = "Q8_0"
        # Standalone llama-server supports a separate physical microbatch.
        self.assertEqual(CampaignConfig.model_validate(raw).runtime.microbatch_size, 64)

    def test_engine_gguf_mismatch_rejected(self):
        raw = self.config.model_dump()
        raw["cells"][1]["model"]["quant"] = "Q8_0"
        with self.assertRaises(ValidationError):
            CampaignConfig.model_validate(raw)

    def test_official_needs_frozen_gpu(self):
        raw = self.config.model_dump()
        raw.update(stage="official", synthetic=False)
        with self.assertRaises(ValidationError):
            CampaignConfig.model_validate(raw)

    def test_schedule_counts_and_determinism(self):
        a, b = make_schedule(self.config), make_schedule(self.config)
        self.assertEqual(a, b)
        self.assertEqual(a["planned_trials"], 36)
        self.assertEqual(validate_schedule(a), self.config)

    def test_schedule_tamper_rejected(self):
        value = make_schedule(self.config)
        value["sessions"][0]["trials"][0]["item"]["text"] = "changed"
        with self.assertRaises(ValueError):
            validate_schedule(value)

    def test_warm_repetitions_divide_evenly(self):
        raw = self.config.model_dump()
        raw["protocol"]["warm_repetitions"] = 5
        with self.assertRaises(ValidationError):
            CampaignConfig.model_validate(raw)

    def test_renderer_preserves_unicode_and_braces(self):
        model = self.config.cells[0].model
        output = render(model, "Xin chào {literal}")
        self.assertIn("Xin chào {literal}", output)
        self.assertEqual(text_hash(output), text_hash(render(model, "Xin chào {literal}")))

    def test_prompts_reject_answer_leak(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text(json.dumps({"id": "safe", "text": "Q", "expected": "A"}))
            with self.assertRaises(ValueError):
                read_items(path)


class MetricTests(unittest.TestCase):
    def test_null_requires_reason(self):
        with self.assertRaises(ValidationError):
            Metric(unit="ms", source="test")

    def test_negative_and_nonfinite_rejected(self):
        for x in (-1.0, math.inf, math.nan):
            with self.assertRaises(ValidationError):
                Metric(value=x, unit="ms", source="test")

    def test_ollama_uncached_denominator_and_ns(self):
        usage, metrics = native_metrics(
            "ollama",
            {
                "prompt_eval_count": 100,
                "prompt_eval_cached_count": 90,
                "prompt_eval_duration": 100_000_000,
                "eval_count": 5,
                "eval_duration": 500_000_000,
            },
        )
        self.assertEqual(usage.input_evaluated, 10)
        self.assertEqual(metrics["prefill_tok_s"].value, 100)
        self.assertEqual(usage.output, 5)
        self.assertEqual(usage.decoded, 4)
        self.assertEqual(metrics["decode_tok_s"].value, 8)

    def test_missing_cache_count_not_guessed_zero(self):
        usage, metrics = native_metrics(
            "ollama", {"prompt_eval_count": 100, "prompt_eval_duration": 1_000_000}
        )
        self.assertIsNone(usage.input_evaluated)
        self.assertIsNone(metrics["prefill_tok_s"].value)

    def test_llama_native_counts_and_ms(self):
        usage, metrics = native_metrics(
            "llamacpp",
            {
                "timings": {
                    "prompt_n": 10,
                    "cache_n": 90,
                    "prompt_ms": 100.0,
                    "predicted_n": 5,
                    "predicted_ms": 500.0,
                }
            },
        )
        self.assertEqual(usage.input_total, 100)
        self.assertEqual(metrics["prefill_tok_s"].value, 100)
        self.assertEqual(usage.output, 5)
        self.assertEqual(usage.decoded, 4)
        self.assertEqual(metrics["decode_tok_s"].value, 8)

    def test_decode_steps_match_audited_native_boundary_for_short_outputs(self):
        # server_slot_stats::n_gen_steps excludes the first prefill-sampled token.
        for count in (0, 1, 8):
            finals = {
                "ollama": {"eval_count": count, "eval_duration": 100_000_000},
                "llamacpp": {
                    "tokens_predicted": count,
                    "timings": {"predicted_n": count, "predicted_ms": 100.0},
                },
            }
            for engine, final in finals.items():
                with self.subTest(engine=engine, count=count):
                    usage, metrics = native_metrics(engine, final)
                    self.assertEqual(usage.output, count)
                    self.assertEqual(usage.decoded, max(0, count - 1))
                    self.assertEqual(metrics["decode_tok_s"].value, max(0, count - 1) * 10)

    def test_zero_duration_is_null_rate(self):
        _, metrics = native_metrics(
            "ollama",
            {"prompt_eval_count": 10, "prompt_eval_cached_count": 0, "prompt_eval_duration": 0},
        )
        self.assertIsNone(metrics["prefill_tok_s"].value)

    def test_inconsistent_token_accounting_rejected(self):
        with self.assertRaises(ValidationError):
            TokenUsage(input_total=2, input_cached=3, input_evaluated=0, source="bad")


class QualityTests(unittest.TestCase):
    def test_exact_normalization_is_declared(self):
        answer = {"scorer": "exact", "expected": "red"}
        self.assertEqual(score(" red\n", answer)["score"], 1)
        self.assertEqual(score("RED", answer)["score"], 0)
        self.assertEqual(score("RED", {**answer, "casefold": True})["score"], 1)

    def test_numeric_tolerance_and_scientific_notation(self):
        a = {"scorer": "numeric", "expected": 1000, "absolute_tolerance": 0.1}
        self.assertEqual(score("+1e3", a)["score"], 1)
        self.assertEqual(score("1000.2", a)["score"], 0)
        self.assertEqual(score("1000.1", a)["score"], 1)  # exact decimal tolerance boundary

    def test_numeric_ambiguous_units_nan_rejected(self):
        a = {"scorer": "numeric", "expected": 2}
        for text in ("2 or 3", "2 meters", "NaN", "inf", "The answer is 2", "1e999"):
            self.assertEqual(score(text, a)["score"], 0)

    def test_json_duplicate_keys_extra_fields_types(self):
        a = {"scorer": "json", "expected": {"age": 1}}
        self.assertEqual(score('{"age":1}', a)["score"], 1)
        self.assertEqual(score('{"age":1.0}', a)["score"], 1)
        for text in (
            '{"age":true}',
            '{"age":1,"x":2}',
            '{"age":2,"age":1}',
            '{"age":NaN}',
            '{"age":1e999}',
        ):
            self.assertEqual(score(text, a)["score"], 0)
            json.dumps(score(text, a), allow_nan=False)

    def test_json_numbers_compare_original_decimal_literals(self):
        cases = [
            ('{"n":1e-999}', {"n": 0}, 0),
            ('{"n":9007199254740993.0}', {"n": 9007199254740992}, 0),
            ('{"n":9007199254740993.0}', {"n": 9007199254740993}, 1),
            ('{"n":1.0000000000000000001}', {"n": 1}, 0),
        ]
        for text, expected, result in cases:
            with self.subTest(output=text, expected=expected):
                scored = score(text, {"scorer": "json", "expected": expected})
                self.assertEqual(scored["score"], result)
                json.dumps(scored, allow_nan=False)

    def test_dataset_has_60_items_and_balanced_categories(self):
        rows = read_items(ROOT / "data/quality/items.jsonl")
        self.assertEqual(len(rows), 60)
        for category in ("exact", "numeric", "structured"):
            self.assertEqual(sum(r["category"] == category for r in rows), 20)


class GateAndStatisticsTests(unittest.TestCase):
    def test_sample_std_and_missing_values(self):
        self.assertEqual(stats([1.0, 3.0])["mean"], 2)
        self.assertAlmostEqual(stats([1.0, 3.0])["std"], math.sqrt(2))
        self.assertIsNone(stats([1.0])["std"])
        self.assertIsNone(stats([])["mean"])

    def test_equivalence_requires_cache_and_prompt_token_parity(self):
        config = load_config(ROOT / "configs/demo.yaml")
        rows = [
            {
                "cell_id": c.id,
                "prompt_id": "p",
                "status": "ok",
                "cache": 0,
                "input": 10,
                "prompt_hash": "same",
            }
            for c in config.cells
        ]
        self.assertTrue(equivalence(rows, config)["passed"])
        rows[0]["cache"] = 3
        self.assertFalse(equivalence(rows, config)["passed"])
        rows[0]["cache"] = None
        self.assertFalse(equivalence(rows, config)["passed"])

    def test_equivalence_missing_engine_fails(self):
        config = load_config(ROOT / "configs/demo.yaml")
        self.assertFalse(
            equivalence(
                [{"cell_id": "ollama", "prompt_id": "p", "status": "ok", "cache": 0, "input": 10}],
                config,
            )["passed"]
        )

    def test_cold_and_warm_cache_counts_are_separate_strata(self):
        config = load_config(ROOT / "configs/demo.yaml")
        config.protocol.allowed_bos_cache_tokens = 1
        rows = [
            {
                "cell_id": c.id,
                "prompt_id": "p",
                "run_mode": mode,
                "status": "ok",
                "cache": cached,
                "input": 10,
                "prompt_hash": "same",
            }
            for c in config.cells
            for mode, cached in (("cold", 0), ("warm", 1))
        ]
        self.assertTrue(equivalence(rows, config)["passed"])

    def test_duplicate_load_observations_rejected(self):
        load = {"observation": {"id": "l"}, "cell_id": "c", "run_mode": "warm", "metrics": {}}
        with self.assertRaises(ValueError):
            aggregate_loads([load, copy.deepcopy(load)])
