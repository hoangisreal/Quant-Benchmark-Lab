"""Native metric mappings preserve denominator and measurement boundary."""

from .schema import Metric, TokenUsage


def measured(value: float | None, unit: str, source: str, reason: str = "unavailable") -> Metric:
    return Metric(value=value, unit=unit, source=source, reason=reason if value is None else None)


def nonnegative_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def duration(value, divisor: float, source: str) -> Metric:
    valid = isinstance(value, (float, int)) and not isinstance(value, bool) and value >= 0
    return measured(float(value) / divisor if valid else None, "ms", source)


def rate(count: int | None, milliseconds: Metric) -> Metric:
    value = None
    if count is not None and milliseconds.value is not None and milliseconds.value > 0:
        value = count * 1000.0 / milliseconds.value
    return measured(value, "tok/s", milliseconds.source, "missing count or nonpositive duration")


def native_metrics(engine: str, final: dict) -> tuple[TokenUsage, dict[str, Metric]]:
    if engine == "ollama":
        total = nonnegative_int(final.get("prompt_eval_count"))
        cached = nonnegative_int(final.get("prompt_eval_cached_count"))
        output = nonnegative_int(final.get("eval_count"))
        evaluated = total - cached if total is not None and cached is not None else None
        prefill = duration(final.get("prompt_eval_duration"), 1e6, "ollama.prompt_eval_duration/ns")
        decode = duration(final.get("eval_duration"), 1e6, "ollama.eval_duration/ns")
        load = duration(final.get("load_duration"), 1e6, "ollama.load_duration/ns")
        source = "ollama final usage; eval_count includes engine-defined EOS/first-token policy"
    elif engine == "llamacpp":
        t = final.get("timings", {})
        evaluated = nonnegative_int(t.get("prompt_n"))
        cached = nonnegative_int(t.get("cache_n"))
        total = evaluated + cached if evaluated is not None and cached is not None else None
        output = nonnegative_int(final.get("tokens_predicted", t.get("predicted_n")))
        prefill = duration(t.get("prompt_ms"), 1, "llamacpp.timings.prompt_ms")
        decode = duration(t.get("predicted_ms"), 1, "llamacpp.timings.predicted_ms")
        load = measured(None, "ms", "llamacpp", "native load timer not exposed by completion")
        source = "llamacpp timings cache_n + prompt_n; predicted_n is engine-defined"
    else:
        raise ValueError(f"unknown metrics adapter {engine}")
    decoded = output if engine == "ollama" else nonnegative_int(final.get("timings", {}).get("predicted_n"))
    usage = TokenUsage(input_total=total, input_cached=cached, input_evaluated=evaluated,
                       output=output, decoded=decoded, source=source,
                       reason="missing native token fields" if total is None or output is None else None)
    return usage, {"prefill_ms": prefill, "decode_ms": decode,
                   "prefill_tok_s": rate(evaluated, prefill), "decode_tok_s": rate(decoded, decode),
                   "request_native_load_ms": load}
