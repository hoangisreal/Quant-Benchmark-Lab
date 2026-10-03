"""Real hardware gate and conservative reference-tokenizer/cache probes."""

import statistics
from pathlib import Path

from .artifacts import verify_artifacts
from .backends.base import Request
from .backends.llamacpp import LlamaCppBackend
from .config import CampaignConfig, Cell
from .environment import capture_environment, environment_identity
from .metrics import native_metrics
from .monitoring.gpu import GPULock, GPUMonitor, wait_idle
from .prompts import eviction_text, read_items, render, text_hash
from .protocol import dataset_identity, equivalence, make_schedule, measurement_identity
from .runner import BenchmarkRunner, make_backend
from .storage import ResultStore, read_jsonl
from .utils import Clock, atomic_json


def reference_counts(config: CampaignConfig, work: Path) -> dict:
    """CPU tokenizer probes do not contribute any benchmark timing observations."""
    counts, token_evidence = {}, {}
    items = read_items(Path(config.performance_path)) + read_items(Path(config.quality_path))
    cpu_config = config.model_copy(update={"deployment": "cpu", "stage": "pilot"})
    for model in {c.model.sha256: c.model for c in config.cells}.values():
        cell = Cell(id="reference", engine="llamacpp", model=model)
        backend = LlamaCppBackend(cell, cpu_config, work / model.sha256, Clock())
        counts[model.sha256], token_evidence[model.sha256] = {}, {}
        try:
            backend.load("reference-tokenizer-only")
            for item in items:
                prompt = render(model, item["text"])
                for text in (prompt, eviction_text(prompt)):
                    response = backend.client.post(
                        "/tokenize",
                        json={"content": text, "add_special": True, "parse_special": True},
                    )
                    response.raise_for_status()
                    ids = response.json()["tokens"]
                    key = text_hash(text)
                    counts[model.sha256][key] = len(ids)
                    token_evidence[model.sha256][key] = ids
                p, e = (counts[model.sha256][text_hash(t)] for t in (prompt, eviction_text(prompt)))
                budget = max(
                    config.generation.performance_max_tokens, config.generation.quality_max_tokens
                )
                if p + budget > config.runtime.context_size or e + 1 > config.runtime.context_size:
                    raise ValueError(
                        f"{model.id}/{item['id']}: request or eviction exceeds configured context"
                    )
                if e < p:
                    raise ValueError(
                        "eviction prompt is shorter in native tokens than measured prompt"
                    )
        finally:
            backend.close()
    atomic_json(work / "token_ids.json", token_evidence)
    return counts


def monitor_overhead(config, counts, work):
    cells = {
        cell.id: _monitor_overhead_cell(config, counts, work / cell.id, cell)
        for cell in config.cells
    }
    return {
        "passed": all(result["passed"] for result in cells.values()),
        "overhead_fraction": max(result["overhead_fraction"] for result in cells.values()),
        "threshold": 0.03,
        "cells": cells,
    }


def _monitor_overhead_cell(config, counts, work, cell):
    clock = Clock()
    backend = make_backend(cell, config, work, clock)
    item = read_items(Path(config.performance_path))[0]
    request = Request(
        render(cell.model, item["text"]),
        config.generation.performance_max_tokens,
        config.generation,
        cell.model.stop,
    )
    observations = []
    env = capture_environment(
        config.runtime.ollama_binary, config.runtime.llama_binary, config.runtime.toolchain_lock
    )
    device = next(d for d in env["gpu"]["devices"] if d["index"] == config.runtime.gpu_index)
    lock = GPULock(device["uuid"])
    warmups = []
    try:
        lock.acquire()
        load = backend.load("monitor-overhead-load")
        for _ in range(config.protocol.warmups):
            warmups.append([e.model_dump() for e in backend.stream(request)])
        for pair in range(5):
            for enabled in [False, True] if pair % 2 == 0 else [True, False]:
                monitor = GPUMonitor(
                    clock,
                    config.protocol.sample_interval_ms,
                    config.runtime.gpu_index,
                    enabled=enabled,
                )
                try:
                    monitor.start()
                    list(backend.stream(backend.prepare_cache(request)))
                    clock.sleep(config.protocol.settle_seconds)
                    monitor.mark("request")
                    start = clock.now()
                    events = list(backend.stream(request))
                    end = clock.now()
                    finals = [e for e in events if e.kind == "final"]
                    if len(finals) != 1:
                        raise ValueError("overhead probe has no valid terminal response")
                    usage, _ = native_metrics(cell.engine, finals[0].payload)
                    if (
                        usage.input_cached is None
                        or usage.input_cached > config.protocol.allowed_bos_cache_tokens
                    ):
                        raise ValueError("overhead probe is not cache-equivalent")
                    if usage.input_total != counts[cell.model.sha256][text_hash(request.prompt)]:
                        raise ValueError("overhead probe input token mismatch")
                    observations.append(
                        {
                            "pair": pair,
                            "monitor_enabled": enabled,
                            "e2e_ms": (end - start) / 1e6,
                            "output_tokens": usage.output,
                            "cached_tokens": usage.input_cached,
                            "events": [e.model_dump() for e in events],
                            "gpu_samples": monitor.samples,
                        }
                    )
                finally:
                    monitor.close()
    finally:
        try:
            backend.close()
        finally:
            lock.close()
    if len({(o["cached_tokens"], o["output_tokens"]) for o in observations}) != 1:
        raise ValueError("monitor overhead probes have unmatched cache/output-token work")
    on = statistics.median(r["e2e_ms"] for r in observations if r["monitor_enabled"])
    off = statistics.median(r["e2e_ms"] for r in observations if not r["monitor_enabled"])
    ratio = on / off - 1
    return {
        "passed": ratio <= 0.03,
        "overhead_fraction": ratio,
        "observations": observations,
        "threshold": 0.03,
        "load_observation": load.model_dump(),
        "warmup_events": warmups,
    }


def preflight(config: CampaignConfig, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = capture_environment(
        config.runtime.ollama_binary, config.runtime.llama_binary, config.runtime.toolchain_lock
    )
    result = {
        "schema_version": 1,
        "passed": False,
        "synthetic": config.synthetic,
        "measurement_hash": measurement_identity(config),
        "dataset_hashes": dataset_identity(config),
        "environment_hash": environment_identity(env),
        "failures": [],
        "input_counts": {},
    }
    try:
        if config.synthetic:
            raise ValueError("synthetic campaigns cannot produce hardware preflight certificates")
        if config.deployment != "gpu":
            raise ValueError("hardware preflight requires GPU deployment")
        if not env["gpu"]["available"]:
            raise ValueError(f"GPU unavailable: {env['gpu']['reason']}")
        device = next(
            (d for d in env["gpu"]["devices"] if d["index"] == config.runtime.gpu_index), None
        )
        if (
            not device
            or "3050" not in device["name"]
            or not 3.5 * 1024**3 <= device["total_bytes"] <= 4.5 * 1024**3
        ):
            raise ValueError("hardware preflight requires RTX 3050 4GB")
        verify_artifacts(config)
        monitor = GPUMonitor(Clock(), config.protocol.sample_interval_ms, config.runtime.gpu_index)
        try:
            monitor.start()
            wait_idle(monitor, config.protocol, monitor.clock)
        finally:
            monitor.close()
        counts = reference_counts(config, output / "reference")
        result["input_counts"] = counts
        pilot_protocol = config.protocol.model_copy(
            update={
                "state": "draft",
                "cold_repetitions": 2,
                "warm_repetitions": 4,
                "warm_sessions": 2,
            }
        )
        pilot = config.model_copy(
            update={"stage": "pilot", "preflight_path": None, "protocol": pilot_protocol}
        )
        schedule = make_schedule(pilot, "both")
        atomic_json(output / "probe_schedule.json", schedule)
        campaign = output / "pilot-campaign"
        BenchmarkRunner(schedule, campaign, evidence={"input_counts": counts}).run()
        store = ResultStore(campaign)
        store.verify()
        records = [r for r in store.records() if r.phase in {"measurement", "quality"}]
        observations = [
            {
                "cell_id": r.cell_id,
                "prompt_id": r.prompt_id,
                "status": r.status,
                "run_mode": r.run_mode,
                "prompt_hash": r.prompt_hash,
                "input": r.usage.input_total,
                "cache": r.usage.input_cached,
                "full_offload": r.effective.get("full_offload"),
            }
            for r in records
        ]
        gate = equivalence(observations, config)
        result["equivalence"] = gate
        result["cache_counts"] = {}
        for r in records:
            result["cache_counts"].setdefault(r.cell_id, {}).setdefault(r.run_mode, {})[
                r.prompt_hash
            ] = r.usage.input_cached
        result["failures"].extend(gate["failures"])
        for r in records:
            if any(
                r.metrics.get(k) is None or r.metrics[k].value is None
                for k in ("prefill_tok_s", "decode_tok_s", "peak_request_vram_bytes")
            ):
                result["failures"].append(f"{r.trial_id}: required native/VRAM metric missing")
        overhead = monitor_overhead(pilot, counts, output / "overhead")
        result["monitor_overhead"] = overhead
        if not overhead["passed"]:
            result["failures"].append(
                "monitor overhead >3%; recalibrate interval then repeat pilot"
            )
        if config.experiment == "model":
            if len(config.cells) < 3:
                result["failures"].append("model experiment needs at least three models")
            near = []
            for r in records:
                peak = r.metrics.get("peak_request_vram_bytes")
                # Require attributable owned process VRAM in addition to global pressure.
                pids = set(r.effective.get("owned_pids", []))
                samples = read_jsonl(store.safe_path(r.paths["gpu.jsonl"]))
                attributable = [
                    sum(
                        p["used_bytes"]
                        for p in (s.get("processes") or [])
                        if p["pid"] in pids and p["used_bytes"] is not None
                    )
                    for s in samples
                ]
                idle = r.metrics.get("idle_vram_bytes")
                if (
                    peak
                    and peak.value is not None
                    and peak.value >= 0.8 * device["total_bytes"]
                    and attributable
                    and idle
                    and idle.value is not None
                    and max(attributable) >= peak.value - idle.value - 64 * 1024**2
                    and max(attributable) >= 0.8 * (device["total_bytes"] - idle.value)
                ):
                    near.append(r.model_id)
            result["near_limit_models"] = sorted(set(near))
            if not near:
                result["failures"].append("no measured, attributable full-offload near-limit model")
        if config.experiment == "quantization" and {c.model.quant for c in config.cells} != {
            "F16",
            "Q8_0",
            "Q4_K_M",
        }:
            result["failures"].append("quantization baseline must include F16/Q8_0/Q4_K_M")
        result["passed"] = bool(records) and not result["failures"]
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    atomic_json(output / "environment.json", env)
    atomic_json(output / "preflight.json", result)
    return result
