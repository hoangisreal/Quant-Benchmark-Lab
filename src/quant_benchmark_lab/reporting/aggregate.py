"""No cross-prompt/model/cold-warm averaging or duplicate session loads."""

import statistics

from ..utils import digest


def stats(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def select_attempts(records):
    groups = {}
    for record in records:
        groups.setdefault(record.trial_id, []).append(record)
    return {key: next((r for r in rows if r.status == "ok"), None) for key, rows in groups.items()}


def aggregate(records, schedule: dict) -> list[dict]:
    flags = {r.synthetic for r in records}
    if len(flags) > 1:
        raise ValueError("cannot mix synthetic and real records")
    relevant = [r for r in records if r.phase == "measurement"]
    selected = select_attempts(relevant)
    result = []
    groups = {}
    for session in schedule["sessions"]:
        if session["mode"] == "quality":
            continue
        for trial in session["trials"]:
            key = (session["cell_id"], session["mode"], trial["item"]["id"])
            groups.setdefault(key, []).append(trial["id"])
    for (cell, mode, prompt), ids in sorted(groups.items()):
        attempts = [r for r in relevant if r.trial_id in ids]
        valid = [selected[i] for i in ids if selected.get(i) is not None]
        if len({digest(r.settings) for r in valid}) > 1:
            raise ValueError("effective requested settings changed within a group")
        names = sorted({n for r in valid for n in r.metrics})
        for metric in names or ["e2e_request_ms"]:
            values = [r.metrics[metric].value for r in valid
                      if metric in r.metrics and r.metrics[metric].value is not None]
            units = {r.metrics[metric].unit for r in valid if metric in r.metrics}
            if len(units) > 1:
                raise ValueError("mixed metric units")
            first_failed = 0
            for trial_id in ids:
                first = next((r for r in attempts if r.trial_id == trial_id), None)
                first_failed += first is not None and first.status != "ok"
            result.append({"cell_id": cell, "run_mode": mode, "prompt_id": prompt,
                "metric": metric, "unit": next(iter(units), "ms"),
                "n_planned": len(ids), "n_attempts": len(attempts), "n_valid_trials": len(valid),
                "n_unresolved": len(ids) - len(valid), "n_initial_failures": first_failed,
                "n_missing_metric": len(valid) - len(values), **stats(values)})
        for field in ("input_total", "input_cached", "input_evaluated", "output"):
            values = [float(getattr(r.usage, field)) for r in valid if getattr(r.usage, field) is not None]
            result.append({"cell_id": cell, "run_mode": mode, "prompt_id": prompt,
                "metric": field + "_tokens", "unit": "tokens", "n_planned": len(ids),
                "n_attempts": len(attempts), "n_valid_trials": len(valid),
                "n_unresolved": len(ids) - len(valid), "n_initial_failures": first_failed,
                "n_missing_metric": len(valid) - len(values), **stats(values)})
    return result


def aggregate_loads(loads: list[dict]) -> list[dict]:
    seen, groups = set(), {}
    for load in loads:
        identity = load["observation"]["id"]
        if identity in seen:
            raise ValueError("duplicate load event")
        seen.add(identity)
        groups.setdefault((load["cell_id"], load["run_mode"]), []).append(load)
    result = []
    for (cell, mode), rows in sorted(groups.items()):
        for name in sorted({n for r in rows for n in r["metrics"]}):
            values = [r["metrics"][name]["value"] for r in rows if r["metrics"][name]["value"] is not None]
            result.append({"cell_id": cell, "run_mode": mode, "metric": name,
                           "unit": rows[0]["metrics"][name]["unit"], "n_sessions": len(rows), **stats(values)})
    return result


def paired_engine_differences(records, schedule: dict) -> list[dict]:
    if schedule["config"]["experiment"] != "engine":
        return []
    selected = select_attempts([r for r in records if r.phase == "measurement"])
    cells = [c["id"] for c in schedule["config"]["cells"]]
    pairs = {}
    for session in schedule["sessions"]:
        if session["mode"] == "quality":
            continue
        for trial in session["trials"]:
            key = (session["block_id"], session["mode"], trial["item"]["id"], trial["repetition"])
            pairs.setdefault(key, {})[session["cell_id"]] = selected.get(trial["id"])
    result = []
    for (block, mode, prompt, repetition), pair in sorted(pairs.items()):
        a, b = pair.get(cells[0]), pair.get(cells[1])
        if a is None or b is None:
            continue
        if (a.prompt_hash, a.usage.input_total, a.usage.input_cached) != (
                b.prompt_hash, b.usage.input_total, b.usage.input_cached):
            continue
        for name in ("ttft_stream_ms", "e2e_request_ms", "prefill_tok_s", "decode_tok_s"):
            left, right = a.metrics.get(name), b.metrics.get(name)
            if left and right and left.value is not None and right.value is not None and left.unit == right.unit:
                result.append({"block_id": block, "run_mode": mode, "prompt_id": prompt,
                    "repetition": repetition, "metric": name, "unit": left.unit,
                    "a_cell": cells[0], "b_cell": cells[1], "delta_b_minus_a": right.value - left.value,
                    "a_output_tokens": a.usage.output, "b_output_tokens": b.usage.output})
    return result
