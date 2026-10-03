"""Deterministic balanced scheduling and fail-closed comparison gates."""

import random
from copy import deepcopy
from pathlib import Path

from .config import CampaignConfig
from .prompts import eviction_text, read_items, render, text_hash, workload_hash
from .quality.dataset import load_answers
from .utils import digest, file_hash


def dataset_identity(config: CampaignConfig) -> dict:
    return {
        key: file_hash(Path(getattr(config, key)))
        for key in ("performance_path", "quality_path", "answers_path")
    }


def measurement_identity(config: CampaignConfig) -> str:
    value = config.model_dump()
    value.pop("stage")
    value.pop("preflight_path")
    value["protocol"].pop("state")
    return digest({"config": value, "datasets": dataset_identity(config)})


def make_schedule(config: CampaignConfig, suite: str = "both") -> dict:
    perf = read_items(Path(config.performance_path))
    quality = read_items(Path(config.quality_path))
    answers = load_answers(Path(config.answers_path))
    if set(answers) != {item["id"] for item in quality}:
        raise ValueError("quality questions and answer key must have identical IDs")
    for item in quality:
        if not isinstance(item.get("category"), str) or not item["category"]:
            raise ValueError("quality item requires a nonempty category")
        if item.get("scorer") != answers[item["id"]]["scorer"]:
            raise ValueError("quality question and answer scorer differ")
    rng = random.Random(config.protocol.schedule_seed)
    sessions = []

    def block(cells, mode, prompt_items, rep, repetitions):
        shuffled = list(cells)
        rng.shuffle(shuffled)
        for cell in shuffled:
            prompt_tag = prompt_items[0]["id"] if mode != "quality" else "suite"
            base = f"{mode}-{prompt_tag}-b{rep}"
            session_id = f"{base}-{cell.id}"
            trials = []
            for n in range(repetitions):
                for item in prompt_items:
                    trials.append(
                        {
                            "id": f"{session_id}-{item['id']}-r{n}",
                            "item": item,
                            "repetition": rep * repetitions + n,
                        }
                    )
            sessions.append(
                {
                    "id": session_id,
                    "block_id": base,
                    "cell_id": cell.id,
                    "mode": mode,
                    "trials": trials,
                }
            )

    if suite in {"both", "performance"}:
        for item in perf:
            for rep in range(config.protocol.cold_repetitions):
                block(config.cells, "cold", [item], rep, 1)
        for item in perf:
            for rep in range(config.protocol.warm_sessions):
                block(
                    config.cells,
                    "warm",
                    [item],
                    rep,
                    config.protocol.warm_repetitions // config.protocol.warm_sessions,
                )
    if suite in {"both", "quality"}:
        # Same seeded order of items in all cells; model order randomized independently.
        rng.shuffle(quality)
        block(config.cells, "quality", quality, 0, 1)
    if suite not in {"both", "performance", "quality"}:
        raise ValueError("unknown suite")
    return {
        "schema_version": 1,
        "config": config.model_dump(),
        "config_hash": digest(config.model_dump()),
        "measurement_hash": measurement_identity(config),
        "workload_hash": workload_hash(perf, read_items(Path(config.quality_path))),
        "answers_sha256": file_hash(Path(config.answers_path)),
        "dataset_hashes": dataset_identity(config),
        "suite": suite,
        "sessions": sessions,
        "planned_trials": sum(len(s["trials"]) for s in sessions),
    }


def validate_schedule(schedule: dict) -> CampaignConfig:
    config = CampaignConfig.model_validate(schedule["config"])
    if digest(config.model_dump()) != schedule["config_hash"]:
        raise ValueError("schedule config checksum mismatch")
    regenerated = make_schedule(config, schedule["suite"])
    if digest(regenerated) != digest(schedule):
        raise ValueError("schedule/dataset/answer key changed; create a new campaign")
    return config


def equivalence(observations: list[dict], config: CampaignConfig) -> dict:
    failures = []
    for row in observations:
        if row.get("status") != "ok":
            failures.append(f"{row.get('cell_id')}: invalid probe")
        if row.get("cache") is None or row["cache"] > config.protocol.allowed_bos_cache_tokens:
            failures.append(f"{row.get('cell_id')}: substantive/unknown prefix cache")
        if row.get("input") is None:
            failures.append(f"{row.get('cell_id')}: missing input count")
        if config.deployment == "gpu" and row.get("full_offload") is not True:
            failures.append(f"{row.get('cell_id')}: full offload not established")
    if config.experiment == "engine":
        groups = {}
        for row in observations:
            groups.setdefault((row["prompt_id"], row.get("run_mode", "probe")), []).append(row)
        for prompt, rows in groups.items():
            if {r["cell_id"] for r in rows} != {c.id for c in config.cells}:
                failures.append(f"{prompt}: engine probe missing")
            if len({(r.get("prompt_hash"), r.get("input"), r.get("cache")) for r in rows}) != 1:
                failures.append(f"{prompt}: prompt/token/cache parity mismatch")
    return {
        "passed": bool(observations) and not failures,
        "failures": failures,
        "observations": deepcopy(observations),
    }


def validate_certificate(config: CampaignConfig, evidence: dict, schedule: dict | None = None):
    if evidence.get("synthetic") or evidence.get("passed") is not True:
        raise ValueError("official run requires a genuine passing preflight")
    if evidence.get("measurement_hash") != measurement_identity(config):
        raise ValueError("preflight measurement/dataset mismatch")
    if evidence.get("dataset_hashes") != dataset_identity(config):
        raise ValueError("preflight dataset identity missing or changed")
    schedule = schedule or make_schedule(config)
    cells = {c.id: c for c in config.cells}
    for session in schedule["sessions"]:
        cell = cells[session["cell_id"]]
        for trial in session["trials"]:
            prompt = render(cell.model, trial["item"]["text"])
            counts = evidence.get("input_counts", {}).get(cell.model.sha256, {})
            texts = [prompt] if session["mode"] == "cold" else [prompt, eviction_text(prompt)]
            if any(
                type(counts.get(text_hash(t))) is not int or counts[text_hash(t)] < 0 for t in texts
            ):
                raise ValueError(f"preflight missing reference tokens: {trial['id']}")
            cached = (
                evidence.get("cache_counts", {})
                .get(cell.id, {})
                .get(session["mode"], {})
                .get(text_hash(prompt))
            )
            if (
                type(cached) is not int
                or not 0 <= cached <= config.protocol.allowed_bos_cache_tokens
            ):
                raise ValueError(f"preflight missing cache evidence: {trial['id']}")
