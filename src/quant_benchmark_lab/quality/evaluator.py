"""Select the first valid completed attempt, never the best answer."""

import json
import statistics
from pathlib import Path

from ..storage import ResultStore
from ..utils import atomic_json, digest, file_hash
from .dataset import load_answers
from .scorers import SCORER_VERSION, score


def evaluate(root: Path, *, persist: bool = True) -> dict:
    store = ResultStore(root)
    store.verify()
    schedule = json.loads((root / "schedule.json").read_text())
    answer_path = root / "answers.jsonl"
    if file_hash(answer_path) != schedule["answers_sha256"]:
        raise ValueError("answer key changed since scheduling")
    answers = load_answers(answer_path)
    records = [r for r in store.records() if r.phase == "quality"]
    by_trial = {}
    for r in records:
        by_trial.setdefault(r.trial_id, []).append(r)
    items = []
    for session in schedule["sessions"]:
        if session["mode"] != "quality":
            continue
        for trial in session["trials"]:
            attempts = by_trial.get(trial["id"], [])
            selected = next((r for r in attempts if r.status == "ok"), None)
            item = trial["item"]
            answer = answers[item["id"]]
            if answer["scorer"] != item["scorer"]:
                raise ValueError("dataset scorer and answer scorer differ")
            result = score(selected.output, answer) if selected else {
                "score": 0.0, "normalized": None, "reason": "unresolved failure or missing trial"}
            items.append({"cell_id": session["cell_id"], "item_id": item["id"],
                          "trial_id": trial["id"], "attempt_id": selected.attempt_id if selected else None,
                          "category": item["category"], "answered": selected is not None,
                          "truncated": selected.stop_reason in {"length", "limit"} if selected else False,
                          "output": selected.output if selected else None, **result})
    summary = []
    for cell in sorted({i["cell_id"] for i in items}):
        rows = [i for i in items if i["cell_id"] == cell]
        categories = {}
        for category in sorted({i["category"] for i in rows}):
            group = [i for i in rows if i["category"] == category]
            answered = [i["score"] for i in group if i["answered"]]
            categories[category] = {"n": len(group), "answered": len(answered),
                "completion_inclusive_accuracy": statistics.mean(i["score"] for i in group),
                "answered_only_accuracy": statistics.mean(answered) if answered else None}
        summary.append({"cell_id": cell, "n_items": len(rows), "categories": categories,
                        "macro_accuracy": statistics.mean(c["completion_inclusive_accuracy"]
                                                          for c in categories.values())})
    manifest = json.loads((root / "manifest.json").read_text())
    identity = {"campaign_id": manifest["campaign_id"], "schedule_sha256": file_hash(root / "schedule.json"),
                "answers_sha256": file_hash(answer_path), "scorer_version": SCORER_VERSION,
                "scorer_sha256": file_hash(Path(__file__).with_name("scorers.py")),
                "quality_records_sha256": digest([r.model_dump() for r in records])}
    result = {"identity": identity, "scorer_version": SCORER_VERSION, "synthetic": manifest["synthetic"],
              "answers_sha256": schedule["answers_sha256"], "items": items, "summary": summary}
    if persist:
        atomic_json(root / "quality.json", result)
    return result
