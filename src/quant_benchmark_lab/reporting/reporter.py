"""Reports are reproducible offline and do not infer winners from invalid probes."""

import csv
import json
from pathlib import Path

from ..quality.evaluator import evaluate
from ..storage import ResultStore, read_jsonl
from ..utils import atomic_json, file_hash
from .aggregate import aggregate, aggregate_loads, paired_engine_differences
from .plots import charts, tradeoff_charts


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def report(root: Path, output: Path) -> dict:
    store = ResultStore(root)
    store.verify()
    manifest = json.loads((root / "manifest.json").read_text())
    schedule = json.loads((root / "schedule.json").read_text())
    records = store.records()
    if any(r.synthetic != manifest["synthetic"] for r in records):
        raise ValueError("synthetic contamination in campaign")
    rows = aggregate(records, schedule)
    loads = aggregate_loads(read_jsonl(root / "loads.jsonl"))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "loads.csv", loads)
    write_csv(output / "paired_engine_differences.csv", paired_engine_differences(records, schedule)
              if manifest["equivalence_verified"] else [])
    session_rows = read_jsonl(root / "sessions.jsonl")
    write_csv(output / "sessions.csv", [{"session_id": r["session_id"], "cell_id": r["cell_id"],
        "run_mode": r["run_mode"], "peak_session_vram_bytes": r["peak_session_vram_bytes"]["value"],
        "n_samples": r["n_samples"], "trace_path": r["trace_path"]} for r in session_rows])
    failures = [{"trial_id": r.trial_id, "attempt_id": r.attempt_id, "status": r.status,
                 "error": r.error, "exclusions": "; ".join(r.exclusions)} for r in records if r.status != "ok"]
    write_csv(output / "failures.csv", failures)
    files = charts(rows, output, manifest["synthetic"])
    lines = ["# Quant Benchmark Lab report", "",
        "**SYNTHETIC DATA — harness demonstration only; no model performance claims.**" if manifest["synthetic"]
        else "Measured data; interpret only within the recorded protocol and hardware environment.", "",
        f"Campaign: `{manifest['campaign_id']}`; experiment: `{schedule['config']['experiment']}`.", "",
        ("Hardware equivalence: unverified; synthetic backend contracts only." if manifest["synthetic"]
         else f"Equivalence verified: `{manifest['equivalence_verified']}`. "
         "No engine ranking is asserted when this gate is false."), "",
        "TTFT is client-observed first-content latency. Cold means model-cold; OS page cache is uncontrolled. "
        "Peak VRAM is sampled device memory. Native load boundaries differ between engines.", "",
        "[Request metrics](metrics.csv) · [Unique session loads](loads.csv) · [Failed/excluded attempts](failures.csv)", "",
        "Mean, median and sample standard deviation (ddof=1) are separate. Missing metrics are empty CSV cells; "
        "no values are imputed. Groups preserve prompt, engine configuration and cold/warm mode.", "",
        "| Cell | Mode | Prompt | Metric | Median | Unit | Valid n | Planned n | Unresolved |", 
        "|---|---|---|---|---:|---|---:|---:|---:|"]
    for row in rows:
        if row["metric"] in {"ttft_stream_ms", "decode_tok_s", "prefill_tok_s", "e2e_request_ms", "peak_request_vram_bytes", "output_tokens"}:
            value = f"{row['median']:.3f}" if row["median"] is not None else "N/A"
            lines.append(f"| {row['cell_id']} | {row['run_mode']} | {row['prompt_id']} | {row['metric']} | "
                         f"{value} | {row['unit']} | {row['n']} | {row['n_planned']} | {row['n_unresolved']} |")
    if any(s["mode"] == "quality" for s in schedule["sessions"]):
        # Always derive scores from the verified snapshot; quality.json is an export/cache only.
        quality = evaluate(root, persist=False)
        write_csv(output / "quality_items.csv", quality["items"])
        files += tradeoff_charts(rows, quality, output, manifest["synthetic"])
        lines += ["", "## Objective quality", "", "Macro-average of category accuracies; unresolved trials score zero.", ""]
        for summary in quality["summary"]:
            lines.append(f"- `{summary['cell_id']}`: {summary['macro_accuracy']:.3f}, n={summary['n_items']} items.")
    lines += ["", "## Figures", ""] + [f"![{name}]({name})" for name in files if name.endswith(".svg")]
    lines += ["", "SVG figures are always available. PNG export requires Matplotlib. "
              "Raw campaign contains schedule, resolved settings, environment, native payloads, prompts and GPU traces.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    index = {"campaign_id": manifest["campaign_id"], "synthetic": manifest["synthetic"],
             "config_hash": manifest["config_hash"], "files": {p.name: file_hash(p) for p in output.iterdir()
                                                                 if p.is_file() and p.name != "manifest.json"}}
    atomic_json(output / "manifest.json", index)
    return index
