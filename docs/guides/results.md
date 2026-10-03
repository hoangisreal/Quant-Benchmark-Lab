# Read, preserve and regenerate results

[Documentation home](../../README.md) · Previous: [run a campaign](reproduction.md)

## Report files

| File | Meaning |
|---|---|
| `report.md` | Human-readable metrics, quality and figure links |
| `metrics.csv` | Per-cell, prompt, cold/warm metric statistics and valid/planned/failure counts |
| `loads.csv` | Unique load observations, not duplicated for every warm request |
| `sessions.csv` | Session peaks, sample counts and raw trace paths |
| `paired_engine_differences.csv` | Paired engine differences only when equivalence is marked verified |
| `failures.csv` | Invalid, failed or excluded attempts with reasons |
| `quality_items.csv` | Individual objective scores when the schedule includes quality |
| `*.svg`, `*.png` | Figures with units and synthetic labels when applicable |
| `manifest.json` | Report file hashes, campaign/config identity and quality scorer/input identity |

A report can be generated for an incomplete campaign; successful generation alone does
not prove completion or hardware validity. Check raw audit output and preflight evidence.
An empty paired-comparison file is not proof that engines perform equally.

## Interpret the numbers

Compare quantizations within E1, engines within E2, and models within E3. Keep cold/warm
and prompt groups separate. TTFT measures first streamed content at the client; prefill
and decode rates use native token counters and durations. Different models tokenize the
same text differently, so their token rates are not identical-work comparisons.

Read E2E latency together with actual output lengths: natural EOS can produce unequal work.
Read sampled device peak VRAM alongside idle, loaded and attributable process memory.
Session peaks also include warmup and eviction; request peaks have a narrower boundary.

Missing metrics are empty CSV cells or N/A, not zero. Raw records retain reasons and native
payloads. Standard deviation uses `ddof=1` and is unavailable for one observation. Analysis
selects the first valid attempt; failed initial attempts remain visible. Quality macro
accuracy averages task-category accuracies, with unresolved questions scoring zero in the
completion-inclusive view. See [methodology](../reference/methodology.md) and [limitations](../reference/limitations.md).

## Regenerate without inference

With the Python environment installed and a complete raw campaign available, run:

```bash
uv run --frozen qbl audit --campaign results/raw/engine-official
uv run --frozen qbl evaluate --campaign results/raw/engine-official
uv run --frozen qbl report --campaign results/raw/engine-official --output reports/engine-regenerated
```

Replace the example campaign path with your actual directory and use a fresh report
directory. This workflow uses the raw snapshots; it does not need model weights or a GPU.
`report` recomputes quality from verified records and answer snapshots instead of trusting
an older `quality.json` export.

Scorer 1.0.1 compares original JSON decimal literals before floating-point rounding.
The normalized display uses ordinary JSON numbers; `output.txt` remains the exact original
answer. Report manifests record scorer version/source and quality-record identity.
To reproduce an older scoring implementation exactly, use its original repository revision;
regeneration with a newer scorer is a new analysis, with an explicit identity.

## Preserve a campaign

Keep the **entire** raw directory, including hidden files, immutable metadata, checksums,
the four JSONL journals, their `journal/` write-ahead entries, per-attempt files, logs and
session traces. A report alone cannot reproduce scores or establish provenance. Also keep
the repository revision, `uv.lock`, reviewed source evidence and artifact creation manifests.

Treat raw data as immutable. Never edit it to remove failures, repair checksums or redact
paths in place. Review prompts, outputs, environment paths and logs before sharing. Publish
a separate clearly labeled redacted copy if needed; it is not the original verifiable raw
campaign. No public raw-results download or measured report is currently provided.

## Storage compatibility

Current campaigns use storage version 2. Earlier campaigns lack the required integrity
history and cannot be audited, resumed or reported with the current reader. Keep them for
historical inspection and start a new campaign directory; no automatic migration fabricates
missing checksums. Older preflight certificates also need replacement after dataset-binding
or transport changes. A new measurement configuration always needs a fresh pilot.
