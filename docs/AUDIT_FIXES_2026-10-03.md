# Audit fixes — 2026-10-03

All nine findings in [the initial audit](AUDIT_2026-10-03.md) now have implementation
changes and offline regression coverage. This establishes harness behavior under tested
conditions; real engine compatibility and GPU performance still require the lab host.

| Finding | Change | Regression coverage |
|---|---|---|
| A01 | Both engines launch with a controlled environment; its effective values enter environment identity. llama.cpp KV offload/speculation are explicit. | Ambient overrides are omitted from actual subprocess env; backend overrides and flags are checked. |
| A02 | Preflight identity includes content hashes for performance, quality and answers. Certificates cover each cell/prompt/run mode, including quality. Official runs require full token/cache coverage before loading. | Edits to each dataset and missing input/cold/warm/quality evidence are rejected. |
| A03 | Session attempt directories are reserved independently of trial attempts and flushed to disk. Every restart gets a distinct load ID and trace. | Crash before load and between warm trials, then resume/report; previous traces remain unchanged. |
| A04 | Storage v2 requires metadata inventory and checksummed write-ahead entries for all four journals, plus session-trace checksums and load references. | Edited/deleted journals, missing entries/metadata and changed traces fail verification; empty paths fail audit. |
| A05 | Received events/output are persisted even if GPU sampling fails. The attempt is invalid with a missing VRAM reason. | Sensor loss at terminal response retains output and raw events and produces verifiable failure artifacts. |
| A06 | Reports recompute quality from verified raw records and answer snapshots. Exported quality includes input/scorer identity. | Evaluate partial campaign, resume, modify stale quality export, report: all completed answers are used and modified scores are ignored. |
| A07 | Reusable async HTTP I/O behind the sequential backend API enforces absolute budgets for headers and byte reads. Cancellation closes the response. | Silent body/header, heartbeat-only, metadata-only, stall after first content, early consumer close, fragmented Unicode and repeated requests. |
| A08 | Recovery applies to runs, loads, sessions and control journals. Committed write-ahead entries repair interrupted appends; torn tails are quarantined. | Partial JSON/UTF-8 and truncated committed appends recover for every journal; report works afterward. |
| A09 | Reused GGUF creation manifests remain byte-for-byte unchanged. Converter/tokenizer/source/quant/parent lineage must match. | Package-version change preserves provenance; converter/parent change fails without rewriting catalog/index. |

## Compatibility and operation

- New campaigns use **storage version 2**. Pre-fix campaigns lack the required write-ahead
  integrity history and are rejected by current audit/report/resume. Keep their original
  artifacts for historical inspection and create a new campaign directory; checksums are
  not retroactively invented for them.
- Certificates created before these changes are invalid. Rerun preflight after pinning the
  toolchain and preparing the workloads. It now includes quality probes and dataset hashes.
- Ambient engine flags are ignored. Express supported settings in repository configuration
  and reviewed backend options. The effective allowlisted host environment is captured.
- HTTP transport implementation changed, so streaming latency and monitor overhead require
  a fresh hardware pilot. Client-observed TTFT remains distinct from native engine timing.
- `qbl report` derives quality directly from raw data. `qbl evaluate` is still available to
  export a separate `quality.json`. `qbl audit` reports trial completion independently of
  file integrity; it does not claim hardware equivalence.
- Journal snapshots commit before JSONL materialization. Recovery handles process crashes
  after that commit. Missing initial metadata or unrelated corrupt data fails closed.

## Validation

- Full suite: **73 tests, 71 passed, 2 real-engine/GPU tests skipped**.
- New coverage: 18 regression tests across campaign integrity/resume, identity/certificates,
  HTTP deadlines, and model provenance, plus updated adapter contracts.
- Initial validation used cached packages. The user subsequently generated `uv.lock` and
  completed frozen dev/prepare installation; `uv lock --check --offline` now passes.
- Ruff import/UTC fixes and deadline-test loop-variable bindings were applied. `ruff check .`
  passes; reports/results and model/vendor artifact directories are explicitly excluded.
- The suite was rerun with installed dependencies: 73 tests, 71 passed, 2 hardware tests skipped.
  Remote CI and real hardware integration remain unverified.
- No real engine build, GGUF conversion, model inference, GPU pilot or official benchmark was run.

Run the checks with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The refreshed synthetic demonstration uses `results/raw/demo-audit-fixed` and
`reports/demo-audit-fixed`. Its measurements are fixtures, not LLM/GPU results.
