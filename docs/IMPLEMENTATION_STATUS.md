# Implementation status

Updated: 2026-10-03, after audit fixes and dependency/lint validation. Status describes code and observed validation, not assumed hardware behavior.

All nine audit findings A01–A09 have implementation fixes and regression coverage. See
[audit fix details and compatibility notes](AUDIT_FIXES_2026-10-03.md). New campaigns require
storage version 2 and new dataset-bound preflight certificates; old raw campaigns are preserved.

| Tasks | Implemented | Validation / outstanding work |
|---|---|---|
| T01 | Package, CLI, pyproject, dependency groups, Python reference version | `uv.lock` present; user completed frozen dev/prepare installation. Offline lock freshness check passes. |
| T02–T03 | Protocol docs, strict resolved config, schema, experiment invariants | Unit tests cover unknown/duplicate keys, schedule identity, nonfinite/negative metrics and invalid experiments. |
| T04 | Environment/doctor, harness source fingerprint, allowlisted runtime variables | Missing binaries/driver/NVML recorded explicitly; actual GPU fields await the lab host. |
| T05–T06 | Explicit-commit bootstrap, binary pinning and F16→quant preparation scripts | Toolchain/model locks remain unresolved. No download/build/conversion or real artifact claim. |
| T07–T08 | Fixed workloads, rendering, hashes, fake clock/backend, stream contracts | Offline tests include Unicode fragmentation, heartbeat/final handling, answer isolation and timing/count semantics. |
| T09–T13 | Managed Ollama/llama-server, native metrics, cache/prompt equivalence gates | HTTP mocks tested. Actual engine integration/source audit/token behavior remain to be verified. |
| T14–T15 | NVML monitor, phase traces, raw storage, checksums, process ownership and crash preservation | Synthetic/missing-sensor/storage tests pass. Real NVML sampling and overhead await hardware. |
| T16–T17 | Balanced schedule, cold/warm state machine, resume and failure history | End-to-end fake campaign verifies counts, warmup exclusion, successful-trial skipping and integrity failures. |
| T18–T20 | Objective scorers, aggregation, paired comparisons, reports/figures | Offline pipeline passes; report tests exercised SVG/CSV/Markdown and installed Matplotlib PNG exports. |
| T21 | Hardware preflight, reference token counts, overhead/near-limit checks, freeze command | A failed local readiness report is saved; no certificate or official freeze. Actual RTX 3050 pilot required. |
| T22 | CI workflow, offline tests, opt-in real engine/GPU tests | Local Ruff and unittest suite pass; hardware tests explicitly skipped. Remote CI remains unverified. |
| T23 | Official execution path and audit commands | Official campaigns not executed: driver, serving binaries, artifacts and passing preflight unavailable. |
| T24 | README, reproduction/methodology/limitations and interview prompts | Synthetic demonstration available. Actual findings and full portfolio results require T23. |

## Implementation refinements

- Tests use `unittest` so meaningful offline verification can run before pytest is installed;
  pytest can discover the same tests. CI executes the standard-library runner.
- Aggregation uses standard-library `statistics`; NumPy remains in declared dependencies for the
  planned environment/preparation workflow, but is not needed to calculate mean/median/sample std.
- Standalone SVG figures work without Matplotlib. PNG is an additional export when Matplotlib is installed.
- Session traces/peaks and initial/unresolved failure rates are explicit. The near-limit check also
  requires substantial owned-process usage above idle, preventing unrelated-memory attribution.
- The initial dependency lock was resolved by the user and passed the offline freshness check.
  Keep the reviewed lock in version control before official campaigns.

## Remaining gates

1. Review and commit the resolved `uv.lock`; verify remote CI when the repository is published.
2. On the target lab host, restore working NVIDIA driver/NVML, install both engines, pin sources/binaries.
3. Prepare immutable GGUF artifacts and complete the reviewed source audit.
4. Run CPU integration and GPU preflight for all three matrices; calibrate thresholds and freeze.
5. Execute official campaigns, score quality and regenerate reports from their raw data.

Improvements I01–I07 are intentionally deferred until these MVP gates are complete.

## Local validation evidence

- Full unittest run after fixes: 73 tests, 71 passed and 2 real-engine/GPU tests explicitly skipped.
- Regression coverage includes session restart, journal recovery/tampering, monitor loss, dataset
  binding, controlled subprocess environments, HTTP deadlines, fresh quality reports and provenance reuse.
- Storage-v2 CLI demo (`demo-audit-fixed`): 36 planned trials, 84 total records including warmup/eviction;
  report generation and raw integrity audit passed with 0 unresolved trials. All data is synthetic.
- Python compilation and bootstrap shell syntax checks passed.
- After the user completed dependency installation, Ruff import/UTC fixes and loop-variable
  binding fixes were applied; `ruff check .` passes. Generated reports/results are explicitly excluded.
- `uv lock --check --offline` passes. Remote CI and hardware validation remain unverified.
