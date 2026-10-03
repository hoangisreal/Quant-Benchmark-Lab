# Development and contribution guide

[Documentation home](../README.md)

## Architecture

`cli.py` parses commands and delegates to focused modules:

| Responsibility | Module |
|---|---|
| Strict configuration, relative path resolution, result types | `config.py`, `schema.py` |
| Balanced schedules, dataset identity, certificate checks | `protocol.py` |
| Sequential cold/warm lifecycle and failure persistence | `runner.py` |
| Managed local engines and streaming transport | `backends/` |
| NVML sampling and exclusive GPU ownership | `monitoring/gpu.py` |
| Metadata, immutable attempts, journals and recovery | `storage.py` |
| Host/binary identity and GGUF lineage | `environment.py`, `artifacts.py` |
| Real tokenizer/cache/offload/overhead probes | `preflight.py` |
| Objective scoring and offline reports | `quality/`, `reporting/` |

The fake backend and clock provide deterministic offline coverage. They cannot issue
hardware certificates. Default tests never download models or start real engines.

## Configuration

All example commands run from the repository root. YAML file paths resolve relative to
the YAML that defines them; binary names without a slash are looked up on `PATH`.
Experiment files include the shared runtime, protocol, generation and model catalog.

| File | Controls |
|---|---|
| `configs/runtime.yaml` | Context, CPU threads, batch, KV cache, GPU index, binaries and ports |
| `configs/generation.yaml` | Explicit greedy settings, seed and output budgets |
| `configs/protocol.yaml` | Repetitions and order seed; further thresholds have typed defaults in `config.py` |
| `configs/models.yaml` | Source revisions, templates, thinking mode, stops and GGUF manifests |
| `configs/experiments/*.yaml` | Which variable changes and which model/engine/quant cells participate |

Unknown fields, duplicate YAML keys and unsupported sampling profiles fail validation.
Resolved schedules contain defaults explicitly. Changes to settings, data, sources or
environment require a new pilot and campaign identity; do not edit a frozen campaign.

## Checks

```bash
uv sync --frozen --python 3.11 --group dev
uv lock --check --offline
uv run --frozen ruff check .
uv run --frozen python -m unittest discover -s tests -v
```

Tests use `unittest`; pytest can also discover them. Unit tests cover metric definitions,
configuration and scoring. Contract tests cover request options and streamed framing/timeouts.
Integration tests cover synthetic campaigns, resume, integrity and report regeneration.
Two opt-in checks skip unless `QBL_CPU_CONFIG` or `QBL_GPU_CONFIG` is set; their setup is
in [reproduction](guides/reproduction.md). An offline pass is not real-engine validation.

Use four-space indentation, `snake_case` functions/modules and `PascalCase` classes.
Ruff is configured for Python 3.11 and a 100-character line target. Format changed Python
files with `uv run --frozen ruff format path/to/file.py`; avoid unrelated bulk formatting.
There is currently no configured type checker or numerical coverage threshold.

CI requires a present, fresh `uv.lock`, installs it with Python 3.11 and frozen resolution,
then runs lint, the offline suite and a CLI demonstration including resume. A missing or
stale lock fails before installation; CI never resolves a replacement lock or downloads weights.
Actions are pinned to immutable commits using Node 24, with uv 0.12.17 and Python 3.11.
Remote CI success must be checked on the actual commit, not inferred from local results.

## Contributions and local files

Use short imperative commit subjects. Describe behavior, motivation and relevant checks
in pull requests; link issues when applicable. Include regression coverage for measurement,
integrity or lifecycle fixes, and disclose changes requiring a new pilot or storage version.

Keep credentials out of configs, logs and version control. Model weights, vendor checkouts,
generated campaign/report directories and internal AI-agent planning notes are ignored.
Public procedures belong in `docs/guides/`, definitions/evidence in `docs/reference/`,
and audit/research summaries in `docs/reviews/`. Machine-specific provenance locks and
`configs/frozen/` are generated locally and ignored; dataset provenance remains tracked.
Ignored notes remain local; removing previously tracked notes does not erase Git history.
