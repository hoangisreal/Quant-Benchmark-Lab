# Quant Benchmark Lab

A single-machine streaming benchmark harness for small GGUF models on an RTX 3050 4GB.
It keeps quantization, serving-engine and model experiments separate, and records raw HTTP
events, native timings, token counts, settings, source identities and GPU telemetry.

## Current status

The mock/CPU harness is implemented and tested. Real engine integration and official GPU
campaigns still need the lab environment. Python dependencies are locked in `uv.lock`;
local Ruff checks and the offline suite pass (71 tests passed, 2 hardware tests skipped).
Pinned serving engines, GGUF artifacts and a passing GPU preflight remain required.
**No measured LLM benchmark results are claimed.**

The nine findings from the initial audit have code fixes and offline regression coverage;
see [audit fixes and storage compatibility](docs/AUDIT_FIXES_2026-10-03.md).

See [implementation status](docs/IMPLEMENTATION_STATUS.md),
[execution plan](docs/IMPLEMENTATION_PLAN.md), [methodology](docs/methodology.md) and
[reproduction instructions](docs/reproduction.md).

## Install

Python 3.11 is the reference interpreter. Install the locked dependencies:

```bash
uv sync --frozen --group dev
```

Keep `uv.lock` in version control. Run `uv lock` only when intentionally updating the dependency
resolution, then review the changes. GPU telemetry uses `nvidia-ml-py`; CPU/mock tests do not require NVML or CUDA.
The heavyweight conversion dependencies are isolated in the `prepare` group.

## Run the synthetic demonstration

```bash
uv run qbl doctor --output /tmp/qbl-environment.json
uv run qbl validate --config configs/demo.yaml
uv run qbl plan --config configs/demo.yaml --output /tmp/qbl-schedule.json
uv run qbl run --schedule /tmp/qbl-schedule.json --campaign results/raw/demo
uv run qbl evaluate --campaign results/raw/demo
uv run qbl report --campaign results/raw/demo --output reports/demo
uv run qbl audit --campaign results/raw/demo
```

Use a new campaign path on each new run, or `--resume` for the same unchanged schedule. Demo
outputs and every figure are labeled **SYNTHETIC**. The fake backend emits fixture text and
clock/memory observations; its quality scores and speeds are not measurements of Qwen.

## Project structure

- `src/quant_benchmark_lab/`: strict configuration/schema, backend adapters, runner, monitor,
  objective scorers and offline reporting.
- `configs/experiments/`: E1 quantization, E2 engine and E3 model matrices; all initially draft.
- `scripts/`: explicit-commit build, binary pinning and source-to-GGUF preparation.
- `data/`: six performance prompts and 60 project-authored quality items; answer keys are separate.
- `locks/`: toolchain/artifact/workload provenance. Unresolved locks block real campaigns.
- `tests/`: methodology unit tests, HTTP streaming contracts, fake campaigns and opt-in hardware tests.
- `results/raw/<campaign>/`: immutable metadata, append-only attempts, native payloads and traces.
- `reports/<campaign>/`: CSV, Markdown and standalone SVG; PNG export when Matplotlib is installed.

## Tests

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check .
uv run ruff format .
```

Tests use standard-library `unittest` and are also discoverable by pytest. Default tests are offline;
real engine tests require `QBL_CPU_CONFIG`, and RTX 3050 tests require `QBL_GPU_CONFIG`. Hardware
tests skip explicitly when these variables are absent. CI runs the offline tests and a CLI demo.

## Real benchmark workflow

Install the target NVIDIA driver/CUDA toolchain and Ollama on the lab host; build llama.cpp at
an explicit commit, pin binary identities, prepare GGUFs, review the source audit, then run
preflight separately for each experiment. A passing preflight is required to freeze an official
configuration. [Reproduction instructions](docs/reproduction.md) give the complete sequence.

Reports preserve cold/warm and prompt strata. TTFT is first streamed content at the client;
prefill throughput counts uncached evaluated tokens; load timing is stored once per session.
Objective quality is intentionally a small task suite. See [limitations](docs/limitations.md)
before drawing conclusions about model capability or engine performance.
# Quant-Benchmark-Lab
