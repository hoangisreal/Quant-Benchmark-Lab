# Quant Benchmark Lab

A reproducible, single-machine benchmark harness for small GGUF language models on an
**RTX 3050 4GB**. It measures objective quality, streaming latency, inference throughput
and GPU memory while keeping three experiments separate:

| Experiment | Changes | Held constant |
|---|---|---|
| Quantization | F16, Q8_0, Q4_K_M | Qwen2.5-0.5B, llama.cpp |
| Engine | Ollama, llama.cpp | The same Qwen2.5-1.5B Q4_K_M GGUF, rendered prompts and generation profile |
| Model | Qwen2.5-0.5B/1.5B/3B, Qwen3-4B | llama.cpp, Q4_K_M and task text |

The Python harness and synthetic workflow are implemented. Dependencies are locked in
`uv.lock`; the current candidate pair has passed real CPU integration locally. Each lab
still needs local artifact preparation and passing GPU preflights before official campaigns.
**No official GPU benchmark results are published here.**
The 4B candidate has not yet been established as a valid near-limit fit on the target GPU.

## Start here

Use Linux and Python 3.11. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Git, then run the following from a terminal. Existing checkouts can start at `uv sync`.

```bash
git clone https://github.com/hoangisreal/Quant-Benchmark-Lab.git
cd Quant-Benchmark-Lab
uv sync --frozen --python 3.11 --group dev
uv run --frozen qbl validate --config configs/demo.yaml
```

Continue with the [complete synthetic walkthrough](docs/guides/getting-started.md). It runs
planning, inference fixtures, scoring, reporting and integrity checks without CUDA or model
downloads. Synthetic numbers demonstrate the harness and cannot establish model performance.

## Documentation: installation to results

The [documentation index](docs/README.md) contains the full reading order and
[audit/research findings](docs/reviews/2026-10-03.md).

| Step | Guide | Outcome |
|---|---|---|
| 1 | [Getting started](docs/guides/getting-started.md) | Installed environment and a complete synthetic report |
| 2 | [Methodology](docs/reference/methodology.md) | Understand timing, cache control, quality scoring and validity gates |
| 3 | [Toolchain and model preparation](docs/guides/toolchain.md) | Pinned binaries, reviewed settings and local GGUF artifacts |
| — | [Engine compatibility](docs/reference/engine-compatibility.md) | Candidate revisions, source evidence and remaining runtime probes |
| 4 | [Reproduce a real campaign](docs/guides/reproduction.md) | CPU integration, GPU pilot, frozen configuration and official runs |
| 5 | [Read and preserve results](docs/guides/results.md) | Interpret reports, inspect failures and regenerate from raw data |
| 6 | [Limitations](docs/reference/limitations.md) | Know which conclusions the measurements support |
| — | [Troubleshooting](docs/guides/troubleshooting.md) | Diagnose installation, preflight, resume and reporting failures |
| — | [Development](docs/development.md) | Architecture, configuration, tests and contribution workflow |

## Measurements

Each campaign records load time, client-observed streaming TTFT, native prefill/decode
rates, end-to-end latency, input/output token counts, and idle/loaded/peak VRAM.
Cold and warm runs remain separate; warmups are excluded. Reports preserve per-prompt
groups and show mean, median, sample standard deviation and failure counts.

Quality uses 60 project-authored exact-match, numeric and JSON tasks. It is a narrow
objective suite, not a general model ranking. Missing metrics remain missing, with reasons
in raw records. Official runs require verified artifacts and a passing preflight bound to
the datasets, configuration and environment.

## Repository layout

```text
src/quant_benchmark_lab/   CLI, backends, runner, storage, monitoring, scoring, reporting
configs/                  Runtime, generation, protocol and experiment matrices
scripts/                  Engine build, toolchain pinning and model preparation
data/                     Performance prompts, quality questions and separate answer keys
locks/                    Toolchain, model and workload provenance
tests/                    Unit, streaming contract, campaign and opt-in hardware checks
docs/guides/              Installation, preparation, campaigns, results, troubleshooting
docs/reference/           Methodology, engine compatibility, limitations
docs/reviews/             Public audit and research summaries
results/raw/              Local campaign data (generated directories ignored)
reports/                  Local reports and figures (generated directories ignored)
```

For development checks, see [Development](docs/development.md). Large weights, local raw
campaigns and internal agent notes are excluded from Git; public guides are self-contained.
Machine-specific toolchain/model locks and frozen configurations stay local; see
[configuration](configs/README.md) and [lock provenance](locks/README.md).
