# Getting started without a GPU

[Documentation home](../../README.md) · Next: [prepare the toolchain](toolchain.md)

## 1. Install the Python environment

Use Linux: process ownership and file locking depend on Unix facilities. Python 3.11 is
the reference interpreter. Git and an initial internet connection are needed for cloning
and dependency installation. CUDA, Ollama and model weights are unnecessary for this guide.

Install uv using its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```bash
uv sync --frozen --python 3.11 --group dev
uv run --frozen qbl --help
uv run --frozen qbl doctor --output /tmp/qbl-environment.json
```

`--frozen` uses the committed dependency resolution. Do not run `uv lock` during ordinary
setup. `doctor` captures the host and records missing engines/GPU explicitly; without
`--require-gpu`, missing GPU hardware does not make this command fail.

## 2. Run a complete synthetic campaign

These examples use Bash. `mktemp` provides fresh paths each time, avoiding accidental reuse
of an existing campaign. Keep this terminal open for the commands that use these variables.

```bash
mkdir -p results/raw reports
qbl_demo_root=$(mktemp -d results/raw/demo-XXXXXX)
qbl_demo_name=$(basename "$qbl_demo_root")
qbl_demo_schedule=$(mktemp /tmp/qbl-demo-schedule-XXXXXX.json)
qbl_demo_report="reports/$qbl_demo_name"

uv run --frozen qbl validate --config configs/demo.yaml
uv run --frozen qbl plan --config configs/demo.yaml --output "$qbl_demo_schedule"
uv run --frozen qbl run --schedule "$qbl_demo_schedule" --dry-run
uv run --frozen qbl run --schedule "$qbl_demo_schedule" --campaign "$qbl_demo_root"
uv run --frozen qbl evaluate --campaign "$qbl_demo_root"
uv run --frozen qbl report --campaign "$qbl_demo_root" --output "$qbl_demo_report"
uv run --frozen qbl audit --campaign "$qbl_demo_root"
```

Structural validation does not certify hardware. Planning produces 36 measured/quality
trials for the current demo; warmup and eviction records are additional. `--dry-run`
checks the schedule without starting inference. The final audit should report verified
integrity, complete trial completion and zero unresolved trials.

## 3. Open the outputs

Open `report.md` inside the report directory printed by the CLI. It links CSV tables and
SVG plots; Matplotlib also exports PNGs. Every demo report is labeled **SYNTHETIC**.
Fixture quality scores and memory/timing values do not measure Qwen or your GPU.

The campaign directory stores the schedule, environment, settings, answers, raw events,
outputs and integrity journals. The [results guide](results.md) explains their use.

To exercise resume with the same unchanged configuration and schedule:

```bash
uv run --frozen qbl run --schedule "$qbl_demo_schedule" --campaign "$qbl_demo_root" --resume
uv run --frozen qbl audit --campaign "$qbl_demo_root"
```

Completed valid trials are skipped. Use a fresh directory when changing the schedule or
datasets. Once this workflow succeeds, follow [toolchain preparation](toolchain.md).
