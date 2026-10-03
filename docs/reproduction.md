# Reproduce a campaign

## Dependencies

Use Python 3.11 and `uv`, then run `uv sync --frozen --group dev` using the repository
`uv.lock`. Add `--group prepare` for model conversion. The initial lock and frozen installation
completed successfully; local Ruff and offline tests now pass. Review and commit the lock
before official campaigns. Rerun `uv lock` only for intentional dependency updates.

Runtime dependency versions, interpreter and harness source hash are captured by `qbl doctor`.
Heavyweight conversion requires `uv sync --frozen --group prepare`. Engine installation/build,
source weight downloads and model import occur outside measurement windows.

## Toolchain and source audit

On the lab machine, first ensure `nvidia-smi` and `nvcc --version` work and install Ollama.
Choose a supported llama.cpp commit exposing the documented native API/launch flags. Build it
at a full immutable SHA (replace the placeholder below; no floating revision is accepted):

```bash
bash scripts/bootstrap_llama.sh <full-llama.cpp-commit-sha> cuda
uv run python scripts/pin_toolchain.py --source vendor/llama.cpp --output locks/toolchain.json
```

The CUDA build targets compute capability 8.6 for the RTX 3050. The pin script records actual
binary hashes, quantizer, compiler, CMake cache, CUDA toolkit and commit; it requires both engines.

Copy `configs/settings_audit.example.json` into a reviewed audit file. Inspect the exact pinned
engine sources for sampler defaults/order, KV dtype, batch semantics, speculation disabled,
BOS/special-token policy and truncation/shift behavior. For each reviewed field include a source
reference with commit/path/line or source digest, record the actual binary SHA and set only fields
whose behavior was established. Keep unsupported settings documented as disabled/inapplicable.

```bash
uv run python scripts/pin_toolchain.py --source vendor/llama.cpp --audit <reviewed-audit.json>
```

This source review is a required methodology artifact; it is not automatic approval or proof
that the GPU ran correctly. Native preflight probes still need to pass.

## Prepare models

The catalog starts with unresolved source revisions/checksums. Preparation resolves the source
revision once, downloads that immutable revision, derives the official non-thinking template,
converts F16, and quantizes directly from it. It records the model-card license and artifact lineage.
Existing mismatched artifacts are never overwritten.

```bash
uv run --group prepare python scripts/prepare_models.py --model qwen-0_5b
uv run --group prepare python scripts/prepare_models.py --model qwen-1_5b --quants Q4_K_M
uv run --group prepare python scripts/prepare_models.py --model qwen-3b --quants Q4_K_M
uv run --group prepare python scripts/prepare_models.py --model qwen-4b --quants Q4_K_M
```

Even Q4-only preparation needs an F16 ancestor; Q8 is generated only when selected.
Disk/RAM checks precede downloads/conversion. Model files stay outside Git; commit manifests,
resolved catalog, toolchain identity and workload hashes. `--dry-run` lists preparation intent
without loading dependencies or downloading weights, but still requires a pinned local toolchain.

## Pilot, freeze, run

Run this sequence separately for quantization, engine and model experiments, with fresh output
directories. Example for the engine experiment:

```bash
uv run qbl doctor --config configs/experiments/engine.yaml --output /tmp/environment.json
uv run qbl validate --config configs/experiments/engine.yaml
uv run qbl preflight --config configs/experiments/engine.yaml --output results/raw/pilot-engine
uv run qbl freeze --config configs/experiments/engine.yaml --preflight results/raw/pilot-engine/preflight.json --output configs/frozen/engine.yaml
uv run qbl plan --config configs/frozen/engine.yaml --output /tmp/engine-schedule.json
uv run qbl run --schedule /tmp/engine-schedule.json --campaign results/raw/engine-official
uv run qbl evaluate --campaign results/raw/engine-official
uv run qbl report --campaign results/raw/engine-official --output reports/engine-official
uv run qbl audit --campaign results/raw/engine-official
```

Preflight first uses the GGUF tokenizer on CPU, then collects a shorter GPU performance campaign
and quality probes plus monitor-overhead probes. Its certificate contains token/cache counts
for each cell/run mode, dataset content hashes and environment/config identity. A failing certificate cannot freeze an official config. If the model matrix lacks a real
near-limit model, adjust the common draft context/selected model and repeat the entire pilot.

Do not update packages/engines/models mid-campaign. After an interruption, run the same command
with `--resume`; failed attempts and torn append tails remain stored. Valid completed trials are
skipped. Storage version 2 assigns a fresh session attempt directory on restart and recovers
all four JSONL journals from checksummed write-ahead entries. Campaigns created before this
format must use a new output directory; old checksums cannot certify missing journal history. New config, source or environment identity requires a new campaign path.

## Offline reports and tests

Reports only require the raw campaign, including its immutable metadata, answer key snapshot,
run records, per-attempt checksums, raw events, outputs, load observations and GPU traces. They
do not re-open source model paths or regenerate the schedule. `qbl report` verifies raw artifact
integrity first. `qbl evaluate` verifies the snapshot answer-key checksum. Reports recompute quality from the
verified raw snapshot, so an older `quality.json` export cannot affect the report.
`qbl audit` reports trial completion separately from file integrity.

```bash
uv run python -m unittest discover -s tests -v
QBL_CPU_CONFIG=<pinned-tiny-model-cpu-config.yaml> uv run python -m unittest tests.integration.test_real_engines -v
QBL_GPU_CONFIG=configs/experiments/engine.yaml uv run python -m unittest tests.integration.test_real_engines -v
```

CPU integration still uses actual pinned engines and a local GGUF, not synthetic data. No test
automatically downloads model weights. The fake campaign is the default offline validation path.
