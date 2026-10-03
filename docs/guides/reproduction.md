# Reproduce a real campaign

[Documentation home](../../README.md) · Previous: [toolchain and models](toolchain.md)
· Next: [read and preserve results](results.md)

## Prerequisites and stopping points

Complete the synthetic walkthrough, engine build/pinning, source audit and GGUF preparation
first. Run all commands from the repository root with the same Python environment. This
guide documents the execution path for the source-reviewed candidate with a local CPU
integration pass. GPU validation and official measurements remain pending.

Use one campaign at a time on the RTX 3050 4GB. Stop unrelated GPU work that you own,
keep power/display conditions stable and ensure sufficient disk space for logs, traces and
Ollama imports. Do not update dependencies, engines, model artifacts or workloads between
preflight and official execution. The initial configuration uses context 4096, greedy
sampling, F16 KV cache, two warmups, five cold and ten warm repetitions per performance
prompt/cell. Quality uses one response per item/cell.

## 1. Exercise both real backends on CPU

Use the supplied tiny-model CPU integration configuration. This check is separate from
the 1.5B GPU engine experiment:

```bash
QBL_CPU_CONFIG=configs/experiments/engine-cpu.yaml uv run --frozen python -m unittest tests.integration.test_real_engines.RealEngineTests.test_cpu_engine_parity -v
```

By default the test deletes its temporary output. To retain logs, reference token IDs and
native responses, rerun with a new, nonexistent output directory:

```bash
QBL_CPU_CONFIG=configs/experiments/engine-cpu.yaml \
QBL_CPU_OUTPUT=results/raw/cpu-check-01 \
uv run --frozen python -m unittest tests.integration.test_real_engines.RealEngineTests.test_cpu_engine_parity -v
```

This check uses the prepared 0.5B Q4_K_M GGUF in both engines, tokenizes reference prompts,
streams responses and checks context, input/cache counters, native rates and unload.
It needs real engines, sufficient RAM and the artifact
lock; it does not download models. It does not replace the settings audit or GPU preflight.

## 2. Select and inspect an experiment

Use the variable below for one experiment at a time. Valid values are `quantization`,
`engine` and `model`; complete this workflow for each. The run label is an output identity,
not an engine version. Change it for a new pilot/campaign.

```bash
qbl_experiment=engine
qbl_run_label=run-01
qbl_config="configs/experiments/$qbl_experiment.yaml"
qbl_pilot="results/raw/pilot-$qbl_experiment-$qbl_run_label"
qbl_frozen="configs/frozen/$qbl_experiment-$qbl_run_label.yaml"
qbl_campaign="results/raw/$qbl_experiment-$qbl_run_label-official"
qbl_report="reports/$qbl_experiment-$qbl_run_label-official"

uv run --frozen qbl doctor --require-gpu --config "$qbl_config" --output /tmp/qbl-lab-environment.json
uv run --frozen qbl validate --config "$qbl_config"
```

Keep these shell variables for subsequent commands. Review the doctor output and the
resolved catalog/locks. A successful doctor/validate command is not a passing pilot.

## 3. Run the GPU pilot

```bash
uv run --frozen qbl preflight --config "$qbl_config" --output "$qbl_pilot"
```

Preflight writes `preflight.json`, reference tokenizer IDs, a shorter performance/quality
campaign and monitor-overhead observations. Inspect `passed` and `failures`; only an actual
passing result permits the next step. A failed pilot remains diagnostic evidence.

The gates include full GPU offload, context/settings evidence, input/cache parity, required
native timing and VRAM metrics, and monitoring overhead. E1 requires F16/Q8_0/Q4_K_M. E3
requires at least three models and a measured, attributable near-limit model. The 4B
candidate is not assumed to fit. See [methodology](../reference/methodology.md) for exact boundaries.

If a gate fails, investigate, revise draft settings where justified, and rerun with a new
label. Record the change and apply shared settings consistently to affected cells. Do not
relax thresholds after seeing official results or copy a certificate from another matrix.

An optional automated GPU check runs another full preflight in a temporary directory:

```bash
QBL_GPU_CONFIG=configs/experiments/engine.yaml uv run --frozen python -m unittest tests.integration.test_real_engines.RealEngineTests.test_target_gpu_preflight -v
```

Use the CLI pilot above to retain evidence for freezing; the test's temporary output is
removed and cannot supply a durable certificate.

## 4. Freeze and plan

```bash
uv run --frozen qbl freeze --config "$qbl_config" --preflight "$qbl_pilot/preflight.json" --output "$qbl_frozen"
qbl_schedule="${qbl_frozen%.yaml}-schedule.json"
uv run --frozen qbl plan --config "$qbl_frozen" --output "$qbl_schedule"
uv run --frozen qbl run --schedule "$qbl_schedule" --dry-run
```

Freeze refuses an existing output or an invalid/synthetic certificate. The resolved
configuration stores absolute artifact/data/certificate paths. Preserve those files and
locations for execution/resume; a moved environment needs preparation and preflight again.
Planning includes performance and quality by default. Keep the generated schedule unchanged.

## 5. Execute, score and audit

```bash
uv run --frozen qbl run --schedule "$qbl_schedule" --campaign "$qbl_campaign"
uv run --frozen qbl audit --campaign "$qbl_campaign"
uv run --frozen qbl evaluate --campaign "$qbl_campaign"
uv run --frozen qbl report --campaign "$qbl_campaign" --output "$qbl_report"
```

Check both `integrity` and `trial_completion`, including `n_unresolved`. Audit returns
exit status 2 when planned trials are unresolved, even if file integrity is verified.
Reports can include
failures and missing metrics; never fill them with estimates. Pilot observations are kept
outside the official directory and excluded from its aggregates.

## 6. Resume an interrupted campaign

```bash
uv run --frozen qbl run --schedule "$qbl_schedule" --campaign "$qbl_campaign" --resume
uv run --frozen qbl audit --campaign "$qbl_campaign"
uv run --frozen qbl evaluate --campaign "$qbl_campaign"
uv run --frozen qbl report --campaign "$qbl_campaign" --output "$qbl_report"
```

Resume requires the same schedule, data, artifacts and compatible environment identity.
It skips valid completed trials, keeps prior failures and creates new session attempts
with fresh warmup. Storage v2 recovers interrupted committed journal writes under a lock;
changed checksums or missing history fail closed. Do not modify raw files to force recovery.
See [troubleshooting](troubleshooting.md) and [storage compatibility](results.md).

## 7. Complete all three experiments

Repeat steps 2–6 for the remaining matrices using distinct pilot, frozen, schedule, raw
and report paths. Keep the toolchain stable across them. Preserve a record of unavoidable
host/session differences. A completed engine experiment cannot substitute for quantization
or model evidence. Use the [results guide](results.md) for offline regeneration and sharing.
