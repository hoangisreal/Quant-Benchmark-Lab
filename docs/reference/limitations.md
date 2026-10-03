# Interpretation and current limitations

[Documentation home](../../README.md) · [Measurement definitions](methodology.md)

- No official GPU results have been collected. Synthetic report numbers are fixture observations.
- Python dependencies are locked in `uv.lock`; frozen installation and offline validation have
  been exercised. Real engine builds, model preparation, GPU preflight and remote CI status
  must be established separately. The source-reviewed candidate has a local CPU integration
  pass, but no GPU-validated engine version pair or official campaign is published yet.
- Real API behavior must match the pinned versions. Cache fields absent in an older Ollama or
  llama.cpp build are rejected rather than inferred; unsupported log formats leave offload unknown.
- The source audit is a documented review artifact. Context/offload/cache/token probes complement
  it; assertions in an audit file alone do not establish empirical correctness.
- Cold is model-cold, not disk-cold. OS page cache, daemon lifecycle, CUDA initialization and
  internal engine startup warmup can differ. Native load timers have different boundaries.
- HTTP first-content time approximates TTFT. Frames may contain multiple tokens; HTTP chunks
  are not token counts. Transport framing and buffering are part of the serving stack.
- Native decode timing can count first token/EOS differently. Raw counts and native timings are
  retained; headline wall latency and engine-reported token rates answer different questions.
- Native tokens differ between models. E3 uses identical task text with model-specific templates;
  an arithmetic average of model token rates is not a common-work comparison.
- Actual output lengths vary under natural EOS. Latency comparisons must be read with output
  counts. Truncated quality outputs may be incorrect even though inference completed normally.
- NVML gives sampled peaks; short transients can be missed. Process accounting, power/clocks or
  throttle telemetry may be unavailable. Global desktop memory is recorded in idle/absolute VRAM.
- Prefix-eviction requests add thermal workload outside timed requests. Both engines use the same
  procedure, and cache counts/settle conditions are checked, but this is not an ideal kernel microbenchmark.
- Monitor overhead uses five alternating pairs and a preregistered 3% median check. Noise can
  fail the gate; a new pilot is needed, not post-hoc deletion of unfavorable observations.
- Quality is a deliberately narrow, project-authored 60-item objective suite. It is not a standard
  leaderboard, coding evaluation or general capability score; contamination/generalization is unknown.
- The six performance texts are distinct but all use a repetitive artificial word pattern
  across three nominal length buckets. They provide little task diversity; bucket names
  do not establish actual native input counts. Broader prompts need a protocol revision.
- The shared greedy profile is a controlled baseline. It differs from Qwen3's recommended
  non-thinking sampling profile, so scores do not represent its best achievable quality.
- Monitoring overhead is checked separately for every cell. The gate needs real GPU probes;
  a passing unit test of the aggregation cannot establish measured overhead.
- One GPU/session environment cannot support broad claims about other GPUs or concurrency.
- Conversion currently supports the local Qwen catalog and official Transformers templates.
  It requires enough disk/RAM and a compatible pinned llama.cpp converter.
- Tables/CSV and SVG export are implemented without plotting-library dependencies. PNG export
  uses Matplotlib and has been exercised by offline tests; that verifies reporting, not GPU inference.
- Coding execution, LLM judges, bootstrap confidence intervals, hybrid offload, cache-hit sweeps,
  disk-cold trials and energy metrics remain planned improvements, not MVP claims.

## Interview questions to prepare

1. Why can a warm prompt appear faster even if the kernel did not improve? Explain prefix cache.
2. What exactly starts/stops TTFT, load and E2E timers? Explain observable versus native boundaries.
3. How is identical GGUF input proven across stacks, and what cannot be proved without token IDs?
4. Why does full offload require evidence beyond “GPU detected” or model-file size?
5. Why keep cold/warm, prompt lengths and native tokenizers in separate analysis groups?
6. How do failed attempts, first-valid selection and disclosed initial failure rates avoid selection bias?
7. What can exact-match/numeric/JSON scores conclude, and what needs a larger external task suite?
8. Which claims require another hardware/session campaign before generalizing the result?
