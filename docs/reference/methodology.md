# Benchmark methodology, protocol v1

[Documentation home](../../README.md) · [Run the protocol](../guides/reproduction.md)

## Controlled experiments

E1 varies F16/Q8_0/Q4_K_M on Qwen2.5-0.5B and llama.cpp. E2 varies Ollama versus llama.cpp
with the exact same local Q4_K_M GGUF and rendered prompt. E3 varies models under llama.cpp
Q4_K_M. The matrices are in `configs/experiments/`. Source revisions and artifact checksums
must be verified by the local preparation pipeline, not guessed from model tags. A resolved
catalog or source audit is not a passing hardware certificate.

Real baseline runs require full GPU layer offload. An OOM, unknown offload state, automatic
context adjustment or CPU layer fallback invalidates the comparison. E3 also needs a model
with measured device peak at least 80% of total VRAM, attribution to owned processes, and
owned-process usage at least 80% of memory available above the idle baseline. This prevents
unrelated GPU memory pressure from creating a near-limit claim.

## Explicit settings

Baseline sampling is greedy: temperature 0, seed 42, top-k 1, top-p 1, min-p 0, penalties
disabled and no speculation. Context starts at 4096, batch/microbatch at 128, F16 KV cache,
flash attention off, one sequence, and physical-core CPU threads. Performance/quality generation
budgets are 128/256 tokens. Natural EOS is retained, actual output length and stop reason recorded.
Same seed does not imply identical output across kernels or engines.

Raw prompts use pinned model templates. Qwen3 uses its non-thinking template; unexpected thinking
events or generated `<think>` markers invalidate the request. User text is rendered once and
hashed before either engine sees it. Reference GGUF tokenization detects context overflow and
BOS/input-count mismatch. Answer keys are never included in model requests.

Some effective sampler/runtime behavior needs a reviewed source audit for the pinned binary.
`configs/settings_audit.example.json` is deliberately unapproved. A successful HTTP response
does not prove an unknown setting was applied. The audit is documented source review;
it is supplemented by token/cache/offload and native-metric probes, not presented as telemetry.

## Timing boundaries

| Metric | Boundary/source |
|---|---|
| Model load wall | Ollama explicit preload → complete; llama-server spawn → health ready |
| Native load | Engine-provided timing if its boundary is known, otherwise null with reason |
| Streaming TTFT | Before request transport → first nonempty generated text, including whitespace |
| Prefill tok/s | Native uncached evaluated tokens / native prefill duration |
| Decode tok/s | Native decoded-token count / native generation duration |
| Request E2E | Before request transport → valid terminal event |
| Cold activation-to-response | Activation start → terminal, including readiness/observation gaps |

Wall-clock timings use monotonic nanoseconds; UTC is an audit timestamp. HTTP JSON encoding
and client transport are included in request time, while reference tokenization/config checks
are excluded. TTFT is a first-content proxy because transport can buffer multiple tokens.
Native token counters, not HTTP chunks, supply token counts. Ollama nanoseconds are converted
to milliseconds; llama.cpp native milliseconds retain their own source labels. No rate is inferred
from TTFT or inter-chunk gaps. Missing or zero-duration rates are null, never fabricated.

For the [audited candidate pair](engine-compatibility.md), `predicted_n` and Ollama's
forwarded `eval_count` include the first token sampled from the final prefill logits.
Their native generation rate uses `max(0, generated - 1)` decoding steps. The adapter
preserves the full output count and uses that separate denominator for decode tok/s;
a single generated token has zero subsequent decoding steps. This mapping must be
reviewed again for any engine revision with different native timer/counter semantics.

## Cold, warm and cache

Cold means model-cold, with uncontrolled OS page cache. Every cold trial unloads the model,
observes idle memory, activates it, observes loaded memory, runs its first inference and unloads.
There is no inference warmup between cold activation and measurement. Ollama daemon/import setup
is outside model activation; llama.cpp process startup is included. Internal engine startup warmup
may differ, so operational load-to-ready is a stack measurement rather than pure tensor I/O.

Warm sessions keep a model resident and discard two warmup requests. Before each measured warm
request, both stacks process the same raw eviction prompt with one generated-token budget and
settle for one second. A single slot/sequence prevents multiple retained prompt histories. The
eviction prompt must fit context and be at least as long in native tokens as the target prompt.

Native cache counts must prove no substantive prefix reuse. A fixed BOS residual can be
allowed only through a protocol change before freeze; cold and warm cache counts are validated
as separate strata. Engine-paired records require matching prompt, input and cache counts.
Cache eviction itself is never proof of correctness. Official measured cache counts must match
the preflight certificate for every cell/run mode/prompt. Preflight covers performance and
quality, and binds the contents of both datasets and the answer key. Quality requests are cache-prepared but not included in performance
aggregates.

## GPU telemetry

NVML is sampled in a separate thread every 20 ms initially. Idle/loaded values are medians of
two-second windows. Load, request and whole-session peaks have distinct scopes. Session peaks
include warmup/eviction; request peaks exclude them. Raw traces retain phase, device memory,
per-process memory when supported, temperature, utilization, clocks and power when available.
Missing sensors are explicit. Sampled peaks can miss transients between samples.

Preflight alternates five monitor-on/off pairs for each cell and checks median overhead
against 3% in every cell. A failure
requires a common interval recalibration and a new pilot. HTTP generation uses a reusable async HTTP client behind the sequential backend interface.
Absolute TTFT/request deadlines apply to headers, raw byte reads, heartbeats and partial frames;
timeout cancellation closes the response. The transport change requires a new hardware pilot.

Before each session, idle temperature
and utilization must meet the frozen thresholds. Memory must return to the initial baseline
within the pinned tolerance after unload. Monitor loss/cleanup failure is a campaign error.

## Repetitions and analysis

For each cell/prompt: five cold repetitions and ten warm repetitions across two sessions.
Seeded balanced blocks randomize configuration order; one request/model is active at a time.
Mean, median, sample std (`ddof=1`), planned/attempted/valid counts and missing/failure counts
are reported separately per configuration, prompt and cold/warm mode. Unique load events are
not duplicated across generation repetitions. Std is null for one observation.

All attempts remain raw. Resume creates a new attempt ID and redoes session warmup. Analysis
selects the first valid completed attempt in append order, never the fastest or best-scoring one.
Successful trials are not rerun. Initial-attempt failures remain visible alongside unresolved trials.
No statistical outliers are deleted after looking at results. Harness-source, config, workload,
environment and artifact identities must match when resuming an official campaign.

Quality uses one greedy response for each of 60 fixed items. Exact-match normalization, numeric
decimal tolerances and strict JSON expectations are explicit in the answer key. Category accuracy
and their macro-average are separate from performance repetitions. Unresolved trials score zero
in the completion-inclusive view; answered-only accuracy is also available. LLM judging is outside MVP.

## API references

Adapters follow [Ollama generate](https://docs.ollama.com/api/generate),
[Ollama usage](https://docs.ollama.com/api/usage),
[GGUF import](https://docs.ollama.com/import) and
[llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
The local binary/source lock and audited API behavior govern the actual experiment. Documentation
at a floating branch is a reference, not a substitute for verification of the pinned build.


## Engine environment and storage integrity

Engine subprocesses receive a controlled environment: PATH, HOME, TMPDIR, LD_LIBRARY_PATH,
CUDA_VISIBLE_DEVICES, CUDA_MODULE_LOADING, OMP_NUM_THREADS and OMP_PROC_BIND when present;
LANG/LC_ALL are C.UTF-8 and TZ is UTC. This effective base environment enters the fingerprint.
Ambient LLAMA_ARG_*, OLLAMA_*, LD_PRELOAD and other variables are omitted; Ollama receives
explicit backend overrides. llama.cpp KV offload and speculation are explicit launch options.
Changes to the supported runtime profile still require review of the pinned engine sources.

Storage version 2 requires the immutable metadata inventory and checksummed write-ahead
entries for runs, loads, sessions and control journals. Each session attempt has its own
reserved directory, load ID and checksummed GPU trace. Recovery restores committed appends
and quarantines incomplete tails. Telemetry failures invalidate an attempt while preserving
already-received inference events/output. Reports recompute objective quality from raw inputs.
