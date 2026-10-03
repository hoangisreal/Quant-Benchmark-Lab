# Engine compatibility evidence

[Documentation home](../../README.md) · [Toolchain](../guides/toolchain.md)

## Candidate identities

This pair is under validation. A local real CPU integration check passed on 2026-10-03
using the same Qwen2.5-0.5B Q4_K_M GGUF in both engines. This does not establish GPU
equivalence or benchmark performance. Local diagnostics are retained under
`results/raw/readiness-20261003/`; vendor payloads remain ignored.

| Component | Immutable identity |
|---|---|
| Standalone llama.cpp | [`b92761a515ea31e852e7fbc1fad5f874b46f3718`](https://github.com/ggml-org/llama.cpp/tree/b92761a515ea31e852e7fbc1fad5f874b46f3718) |
| Ollama release | [`v0.35.1`](https://github.com/ollama/ollama/releases/tag/v0.35.1) |
| Ollama source | [`b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce`](https://github.com/ollama/ollama/tree/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce) |
| Ollama bundled llama.cpp | [`6f767fe960c3b97cf37fac4626c86400561ca1e4`](https://github.com/ggml-org/llama.cpp/tree/6f767fe960c3b97cf37fac4626c86400561ca1e4), tag `b11232` |

The official Linux AMD64 Ollama archive has SHA-256
`9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10`.
Its client reports 0.35.1; its bundled server reports commit `6f767fe96`.
Keep the extracted `bin/` and `lib/ollama/` tree together: Ollama launches a separate
server and loads shared backend libraries. Preserve archive and payload hashes with
the binary/source lock. Different llama.cpp revisions/builds are part of this comparison;
identical GGUF/settings do not establish identical kernels or output text.

## Source contract and required probes

The immutable references below support source assertions. Every row also needs real probes.

| Requirement | Source evidence | Required runtime evidence |
|---|---|---|
| Streaming, cache and units | Ollama [terminal mapping](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L1824), [cache total](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L1595); standalone [timings](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/tools/server/server-common.cpp#L82) | One final, first content; total/cached/evaluated native counts; Ollama ns versus llama ms |
| Preload, context and unload | Ollama [empty preload](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/routes.go#L503), [shift reload](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/sched.go#L1422) | No generation during load; effective context 4096; stable owned PIDs, removed residency |
| Greedy sampling | Ollama [request mapping](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L1654); bundled [defaults](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/common/common.h#L223), [chain](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/common/sampling.cpp#L340); standalone [chain](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/common/sampling.cpp) | Temp 0, top-k 1, top-p 1, min-p 0, seed 42; penalties/DRY/XTC/mirostat inactive |
| Batch, F16 KV, flash, offload | Ollama [launch](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L397), [batch/flash](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L584); standalone [flags](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/common/arg.cpp) | Batch/microbatch 128, K/V F16, flash off, one slot; complete offload and owned VRAM |
| Raw, BOS, special tokens | Ollama [raw route](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/routes.go#L545), [BOS handling](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L246); bundled [tokenization](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/tools/server/server-common.cpp#L867) | Identical prompt bytes/hash; reference token IDs/counts; evicted-prefix cache within protocol |
| Truncation and shift disabled | Ollama [propagation](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/routes.go#L703), [prompt handling](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L282); standalone [shift flag](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/common/arg.cpp#L1741) | False at preload and requests; over-context rejection without shortening |
| Speculation, stops, thinking | Ollama [draft guard](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L806), [raw parser guard](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/routes.go#L495); standalone [spec flag](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/common/arg.cpp#L4264) | No draft decoding; same stops/budgets; Qwen3 closed thinking block, no thinking emitted |

Ollama's longer sampler chain has inactive extra samplers under the reviewed greedy
defaults. Do not generalize that equivalence to other profiles. Preserve effective
settings and terminal payloads in CPU/GPU probes.

This Ollama preload response has no native load duration; wall preload-to-residency is
measurable. Request load duration includes scheduler work; llama load wall time includes
process startup. Preserve native decode boundaries and compare output lengths, client
TTFT and end-to-end latency. Ollama may abort excessive repeated chunks; retain failures.

## Host and validation status

The inspected host has Fedora 44, GCC 16.2.1, CMake 4.3.0 and CUDA 13.4.92 under
`/usr/local/cuda-13.4`. Fedora 44/GCC 16 appears in the
[CUDA 13.4 support matrix](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
Put that toolkit directory on PATH for build, pin, doctor and campaigns. NVML recognizes
the 4 GiB RTX 3050 Laptop GPU. Desktop/Steam workloads were present and were not stopped.

Preload omitted `shift` and used default true, while measured requests used false; source
shows reload on that difference. The adapter now uses false at preload too, with a
contract regression test. The retained CPU probe (`cpu-probe-03`) passed context 4096,
streaming/final-event checks, reference input/cache counts, native decode-rate checks,
stable owned processes and unload checks in both engines. The earlier failed probes
remain local diagnostics. GPU lifecycle, pilots and official campaigns remain pending.
No downloadable probe bundle is published; reproduce the CPU check using the
[integration instructions](../guides/reproduction.md). Do not freeze from CPU/source evidence alone.

The selected servers' native generation rate divides `max(0, predicted_n - 1)` by
generation duration: the first token is sampled from prefill. Ollama forwards both
`predicted_n` as `eval_count` and that native duration. The harness now retains output
count separately and matches the native decoding-step denominator. References:
[standalone stats](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/tools/server/server-common.h#L433),
[bundled stats](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/tools/server/server-common.h#L415).

Source review also found the bundled server defaults to an 8192 MiB RAM prompt cache;
restoring an old prefix can defeat eviction. Ollama forwards its process environment to
the child server. The adapter explicitly sets `LLAMA_ARG_CACHE_RAM=0`, `LLAMA_ARG_FIT=off`,
`LLAMA_ARG_LOAD_MODE=mmap` and `LLAMA_ARG_LAZY_MODE=off`, matching the standalone flags.
These fixed overrides do not inherit ambient `LLAMA_ARG_*` values. References:
[child environment](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L448),
[RAM-cache override](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/common/arg.cpp#L1713),
[fit override](https://github.com/ggml-org/llama.cpp/blob/6f767fe960c3b97cf37fac4626c86400561ca1e4/common/arg.cpp#L2863).
Native cache counts and effective context/offload must still pass the real pilot.

Ollama maps `num_batch` to both `-b` and `-ub`. Real configurations containing Ollama
therefore reject unequal batch/microbatch sizes instead of accepting an ineffective
microbatch setting. Standalone llama.cpp can retain separate sizes.

The selected llama.cpp source logs per-sequence context as `n_ctx_seq`; older sources
used `n_ctx_per_seq`. The adapter recognizes both labels and does not infer per-sequence
capacity from global `n_ctx`. Source:
[context log](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/src/llama-context.cpp#L310).

Real CPU integration exposed suppressed startup evidence: standalone verbosity 3 hides
library INFO messages because the log bridge maps them to trace level 4. Its launch now
explicitly uses verbosity 4, matching Ollama's bundled server, retaining native context,
KV and offload logs. References:
[log bridge](https://github.com/ggml-org/llama.cpp/blob/b92761a515ea31e852e7fbc1fad5f874b46f3718/common/log.cpp#L529),
[Ollama launch verbosity](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go#L575).
