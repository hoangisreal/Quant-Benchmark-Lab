# Prepare the lab toolchain and models

[Documentation home](../../README.md) · Previous: [synthetic walkthrough](getting-started.md)
· Next: [real campaigns](reproduction.md)

## 1. Establish the host prerequisites

The reference target is Linux with an RTX 3050 reporting approximately 4 GiB VRAM.
Install Git, CMake, a C/C++ compiler, a working NVIDIA driver and a compatible CUDA Toolkit.
Use the [NVIDIA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
for your distribution and its supported host compiler. Record the versions you install.

```bash
nvidia-smi
nvcc --version
c++ --version
cmake --version
```

`nvidia-smi` confirms driver visibility. Its displayed CUDA version does not establish that
the Toolkit compiler is installed: `nvcc` must also be available on `PATH` for a CUDA build.
Python CUDA libraries alone do not provide the required compiler. If a Toolkit is installed
outside `PATH`, add its actual `bin` directory to your shell configuration before continuing.

Keep unrelated GPU workloads stopped during measurement, connect laptop power and record
the power/display setup. The harness records idle memory and temperature; desktop memory
is included in device-wide VRAM. A GPU visible to the driver can still fail an NVML probe.

## 2. Install the candidate engine versions

The current source-reviewed candidate is Ollama **v0.35.1** with standalone llama.cpp
**b92761a515ea31e852e7fbc1fad5f874b46f3718**. Real CPU integration has passed locally;
GPU equivalence and official performance remain unverified. See the
[compatibility guide](../reference/engine-compatibility.md) for evidence and remaining probes.

The following examples target a fresh Linux AMD64 checkout. They install Ollama locally,
without a system service. You need `curl`, `tar` and `zstd` in addition to the prerequisites
above. If these directories already contain the same verified candidate, reuse them;
inspect a different installation before replacing anything.
Stop at any failed command. Extract the archive only after its checksum check passes.

```bash
export QBL_LLAMA_COMMIT=b92761a515ea31e852e7fbc1fad5f874b46f3718
export QBL_OLLAMA_RELEASE=v0.35.1
mkdir -p vendor
curl -fL --retry 3 \
  "https://github.com/ollama/ollama/releases/download/$QBL_OLLAMA_RELEASE/ollama-linux-amd64.tar.zst" \
  -o "vendor/ollama-linux-amd64-$QBL_OLLAMA_RELEASE.tar.zst"
sha256sum --check <<'EOF'
9fcd79ac4575b2bd31b992eee18b1000c8ad126b451627c8f8cd091714cfbb10  vendor/ollama-linux-amd64-v0.35.1.tar.zst
EOF
mkdir "vendor/ollama-release-$QBL_OLLAMA_RELEASE"
tar --zstd -xf "vendor/ollama-linux-amd64-$QBL_OLLAMA_RELEASE.tar.zst" \
  -C "vendor/ollama-release-$QBL_OLLAMA_RELEASE"
export PATH="$PWD/vendor/ollama-release-$QBL_OLLAMA_RELEASE/bin:$PATH"
ollama --version
bash scripts/bootstrap_llama.sh "$QBL_LLAMA_COMMIT" cuda
```

Keep this PATH in the shell used for pinning, doctor, tests and campaigns. Keep the
archive's `bin/` and `lib/ollama/` directories together. For a system installation or
another architecture, consult the [official Linux guide](https://docs.ollama.com/linux)
and review compatibility rather than reusing the AMD64 checksum.

The harness creates its own daemon on port 11435 and uses port 8081 for llama-server;
both must be free. It imports local GGUF files itself; no `ollama pull` is required.

The script builds `llama-server` and `llama-quantize` under `vendor/llama.cpp/build/bin`,
using Release mode and CUDA compute capability 8.6. See the upstream
[build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
when resolving build failures; source at the chosen commit governs compatibility.

Inspect `backends/llamacpp.py` and `backends/ollama.py` under `src/quant_benchmark_lab/`
against the selected sources. Required behavior includes native streaming usage/cache
counters, special-token handling, context reporting, full-offload evidence and the actual
launch flags. Unsupported behavior is a compatibility blocker, not permission to invent
missing metrics or silently remove controls.

## 3. Pin binaries and audit settings

Keep a clean Ollama source checkout at the installed release tag as well. Use the
`QBL_OLLAMA_RELEASE` set above. Existing checkouts must have clean source and HEAD at
the corresponding tag; skip cloning when that condition is already satisfied:

```bash
git clone --depth 1 --branch "$QBL_OLLAMA_RELEASE" https://github.com/ollama/ollama.git vendor/ollama
uv run --frozen python scripts/pin_toolchain.py --source vendor/llama.cpp \
  --ollama-source vendor/ollama --ollama-release "$QBL_OLLAMA_RELEASE"
test -f configs/settings_audit.json || cp configs/settings_audit.example.json configs/settings_audit.json
```

Pinning requires both engine binaries, clean llama.cpp and tagged Ollama checkouts, the quantizer and
`CMakeCache.txt`. It writes `locks/toolchain.json` with hashes, source commit, compiler,
CUDA details, Ollama release/source references, native payload hashes and server help.
This file and the generated `locks/models.json` are local and ignored because they contain
build-specific absolute paths. The tracked `locks/toolchain.example.json` is unresolved;
it cannot certify another host. Preserve generated locks with campaign snapshots.
Shared implementations and discovered GPU backends are included; an unchanged launcher
hash cannot hide changed native code. Old pinned locks without payload provenance must
be repinned before real runs. For a nonstandard Ollama layout, use `--ollama-payload`
to identify its native `lib/ollama` directory. It does not approve runtime equivalence. If Ollama has a
custom path, pass `--ollama /absolute/path/to/ollama` and set the same runtime binary.

Review each engine in `configs/settings_audit.json`. The supplied review is for the
candidate sources, with hashes from the local build. A fresh build can have different
binary hashes; bind the review to your actual binaries only after checking the sources
and effective controls. Do not replace an existing review with the example:

| Field | Evidence required |
|---|---|
| `binary_sha256` | The corresponding binary SHA from `locks/toolchain.json` |
| `source_references` | Nonempty list of immutable source references with commit, path and line or source digest |
| `samplers` | Greedy sampler behavior/order and disabled penalties/additional sampling |
| `kv_f16` | Actual F16 K/V cache behavior |
| `batch` | Batch/microbatch semantics for the requested runtime |
| `no_speculation` | Speculative decoding disabled or established inapplicable |
| `bos_policy` | BOS/special-token handling consistent with reference tokenization |
| `truncate_disabled` | Oversized prompts cannot silently truncate/shift |
| `reviewed` | Set true only after the above review is complete |

A list entry can be an immutable source URL plus an explanation of the relevant lines.
Keep unverified fields false and record unsupported behavior. The harness checks these
assertions and source references but cannot prove their truth; real probes remain required.
Preserve the reviewed file as public methodology evidence, then bind it into the lock:

```bash
uv run --frozen python scripts/pin_toolchain.py --source vendor/llama.cpp \
  --ollama-source vendor/ollama --ollama-release "$QBL_OLLAMA_RELEASE" \
  --audit configs/settings_audit.json
```

Do not repin without `--audit` afterward: that replaces the saved audit with an empty object.

## 4. Prepare immutable GGUF artifacts

Install the separate conversion dependency group before collecting any pilot evidence:

```bash
uv sync --frozen --python 3.11 --group dev --group prepare
uv run --frozen --group prepare python scripts/prepare_models.py --model qwen-0_5b --dry-run
uv run --frozen --group prepare python scripts/prepare_models.py --model qwen-0_5b
uv run --frozen --group prepare python scripts/prepare_models.py --model qwen-1_5b --quants Q4_K_M
uv run --frozen --group prepare python scripts/prepare_models.py --model qwen-3b --quants Q4_K_M
uv run --frozen --group prepare python scripts/prepare_models.py --model qwen-4b --quants Q4_K_M
```

`--dry-run` requires the pinned local toolchain, but does not download weights. Actual
preparation downloads the source revision, checks its license, derives the official
non-thinking template, converts F16 and quantizes directly from that F16 parent. Q4-only
requests still create an F16 ancestor. The 0.5B default produces all three quantizations.

The 3B source card labels its license `other`, specifically the
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE).
It limits use to noncommercial research/evaluation. Retain that license identity;
do not describe all four models as Apache-licensed. Source weights and generated GGUF
files stay local; this repository publishes methodology and benchmark evidence.

Budget disk for source weights, F16, selected quants, temporary outputs and campaign-local
Ollama imports. The script estimates a minimum conversion disk budget from parameter count;
its RAM check is only a basic guard, not proof that conversion will fit. No measured runtime
or total storage estimate is available yet.

Outputs include `models/gguf/<model-id>/`, creation manifests beside each GGUF, a resolved
`configs/models.yaml`, and provenance in `locks/models.json`. Existing artifacts must match
their provenance; mismatches and partial files require inspection rather than overwrite.
Weights and source caches remain ignored. Preserve manifests, resolved catalogs and locks
for reproduction; inspect absolute host paths before publishing them.

## 5. Verify readiness

```bash
uv run --frozen qbl doctor --require-gpu --config configs/experiments/engine.yaml --output /tmp/qbl-lab-environment.json
uv run --frozen qbl validate --config configs/experiments/engine.yaml
```

Doctor checks GPU availability and records versions; it does not validate every engine or
artifact. Config validation is structural. Continue to CPU integration and GPU preflight
in [reproduction](reproduction.md) before freezing any official experiment.
