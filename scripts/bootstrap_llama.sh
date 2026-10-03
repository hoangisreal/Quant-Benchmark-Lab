#!/usr/bin/env bash
set -euo pipefail
# Explicit commit only. No floating revision is accepted or checked out.
commit="${1:?Usage: bootstrap_llama.sh <40-character-commit> [cpu|cuda]}"
backend="${2:-cuda}"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected full llama.cpp commit SHA" >&2; exit 2; }
[[ "$backend" == "cuda" || "$backend" == "cpu" ]] || exit 2
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$repo_root/vendor/llama.cpp"
mkdir -p "$repo_root/vendor"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$source_dir"
fi
[[ -z "$(git -C "$source_dir" status --porcelain)" ]] || { echo "llama.cpp source is dirty" >&2; exit 2; }
git -C "$source_dir" fetch origin "$commit"
git -C "$source_dir" checkout --detach "$commit"
cuda_flag=OFF
[[ "$backend" == "cuda" ]] && cuda_flag=ON
cuda_args=()
[[ "$backend" == "cuda" ]] && cuda_args=(-DCMAKE_CUDA_ARCHITECTURES=86)
cmake -S "$source_dir" -B "$source_dir/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA="$cuda_flag" -DLLAMA_CURL=OFF "${cuda_args[@]}"
cmake --build "$source_dir/build" --config Release --target llama-server llama-quantize --parallel 2
echo "Build complete. Pin binary hashes with scripts/pin_toolchain.py before preparing artifacts."
