#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing llama-server to avoid port conflicts
pkill -f 'llama-server' || true
sleep 2

# Use uv venv CUDA 12.8 libs (required for V100/SM70)
VENV_LIBS="$SCRIPT_DIR/.venv-1cat/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$VENV_LIBS/cuda_runtime/lib:$VENV_LIBS/cublas/lib:$VENV_LIBS/cudnn/lib:${LD_LIBRARY_PATH:-}"

# llama.cpp without speculative decoding, with multimodal (mmproj)
# For a fair comparison against vLLM 1.2.0/1.2.1 multimodal configs.
exec ./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c 65536 -ngl 99 -np 2 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000 \
  "$@"
