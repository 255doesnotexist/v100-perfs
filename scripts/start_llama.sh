#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing llama-server to avoid port conflicts
pkill -f 'llama-server' || true
sleep 2

# Use uv venv CUDA 12.8 libs (required for V100/SM70)
VENV_LIBS="$SCRIPT_DIR/.venv-1cat/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$VENV_LIBS/cuda_runtime/lib:$VENV_LIBS/cublas/lib:$VENV_LIBS/cudnn/lib:$LD_LIBRARY_PATH"

# llama.cpp DFlash configuration (OPTIMAL):
# - Target: Qwen3.6-27B-Q3_K_M (13 GB, from unsloth@ModelScope)
# - Draft:  DFlash bf16 (3.46 GB, converted via convert_hf_to_gguf.py)
# - mmproj: vision encoder (0.93 GB, for multimodal/image input)
# - DFlash: 15 speculative tokens per draft step
# - KV cache: q8_0 (8-bit, halves KV memory, prevents OOM on long prompts)
# - Context: 65536 (with mmproj) -> 32768 usable per slot with -np 2
# - Flash attention: on, all layers on GPU
# - Parallel slots: 2 (best balance of DFlash speed + 5-concurrent throughput)
#
# Performance (V100-32GB):
#   Single short prompt:     ~51 tok/s
#   2x concurrent short:     ~76 tok/s total throughput
#   5x concurrent short:     ~78 tok/s total throughput
#   vLLM AWQ baseline:       21.7 tok/s

exec ./llama.cpp/build/bin/llama-server \
  -m ./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf \
  -md ./models/Qwen3.6-27B-DFlash/dflash.gguf \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  -c 65536 -ngl 99 -np 2 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000 \
  "$@"
