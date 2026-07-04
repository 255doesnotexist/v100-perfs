#!/usr/bin/env bash
set -euo pipefail

# Production-ready TurboQuant config: 4 slots × 140K context per slot on V100-32GB.
# Verified: startup VRAM ~30.0 GB, 4-agent stress peak ~30.2 GB.
#
# Usage:
#   ./start_llama_turboquant_4x140k.sh
#   PORT=8080 ./start_llama_turboquant_4x140k.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"

# 4 slots × 140K context = 573,440 total KV cache budget
export TURBO_CTX=573440
export TURBO_SLOTS=4
export TURBO_CACHE_K=turbo3
export TURBO_CACHE_V=turbo3
export PROMPT_CACHE_MB=0

# Keep the same model alias as vLLM for client compatibility
export TURBO_MODEL="${TURBO_MODEL:-./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf}"
PORT="${PORT:-8000}"

exec ./llama.cpp-turboquant/build/bin/llama-server \
  -m "$TURBO_MODEL" \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c "$TURBO_CTX" -ngl 99 -np "$TURBO_SLOTS" -fa on \
  --cache-type-k "$TURBO_CACHE_K" --cache-type-v "$TURBO_CACHE_V" \
  --cache-ram 0 \
  --jinja \
  --chat-template-file ./chat_templates/qwen3.6_merged.jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port "$PORT" \
  "$@"
