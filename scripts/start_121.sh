#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing vLLM server to avoid port conflicts
pkill -f 'vllm.entrypoints.openai.api_server' || true
sleep 2

source .venv-1cat/bin/activate

# Keep-alive timeout: 5s default kills long-running auxiliary calls (compression, etc.)
# Set to 300s (5 min) to survive context compression and other slow operations.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600

VLLM_SM70_AWQ_TURBOMIND=1 exec python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 98784 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-cache-gb 0 \
  --cpu-offload-gb 0 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
