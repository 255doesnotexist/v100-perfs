#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pkill -f 'vllm.entrypoints.openai.api_server' || true
sleep 2

source "$SCRIPT_DIR/.venv-1cat-120/bin/activate"

export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600

VLLM_SM70_AWQ_TURBOMIND=1 exec python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 20400 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enforce-eager \
  --disable-custom-all-reduce \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
