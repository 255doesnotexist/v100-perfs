#!/usr/bin/env bash
set -euo pipefail

# Serial multi-engine agent benchmark runner.
# Runs agent_bench.py against one engine at a time to keep GPU measurements clean.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="${1:-/tmp/agent_bench_results}"
mkdir -p "$OUT_DIR"

BENCH="$SCRIPT_DIR/agent_bench.py"
PYTHON="$SCRIPT_DIR/.venv-1cat-120/bin/python"

wait_health() {
  local url=$1 deadline
  deadline=$(($(date +%s) + 300))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s "$url/health" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q '^200$'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_benchmark_set() {
  local name=$1 endpoint=$2 model=$3 max_model_len=$4 shared_prefix=$5 turns=$6 max_tokens=$7
  shift 7
  for agents in 1 2 4; do
    local out="$OUT_DIR/${name}_agents${agents}.json"
    echo "=== $name | agents=$agents | max_len=$max_model_len | shared=$shared_prefix | turns=$turns ==="
    "$PYTHON" "$BENCH" \
      --endpoint "$endpoint" \
      --model "$model" \
      --num-agents "$agents" \
      --max-model-len "$max_model_len" \
      --shared-prefix-tokens "$shared_prefix" \
      --turns "$turns" \
      --max-tokens "$max_tokens" \
      --compact-ratio 0.6 \
      --warmup-turns 1 \
      --output "$out"
    sleep 5
  done
}

# Profile 1: 1Cat-vLLM 1.2.0 (multimodal, no MTP, max-model-len=51744)
echo ""
echo "#############################################"
echo "# Profile 1: 1Cat-vLLM 1.2.0 (multimodal)   #"
echo "#############################################"
./start.sh > "$OUT_DIR/v120.log" 2>&1 &
wait_health http://127.0.0.1:8000
# shared prefix ~45% of max len; should trigger compaction at 60% within ~35 turns
run_benchmark_set "v120" "http://127.0.0.1:8000" "qwen3.6-27b-awq" 51744 23285 35 256
pkill -f 'vllm.entrypoints.openai.api_server' || true
sleep 5

# Profile 2: 1Cat-vLLM 1.2.1 (multimodal, fp8_e5m2, max-model-len=98784)
echo ""
echo "#############################################"
echo "# Profile 2: 1Cat-vLLM 1.2.1 (multimodal)   #"
echo "#############################################"
./start_121.sh > "$OUT_DIR/v121.log" 2>&1 &
if wait_health http://127.0.0.1:8000; then
  run_benchmark_set "v121" "http://127.0.0.1:8000" "qwen3.6-27b-awq" 98784 44453 52 256
else
  echo "1.2.1 failed to start, skipping"
fi
pkill -f 'vllm.entrypoints.openai.api_server' || true
sleep 5

# Profile 3: llama.cpp without speculative decoding, with multimodal
echo ""
echo "#############################################"
echo "# Profile 3: llama.cpp no-spec + multimodal #"
echo "#############################################"
./start_llama_nospec.sh > "$OUT_DIR/llama_nospec.log" 2>&1 &
if wait_health http://127.0.0.1:8000; then
  # llama.cpp c=65536; n_slots=2 means 32768 per slot
  run_benchmark_set "llama_nospec" "http://127.0.0.1:8000" "qwen3.6-27b-awq" 65536 29491 40 256
else
  echo "llama.cpp no-spec failed to start, skipping"
fi
pkill -f 'llama-server' || true
sleep 5

echo ""
echo "All benchmarks complete. Results in $OUT_DIR"
ls -la "$OUT_DIR"
