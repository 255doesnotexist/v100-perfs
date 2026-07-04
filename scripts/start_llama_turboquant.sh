#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# This binary was built with the conda tsenv CUDA 12.9 toolchain,
# so we must load its runtime libraries (libcudart, libcublas, libcudnn).
TSENV_LIBS="/home/ezra/.conda/envs/tsenv/lib"
export LD_LIBRARY_PATH="$TSENV_LIBS:${LD_LIBRARY_PATH:-}"

# TurboQuant fork build. For comparison against upstream llama.cpp no-spec.
# Supports --cache-type-k/-v {turbo2,turbo3,turbo4} for KV-cache compression.

# Default: balanced quality/compression. Override via env:
#   TURBO_MODEL=...                     # model GGUF path (default Q3_K_M)
#   TURBO_CACHE_K=turbo3 TURBO_CACHE_V=turbo3
#   TURBO_CTX=131072 TURBO_SLOTS=2
#   USE_YARN=1 YARN_SCALE=2
#   PROMPT_CACHE_MB=0                   # disable prompt cache to save VRAM (minor effect)
#   USE_MTP=1 MTP_DRAFT_N_MAX=2         # requires MTP-compatible GGUF (e.g. IQ4_XS MTP)

MODEL="${TURBO_MODEL:-./models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q3_K_M.gguf}"

CACHE_K="${TURBO_CACHE_K:-turbo4}"
CACHE_V="${TURBO_CACHE_V:-turbo4}"
CTX="${TURBO_CTX:-65536}"
SLOTS="${TURBO_SLOTS:-2}"

MTP_ARGS=()
if [[ "${USE_MTP:-0}" == "1" ]]; then
    MTP_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "${MTP_DRAFT_N_MAX:-2}")
fi

YARN_ARGS=()
if [[ "${USE_YARN:-0}" == "1" ]]; then
    SCALE="${YARN_SCALE:-2}"
    YARN_ARGS+=(
        --rope-scaling yarn
        --rope-scale "$SCALE"
        --yarn-orig-ctx 262144
    )
fi

PROMPT_CACHE_MB="${PROMPT_CACHE_MB:-8192}"
PROMPT_CACHE_ARG=()
if [[ "$PROMPT_CACHE_MB" == "0" ]]; then
    PROMPT_CACHE_ARG=(--cache-ram 0)
else
    PROMPT_CACHE_ARG=(--cache-ram "$PROMPT_CACHE_MB")
fi

exec ./llama.cpp-turboquant/build/bin/llama-server \
  -m "$MODEL" \
  --mmproj ./models/Qwen3.6-27B-DFlash-GGUF/mmproj-BF16.gguf \
  -c "$CTX" -ngl 99 -np "$SLOTS" -fa on \
  --cache-type-k "$CACHE_K" --cache-type-v "$CACHE_V" \
  "${YARN_ARGS[@]}" \
  "${PROMPT_CACHE_ARG[@]}" \
  "${MTP_ARGS[@]}" \
  --jinja \
  --alias qwen3.6-27b-awq \
  --host 0.0.0.0 --port 8000 \
  "$@"
