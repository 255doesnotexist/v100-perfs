#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LD_LIBRARY_PATH="/home/ezra/.conda/envs/tsenv/lib:${LD_LIBRARY_PATH:-}"

export AUTH_TOKEN="${AUTH_TOKEN:-CHANGE_ME}"
export BACKEND_PORT="${BACKEND_PORT:-8001}"
export PROXY_PORT="${PROXY_PORT:-8000}"

exec .venv-1cat/bin/python thinking_proxy.py
