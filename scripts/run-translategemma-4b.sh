#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-models/translategemma-4b-it-Q4_K_M.gguf}
PROMPT=${1:-"Translate into Spanish: The weather is beautiful today."}
THREADS=${THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 8)}
CTX_SIZE=${CTX_SIZE:-4096}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMP=${TEMP:-0.2}
TOP_P=${TOP_P:-0.95}

if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL" >&2
  echo "run: ./scripts/download-translategemma-4b-gguf.sh" >&2
  exit 1
fi

if command -v llama-cli >/dev/null 2>&1; then
  LLAMA_CLI=llama-cli
elif command -v llama >/dev/null 2>&1; then
  LLAMA_CLI=llama
else
  echo "error: llama.cpp CLI not found" >&2
  echo "install: brew install llama.cpp" >&2
  exit 1
fi

exec "$LLAMA_CLI" \
  --model "$MODEL" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --predict "$MAX_TOKENS" \
  --prompt "$PROMPT"
