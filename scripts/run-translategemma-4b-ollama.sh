#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-translategemma:latest}
PROMPT=${1:-"The weather is beautiful today."}
SOURCE=${SOURCE:-English}
TARGET=${TARGET:-Spanish}

if ! command -v ollama >/dev/null 2>&1; then
  echo "error: ollama not found" >&2
  echo "install: https://ollama.com" >&2
  exit 1
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -qx "$MODEL"; then
  echo "error: Ollama model not found: $MODEL" >&2
  echo "available models:" >&2
  ollama list >&2
  exit 1
fi

REQUEST="Translate from $SOURCE to $TARGET. Output only the translation, with no explanation.\n\n$PROMPT"

exec ollama run "$MODEL" --nowordwrap "$REQUEST"
