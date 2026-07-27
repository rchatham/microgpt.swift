#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-models/translategemma-4b-it-Q4_K_M.gguf}
TEXT=${1:-Hello world}
EXPECTED_IDS=${EXPECTED_IDS:-"2 9259 1902"}

output=$(swift run translategemma-native-tokenize "$MODEL" "$TEXT")
printf '%s\n' "$output"

actual_ids=$(printf '%s\n' "$output" | awk -F': ' '/^ids: / { print $2 }')
if [[ "$actual_ids" != "$EXPECTED_IDS" ]]; then
  echo "expected ids: $EXPECTED_IDS" >&2
  echo "actual ids:   $actual_ids" >&2
  exit 1
fi
