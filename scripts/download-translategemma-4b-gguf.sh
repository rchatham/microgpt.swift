#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-bullerwins/translategemma-4b-it-GGUF}
FILE=${FILE:-translategemma-4b-it-Q4_K_M.gguf}
OUT_DIR=${OUT_DIR:-models}
OUT_PATH="$OUT_DIR/$FILE"
URL="https://huggingface.co/$REPO/resolve/main/$FILE?download=true"

mkdir -p "$OUT_DIR"

if [[ -f "$OUT_PATH" ]]; then
  echo "already exists: $OUT_PATH"
  exit 0
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" "$FILE" --local-dir "$OUT_DIR" --local-dir-use-symlinks False
else
  curl -L --fail --continue-at - "$URL" -o "$OUT_PATH"
fi

echo "downloaded: $OUT_PATH"
