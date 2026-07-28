#!/usr/bin/env bash
set -euo pipefail

VERSION=${VERSION:-0.5.0}
DEST=${DEST:-Vendor/LocalLLMClient}
URL=${URL:-https://github.com/tattn/LocalLLMClient.git}

mkdir -p "$(dirname "$DEST")"

if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --tags origin
  git -C "$DEST" checkout "$VERSION"
else
  git clone --branch "$VERSION" --depth 1 "$URL" "$DEST"
fi

chmod -R u+w "$DEST"
python3 - <<'PY'
from pathlib import Path
path = Path('Vendor/LocalLLMClient/Sources/LocalLLMClientLlama/Model.swift')
text = path.read_text()
old = '    func buildChatParams(tools: [AnyLLMTool]) -> UnsafeMutablePointer<llm_chat_params>? {\n        let inputs = create_chat_templates_inputs()'
new = '    func buildChatParams(tools: [AnyLLMTool]) -> UnsafeMutablePointer<llm_chat_params>? {\n        guard !tools.isEmpty else { return nil }\n        let inputs = create_chat_templates_inputs()'
if old in text:
    path.write_text(text.replace(old, new))
PY

echo "LocalLLMClient ready at $DEST ($VERSION)"
