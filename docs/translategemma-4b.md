# TranslateGemma 4B runner notes

This branch experiments with running Google's TranslateGemma 4B instruction model from Swift/macOS-adjacent tooling.

## Model

Primary target:

- Official gated model: `google/translategemma-4b-it`
- License/terms: Gemma license; accept terms on Hugging Face before using official weights.
- Official format: Safetensors.

For local macOS prototyping this branch starts with a community GGUF conversion:

- `bullerwins/translategemma-4b-it-GGUF`
- Recommended first quant: `translategemma-4b-it-Q4_K_M.gguf`

GGUF + `llama.cpp` is the quickest way to verify the model locally before attempting a native Swift/Metal integration.

## Install prerequisites

```bash
brew install llama.cpp
```

Optional if using Hugging Face tooling directly:

```bash
brew install huggingface-cli
# or: pipx install huggingface-hub
```

## Download GGUF

```bash
./scripts/download-translategemma-4b-gguf.sh
```

This downloads to:

```text
models/translategemma-4b-it-Q4_K_M.gguf
```

## Run weights directly from Swift

This is the repo's main goal: load local GGUF model weights from a Swift executable using `LocalLLMClient`/llama.cpp.

Set up the local Swift llama.cpp wrapper dependency, then build the Swift runner:

```bash
./scripts/setup-localllmclient.sh
swift build -c release --product translategemma-swift
```

`LocalLLMClient` currently depends on llama.cpp targets with unsafe build flags. SwiftPM does not allow those through a normal remote package dependency, so this repo uses a local path dependency under `Vendor/LocalLLMClient` created by the setup script.

Run with a local GGUF file:

```bash
MODEL=models/translategemma-4b-it-Q4_K_M.gguf \
SOURCE=English TARGET=Spanish \
.build/release/translategemma-swift "The weather is beautiful today."
```

Pipe input via stdin:

```bash
echo "Good morning, how are you?" | TARGET=Japanese .build/release/translategemma-swift
```

The Ollama and shell scripts below are only smoke-test/prototyping helpers; they are not the intended final architecture.

## Run with Ollama if you already have the model

If `ollama list` shows `translategemma:latest`, no GGUF download is needed:

```bash
./scripts/run-translategemma-4b-ollama.sh "The weather is beautiful today."
```

Target a different language:

```bash
TARGET=Japanese ./scripts/run-translategemma-4b-ollama.sh "Good morning, how are you?"
```

Use a different Ollama model name:

```bash
MODEL=translategemma:latest SOURCE=English TARGET=French ./scripts/run-translategemma-4b-ollama.sh "I would like a coffee."
```

## Run a GGUF translation smoke test

```bash
./scripts/run-translategemma-4b.sh "Translate into Spanish: The weather is beautiful today."
```

With explicit target/source wording:

```bash
./scripts/run-translategemma-4b.sh "Translate from English to Japanese: Good morning, how are you?"
```

## Tuning knobs

```bash
MODEL=models/translategemma-4b-it-Q4_K_M.gguf \
THREADS=8 \
CTX_SIZE=4096 \
MAX_TOKENS=256 \
TEMP=0.2 \
TOP_P=0.95 \
./scripts/run-translategemma-4b.sh "Translate into French: I would like a coffee."
```

Translation usually benefits from low temperature (`0.0` to `0.3`).

## Next Swift integration paths

1. **CLI bridge first**: call `llama-cli` from Swift `Process` for a working prototype.
2. **llama.cpp Swift package / C API**: remove shell dependency while keeping GGUF.
3. **MLX Swift**: convert official Safetensors to MLX for a more Apple-native Metal path.
4. **MediaPipe LiteRT**: use `litert-community/TranslateGemma-4B-IT` if targeting iOS/MediaPipe.

## Caveats

- Community GGUF conversions are not Google-verified.
- Official weights are gated and governed by the Gemma license.
- Prompt formatting may need adjustment after smoke testing; TranslateGemma/Gemma chat templates can affect quality.
