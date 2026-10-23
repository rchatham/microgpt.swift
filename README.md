# microgpt.swift

Two ends of the same idea in Swift:

- **`microgpt.swift`**: a dependency-free port of Karpathy's microgpt. It uses scalar autograd to train a 1-layer GPT on names.
- **`MicroGemma`**: Gemma 3 inference written in Swift on top of [MLX](https://github.com/ml-explore/mlx-swift) arrays. It runs Google's TranslateGemma. MLX supplies the tensors, Metal kernels and quantized layers; the model, KV cache, weight loading and prompt are in [`Sources/MicroGemma`](Sources/MicroGemma).

## Build

MLX's Metal shaders only compile under `xcodebuild` (not `swift build`). The package pins `mlx-swift` 0.31.x because newer releases need Swift tools 6.3.

```bash
xcodebuild build -scheme translate -destination 'platform=macOS' -configuration Release -derivedDataPath .build/xcode -skipMacroValidation
```

`microgpt.swift` has no dependencies and still runs with `swift run microgpt.swift`.

## Translate

Download [mlx-community/translategemma-4b-it-4bit](https://huggingface.co/mlx-community/translategemma-4b-it-4bit) (2.2 GB) to `~/.cache/microgpt.swift/translategemma-4b-it-4bit`, or pass `--model DIR` / set `MICROGEMMA_MODEL`. The 12B and 27B conversions load the same way.

```bash
.build/xcode/Build/Products/Release/translate --from en --to fr "Language models can now run on a laptop."
```

Language codes come from the model's own `chat_template.jinja` (`en`, `pt-BR`, `zh-Hant`, …). Text can also come on stdin.

Decoding is greedy by default, which is usually best for translation. `--sample` switches to top-k/top-p sampling with TranslateGemma's suggested k = 64 and p = 0.95, which gives varied phrasings with the same meaning. `gemma-check --sampling` tests the sampler on a toy vocabulary, against a plain-Swift reference, without loading the model.

## Check against the reference

`gemma-check` runs MicroGemma and `mlx-swift-lm`'s `Gemma3TextModel` on the same tokens and diffs their logits. It also compares the KV cache against a full forward pass beyond the 1024-token sliding window. `mlx-swift-lm` is only a dependency of this tool.

```bash
xcodebuild build -scheme gemma-check -destination 'platform=macOS' -configuration Release -derivedDataPath .build/xcode -skipMacroValidation
.build/xcode/Build/Products/Release/gemma-check
```

`gemma-check --bench` times greedy generation (256 tokens, short and 1275-token prompts) for both implementations. It alternates which one goes first, reports the median of paired MicroGemma/mlx-swift-lm decode ratios, and splits a MicroGemma decode step into CPU graph-building time and GPU wait. Decoding is pipelined (step n+1 is queued with `asyncEval` before token n is read), and the KV cache writes into preallocated buffers; sliding layers use a 1024-slot ring.

TranslateGemma's config gives its ×8 linear RoPE factor only under the newer `rope_parameters` key. `mlx-swift-lm` 3.31.4 reads only the legacy `rope_scaling` key, so as shipped it drops the factor on the global-attention layers. MicroGemma reads both keys. With the factor injected into the reference, the two implementations' logits are identical.
