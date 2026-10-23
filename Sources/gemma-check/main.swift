import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MicroGemma

// Dev tool: runs MicroGemma and mlx-swift-lm's Gemma3TextModel on identical token IDs and
// reports how far apart their logits are. Also checks MicroGemma's KV cache against a plain
// full-sequence forward pass, including past the sliding window.
//
// Usage: gemma-check [--model DIR]

var arguments = Array(CommandLine.arguments.dropFirst())
let modelPath: String = {
    if let i = arguments.firstIndex(of: "--model"), i + 1 < arguments.count { return arguments[i + 1] }
    return ProcessInfo.processInfo.environment["MICROGEMMA_MODEL"]
        ?? NSString(string: "~/.cache/microgpt.swift/translategemma-4b-it-4bit").expandingTildeInPath
}()
let directory = URL(fileURLWithPath: modelPath)

/// Loads the reference model. With `forceLinearRope`, injects the legacy `rope_scaling` key
/// so the reference sees the ×8 factor that TranslateGemma only states in `rope_parameters`.
func loadReference(forceLinearRope: Bool) throws -> MLXLLM.Gemma3TextModel {
    var json = try JSONSerialization.jsonObject(
        with: Data(contentsOf: directory.appending(path: "config.json"))) as! [String: Any]
    if forceLinearRope, var text = json["text_config"] as? [String: Any] {
        text["rope_scaling"] = ["rope_type": "linear", "factor": 8.0]
        json["text_config"] = text
    }
    let config = try JSONDecoder().decode(
        Gemma3TextConfiguration.self, from: JSONSerialization.data(withJSONObject: json))
    let model = MLXLLM.Gemma3TextModel(config)
    let q = (json["quantization"] as? [String: Any]).map {
        BaseConfiguration.Quantization(
            groupSize: ($0["group_size"] as! NSNumber).intValue,
            bits: ($0["bits"] as! NSNumber).intValue)
    }
    try loadWeights(modelDirectory: directory, model: model, quantization: q)
    return model
}

struct Diff: CustomStringConvertible {
    let maxAbs: Float
    let meanAbs: Float
    let argmaxAgree: Int
    let positions: Int

    init(_ a: MLXArray, _ b: MLXArray) {
        let a = a.asType(.float32), b = b.asType(.float32)
        let d = abs(a - b)
        maxAbs = d.max().item(Float.self)
        meanAbs = d.mean().item(Float.self)
        argmaxAgree = (argMax(a, axis: -1) .== argMax(b, axis: -1)).asType(.int32).sum().item(Int.self)
        positions = a.dim(-2)
    }

    var description: String {
        String(format: "max|Δ| %7.4f  mean|Δ| %.5f  argmax agree %d/%d", maxAbs, meanAbs, argmaxAgree, positions)
    }
}

/// Greedy decoding with the reference model, recomputing the whole sequence each step.
func referenceGreedy(_ model: MLXLLM.Gemma3TextModel, _ prompt: [Int], stop: Set<Int>, maxTokens: Int) -> [Int] {
    var tokens = prompt
    var out: [Int] = []
    for _ in 0..<maxTokens {
        let logits = model(MLXArray(tokens.map(Int32.init))[.newAxis], cache: nil)
        let next = argMax(logits[0, -1]).item(Int.self)
        if stop.contains(next) { break }
        out.append(next)
        tokens.append(next)
    }
    return out
}

print("loading MicroGemma from \(directory.path)")
let translator = try await Translator.load(from: directory)
let ours = translator.model
print("global-layer RoPE scale (MicroGemma): \(ours.config.globalRope.scale)")
print("loading reference as shipped, and with rope_scaling injected")
let refShipped = try loadReference(forceLinearRope: false)
let refScaled = try loadReference(forceLinearRope: true)

if arguments.contains("--bench") {
    let short = try translator.promptTokens(
        text: "Could you tell me where the nearest train station is? I think I'm lost, and my "
            + "phone battery died an hour ago, so I can't look it up myself.",
        source: "en", target: "de")
    let long = try translator.promptTokens(
        text: String(
            repeating: "Machine translation has improved dramatically over the past decade, "
                + "moving from phrase tables to neural networks that read whole sentences at once. ",
            count: 50),
        source: "en", target: "fr")
    try runBenchmark(
        ours: ours, reference: refScaled, tokenizer: translator.tokenizer, prompts: [("short", short), ("long", long)],
        count: 256, runs: 9)
    exit(0)
}

let stop = Set([translator.tokenizer.convertTokenToId("<end_of_turn>"), translator.tokenizer.eosTokenId].compactMap { $0 })
let cases: [(String, String, String)] = [
    ("en", "es", "The quick brown fox jumps over the lazy dog."),
    ("en", "de", "Could you tell me where the nearest train station is? I think I'm lost."),
    ("ja", "en", "明日の会議は午後三時に変更になりました。資料を忘れずに持ってきてください。"),
]

for (source, target, text) in cases {
    let tokens = try translator.promptTokens(text: text, source: source, target: target)
    let input = MLXArray(tokens.map(Int32.init))[.newAxis]
    let mine = ours(input, caches: ours.makeCaches())
    print("\n[\(source)→\(target)] \(text)  (\(tokens.count) tokens)")
    print("  vs reference as shipped:   \(Diff(mine, refShipped(input, cache: nil)))")
    print("  vs reference + rope scale: \(Diff(mine, refScaled(input, cache: nil)))")

    let (translation, _) = try translator.translate(text, from: source, to: target, maxTokens: 64)
    let decode = { (ids: [Int]) in translator.tokenizer.decode(tokens: ids, skipSpecialTokens: true) }
    print("  MicroGemma:                \(translation)")
    print("  reference as shipped:      \(decode(referenceGreedy(refShipped, tokens, stop: stop, maxTokens: 64)))")
    print("  reference + rope scale:    \(decode(referenceGreedy(refScaled, tokens, stop: stop, maxTokens: 64)))")
}

/// Runs `tokens` once as a single forward pass and once as a prefill of `split` tokens
/// followed by one-token steps, and diffs the logits for positions `split...`.
/// `makeForward` returns a forward function bound to a fresh cache.
func cacheCheck(_ tokens: [Int], split: Int, _ makeForward: () -> (MLXArray) -> MLXArray) -> Diff {
    let full = makeForward()(MLXArray(tokens.map(Int32.init))[.newAxis])
    let forward = makeForward()
    _ = forward(MLXArray(tokens[..<split].map(Int32.init))[.newAxis])
    var steps: [MLXArray] = []
    for t in tokens[split...] {
        let logits = forward(MLXArray([Int32(t)])[.newAxis])
        eval(logits)  // MLX evaluation, keeps each step's graph small
        steps.append(logits[0..., -1, 0...])
    }
    return Diff(full[0..., split..., 0...], stacked(steps, axis: 1))
}
let oursForward = { () -> (MLXArray) -> MLXArray in
    let caches = ours.makeCaches()
    return { ours($0, caches: caches) }
}
let referenceForward = { () -> (MLXArray) -> MLXArray in
    let cache = refScaled.newCache(parameters: nil)
    return { refScaled($0, cache: cache) }
}

// Cache check. Single-token steps take a different attention kernel than a full pass, so
// bf16 rounding alone makes them drift apart; the reference's own cache shows the expected
// size of that drift. The long prompt crosses the 1024-token sliding window.
let shortTokens = try translator.promptTokens(text: cases[1].2, source: cases[1].0, target: cases[1].1)
print("\nshort prompt: \(shortTokens.count) tokens, steps from 40")
print("  MicroGemma cached vs full: \(cacheCheck(shortTokens, split: 40, oursForward))")
print("  reference cached vs full:  \(cacheCheck(shortTokens, split: 40, referenceForward))")

let paragraph = "Machine translation has improved dramatically over the past decade, "
    + "moving from phrase tables to neural networks that read whole sentences at once. "
let longTokens = try translator.promptTokens(
    text: String(repeating: paragraph, count: 50), source: "en", target: "fr")
let window = ours.config.slidingWindow
print("\nlong prompt: \(longTokens.count) tokens (sliding window \(window)), steps from \(window + 50)")
print("  MicroGemma cached vs full: \(cacheCheck(longTokens, split: window + 50, oursForward))")
print("  reference cached vs full:  \(cacheCheck(longTokens, split: window + 50, referenceForward))")
print("  MicroGemma steps wrapping the ring (from 1000): \(cacheCheck(longTokens, split: 1000, oursForward))")

// Prefill in 300-token chunks: later chunks rewrite a ring that has already wrapped.
let chunkForward = oursForward()
let chunked = stride(from: 0, to: longTokens.count, by: 300).map { start in
    let chunk = longTokens[start ..< min(start + 300, longTokens.count)]
    let logits = chunkForward(MLXArray(chunk.map(Int32.init))[.newAxis])
    eval(logits)  // MLX evaluation of this chunk
    return logits
}
let fullPass = oursForward()(MLXArray(longTokens.map(Int32.init))[.newAxis])
print("  MicroGemma 300-token chunked prefill vs full:   \(Diff(concatenated(chunked, axis: 1), fullPass))")

let longInput = MLXArray(longTokens.map(Int32.init))[.newAxis]
print("  MicroGemma vs reference + rope scale (full pass): \(Diff(ours(longInput, caches: ours.makeCaches()), refScaled(longInput, cache: nil)))")
