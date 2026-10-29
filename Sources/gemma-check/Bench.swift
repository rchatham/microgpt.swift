import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MicroGemma
import Tokenizers

/// Prefill and decode rates for one generation, measured the same way for both models:
/// prefill runs until the first new token is known, decode from then until the last.
struct BenchRun {
    var prefillTokensPerSecond: Double
    var decodeTokensPerSecond: Double
    var tokens: [Int]
}

func benchMicroGemma(_ model: MicroGemma.Gemma3TextModel, prompt: [Int], count: Int) -> BenchRun {
    var tokens: [Int] = []
    let stats = model.generate(prompt: prompt, maxTokens: count) { tokens.append($0); return true }
    return BenchRun(
        prefillTokensPerSecond: stats.prefillTokensPerSecond,
        decodeTokensPerSecond: stats.decodeTokensPerSecond, tokens: tokens)
}

func benchReference(_ model: MLXLLM.Gemma3TextModel, prompt: [Int], count: Int) throws -> BenchRun {
    let start = Date()
    var iterator = try TokenIterator(
        input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
        parameters: GenerateParameters(maxTokens: count, temperature: 0))
    var tokens = [iterator.next()!]
    let firstToken = Date()
    while let t = iterator.next() { tokens.append(t) }
    let end = Date()
    return BenchRun(
        prefillTokensPerSecond: Double(prompt.count) / firstToken.timeIntervalSince(start),
        decodeTokensPerSecond: Double(tokens.count - 1) / end.timeIntervalSince(firstToken),
        tokens: tokens)
}

/// Splits one MicroGemma decode step into the CPU time spent building the lazy graph and
/// the time spent waiting for the GPU to finish it. Prints milliseconds per step.
func profileMicroGemma(_ model: MicroGemma.Gemma3TextModel, prompt: [Int], count: Int) {
    let caches = model.makeCaches()
    var logits = model(MLXArray(prompt.map(Int32.init))[.newAxis], caches: caches)
    var build = 0.0, wait = 0.0
    for _ in 0..<count {
        let next = argMax(logits[0, -1]).item(Int.self)
        let t0 = Date()
        logits = model(MLXArray([Int32(next)])[.newAxis], caches: caches)
        let t1 = Date()
        eval([logits] + caches.flatMap(\.state))  // MLX evaluation of the step just built
        let t2 = Date()
        build += t1.timeIntervalSince(t0)
        wait += t2.timeIntervalSince(t1)
    }
    print(String(
        format: "  MicroGemma per step: graph build %.2f ms, GPU wait %.2f ms",
        build / Double(count) * 1000, wait / Double(count) * 1000))
}

/// Times the streaming detokenization `Translator.translate` does: re-decoding the whole
/// reply after every token. Prints the average and final per-token cost in milliseconds.
func profileDetokenize(_ tokenizer: any Tokenizers.Tokenizer, tokens: [Int]) {
    var total = 0.0, last = 0.0
    for n in 1...tokens.count {
        let start = Date()
        _ = tokenizer.decode(tokens: Array(tokens[..<n]), skipSpecialTokens: true)
        last = Date().timeIntervalSince(start)
        total += last
    }
    print(String(
        format: "  streaming detokenize: %.2f ms/token average, %.2f ms at %d tokens",
        total / Double(tokens.count) * 1000, last * 1000, tokens.count))
}

func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

/// Generates `count` tokens (no stop token) from each prompt with both models, `runs` times
/// after a warm-up, and prints median rates. Greedy decoding, so the token streams should
/// match except where bf16 near-ties flip an argmax.
func runBenchmark(
    ours: MicroGemma.Gemma3TextModel, reference: MLXLLM.Gemma3TextModel,
    tokenizer: any Tokenizers.Tokenizer, prompts: [(String, [Int])], count: Int, runs: Int
) throws {
    _ = benchMicroGemma(ours, prompt: prompts[0].1, count: 8)
    _ = try benchReference(reference, prompt: prompts[0].1, count: 8)

    for (name, prompt) in prompts {
        var mine: [BenchRun] = [], theirs: [BenchRun] = []
        // Alternate which model goes first, and compare runs in adjacent pairs, so load from
        // other processes affects both sides about equally.
        for run in 0..<runs {
            if run.isMultiple(of: 2) {
                mine.append(benchMicroGemma(ours, prompt: prompt, count: count))
                theirs.append(try benchReference(reference, prompt: prompt, count: count))
            } else {
                theirs.append(try benchReference(reference, prompt: prompt, count: count))
                mine.append(benchMicroGemma(ours, prompt: prompt, count: count))
            }
        }
        let same = zip(mine[0].tokens, theirs[0].tokens).prefix { $0 == $1 }.count
        print("\n\(name): \(prompt.count) prompt tokens, \(count) generated, median of \(runs)")
        for (label, r) in [("MicroGemma  ", mine), ("mlx-swift-lm", theirs)] {
            print(String(
                format: "  %@  prefill %6.0f tok/s   decode %6.1f tok/s", label,
                median(r.map(\.prefillTokensPerSecond)), median(r.map(\.decodeTokensPerSecond))))
        }
        let ratios = zip(mine, theirs).map { $0.decodeTokensPerSecond / $1.decodeTokensPerSecond }
        print(String(
            format: "  decode speed, MicroGemma / mlx-swift-lm: median %.2fx (pairs %@)",
            median(ratios), ratios.map { String(format: "%.2f", $0) }.joined(separator: " ")))
        print("  identical greedy prefix: \(same)/\(count) tokens")
        profileMicroGemma(ours, prompt: prompt, count: 64)
        profileDetokenize(tokenizer, tokens: mine[0].tokens)
    }
}
