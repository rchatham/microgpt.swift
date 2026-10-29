import Foundation
import MLX

public struct GenerationStats: Sendable {
    public var promptTokens = 0
    public var generatedTokens = 0
    /// From the start of the call until the first new token is known.
    public var prefillSeconds = 0.0
    /// From the first new token until the last.
    public var decodeSeconds = 0.0

    public var prefillTokensPerSecond: Double { Double(promptTokens) / prefillSeconds }
    /// The first token comes out of prefill, so decoding produced one fewer.
    public var decodeTokensPerSecond: Double { Double(max(generatedTokens - 1, 0)) / decodeSeconds }
}

extension Gemma3TextModel {
    /// Runs `prompt` through the model and then generates up to `maxTokens` tokens, passing
    /// each to `onToken`. Return `false` from `onToken` to stop early.
    ///
    /// Steps are pipelined: the graph for token n+1 is built on the still-unknown token n and
    /// queued with `asyncEval` before we wait for token n. MLX encodes Metal work on the CPU,
    /// so this overlaps that encoding (and our Swift) with the GPU running the previous step.
    /// The cost is one speculative step computed after the last token, which is discarded.
    @discardableResult
    public func generate(
        prompt: [Int], maxTokens: Int, sampling: Sampling = .greedy,
        onToken: (Int) -> Bool
    ) -> GenerationStats {
        var stats = GenerationStats(promptTokens: prompt.count)
        guard maxTokens > 0 else { return stats }
        let caches = makeCaches()

        var clock = Date()
        let logits = self(MLXArray(prompt.map(Int32.init))[.newAxis], caches: caches)
        var token = sampling.token(from: logits[0, -1])
        asyncEval(token)

        for step in 0..<maxTokens {
            var nextToken: MLXArray?
            if step < maxTokens - 1 {
                let logits = self(token.reshaped(1, 1), caches: caches)
                nextToken = sampling.token(from: logits[0, -1])
                asyncEval([nextToken!] + caches.flatMap(\.state))
            }

            let id = token.item(Int.self)  // waits for this token only, not the queued step
            if step == 0 {
                stats.prefillSeconds = Date().timeIntervalSince(clock)
                clock = Date()
            }
            stats.generatedTokens += 1
            guard onToken(id), let nextToken else { break }
            token = nextToken
        }

        stats.decodeSeconds = Date().timeIntervalSince(clock)
        return stats
    }
}
