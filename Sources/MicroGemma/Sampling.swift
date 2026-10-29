import MLX
import MLXRandom

/// How the next token is picked from the model's logits.
public enum Sampling: Sendable {
    /// Always take the most likely token. Deterministic, and usually the right choice for
    /// translation, where there is one meaning to convey rather than many to explore.
    case greedy
    /// Sample from the `k` most likely tokens, further cut to the smallest set whose
    /// probability reaches `p`, after dividing logits by `temperature`.
    /// TranslateGemma's `generation_config.json` suggests k = 64, p = 0.95.
    case topKTopP(k: Int, p: Float, temperature: Float)

    /// Picks one token from `logits`, a `[vocab]` array for the last position.
    ///
    /// Returns a lazy scalar array rather than an `Int`, so the next decode step can be built
    /// on top of it before the GPU has produced it (see `Gemma3TextModel.generate`).
    public func token(from logits: MLXArray) -> MLXArray {
        switch self {
        case .greedy:
            return argMax(logits)
        case .topKTopP(let k, let p, let temperature):
            return sampleTopKTopP(logits, k: k, p: p, temperature: temperature)
        }
    }
}

/// Samples a token from the top-k / nucleus (top-p) distribution of `logits`, returning it as
/// a scalar int array.
///
/// microgpt's `rng.weightedChoice` did this over 27 characters with plain Swift arrays;
/// here the vocabulary is 262k, so the work stays in MLX until a single index is left — and
/// stays lazy (no `.item()`), so decoding can keep running ahead of the GPU.
/// The randomness comes from MLX's global key; call `MLXRandom.seed(_:)` to make it repeatable.
func sampleTopKTopP(_ logits: MLXArray, k: Int, p: Float, temperature: Float) -> MLXArray {
    guard temperature > 0 else { return argMax(logits) }
    // float32: bf16 is too coarse for the softmax and running sum below.
    let scaled = logits.asType(.float32) / temperature
    let k = min(max(k, 1), scaled.dim(-1))

    // Top-k: partitioning finds the k largest in linear time (no sort over 262k entries);
    // then only those k are sorted, most likely first.
    let candidates = argPartition(-scaled, kth: k - 1)[..<k]
    let ranked = take(candidates, argSort(-take(scaled, candidates)))
    let rankedLogits = take(scaled, ranked)

    // Top-p: keep a token while the probability mass ranked above it is still below p. The
    // token that crosses p stays in, and so does the top token for any p.
    let massBefore = cumsum(softmax(rankedLogits), inclusive: false)
    let nucleus = MLX.where(massBefore .< p, rankedLogits, MLXArray(-Float.infinity))

    // `categorical` picks a position in the ranked subset; map it back to a vocabulary ID.
    return take(ranked, MLXRandom.categorical(nucleus))
}
