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

    /// Picks one token ID from `logits`, a `[vocab]` array for the last position.
    func chooseToken(from logits: MLXArray) -> Int {
        switch self {
        case .greedy:
            return argMax(logits).item(Int.self)
        case .topKTopP(let k, let p, let temperature):
            return sampleTopKTopP(logits, k: k, p: p, temperature: temperature)
        }
    }
}

/// Samples a token from the top-k / nucleus (top-p) distribution of `logits`.
///
/// microgpt's `rng.weightedChoice` did this over 27 characters with plain Swift arrays;
/// here the vocabulary is 262k, so the work should stay in MLX until a single index is left.
/// Useful pieces: `argPartition(-logits, kth: k - 1)` (indices of the k largest, unordered),
/// `take(_:_:)`, `sorted`/`argSort`, `softmax`, `cumsum`, `MLX.where`, and
/// `MLXRandom.categorical(logits)` (samples an index from unnormalized log-probabilities).
func sampleTopKTopP(_ logits: MLXArray, k: Int, p: Float, temperature: Float) -> Int {
    // TODO: implement top-k + top-p sampling (about 5–10 lines).
    // Until then this falls back to greedy so `--sample` still runs.
    return argMax(logits).item(Int.self)
}
