import Foundation
import MLX
import MLXRandom
import MicroGemma

/// Plain-Swift reference for top-k / top-p: the tokens that may be drawn and the probability
/// of each, written the slow obvious way to check the MLX version against.
func expectedDistribution(_ logits: [Float], k: Int, p: Float, temperature: Float) -> [Int: Double] {
    let ranked = Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(k))
    let scaled = ranked.map { Double(logits[$0] / temperature) }
    let top = scaled.max()!
    let weights = scaled.map { exp($0 - top) }
    let total = weights.reduce(0, +)

    var kept: [(id: Int, weight: Double)] = []
    var massBefore = 0.0
    for (id, w) in zip(ranked, weights) {
        if massBefore < Double(p) { kept.append((id, w)) }
        massBefore += w / total
    }
    let keptTotal = kept.map(\.weight).reduce(0, +)
    return Dictionary(uniqueKeysWithValues: kept.map { ($0.id, $0.weight / keptTotal) })
}

/// Draws `count` tokens, all queued lazily and evaluated together, and returns how often
/// each ID came up.
func drawCounts(_ sampling: Sampling, from logits: MLXArray, count: Int) -> [Int: Int] {
    let draws = stacked((0..<count).map { _ in sampling.token(from: logits) })
    var counts: [Int: Int] = [:]
    for id in draws.asType(.int32).asArray(Int32.self) { counts[Int(id), default: 0] += 1 }
    return counts
}

/// Checks `Sampling` on a hand-made 8-token logits vector. Returns false if any check fails.
func runSamplingChecks() -> Bool {
    // bf16, like the model's logits. Token 1 is the argmax; 7 and 4 are close behind.
    let logits = MLXArray([1.0, 3.0, 2.0, 0.5, 2.9, -1.0, 0.1, 2.95] as [Float]).asType(.bfloat16)
    let values = logits.asType(.float32).asArray(Float.self)  // after bf16 rounding
    let argmax = values.indices.max { values[$0] < values[$1] }!
    let draws = 4000
    MLXRandom.seed(1234)

    let cases: [(name: String, k: Int, p: Float, temperature: Float)] = [
        ("k = 1 always gives the argmax", 1, 1, 1),
        ("tiny p always gives the argmax", 8, 1e-6, 1),
        ("temperature 0 is greedy", 8, 0.95, 0),
        ("k = 3, p = 1: draws stay within the top 3", 3, 1, 1),
        ("k = 8, p = 0.5: draws stay within the nucleus", 8, 0.5, 1),
        ("k = 64 (> vocab), p = 0.95, temperature 0.7", 64, 0.95, 0.7),
    ]

    print("sampling checks (\(draws) seeded draws each, toy vocabulary of \(values.count))")
    var allPassed = true
    for c in cases {
        let counts = drawCounts(.topKTopP(k: c.k, p: c.p, temperature: c.temperature), from: logits, count: draws)
        let expected = c.temperature > 0
            ? expectedDistribution(values, k: c.k, p: c.p, temperature: c.temperature)
            : [argmax: 1.0]

        let outside = counts.keys.filter { expected[$0] == nil }.sorted()
        let worstError = expected.map { id, prob in
            abs(Double(counts[id, default: 0]) / Double(draws) - prob)
        }.max() ?? 0
        // 4000 draws put one standard deviation of a frequency at ≤ 0.008; allow ~4.
        let passed = outside.isEmpty && worstError < 0.03
        allPassed = allPassed && passed

        print(String(
            format: "  %@  %@: allowed %@, drawn %@, max frequency error %.3f",
            passed ? "PASS" : "FAIL", c.name,
            expected.keys.sorted().description, counts.keys.sorted().description, worstError))
        if !outside.isEmpty { print("        drew tokens outside the allowed set: \(outside)") }
    }
    return allPassed
}
