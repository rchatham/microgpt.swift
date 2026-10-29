import Foundation
import MLX
import MLXNN

// Gemma 3 text decoder. Compared with microgpt's `gpt(tokenID:posID:keys:values:)` the
// skeleton is the same — embed, N × (attention + MLP) with residuals, norm, project to
// vocab — but every piece is upgraded:
//   - learned position table      -> rotary embeddings (two bases, see `Gemma3Config.Rope`)
//   - plain RMSNorm               -> Gemma RMSNorm, scaled by (1 + w), four per block
//   - one KV head per query head  -> grouped-query attention plus RMSNorm on q and k
//   - ReLU MLP                    -> GeGLU: down(gelu(gate(x)) * up(x))
//   - full attention everywhere   -> 1024-token sliding window on 5 of every 6 layers
//   - scalar `Value` autograd     -> MLXArray tensors, inference only

/// RMSNorm whose stored weight is an offset from 1, as Gemma checkpoints store it.
final class GemmaRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class Attention: Module {
    let headCount: Int
    let kvHeadCount: Int
    let headDim: Int
    let scale: Float
    let rope: Gemma3Config.Rope
    let window: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: GemmaRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: GemmaRMSNorm

    init(_ config: Gemma3Config, sliding: Bool) {
        headCount = config.headCount
        kvHeadCount = config.kvHeadCount
        headDim = config.headDim
        scale = 1 / config.queryPreAttnScalar.squareRoot()
        rope = sliding ? config.localRope : config.globalRope
        window = sliding ? config.slidingWindow : nil

        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, headCount * headDim, bias: false)
        _kProj.wrappedValue = Linear(d, kvHeadCount * headDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvHeadCount * headDim, bias: false)
        _oProj.wrappedValue = Linear(headCount * headDim, d, bias: false)
        _qNorm.wrappedValue = GemmaRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = GemmaRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        // [B, L, H*D] -> [B, H, L, D]
        var q = qProj(x).reshaped(b, l, headCount, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, l, kvHeadCount, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeadCount, headDim).transposed(0, 2, 1, 3)

        q = qNorm(q)
        k = kNorm(k)

        let position = cache.offset
        q = MLXFast.RoPE(
            q, dimensions: headDim, traditional: false, base: rope.base, scale: rope.scale,
            offset: position)
        k = MLXFast.RoPE(
            k, dimensions: headDim, traditional: false, base: rope.base, scale: rope.scale,
            offset: position)

        let (keys, values, keyStart) = cache.update(keys: k, values: v)
        let mask = attentionMask(
            queryStart: position, queryCount: l, keyStart: keyStart, keyCount: keys.dim(2),
            window: window, dtype: q.dtype)

        // The fused kernel broadcasts the 4 KV heads across the 8 query heads itself.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

/// Additive mask letting query `i` see key `j` when `j <= i` and, for sliding layers,
/// `j > i - window`. Returns nil when every key is visible (single-token decode).
func attentionMask(
    queryStart: Int, queryCount: Int, keyStart: Int, keyCount: Int, window: Int?,
    dtype: DType
) -> MLXArray? {
    let lastQuery = queryStart + queryCount - 1
    let firstQuery = queryStart
    let causalOK = keyStart + keyCount - 1 <= firstQuery
    let windowOK = window.map { keyStart > lastQuery - $0 } ?? true
    if causalOK && windowOK { return nil }

    let q = MLXArray(Int32(queryStart) ..< Int32(queryStart + queryCount)).expandedDimensions(axis: 1)
    let k = MLXArray(Int32(keyStart) ..< Int32(keyStart + keyCount)).expandedDimensions(axis: 0)
    var visible = k .<= q
    if let window {
        visible = visible .&& (k .> (q - Int32(window)))
    }
    return MLX.where(visible, MLXArray(0, dtype: dtype), MLXArray(-Float.infinity, dtype: dtype))
}

final class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hidden: Int) {
        _gate.wrappedValue = Linear(dimensions, hidden, bias: false)
        _up.wrappedValue = Linear(dimensions, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApproximate(gate(x)) * up(x))
    }
}

final class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: GemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: GemmaRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardNorm: GemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardNorm: GemmaRMSNorm

    init(_ config: Gemma3Config, sliding: Bool) {
        let d = config.hiddenSize
        let eps = config.rmsNormEps
        _attention.wrappedValue = Attention(config, sliding: sliding)
        mlp = MLP(dimensions: d, hidden: config.intermediateSize)
        _inputNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: eps)
        _postAttentionNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: eps)
        _preFeedforwardNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: eps)
        _postFeedforwardNorm.wrappedValue = GemmaRMSNorm(dimensions: d, eps: eps)
    }

    // "Sandwich" norms: each sublayer's output is normalized again before the residual add.
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + postAttentionNorm(attention(inputNorm(x), cache: cache))
        return h + postFeedforwardNorm(mlp(preFeedforwardNorm(h)))
    }
}

public final class Gemma3TextModel: Module {
    public let config: Gemma3Config

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [TransformerBlock]
    let norm: GemmaRMSNorm

    public init(_ config: Gemma3Config) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = config.isSliding.map { TransformerBlock(config, sliding: $0) }
        norm = GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func makeCaches() -> [KVCache] {
        config.isSliding.map { KVCache(window: $0 ? config.slidingWindow : nil) }
    }

    /// Runs `tokens` ([B, L] int32) through the model, extending `caches`, and returns logits
    /// [B, L, vocab].
    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]) -> MLXArray {
        var h = embedTokens(tokens)
        // Gemma scales embeddings by sqrt(d); the constant is rounded to bf16 like the original.
        h = h * MLXArray(Float(config.hiddenSize).squareRoot(), dtype: .bfloat16).asType(h.dtype)
        for (layer, cache) in zip(layers, caches) {
            h = layer(h, cache: cache)
        }
        // The output projection shares weights with the input embedding table.
        return embedTokens.asLinear(norm(h))
    }
}
