import Foundation

/// The subset of a Gemma 3 `config.json` that text inference needs.
///
/// TranslateGemma ships as `Gemma3ForConditionalGeneration`, so the text model's settings
/// live under `text_config`; plain `gemma3_text` checkpoints keep them at the top level.
public struct Gemma3Config: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var intermediateSize: Int
    public var headCount: Int
    public var kvHeadCount: Int
    public var headDim: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var queryPreAttnScalar: Float
    public var slidingWindow: Int
    /// One entry per layer: `true` when the layer attends only to the last `slidingWindow` tokens.
    public var isSliding: [Bool]
    public var localRope: Rope
    public var globalRope: Rope
    public var quantization: Quantization?

    public struct Rope: Sendable {
        public var base: Float
        /// Multiplier applied to positions. Linear RoPE scaling with factor `f` is `1 / f`.
        public var scale: Float
    }

    public struct Quantization: Sendable {
        public var groupSize: Int
        public var bits: Int
    }
}

extension Gemma3Config {
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MicroGemmaError.invalidConfig("config.json is not an object")
        }
        let text = root["text_config"] as? [String: Any] ?? root

        func int(_ key: String) throws -> Int {
            guard let v = text[key] as? NSNumber else { throw MicroGemmaError.invalidConfig(key) }
            return v.intValue
        }
        func float(_ key: String, default value: Float) -> Float {
            (text[key] as? NSNumber)?.floatValue ?? value
        }

        hiddenSize = try int("hidden_size")
        layerCount = try int("num_hidden_layers")
        intermediateSize = try int("intermediate_size")
        headCount = try int("num_attention_heads")
        kvHeadCount = try int("num_key_value_heads")
        headDim = try int("head_dim")
        vocabSize = try int("vocab_size")
        rmsNormEps = float("rms_norm_eps", default: 1e-6)
        queryPreAttnScalar = float("query_pre_attn_scalar", default: Float(headDim))
        slidingWindow = try int("sliding_window")

        if let types = text["layer_types"] as? [String] {
            isSliding = types.map { $0 == "sliding_attention" }
        } else {
            let pattern = (text["sliding_window_pattern"] ?? text["_sliding_window_pattern"])
                .flatMap { ($0 as? NSNumber)?.intValue } ?? 6
            isSliding = (0..<layerCount).map { ($0 + 1) % pattern != 0 }
        }

        localRope = Rope(base: float("rope_local_base_freq", default: 10_000), scale: 1)
        globalRope = Rope(
            base: float("rope_theta", default: 1_000_000),
            scale: Self.linearRopeScale(text, layerType: "full_attention"))

        let q = (root["quantization"] ?? root["quantization_config"]) as? [String: Any]
        if let q, let g = q["group_size"] as? NSNumber, let b = q["bits"] as? NSNumber {
            quantization = Quantization(groupSize: g.intValue, bits: b.intValue)
        }
    }

    /// Reads the RoPE position scale for one layer type.
    ///
    /// Newer `transformers` exports write `rope_parameters`, either flat or keyed by layer type,
    /// and set the legacy `rope_scaling` to null. Both spellings are honored here.
    static func linearRopeScale(_ text: [String: Any], layerType: String) -> Float {
        var params = text["rope_parameters"] as? [String: Any]
        if let perType = params?[layerType] as? [String: Any] { params = perType }
        let candidates = [params, text["rope_scaling"] as? [String: Any]].compactMap { $0 }
        for p in candidates {
            let type = (p["rope_type"] ?? p["type"]) as? String
            if type == "linear", let factor = (p["factor"] as? NSNumber)?.floatValue {
                return 1 / factor
            }
        }
        return 1
    }
}

public enum MicroGemmaError: Error, CustomStringConvertible {
    case invalidConfig(String)
    case missingWeights(URL)
    case unknownLanguage(String)

    public var description: String {
        switch self {
        case .invalidConfig(let key): "config.json: missing or invalid \(key)"
        case .missingWeights(let dir): "no .safetensors files in \(dir.path)"
        case .unknownLanguage(let code): "unknown language code \(code)"
        }
    }
}
