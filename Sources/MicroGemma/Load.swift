import Foundation
import MLX
import MLXNN

extension Gemma3TextModel {
    /// Builds the model described by `directory/config.json` and fills it from the
    /// `.safetensors` files beside it.
    public static func load(from directory: URL) throws -> Gemma3TextModel {
        let config = try Gemma3Config(contentsOf: directory.appending(path: "config.json"))
        let model = Gemma3TextModel(config)

        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else { throw MicroGemmaError.missingWeights(directory) }

        var weights = [String: MLXArray]()
        for file in files {
            for (name, array) in try loadArrays(url: file) {
                // TranslateGemma stores the text decoder under `language_model.model.` next to
                // the vision tower; plain text checkpoints use `model.`. Keep only the decoder.
                for prefix in ["language_model.model.", "model."] where name.hasPrefix(prefix) {
                    weights[String(name.dropFirst(prefix.count))] = array
                }
            }
        }

        // Swap Linear/Embedding for their quantized versions wherever the checkpoint has
        // scales, so the packed 4-bit weights load as-is and are never expanded.
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? (q.groupSize, q.bits, .affine) : nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        // MLX is lazy; `eval` reads the tensors into memory now rather than on first use.
        eval(model)
        return model
    }
}
