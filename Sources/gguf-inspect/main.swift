import Foundation
import GGUFCore

struct InspectOptions {
    let path: String
    let tensorName: String?
    let valueCount: Int
}

struct GGUFInspect {
    static func main() throws {
        let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
        let file = try GGUFReader().read(path: options.path)
        printSummary(file)
        if let tensorName = options.tensorName {
            try printTensor(file, name: tensorName, valueCount: options.valueCount)
        }
    }

    private static func parseOptions(_ arguments: [String]) throws -> InspectOptions {
        guard let path = arguments.first, !path.hasPrefix("--") else {
            throw GGUFError.invalidUsage
        }
        var tensorName: String?
        var valueCount = 16
        var index = 1
        while index < arguments.count {
            switch arguments[index] {
            case "--tensor":
                guard index + 1 < arguments.count else { throw GGUFError.invalidUsage }
                tensorName = arguments[index + 1]
                index += 2
            case "--values":
                guard index + 1 < arguments.count else { throw GGUFError.invalidUsage }
                guard let parsed = Int(arguments[index + 1]), parsed >= 0 else {
                    throw GGUFError.invalidValueCount(arguments[index + 1])
                }
                valueCount = parsed
                index += 2
            default:
                throw GGUFError.invalidUsage
            }
        }
        return InspectOptions(path: path, tensorName: tensorName, valueCount: valueCount)
    }

    private static func printSummary(_ file: GGUFFile) {
        let architecture = metadata(file, "general.architecture")
        let name = metadata(file, "general.name")
        print("GGUF v\(file.version)")
        print("architecture: \(architecture)")
        print("name: \(name)")
        print("tensors: \(file.tensorCount)")
        print("metadata entries: \(file.metadata.count)")
        print("tensor data start: \(file.tensorDataStart)")
        let architecturePrefix = architecture == "<missing>" ? "gemma3" : architecture
        let dynamicBlockCount = metadata(file, "\(architecturePrefix).block_count")
        let dynamicContextLength = metadata(file, "\(architecturePrefix).context_length")
        let dynamicEmbeddingLength = metadata(file, "\(architecturePrefix).embedding_length")
        let dynamicFeedForwardLength = metadata(file, "\(architecturePrefix).feed_forward_length")
        let dynamicAttentionHeads = metadata(file, "\(architecturePrefix).attention.head_count")
        let dynamicAttentionKVHeads = metadata(file, "\(architecturePrefix).attention.head_count_kv")

        print("")
        print("Model")
        print("  block count: \(dynamicBlockCount)")
        print("  context length: \(dynamicContextLength)")
        print("  embedding length: \(dynamicEmbeddingLength)")
        print("  feed-forward length: \(dynamicFeedForwardLength)")
        print("  attention heads: \(dynamicAttentionHeads)")
        print("  attention kv heads: \(dynamicAttentionKVHeads)")
        print("  vocab size: \(vocabSize(file))")
        print("")
        print("Tensor types")
        for (typeName, count) in tensorTypeCounts(file) {
            print("  \(typeName): \(count)")
        }
        print("")
        print("Tensors")
        for tensor in file.tensors.prefix(24) {
            let dimensions = tensor.dimensions.map(String.init).joined(separator: " x ")
            print("  \(tensor.name): [\(dimensions)] \(tensor.typeName) @ +\(tensor.offset)")
        }
        if file.tensors.count > 24 {
            print("  … \(file.tensors.count - 24) more")
        }
    }

    private static func printTensor(_ file: GGUFFile, name: String, valueCount: Int) throws {
        guard let tensor = file.tensor(named: name) else {
            throw GGUFError.tensorNotFound(name)
        }
        let byteCount = try file.bytes(for: tensor).count
        let dimensions = tensor.dimensions.map(String.init).joined(separator: " x ")
        print("")
        print("Tensor \(tensor.name)")
        print("  shape: [\(dimensions)]")
        print("  type: \(tensor.typeName)")
        print("  elements: \(tensor.elementCount)")
        print("  relative offset: \(tensor.offset)")
        print("  byte count: \(byteCount)")
        if valueCount > 0 {
            if tensor.type == 0 || tensor.type == 1 {
                let values = try file.decodedFloatPrefix(tensor: tensor, count: valueCount)
                let rendered = values.map { String(format: "%.6g", Double($0)) }.joined(separator: ", ")
                print("  first \(values.count) values: [\(rendered)]")
            } else {
                print("  first values: <\(tensor.typeName) decode not implemented; use --values 0 to suppress>")
            }
        }
    }

    private static func metadata(_ file: GGUFFile, _ key: String) -> String {
        file.metadata[key]?.description ?? "<missing>"
    }

    private static func vocabSize(_ file: GGUFFile) -> String {
        if case .array(_, let count, _) = file.metadata["tokenizer.ggml.tokens"] {
            return String(count)
        }
        return "<missing>"
    }

    private static func tensorTypeCounts(_ file: GGUFFile) -> [(String, Int)] {
        let counts = Dictionary(grouping: file.tensors, by: { $0.typeName })
            .mapValues(\.count)
        return counts.sorted { left, right in
            if left.key == right.key { return left.value < right.value }
            return left.key < right.key
        }
    }
}

do {
    try GGUFInspect.main()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
