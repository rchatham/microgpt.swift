import Foundation
import GGUFCore

enum TokenizerError: Error, CustomStringConvertible {
    case usage
    case missingMetadata(String)
    case unsupportedTokenizer(String)

    var description: String {
        switch self {
        case .usage:
            "usage: translategemma-native-tokenize <model.gguf> <text>"
        case .missingMetadata(let key):
            "missing tokenizer metadata: \(key)"
        case .unsupportedTokenizer(let model):
            "unsupported tokenizer model: \(model)"
        }
    }
}

struct GGUFTokenizerMetadata {
    let model: String
    let tokens: [String]
    let scores: [Float]
    let tokenTypes: [Int32]
    let unknownTokenID: Int
    let bosTokenID: Int
    let eosTokenID: Int
    let addBOS: Bool
    let addEOS: Bool
    let addSpacePrefix: Bool

    init(file: GGUFFile) throws {
        model = try file.stringMetadata("tokenizer.ggml.model")
        tokens = try file.stringArrayMetadata("tokenizer.ggml.tokens")
        scores = file.floatArrayMetadata("tokenizer.ggml.scores") ?? []
        tokenTypes = file.int32ArrayMetadata("tokenizer.ggml.token_type") ?? []
        unknownTokenID = Int(file.uint32Metadata("tokenizer.ggml.unknown_token_id") ?? 0)
        bosTokenID = Int(file.uint32Metadata("tokenizer.ggml.bos_token_id") ?? 2)
        eosTokenID = Int(file.uint32Metadata("tokenizer.ggml.eos_token_id") ?? 1)
        addBOS = file.boolMetadata("tokenizer.ggml.add_bos_token") ?? true
        addEOS = file.boolMetadata("tokenizer.ggml.add_eos_token") ?? false
        addSpacePrefix = file.boolMetadata("tokenizer.ggml.add_space_prefix") ?? true
    }
}

struct GreedySentencePieceTokenizer {
    let metadata: GGUFTokenizerMetadata
    let tokenToID: [String: Int]
    let byteTokenIDs: [UInt8: Int]
    let maxTokenLength: Int

    init(metadata: GGUFTokenizerMetadata) throws {
        guard metadata.model == "llama" else { throw TokenizerError.unsupportedTokenizer(metadata.model) }
        self.metadata = metadata
        let tokenToID = metadata.tokens.enumerated().reduce(into: [:]) { result, item in
            result[item.element] = result[item.element] ?? item.offset
        }
        self.tokenToID = tokenToID
        self.byteTokenIDs = (UInt8.min...UInt8.max).reduce(into: [:]) { result, byte in
            result[byte] = tokenToID[String(format: "<0x%02X>", byte)]
        }
        self.maxTokenLength = metadata.tokens.map(\.count).max() ?? 0
    }

    func encode(_ text: String) -> [Int] {
        var normalized = text.replacingOccurrences(of: " ", with: "▁")
        if metadata.addSpacePrefix, !normalized.hasPrefix("▁") {
            normalized = "▁" + normalized
        }

        var ids: [Int] = []
        if metadata.addBOS { ids.append(metadata.bosTokenID) }

        var index = normalized.startIndex
        while index < normalized.endIndex {
            let remaining = normalized[index...]
            var match: (id: Int, end: String.Index)?
            var end = normalized.index(index, offsetBy: min(maxTokenLength, remaining.count), limitedBy: normalized.endIndex) ?? normalized.endIndex
            while end > index {
                let piece = String(normalized[index..<end])
                if let id = tokenToID[piece] {
                    match = (id, end)
                    break
                }
                end = normalized.index(before: end)
            }
            if let match {
                ids.append(match.id)
                index = match.end
            } else {
                let character = normalized[index]
                let fallbackIDs = String(character).utf8.map { byteTokenIDs[$0] ?? metadata.unknownTokenID }
                ids.append(contentsOf: fallbackIDs)
                index = normalized.index(after: index)
            }
        }

        if metadata.addEOS { ids.append(metadata.eosTokenID) }
        return ids
    }

    func pieces(for ids: [Int]) -> [String] {
        ids.map { id in
            guard metadata.tokens.indices.contains(id) else { return "<invalid>" }
            return metadata.tokens[id]
        }
    }
}

private extension GGUFFile {
    func stringMetadata(_ key: String) throws -> String {
        guard case .string(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    func stringArrayMetadata(_ key: String) throws -> [String] {
        guard case .array(.string, _, let values) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return values
    }

    func floatArrayMetadata(_ key: String) -> [Float]? {
        guard case .array(.float32, _, let values) = metadata[key] else { return nil }
        return values.map(Float.init).compactMap { $0 }
    }

    func int32ArrayMetadata(_ key: String) -> [Int32]? {
        guard case .array(.int32, _, let values) = metadata[key] else { return nil }
        return values.map(Int32.init).compactMap { $0 }
    }

    func uint32Metadata(_ key: String) -> UInt32? {
        guard case .scalar(let value) = metadata[key] else { return nil }
        return UInt32(value)
    }

    func boolMetadata(_ key: String) -> Bool? {
        guard case .scalar(let value) = metadata[key] else { return nil }
        return Bool(value)
    }
}

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { throw TokenizerError.usage }
    let path = args[0]
    let text = args.dropFirst().joined(separator: " ")
    let file = try GGUFReader().read(path: path)
    let metadata = try GGUFTokenizerMetadata(file: file)
    let tokenizer = try GreedySentencePieceTokenizer(metadata: metadata)
    let ids = tokenizer.encode(text)
    let pieces = tokenizer.pieces(for: ids)

    print("model: \(metadata.model)")
    print("vocab: \(metadata.tokens.count)")
    print("unk: \(metadata.unknownTokenID) bos: \(metadata.bosTokenID) eos: \(metadata.eosTokenID)")
    print("ids: \(ids.map(String.init).joined(separator: " "))")
    print("pieces: \(pieces.map { $0.debugDescription }.joined(separator: " "))")
}

do {
    try main()
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
