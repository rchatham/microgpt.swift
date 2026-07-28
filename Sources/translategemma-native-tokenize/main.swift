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

struct SentencePieceUnigramTokenizer {
    private struct Match {
        let id: Int
        let length: Int
        let score: Float
    }

    private struct Backpointer {
        let previous: Int
        let id: Int
    }

    let metadata: GGUFTokenizerMetadata
    let tokenToID: [String: Int]
    let byteTokenIDs: [UInt8: Int]
    let tokenScores: [Float]
    let maxTokenLength: Int
    let unknownScore: Float

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
        self.tokenScores = metadata.tokens.indices.map { index in
            metadata.scores.indices.contains(index) ? metadata.scores[index] : 0
        }
        self.maxTokenLength = metadata.tokens.map(\.count).max() ?? 0
        self.unknownScore = (metadata.scores.min() ?? -10) - 10
    }

    func encode(_ text: String) -> [Int] {
        let normalized = normalize(text)
        let characters = Array(normalized)
        var bestScores = Array(repeating: -Float.infinity, count: characters.count + 1)
        var backpointers = Array<Backpointer?>(repeating: nil, count: characters.count + 1)
        bestScores[0] = 0

        for start in 0..<characters.count where bestScores[start].isFinite {
            for match in matches(in: characters, at: start) {
                let end = start + match.length
                let score = bestScores[start] + match.score
                if score > bestScores[end] {
                    bestScores[end] = score
                    backpointers[end] = Backpointer(previous: start, id: match.id)
                }
            }
        }

        var ids = backtrack(backpointers: backpointers, end: characters.count)
        if metadata.addBOS { ids.insert(metadata.bosTokenID, at: 0) }
        if metadata.addEOS { ids.append(metadata.eosTokenID) }
        return ids
    }

    func pieces(for ids: [Int]) -> [String] {
        ids.map { id in
            guard metadata.tokens.indices.contains(id) else { return "<invalid>" }
            return metadata.tokens[id]
        }
    }

    private func normalize(_ text: String) -> String {
        var normalized = text.replacingOccurrences(of: " ", with: "▁")
        if metadata.addSpacePrefix, !normalized.hasPrefix("▁") {
            normalized = "▁" + normalized
        }
        return normalized
    }

    private func matches(in characters: [Character], at start: Int) -> [Match] {
        var matches: [Match] = []
        let maxEnd = min(characters.count, start + maxTokenLength)
        if start < maxEnd {
            for end in (start + 1)...maxEnd {
                let piece = String(characters[start..<end])
                if let id = tokenToID[piece] {
                    matches.append(Match(id: id, length: end - start, score: tokenScores[id]))
                }
            }
        }
        if matches.isEmpty {
            let fallbackIDs = String(characters[start]).utf8.map { byteTokenIDs[$0] ?? metadata.unknownTokenID }
            matches = fallbackIDs.map { Match(id: $0, length: 1, score: unknownScore) }
        }
        return matches
    }

    private func backtrack(backpointers: [Backpointer?], end: Int) -> [Int] {
        var ids: [Int] = []
        var index = end
        while index > 0, let backpointer = backpointers[index] {
            ids.append(backpointer.id)
            index = backpointer.previous
        }
        return ids.reversed()
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
    let tokenizer = try SentencePieceUnigramTokenizer(metadata: metadata)
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
