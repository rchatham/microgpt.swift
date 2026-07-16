import Foundation

enum TokenizerError: Error, CustomStringConvertible {
    case usage
    case readFailed(String)
    case invalidMagic(String)
    case invalidUTF8
    case unexpectedEOF
    case unknownValueType(UInt32)
    case missingMetadata(String)
    case unsupportedTokenizer(String)

    var description: String {
        switch self {
        case .usage:
            "usage: translategemma-native-tokenize <model.gguf> <text>"
        case .readFailed(let path):
            "failed to read file: \(path)"
        case .invalidMagic(let magic):
            "invalid GGUF magic: \(magic)"
        case .invalidUTF8:
            "invalid UTF-8 in GGUF metadata"
        case .unexpectedEOF:
            "unexpected EOF while reading GGUF metadata"
        case .unknownValueType(let raw):
            "unknown GGUF metadata value type: \(raw)"
        case .missingMetadata(let key):
            "missing tokenizer metadata: \(key)"
        case .unsupportedTokenizer(let model):
            "unsupported tokenizer model: \(model)"
        }
    }
}

enum GGUFValueType: UInt32 {
    case uint8 = 0, int8 = 1, uint16 = 2, int16 = 3, uint32 = 4, int32 = 5
    case float32 = 6, bool = 7, string = 8, array = 9, uint64 = 10, int64 = 11, float64 = 12
}

enum MetadataValue {
    case string(String)
    case uint32(UInt32)
    case bool(Bool)
    case strings([String])
    case floats([Float])
    case int32s([Int32])
    case skipped
}

struct BinaryReader {
    let data: Data
    var offset = 0

    mutating func readBytes(_ count: Int) throws -> [UInt8] {
        guard count >= 0, offset <= data.count - count else { throw TokenizerError.unexpectedEOF }
        var bytes = [UInt8](repeating: 0, count: count)
        data.copyBytes(to: &bytes, from: offset..<(offset + count))
        offset += count
        return bytes
    }

    mutating func readUInt8() throws -> UInt8 { try readInteger(UInt8.self) }
    mutating func readInt8() throws -> Int8 { try readInteger(Int8.self) }
    mutating func readUInt16() throws -> UInt16 { UInt16(littleEndian: try readInteger(UInt16.self)) }
    mutating func readInt16() throws -> Int16 { Int16(littleEndian: try readInteger(Int16.self)) }
    mutating func readUInt32() throws -> UInt32 { UInt32(littleEndian: try readInteger(UInt32.self)) }
    mutating func readInt32() throws -> Int32 { Int32(littleEndian: try readInteger(Int32.self)) }
    mutating func readUInt64() throws -> UInt64 { UInt64(littleEndian: try readInteger(UInt64.self)) }
    mutating func readInt64() throws -> Int64 { Int64(littleEndian: try readInteger(Int64.self)) }
    mutating func readFloat32() throws -> Float { Float(bitPattern: try readUInt32()) }
    mutating func readFloat64() throws -> Double { Double(bitPattern: try readUInt64()) }

    mutating func readString() throws -> String {
        let count = try readUInt64()
        let bytes = try readBytes(Int(count))
        guard let value = String(bytes: bytes, encoding: .utf8) else { throw TokenizerError.invalidUTF8 }
        return value
    }

    private mutating func readInteger<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let count = MemoryLayout<T>.size
        guard offset <= data.count - count else { throw TokenizerError.unexpectedEOF }
        let value = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
        offset += count
        return value
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
}

struct GGUFMetadataReader {
    func readTokenizer(path: String) throws -> GGUFTokenizerMetadata {
        let url = URL(fileURLWithPath: path)
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw TokenizerError.readFailed(path)
        }

        var reader = BinaryReader(data: data)
        let magic = String(bytes: try reader.readBytes(4), encoding: .ascii) ?? "<non-ascii>"
        guard magic == "GGUF" else { throw TokenizerError.invalidMagic(magic) }
        _ = try reader.readUInt32()
        let tensorCount = try reader.readUInt64()
        let metadataCount = try reader.readUInt64()

        var metadata: [String: MetadataValue] = [:]
        for _ in 0..<metadataCount {
            let key = try reader.readString()
            metadata[key] = try readValue(reader: &reader)
        }

        // Skip tensor directory so malformed metadata readers do not accidentally leave unconsumed arrays unnoticed.
        for _ in 0..<tensorCount {
            _ = try reader.readString()
            let dimensions = try reader.readUInt32()
            for _ in 0..<dimensions { _ = try reader.readUInt64() }
            _ = try reader.readUInt32()
            _ = try reader.readUInt64()
        }

        let model = try string(metadata, "tokenizer.ggml.model")
        let tokens = try strings(metadata, "tokenizer.ggml.tokens")
        let scores = (try? floats(metadata, "tokenizer.ggml.scores")) ?? []
        let tokenTypes = (try? int32s(metadata, "tokenizer.ggml.token_type")) ?? []
        let unknown = Int((try? uint32(metadata, "tokenizer.ggml.unknown_token_id")) ?? 0)
        let bos = Int((try? uint32(metadata, "tokenizer.ggml.bos_token_id")) ?? 2)
        let eos = Int((try? uint32(metadata, "tokenizer.ggml.eos_token_id")) ?? 1)
        let addBOS = (try? bool(metadata, "tokenizer.ggml.add_bos_token")) ?? true
        let addEOS = (try? bool(metadata, "tokenizer.ggml.add_eos_token")) ?? false
        let addSpacePrefix = (try? bool(metadata, "tokenizer.ggml.add_space_prefix")) ?? true
        return .init(model: model, tokens: tokens, scores: scores, tokenTypes: tokenTypes, unknownTokenID: unknown, bosTokenID: bos, eosTokenID: eos, addBOS: addBOS, addEOS: addEOS, addSpacePrefix: addSpacePrefix)
    }

    private func readValue(reader: inout BinaryReader) throws -> MetadataValue {
        let rawType = try reader.readUInt32()
        guard let type = GGUFValueType(rawValue: rawType) else { throw TokenizerError.unknownValueType(rawType) }
        switch type {
        case .uint8: _ = try reader.readUInt8(); return .skipped
        case .int8: _ = try reader.readInt8(); return .skipped
        case .uint16: _ = try reader.readUInt16(); return .skipped
        case .int16: _ = try reader.readInt16(); return .skipped
        case .uint32: return .uint32(try reader.readUInt32())
        case .int32: _ = try reader.readInt32(); return .skipped
        case .float32: _ = try reader.readFloat32(); return .skipped
        case .bool: return .bool(try reader.readUInt8() != 0)
        case .string: return .string(try reader.readString())
        case .array: return try readArray(reader: &reader)
        case .uint64: _ = try reader.readUInt64(); return .skipped
        case .int64: _ = try reader.readInt64(); return .skipped
        case .float64: _ = try reader.readFloat64(); return .skipped
        }
    }

    private func readArray(reader: inout BinaryReader) throws -> MetadataValue {
        let rawElementType = try reader.readUInt32()
        guard let elementType = GGUFValueType(rawValue: rawElementType) else { throw TokenizerError.unknownValueType(rawElementType) }
        let count = Int(try reader.readUInt64())
        switch elementType {
        case .string:
            var values: [String] = []
            values.reserveCapacity(count)
            for _ in 0..<count { values.append(try reader.readString()) }
            return .strings(values)
        case .float32:
            var values: [Float] = []
            values.reserveCapacity(count)
            for _ in 0..<count { values.append(try reader.readFloat32()) }
            return .floats(values)
        case .int32:
            var values: [Int32] = []
            values.reserveCapacity(count)
            for _ in 0..<count { values.append(try reader.readInt32()) }
            return .int32s(values)
        default:
            for _ in 0..<count { try skipArrayElement(elementType, reader: &reader) }
            return .skipped
        }
    }

    private func skipArrayElement(_ type: GGUFValueType, reader: inout BinaryReader) throws {
        switch type {
        case .uint8, .int8, .bool: _ = try reader.readBytes(1)
        case .uint16, .int16: _ = try reader.readBytes(2)
        case .uint32, .int32, .float32: _ = try reader.readBytes(4)
        case .uint64, .int64, .float64: _ = try reader.readBytes(8)
        case .string: _ = try reader.readString()
        case .array: throw TokenizerError.unsupportedTokenizer("nested GGUF metadata arrays")
        }
    }

    private func string(_ metadata: [String: MetadataValue], _ key: String) throws -> String {
        guard case .string(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    private func strings(_ metadata: [String: MetadataValue], _ key: String) throws -> [String] {
        guard case .strings(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    private func floats(_ metadata: [String: MetadataValue], _ key: String) throws -> [Float] {
        guard case .floats(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    private func int32s(_ metadata: [String: MetadataValue], _ key: String) throws -> [Int32] {
        guard case .int32s(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    private func uint32(_ metadata: [String: MetadataValue], _ key: String) throws -> UInt32 {
        guard case .uint32(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    private func bool(_ metadata: [String: MetadataValue], _ key: String) throws -> Bool {
        guard case .bool(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }
}

struct GreedySentencePieceTokenizer {
    let metadata: GGUFTokenizerMetadata
    let tokenToID: [String: Int]
    let maxTokenLength: Int

    init(metadata: GGUFTokenizerMetadata) throws {
        guard metadata.model == "llama" else { throw TokenizerError.unsupportedTokenizer(metadata.model) }
        self.metadata = metadata
        self.tokenToID = metadata.tokens.enumerated().reduce(into: [:]) { result, item in
            result[item.element] = result[item.element] ?? item.offset
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
                let scalar = normalized[index]
                ids.append(tokenToID[String(scalar)] ?? metadata.unknownTokenID)
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

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { throw TokenizerError.usage }
    let path = args[0]
    let text = args.dropFirst().joined(separator: " ")
    let metadata = try GGUFMetadataReader().readTokenizer(path: path)
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
