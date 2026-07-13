import Foundation

enum GGUFError: Error, CustomStringConvertible {
    case invalidUsage
    case failedToReadFile(String)
    case unexpectedEOF(offset: Int, needed: Int)
    case invalidMagic(String)
    case unsupportedVersion(UInt32)
    case invalidUTF8
    case unknownValueType(UInt32)
    case unsupportedArrayElement(GGUFValueType)

    var description: String {
        switch self {
        case .invalidUsage:
            return "usage: gguf-inspect <model.gguf>"
        case .failedToReadFile(let path):
            return "failed to read file: \(path)"
        case .unexpectedEOF(let offset, let needed):
            return "unexpected EOF at byte \(offset), needed \(needed) bytes"
        case .invalidMagic(let magic):
            return "invalid GGUF magic: \(magic)"
        case .unsupportedVersion(let version):
            return "unsupported GGUF version: \(version)"
        case .invalidUTF8:
            return "invalid UTF-8 string in GGUF"
        case .unknownValueType(let raw):
            return "unknown GGUF metadata value type: \(raw)"
        case .unsupportedArrayElement(let type):
            return "unsupported GGUF array element type: \(type)"
        }
    }
}

enum GGUFValueType: UInt32, CustomStringConvertible {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12

    var description: String {
        switch self {
        case .uint8: "uint8"
        case .int8: "int8"
        case .uint16: "uint16"
        case .int16: "int16"
        case .uint32: "uint32"
        case .int32: "int32"
        case .float32: "float32"
        case .bool: "bool"
        case .string: "string"
        case .array: "array"
        case .uint64: "uint64"
        case .int64: "int64"
        case .float64: "float64"
        }
    }
}

enum GGUFMetadataValue: CustomStringConvertible {
    case scalar(String)
    case string(String)
    case array(type: GGUFValueType, count: UInt64, preview: [String])

    var description: String {
        switch self {
        case .scalar(let value), .string(let value):
            return value
        case .array(let type, let count, let preview):
            let suffix = count > UInt64(preview.count) ? ", …" : ""
            return "[\(type); \(count)] [\(preview.joined(separator: ", "))\(suffix)]"
        }
    }
}

struct GGUFTensorInfo {
    let name: String
    let dimensions: [UInt64]
    let type: UInt32
    let offset: UInt64

    var elementCount: UInt64 {
        dimensions.reduce(1, *)
    }

    var typeName: String {
        GGMLType.name(for: type)
    }
}

struct GGUFFile {
    let version: UInt32
    let tensorCount: UInt64
    let metadata: [String: GGUFMetadataValue]
    let tensors: [GGUFTensorInfo]
    let tensorDataStart: Int
}

enum GGMLType {
    static func name(for raw: UInt32) -> String {
        switch raw {
        case 0: "F32"
        case 1: "F16"
        case 2: "Q4_0"
        case 3: "Q4_1"
        case 6: "Q5_0"
        case 7: "Q5_1"
        case 8: "Q8_0"
        case 9: "Q8_1"
        case 10: "Q2_K"
        case 11: "Q3_K"
        case 12: "Q4_K"
        case 13: "Q5_K"
        case 14: "Q6_K"
        case 15: "Q8_K"
        case 16: "IQ2_XXS"
        case 17: "IQ2_XS"
        case 18: "IQ3_XXS"
        case 19: "IQ1_S"
        case 20: "IQ4_NL"
        case 21: "IQ3_S"
        case 22: "IQ2_S"
        case 23: "IQ4_XS"
        case 24: "I8"
        case 25: "I16"
        case 26: "I32"
        case 27: "I64"
        case 28: "F64"
        case 29: "IQ1_M"
        case 30: "BF16"
        default: "UNKNOWN(\(raw))"
        }
    }
}

struct BinaryReader {
    private let data: Data
    private(set) var offset: Int = 0

    init(data: Data) {
        self.data = data
    }

    mutating func readBytes(count: Int) throws -> [UInt8] {
        guard count >= 0, offset <= data.count - count else {
            throw GGUFError.unexpectedEOF(offset: offset, needed: count)
        }
        var value = [UInt8](repeating: 0, count: count)
        data.copyBytes(to: &value, from: offset..<(offset + count))
        offset += count
        return value
    }

    mutating func readUInt8() throws -> UInt8 { try readInteger(UInt8.self) }
    mutating func readInt8() throws -> Int8 { try readInteger(Int8.self) }
    mutating func readUInt16() throws -> UInt16 { UInt16(littleEndian: try readInteger(UInt16.self)) }
    mutating func readInt16() throws -> Int16 { Int16(littleEndian: try readInteger(Int16.self)) }
    mutating func readUInt32() throws -> UInt32 { UInt32(littleEndian: try readInteger(UInt32.self)) }
    mutating func readInt32() throws -> Int32 { Int32(littleEndian: try readInteger(Int32.self)) }
    mutating func readUInt64() throws -> UInt64 { UInt64(littleEndian: try readInteger(UInt64.self)) }
    mutating func readInt64() throws -> Int64 { Int64(littleEndian: try readInteger(Int64.self)) }

    mutating func readFloat32() throws -> Float32 {
        Float32(bitPattern: try readUInt32())
    }

    mutating func readFloat64() throws -> Float64 {
        Float64(bitPattern: try readUInt64())
    }

    mutating func readGGUFString() throws -> String {
        let count = try readUInt64()
        let bytes = try readBytes(count: Int(count))
        guard let string = String(bytes: bytes, encoding: .utf8) else {
            throw GGUFError.invalidUTF8
        }
        return string
    }

    private mutating func readInteger<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let count = MemoryLayout<T>.size
        guard offset <= data.count - count else {
            throw GGUFError.unexpectedEOF(offset: offset, needed: count)
        }
        let value = data.withUnsafeBytes { bytes in
            bytes.loadUnaligned(fromByteOffset: offset, as: T.self)
        }
        offset += count
        return value
    }
}

struct GGUFReader {
    func read(path: String) throws -> GGUFFile {
        let url = URL(fileURLWithPath: path)
        let data: Data
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
        } catch {
            throw GGUFError.failedToReadFile(path)
        }
        var reader = BinaryReader(data: data)

        let magicBytes = try reader.readBytes(count: 4)
        let magic = String(bytes: magicBytes, encoding: .ascii) ?? "<non-ascii>"
        guard magic == "GGUF" else {
            throw GGUFError.invalidMagic(magic)
        }

        let version = try reader.readUInt32()
        guard version == 2 || version == 3 else {
            throw GGUFError.unsupportedVersion(version)
        }

        let tensorCount = try reader.readUInt64()
        let metadataCount = try reader.readUInt64()

        var metadata: [String: GGUFMetadataValue] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let key = try reader.readGGUFString()
            metadata[key] = try readMetadataValue(reader: &reader)
        }

        var tensors: [GGUFTensorInfo] = []
        tensors.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            let name = try reader.readGGUFString()
            let dimensionCount = try reader.readUInt32()
            var dimensions: [UInt64] = []
            dimensions.reserveCapacity(Int(dimensionCount))
            for _ in 0..<dimensionCount {
                dimensions.append(try reader.readUInt64())
            }
            let type = try reader.readUInt32()
            let offset = try reader.readUInt64()
            tensors.append(.init(name: name, dimensions: dimensions, type: type, offset: offset))
        }

        let alignment = metadata["general.alignment"]?.uint64Value ?? 32
        let tensorDataStart = align(reader.offset, to: Int(alignment))
        return GGUFFile(
            version: version,
            tensorCount: tensorCount,
            metadata: metadata,
            tensors: tensors,
            tensorDataStart: tensorDataStart
        )
    }

    private func readMetadataValue(reader: inout BinaryReader) throws -> GGUFMetadataValue {
        let rawType = try reader.readUInt32()
        guard let type = GGUFValueType(rawValue: rawType) else {
            throw GGUFError.unknownValueType(rawType)
        }
        switch type {
        case .uint8: return .scalar(String(try reader.readUInt8()))
        case .int8: return .scalar(String(try reader.readInt8()))
        case .uint16: return .scalar(String(try reader.readUInt16()))
        case .int16: return .scalar(String(try reader.readInt16()))
        case .uint32: return .scalar(String(try reader.readUInt32()))
        case .int32: return .scalar(String(try reader.readInt32()))
        case .float32: return .scalar(String(try reader.readFloat32()))
        case .bool: return .scalar(String(try reader.readUInt8() != 0))
        case .string: return .string(try reader.readGGUFString())
        case .array: return try readArray(reader: &reader)
        case .uint64: return .scalar(String(try reader.readUInt64()))
        case .int64: return .scalar(String(try reader.readInt64()))
        case .float64: return .scalar(String(try reader.readFloat64()))
        }
    }

    private func readArray(reader: inout BinaryReader) throws -> GGUFMetadataValue {
        let rawElementType = try reader.readUInt32()
        guard let elementType = GGUFValueType(rawValue: rawElementType) else {
            throw GGUFError.unknownValueType(rawElementType)
        }
        let count = try reader.readUInt64()
        var preview: [String] = []
        preview.reserveCapacity(min(Int(count), 8))
        for index in 0..<count {
            let value = try readArrayElement(type: elementType, reader: &reader)
            if index < 8 {
                preview.append(value)
            }
        }
        return .array(type: elementType, count: count, preview: preview)
    }

    private func readArrayElement(type: GGUFValueType, reader: inout BinaryReader) throws -> String {
        switch type {
        case .uint8: return String(try reader.readUInt8())
        case .int8: return String(try reader.readInt8())
        case .uint16: return String(try reader.readUInt16())
        case .int16: return String(try reader.readInt16())
        case .uint32: return String(try reader.readUInt32())
        case .int32: return String(try reader.readInt32())
        case .float32: return String(try reader.readFloat32())
        case .bool: return String(try reader.readUInt8() != 0)
        case .string: return try reader.readGGUFString()
        case .uint64: return String(try reader.readUInt64())
        case .int64: return String(try reader.readInt64())
        case .float64: return String(try reader.readFloat64())
        case .array: throw GGUFError.unsupportedArrayElement(type)
        }
    }

    private func align(_ value: Int, to alignment: Int) -> Int {
        guard alignment > 0 else { return value }
        let remainder = value % alignment
        return remainder == 0 ? value : value + alignment - remainder
    }
}

private extension GGUFMetadataValue {
    var uint64Value: UInt64? {
        switch self {
        case .scalar(let value): UInt64(value)
        case .string, .array: nil
        }
    }
}

struct GGUFInspect {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 1 else {
            throw GGUFError.invalidUsage
        }

        let file = try GGUFReader().read(path: arguments[0])
        printSummary(file)
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
