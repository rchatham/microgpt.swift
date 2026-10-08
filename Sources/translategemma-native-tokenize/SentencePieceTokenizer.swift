import Foundation
import GGUFCore

// MARK: - Tokenizer Error

enum TokenizerError: Error, CustomStringConvertible {
    case usage
    case missingMetadata(String)
    case unsupportedTokenizer(String)
    case invalidMetadata(String)

    var description: String {
        switch self {
        case .usage:
            "usage: translategemma-native-tokenize <model.gguf> <text>"
        case .missingMetadata(let key):
            "missing tokenizer metadata: \(key)"
        case .unsupportedTokenizer(let model):
            "unsupported tokenizer model: \(model)"
        case .invalidMetadata(let detail):
            "invalid tokenizer metadata: \(detail)"
        }
    }
}

// MARK: - Tokenizer Metadata

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

        guard model == "llama" else {
            throw TokenizerError.unsupportedTokenizer(model)
        }

        tokens = try file.stringArrayMetadata("tokenizer.ggml.tokens")

        // Validate scores: must be present, correct type, count matches tokens, all parseable and finite
        guard let rawScores = file.metadata["tokenizer.ggml.scores"] else {
            throw TokenizerError.missingMetadata("tokenizer.ggml.scores")
        }
        guard case .array(.float32, let storedCount, let values) = rawScores else {
            throw TokenizerError.invalidMetadata("tokenizer.ggml.scores must be float32 array")
        }
        guard storedCount == values.count else {
            throw TokenizerError.invalidMetadata(
                "tokenizer.ggml.scores count mismatch: stored \(storedCount), parsed \(values.count)"
            )
        }
        guard values.count == tokens.count else {
            throw TokenizerError.invalidMetadata(
                "tokenizer.ggml.scores count \(values.count) != tokens count \(tokens.count)"
            )
        }

        var parsedScores: [Float] = []
        parsedScores.reserveCapacity(values.count)
        for (i, string) in values.enumerated() {
            guard let floatVal = Float(string) else {
                throw TokenizerError.invalidMetadata("tokenizer.ggml.scores[\(i)] unparsable: \"\(string)\"")
            }
            guard floatVal.isFinite else {
                throw TokenizerError.invalidMetadata("tokenizer.ggml.scores[\(i)] non-finite: \(floatVal)")
            }
            parsedScores.append(floatVal)
        }
        scores = parsedScores

        tokenTypes = file.int32ArrayMetadata("tokenizer.ggml.token_type") ?? []
        unknownTokenID = Int(file.uint32Metadata("tokenizer.ggml.unknown_token_id") ?? 0)
        bosTokenID = Int(file.uint32Metadata("tokenizer.ggml.bos_token_id") ?? 2)
        eosTokenID = Int(file.uint32Metadata("tokenizer.ggml.eos_token_id") ?? 1)
        addBOS = file.boolMetadata("tokenizer.ggml.add_bos_token") ?? true
        addEOS = file.boolMetadata("tokenizer.ggml.add_eos_token") ?? false
        addSpacePrefix = file.boolMetadata("tokenizer.ggml.add_space_prefix") ?? true
    }
}

// MARK: - SentencePiece Tokenizer (Merge-Based)

struct SentencePieceTokenizer {

    // MARK: - Internal Types

    private struct Symbol {
        var offset: Int    // byte offset in the UTF-8 buffer
        var length: Int    // current byte length (0 = consumed)
        var prev: Int      // previous symbol index, or -1
        var next: Int      // next symbol index, or -1
    }

    private struct MergeCandidate {
        let left: Int
        let right: Int
        let byteLength: Int   // expected combined byte length at time of seeding
        let score: Float
    }

    // MARK: - Properties

    let metadata: GGUFTokenizerMetadata
    let tokenToID: [[UInt8]: Int]
    let byteTokenToID: [UInt8: Int]
    let tokenScores: [Float]

    // MARK: - Initialization

    init(metadata: GGUFTokenizerMetadata) throws {
        guard metadata.model == "llama" else {
            throw TokenizerError.unsupportedTokenizer(metadata.model)
        }
        self.metadata = metadata

        // Build token-to-ID map keyed by exact UTF-8 bytes (not String) to avoid
        // Swift canonical String equality changing token byte representations.
        var tokenToID: [[UInt8]: Int] = [:]
        for (index, token) in metadata.tokens.enumerated() {
            let bytes = Array(token.utf8)
            // First occurrence wins (preserve lowest ID for duplicates)
            if tokenToID[bytes] == nil {
                tokenToID[bytes] = index
            }
        }
        self.tokenToID = tokenToID

        // Map each byte value to its <0xNN> token ID
        self.byteTokenToID = (UInt8.min...UInt8.max).reduce(into: [:]) { result, byte in
            let hexKey = Array(String(format: "<0x%02X>", byte).utf8)
            result[byte] = tokenToID[hexKey]
        }

        self.tokenScores = metadata.scores
    }

    // MARK: - Public API

    func encode(_ text: String) -> [Int] {
        let normalized = normalize(text)
        let utf8Bytes = Array(normalized.utf8)

        guard !utf8Bytes.isEmpty else {
            var ids: [Int] = []
            if metadata.addBOS { ids.append(metadata.bosTokenID) }
            if metadata.addEOS { ids.append(metadata.eosTokenID) }
            return ids
        }

        // Build symbols: one per Unicode scalar (not Swift Character)
        var symbols = buildSymbols(from: normalized)

        // Seed heap with known adjacent pairs
        var heap = buildHeap(symbols: symbols, utf8Bytes: utf8Bytes)

        // Main merge loop
        while let candidate = popMax(&heap) {
            applyCandidate(candidate, symbols: &symbols, utf8Bytes: utf8Bytes, heap: &heap)
        }

        // Traverse surviving symbols to emit token IDs
        var ids = emitTokens(symbols: symbols, utf8Bytes: utf8Bytes)

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

    // MARK: - Normalization

    private func normalize(_ text: String) -> String {
        var normalized = text.replacingOccurrences(of: " ", with: "▁")
        if metadata.addSpacePrefix, !normalized.hasPrefix("▁") {
            normalized = "▁" + normalized
        }
        return normalized
    }

    // MARK: - Symbol Construction

    private func buildSymbols(from text: String) -> [Symbol] {
        var symbols: [Symbol] = []
        var byteOffset = 0

        for scalar in text.unicodeScalars {
            let len = scalar.utf8.count
            symbols.append(Symbol(
                offset: byteOffset,
                length: len,
                prev: symbols.count - 1,
                next: symbols.count + 1
            ))
            byteOffset += len
        }

        // Fix last symbol's next
        if !symbols.isEmpty {
            symbols[symbols.count - 1].next = -1
        }

        return symbols
    }

    // MARK: - Heap Operations (Binary Max-Heap)

    private func buildHeap(symbols: [Symbol], utf8Bytes: [UInt8]) -> [MergeCandidate] {
        var heap: [MergeCandidate] = []

        for i in 0..<(symbols.count - 1) {
            pushNeighborCandidate(left: i, right: i + 1, symbols: symbols, utf8Bytes: utf8Bytes, heap: &heap)
        }

        return heap
    }

    private func pushMax(_ heap: inout [MergeCandidate], _ entry: MergeCandidate) {
        heap.append(entry)
        siftUp(&heap, heap.count - 1)
    }

    private func popMax(_ heap: inout [MergeCandidate]) -> MergeCandidate? {
        guard !heap.isEmpty else { return nil }
        if heap.count == 1 { return heap.removeLast() }
        let result = heap[0]
        heap[0] = heap.removeLast()
        if !heap.isEmpty { siftDown(&heap, 0) }
        return result
    }

    private func siftUp(_ heap: inout [MergeCandidate], _ i: Int) {
        var child = i
        while child > 0 {
            let parent = (child - 1) / 2
            if !isBetter(heap[child], than: heap[parent]) { break }
            heap.swapAt(parent, child)
            child = parent
        }
    }

    private func siftDown(_ heap: inout [MergeCandidate], _ i: Int) {
        var parent = i
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var best = parent
            if left < heap.count, isBetter(heap[left], than: heap[best]) { best = left }
            if right < heap.count, isBetter(heap[right], than: heap[best]) { best = right }
            if best == parent { break }
            heap.swapAt(parent, best)
            parent = best
        }
    }

    /// Higher score wins; lower left index breaks ties.
    private func isBetter(_ a: MergeCandidate, than b: MergeCandidate) -> Bool {
        if a.score != b.score { return a.score > b.score }
        return a.left < b.left
    }

    // MARK: - Merge Logic

    private func applyCandidate(
        _ candidate: MergeCandidate,
        symbols: inout [Symbol],
        utf8Bytes: [UInt8],
        heap: inout [MergeCandidate]
    ) {
        // Validate: both symbols must still be alive and adjacent
        guard candidate.left >= 0, candidate.left < symbols.count else { return }
        guard candidate.right >= 0, candidate.right < symbols.count else { return }

        let left = symbols[candidate.left]
        guard left.length > 0 else { return }
        guard left.next == candidate.right else { return }

        let right = symbols[candidate.right]
        guard right.length > 0 else { return }

        let currentByteLength = left.length + right.length
        guard currentByteLength == candidate.byteLength else { return }

        // Merge right into left
        symbols[candidate.left].length = currentByteLength
        symbols[candidate.right].length = 0
        symbols[candidate.left].next = right.next
        if right.next >= 0, right.next < symbols.count {
            symbols[right.next].prev = candidate.left
        }

        // Push new neighbor candidates
        let leftPrev = left.prev
        if leftPrev >= 0, leftPrev < symbols.count, symbols[leftPrev].length > 0 {
            pushNeighborCandidate(left: leftPrev, right: candidate.left, symbols: symbols, utf8Bytes: utf8Bytes, heap: &heap)
        }

        let leftNext = symbols[candidate.left].next
        if leftNext >= 0, leftNext < symbols.count, symbols[leftNext].length > 0 {
            pushNeighborCandidate(left: candidate.left, right: leftNext, symbols: symbols, utf8Bytes: utf8Bytes, heap: &heap)
        }
    }

    private func pushNeighborCandidate(
        left leftIdx: Int,
        right rightIdx: Int,
        symbols: [Symbol],
        utf8Bytes: [UInt8],
        heap: inout [MergeCandidate]
    ) {
        let leftStart = symbols[leftIdx].offset
        let rightEnd = symbols[rightIdx].offset + symbols[rightIdx].length
        let mergedBytes = Array(utf8Bytes[leftStart..<rightEnd])

        if let id = tokenToID[mergedBytes] {
            let byteLen = rightEnd - leftStart
            pushMax(&heap, MergeCandidate(
                left: leftIdx,
                right: rightIdx,
                byteLength: byteLen,
                score: tokenScores[id]
            ))
        }
    }

    // MARK: - Token Emission

    private func emitTokens(symbols: [Symbol], utf8Bytes: [UInt8]) -> [Int] {
        var ids: [Int] = []
        var idx = 0

        while idx >= 0, idx < symbols.count {
            let symbol = symbols[idx]
            if symbol.length > 0 {
                let bytes = Array(utf8Bytes[symbol.offset..<(symbol.offset + symbol.length)])
                if let id = tokenToID[bytes] {
                    ids.append(id)
                } else {
                    // Byte-level fallback: emit each byte's token ID
                    for byte in bytes {
                        ids.append(byteTokenToID[byte] ?? metadata.unknownTokenID)
                    }
                }
            }
            idx = symbol.next
        }

        return ids
    }
}

// MARK: - GGUFFile Metadata Helpers

private extension GGUFFile {
    func stringMetadata(_ key: String) throws -> String {
        guard case .string(let value) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return value
    }

    func stringArrayMetadata(_ key: String) throws -> [String] {
        guard case .array(.string, _, let values) = metadata[key] else { throw TokenizerError.missingMetadata(key) }
        return values
    }

    func int32ArrayMetadata(_ key: String) -> [Int32]? {
        guard case .array(.int32, _, let values) = metadata[key] else { return nil }
        return values.compactMap(Int32.init)
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