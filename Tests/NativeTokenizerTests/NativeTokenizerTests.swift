import Foundation
import XCTest
@testable import GGUFCore
@testable import translategemma_native_tokenize

final class NativeTokenizerTests: XCTestCase {
    func testMergePriorityIsNotAnAdditiveUnigramScore() throws {
        let tokenizer = try makeTokenizer(tokens: ["a", "b", "ab"], scores: [0, 0, -100])
        XCTAssertEqual(tokenizer.encode("ab"), [2])
    }

    func testChainedMergesContinueEvenWithLowerScores() throws {
        let tokenizer = try makeTokenizer(
            tokens: ["a", "b", "c", "ab", "bc", "abc"], scores: [0, 0, 0, -5, -5, -100]
        )
        XCTAssertEqual(tokenizer.encode("abc"), [5])
    }

    func testEqualScoreCompetingPairsChooseLeftmost() throws {
        let tokenizer = try makeTokenizer(tokens: ["a", "b", "c", "ab", "bc"], scores: [0, 0, 0, -5, -5])
        XCTAssertEqual(tokenizer.encode("abc"), [3, 2])
    }

    func testHigherScoreWinsOverLeftmostPair() throws {
        let tokenizer = try makeTokenizer(tokens: ["a", "b", "c", "ab", "bc"], scores: [0, 0, 0, -10, -5])
        XCTAssertEqual(tokenizer.encode("abc"), [0, 4])
    }

    func testConsumedLeftSymbolInvalidatesQueuedPair() throws {
        // ab wins; the originally queued bc must not merge the consumed b.
        let tokenizer = try makeTokenizer(
            tokens: ["a", "b", "c", "d", "ab", "bc", "cd"], scores: [0, 0, 0, 0, 10, 9, 5]
        )
        XCTAssertEqual(tokenizer.encode("abcd"), [4, 6])
    }

    func testGrownRightSymbolInvalidatesOldByteLength() throws {
        // cd grows c without consuming it. The stale bc entry has higher priority
        // than the new bcd entry, but must not merge the now-longer c as though it were bc.
        let tokenizer = try makeTokenizer(
            tokens: ["a", "b", "c", "d", "ab", "bc", "cd", "bcd"],
            scores: [0, 0, 0, 0, 8, 9, 10, 1]
        )
        XCTAssertEqual(tokenizer.encode("abcd"), [4, 6])
    }

    func testCombiningMarksStartAsSeparateCodepoints() throws {
        let tokenizer = try makeTokenizer(tokens: ["e", "\u{0301}"])
        XCTAssertEqual(tokenizer.encode("e\u{0301}"), [0, 1])
    }

    func testVocabularyMatchingUsesExactUTF8Bytes() throws {
        let tokenizer = try makeTokenizer(tokens: ["é", "e\u{0301}"])
        XCTAssertEqual(tokenizer.encode("é"), [0])
        XCTAssertEqual(tokenizer.encode("e\u{0301}"), [1])
    }

    func testKnownEmojiIsOneCodepointToken() throws {
        let tokenizer = try makeTokenizer(tokens: ["<unk>", "🧪"])
        XCTAssertEqual(tokenizer.encode("🧪"), [1])
    }

    func testMultibyteFallbackKeepsByteAndTokenOrder() throws {
        let tokenizer = try makeTokenizer(tokens: ["<unk>", "a", "<0xF0>", "<0x9F>", "<0xA7>", "<0xAA>", "b"])
        XCTAssertEqual(tokenizer.encode("a🧪b"), [1, 2, 3, 4, 5, 6])
    }

    func testMissingByteTokenFallsBackToUnknownWithoutDroppingBytes() throws {
        let tokenizer = try makeTokenizer(tokens: ["<unk>", "<0xF0>", "<0xA7>"])
        XCTAssertEqual(tokenizer.encode("🧪"), [1, 0, 2, 0])
    }

    func testCompetingLongerPieceDoesNotSuppressLeftoverFallback() throws {
        let tokenizer = try makeTokenizer(
            tokens: ["<unk>", "ab", "bc", "<0x61>", "<0x63>"], scores: [0, -100, -1, 0, 0]
        )
        XCTAssertEqual(tokenizer.encode("abc"), [3, 2])
    }

    func testEmptyInputWithoutPrefixOrSpecialTokens() throws {
        XCTAssertEqual(try makeTokenizer(tokens: ["a"]).encode(""), [])
    }

    func testBOSAndEOSFlagsAndEmptyInput() throws {
        let tokenizer = try makeTokenizer(tokens: ["<unk>", "<bos>", "<eos>", "a"], addBOS: true, addEOS: true)
        XCTAssertEqual(tokenizer.encode("a"), [1, 3, 2])
        XCTAssertEqual(tokenizer.encode(""), [1, 2])
    }

    func testSpaceReplacementAndOptionalPrefixRemainUnchanged() throws {
        let tokenizer = try makeTokenizer(tokens: ["▁", "a", "▁a"], addSpacePrefix: true)
        XCTAssertEqual(tokenizer.encode("a a"), [2, 2])
        XCTAssertEqual(tokenizer.encode("▁a"), [2])
        XCTAssertEqual(tokenizer.encode(""), [0])
    }

    func testDuplicateTokenBytesKeepFirstID() throws {
        XCTAssertEqual(try makeTokenizer(tokens: ["a", "a"], scores: [0, -100]).encode("a"), [0])
    }

    func testPiecesForValidAndInvalidIDs() throws {
        let tokenizer = try makeTokenizer(tokens: ["a", "b"])
        XCTAssertEqual(tokenizer.pieces(for: [0, 1, -1, 999]), ["a", "b", "<invalid>", "<invalid>"])
    }

    func testUnsupportedModelsAreRejectedBeforeScoreParsing() throws {
        for model in ["t5", "gpt2", "replit"] {
            let file = try readFixture(metadata: ["tokenizer.ggml.model": .string(model)])
            XCTAssertThrowsError(try GGUFTokenizerMetadata(file: file)) { error in
                guard case TokenizerError.unsupportedTokenizer(let actual) = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
                XCTAssertEqual(actual, model)
            }
        }
    }

    func testMissingScoresAreRejected() throws {
        var values = fixtureMetadata(tokens: ["a"], scores: [0])
        values.removeValue(forKey: "tokenizer.ggml.scores")
        let file = try readFixture(metadata: values)
        XCTAssertThrowsError(try GGUFTokenizerMetadata(file: file)) { error in
            guard case TokenizerError.missingMetadata("tokenizer.ggml.scores") = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testWrongScoreArrayTypeIsRejected() throws {
        var values = fixtureMetadata(tokens: ["a"], scores: [0])
        values["tokenizer.ggml.scores"] = .array(type: .string, count: 1, values: ["0"])
        assertInvalidScores(try readFixture(metadata: values), contains: "float32 array")
    }

    func testScoreCountTooShortIsRejected() throws {
        assertInvalidScores(try readFixture(metadata: fixtureMetadata(tokens: ["a", "b"], scores: [0])), contains: "tokens count")
    }

    func testScoreCountTooLongIsRejected() throws {
        assertInvalidScores(try readFixture(metadata: fixtureMetadata(tokens: ["a"], scores: [0, 0])), contains: "tokens count")
    }

    func testNaNScoreIsRejected() throws {
        assertInvalidScores(try readFixture(metadata: fixtureMetadata(tokens: ["a"], scores: [.nan])), contains: "non-finite")
    }

    func testPositiveInfinityScoreIsRejected() throws {
        assertInvalidScores(try readFixture(metadata: fixtureMetadata(tokens: ["a"], scores: [.infinity])), contains: "non-finite")
    }

    func testNegativeInfinityCannotSilentlyEraseInput() throws {
        let file = try readFixture(metadata: fixtureMetadata(tokens: ["<unk>", "a", "<0x62>"], scores: [0, -1, -.infinity]))
        assertInvalidScores(file, contains: "non-finite")
    }

    func testUnparseableScoreIsNotSilentlyDropped() {
        var values = fixtureMetadata(tokens: ["a", "b"], scores: [0, 0])
        values["tokenizer.ggml.scores"] = .array(type: .float32, count: 2, values: ["0", "bad"])
        assertInvalidScores(makeFile(metadata: values), contains: "unparsable")
    }

    func testInconsistentStoredScoreCountIsRejected() {
        var values = fixtureMetadata(tokens: ["a"], scores: [0])
        values["tokenizer.ggml.scores"] = .array(type: .float32, count: 2, values: ["0"])
        assertInvalidScores(makeFile(metadata: values), contains: "count mismatch")
    }

    private func assertInvalidScores(_ file: GGUFFile, contains detail: String, file sourceFile: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try GGUFTokenizerMetadata(file: file), file: sourceFile, line: line) { error in
            guard case TokenizerError.invalidMetadata(let message) = error else {
                return XCTFail("Unexpected error: \(error)", file: sourceFile, line: line)
            }
            XCTAssertTrue(message.contains(detail), message, file: sourceFile, line: line)
        }
    }

    private func makeTokenizer(tokens: [String], scores: [Float]? = nil, addBOS: Bool = false, addEOS: Bool = false, addSpacePrefix: Bool = false) throws -> SentencePieceTokenizer {
        var values = fixtureMetadata(tokens: tokens, scores: scores ?? Array(repeating: 0, count: tokens.count))
        values["tokenizer.ggml.add_bos_token"] = .scalar(String(addBOS))
        values["tokenizer.ggml.add_eos_token"] = .scalar(String(addEOS))
        values["tokenizer.ggml.add_space_prefix"] = .scalar(String(addSpacePrefix))
        return try SentencePieceTokenizer(metadata: GGUFTokenizerMetadata(file: readFixture(metadata: values)))
    }

    private func fixtureMetadata(tokens: [String], scores: [Float]) -> [String: GGUFMetadataValue] {
        [
            "tokenizer.ggml.model": .string("llama"),
            "tokenizer.ggml.tokens": .array(type: .string, count: UInt64(tokens.count), values: tokens),
            "tokenizer.ggml.scores": .array(type: .float32, count: UInt64(scores.count), values: scores.map { String($0) }),
            "tokenizer.ggml.unknown_token_id": .scalar("0"),
            "tokenizer.ggml.bos_token_id": .scalar("1"),
            "tokenizer.ggml.eos_token_id": .scalar("2"),
            "tokenizer.ggml.add_bos_token": .scalar("false"),
            "tokenizer.ggml.add_eos_token": .scalar("false"),
            "tokenizer.ggml.add_space_prefix": .scalar("false"),
        ]
    }

    // Use @testable GGUFCore only for string-level corruption impossible in a binary float32.
    private func makeFile(metadata: [String: GGUFMetadataValue]) -> GGUFFile {
        GGUFFile(version: 3, tensorCount: 0, metadata: metadata, tensors: [], tensorDataStart: 0, data: Data())
    }

    private func readFixture(metadata: [String: GGUFMetadataValue]) throws -> GGUFFile {
        var data = Data("GGUF".utf8)
        appendInteger(UInt32(3), to: &data)
        appendInteger(UInt64(0), to: &data)
        appendInteger(UInt64(metadata.count), to: &data)
        for key in metadata.keys.sorted() {
            appendString(key, to: &data)
            switch metadata[key]! {
            case .string(let value):
                appendInteger(GGUFValueType.string.rawValue, to: &data)
                appendString(value, to: &data)
            case .scalar(let value):
                if let bool = Bool(value) {
                    appendInteger(GGUFValueType.bool.rawValue, to: &data)
                    data.append(bool ? 1 : 0)
                } else {
                    appendInteger(GGUFValueType.uint32.rawValue, to: &data)
                    appendInteger(try XCTUnwrap(UInt32(value)), to: &data)
                }
            case .array(let type, let count, let values):
                appendInteger(GGUFValueType.array.rawValue, to: &data)
                appendInteger(type.rawValue, to: &data)
                appendInteger(count, to: &data)
                for value in values {
                    if type == .string {
                        appendString(value, to: &data)
                    } else {
                        XCTAssertEqual(type, .float32)
                        appendInteger(try XCTUnwrap(Float(value)).bitPattern, to: &data)
                    }
                }
            }
        }
        data.append(Data(count: (32 - data.count % 32) % 32))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        try data.write(to: url)
        defer {
            do { try FileManager.default.removeItem(at: url) }
            catch { XCTFail("Failed to remove fixture: \(error)") }
        }
        return try GGUFReader().read(path: url.path)
    }

    private func appendInteger<T: FixedWidthInteger>(_ integer: T, to data: inout Data) {
        var value = integer.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }

    private func appendString(_ string: String, to data: inout Data) {
        let bytes = Array(string.utf8)
        appendInteger(UInt64(bytes.count), to: &data)
        data.append(contentsOf: bytes)
    }
}
