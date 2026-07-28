import XCTest
@testable import GGUFCore

final class GGUFCoreTests: XCTestCase {
    func testGGMLTypeNames() {
        XCTAssertEqual(GGMLType.name(for: 0), "F32")
        XCTAssertEqual(GGMLType.name(for: 1), "F16")
        XCTAssertEqual(GGMLType.name(for: 12), "Q4_K")
        XCTAssertEqual(GGMLType.name(for: 14), "Q6_K")
        XCTAssertEqual(GGMLType.name(for: 999), "UNKNOWN(999)")
    }

    func testUnquantizedByteCounts() {
        XCTAssertEqual(GGMLType.byteCount(type: 0, elementCount: 10), 40)
        XCTAssertEqual(GGMLType.byteCount(type: 1, elementCount: 10), 20)
        XCTAssertEqual(GGMLType.byteCount(type: 24, elementCount: 10), 10)
        XCTAssertEqual(GGMLType.byteCount(type: 27, elementCount: 10), 80)
        XCTAssertEqual(GGMLType.byteCount(type: 30, elementCount: 10), 20)
    }

    func testQuantizedBlockByteCountsRoundUp() {
        XCTAssertEqual(GGMLType.byteCount(type: 12, elementCount: 256), 144)
        XCTAssertEqual(GGMLType.byteCount(type: 12, elementCount: 257), 288)
        XCTAssertEqual(GGMLType.byteCount(type: 14, elementCount: 256), 210)
        XCTAssertEqual(GGMLType.byteCount(type: 14, elementCount: 512), 420)
        XCTAssertEqual(GGMLType.byteCount(type: 2, elementCount: 32), 18)
        XCTAssertEqual(GGMLType.byteCount(type: 2, elementCount: 33), 36)
    }

    func testUnsupportedByteCountReturnsNil() {
        XCTAssertNil(GGMLType.byteCount(type: 16, elementCount: 256))
        XCTAssertNil(GGMLType.byteCount(type: 999, elementCount: 1))
    }

    func testBinaryReaderReadsLittleEndianValues() throws {
        let data = Data([
            0x34, 0x12,
            0x78, 0x56, 0x34, 0x12,
            0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12,
        ])
        var reader = BinaryReader(data: data)
        XCTAssertEqual(try reader.readUInt16(), 0x1234)
        XCTAssertEqual(try reader.readUInt32(), 0x12345678)
        XCTAssertEqual(try reader.readUInt64(), 0x1234567890abcdef)
        XCTAssertEqual(reader.offset, data.count)
    }

    func testMetadataDescriptionPreviewsLargeArrays() {
        let value = GGUFMetadataValue.array(
            type: .string,
            count: 10,
            values: (0..<10).map { "token\($0)" }
        )
        XCTAssertTrue(value.description.contains("token0"))
        XCTAssertTrue(value.description.contains("token7"))
        XCTAssertFalse(value.description.contains("token8"))
        XCTAssertTrue(value.description.contains("…"))
    }
}
