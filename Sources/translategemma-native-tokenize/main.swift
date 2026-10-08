import Foundation
import GGUFCore

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    guard args.count >= 2 else { throw TokenizerError.usage }
    let path = args[0]
    let text = args.dropFirst().joined(separator: " ")
    let file = try GGUFReader().read(path: path)
    let metadata = try GGUFTokenizerMetadata(file: file)
    let tokenizer = try SentencePieceTokenizer(metadata: metadata)
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
