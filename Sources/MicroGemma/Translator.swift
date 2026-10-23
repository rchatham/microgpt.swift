import Foundation
import MLX
import Tokenizers

/// Loads a TranslateGemma checkpoint directory and translates text with it.
public final class Translator {
    public let model: Gemma3TextModel
    public let tokenizer: any Tokenizer
    public let prompt: TranslationPrompt
    let stopTokens: Set<Int>

    public init(model: Gemma3TextModel, tokenizer: any Tokenizer, prompt: TranslationPrompt) {
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        // The model ends its turn with <end_of_turn>; <eos> is a backstop.
        self.stopTokens = Set(
            [tokenizer.convertTokenToId("<end_of_turn>"), tokenizer.eosTokenId].compactMap { $0 })
    }

    public static func load(from directory: URL) async throws -> Translator {
        let model = try Gemma3TextModel.load(from: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let prompt = try TranslationPrompt(chatTemplate: directory.appending(path: "chat_template.jinja"))
        return Translator(model: model, tokenizer: tokenizer, prompt: prompt)
    }

    /// Token IDs for the full prompt, starting with `<bos>`.
    public func promptTokens(text: String, source: String, target: String) throws -> [Int] {
        let rendered = try prompt.render(text: text, source: source, target: target)
        let bos = tokenizer.bosTokenId.map { [$0] } ?? []
        return bos + tokenizer.encode(text: rendered, addSpecialTokens: false)
    }

    /// Translates `text`, calling `onText` with each newly decoded piece as it is generated.
    @discardableResult
    public func translate(
        _ text: String, from source: String, to target: String,
        maxTokens: Int = 512, sampling: Sampling = .greedy,
        onText: (String) -> Void = { _ in }
    ) throws -> (text: String, stats: GenerationStats) {
        let tokens = try promptTokens(text: text, source: source, target: target)
        var generated: [Int] = []
        var emitted = ""

        let stats = model.generate(prompt: tokens, maxTokens: maxTokens, sampling: sampling) { next in
            if stopTokens.contains(next) { return false }
            generated.append(next)

            // Re-decode the whole reply so multi-token characters come out whole; hold back
            // a trailing U+FFFD, which means a UTF-8 sequence is still incomplete.
            let decoded = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
            if !decoded.hasSuffix("\u{FFFD}"), decoded.count > emitted.count {
                onText(String(decoded.dropFirst(emitted.count)))
                emitted = decoded
            }
            return true
        }
        return (tokenizer.decode(tokens: generated, skipSpecialTokens: true), stats)
    }
}
