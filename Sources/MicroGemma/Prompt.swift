import Foundation

/// TranslateGemma's prompt, rebuilt in Swift from its `chat_template.jinja`.
///
/// The template is fixed text around three inputs — source language, target language and the
/// text — so string building replaces a Jinja engine. The language-code table is read from
/// the template itself rather than copied here.
public struct TranslationPrompt: Sendable {
    /// Language code (e.g. `en`, `pt-BR`, `zh-Hant`) to English name.
    public let languages: [String: String]

    public init(chatTemplate url: URL) throws {
        let template = try String(contentsOf: url, encoding: .utf8)
        // The table is a Jinja dict literal: `"aa-DJ": "Afar",` one entry per line.
        let entry = /"([A-Za-z0-9-]+)":\s*"([^"]+)"/
        var languages = [String: String]()
        for match in template.matches(of: entry) {
            languages[String(match.1)] = String(match.2)
        }
        self.languages = languages
    }

    /// The prompt text after `<bos>`, ending where the model starts its reply.
    public func render(text: String, source: String, target: String) throws -> String {
        let sourceCode = source.replacingOccurrences(of: "_", with: "-")
        let targetCode = target.replacingOccurrences(of: "_", with: "-")
        guard let sourceName = languages[sourceCode] else {
            throw MicroGemmaError.unknownLanguage(source)
        }
        guard let targetName = languages[targetCode] else {
            throw MicroGemmaError.unknownLanguage(target)
        }
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)

        return "<start_of_turn>user\n"
            + "You are a professional \(sourceName) (\(sourceCode)) to \(targetName) (\(targetCode)) "
            + "translator. Your goal is to accurately convey the meaning and nuances of the "
            + "original \(sourceName) text while adhering to \(targetName) grammar, vocabulary, "
            + "and cultural sensitivities.\n"
            + "Produce only the \(targetName) translation, without any additional explanations "
            + "or commentary. Please translate the following \(sourceName) text into "
            + "\(targetName):\n\n\n"
            + body
            + "<end_of_turn>\n"
            + "<start_of_turn>model\n"
    }
}
