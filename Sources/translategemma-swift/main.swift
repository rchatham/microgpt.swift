import Foundation
import LocalLLMClient
import LocalLLMClientLlama

@main
struct TranslateGemmaSwift {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func run() async throws {
        let env = ProcessInfo.processInfo.environment
        if CommandLine.arguments.contains("--help") || env["HELP"] == "1" {
            printHelp()
            return
        }

        let modelPath = env["MODEL"] ?? "models/translategemma-4b-it-Q4_K_M.gguf"
        let sourceLanguage = env["SOURCE"] ?? "English"
        let targetLanguage = env["TARGET"] ?? "Spanish"
        let contextSize = Int(env["CTX_SIZE"] ?? "4096") ?? 4096
        let temperature = Float(env["TEMP"] ?? "0.1") ?? 0.1
        let topK = Int(env["TOP_K"] ?? "40") ?? 40
        let topP = Float(env["TOP_P"] ?? "0.95") ?? 0.95
        let input = inputText()

        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw TranslateGemmaError.modelNotFound(modelPath)
        }

        let userPrompt = input.isEmpty ? "The weather is beautiful today." : input
        let prompt = """
        <bos><start_of_turn>user
        Translate from \(sourceLanguage) to \(targetLanguage). Output only the translation, with no explanation.

        \(userPrompt)<end_of_turn>
        <start_of_turn>model
        """

        let client = try await LocalLLMClient.llama(
            url: modelURL,
            parameter: .init(
                context: contextSize,
                temperature: temperature,
                topK: topK,
                topP: topP
            )
        )

        let stream = try await client.textStream(from: .plain(prompt))

        var output = ""
        for try await text in stream {
            output += text
        }
        print(cleanTranslation(output))
    }

    private static func inputText() -> String {
        let args = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") }
        if !args.isEmpty { return args.joined(separator: " ") }

        if !FileHandle.standardInput.isatty {
            let data = FileHandle.standardInput.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        }

        return ""
    }

    private static func cleanTranslation(_ output: String) -> String {
        let lines = output
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return lines.last ?? output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func printHelp() {
        print("""
        translategemma-swift - run TranslateGemma GGUF weights from Swift

        Build:
          swift build -c release --product translategemma-swift

        Download weights:
          ./scripts/download-translategemma-4b-gguf.sh

        Run:
          MODEL=models/translategemma-4b-it-Q4_K_M.gguf \\
          SOURCE=English TARGET=Spanish \\
          .build/release/translategemma-swift "The weather is beautiful today."

        Input can also come from stdin:
          echo "Good morning" | TARGET=Japanese .build/release/translategemma-swift

        Environment:
          MODEL       Path to local .gguf weights
          SOURCE      Source language, default English
          TARGET      Target language, default Spanish
          CTX_SIZE    Context size, default 4096
          TEMP        Temperature, default 0.1
          TOP_K       Top-k sampling, default 40
          TOP_P       Top-p sampling, default 0.95
        """)
    }
}

enum TranslateGemmaError: Error, CustomStringConvertible {
    case modelNotFound(String)

    var description: String {
        switch self {
        case .modelNotFound(let path):
            "model not found at \(path); run ./scripts/download-translategemma-4b-gguf.sh or set MODEL=/path/to/model.gguf"
        }
    }
}

private extension FileHandle {
    var isatty: Bool {
        Darwin.isatty(fileDescriptor) == 1
    }
}
