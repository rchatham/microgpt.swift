import Foundation
import MicroGemma

// Usage: translate --from en --to es [--model DIR] [--max-tokens N] [--sample] [TEXT]
// TEXT defaults to stdin. DIR defaults to $MICROGEMMA_MODEL, then
// ~/.cache/microgpt.swift/translategemma-4b-it-4bit.

var arguments = Array(CommandLine.arguments.dropFirst())
@MainActor func option(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
    defer { arguments.removeSubrange(i...(i + 1)) }
    return arguments[i + 1]
}
@MainActor func flag(_ name: String) -> Bool {
    guard let i = arguments.firstIndex(of: name) else { return false }
    arguments.remove(at: i)
    return true
}

let source = option("--from") ?? "en"
let target = option("--to") ?? "es"
let maxTokens = option("--max-tokens").flatMap(Int.init) ?? 512
let modelPath = option("--model")
    ?? ProcessInfo.processInfo.environment["MICROGEMMA_MODEL"]
    ?? NSString(string: "~/.cache/microgpt.swift/translategemma-4b-it-4bit").expandingTildeInPath
let sampling: Sampling = flag("--sample") ? .topKTopP(k: 64, p: 0.95, temperature: 1) : .greedy

let text = arguments.isEmpty
    ? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
    : arguments.joined(separator: " ")

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("translate: \(message)\n".utf8))
    exit(1)
}

let loadStart = Date()
let translator: Translator
do {
    translator = try await Translator.load(from: URL(fileURLWithPath: modelPath))
} catch {
    fail("could not load model from \(modelPath): \(error)")
}
let loadSeconds = Date().timeIntervalSince(loadStart)

let stats: Translator.Stats
do {
    stats = try translator.translate(
        text, from: source, to: target, maxTokens: maxTokens, sampling: sampling
    ) { piece in
        print(piece, terminator: "")
        fflush(stdout)
    }.stats
} catch {
    fail("\(error)")
}
print()
fflush(stdout)

let note = String(
    format: "load %.1fs | prompt %d tok, prefill %.0f tok/s | %d tok, %.1f tok/s",
    loadSeconds, stats.promptTokens, Double(stats.promptTokens) / stats.prefillSeconds,
    stats.generatedTokens, Double(stats.generatedTokens) / stats.decodeSeconds)
FileHandle.standardError.write(Data((note + "\n").utf8))
