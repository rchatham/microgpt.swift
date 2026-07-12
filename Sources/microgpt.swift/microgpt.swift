import Foundation

/// A tiny, dependency-free Swift port of Andrej Karpathy's atomic GPT trainer.
/// Run with: `swift run`
@main
struct MicroGPT {
    static func main() {
        do {
            try run()
        } catch let error as MicroGPTError {
            fputs("error: \(error.description)\n", stderr)
            exit(1)
        } catch {
            fputs("error: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    static func run() throws {
        let env = ProcessInfo.processInfo.environment
        if CommandLine.arguments.contains("--help") || boolEnv(env["HELP"], default: false) {
            printHelp()
            return
        }
        let seed = UInt64(env["SEED"] ?? "42") ?? 42
        var rng = SeededRandomNumberGenerator(seed: seed)

        if !FileManager.default.fileExists(atPath: "input.txt") {
            let namesURL = URL(string: "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt")!
            let data = try Data(contentsOf: namesURL)
            try data.write(to: URL(fileURLWithPath: "input.txt"))
        }

        let inputPath = env["INPUT"] ?? "input.txt"
        let input = try String(contentsOfFile: inputPath, encoding: .utf8)
        var docs = input.split(whereSeparator: \ .isNewline).map(String.init).filter { !$0.isEmpty }
        docs.shuffle(using: &rng)
        let knownNames = Set(docs.map { $0.lowercased() })
        print("num docs: \(docs.count)")

        let uchars = Array(Set(docs.joined())).sorted()
        let charToID = Dictionary(uniqueKeysWithValues: uchars.enumerated().map { ($0.element, $0.offset) })
        let bos = uchars.count
        let vocabSize = uchars.count + 1
        print("vocab size: \(vocabSize)")

        let nLayer = Int(env["N_LAYER"] ?? "1") ?? 1
        let nEmbd = Int(env["N_EMBD"] ?? "16") ?? 16
        let blockSize = Int(env["BLOCK_SIZE"] ?? "16") ?? 16
        let nHead = Int(env["N_HEAD"] ?? "4") ?? 4
        let headDim = nEmbd / nHead
        print("config: layers=\(nLayer) embd=\(nEmbd) heads=\(nHead) block=\(blockSize) seed=\(seed)")

        func matrix(_ nout: Int, _ nin: Int, std: Double = 0.08) -> [[Value]] {
            (0..<nout).map { _ in (0..<nin).map { _ in Value(rng.gaussian(mean: 0, std: std)) } }
        }

        var stateDict: [String: [[Value]]] = [
            "wte": matrix(vocabSize, nEmbd),
            "wpe": matrix(blockSize, nEmbd),
            "lm_head": matrix(vocabSize, nEmbd),
        ]

        for i in 0..<nLayer {
            stateDict["layer\(i).attn_wq"] = matrix(nEmbd, nEmbd)
            stateDict["layer\(i).attn_wk"] = matrix(nEmbd, nEmbd)
            stateDict["layer\(i).attn_wv"] = matrix(nEmbd, nEmbd)
            stateDict["layer\(i).attn_wo"] = matrix(nEmbd, nEmbd)
            stateDict["layer\(i).mlp_fc1"] = matrix(4 * nEmbd, nEmbd)
            stateDict["layer\(i).mlp_fc2"] = matrix(nEmbd, 4 * nEmbd)
        }

        if let checkpointPath = env["LOAD_CHECKPOINT"] {
            let checkpoint = try JSONDecoder().decode(Checkpoint.self, from: Data(contentsOf: URL(fileURLWithPath: checkpointPath)))
            guard checkpoint.uchars == uchars.map(String.init),
                  checkpoint.nLayer == nLayer,
                  checkpoint.nEmbd == nEmbd,
                  checkpoint.blockSize == blockSize,
                  checkpoint.nHead == nHead else {
                throw MicroGPTError.incompatibleCheckpoint
            }
            for (key, savedMatrix) in checkpoint.stateDict {
                stateDict[key] = savedMatrix.map { row in row.map { Value($0) } }
            }
            print("loaded checkpoint: \(checkpointPath)")
        }

        let paramKeys = stateDict.keys.sorted()
        let params = paramKeys.flatMap { key in stateDict[key]!.flatMap { $0 } }
        print("num params: \(params.count)")

        func linear(_ x: [Value], _ w: [[Value]]) -> [Value] {
            w.map { row in zip(row, x).map(*).reduce(Value(0), +) }
        }

        func softmax(_ logits: [Value]) -> [Value] {
            let maxValue = logits.map(\.data).max()!
            let exps = logits.map { ($0 - maxValue).exp() }
            let total = exps.reduce(Value(0), +)
            return exps.map { $0 / total }
        }

        func rmsnorm(_ x: [Value]) -> [Value] {
            let ms = x.map { $0 * $0 }.reduce(Value(0), +) / Double(x.count)
            let scale = (ms + 1e-5).pow(-0.5)
            return x.map { $0 * scale }
        }

        func gpt(tokenID: Int, posID: Int, keys: inout [[[Value]]], values: inout [[[Value]]]) -> [Value] {
            let tokEmb = stateDict["wte"]![tokenID]
            let posEmb = stateDict["wpe"]![posID]
            var x = zip(tokEmb, posEmb).map(+)
            x = rmsnorm(x)

            for li in 0..<nLayer {
                var xResidual = x
                x = rmsnorm(x)
                let q = linear(x, stateDict["layer\(li).attn_wq"]!)
                let k = linear(x, stateDict["layer\(li).attn_wk"]!)
                let v = linear(x, stateDict["layer\(li).attn_wv"]!)
                keys[li].append(k)
                values[li].append(v)

                var xAttn: [Value] = []
                for h in 0..<nHead {
                    let hs = h * headDim
                    let qH = Array(q[hs..<(hs + headDim)])
                    let kH = keys[li].map { Array($0[hs..<(hs + headDim)]) }
                    let vH = values[li].map { Array($0[hs..<(hs + headDim)]) }
                    let attnLogits = kH.map { kt in
                        zip(qH, kt).map(*).reduce(Value(0), +) / sqrt(Double(headDim))
                    }
                    let attnWeights = softmax(attnLogits)
                    let headOut = (0..<headDim).map { j in
                        (0..<vH.count).map { t in attnWeights[t] * vH[t][j] }.reduce(Value(0), +)
                    }
                    xAttn.append(contentsOf: headOut)
                }

                x = linear(xAttn, stateDict["layer\(li).attn_wo"]!)
                x = zip(x, xResidual).map(+)

                xResidual = x
                x = rmsnorm(x)
                x = linear(x, stateDict["layer\(li).mlp_fc1"]!)
                x = x.map { $0.relu() }
                x = linear(x, stateDict["layer\(li).mlp_fc2"]!)
                x = zip(x, xResidual).map(+)
            }

            return linear(x, stateDict["lm_head"]!)
        }

        let learningRate = Double(env["LEARNING_RATE"] ?? "0.01") ?? 0.01
        let beta1 = Double(env["BETA1"] ?? "0.85") ?? 0.85
        let beta2 = Double(env["BETA2"] ?? "0.99") ?? 0.99
        let epsAdam = 1e-8
        var m = Array(repeating: 0.0, count: params.count)
        var v = Array(repeating: 0.0, count: params.count)

        let numSteps = Int(env["NUM_STEPS"] ?? "1000") ?? 1000
        for step in 0..<numSteps {
            let doc = docs[step % docs.count]
            let tokens = [bos] + doc.compactMap { charToID[$0] } + [bos]
            let n = min(blockSize, tokens.count - 1)

            var keys = Array(repeating: [[Value]](), count: nLayer)
            var values = Array(repeating: [[Value]](), count: nLayer)
            var losses: [Value] = []
            for posID in 0..<n {
                let tokenID = tokens[posID]
                let targetID = tokens[posID + 1]
                let logits = gpt(tokenID: tokenID, posID: posID, keys: &keys, values: &values)
                let probs = softmax(logits)
                losses.append(-probs[targetID].log())
            }
            let loss = losses.reduce(Value(0), +) / Double(n)
            loss.backward()

            let lrT = learningRate * (1 - Double(step) / Double(numSteps))
            for (i, p) in params.enumerated() {
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad * p.grad
                let mHat = m[i] / (1 - pow(beta1, Double(step + 1)))
                let vHat = v[i] / (1 - pow(beta2, Double(step + 1)))
                p.data -= lrT * mHat / (sqrt(vHat) + epsAdam)
                p.grad = 0
            }

            print(String(format: "step %4d / %4d | loss %.4f", step + 1, numSteps, loss.data), terminator: "\r")
            fflush(stdout)
        }

        if let checkpointPath = env["SAVE_CHECKPOINT"] {
            let savedState = Dictionary(uniqueKeysWithValues: stateDict.map { key, matrix in
                (key, matrix.map { row in row.map(\.data) })
            })
            let checkpoint = Checkpoint(
                uchars: uchars.map(String.init),
                nLayer: nLayer,
                nEmbd: nEmbd,
                blockSize: blockSize,
                nHead: nHead,
                stateDict: savedState
            )
            let data = try JSONEncoder().encode(checkpoint)
            try data.write(to: URL(fileURLWithPath: checkpointPath))
            print("\nsaved checkpoint: \(checkpointPath)")
        }

        let temperature = Double(env["TEMPERATURE"] ?? "0.5") ?? 0.5
        let topK = Int(env["TOP_K"] ?? "0") ?? 0
        let topP = Double(env["TOP_P"] ?? "1.0") ?? 1.0
        let prefix = (env["PREFIX"] ?? "").lowercased()
        let startsWith = (env["STARTS_WITH"] ?? "").lowercased()
        let minLength = Int(env["MIN_LENGTH"] ?? "0") ?? 0
        let maxLength = Int(env["MAX_LENGTH"] ?? String(blockSize)) ?? blockSize
        let maxRepeat = Int(env["MAX_REPEAT"] ?? "2") ?? 2
        let uniqueOnly = boolEnv(env["UNIQUE"], default: true)
        let excludeKnown = boolEnv(env["EXCLUDE_KNOWN"], default: false)
        let requireVowel = boolEnv(env["REQUIRE_VOWEL"], default: true)
        let bannedPatterns = (env["BANNED_PATTERNS"] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }.filter { !$0.isEmpty }
        let numSamples = Int(env["SAMPLES"] ?? "20") ?? 20

        guard prefix.count < blockSize else { throw MicroGPTError.prefixTooLong }
        guard prefix.allSatisfy({ charToID[$0] != nil }) else { throw MicroGPTError.unknownPrefixCharacter(prefix) }
        guard startsWith.isEmpty || startsWith.count == 1 else { throw MicroGPTError.startsWithMustBeOneCharacter }
        guard startsWith.allSatisfy({ charToID[$0] != nil }) else { throw MicroGPTError.unknownStartsWithCharacter(startsWith) }
        guard minLength >= 0, maxLength >= minLength, maxLength <= blockSize else { throw MicroGPTError.invalidLengthRange }

        print("sampling: temp=\(temperature) topK=\(topK) topP=\(topP) prefix='\(prefix)' startsWith='\(startsWith)' length=\(minLength)...\(maxLength) unique=\(uniqueOnly) excludeKnown=\(excludeKnown)")
        print("\n--- inference (new, hallucinated names) ---")
        var printedSamples = 0
        var attempts = 0
        var generatedNames = Set<String>()
        let maxAttempts = max(numSamples * 100, numSamples)
        while printedSamples < numSamples && attempts < maxAttempts {
            attempts += 1
            var keys = Array(repeating: [[Value]](), count: nLayer)
            var values = Array(repeating: [[Value]](), count: nLayer)
            var tokenID = bos
            var sample: [Character] = []
            var posID = 0

            for character in prefix where posID < blockSize {
                _ = gpt(tokenID: tokenID, posID: posID, keys: &keys, values: &values)
                tokenID = charToID[character]!
                sample.append(character)
                posID += 1
            }

            while posID < blockSize {
                let logits = gpt(tokenID: tokenID, posID: posID, keys: &keys, values: &values)
                let probs = softmax(logits.map { $0 / temperature })
                tokenID = rng.weightedChoice(weights: filteredWeights(probs.map(\.data), topK: topK, topP: topP))
                if tokenID == bos { break }
                sample.append(uchars[tokenID])
                posID += 1
                if sample.count >= maxLength { break }
            }

            let generated = String(sample)
            if generated.count < minLength || generated.count > maxLength { continue }
            if hasTooManyRepeats(generated, maxRepeat: maxRepeat) { continue }
            if requireVowel && !containsVowel(generated) { continue }
            if bannedPatterns.contains(where: { generated.contains($0) }) { continue }
            if uniqueOnly && generatedNames.contains(generated) { continue }
            if excludeKnown && knownNames.contains(generated) { continue }
            if prefix.isEmpty && !startsWith.isEmpty && !generated.hasPrefix(startsWith) { continue }
            generatedNames.insert(generated)
            printedSamples += 1
            print(String(format: "sample %2d: %@", printedSamples, generated))
        }

        if printedSamples < numSamples {
            print("warning: only generated \(printedSamples) matching samples after \(attempts) attempts")
        }
    }
}

struct Checkpoint: Codable {
    let uchars: [String]
    let nLayer: Int
    let nEmbd: Int
    let blockSize: Int
    let nHead: Int
    let stateDict: [String: [[Double]]]
}

enum MicroGPTError: Error, CustomStringConvertible {
    case incompatibleCheckpoint
    case prefixTooLong
    case unknownPrefixCharacter(String)
    case startsWithMustBeOneCharacter
    case unknownStartsWithCharacter(String)
    case invalidLengthRange

    var description: String {
        switch self {
        case .incompatibleCheckpoint:
            "checkpoint does not match dataset/model config; pass the same N_LAYER, N_EMBD, N_HEAD, and BLOCK_SIZE used for training"
        case .prefixTooLong:
            "PREFIX must be shorter than BLOCK_SIZE"
        case .unknownPrefixCharacter(let prefix):
            "PREFIX contains characters not present in the dataset vocabulary: \(prefix)"
        case .startsWithMustBeOneCharacter:
            "STARTS_WITH must be empty or exactly one character"
        case .unknownStartsWithCharacter(let value):
            "STARTS_WITH character is not present in the dataset vocabulary: \(value)"
        case .invalidLengthRange:
            "invalid MIN_LENGTH/MAX_LENGTH; require 0 <= MIN_LENGTH <= MAX_LENGTH <= BLOCK_SIZE"
        }
    }
}

final class Value: Hashable {
    var data: Double
    var grad: Double
    private let children: [Value]
    private let localGrads: [Double]

    init(_ data: Double, children: [Value] = [], localGrads: [Double] = []) {
        self.data = data
        self.grad = 0
        self.children = children
        self.localGrads = localGrads
    }

    static func == (lhs: Value, rhs: Value) -> Bool { lhs === rhs }
    func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }

    func pow(_ exponent: Double) -> Value {
        Value(Foundation.pow(data, exponent), children: [self], localGrads: [exponent * Foundation.pow(data, exponent - 1)])
    }

    func log() -> Value {
        Value(Foundation.log(data), children: [self], localGrads: [1 / data])
    }

    func exp() -> Value {
        let out = Foundation.exp(data)
        return Value(out, children: [self], localGrads: [out])
    }

    func relu() -> Value {
        Value(Swift.max(0, data), children: [self], localGrads: [data > 0 ? 1 : 0])
    }

    func backward() {
        var topo: [Value] = []
        var visited = Set<Value>()

        func buildTopo(_ v: Value) {
            if visited.insert(v).inserted {
                for child in v.children { buildTopo(child) }
                topo.append(v)
            }
        }

        buildTopo(self)
        grad = 1
        for v in topo.reversed() {
            for (child, localGrad) in zip(v.children, v.localGrads) {
                child.grad += localGrad * v.grad
            }
        }
    }
}

func + (lhs: Value, rhs: Value) -> Value {
    Value(lhs.data + rhs.data, children: [lhs, rhs], localGrads: [1, 1])
}

func + (lhs: Value, rhs: Double) -> Value { lhs + Value(rhs) }
func + (lhs: Double, rhs: Value) -> Value { Value(lhs) + rhs }

func - (lhs: Value, rhs: Value) -> Value { lhs + (-rhs) }
func - (lhs: Value, rhs: Double) -> Value { lhs - Value(rhs) }
func - (lhs: Double, rhs: Value) -> Value { Value(lhs) - rhs }
prefix func - (value: Value) -> Value { value * -1 }

func * (lhs: Value, rhs: Value) -> Value {
    Value(lhs.data * rhs.data, children: [lhs, rhs], localGrads: [rhs.data, lhs.data])
}

func * (lhs: Value, rhs: Double) -> Value { lhs * Value(rhs) }
func * (lhs: Double, rhs: Value) -> Value { Value(lhs) * rhs }

func / (lhs: Value, rhs: Value) -> Value { lhs * rhs.pow(-1) }
func / (lhs: Value, rhs: Double) -> Value { lhs / Value(rhs) }
func / (lhs: Double, rhs: Value) -> Value { Value(lhs) / rhs }

func filteredWeights(_ weights: [Double], topK: Int, topP: Double) -> [Double] {
    let sorted = weights.enumerated().sorted { $0.element > $1.element }
    var keep = Set(sorted.map(\.offset))

    if topK > 0, topK < weights.count {
        keep = Set(sorted.prefix(topK).map(\.offset))
    }

    if topP > 0, topP < 1 {
        var cumulative = 0.0
        var nucleus = Set<Int>()
        for item in sorted where keep.contains(item.offset) {
            nucleus.insert(item.offset)
            cumulative += item.element
            if cumulative >= topP { break }
        }
        keep = nucleus
    }

    return weights.enumerated().map { keep.contains($0.offset) ? $0.element : 0 }
}

func hasTooManyRepeats(_ text: String, maxRepeat: Int) -> Bool {
    guard maxRepeat > 0 else { return false }
    var previous: Character?
    var count = 0
    for character in text {
        if character == previous {
            count += 1
            if count > maxRepeat { return true }
        } else {
            previous = character
            count = 1
        }
    }
    return false
}

func containsVowel(_ text: String) -> Bool {
    text.contains { "aeiouy".contains($0) }
}

func boolEnv(_ value: String?, default defaultValue: Bool) -> Bool {
    guard let value else { return defaultValue }
    switch value.lowercased() {
    case "1", "true", "yes", "on": return true
    case "0", "false", "no", "off": return false
    default: return defaultValue
    }
}

func printHelp() {
    print("""
    microgpt.swift - tiny dependency-free character GPT trainer/generator

    Training:
      NUM_STEPS=10000 SAVE_CHECKPOINT=model.json .build/release/microgpt.swift

    Inference:
      NUM_STEPS=0 LOAD_CHECKPOINT=model.json SAMPLES=50 .build/release/microgpt.swift

    Model config, must match checkpoint when loading:
      N_LAYER=1 N_EMBD=32 N_HEAD=4 BLOCK_SIZE=16

    Sampling controls:
      TEMPERATURE=0.65 TOP_K=8 TOP_P=0.9 SEED=42 SAMPLES=20
      PREFIX=sha              Continue from a prefix; best for rare starts like z/q/x
      STARTS_WITH=m           Rejection-filter by first letter
      MIN_LENGTH=4 MAX_LENGTH=8 MAX_REPEAT=2
      UNIQUE=true             Suppress duplicate outputs
      EXCLUDE_KNOWN=false     Suppress names present in the training data
      REQUIRE_VOWEL=true      Reject names without a/e/i/o/u/y
      BANNED_PATTERNS=aaa,zzz Reject comma-separated substrings

    Data:
      INPUT=input.txt          One training name/document per line

    Good generation recipe:
      SEED=11 N_EMBD=32 N_HEAD=4 NUM_STEPS=0 LOAD_CHECKPOINT=model-32-10000.json \\
        TEMPERATURE=0.65 TOP_K=8 TOP_P=0.9 MIN_LENGTH=4 MAX_LENGTH=7 \\
        UNIQUE=true EXCLUDE_KNOWN=true SAMPLES=50 .build/release/microgpt.swift
    """)
}

struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64
    private var cachedGaussian: Double?

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func uniform01() -> Double {
        Double(next() >> 11) / Double(1 << 53)
    }

    mutating func gaussian(mean: Double, std: Double) -> Double {
        if let cachedGaussian {
            self.cachedGaussian = nil
            return mean + std * cachedGaussian
        }
        let u1 = max(uniform01(), Double.leastNonzeroMagnitude)
        let u2 = uniform01()
        let radius = sqrt(-2 * log(u1))
        let theta = 2 * Double.pi * u2
        cachedGaussian = radius * sin(theta)
        return mean + std * radius * cos(theta)
    }

    mutating func weightedChoice(weights: [Double]) -> Int {
        let total = weights.reduce(0, +)
        var threshold = uniform01() * total
        for (index, weight) in weights.enumerated() {
            threshold -= weight
            if threshold <= 0 { return index }
        }
        return weights.count - 1
    }
}
