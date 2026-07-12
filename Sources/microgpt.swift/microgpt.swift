import Foundation

/// A tiny, dependency-free Swift port of Andrej Karpathy's atomic GPT trainer.
/// Run with: `swift run`
@main
struct MicroGPT {
    static func main() throws {
        var rng = SeededRandomNumberGenerator(seed: 42)

        if !FileManager.default.fileExists(atPath: "input.txt") {
            let namesURL = URL(string: "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt")!
            let data = try Data(contentsOf: namesURL)
            try data.write(to: URL(fileURLWithPath: "input.txt"))
        }

        let input = try String(contentsOfFile: "input.txt", encoding: .utf8)
        var docs = input.split(whereSeparator: \ .isNewline).map(String.init).filter { !$0.isEmpty }
        docs.shuffle(using: &rng)
        print("num docs: \(docs.count)")

        let uchars = Array(Set(docs.joined())).sorted()
        let charToID = Dictionary(uniqueKeysWithValues: uchars.enumerated().map { ($0.element, $0.offset) })
        let bos = uchars.count
        let vocabSize = uchars.count + 1
        print("vocab size: \(vocabSize)")

        let nLayer = 1
        let nEmbd = 16
        let blockSize = 16
        let nHead = 4
        let headDim = nEmbd / nHead

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

        let params = stateDict.values.flatMap { matrix in matrix.flatMap { $0 } }
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

        let learningRate = 0.01
        let beta1 = 0.85
        let beta2 = 0.99
        let epsAdam = 1e-8
        var m = Array(repeating: 0.0, count: params.count)
        var v = Array(repeating: 0.0, count: params.count)

        let numSteps = Int(ProcessInfo.processInfo.environment["NUM_STEPS"] ?? "1000") ?? 1000
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

        let temperature = 0.5
        print("\n--- inference (new, hallucinated names) ---")
        for sampleIdx in 0..<20 {
            var keys = Array(repeating: [[Value]](), count: nLayer)
            var values = Array(repeating: [[Value]](), count: nLayer)
            var tokenID = bos
            var sample: [Character] = []
            for posID in 0..<blockSize {
                let logits = gpt(tokenID: tokenID, posID: posID, keys: &keys, values: &values)
                let probs = softmax(logits.map { $0 / temperature })
                tokenID = rng.weightedChoice(weights: probs.map(\.data))
                if tokenID == bos { break }
                sample.append(uchars[tokenID])
            }
            print(String(format: "sample %2d: %@", sampleIdx + 1, String(sample)))
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
