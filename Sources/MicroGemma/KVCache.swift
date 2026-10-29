import MLX

/// Per-layer key/value history, the tensor version of microgpt's `keys`/`values` arrays.
///
/// Keys and values live in preallocated buffers that new positions are written into, so a
/// decode step writes one slot instead of copying the whole history.
///
/// - Full-attention layers (`window == nil`) append to a buffer that grows in `step`-sized
///   chunks.
/// - Sliding layers keep a ring of `window` slots: position `p` lives in slot `p % window`.
///   Attention doesn't care about slot order — RoPE has already baked each key's position
///   in — so single-token steps just overwrite the oldest slot and read the whole ring.
///   Multi-token updates (prefill) unroll the ring into position order first, because their
///   causal/window mask is built from positions.
public final class KVCache {
    public let window: Int?
    /// Absolute position of the next token to be appended.
    public private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?
    private let step = 256

    public init(window: Int?) {
        self.window = window
    }

    /// Appends `[B, H, L, D]` keys and values and returns everything attention should see
    /// for these queries, with the absolute position of the first returned key.
    func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (keys: MLXArray, values: MLXArray, start: Int)
    {
        if let window {
            if newKeys.dim(2) == 1 {
                return appendToRing(newKeys, newValues, window: window)
            }
            return rewriteRing(newKeys, newValues, window: window)
        }
        return append(newKeys, newValues)
    }

    private func append(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray, Int) {
        let l = k.dim(2)
        if keys == nil || offset + l > keys!.dim(2) {
            let grow = (l + step - 1) / step * step
            keys = Self.extend(keys, by: grow, like: k, upTo: offset)
            values = Self.extend(values, by: grow, like: v, upTo: offset)
        }
        keys![0..., 0..., offset ..< offset + l, 0...] = k
        values![0..., 0..., offset ..< offset + l, 0...] = v
        offset += l
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...], 0)
    }

    private func appendToRing(_ k: MLXArray, _ v: MLXArray, window: Int)
        -> (MLXArray, MLXArray, Int)
    {
        if keys == nil {
            keys = Self.extend(nil, by: window, like: k, upTo: 0)
            values = Self.extend(nil, by: window, like: v, upTo: 0)
        }
        let slot = offset % window
        keys![0..., 0..., slot ..< slot + 1, 0...] = k
        values![0..., 0..., slot ..< slot + 1, 0...] = v
        offset += 1
        let count = min(offset, window)
        // Slots are out of position order once the ring has wrapped; `start` still names the
        // oldest visible position, which is all the mask logic needs.
        return (keys![0..., 0..., ..<count, 0...], values![0..., 0..., ..<count, 0...], offset - count)
    }

    private func rewriteRing(_ k: MLXArray, _ v: MLXArray, window: Int)
        -> (MLXArray, MLXArray, Int)
    {
        let history = min(offset, window)
        let start = offset - history
        let allKeys = keys.map { concatenated([Self.unroll($0, offset: offset, window: window), k], axis: 2) } ?? k
        let allValues = values.map { concatenated([Self.unroll($0, offset: offset, window: window), v], axis: 2) } ?? v
        offset += k.dim(2)

        // Keep the newest `window` positions, laid out so position p sits in slot p % window.
        let kept = min(offset, window)
        var ringKeys = allKeys[0..., 0..., (-kept)..., 0...]
        var ringValues = allValues[0..., 0..., (-kept)..., 0...]
        if kept == window {
            ringKeys = roll(ringKeys, shift: offset % window, axis: 2)
            ringValues = roll(ringValues, shift: offset % window, axis: 2)
        } else {
            ringKeys = Self.extend(ringKeys, by: window - kept, like: k, upTo: kept)
            ringValues = Self.extend(ringValues, by: window - kept, like: v, upTo: kept)
        }
        keys = ringKeys
        values = ringValues
        return (allKeys, allValues, start)
    }

    /// The ring's contents in position order.
    private static func unroll(_ ring: MLXArray, offset: Int, window: Int) -> MLXArray {
        if offset <= window { return ring[0..., 0..., ..<offset, 0...] }
        let oldest = offset % window
        return concatenated([ring[0..., 0..., oldest..., 0...], ring[0..., 0..., ..<oldest, 0...]], axis: 2)
    }

    /// `buffer[..<used]` followed by `extra` zeroed slots, shaped like `like` along the other axes.
    private static func extend(_ buffer: MLXArray?, by extra: Int, like: MLXArray, upTo used: Int)
        -> MLXArray
    {
        let (b, h, d) = (like.dim(0), like.dim(1), like.dim(3))
        let zeros = MLXArray.zeros([b, h, extra, d], dtype: like.dtype)
        guard let buffer, used > 0 else { return zeros }
        return concatenated([buffer[0..., 0..., ..<used, 0...], zeros], axis: 2)
    }

    /// The arrays to hand to `eval` so the cache does not keep a growing lazy graph alive.
    public var state: [MLXArray] { [keys, values].compactMap { $0 } }
}
