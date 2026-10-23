import MLX

/// Per-layer key/value history, the tensor version of microgpt's `keys`/`values` arrays.
///
/// With a `window`, only the last `window - 1` positions are kept between calls — exactly
/// what the next query can still see — so sliding layers use constant memory however long
/// the text gets. This grows by concatenation for clarity; a production cache would
/// preallocate and write in place.
public final class KVCache {
    public let window: Int?
    /// Absolute position of the next token to be appended.
    public private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    public init(window: Int?) {
        self.window = window
    }

    /// Appends `[B, H, L, D]` keys and values and returns everything attention should see
    /// for these queries, with the absolute position of the first returned key.
    func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (keys: MLXArray, values: MLXArray, start: Int)
    {
        let allKeys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let allValues = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        let start = offset - (keys?.dim(2) ?? 0)
        offset += newKeys.dim(2)

        if let window, allKeys.dim(2) > window - 1 {
            let keep = window - 1
            keys = allKeys[0..., 0..., (-keep)...]
            values = allValues[0..., 0..., (-keep)...]
        } else {
            keys = allKeys
            values = allValues
        }
        return (allKeys, allValues, start)
    }

    /// The arrays to hand to `eval` so the cache does not keep a growing lazy graph alive.
    public var state: [MLXArray] { [keys, values].compactMap { $0 } }
}
