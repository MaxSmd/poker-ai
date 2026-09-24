//! Shared deterministic PRNG and distribution sampling.
//!
//! A seeded `xorshift64*` generator — small, fast, and reproducible, so the
//! solver, evaluators, and clustering all draw the same way without a `rand`
//! dependency.  The math here is the single source of truth previously
//! copy-pasted across `mccfr`, `aivat`, `local_br`, `clustering`, the agent,
//! and the test helpers.
//!
//! Two other generators live here too, each kept bit-for-bit because seeded
//! results elsewhere depend on its exact stream: SplitMix64 (per-iteration
//! MCCFR streams, `vector_br`'s board sampling) and plain `xorshift64` (LBR's
//! deals).  Changing any of them silently makes new runs incomparable with
//! old ones at the same seed.

/// One `xorshift64*` step: advance `state` and return the scrambled `u64`.
#[inline]
pub fn xorshift_next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// One `xorshift64*` step yielding a uniform value in `[0, 1)`, built from the
/// top 53 bits (the f64 mantissa width).
#[inline]
pub fn xorshift_next_unit(state: &mut u64) -> f64 {
    (xorshift_next_u64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// One step of Marsaglia's plain `xorshift64` (shifts 13, 7, 17) — no `*`
/// scrambler, so weaker than [`xorshift_next_u64`] in its low bits.  Only for
/// streams that must stay comparable with earlier seeded runs.
#[inline]
pub fn xorshift64_plain_next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// SplitMix64's increment, `2⁶⁴ / φ` — also the conventional multiplier for
/// decorrelating derived seeds.
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's output function: a bijective avalanche of `z`.
#[inline]
pub fn splitmix_mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One SplitMix64 step yielding a uniform value in `[0, 1)`: advance `state`
/// by [`GOLDEN_GAMMA`] and mix.  Any `state`, including zero, is a valid seed.
#[inline]
pub fn splitmix_next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    (splitmix_mix(*state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Seed for stream `stream` of base seed `seed` — independent, reproducible
/// per-iteration streams for parallel workers.  Forced odd, so it is a valid
/// `xorshift64*` state.
#[inline]
pub fn splitmix_stream(seed: u64, stream: u64) -> u64 {
    splitmix_mix(seed ^ stream.wrapping_mul(GOLDEN_GAMMA)) | 1
}

/// Sample an index from a probability stream given a uniform draw `r ∈ [0, 1)`.
/// Returns the last index if rounding leaves `r` past the cumulative sum.
#[inline]
pub fn sample_index(probs: impl Iterator<Item = f64>, r: f64) -> usize {
    let mut acc = 0.0;
    let mut last = 0;
    for (i, p) in probs.enumerate() {
        last = i;
        acc += p;
        if r < acc {
            return i;
        }
    }
    last
}

/// A seeded `xorshift64*` generator with ergonomic draw helpers.
pub struct Rng(pub u64);

impl Rng {
    /// Seed the generator.  The seed is forced odd (`| 1`) since `xorshift64*`
    /// must not start from zero.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Next uniform value in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        xorshift_next_unit(&mut self.0)
    }

    /// Sample an index from a probability distribution.
    #[inline]
    pub fn sample(&mut self, probs: &[f64]) -> usize {
        let r = self.unit();
        sample_index(probs.iter().copied(), r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first draws of every generator, pinned.  Seeded results across the
    /// crate (training runs, LBR matches, `vector_br` board samples) are only
    /// comparable while these streams stay bit-for-bit the same, so a change
    /// here must be deliberate.  The expected values were computed from the
    /// inline copies these functions replaced.
    #[test]
    fn streams_are_pinned() {
        // Units are 53-bit integers over 2⁵³, so scaling back up is exact.
        let units = |next: fn(&mut u64) -> f64, mut s: u64| -> Vec<u64> {
            (0..3).map(|_| (next(&mut s) * (1u64 << 53) as f64) as u64).collect()
        };
        assert_eq!(
            units(xorshift_next_unit, 0xBADC_0FFE),
            [0xc5f86be3cb378, 0x1aacb65b75e70a, 0xa2912fc54a006]
        );
        assert_eq!(
            units(splitmix_next_unit, 7),
            [0xc797c3c8b2641, 0x89879afe878c, 0x1cd30810175625]
        );
        let mut s = 3;
        let plain: Vec<u64> = (0..3).map(|_| xorshift64_plain_next_u64(&mut s)).collect();
        assert_eq!(plain, [0xc18660c3, 0x3000c30a14013cc2, 0xad230d75bb884e3b]);
        let streams: Vec<u64> = (0..3).map(|t| splitmix_stream(42, t)).collect();
        assert_eq!(streams, [0xa759ea27d4727623, 0xbdd732262feb6e95, 0xd9639a006c85adb1]);
    }
}
