//! Shared deterministic PRNG and distribution sampling.
//!
//! A seeded `xorshift64*` generator — small, fast, and reproducible, so the
//! solver, evaluators, and clustering all draw the same way without a `rand`
//! dependency.  The math here is the single source of truth previously
//! copy-pasted across `mccfr`, `aivat`, `local_br`, `clustering`, and the
//! abstraction test helpers.

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
