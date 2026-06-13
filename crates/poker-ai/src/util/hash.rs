//! FNV-1a hashing — folds a variable-length information-set descriptor into the
//! `u64` key the [`Game`](crate::games::Game) trait requires.  64-bit collision
//! risk is negligible for the thousands of info sets in these games.
//!
//! Shared here (rather than inside one game module) because the resolver and
//! several games all key their info sets with the same hash.

/// One-shot FNV-1a over a byte slice.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.write_all(bytes);
    h.finish()
}

/// Streaming FNV-1a — the same hash as [`fnv1a`] but fed incrementally, so an
/// information-set key can be folded directly from its parts without first
/// materializing a `Vec<u8>`.  This is what lets the cursor-based hot path
/// (see [`crate::games::CursorGame`]) build keys with zero per-node allocation.
pub(crate) struct Fnv1a(u64);

impl Fnv1a {
    #[inline]
    pub(crate) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    #[inline]
    pub(crate) fn write(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }

    #[inline]
    pub(crate) fn write_all(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write(b);
        }
    }

    #[inline]
    pub(crate) fn finish(self) -> u64 {
        self.0
    }
}
