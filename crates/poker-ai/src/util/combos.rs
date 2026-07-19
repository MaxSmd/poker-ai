//! The canonical two-card-combo bijection: `{a, b} ↔ 0..1326`.
//!
//! **This is the only combo ordering in the crate.**  Every reach vector,
//! belief distribution, CFV vector, and sweep output indexes hands with these
//! two functions; `features` and `belief_state` re-export them for locality.
//!
//! History (why this module exists): the crate once carried *two different*
//! bijections — this lower-triangular one in `features` and a first-card-major
//! one in `belief_state`.  Seeding a `features`-ordered reach vector in
//! `belief_state` order misindexed every opponent hand and converged CFR to a
//! stable **wrong** fixed point (silent — no panic, plausible-looking output).
//! One implementation makes that class of bug unrepresentable.
//!
//! The ordering is lower-triangular, high-card-major: `index = hi·(hi−1)/2 + lo`
//! with `lo < hi`.  The equity sweeps (`board_equities` and friends) iterate
//! hands in exactly this order in their hot loops, which is why it — and not
//! the retired first-card-major form — is the canonical one.

/// Number of distinct two-card combinations: `C(52, 2)`.
pub const NUM_COMBOS: usize = 1326;

/// Lower-triangular index of a hole pair `{a, b}` over the 52 cards into
/// `0..NUM_COMBOS` (order-independent).
#[inline]
pub fn combo_index(a: u8, b: u8) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    (hi as usize) * (hi as usize - 1) / 2 + lo as usize
}

/// Inverse of [`combo_index`]: the `[lo, hi]` cards (`lo < hi`) for a combo
/// index in `0..NUM_COMBOS`.
#[inline]
pub fn combo_cards(index: usize) -> [u8; 2] {
    let mut hi = 1usize;
    while (hi + 1) * hi / 2 <= index {
        hi += 1;
    }
    let lo = index - hi * (hi - 1) / 2;
    [lo as u8, hi as u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full bijection property: every pair round-trips, order of the input
    /// cards is irrelevant, and the 1326 indices are exactly covered.
    #[test]
    fn combo_index_and_cards_round_trip() {
        let mut seen = vec![false; NUM_COMBOS];
        for a in 0u8..52 {
            for b in (a + 1)..52 {
                let i = combo_index(a, b);
                assert_eq!(combo_cards(i), [a, b], "round trip at ({a},{b})");
                assert_eq!(combo_index(b, a), i, "order-independent");
                assert!(!seen[i], "index {i} hit twice");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "all {NUM_COMBOS} indices covered");
    }

    /// Pin the ordering itself (not just the bijection property): the sweeps'
    /// inner loops depend on lower-triangular hi-major order, so a reordering
    /// that still round-trips must fail here.
    #[test]
    fn ordering_is_lower_triangular_hi_major() {
        assert_eq!(combo_index(0, 1), 0);
        assert_eq!(combo_index(0, 2), 1);
        assert_eq!(combo_index(1, 2), 2);
        assert_eq!(combo_index(0, 3), 3);
        assert_eq!(combo_index(50, 51), NUM_COMBOS - 1);
    }
}
