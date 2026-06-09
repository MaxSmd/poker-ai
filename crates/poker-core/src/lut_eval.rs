//! Lookup-table (LUT) based hand evaluator.
//!
//! Two precomputed tables are embedded at compile time via the build script
//! (`build.rs`):
//!
//! * **`FLUSH_LUT`** – `[u32; 8192]` indexed by a 13-bit rank bitmask.
//!   Every entry whose popcount equals 5 holds the hand rank for the
//!   corresponding flush (or straight-flush) hand.  Other entries are zero.
//!
//! * **`NOFLUSH_LUT`** – `[(u32, u32); 16384]` open-addressed hash table
//!   keyed by the product of the five rank primes (à la Cactus Kev).
//!   An entry whose first field is 0 is empty.
//!
//! `evaluate_5_lut` is a near-drop-in replacement for `evaluate_5` and
//! returns bit-for-bit identical results while being significantly faster:
//! flush classification becomes a single array lookup; non-flush
//! classification requires only a prime-product multiply and 1–2 probes.
//!
//! `evaluate_7_lut` (like `evaluate_6_lut`) uses a single-pass frequency and
//! per-suit bitmask approach — no C(7,5) = 21 subset enumeration needed.

use crate::evaluator::{best_non_flush_rank, find_best_straight, make_hand, rank_of, suit_of};

// ── Generated tables (written to $OUT_DIR/lut_tables.rs by build.rs) ─────────

include!(concat!(env!("OUT_DIR"), "/lut_tables.rs"));

// ── Constants (must match build.rs) ──────────────────────────────────────────

const NOFLUSH_SIZE: usize = 16384;
const NOFLUSH_MASK: usize = NOFLUSH_SIZE - 1;
const HASH_MUL: u32 = 2_654_435_761;

/// Primes for ranks 0..=12 (two through ace).
const RANK_PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

// Compile-time assertions: verify that the generated tables match the
// constants declared here.  A mismatch would cause silent wrong evaluations.
const _: () = assert!(FLUSH_LUT.len() == 8192, "FLUSH_LUT size mismatch with build.rs");
const _: () = assert!(NOFLUSH_LUT.len() == NOFLUSH_SIZE, "NOFLUSH_LUT size mismatch with build.rs");

// ── Core lookup helpers ───────────────────────────────────────────────────────

/// Look up the hand rank for a flush (or straight-flush) hand described by a
/// 13-bit rank bitmask.  Callers must ensure `popcount(rank_bits) == 5` and
/// that all five cards share the same suit.
#[inline(always)]
fn flush_rank(rank_bits: u16) -> u32 {
    FLUSH_LUT[rank_bits as usize]
}

/// Return a new bitmask containing only the top 5 set bits of `bits`.
/// Used to reduce a 6- or 7-card suited hand to exactly 5 ranks before
/// indexing `FLUSH_LUT`, which only has valid entries for 5-bit masks.
///
/// Clears the lowest set bit repeatedly until exactly 5 remain, keeping
/// the highest 5 ranks.  Runs at most 2 iterations for 6- or 7-card hands.
#[inline(always)]
fn top5_bits(mut bits: u16) -> u16 {
    while bits.count_ones() > 5 {
        bits &= bits - 1; // clear lowest set bit
    }
    bits
}

/// Look up the hand rank for a non-flush hand described by the product of its
/// five rank primes.  Uses linear probing; expected probe count < 2.
///
/// Panics if the product is not found after a full table scan, which indicates
/// either a corrupt LUT (build script bug) or an invalid input product.
#[inline(always)]
fn noflush_rank(product: u32) -> u32 {
    let mut idx = (product.wrapping_mul(HASH_MUL) as usize) & NOFLUSH_MASK;
    for _ in 0..NOFLUSH_SIZE {
        let (key, val) = NOFLUSH_LUT[idx];
        if key == product {
            return val;
        }
        idx = (idx + 1) & NOFLUSH_MASK;
    }
    panic!("noflush_rank: product {product} not found — corrupt LUT or invalid non-flush hand");
}

// ── Public evaluators ─────────────────────────────────────────────────────────

/// Evaluate a 5-card hand using precomputed lookup tables.
///
/// Returns the same `u32` rank encoding as [`crate::evaluate_5`] —
/// higher is better, category in bits 23-20, tiebreaker nibbles below.
///
/// **Note on invalid inputs:** this function assumes a valid 5-card hand from
/// a standard deck (all cards distinct).  If the 5 cards share a suit but have
/// duplicate ranks (impossible in a real deck), `rank_bits` will have fewer
/// than 5 bits set, `FLUSH_LUT` will return 0, and the non-flush path is
/// taken — the result is then undefined.  Real-deck hands are always handled
/// correctly.
#[inline]
pub fn evaluate_5_lut(cards: &[u8; 5]) -> u32 {
    let mut rank_bits = 0u16;
    let first_suit = suit_of(cards[0]);
    let mut is_flush = true;
    let mut product = 1u32;

    for &c in cards {
        let r = rank_of(c) as usize;
        rank_bits |= 1u16 << r;
        product = product.wrapping_mul(RANK_PRIMES[r]);
        if suit_of(c) != first_suit {
            is_flush = false;
        }
    }

    if is_flush {
        flush_rank(rank_bits)
    } else {
        noflush_rank(product)
    }
}

/// Evaluate the best 5-card hand from 6 cards using lookup tables.
///
/// Uses a single pass to build rank-frequency and per-suit rank-bitmask tables,
/// then uses [`FLUSH_LUT`] for flush/straight-flush detection and the existing
/// frequency-table logic for non-flush hands.
#[inline]
pub fn evaluate_6_lut(cards: &[u8; 6]) -> u32 {
    let mut freq = [0u8; 13];
    let mut suit_rank_bits = [0u16; 4];
    let mut suit_count = [0u8; 4];

    for &c in cards {
        let r = rank_of(c) as usize;
        let s = suit_of(c) as usize;
        freq[r] += 1;
        suit_count[s] += 1;
        suit_rank_bits[s] |= 1u16 << r;
    }

    evaluate_from_tables_lut(&freq, &suit_rank_bits, &suit_count)
}

/// Evaluate the best 5-card hand from 7 cards using lookup tables.
///
/// Uses a single pass to build rank-frequency and per-suit rank-bitmask tables,
/// then uses [`FLUSH_LUT`] for flush/straight-flush detection and the existing
/// frequency-table logic for non-flush hands.
#[inline]
pub fn evaluate_7_lut(cards: &[u8; 7]) -> u32 {
    let mut freq = [0u8; 13];
    let mut suit_rank_bits = [0u16; 4];
    let mut suit_count = [0u8; 4];

    for &c in cards {
        let r = rank_of(c) as usize;
        let s = suit_of(c) as usize;
        freq[r] += 1;
        suit_count[s] += 1;
        suit_rank_bits[s] |= 1u16 << r;
    }

    evaluate_from_tables_lut(&freq, &suit_rank_bits, &suit_count)
}

/// Shared core for [`evaluate_6_lut`] and [`evaluate_7_lut`]: derives the best
/// hand from pre-built frequency and per-suit rank-bitmask tables, using
/// [`FLUSH_LUT`] for flush and straight-flush classification.  Non-flush
/// classification is delegated to [`best_non_flush_rank`] from `evaluator.rs`
/// so the logic lives in exactly one place.
fn evaluate_from_tables_lut(
    freq: &[u8; 13],
    suit_rank_bits: &[u16; 4],
    suit_count: &[u8; 4],
) -> u32 {
    // Overall rank bitmask — needed by best_non_flush_rank for straight check.
    let mut rank_bits = 0u16;
    for (r, &f) in freq.iter().enumerate() {
        if f > 0 { rank_bits |= 1u16 << r; }
    }

    // ── flush / straight-flush via FLUSH_LUT ─────────────────────────────────
    // Check SF on the full suited bitmask (catches wheel SFs where A isn't
    // among the top 5 ranks), then reduce to the top 5 for a plain flush.
    let mut best_flush = 0u32;
    for s in 0..4 {
        if suit_count[s] >= 5 {
            let bits = suit_rank_bits[s];
            let (is_sf, sf_top) = find_best_straight(bits);
            if is_sf {
                return make_hand(8, sf_top, 0, 0, 0, 0);
            }
            let fv = flush_rank(top5_bits(bits));
            if fv > best_flush { best_flush = fv; }
        }
    }

    best_non_flush_rank(freq, rank_bits).max(best_flush)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{evaluate_5, evaluate_6, evaluate_7, make_card};

    fn c(rank: u8, suit: u8) -> u8 {
        make_card(rank, suit)
    }

    // Verify that the LUT evaluator agrees with the reference evaluator on a
    // wide variety of hands.

    #[test]
    fn lut_matches_reference_high_card() {
        let hand = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(7, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_pair() {
        let hand = [c(12, 0), c(12, 1), c(11, 2), c(10, 3), c(9, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_two_pair() {
        let hand = [c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(10, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_trips() {
        let hand = [c(7, 0), c(7, 1), c(7, 2), c(11, 3), c(10, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_straight() {
        let hand = [c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_wheel() {
        let hand = [c(12, 0), c(0, 1), c(1, 2), c(2, 3), c(3, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_flush() {
        let hand = [c(12, 2), c(11, 2), c(10, 2), c(9, 2), c(7, 2)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_full_house() {
        let hand = [c(11, 0), c(11, 1), c(11, 2), c(0, 0), c(0, 1)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_quads() {
        let hand = [c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(11, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_straight_flush() {
        let hand = [c(8, 3), c(9, 3), c(10, 3), c(11, 3), c(12, 3)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_matches_reference_royal_flush() {
        let hand = [c(12, 0), c(11, 0), c(10, 0), c(9, 0), c(8, 0)];
        assert_eq!(evaluate_5_lut(&hand), evaluate_5(&hand));
    }

    #[test]
    fn lut_ordering_preserved() {
        // Confirm relative ordering matches reference across hand categories.
        let hc = evaluate_5_lut(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(7, 0)]);
        let pair = evaluate_5_lut(&[c(12, 0), c(12, 1), c(11, 2), c(10, 3), c(9, 0)]);
        let two_pair = evaluate_5_lut(&[c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(10, 0)]);
        let trips = evaluate_5_lut(&[c(7, 0), c(7, 1), c(7, 2), c(11, 3), c(10, 0)]);
        let straight = evaluate_5_lut(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(8, 0)]);
        let flush = evaluate_5_lut(&[c(12, 2), c(11, 2), c(10, 2), c(9, 2), c(7, 2)]);
        let fh = evaluate_5_lut(&[c(11, 0), c(11, 1), c(11, 2), c(0, 0), c(0, 1)]);
        let quads = evaluate_5_lut(&[c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(11, 0)]);
        let sf = evaluate_5_lut(&[c(8, 3), c(9, 3), c(10, 3), c(11, 3), c(12, 3)]);

        assert!(hc < pair, "high-card < pair");
        assert!(pair < two_pair, "pair < two-pair");
        assert!(two_pair < trips, "two-pair < trips");
        assert!(trips < straight, "trips < straight");
        assert!(straight < flush, "straight < flush");
        assert!(flush < fh, "flush < full-house");
        assert!(fh < quads, "full-house < quads");
        assert!(quads < sf, "quads < straight-flush");
    }

    #[test]
    fn evaluate_6_lut_matches_reference() {
        let cards: [u8; 6] = [
            c(12, 3), c(11, 3), c(10, 3), c(9, 3), c(8, 3),
            c(0, 0),
        ];
        assert_eq!(evaluate_6_lut(&cards), evaluate_6(&cards));
    }

    #[test]
    fn evaluate_7_lut_matches_reference() {
        let cards: [u8; 7] = [
            c(8, 3), c(9, 3), c(10, 3), c(11, 3), c(12, 3),
            c(0, 0), c(1, 1),
        ];
        assert_eq!(evaluate_7_lut(&cards), evaluate_7(&cards));
    }

    #[test]
    fn evaluate_7_lut_quads_beat_flush() {
        let cards: [u8; 7] = [
            c(12, 0), c(12, 1), c(12, 2), c(12, 3),
            c(11, 2), c(10, 2), c(9, 2),
        ];
        assert_eq!(evaluate_7_lut(&cards) >> 20, 7, "expected quads");
    }

    #[test]
    fn evaluate_7_lut_high_card_matches_reference() {
        // 7 cards, no pairs/straights/flushes — exercises the high-card path
        // where rank_bits has 7 bits set.
        let cards: [u8; 7] = [
            c(0, 0),  // 2c
            c(1, 1),  // 3d
            c(3, 2),  // 5h
            c(5, 3),  // 7s
            c(7, 0),  // 9c
            c(9, 1),  // Jd
            c(12, 2), // Ah
        ];
        assert_eq!(
            evaluate_7_lut(&cards),
            evaluate_7(&cards),
            "high-card 7-card LUT mismatch"
        );
    }

    #[test]
    fn evaluate_6_lut_high_card_matches_reference() {
        // 6 cards, no pairs/straights/flushes.
        let cards: [u8; 6] = [
            c(0, 0), c(1, 1), c(3, 2), c(5, 3), c(7, 0), c(12, 1),
        ];
        assert_eq!(evaluate_6_lut(&cards), evaluate_6(&cards));
    }

    // ── 6+ suited cards: flush-detection regression tests ─────────────────────

    #[test]
    fn evaluate_6_lut_six_suited_flush() {
        // 6 hearts — best 5 = A K Q J 10 flush (not straight flush: 10 absent)
        let cards: [u8; 6] = [
            c(12, 2), c(11, 2), c(10, 2), c(9, 2), c(7, 2), c(6, 2),
        ];
        assert_eq!(evaluate_6_lut(&cards), evaluate_6(&cards));
        assert_eq!(evaluate_6_lut(&cards) >> 20, 5, "expected flush");
    }

    #[test]
    fn evaluate_6_lut_six_suited_straight_flush() {
        // 6 spades: A K Q J 10 9 — best 5 = A K Q J 10 straight flush
        let cards: [u8; 6] = [
            c(12, 3), c(11, 3), c(10, 3), c(9, 3), c(8, 3), c(7, 3),
        ];
        assert_eq!(evaluate_6_lut(&cards), evaluate_6(&cards));
        assert_eq!(evaluate_6_lut(&cards) >> 20, 8, "expected straight flush");
    }

    #[test]
    fn evaluate_7_lut_seven_suited_flush() {
        // 7 clubs (ranks 0-6, i.e. 2-3-4-5-6-7-8): best 5 = 4-5-6-7-8 straight flush
        let cards: [u8; 7] = [
            c(0, 0), c(1, 0), c(2, 0), c(3, 0), c(4, 0), c(5, 0), c(6, 0),
        ];
        assert_eq!(evaluate_7_lut(&cards), evaluate_7(&cards));
        assert_eq!(evaluate_7_lut(&cards) >> 20, 8, "expected straight flush");
    }

    #[test]
    fn evaluate_7_lut_six_suited_wheel_straight_flush() {
        // 6 diamonds: A 5 4 3 2 K — wheel straight flush (A-2-3-4-5) should win
        let cards: [u8; 6] = [
            c(12, 1), c(3, 1), c(2, 1), c(1, 1), c(0, 1), c(11, 1),
        ];
        assert_eq!(evaluate_6_lut(&cards), evaluate_6(&cards));
        assert_eq!(evaluate_6_lut(&cards) >> 20, 8, "expected straight flush (wheel)");
    }

    /// Verify that RANK_PRIMES in lut_eval.rs match the build script by
    /// spot-checking known prime products against the NOFLUSH_LUT.
    #[test]
    fn lut_constants_consistent_with_build() {
        // A-K-Q-J-T (ranks 12,11,10,9,8) should have a valid non-flush entry.
        let product: u32 = [12, 11, 10, 9, 8]
            .iter()
            .map(|&r| RANK_PRIMES[r as usize])
            .product();
        let rank = noflush_rank(product);
        // This should be a straight (category 4).
        assert_eq!(rank >> 20, 4, "A-K-Q-J-T should be a straight in NOFLUSH_LUT");

        // 2-2-2-2-3 (four deuces + three) — product = 2^4 * 3 = 48
        let product_quads: u32 = RANK_PRIMES[0].pow(4) * RANK_PRIMES[1];
        let rank_quads = noflush_rank(product_quads);
        assert_eq!(rank_quads >> 20, 7, "four deuces should be quads in NOFLUSH_LUT");
    }

    // ── Exhaustive validation (skipped by default; run with --ignored) ─────────

    /// Verify that `evaluate_5_lut` agrees with `evaluate_5` on all
    /// C(52,5) = 2 598 960 possible 5-card hands drawn from a standard deck.
    ///
    /// Run with:
    ///   cargo test -p poker-core -- --ignored exhaustive_evaluate_5_lut
    #[test]
    #[ignore]
    fn exhaustive_evaluate_5_lut_matches_reference() {
        use crate::evaluator::make_card;
        let deck: Vec<u8> = (0u8..52).map(|i| make_card(i / 4, i % 4)).collect();
        let mut mismatches = 0u64;
        for a in 0..52usize {
            for b in (a + 1)..52 {
                for c in (b + 1)..52 {
                    for d in (c + 1)..52 {
                        for e in (d + 1)..52 {
                            let hand = [deck[a], deck[b], deck[c], deck[d], deck[e]];
                            let lut = evaluate_5_lut(&hand);
                            let ref_ = crate::evaluator::evaluate_5(&hand);
                            if lut != ref_ {
                                mismatches += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(mismatches, 0, "{mismatches} mismatches between evaluate_5_lut and evaluate_5");
    }
}
