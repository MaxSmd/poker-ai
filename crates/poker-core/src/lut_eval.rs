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

use crate::evaluator::{rank_of, suit_of};

// ── Generated tables (written to $OUT_DIR/lut_tables.rs by build.rs) ─────────

include!(concat!(env!("OUT_DIR"), "/lut_tables.rs"));

// ── Constants (must match build.rs) ──────────────────────────────────────────

const NOFLUSH_SIZE: usize = 16384;
const NOFLUSH_MASK: usize = NOFLUSH_SIZE - 1;
const HASH_MUL: u32 = 2_654_435_761;

/// Primes for ranks 0..=12 (two through ace).
const RANK_PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Mask that keeps only the tiebreaker nibbles (bits 19-0), stripping the
/// 4-bit category field in bits 23-20.
const TIEBREAKER_MASK: u32 = 0x000F_FFFF;

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
#[inline(always)]
fn top5_bits(bits: u16) -> u16 {
    let mut result = 0u16;
    let mut count = 0u8;
    for bit in (0..13).rev() {
        if (bits >> bit) & 1 == 1 {
            result |= 1u16 << bit;
            count += 1;
            if count == 5 {
                break;
            }
        }
    }
    result
}

/// Look up the hand rank for a non-flush hand described by the product of its
/// five rank primes.  Uses linear probing; expected probe count < 2.
#[inline(always)]
fn noflush_rank(product: u32) -> u32 {
    let mut idx = (product.wrapping_mul(HASH_MUL) as usize) & NOFLUSH_MASK;
    loop {
        let (key, val) = NOFLUSH_LUT[idx];
        if key == product {
            return val;
        }
        // key == 0 would indicate a missing entry, which should never happen
        // for a valid 5-card non-flush hand.
        idx = (idx + 1) & NOFLUSH_MASK;
    }
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
/// [`FLUSH_LUT`] for flush and straight-flush classification (a single array
/// lookup replaces the straight-scan + rank-extraction done by the bitmask
/// evaluator).
fn evaluate_from_tables_lut(
    freq: &[u8; 13],
    suit_rank_bits: &[u16; 4],
    suit_count: &[u8; 4],
) -> u32 {
    use crate::evaluator::find_best_straight;

    // ── frequency-based components ───────────────────────────────────────────
    let mut quad_rank = 0u8;
    let mut trips = [0u8; 2];
    let mut pairs = [0u8; 3];
    let mut num_trips = 0usize;
    let mut num_pairs = 0usize;
    let mut has_quads = false;

    for r in (0..13u8).rev() {
        match freq[r as usize] {
            4 => {
                quad_rank = r;
                has_quads = true;
            }
            3 => {
                if num_trips < 2 {
                    trips[num_trips] = r;
                    num_trips += 1;
                }
            }
            2 => {
                if num_pairs < 3 {
                    pairs[num_pairs] = r;
                    num_pairs += 1;
                }
            }
            _ => {}
        }
    }

    let rank_bits: u16 = freq
        .iter()
        .enumerate()
        .filter(|(_, &f)| f > 0)
        .fold(0u16, |bits, (r, _)| bits | (1u16 << r));

    // ── flush / straight-flush via FLUSH_LUT ─────────────────────────────────
    let mut best_flush_or_sf = 0u32;
    for s in 0..4 {
        if suit_count[s] >= 5 {
            // When 6 or 7 cards share a suit, suit_rank_bits[s] has 6/7 bits
            // set, but FLUSH_LUT is only defined for exactly-5-bit masks.
            // Check for straight-flush first using the full bitmask (so that
            // wheel SFs like A-2-3-4-5 are detected even when A is not among
            // the top 5 ranks), then reduce to the top 5 for a plain flush.
            let bits = suit_rank_bits[s];
            let (is_sf, sf_top) = find_best_straight(bits);
            if is_sf {
                // Straight flush — nothing can beat it.
                return make_hand_lut(8, sf_top, 0, 0, 0, 0);
            }
            // Regular flush: look up best 5 suited ranks.
            let fv = flush_rank(top5_bits(bits));
            if fv > best_flush_or_sf {
                best_flush_or_sf = fv;
            }
        }
    }

    // ── non-flush hand ───────────────────────────────────────────────────────
    let best_non_flush = if has_quads {
        let kicker = (0..13u8)
            .rev()
            .find(|&r| r != quad_rank && freq[r as usize] > 0)
            .unwrap_or(0);
        make_hand_lut(7, quad_rank, kicker, 0, 0, 0)
    } else if num_trips >= 1 && (num_pairs >= 1 || num_trips >= 2) {
        let pair_part = if num_trips >= 2 { trips[1] } else { pairs[0] };
        make_hand_lut(6, trips[0], pair_part, 0, 0, 0)
    } else {
        let (is_straight, straight_top) = find_best_straight(rank_bits);
        if is_straight {
            make_hand_lut(4, straight_top, 0, 0, 0, 0)
        } else if num_trips == 1 {
            let mut k = [0u8; 2];
            let mut ki = 0;
            for r in (0..13u8).rev() {
                if freq[r as usize] > 0 && r != trips[0] {
                    k[ki] = r;
                    ki += 1;
                    if ki == 2 {
                        break;
                    }
                }
            }
            make_hand_lut(3, trips[0], k[0], k[1], 0, 0)
        } else if num_pairs >= 2 {
            let kicker = (0..13u8)
                .rev()
                .find(|&r| r != pairs[0] && r != pairs[1] && freq[r as usize] > 0)
                .unwrap_or(0);
            make_hand_lut(2, pairs[0], pairs[1], kicker, 0, 0)
        } else if num_pairs == 1 {
            let mut k = [0u8; 3];
            let mut ki = 0;
            for r in (0..13u8).rev() {
                if freq[r as usize] > 0 && r != pairs[0] {
                    k[ki] = r;
                    ki += 1;
                    if ki == 3 {
                        break;
                    }
                }
            }
            make_hand_lut(1, pairs[0], k[0], k[1], k[2], 0)
        } else {
            // High card: find the top 5 ranks from rank_bits (which may have
            // more than 5 bits set for 6- or 7-card hands).
            // Build a 5-bit bitmask of the top 5 set bits, look up in
            // FLUSH_LUT (category 5 = flush), then strip the category to 0.
            let mut top5 = 0u16;
            let mut count = 0u8;
            for bit in (0..13i32).rev() {
                if (rank_bits >> bit) & 1 == 1 {
                    top5 |= 1u16 << bit;
                    count += 1;
                    if count == 5 {
                        break;
                    }
                }
            }
            flush_rank(top5) & TIEBREAKER_MASK
        }
    };

    best_non_flush.max(best_flush_or_sf)
}

/// Pack category and tiebreaker ranks into a hand value (mirrors `make_hand`
/// in `evaluator.rs`; kept private to this module).
#[inline(always)]
fn make_hand_lut(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
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
