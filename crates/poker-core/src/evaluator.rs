//! Hand strength evaluation using a fast bitmask-based evaluator.
//!
//! Cards are encoded as `rank * 4 + suit` where:
//!   - rank: 0 = 2, 1 = 3, …, 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A
//!   - suit: 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades
//!
//! Hand strength is returned as a `u32` where **higher is better**.
//! Encoding:  bits 23-20 = category (0–8), bits 19-0 = tiebreaker ranks.
//! Category codes:
//!   8 = Straight Flush, 7 = Four of a Kind, 6 = Full House, 5 = Flush,
//!   4 = Straight,       3 = Three of a Kind, 2 = Two Pair, 1 = One Pair,
//!   0 = High Card.
//!
//! ## Performance
//! `evaluate_5` avoids sorting by building a 13-element rank-frequency table
//! and a 13-bit rank bitmask in a single pass.  Straight detection uses a
//! precomputed table of the 10 possible 5-card straight patterns, avoiding the
//! sort-then-gap-check approach.
//!
//! `evaluate_7` directly evaluates all 7 cards without calling `evaluate_5`
//! 21 times.  It builds the rank-frequency and per-suit rank-bitmask tables
//! once, then derives the best hand category in O(1) table lookups.  This is
//! significantly faster than the previous C(7,5)=21 subset approach.

/// Precomputed lookup table: the 10 possible 5-card straight patterns.
/// Each entry is `(rank_bitmask, top_rank)`.  The wheel (A-2-3-4-5) is
/// represented with the ace bit set at position 12 and the 2-5 bits at 0-3.
/// The "top rank" for the wheel is 3 (five-high), matching the original
/// `straight_top` logic.
const STRAIGHT_TABLE: [(u16, u8); 10] = [
    (0b1_1111_0000_0000, 12), // A-K-Q-J-T
    (0b0_1111_1000_0000, 11), // K-Q-J-T-9
    (0b0_0111_1100_0000, 10), // Q-J-T-9-8
    (0b0_0011_1110_0000,  9), // J-T-9-8-7
    (0b0_0001_1111_0000,  8), // T-9-8-7-6
    (0b0_0000_1111_1000,  7), // 9-8-7-6-5
    (0b0_0000_0111_1100,  6), // 8-7-6-5-4
    (0b0_0000_0011_1110,  5), // 7-6-5-4-3
    (0b0_0000_0001_1111,  4), // 6-5-4-3-2
    (0b1_0000_0000_1111,  3), // A-5-4-3-2 (wheel, five-high)
];

/// Find the highest straight present in `rank_bits` (a 13-bit mask of ranks).
/// For a 5-card hand pass the exact 5-bit mask; for 6-/7-card hands the mask
/// may have more bits set — the first (highest) matching pattern wins.
/// Returns `(true, top_rank)` or `(false, 0)`.
#[inline]
fn find_best_straight(rank_bits: u16) -> (bool, u8) {
    for (mask, top) in STRAIGHT_TABLE {
        if rank_bits & mask == mask {
            return (true, top);
        }
    }
    (false, 0)
}

/// Collect the top `N` set bits (highest ranks first) from a 13-bit mask.
#[inline]
fn top_n_ranks<const N: usize>(rank_bits: u16) -> [u8; N] {
    let mut out = [0u8; N];
    let mut i = 0;
    let mut rank = 12i8;
    while rank >= 0 && i < N {
        if (rank_bits >> rank) & 1 == 1 {
            out[i] = rank as u8;
            i += 1;
        }
        rank -= 1;
    }
    out
}

/// Extract the rank from a card byte (0 = 2, …, 12 = Ace).
#[inline]
pub fn rank_of(card: u8) -> u8 {
    card >> 2
}

/// Extract the suit from a card byte (0–3).
#[inline]
pub fn suit_of(card: u8) -> u8 {
    card & 3
}

/// Build a card byte from rank and suit.
#[inline]
pub fn make_card(rank: u8, suit: u8) -> u8 {
    (rank << 2) | suit
}

/// Pack category and up-to-5 tiebreaker ranks into a comparable `u32`.
/// `cat` is 0–8; each `rN` is a rank 0–12 stored in a 4-bit nibble.
#[inline]
fn make_hand(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
}

/// Evaluate a 5-card hand.  Returns a rank where higher is better.
///
/// Uses a single-pass rank-frequency table and bitmask straight detection to
/// avoid the sort required by the previous implementation.
pub fn evaluate_5(cards: &[u8; 5]) -> u32 {
    let mut freq = [0u8; 13];
    let mut rank_bits = 0u16;
    let first_suit = suit_of(cards[0]);
    let mut is_flush = true;

    for &c in cards {
        let r = rank_of(c);
        freq[r as usize] += 1;
        rank_bits |= 1u16 << r;
        if suit_of(c) != first_suit {
            is_flush = false;
        }
    }

    // ── frequency-based hand components (scan high→low for tiebreaker order) ──
    let mut quad_rank = 0u8;
    let mut trips_rank = 0u8;
    let mut pair1 = 0u8;
    let mut pair2 = 0u8;
    let mut num_pairs = 0u8;
    let mut has_quads = false;
    let mut has_trips = false;

    for r in (0..13u8).rev() {
        match freq[r as usize] {
            4 => {
                quad_rank = r;
                has_quads = true;
            }
            3 => {
                trips_rank = r;
                has_trips = true;
            }
            2 => {
                if num_pairs == 0 {
                    pair1 = r;
                } else {
                    pair2 = r;
                }
                num_pairs += 1;
            }
            _ => {}
        }
    }

    // ── straight detection via lookup table (no sort needed) ────────────────
    let (is_straight, straight_top) = if rank_bits.count_ones() == 5 {
        find_best_straight(rank_bits)
    } else {
        (false, 0)
    };

    // ── classify ─────────────────────────────────────────────────────────────
    if is_straight && is_flush {
        make_hand(8, straight_top, 0, 0, 0, 0)
    } else if has_quads {
        let kicker = (0..13u8)
            .rev()
            .find(|&r| r != quad_rank && freq[r as usize] > 0)
            .unwrap_or(0);
        make_hand(7, quad_rank, kicker, 0, 0, 0)
    } else if has_trips && num_pairs >= 1 {
        make_hand(6, trips_rank, pair1, 0, 0, 0)
    } else if is_flush {
        let r = top_n_ranks::<5>(rank_bits);
        make_hand(5, r[0], r[1], r[2], r[3], r[4])
    } else if is_straight {
        make_hand(4, straight_top, 0, 0, 0, 0)
    } else if has_trips {
        let mut k = [0u8; 2];
        let mut ki = 0;
        for r in (0..13u8).rev() {
            if freq[r as usize] > 0 && r != trips_rank {
                k[ki] = r;
                ki += 1;
                if ki == 2 {
                    break;
                }
            }
        }
        make_hand(3, trips_rank, k[0], k[1], 0, 0)
    } else if num_pairs >= 2 {
        let kicker = (0..13u8)
            .rev()
            .find(|&r| r != pair1 && r != pair2 && freq[r as usize] > 0)
            .unwrap_or(0);
        make_hand(2, pair1, pair2, kicker, 0, 0)
    } else if num_pairs == 1 {
        let mut k = [0u8; 3];
        let mut ki = 0;
        for r in (0..13u8).rev() {
            if freq[r as usize] > 0 && r != pair1 {
                k[ki] = r;
                ki += 1;
                if ki == 3 {
                    break;
                }
            }
        }
        make_hand(1, pair1, k[0], k[1], k[2], 0)
    } else {
        let r = top_n_ranks::<5>(rank_bits);
        make_hand(0, r[0], r[1], r[2], r[3], r[4])
    }
}

// ── Direct multi-card evaluators ─────────────────────────────────────────────

/// Evaluate the best possible hand from a set of cards given pre-computed
/// rank-frequency and per-suit rank-bit tables.
///
/// This is the core of both `evaluate_6` and `evaluate_7`: build the tables
/// once, then derive the best hand.
///
/// `freq`           – rank frequency histogram (index = rank 0..12)
/// `suit_rank_bits` – for each suit, the bitmask of ranks present in that suit
/// `suit_count`     – number of cards of each suit
fn evaluate_from_tables(
    freq: &[u8; 13],
    suit_rank_bits: &[u16; 4],
    suit_count: &[u8; 4],
) -> u32 {
    // ── frequency-based components ───────────────────────────────────────────
    let mut quad_rank = 0u8;
    let mut trips = [0u8; 2]; // up to 2 trip ranks in 7 cards
    let mut pairs = [0u8; 3]; // up to 3 pair ranks in 7 cards
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

    // Overall rank bitmask (union of all suits).
    let mut rank_bits = 0u16;
    for r in 0..13 {
        if freq[r] > 0 {
            rank_bits |= 1u16 << r;
        }
    }

    // ── flush / straight-flush ───────────────────────────────────────────────
    let mut best_flush_or_sf = 0u32;
    for s in 0..4 {
        if suit_count[s] >= 5 {
            let fb = suit_rank_bits[s];
            let (is_sf, sf_top) = find_best_straight(fb);
            if is_sf {
                // Straight flush beats everything else; return immediately.
                return make_hand(8, sf_top, 0, 0, 0, 0);
            }
            // Regular flush: best 5 cards from this suit.
            let r = top_n_ranks::<5>(fb);
            let fv = make_hand(5, r[0], r[1], r[2], r[3], r[4]);
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
        make_hand(7, quad_rank, kicker, 0, 0, 0)
    } else if num_trips >= 1 && (num_pairs >= 1 || num_trips >= 2) {
        // Full house: best trips + best "pair" (second trips or highest pair).
        let pair_part = if num_trips >= 2 { trips[1] } else { pairs[0] };
        make_hand(6, trips[0], pair_part, 0, 0, 0)
    } else {
        // Straight check.
        let (is_straight, straight_top) = find_best_straight(rank_bits);

        if is_straight {
            // Might still lose to a flush (already computed above).
            make_hand(4, straight_top, 0, 0, 0, 0)
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
            make_hand(3, trips[0], k[0], k[1], 0, 0)
        } else if num_pairs >= 2 {
            let kicker = (0..13u8)
                .rev()
                .find(|&r| r != pairs[0] && r != pairs[1] && freq[r as usize] > 0)
                .unwrap_or(0);
            make_hand(2, pairs[0], pairs[1], kicker, 0, 0)
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
            make_hand(1, pairs[0], k[0], k[1], k[2], 0)
        } else {
            let r = top_n_ranks::<5>(rank_bits);
            make_hand(0, r[0], r[1], r[2], r[3], r[4])
        }
    };

    // A flush (cat 5) beats trips, two-pair, pair, high-card but loses to
    // quads (cat 7) and full house (cat 6).  The `max` correctly selects
    // the winner because the category is encoded in the high bits.
    best_non_flush.max(best_flush_or_sf)
}

/// Evaluate the best 5-card hand from 6 cards (e.g., turn + hole cards).
/// Direct single-pass evaluation — no heap allocation.
pub fn evaluate_6(cards: &[u8; 6]) -> u32 {
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

    evaluate_from_tables(&freq, &suit_rank_bits, &suit_count)
}

/// Evaluate the strength of a 7-card hand. Returns a rank where higher is better.
/// Direct single-pass evaluation — no heap allocation and no repeated subset
/// enumeration.
pub fn evaluate_7(cards: &[u8; 7]) -> u32 {
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

    evaluate_from_tables(&freq, &suit_rank_bits, &suit_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> u8 {
        make_card(rank, suit)
    }

    #[test]
    fn high_card_less_than_pair() {
        // High card: A K Q J 9
        let hc = evaluate_5(&[c(12, 0), c(11, 1), c(10, 2), c(9, 3), c(7, 0)]);
        // One pair: A A K Q J
        let pair = evaluate_5(&[c(12, 0), c(12, 1), c(11, 2), c(10, 3), c(9, 0)]);
        assert!(hc < pair);
    }

    #[test]
    fn straight_flush_beats_quads() {
        // Straight flush 9-T-J-Q-K spades
        let sf = evaluate_5(&[c(7, 3), c(8, 3), c(9, 3), c(10, 3), c(11, 3)]);
        // Four aces + king
        let quads = evaluate_5(&[c(12, 0), c(12, 1), c(12, 2), c(12, 3), c(11, 0)]);
        assert!(sf > quads);
    }

    #[test]
    fn wheel_straight_is_five_high() {
        // A-2-3-4-5 (wheel) – top rank = 3 (five)
        let wheel = evaluate_5(&[c(12, 0), c(0, 1), c(1, 2), c(2, 3), c(3, 0)]);
        // 2-3-4-5-6 straight – top rank = 4 (six)
        let six_high = evaluate_5(&[c(0, 0), c(1, 1), c(2, 2), c(3, 3), c(4, 0)]);
        assert!(wheel < six_high, "wheel {wheel:#x} should be < 6-high {six_high:#x}");
    }

    #[test]
    fn full_house_beats_flush() {
        // Full house: K-K-K-2-2
        let fh = evaluate_5(&[c(11, 0), c(11, 1), c(11, 2), c(0, 0), c(0, 1)]);
        // Flush: A-K-Q-J-9 hearts
        let flush = evaluate_5(&[c(12, 2), c(11, 2), c(10, 2), c(9, 2), c(7, 2)]);
        assert!(fh > flush);
    }

    #[test]
    fn evaluate_7_finds_best_hand() {
        // 7 cards containing a straight flush in spades (T-J-Q-K-A)
        // plus two unrelated cards
        let cards: [u8; 7] = [
            c(8, 3),  // T spades
            c(9, 3),  // J spades
            c(10, 3), // Q spades
            c(11, 3), // K spades
            c(12, 3), // A spades
            c(0, 0),  // 2 clubs
            c(1, 1),  // 3 diamonds
        ];
        let rank = evaluate_7(&cards);
        // Category should be 8 (straight flush)
        assert_eq!(rank >> 20, 8, "expected straight flush category");
    }

    #[test]
    fn two_pair_vs_two_pair_kicker() {
        // A-A-K-K-Q vs A-A-K-K-J — queen kicker beats jack kicker
        let high_kicker = evaluate_5(&[c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(10, 0)]);
        let low_kicker = evaluate_5(&[c(12, 0), c(12, 1), c(11, 0), c(11, 1), c(9, 0)]);
        assert!(high_kicker > low_kicker);
    }

    #[test]
    fn evaluate_7_quads_beat_flush() {
        // 4 aces + 3 hearts (flush) — quads should win
        let cards: [u8; 7] = [
            c(12, 0), c(12, 1), c(12, 2), c(12, 3), // 4 aces
            c(11, 2), c(10, 2), c(9, 2), // K Q J hearts (flush)
        ];
        let rank = evaluate_7(&cards);
        assert_eq!(rank >> 20, 7, "expected quads category");
    }

    #[test]
    fn evaluate_7_two_trip_ranks_full_house() {
        // 3 kings + 3 queens + one unrelated card — best full house = K-K-K/Q-Q
        let cards: [u8; 7] = [
            c(11, 0), c(11, 1), c(11, 2), // 3 kings
            c(10, 0), c(10, 1), c(10, 2), // 3 queens
            c(0, 3),                       // 2 spades (filler)
        ];
        let rank = evaluate_7(&cards);
        assert_eq!(rank >> 20, 6, "expected full house category");
        // trips part = K (11), pair part = Q (10)
        let trips_part = (rank >> 16) & 0xF;
        let pair_part = (rank >> 12) & 0xF;
        assert_eq!(trips_part, 11, "trips should be kings");
        assert_eq!(pair_part, 10, "pair should be queens");
    }

    #[test]
    fn evaluate_6_finds_flush() {
        // 5 spades + 1 unrelated
        let cards: [u8; 6] = [
            c(12, 3), c(11, 3), c(10, 3), c(9, 3), c(8, 3), // A-K-Q-J-T spades (royal SF)
            c(0, 0),                                           // 2 clubs (filler)
        ];
        let rank = evaluate_6(&cards);
        assert_eq!(rank >> 20, 8, "expected straight flush category");
    }
}
