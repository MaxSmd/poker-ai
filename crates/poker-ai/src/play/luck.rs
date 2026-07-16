//! Luck-adjusted match scoring — an AIVAT-style chance-node control variate
//! (Burch et al. 2018, restricted to chance events) computed per hand once
//! both hole cards are known (Slumbot reveals its cards after every hand).
//!
//! At each board reveal the pot is fixed and the reveal is uniform over the
//! unseen cards, so with the *check-down value* `v = (equity − ½) · pot` as
//! the value function, the term
//!
//! ```text
//!   c_street = pot_at_reveal · (equity_after_reveal − equity_before_reveal)
//! ```
//!
//! has exactly zero mean given everything that happened before the reveal
//! (the tower property of exact enumerated equity: the mean of the post-card
//! equity over all possible cards IS the pre-card equity).  Subtracting the
//! summed corrections from the raw winnings is therefore **unbiased** for the
//! true expected winnings, while cancelling run-out luck in proportion to the
//! pot it swung — the dominant variance source (all-ins, big-pot rivers).
//!
//! For a hand that got all-in on street `s` and checked down, the telescoped
//! adjustment reduces raw winnings to `pot · (equity_at_s − ½)` — exact
//! all-in equity cashout — which is the sanity anchor the tests pin down.
//!
//! What this deliberately does NOT correct: the luck of the *deal* itself
//! (that needs a value function approximating `E[winnings | holes]`; the
//! ½-pot check-down value at a 1.5 bb pot removes almost nothing) and the
//! opponent's action luck (needs a model of their policy).  Corrections are
//! only computable when the opponent's cards are known; for hands without
//! them, use a zero correction — that stays unbiased.

use crate::abstraction::features::hand_vs_hand_equity;
use crate::play::protocol::{parse_action, Parsed};

/// Board-prefix length that street `s` (1 = flop, 2 = turn, 3 = river) reveals.
fn prefix_len(street: u8) -> usize {
    match street {
        1 => 3,
        2 => 4,
        _ => 5,
    }
}

/// Total pot at the moment street `s`'s cards were revealed: commitments never
/// change between streets, so it is the pot before the first action on any
/// street ≥ `s` — or the final pot if the hand had no further actions
/// (all-in run-out, empty trailing streets).
fn pot_at_reveal(parsed: &Parsed, street: u8) -> u32 {
    parsed
        .events
        .iter()
        .find(|e| e.street >= street)
        .map(|e| e.pot_before)
        .unwrap_or_else(|| parsed.pot())
}

/// The summed chance-node corrections for one hand, in chips, from OUR
/// perspective: positive = the run-outs favoured us.  `adjusted winnings =
/// raw winnings − luck`.  `board` is the final board actually revealed
/// (0–5 cards); `action` is the full Slumbot action string.
pub fn luck_adjustment(
    our_hole: [u8; 2],
    opp_hole: [u8; 2],
    board: &[u8],
    action: &str,
) -> Result<f64, String> {
    let parsed = parse_action(action)?;
    let mut luck = 0.0;
    let mut eq_before = hand_vs_hand_equity(our_hole, opp_hole, &[]);
    for street in 1..=3u8 {
        let n = prefix_len(street);
        if board.len() < n {
            break;
        }
        let eq_after = hand_vs_hand_equity(our_hole, opp_hole, &board[..n]);
        luck += pot_at_reveal(&parsed, street) as f64 * (eq_after - eq_before);
        eq_before = eq_after;
    }
    Ok(luck)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play::protocol::STACK_SIZE;

    // Engine encoding: card = rank * 4 + suit, rank 0 = deuce … 12 = ace.
    fn c(rank: u8, suit: u8) -> u8 {
        rank * 4 + suit
    }

    #[test]
    fn preflop_fold_has_zero_luck() {
        let luck = luck_adjustment([c(12, 0), c(12, 1)], [c(0, 2), c(5, 3)], &[], "f").unwrap();
        assert_eq!(luck, 0.0);
    }

    /// Preflop all-in + call: the whole board is luck.  The adjustment must
    /// telescope so that `raw − luck = pot · (preflop equity − ½)` — exact
    /// all-in equity cashout — for EVERY run-out.
    #[test]
    fn preflop_all_in_adjusts_to_equity_cashout() {
        let ours = [c(12, 0), c(12, 1)]; // AA
        let theirs = [c(11, 2), c(11, 3)]; // KK
        let pot = (2 * STACK_SIZE) as f64;
        let eq_pre = hand_vs_hand_equity(ours, theirs, &[]);

        // A run-out we win and one we lose.
        let win_board = [c(2, 0), c(7, 1), c(9, 2), c(3, 3), c(4, 0)];
        let lose_board = [c(11, 0), c(7, 1), c(9, 2), c(3, 3), c(4, 0)];
        for (board, raw) in [(win_board, pot / 2.0), (lose_board, -pot / 2.0)] {
            let luck = luck_adjustment(ours, theirs, &board, "b20000c").unwrap();
            let adjusted = raw - luck;
            let cashout = pot * (eq_pre - 0.5);
            assert!(
                (adjusted - cashout).abs() < 1e-6,
                "adjusted {adjusted} != cashout {cashout}"
            );
        }
    }

    /// The correction at each reveal uses the pot at that moment, so betting
    /// AFTER a reveal must scale later corrections but not earlier ones.
    #[test]
    fn corrections_use_pot_at_reveal_time() {
        let ours = [c(12, 0), c(12, 1)];
        let theirs = [c(4, 2), c(5, 3)];
        let board = [c(2, 0), c(7, 1), c(9, 2), c(3, 3), c(8, 0)];
        // Limped pot (200) sees the flop; flop bet 300 called → turn pot 800;
        // turn checked through; river checked through.
        let action = "ck/b300c/kk/kk";
        let eq0 = hand_vs_hand_equity(ours, theirs, &[]);
        let eq_f = hand_vs_hand_equity(ours, theirs, &board[..3]);
        let eq_t = hand_vs_hand_equity(ours, theirs, &board[..4]);
        let eq_r = hand_vs_hand_equity(ours, theirs, &board);
        let expect = 200.0 * (eq_f - eq0) + 800.0 * (eq_t - eq_f) + 800.0 * (eq_r - eq_t);
        let luck = luck_adjustment(ours, theirs, &board, action).unwrap();
        assert!((luck - expect).abs() < 1e-9, "luck {luck} != {expect}");
    }

    /// Zero-mean check for one reveal: the turn correction summed over every
    /// possible turn card is exactly zero (the tower property the estimator's
    /// unbiasedness rests on).
    #[test]
    fn turn_correction_is_zero_mean_over_all_cards() {
        let ours = [c(12, 0), c(11, 1)];
        let theirs = [c(9, 2), c(9, 3)];
        let flop = [c(2, 0), c(7, 1), c(12, 2)];
        let eq_flop = hand_vs_hand_equity(ours, theirs, &flop);
        let mut used = 0u64;
        for &card in ours.iter().chain(theirs.iter()).chain(flop.iter()) {
            used |= 1 << card;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for card in 0u8..52 {
            if used & (1 << card) != 0 {
                continue;
            }
            let board = [flop[0], flop[1], flop[2], card];
            sum += hand_vs_hand_equity(ours, theirs, &board) - eq_flop;
            count += 1;
        }
        assert_eq!(count, 45);
        assert!(sum.abs() < 1e-9, "mean turn correction {} != 0", sum / 45.0);
    }
}
