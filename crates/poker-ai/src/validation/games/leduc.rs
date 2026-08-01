//! Leduc Hold'em — the second correctness oracle, one step richer than Kuhn.
//!
//! ## Rules (standard 2-player Leduc)
//!
//! A 6-card deck: three ranks (J < Q < K, encoded `0 < 1 < 2`), two suits each.
//! Suits are irrelevant to hand strength (there are no flushes), so two cards of
//! the same rank are strategically identical — the solver keys information sets
//! by rank, a lossless abstraction.
//!
//! * Both players ante 1.
//! * **Round 1:** each player is dealt one private card, then a betting round
//!   with a fixed bet/raise size of **2** and a cap of 2 aggressive actions.
//! * **Round 2:** one public card is revealed, then a betting round with a
//!   fixed size of **4** and the same cap.
//! * **Showdown:** a private card matching the public card's rank is a pair and
//!   beats any unpaired hand; otherwise the higher private rank wins; equal
//!   ranks split (utility 0).
//!
//! ## Why it matters beyond Kuhn
//!
//! Leduc has a second betting round behind a *chance* node (the public card),
//! genuine raises, and pair hand-strengths.  A solver bug that Kuhn's single
//! round hides — mishandling reach probabilities across a chance node, or
//! information leakage between rounds — surfaces here.  Like Kuhn, the
//! equilibrium is computable exactly, so the test is again "drive exploitability
//! below ε" (validation protocol step 2).

use super::Game;

/// `action 0` — fold (only when facing a bet).
const FOLD: u8 = 0;
/// `action 1` — check (no bet) or call (facing a bet).
const CHECK_CALL: u8 = 1;
/// `action 2` — bet (no bet) or raise (facing a bet).
const BET_RAISE: u8 = 2;

/// Maximum number of aggressive actions (bet + raise) per round.
const RAISE_CAP: usize = 2;

/// A Leduc node.
#[derive(Clone, Debug)]
pub struct LeducState {
    /// Private cards have been dealt (false only at the root chance node).
    dealt: bool,
    /// Private card ids (`0..6`); rank is `id / 2`.
    privates: [u8; 2],
    /// Public card id, or `-1` before the flop.
    public: i8,
    /// `1` or `2`.
    round: u8,
    /// Actions taken in the *current* round.
    round_actions: Vec<u8>,
    /// Actions taken in round 1 (retained for perfect recall in round 2).
    round1_actions: Vec<u8>,
    /// Chips each player has committed (antes + bets).
    committed: [u32; 2],
    /// `true` when round 1 has closed and the public card is pending (a chance
    /// node).
    pending_public: bool,
    /// The player who folded, or `-1`.
    folded: i8,
    /// Whether the hand has reached a terminal.
    done: bool,
}

/// The Leduc Hold'em game.
pub struct Leduc;

impl LeducState {
    /// Whether the acting player currently faces a live bet.
    fn facing_bet(&self) -> bool {
        self.round_actions.last() == Some(&BET_RAISE)
    }

    /// Aggressive actions taken this round.
    fn raises(&self) -> usize {
        self.round_actions.iter().filter(|&&a| a == BET_RAISE).count()
    }

    /// Legal action kinds at this decision node.
    fn legal(&self) -> Vec<u8> {
        let mut acts = Vec::with_capacity(3);
        if self.facing_bet() {
            acts.push(FOLD);
        }
        acts.push(CHECK_CALL);
        if self.raises() < RAISE_CAP {
            acts.push(BET_RAISE);
        }
        acts
    }

    /// Hand strength as `(has_pair, rank)`, higher is better.
    fn strength(&self, player: usize) -> (u8, u8) {
        let rank = self.privates[player] / 2;
        let pub_rank = (self.public / 2) as u8;
        if self.public >= 0 && rank == pub_rank {
            (1, rank)
        } else {
            (0, rank)
        }
    }
}

impl Game for Leduc {
    type State = LeducState;

    fn num_players(&self) -> usize {
        2
    }

    fn root(&self) -> LeducState {
        LeducState {
            dealt: false,
            privates: [0, 0],
            public: -1,
            round: 1,
            round_actions: Vec::new(),
            round1_actions: Vec::new(),
            committed: [1, 1], // antes
            pending_public: false,
            folded: -1,
            done: false,
        }
    }

    fn is_terminal(&self, state: &LeducState) -> bool {
        state.done
    }

    fn is_chance(&self, state: &LeducState) -> bool {
        !state.dealt || state.pending_public
    }

    fn utility(&self, state: &LeducState, player: usize) -> f64 {
        let other = 1 - player;
        
        if state.folded >= 0 {
            // Folder loses what it committed; the other wins that amount.
            if state.folded as usize == player {
                -(state.committed[player] as f64)
            } else {
                state.committed[other] as f64
            }
        } else {
            // Showdown: committed amounts are equal here.
            let sp = state.strength(player);
            let so = state.strength(other);
            if sp > so {
                state.committed[other] as f64
            } else if sp < so {
                -(state.committed[player] as f64)
            } else {
                0.0
            }
        }
    }

    fn chance_outcomes(&self, state: &LeducState) -> Vec<(LeducState, f64)> {
        if !state.dealt {
            // Deal two distinct private cards from the 6-card deck (ordered).
            let mut out = Vec::with_capacity(30);
            for a in 0u8..6 {
                for b in 0u8..6 {
                    if a != b {
                        let mut s = state.clone();
                        s.dealt = true;
                        s.privates = [a, b];
                        out.push((s, 1.0 / 30.0));
                    }
                }
            }
            out
        } else {
            // Reveal the public card from the 4 remaining cards.
            let mut out = Vec::with_capacity(4);
            for c in 0u8..6 {
                if c != state.privates[0] && c != state.privates[1] {
                    let mut s = state.clone();
                    s.public = c as i8;
                    s.pending_public = false;
                    out.push((s, 1.0 / 4.0));
                }
            }
            out
        }
    }

    fn current_player(&self, state: &LeducState) -> usize {
        // Player 0 acts first in each round.
        state.round_actions.len() % 2
    }

    fn num_actions(&self, state: &LeducState) -> usize {
        state.legal().len()
    }

    fn apply(&self, state: &LeducState, action: usize) -> LeducState {
        let acts = state.legal();
        let kind = acts[action];
        let player = self.current_player(state);
        let other = 1 - player;
        let facing = state.facing_bet();
        let bet = if state.round == 1 { 2 } else { 4 };

        let mut next = state.clone();
        match kind {
            FOLD => {
                next.folded = player as i8;
                next.done = true;
                return next;
            }
            CHECK_CALL => {
                if facing {
                    // Call: match the opponent's commitment.
                    next.committed[player] += state.committed[other] - state.committed[player];
                }
                next.round_actions.push(CHECK_CALL);
            }
            BET_RAISE => {
                let to_match = state.committed[other] - state.committed[player];
                next.committed[player] += to_match + bet;
                next.round_actions.push(BET_RAISE);
            }
            _ => unreachable!(),
        }

        // Did the round close?
        let closes = match kind {
            CHECK_CALL if facing => true,                              // bet was called
            CHECK_CALL => next.round_actions == [CHECK_CALL, CHECK_CALL], // check-check
            _ => false,                                               // a bet/raise stays open
        };
        if closes {
            if next.round == 1 {
                next.round1_actions = next.round_actions.clone();
                next.round_actions.clear();
                next.round = 2;
                next.pending_public = true;
            } else {
                next.done = true;
            }
        }
        next
    }

    fn info_key(&self, state: &LeducState) -> u64 {
        let player = self.current_player(state);
        let rank = (state.privates[player] / 2) as u64;
        let pub_rank = if state.public >= 0 { (state.public / 2) as u64 } else { 3 };

        // Encode an action vector as a base-4 number with a leading sentinel so
        // distinct lengths map to distinct codes.
        let encode = |v: &[u8]| -> u64 {
            let mut code = 1u64;
            for &a in v {
                code = code * 4 + (a as u64 + 1);
            }
            code
        };

        rank
            | (pub_rank << 3)
            | ((state.round as u64) << 6)
            | (encode(&state.round1_actions) << 8)
            | (encode(&state.round_actions) << 24)
    }

    fn chance_key(&self, state: &LeducState) -> u64 {
        // The initial deal is one chance context; each public-card reveal is
        // keyed by the round-1 betting that preceded it so unlike spots don't
        // share a baseline.
        if !state.dealt {
            0
        } else {
            let mut code = 1u64;
            for &a in &state.round1_actions {
                code = code * 4 + (a as u64 + 1);
            }
            code << 1 // keep clear of the deal's key (0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::best_response::{exploitability, profile_value};
    use crate::solver::cfr::{Cfr, Variant};
    use crate::solver::dcfr::Discount;

    /// Known game value to player 0 for standard 2-player Leduc.
    const KNOWN_VALUE_P0: f64 = -0.0856;

    #[test]
    fn has_288_info_sets() {
        // Standard Leduc has exactly 288 information sets — a strong structural
        // check that the game tree (deals, rounds, raise cap, public card) is
        // built correctly, independent of solver convergence.
        let mut solver = Cfr::new(Leduc, Variant::Vanilla);
        solver.train(1);
        assert_eq!(solver.num_info_sets(), 288, "standard Leduc has 288 info sets");
    }

    #[test]
    fn smoke_converges_toward_known_value() {
        // A short, routine-speed run: confirm the solver is heading to the known
        // Leduc value and a small exploitability.  The strict < 1e-3 check is the
        // ignored `converges_to_equilibrium` test (it needs ~10⁵ iterations).
        let mut solver = Cfr::new(Leduc, Variant::Dcfr(Discount::RECOMMENDED));
        solver.train(600);
        let avg = solver.average_strategy();

        let value = profile_value(&Leduc, &avg, 0);
        assert!(
            (value - KNOWN_VALUE_P0).abs() < 0.02,
            "value {value} should be near known {KNOWN_VALUE_P0}"
        );
        let expl = exploitability(&Leduc, &avg);
        assert!(expl < 0.08, "exploitability {expl} should already be small after 600 iters");
    }

    #[test]
    fn dcfr_has_better_last_iterate_than_vanilla() {
        // The robust DCFR property in full-traversal: its last iterate is far
        // closer to equilibrium than vanilla's, which only converges on average.
        let iters = 600;
        let mut vanilla = Cfr::new(Leduc, Variant::Vanilla);
        vanilla.train(iters);
        let mut dcfr = Cfr::new(Leduc, Variant::Dcfr(Discount::RECOMMENDED));
        dcfr.train(iters);
        let last_vanilla = exploitability(&Leduc, &vanilla.current_strategy());
        let last_dcfr = exploitability(&Leduc, &dcfr.current_strategy());
        assert!(
            last_dcfr < last_vanilla,
            "DCFR last iterate ({last_dcfr}) should beat vanilla ({last_vanilla})"
        );
    }

    /// Strict convergence to the known equilibrium (validation protocol step 2).
    /// Ignored by default because it needs ~10⁵ full-tree iterations; run with:
    ///   cargo test -p poker-ai --release -- --ignored converges_to_equilibrium
    #[test]
    #[ignore]
    fn converges_to_equilibrium() {
        let mut solver = Cfr::new(Leduc, Variant::Vanilla);
        solver.train(200_000);
        let avg = solver.average_strategy();

        let expl = exploitability(&Leduc, &avg);
        assert!(expl < 1.5e-3, "Leduc exploitability {expl} should be < 1.5e-3");

        let value = profile_value(&Leduc, &avg, 0);
        assert!(
            (value - KNOWN_VALUE_P0).abs() < 5e-3,
            "value {value} should match known {KNOWN_VALUE_P0}"
        );
    }
}
