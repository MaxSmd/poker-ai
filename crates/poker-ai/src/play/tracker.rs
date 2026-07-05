//! Dual-state tracking: mirror a **real** hand inside the **abstract** game the
//! blueprint was trained on.
//!
//! The real table (Slumbot: 20 000-chip stacks, arbitrary bet sizes) and the
//! blueprint's abstract game (small integer chips, pot-fraction raise menu,
//! raise cap) drift apart the moment an opponent uses an off-tree size.  The
//! classic remedy is to track both states and translate actions between them
//! in **pot-fraction space**, which is scale-invariant:
//!
//! * an observed real bet maps to one of the abstract aggressive actions by
//!   **randomized pseudo-harmonic mapping**
//!   ([`poker_core::betting::pseudo_harmonic_weight`], Ganzfried & Sandholm
//!   2013) over the bracket of neighbouring abstract sizes;
//! * our own abstract raise translates back to a real amount with the same
//!   pot fraction, clamped to the real game's legal range.
//!
//! Design choice: a real bet is **never mapped to a passive abstract action**.
//! Mapping a tiny bet to "check" desynchronizes the two states (the abstract
//! street closes while the real one stays open), which corrupts every later
//! key in the hand — strictly worse than the slight distortion of rounding the
//! bet up to the smallest abstract size.  The one exception is aggression past
//! the abstract raise cap, where no aggressive action exists: the bet maps to
//! `Call` (exactly how the cap was trained) and the tracker reports the
//! desync-tolerant [`MapOutcome::None`] for any follow-up the abstract game
//! has no node for.

use poker_core::action::{Action, ActionList};
use poker_core::betting::pseudo_harmonic_weight;
use poker_core::state::{GameState, NO_CARD};

use crate::games::blueprint::{BlueprintHoldem, BlueprintState};
use crate::games::Game;
use crate::play::protocol::EventKind;

/// Result of translating one real event into the abstract game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapOutcome {
    /// Apply this abstract action index.
    Index(u8),
    /// The abstract game has no node for this event (post-cap aggression
    /// follow-ups) — skip it; the states re-align at the next street.
    None,
}

/// A real hand mirrored in the abstract blueprint game.
pub struct AbstractHand {
    /// Engine seats: `0` = small blind / button, `1` = big blind.
    holes: [[u8; 2]; 2],
    board: [u8; 5],
    history: Vec<u8>,
    state: BlueprintState,
}

impl AbstractHand {
    /// Start a hand: our engine seat and hole cards are known, the opponent
    /// holds placeholder cards (never read — every key we compute is either for
    /// our own seat or for an explicit hypothetical holding).
    pub fn new(game: &BlueprintHoldem, my_seat: usize, my_hole: [u8; 2]) -> Self {
        let board = [NO_CARD; 5];
        let holes = Self::holes_with_placeholder(my_seat, my_hole, &board);
        let state = game.play_state(holes, board);
        Self { holes, board, history: Vec::new(), state }
    }

    /// Placeholder opponent cards: the two lowest cards not colliding with our
    /// hand or the known board.
    fn holes_with_placeholder(my_seat: usize, my_hole: [u8; 2], board: &[u8; 5]) -> [[u8; 2]; 2] {
        let mut used = 0u64;
        used |= 1 << my_hole[0];
        used |= 1 << my_hole[1];
        for &c in board {
            if c != NO_CARD {
                used |= 1 << c;
            }
        }
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        let opp = [spare.next().unwrap(), spare.next().unwrap()];
        let mut holes = [opp, opp];
        holes[my_seat] = my_hole;
        holes
    }

    /// Reveal board cards (cumulative list, engine encoding) and rebuild the
    /// abstract state by replaying the recorded history.  Also re-picks the
    /// opponent placeholder if a revealed card collided with it.
    pub fn set_board(&mut self, game: &BlueprintHoldem, board_cards: &[u8], my_seat: usize) {
        let mut board = [NO_CARD; 5];
        board[..board_cards.len()].copy_from_slice(board_cards);
        self.board = board;
        self.holes = Self::holes_with_placeholder(my_seat, self.holes[my_seat], &board);
        self.state = game.play_state(self.holes, self.board);
        for i in 0..self.history.len() {
            self.state = game.apply(&self.state, self.history[i] as usize);
        }
    }

    /// Apply an abstract action by index into [`actions`](Self::actions).
    pub fn apply(&mut self, game: &BlueprintHoldem, index: u8) {
        self.state = game.apply(&self.state, index as usize);
        self.history.push(index);
    }

    /// The capped abstract action menu at the current node.
    pub fn actions(&self, game: &BlueprintHoldem) -> ActionList {
        game.actions(&self.state)
    }

    /// The wrapped abstract engine state.
    pub fn gs<'g>(&'g self, game: &'g BlueprintHoldem) -> &'g GameState {
        game.game_state(&self.state)
    }

    /// Info key of the acting player with its dealt cards (only meaningful when
    /// **we** act — the opponent's dealt cards are placeholders).
    pub fn info_key(&self, game: &BlueprintHoldem) -> u64 {
        game.info_key(&self.state)
    }

    /// Info key the acting player would have holding `hole` — the likelihood
    /// primitive for belief updates.
    pub fn key_with_hole(&self, game: &BlueprintHoldem, hole: [u8; 2]) -> u64 {
        game.info_key_with_hole(&self.state, hole)
    }

    /// Whether the abstract hand has reached a terminal.
    pub fn is_terminal(&self, game: &BlueprintHoldem) -> bool {
        game.is_terminal(&self.state)
    }

    /// True when the abstract node expects `seat` to act on `street` — the
    /// desync guard consulted before every translation.
    pub fn expects(&self, game: &BlueprintHoldem, seat: usize, street: u8) -> bool {
        if self.is_terminal(game) {
            return false;
        }
        let gs = self.gs(game);
        gs.current_player() == seat && gs.street == street
    }

    /// Translate one observed **real** action into an abstract action index.
    ///
    /// `real_pot` / `real_bet` are the real pot and outstanding street bet
    /// level *before* the action (any chip unit — only fractions matter);
    /// `unit` draws uniform `[0,1)` samples for the randomized mapping.
    pub fn map_real(
        &self,
        game: &BlueprintHoldem,
        kind: EventKind,
        real_pot: f64,
        real_bet: f64,
        unit: &mut impl FnMut() -> f64,
    ) -> MapOutcome {
        let acts = self.actions(game);
        let position = |pred: &dyn Fn(&Action) -> bool| {
            acts.iter().position(pred).map(|i| MapOutcome::Index(i as u8))
        };
        let passive = || {
            position(&|a| matches!(a, Action::Check))
                .or_else(|| position(&|a| matches!(a, Action::Call | Action::AllIn)))
                .unwrap_or(MapOutcome::None)
        };
        match kind {
            EventKind::Check => passive(),
            // A call that matches an all-in appears as `AllIn` in the menu.
            EventKind::Call => position(&|a| matches!(a, Action::Call | Action::AllIn))
                .or_else(|| position(&|a| matches!(a, Action::Check)))
                .unwrap_or(MapOutcome::None),
            EventKind::Fold => position(&|a| matches!(a, Action::Fold)).unwrap_or(MapOutcome::None),
            EventKind::BetTo(to) => {
                let x = (to as f64 - real_bet) / (real_pot + real_bet);
                self.map_bet_fraction(game, &acts, x, unit)
            }
        }
    }

    /// Pseudo-harmonic mapping of a raise of pot-fraction `x` onto the abstract
    /// aggressive menu.
    fn map_bet_fraction(
        &self,
        game: &BlueprintHoldem,
        acts: &ActionList,
        x: f64,
        unit: &mut impl FnMut() -> f64,
    ) -> MapOutcome {
        let gs = self.gs(game);
        let pot = gs.pot as f64;
        let cb = gs.current_bet as f64;
        let actor = gs.current_player();
        let max_bet = (gs.stacks[actor] + gs.street_bets[actor]) as f64;

        // Aggressive candidates in ascending pot-fraction order (the menu is
        // built ascending: raises small→large, then all-in).
        let mut cands: Vec<(usize, f64)> = Vec::with_capacity(acts.len());
        for (i, &a) in acts.iter().enumerate() {
            let level = match a {
                Action::Raise(l) => l as f64,
                Action::AllIn if max_bet > cb => max_bet,
                _ => continue,
            };
            cands.push((i, (level - cb) / (pot + cb)));
        }
        match cands.as_slice() {
            [] => {
                // Aggression past the raise cap: trained as a call.
                let i = acts.iter().position(|a| matches!(a, Action::Call | Action::AllIn));
                i.map(|i| MapOutcome::Index(i as u8)).unwrap_or(MapOutcome::None)
            }
            [(only, _)] => MapOutcome::Index(*only as u8),
            _ => {
                // Bracket x between neighbouring candidate fractions.
                let hi_pos = cands.iter().position(|&(_, f)| f >= x).unwrap_or(cands.len() - 1);
                let lo_pos = hi_pos.saturating_sub(if cands[hi_pos].1 >= x && hi_pos > 0 { 1 } else { 0 });
                let (lo_i, lo_f) = cands[lo_pos];
                let (hi_i, hi_f) = cands[hi_pos];
                let w = pseudo_harmonic_weight(lo_f, hi_f, x);
                MapOutcome::Index(if unit() < w { lo_i as u8 } else { hi_i as u8 })
            }
        }
    }

    /// Translate one of our abstract actions into a real-game move, expressed
    /// in real chips.  `real_bet` is the outstanding real street bet level,
    /// `real_pot` the real pot, `facing` whether there is an outstanding bet to
    /// call (the protocol's `last_bet_size > 0` — NOT derivable from
    /// `real_bet` alone: the big blind's option after a limp has a bet level
    /// but nothing to call), `real_min_size` the minimum legal raise size, and
    /// `real_remaining` the raise size that puts the actor all-in (Slumbot
    /// semantics: `STACK − total_last_bet_to`).
    #[allow(clippy::too_many_arguments)]
    pub fn abstract_to_real(
        &self,
        game: &BlueprintHoldem,
        action: Action,
        real_pot: f64,
        real_bet: u32,
        facing: bool,
        real_min_size: u32,
        real_remaining: u32,
    ) -> RealMove {
        let facing_bet = facing;
        match action {
            Action::Fold => {
                if facing_bet {
                    RealMove::Fold
                } else {
                    RealMove::Check
                }
            }
            Action::Check => {
                if facing_bet {
                    RealMove::Call
                } else {
                    RealMove::Check
                }
            }
            Action::Call => {
                if facing_bet {
                    RealMove::Call
                } else {
                    RealMove::Check
                }
            }
            Action::Raise(level) => {
                let gs = self.gs(game);
                let frac = (level as f64 - gs.current_bet as f64)
                    / (gs.pot as f64 + gs.current_bet as f64);
                let size = (frac * (real_pot + real_bet as f64)).round() as i64;
                let size = (size.max(real_min_size as i64) as u32).min(real_remaining);
                if size >= real_remaining {
                    RealMove::BetTo(real_bet + real_remaining)
                } else {
                    RealMove::BetTo(real_bet + size)
                }
            }
            Action::AllIn => {
                // The abstract all-in may be a forced call of an all-in bet.
                let gs = self.gs(game);
                let actor = gs.current_player();
                if gs.stacks[actor] + gs.street_bets[actor] <= gs.current_bet {
                    return if facing_bet { RealMove::Call } else { RealMove::Check };
                }
                if real_remaining == 0 {
                    return if facing_bet { RealMove::Call } else { RealMove::Check };
                }
                RealMove::BetTo(real_bet + real_remaining)
            }
        }
    }
}

/// A move in the real game, ready to serialize to the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealMove {
    Fold,
    Check,
    Call,
    /// Bet/raise **to** this street level (real chips).
    BetTo(u32),
}

impl RealMove {
    /// Slumbot wire encoding.
    pub fn to_incr(self) -> String {
        match self {
            RealMove::Fold => "f".into(),
            RealMove::Check => "k".into(),
            RealMove::Call => "c".into(),
            RealMove::BetTo(n) => format!("b{n}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::make_card;

    /// 200 bb abstract game in blueprint chips (bb = 2), raise cap 3, no card
    /// abstraction (raw suit-isomorphic keys — fine for tracking mechanics).
    fn game() -> BlueprintHoldem {
        BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3)
    }

    fn hand(g: &BlueprintHoldem) -> AbstractHand {
        AbstractHand::new(g, 0, [make_card(12, 0), make_card(11, 0)]) // we are SB with AKs
    }

    #[test]
    fn tracks_streets_and_actors_through_a_scripted_hand() {
        let g = game();
        let mut h = hand(&g);
        assert!(h.expects(&g, 0, 0), "SB (engine seat 0) opens preflop");

        // SB raises (some aggressive index), BB calls, flop comes.
        let acts = h.actions(&g);
        let raise = acts.iter().position(|a| matches!(a, Action::Raise(_))).unwrap();
        h.apply(&g, raise as u8);
        assert!(h.expects(&g, 1, 0), "BB to act facing the raise");
        let call = h.actions(&g).iter().position(|a| matches!(a, Action::Call)).unwrap();
        h.apply(&g, call as u8);

        let flop = [make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        h.set_board(&g, &flop, 0);
        assert!(h.expects(&g, 1, 1), "BB first to act on the flop");
        assert!(!h.expects(&g, 0, 1));
        assert!(!h.expects(&g, 1, 0), "street guard rejects stale events");
    }

    #[test]
    fn board_rebuild_preserves_the_replayed_key() {
        // Replaying the same history after a board patch must key identically
        // to a hand built with the full board up front.
        let g = game();
        let mut incremental = hand(&g);
        let acts = incremental.actions(&g);
        let raise = acts.iter().position(|a| matches!(a, Action::Raise(_))).unwrap() as u8;
        incremental.apply(&g, raise);
        let call =
            incremental.actions(&g).iter().position(|a| matches!(a, Action::Call)).unwrap() as u8;
        incremental.apply(&g, call);
        let flop = [make_card(7, 2), make_card(2, 3), make_card(0, 0)];
        incremental.set_board(&g, &flop, 0);

        let mut upfront = hand(&g);
        upfront.set_board(&g, &flop, 0);
        upfront.apply(&g, raise);
        upfront.apply(&g, call);

        assert_eq!(incremental.info_key(&g), upfront.info_key(&g));
        assert_eq!(
            incremental.key_with_hole(&g, [make_card(5, 1), make_card(5, 2)]),
            upfront.key_with_hole(&g, [make_card(5, 1), make_card(5, 2)])
        );
    }

    #[test]
    fn real_bets_map_pseudo_harmonically_never_passively() {
        let g = game();
        let h = hand(&g);
        // Real: SB opens to 250 into blinds 50/100 (slumbot chips).
        // Fraction = (250 − 100) / (150 + 100) = 0.6.
        let acts = h.actions(&g);
        let mut low = || 0.0; // rng draw below any weight → lower bracket side
        let m = h.map_real(&g, EventKind::BetTo(250), 150.0, 100.0, &mut low);
        let MapOutcome::Index(i) = m else { panic!("bet must map") };
        assert!(
            matches!(acts[i as usize], Action::Raise(_)),
            "an opening raise maps to an abstract raise, got {:?}",
            acts[i as usize]
        );

        // A min-bet never maps to a passive action (the desync trap).
        let mut anyr = || 0.5;
        let m = h.map_real(&g, EventKind::BetTo(200), 150.0, 100.0, &mut anyr);
        let MapOutcome::Index(i) = m else { panic!("min-bet must map") };
        assert!(matches!(acts[i as usize], Action::Raise(_) | Action::AllIn));

        // A shove maps to the all-in end of the menu.
        let mut hi = || 0.999;
        let m = h.map_real(&g, EventKind::BetTo(20_000), 150.0, 100.0, &mut hi);
        let MapOutcome::Index(i) = m else { panic!("shove must map") };
        assert!(matches!(acts[i as usize], Action::AllIn));
    }

    #[test]
    fn mapping_is_deterministic_at_exact_abstract_sizes() {
        // A real bet whose fraction equals an abstract size maps to it with
        // probability 1 regardless of the random draw.
        let g = game();
        let h = hand(&g);
        let acts = h.actions(&g);
        let gs = h.gs(&g);
        let (pot, cb) = (gs.pot as f64, gs.current_bet as f64);
        for (i, &a) in acts.iter().enumerate() {
            let Action::Raise(level) = a else { continue };
            let frac = (level as f64 - cb) / (pot + cb);
            // Present the same fraction in real chips (pot 150, bet 100).
            let real_to = 100.0 + frac * 250.0;
            for draw in [0.0, 0.5, 0.999_999] {
                let mut unit = || draw;
                let m = h.map_real(&g, EventKind::BetTo(real_to.round() as u32), 150.0, 100.0, &mut unit);
                assert_eq!(m, MapOutcome::Index(i as u8), "exact size at draw {draw}");
            }
        }
    }

    #[test]
    fn cap_overflow_maps_to_call_and_reports_none_after() {
        let g = game();
        let mut h = hand(&g);
        // Drive the preflop to the cap: raise, reraise, reraise (3 = cap).
        for _ in 0..3 {
            let acts = h.actions(&g);
            let i = acts.iter().position(|a| matches!(a, Action::Raise(_))).unwrap_or_else(|| {
                acts.iter().position(|a| matches!(a, Action::AllIn)).unwrap()
            });
            h.apply(&g, i as u8);
        }
        // At the cap the menu is passive-only; a further real raise maps to Call.
        let acts = h.actions(&g);
        assert!(acts.iter().all(|a| !matches!(a, Action::Raise(_))), "cap reached");
        let mut unit = || 0.5;
        let m = h.map_real(&g, EventKind::BetTo(5_000), 3_000.0, 2_000.0, &mut unit);
        let MapOutcome::Index(i) = m else { panic!("cap overflow maps to the passive close") };
        assert!(matches!(acts[i as usize], Action::Call | Action::AllIn));
    }

    #[test]
    fn abstract_raise_translates_to_a_legal_real_size() {
        let g = game();
        let h = hand(&g);
        let acts = h.actions(&g);
        let raise = acts.iter().find(|a| matches!(a, Action::Raise(_))).copied().unwrap();
        // Real preflop: pot 150, outstanding 100, min raise 100, remaining 19900.
        let m = h.abstract_to_real(&g, raise, 150.0, 100, true, 100, 19_900);
        let RealMove::BetTo(to) = m else { panic!("raise stays a raise") };
        assert!(to >= 200, "at least a min-raise, got {to}");
        assert!(to <= 20_000, "never beyond the stack");

        // Passive and fold translations respect the real context.
        assert_eq!(h.abstract_to_real(&g, Action::Fold, 150.0, 100, true, 100, 19_900), RealMove::Fold);
        assert_eq!(h.abstract_to_real(&g, Action::Call, 150.0, 100, true, 100, 19_900), RealMove::Call);
        assert_eq!(h.abstract_to_real(&g, Action::Fold, 200.0, 0, false, 0, 19_900), RealMove::Check, "no bet to fold to");
        // All-in translates to the full remaining size.
        assert_eq!(
            h.abstract_to_real(&g, Action::AllIn, 150.0, 100, true, 100, 19_900),
            RealMove::BetTo(20_000)
        );
    }
}
