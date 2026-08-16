//! Where the agent actually chooses a move.
//!
//! Three regimes, dispatched by `should_resolve` and the tracker's sync state:
//! the **blueprint** lookup (preflop/flop/turn by default), a full-range
//! **vectorized re-solve** of the real state (river by default, optionally
//! turn/flop), and the **desynced** fallback for spots the abstraction has no
//! node for (an opponent raise past the cap), where hand-vs-range equity
//! against the frozen range is the only well-defined signal left.

use poker_core::action::Action;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::{Bot, HandState};
use crate::games::blueprint::BlueprintHoldem;
use crate::play::equity::equity_vs_range;
use crate::play::protocol::{EventKind, Parsed, BIG_BLIND, SMALL_BLIND, STACK_SIZE};
use crate::play::tracker::{MapOutcome, RealMove};
use crate::resolving::belief_state::{combo_cards, combo_index, NUM_COMBOS};
use crate::resolving::vector_cfr::{
    capped_root_actions, subgame_info_key, VectorCfr,
};

impl Bot {

    /// Blueprint decision (preflop / flop / turn, and the river fallback).
    pub(super) fn decide_blueprint(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let facing = parsed.last_bet_size > 0;
        if !hs.hand.expects(&self.game, hs.my_seat, parsed.street) {
            hs.pending_self = Some(None);
            return self.decide_desynced(hs, parsed, board);
        }

        let acts = hs.hand.actions(&self.game);
        let key = hs.hand.info_key(&self.game);
        let probs = self.policy.probs_or_uniform(key, acts.len());
        let idx = self.sample(&probs);

        let real_min = parsed.last_bet_size.max(BIG_BLIND);
        let remaining = STACK_SIZE - parsed.total_last_bet_to;
        let mv = hs.hand.abstract_to_real(
            &self.game,
            acts[idx],
            parsed.pot() as f64,
            parsed.street_last_bet_to,
            facing,
            real_min.min(remaining),
            remaining,
        );

        self.update_range(hs, hs.my_seat, idx as u8);
        hs.hand.apply(&self.game, idx as u8);
        hs.pending_self = Some(Some(idx as u8));
        mv
    }

    /// Decision when the abstract game has no node for this spot.
    ///
    /// This happens when the opponent raises past the blueprint's cap: the
    /// abstraction offers no aggressive action, the event cannot be mapped, and
    /// the tracker stops advancing for the rest of the hand.  The blueprint is
    /// unusable here, so fall back on the one thing that is still well defined
    /// — our hand's equity against the opponent's belief range, weighed against
    /// the price we are being laid.
    ///
    /// The range is frozen at the last node we could translate, which is stale
    /// but blueprint-consistent up to that point; the alternative is no
    /// information at all.  We never raise from a desynced state.
    fn decide_desynced(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let me = hs.my_pos as usize;
        let stack = STACK_SIZE - parsed.total_committed[me];
        let to_call = (parsed.street_last_bet_to - parsed.street_committed[me]).min(stack);
        if to_call == 0 {
            return RealMove::Check;
        }

        let opp = &hs.ranges[1 - hs.my_seat];
        let mut rng = self.rng;
        let equity = equity_vs_range(hs.my_hole, board, opp, &mut rng);
        self.rng = rng;

        // Price of the call: chips in versus the pot we would be contesting.
        let odds = to_call as f64 / (parsed.pot() + to_call) as f64;
        if equity >= odds {
            RealMove::Call
        } else {
            RealMove::Fold
        }
    }

    /// Postflop decision by full-range vectorized re-solve of the real state —
    /// exact to showdown on the river, cut at the undealt street on turn/flop
    /// (runout leaves with K continuations).  Dispatched by [`Self::should_resolve`].
    pub(super) fn decide_resolve(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let street = parsed.street;
        let root = self.resolve_root(hs, parsed, board);
        let acts = capped_root_actions(&root, self.cfg.river_cap);
        if acts.is_empty() {
            return self.decide_blueprint(hs, parsed, board);
        }

        // Ranges: card removal, opponent additionally filtered by our cards,
        // and our actual hand floored to nonzero mass so the resolve always
        // covers it.
        let opp_seat = 1 - hs.my_seat;
        let mut beliefs = [hs.ranges[0].clone(), hs.ranges[1].clone()];
        for b in &mut beliefs {
            b.remove_board(board);
        }
        let mut mask = vec![1.0; NUM_COMBOS];
        for (i, m) in mask.iter_mut().enumerate() {
            let [a, b] = combo_cards(i);
            if a == hs.my_hole[0] || a == hs.my_hole[1] || b == hs.my_hole[0] || b == hs.my_hole[1] {
                *m = 0.0;
            }
        }
        beliefs[opp_seat].update(&mask);
        let me_idx = combo_index(hs.my_hole[0], hs.my_hole[1]);
        let floor = beliefs[hs.my_seat].probs.iter().cloned().fold(0.0, f64::max) * 0.02;
        if beliefs[hs.my_seat].probs[me_idx] < floor {
            beliefs[hs.my_seat].probs[me_idx] = floor.max(1e-6);
            let ones = vec![1.0; NUM_COMBOS];
            beliefs[hs.my_seat].update(&ones); // renormalize
        }

        // River: exact showdown terminals.  Turn (full-river mode): the river
        // is dealt as a chance node and the real river betting solved below —
        // exact to showdown, no leaf model.  Otherwise turn/flop cut at the
        // undealt street with the K-continuation check-down leaf.
        let (mut solver, iters) = if board.len() == 5 {
            (VectorCfr::new_capped(&root, &beliefs, self.cfg.river_cap), self.cfg.river_iters)
        } else if board.len() == 4 && self.cfg.turn_full_river {
            (
                VectorCfr::new_full(&root, &beliefs, self.cfg.river_cap, vec![0.0], true),
                self.cfg.turn_iters,
            )
        } else {
            // Turn and flop cut at the undealt street, so every leaf pays for a
            // runout sweep once per iteration — 48 completions on the turn and
            // 1176 on the flop.  `runout_sample` is what keeps that inside a
            // live clock; the river arm above never reaches here (a complete
            // board has no runout to sample).
            (
                VectorCfr::new_capped_multi(
                    &root,
                    &beliefs,
                    self.cfg.river_cap,
                    self.cfg.continuations.clone(),
                )
                .with_runout_sample(self.cfg.runout_sample),
                self.cfg.turn_iters,
            )
        };
        // Continual re-solving: constrain by the opponent CFVs carried from
        // our previous resolve this hand (the first resolve bootstraps), then
        // refresh the carry from the new solution.
        if self.cfg.continual {
            if let Some(cfvs) = &hs.carried_cfvs {
                solver = solver.with_opponent_gadget(*cfvs.clone());
            }
        }
        solver.run(iters);
        if self.cfg.continual {
            hs.carried_cfvs = Some(Box::new(solver.opponent_cfvs()));
        }
        let resolved = solver.into_resolved();

        let mut hole = hs.my_hole;
        hole.sort_unstable();
        // The root key filters NO_CARD, so a padded board yields the visible
        // cards only — matching the solver's own root emission.
        let mut board5 = [NO_CARD; 5];
        board5[..board.len()].copy_from_slice(board);
        let key = subgame_info_key(hs.my_seat, hole, &board5, &[]);
        let Some(probs) = resolved.strategy.get(&key).cloned() else {
            // Unreached in the resolve (shouldn't happen with the floor):
            // degrade to the blueprint.
            return self.decide_blueprint(hs, parsed, board);
        };

        let idx = self.sample(&probs);
        let action = acts[idx];

        // Keep the abstract tracker and our range coherent with what we do:
        // map the real move back into the abstract game as if observed.
        let facing = parsed.last_bet_size > 0;
        let mv = match action {
            Action::Fold => RealMove::Fold,
            Action::Check => RealMove::Check,
            Action::Call => RealMove::Call,
            Action::Raise(level) => RealMove::BetTo(level_to_pos_level(&root, hs.my_seat, level)),
            Action::AllIn => {
                let gs = &root;
                let level = gs.street_bets[hs.my_seat] + gs.stacks[hs.my_seat];
                if level <= gs.current_bet {
                    if facing {
                        RealMove::Call
                    } else {
                        RealMove::Check
                    }
                } else {
                    RealMove::BetTo(level)
                }
            }
        };
        let kind = match mv {
            RealMove::Fold => EventKind::Fold,
            RealMove::Check => EventKind::Check,
            RealMove::Call => EventKind::Call,
            RealMove::BetTo(n) => EventKind::BetTo(n),
        };
        if hs.hand.expects(&self.game, hs.my_seat, street) {
            let mut rng = self.rng;
            let mut unit = || {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
            };
            let mapped = hs.hand.map_real(
                &self.game,
                kind,
                parsed.pot() as f64,
                parsed.street_last_bet_to as f64,
                &mut unit,
            );
            self.rng = rng;
            if let MapOutcome::Index(i) = mapped {
                self.update_range(hs, hs.my_seat, i);
                hs.hand.apply(&self.game, i);
                hs.pending_self = Some(Some(i));
            } else {
                hs.pending_self = Some(None);
            }
        } else {
            hs.pending_self = Some(None);
        }
        mv
    }

    /// Synthesize the real postflop public state (Slumbot chips) as an engine
    /// `GameState` at `parsed.street` — the resolve root.  The board is padded
    /// with `NO_CARD` past the revealed cards (turn/flop), which the solver
    /// reads as the depth cut.
    fn resolve_root(&self, hs: &HandState, parsed: &Parsed, board: &[u8]) -> GameState {
        let mut board5 = [NO_CARD; 5];
        board5[..board.len()].copy_from_slice(board);
        // Placeholder opponent cards (never read by the public-tree solver).
        let mut used = 0u64;
        for &c in board {
            used |= 1 << c;
        }
        used |= 1 << hs.my_hole[0];
        used |= 1 << hs.my_hole[1];
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        let opp_cards = [spare.next().unwrap(), spare.next().unwrap()];
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[hs.my_seat] = hs.my_hole;
        holes[1 - hs.my_seat] = opp_cards;

        let mut gs = GameState::new(2, BIG_BLIND, SMALL_BLIND, [STACK_SIZE; MAX_PLAYERS], holes, board5, 0);
        gs.street = parsed.street;
        for pos in 0..2usize {
            let seat = 1 - pos;
            gs.total_committed[seat] = parsed.total_committed[pos];
            gs.stacks[seat] = STACK_SIZE - parsed.total_committed[pos];
            gs.street_bets[seat] = parsed.street_committed[pos];
        }
        gs.pot = parsed.pot();
        gs.current_bet = parsed.street_committed[0].max(parsed.street_committed[1]);
        gs.min_raise = parsed.last_bet_size.max(BIG_BLIND);
        gs.to_act = hs.my_seat as u8;
        gs.folded = 0;
        gs.allin = 0;
        for seat in 0..2 {
            if gs.stacks[seat] == 0 {
                gs.allin |= 1 << seat;
            }
        }
        let street_fresh = !parsed.events.iter().any(|e| e.street == parsed.street);
        gs.players_to_act = if street_fresh && gs.current_bet == 0 { 2 } else { 1 };
        gs.last_aggressor = if parsed.last_bettor >= 0 {
            (1 - parsed.last_bettor) as u8
        } else {
            gs.to_act
        };
        gs
    }

    /// Purify + sample an action index from a distribution.
    fn sample(&mut self, probs: &[f64]) -> usize {
        let mut kept: Vec<f64> = probs.iter().map(|&p| if p < self.cfg.purify { 0.0 } else { p }).collect();
        let total: f64 = kept.iter().sum();
        if total <= 0.0 {
            // Everything purified away: play the argmax.
            return probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        for p in &mut kept {
            *p /= total;
        }
        let draw = self.unit();
        let mut acc = 0.0;
        for (i, &p) in kept.iter().enumerate() {
            acc += p;
            if draw < acc {
                return i;
            }
        }
        kept.len() - 1
    }

    pub fn game(&self) -> &BlueprintHoldem {
        &self.game
    }
}

/// An engine `Raise(level)` at the resolve root is already the actor's new
/// street-bet level in real chips — identical to Slumbot's `b<level>`
/// semantics.  Kept as a named function to make that unit statement explicit.
fn level_to_pos_level(_root: &GameState, _seat: usize, level: u32) -> u32 {
    level
}
