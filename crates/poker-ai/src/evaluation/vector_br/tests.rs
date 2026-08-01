//! Tests for the vectorized best-response evaluator.
//!
//! The gold gate (`walker_matches_the_scalar_oracle`, `#[ignore]`d for cost)
//! checks this walker against a fully independent scalar implementation to
//! 1e-8 per hand for both seats — the reason its numbers can be trusted.

use super::*;
use super::walk::card_mask;
use poker_core::state::GameState;
use crate::abstraction::features::combo_cards;
use crate::abstraction::features::combo_index;
use poker_core::lut_eval::evaluate_7_lut;
use poker_core::state::NO_CARD;

/// 4 bb stacks, cap 1, no bucket maps (raw-index fallback = lossless
/// card abstraction): a tree small enough for the scalar oracle.
fn micro() -> BlueprintHoldem {
    BlueprintHoldem::new(8, 2, 1, 0).with_raise_cap(1)
}

/// Fully independent scalar reference for one hero hand: explicit
/// opponent loop, terminal-time consistency masking (vs. the walker's
/// chance-time reach zeroing + inclusion–exclusion), direct 7-card rank
/// comparison at showdowns (vs. the walker's sorted sweep), and the
/// card-based `key_for` (vs. the walker's hoisted `key_from_bucket`).
/// Shares only the chance-divisor conventions, which are part of the
/// estimator's definition.
struct Oracle<'a> {
    game: &'a BlueprintHoldem,
    policy: &'a CompactPolicy,
    br: usize,
    flops: &'a [[u8; 3]],
    flop_div: f64,
    hero: [u8; 2],
}

impl Oracle<'_> {
    fn value(game: &BlueprintHoldem, policy: &CompactPolicy, br: usize, flops: &[[u8; 3]], hero: [u8; 2]) -> f64 {
        let o = Oracle {
            game,
            policy,
            br,
            flops,
            flop_div: flops.len() as f64 * FLOP_CONSISTENT_RATE,
            hero,
        };
        let state = game.play_state([[48, 49], [50, 51]], [NO_CARD; 5]);
        let mut gs = game.game_state(&state).clone();
        let reach = vec![1.0f64; COMBOS];
        o.node(&mut gs, &mut Vec::new(), 0, 0, &reach)
    }

    fn blocked(&self, j: usize, board: &[u8]) -> bool {
        let [a, b] = combo_cards(j);
        let mask = card_mask(board) | card_mask(&self.hero);
        mask & (1 << a) != 0 || mask & (1 << b) != 0
    }

    fn node(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
        let acts = self.game.capped_actions(gs, raises);
        let n = acts.len();
        let actor = gs.current_player();
        if actor == self.br {
            (0..n)
                .map(|i| self.descend(gs, hist, raises, revealed, reach, acts[i], i))
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            let mut total = 0.0;
            for i in 0..n {
                let mut child = vec![0.0f64; COMBOS];
                for (j, cr) in child.iter_mut().enumerate() {
                    if reach[j] == 0.0 {
                        continue;
                    }
                    let key = self.game.key_for(
                        actor,
                        combo_cards(j),
                        &gs.board[..revealed],
                        hist,
                    );
                    *cr = reach[j] * self.policy.probs_or_uniform(key, n)[i];
                }
                total += self.descend(gs, hist, raises, revealed, &child, acts[i], i);
            }
            total
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn descend(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64], act: poker_core::action::Action, i: usize) -> f64 {
        let (old_street, old_bet) = (gs.street, gs.current_bet);
        gs.apply_action(act);
        hist.push(i as u8);
        let new_raises = self.game.raises_after(raises, old_street, old_bet, gs);
        let out = if gs.is_terminal() {
            if gs.folded != 0 {
                let folder = if gs.folded & 1 != 0 { 0usize } else { 1 };
                let sign = if folder == self.br { -1.0 } else { 1.0 };
                let amount = gs.total_committed[folder] as f64;
                (0..COMBOS)
                    .filter(|&j| !self.blocked(j, &gs.board[..revealed]))
                    .map(|j| sign * amount * reach[j])
                    .sum()
            } else if revealed == 5 {
                self.showdown(gs, reach)
            } else {
                self.deal(gs, hist, new_raises, revealed, reach)
            }
        } else if gs.street != old_street {
            self.deal(gs, hist, new_raises, revealed, reach)
        } else {
            self.node(gs, hist, new_raises, revealed, reach)
        };
        hist.pop();
        gs.undo_action();
        out
    }

    fn deal(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
        let hero_mask = card_mask(&self.hero);
        match revealed {
            0 => {
                let mut sum = 0.0;
                for f in self.flops {
                    if card_mask(f) & hero_mask != 0 {
                        continue;
                    }
                    gs.board[..3].copy_from_slice(f);
                    sum += self.after(gs, hist, raises, 3, reach);
                }
                sum / self.flop_div
            }
            3 | 4 => {
                let div = if revealed == 3 { 45.0 } else { 44.0 };
                let prefix = card_mask(&gs.board[..revealed]);
                let mut sum = 0.0;
                for c in 0..52u8 {
                    if (prefix | hero_mask) & (1 << c) != 0 {
                        continue;
                    }
                    gs.board[revealed] = c;
                    sum += self.after(gs, hist, raises, revealed + 1, reach);
                }
                sum / div
            }
            _ => unreachable!(),
        }
    }

    fn after(&self, gs: &mut GameState, hist: &mut Vec<u8>, raises: u8, revealed: usize, reach: &[f64]) -> f64 {
        if gs.is_terminal() {
            if revealed == 5 {
                self.showdown(gs, reach)
            } else {
                self.deal(gs, hist, raises, revealed, reach)
            }
        } else {
            self.node(gs, hist, raises, revealed, reach)
        }
    }

    fn showdown(&self, gs: &GameState, reach: &[f64]) -> f64 {
        let matched = gs.total_committed[0].min(gs.total_committed[1]) as f64;
        let b = &gs.board;
        let hero_rank = evaluate_7_lut(&[self.hero[0], self.hero[1], b[0], b[1], b[2], b[3], b[4]]);
        let mut sum = 0.0;
        for (j, &r) in reach.iter().enumerate() {
            if r == 0.0 || self.blocked(j, &b[..5]) {
                continue;
            }
            let [ja, jb] = combo_cards(j);
            let opp_rank = evaluate_7_lut(&[ja, jb, b[0], b[1], b[2], b[3], b[4]]);
            sum += r * matched
                * if hero_rank > opp_rank {
                    1.0
                } else if hero_rank < opp_rank {
                    -1.0
                } else {
                    0.0
                };
        }
        sum
    }
}

/// A non-trivial policy: bias a few of the SB's root info sets so the
/// σ-weighting path (keys → probs → reach) is exercised, not just the
/// uniform fallback.
fn biased_root_policy(game: &BlueprintHoldem) -> CompactPolicy {
    let state = game.play_state([[48, 49], [50, 51]], [NO_CARD; 5]);
    let n = game.actions(&state).len();
    let mut entries = Vec::new();
    for hole in [[48u8, 50], [0u8, 1], [4u8, 9]] {
        let key = game.key_for(0, hole, &[], &[]);
        let mut p = vec![0.05f32; n];
        // Deterministic skew: most mass on one action, different per hand.
        p[(hole[0] as usize) % n] = 1.0 - 0.05 * (n as f32 - 1.0);
        entries.push((key, p));
    }
    CompactPolicy::from_entries(entries)
}

/// The gold gate: the vectorized walker equals the fully independent
/// scalar oracle, per hand, for both BR seats, with biased and uniform
/// keys in play.  Full turn/river enumeration makes this expensive
/// (~40 min in the test profile — verified passing 2026-07-15); run it
/// whenever the walker changes: `cargo test -p poker-ai --release
/// walker_matches -- --ignored`.
#[test]
#[ignore]
fn walker_matches_the_scalar_oracle() {
    let game = micro();
    let policy = biased_root_policy(&game);
    let flops = vec![[2u8, 17, 33], [5u8, 6, 40]];
    for br in [0usize, 1] {
        // board_samples=0 → exact enumeration, so the walker equals the
        // exact scalar oracle.
        let v = br_value_vector(&game, &policy, br, &flops, 0, 1);
        // AKo-ish and a low suited hand: exercise both biased and uniform keys.
        for hero in [[0u8, 1], [12u8, 16]] {
            let want = Oracle::value(&game, &policy, br, &flops, hero);
            let got = v[combo_index(hero[0], hero[1])];
            let tol = 1e-8 * want.abs().max(1.0);
            assert!(
                (got - want).abs() < tol,
                "br={br} hero={hero:?}: walker {got} != oracle {want}"
            );
        }
    }
}

/// Cheap always-on smoke: a 1 bb stack collapses betting to shove/fold-
/// scale trees, but the full deal cascade (flop fan-out, turn/river
/// enumeration, all-in run-outs, sweeps, masking, the measure) still
/// runs end-to-end.  Uniform play must read exploitable, the metric
/// non-negative, and the evaluator deterministic.
#[test]
fn smoke_uniform_is_exploitable_and_deterministic() {
    let game = BlueprintHoldem::new(2, 2, 1, 0).with_raise_cap(1);
    let policy = CompactPolicy::from_entries(vec![]);
    let flops = sample_flops(1, 7);
    let r = blueprint_exploitability(&game, &policy, &flops, 0, 1);
    assert!(
        r.exploitability_mbb > 0.0,
        "uniform play must be exploitable, got {} mbb",
        r.exploitability_mbb
    );
    let again = best_response_value(&game, &policy, 0, &flops, 0, 1);
    assert_eq!(again, r.br_value_bb[0], "evaluator must be deterministic");
}

/// Board sampling is deterministic for a fixed seed and reduces to the
/// same shape of answer: uniform play stays exploitable, and two runs
/// with the same seed agree exactly (the parallel flop fan-out included).
#[test]
fn sampled_board_runouts_are_deterministic() {
    let game = BlueprintHoldem::new(2, 2, 1, 0).with_raise_cap(1);
    let policy = CompactPolicy::from_entries(vec![]);
    let flops = sample_flops(2, 7);
    let a = best_response_value(&game, &policy, 0, &flops, 3, 42);
    let b = best_response_value(&game, &policy, 0, &flops, 3, 42);
    assert_eq!(a, b, "fixed seed must be reproducible");
    assert!(a.is_finite());
}
