//! The cursor fast path: zero-per-node-allocation traversals.
//!
//! The clone-based [`Game`] trait returns a freshly-allocated child on every
//! `apply`.  For the real-mechanics blueprint games that wrap a
//! `poker_core::GameState`, that clone drags along a pre-allocated `UndoStack`,
//! so the traversal heap-allocates on *every node* — discarding the
//! zero-allocation mutate-and-undo design `poker_core` was built for.  These
//! methods are the same external-sampling MCCFR as the clone path, but driven
//! through [`CursorGame`]: a single `GameState` is walked in place
//! (`apply`/`undo`), the legal-action list is computed once per node and held on
//! the stack frame, and information keys are folded without a per-node `Vec`.
//! They reuse every regret/strategy/baseline helper unchanged and are
//! **bit-identical** to the clone-based path for a fixed seed (proven by the
//! `*_matches_clone_*` tests).

use rayon::prelude::*;

use super::parallel::{record_strategy_delta, record_traverser_delta, splitmix, Delta};
use super::{Mccfr, Node};
use crate::games::{CursorGame, Game};
use crate::solver::cfr::Variant;
use crate::util::rng::{sample_index, xorshift_next_unit};

impl<G: Game + CursorGame> Mccfr<G> {
    /// Cursor-based counterpart of [`train`](Self::train): zero per-node
    /// allocation.  Bit-identical to `train` for a fixed seed on a game that
    /// implements both [`Game`] and [`CursorGame`].
    pub fn train_fast(&mut self, iters: u64) {
        let mut cursor = CursorGame::root(&self.game);
        for _ in 0..iters {
            self.iterations += 1;
            let t = self.iterations;
            let players = CursorGame::num_players(&self.game);
            for traverser in 0..players {
                // `cursor` starts (and is left) at the pre-deal root; the chance
                // branch deals it in place and restores it before returning.
                self.traverse_cursor(&mut cursor, traverser, t);
            }
        }
    }

    /// External-sampling traversal over a [`CursorGame`] cursor — the structural
    /// mirror of [`traverse`](Self::traverse), reaching children via
    /// `apply`/`undo` instead of cloning.  Leaves `cursor` exactly as it found it.
    fn traverse_cursor(
        &mut self,
        cursor: &mut <G as CursorGame>::Cursor,
        traverser: usize,
        t: u64,
    ) -> f64 {
        self.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            // Non-enumerable chance (a full random deal): deal in place, recurse,
            // then restore the pre-deal root.  No per-outcome control variate is
            // possible (the outcome list does not exist).
            let mut r = self.rng;
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(&mut r));
            self.rng = r;
            let v = self.traverse_cursor(cursor, traverser, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let key = CursorGame::info_key(&self.game, cursor);
        // Legal actions computed once per node and held on this stack frame.
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let strategy = self.nodes.entry(key).or_insert_with(|| Node::new(num_actions)).strategy();

        if player == traverser {
            let consec = self.pruned_streaks(key, t);

            let mut util = vec![0.0; num_actions];
            let mut traversed = vec![true; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                let pruned = match (&consec, self.pruning) {
                    (Some(cb), Some((cfg, _))) => cfg.should_prune(cb[a]),
                    _ => false,
                };
                if pruned {
                    traversed[a] = false;
                    continue;
                }
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_cursor(cursor, traverser, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            self.update_regret(key, t, &util, node_value, &traversed);
            self.refresh_traverser_baseline(key, traverser, &util, &traversed);
            node_value
        } else {
            self.update_strategy(key, t, &strategy);
            let a = self.sample(&strategy);
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_cursor(cursor, traverser, t);
            CursorGame::undo(&self.game, cursor);
            self.corrected_opponent_value_serial(key, &strategy, a, v_child, traverser)
        }
    }

    /// Cursor-based counterpart of [`train_parallel`](Self::train_parallel):
    /// mini-batch MCCFR with each worker walking its own in-place cursor.
    /// Bit-identical to `train_parallel` for a fixed seed and `batch`.
    pub fn train_parallel_fast(&mut self, total_iters: u64, batch: u64)
    where
        G: Sync,
    {
        let batch = batch.max(1);
        let players = CursorGame::num_players(&self.game);
        let mut done = 0u64;
        while done < total_iters {
            let this = batch.min(total_iters - done);
            let base = self.iterations;

            let deltas: Vec<Delta> = (0..this)
                .into_par_iter()
                .map(|i| {
                    let t = base + i + 1;
                    let mut rng = splitmix(self.rng, t);
                    let mut delta = Delta::default();
                    let mut cursor = CursorGame::root(&self.game);
                    for traverser in 0..players {
                        self.traverse_ro_cursor(&mut cursor, traverser, &mut rng, &mut delta, t);
                    }
                    delta
                })
                .collect();

            for (i, delta) in deltas.into_iter().enumerate() {
                self.iterations += 1;
                self.apply_delta(delta, base + i as u64 + 1);
            }
            done += this;
        }
    }

    /// Read-only cursor traversal writing into `delta` — the cursor counterpart
    /// of [`traverse_ro`](Self::traverse_ro).  Leaves `cursor` as it found it.
    fn traverse_ro_cursor(
        &self,
        cursor: &mut <G as CursorGame>::Cursor,
        traverser: usize,
        rng: &mut u64,
        delta: &mut Delta,
        t: u64,
    ) -> f64 {
        delta.nodes_visited += 1;
        if CursorGame::is_terminal(&self.game, cursor) {
            return CursorGame::utility(&self.game, cursor, traverser);
        }
        if CursorGame::is_chance(&self.game, cursor) {
            CursorGame::sample_chance(&self.game, cursor, || xorshift_next_unit(rng));
            let v = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
            CursorGame::undo_chance(&self.game, cursor);
            return v;
        }

        let player = CursorGame::current_player(&self.game, cursor);
        let key = CursorGame::info_key(&self.game, cursor);
        let actions = CursorGame::legal(&self.game, cursor);
        let acts = actions.as_ref();
        let num_actions = acts.len();
        let strategy = self.strategy_ro(key, num_actions);

        if player == traverser {
            let mut util = vec![0.0; num_actions];
            let mut node_value = 0.0;
            for a in 0..num_actions {
                CursorGame::apply(&self.game, cursor, a, acts[a]);
                util[a] = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
                CursorGame::undo(&self.game, cursor);
                node_value += strategy[a] * util[a];
            }
            let sgn = self.use_baseline.then_some(Self::sign(traverser));
            record_traverser_delta(delta, key, &util, node_value, sgn);
            node_value
        } else {
            let weight = match self.variant {
                Variant::Vanilla => 1.0,
                Variant::Dcfr(d) => d.strategy_weight(t),
            };
            record_strategy_delta(delta, key, weight, &strategy);
            let a = sample_index(strategy.iter().copied(), xorshift_next_unit(rng));
            CursorGame::apply(&self.game, cursor, a, acts[a]);
            let v_child = self.traverse_ro_cursor(cursor, traverser, rng, delta, t);
            CursorGame::undo(&self.game, cursor);
            self.corrected_opponent_value(key, &strategy, a, v_child, traverser, delta)
        }
    }
}
