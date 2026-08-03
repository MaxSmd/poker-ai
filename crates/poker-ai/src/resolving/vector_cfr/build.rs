//! Public-tree construction: the recursive walk that turns a root
//! [`GameState`] into the flat node arena.
//!
//! Every leaf decision (fold / river showdown / depth-cut runout / river
//! chance reveal / continuation chooser) is made here, in one place, so the
//! tree's shape is readable without tracing the solver.

use poker_core::legal_actions;
use poker_core::state::{GameState, NO_CARD};

use super::keys::{MARKER_CONTINUATION, MARKER_NONE};
use super::node::{NodeKind, NodeStore};
use super::VectorCfr;
use crate::abstraction::features::PreparedShowdown;

impl VectorCfr {
    /// The subgame's legal actions under the raise cap: past `raises` ≥ cap,
    /// drop every `Raise` and any *voluntary* `AllIn` (one where a passive
    /// action exists) — the same filter as `BlueprintHoldem::capped_legal`.
    fn capped_legal(&self, gs: &GameState, raises: u32) -> Vec<poker_core::action::Action> {
        use poker_core::action::Action;
        let full = legal_actions(gs);
        if raises < self.raise_cap {
            return full.to_vec();
        }
        let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
        full.iter()
            .copied()
            .filter(|a| !(matches!(a, Action::Raise(_)) || (matches!(a, Action::AllIn) && has_passive)))
            .collect()
    }

    pub(super) fn build(&mut self, gs: GameState, history: Vec<u8>, raises: u32, prep: usize) -> usize {
        // A node is a leaf when the hand ends (fold / river showdown) or when the
        // current street wants a board card the resolve root does not have — the
        // depth cut of a turn or flop subgame (or an all-in run-out past it).
        let needs_runout = gs.board[..gs.board_cards_count()].contains(&NO_CARD);
        if gs.is_terminal() || needs_runout {
            let active = (0..gs.num_players as usize).filter(|&i| gs.folded & (1 << i) == 0).count();
            let real_cards = gs.board.iter().filter(|&&c| c != NO_CARD).count();
            let half_pot = (gs.pot as f64 / 2.0) / self.big_blind;
            if active <= 1 {
                // Someone folded: the payoff is board-independent and exact.
                let p = gs.terminal_payoffs();
                let id = self.kinds.len();
                self.kinds.push(NodeKind::Fold {
                    payoffs: [p[0] as f64 / self.big_blind, p[1] as f64 / self.big_blind],
                });
                return id;
            }
            if real_cards == 5 {
                // Complete board: exact river showdown.
                let id = self.kinds.len();
                self.kinds.push(NodeKind::Showdown { half_pot, prep });
                return id;
            }
            // Full-river mode, betting still open, only the river missing:
            // deal it as an explicit chance node and solve the real river
            // betting below — the exact replacement for the depth-cut leaf.
            // (A terminal here is an all-in run-out: no betting remains, so
            // the plain runout check-down is already exact.)
            if self.full_river && !gs.is_terminal() && real_cards == 4 {
                return self.build_river_chance(gs, history);
            }
            // Board undealt (depth cut or all-in run-out): check-down showdown
            // averaged over the runout.  With K > 1 continuations, the opponent
            // first chooses among them; otherwise a plain leaf.
            // In full-river mode a turn all-in run-out gets NO chooser: with no
            // betting left, the plain check-down is exact and a continuation
            // choice would hand the opponent fictitious post-all-in leverage.
            let exact_runout = self.full_river && real_cards == 4;
            if self.scales.len() > 1 && !exact_runout {
                return self.build_continuation_chooser(half_pot, gs.board, history);
            }
            let id = self.kinds.len();
            self.kinds.push(NodeKind::RunoutShowdown { half_pot });
            return id;
        }

        let player = gs.current_player();
        let acts = self.capped_legal(&gs, raises);
        let mut children = Vec::with_capacity(acts.len());
        for (i, &act) in acts.iter().enumerate() {
            let old_bet = gs.current_bet;
            let mut next = gs.clone();
            next.apply_action(act);
            let r = if next.current_bet > old_bet { raises + 1 } else { raises };
            let mut h = history.clone();
            h.push(i as u8);
            children.push(self.build(next, h, r, prep));
        }
        let store = self.stores.len();
        self.stores.push(NodeStore::new(acts.len()));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision {
            player,
            store,
            children,
            board: gs.board,
            history,
            marker: MARKER_NONE,
        });
        id
    }

    /// Deal the river inside a full-river turn resolve: one branch per live
    /// card, each with its own pre-sorted showdown board and the real river
    /// betting tree below.  The action history continues across the reveal
    /// (branches are distinguished by each decision node's own `board`, which
    /// the info key already includes).
    fn build_river_chance(&mut self, gs: GameState, history: Vec<u8>) -> usize {
        let mut used = 0u64;
        for &c in &gs.board[..4] {
            used |= 1 << c;
        }
        let mut children = Vec::with_capacity(48);
        for c in 0..52u8 {
            if used & (1 << c) != 0 {
                continue;
            }
            let mut next = gs.clone();
            next.board[4] = c;
            let prep = self.prepared.len();
            self.prepared.push(PreparedShowdown::new(next.board));
            // The raise counter resets on the new street, mirroring the
            // blueprint's per-street cap semantics.
            children.push((c, self.build(next, history.clone(), 0, prep)));
        }
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Chance { children });
        id
    }

    /// Build the depth-limit **continuation-choice** node: a
    /// decision owned by the fixed [`chooser`](Self::chooser) with one action per
    /// `scales` entry, whose `i`-th child is a `RunoutShowdown` at the inflated
    /// pot `half_pot·(1 + scales[i])`.  Inflating a check-down pot by `s` scales
    /// the (chop-relative) showdown value by exactly `1 + s`, so a scaled
    /// `RunoutShowdown` reproduces `MultiContinuationLeaf`'s continuation `i`
    /// without a new node kind.
    fn build_continuation_chooser(&mut self, half_pot: f64, board: [u8; 5], history: Vec<u8>) -> usize {
        let mut children = Vec::with_capacity(self.scales.len());
        for i in 0..self.scales.len() {
            let s = self.scales[i];
            let child = self.kinds.len();
            self.kinds.push(NodeKind::RunoutShowdown { half_pot: half_pot * (1.0 + s) });
            children.push(child);
        }
        let store = self.stores.len();
        self.stores.push(NodeStore::new(children.len()));
        let id = self.kinds.len();
        self.kinds.push(NodeKind::Decision {
            player: self.chooser,
            store,
            children,
            board,
            history,
            marker: MARKER_CONTINUATION,
        });
        id
    }
}
