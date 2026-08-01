//! Public-tree node kinds and per-node regret storage.
//!
//! The tree is a flat arena: [`NodeKind`] values indexed by `usize`, with
//! decision nodes pointing at a [`NodeStore`] row block.  Terminal *valuation*
//! (showdown, fold, gadget carry) lives here too, since it is a property of the
//! node kind rather than of the traversal.

use crate::abstraction::features::{combo_cards, PreparedRunout, PreparedShowdown};
use crate::resolving::belief_state::NUM_COMBOS;
use poker_core::state::NO_CARD;


/// A terminal or decision node of the public betting tree.
pub(super) enum NodeKind {
    /// River showdown: each player's value is the reach-weighted sweep over
    /// the opponent with `half_pot` at stake (bb units), on the pre-sorted
    /// complete board `prepared[prep]` (a full-river turn resolve has one
    /// prepared board per river card; a river resolve has exactly one).
    Showdown { half_pot: f64, prep: usize },
    /// Turn/flop depth-limit or all-in leaf: the board is incomplete, so the
    /// showdown is the check-down value averaged over the runout
    /// (`board_runout_cfvs`).  The vectorized analogue of the explicit oracle's
    /// `CheckdownLeafEval`.
    RunoutShowdown { half_pot: f64 },
    /// A fold terminal: card-independent per-player net payoff (bb units),
    /// weighted at solve time by the blocker-corrected opponent reach.
    Fold { payoffs: [f64; 2] },
    /// The river reveal inside a **full-river turn resolve**: one child per
    /// live river card.  Both players' reaches are masked per branch (combos
    /// using the card are impossible) and the sum is divided by 44 — the
    /// exact number of rivers consistent with any two disjoint holdings — so
    /// each hand pair's value is an exact conditional expectation, the same
    /// per-pair convention as `board_runout_cfvs`.
    Chance { children: Vec<(u8, usize)> },
    /// The re-solving gadget's **Terminate** terminal (Burch–Johanson–Bowling
    /// 2014, vectorized): the constrained opponent opts out of the subgame and
    /// banks its carried per-hand counterfactual value instead
    /// ([`VectorCfr::carried`], bb).  The resolver's seat receives the
    /// negation.  Constraining the resolve so Follow can never beat the carry
    /// is what makes re-solving *safe*: the opponent cannot profit from our
    /// strategy having been recomputed since the values were extracted.
    CfvTerminal,
    /// A betting decision for `player`; `children[a]` is the node after legal
    /// action `a`.  `store` indexes the regret/strategy arrays; `board`/`history`
    /// reproduce the explicit info key when emitting the strategy.
    ///
    /// `marker` namespaces the emitted key: `MARKER_NONE` for a betting node;
    /// `MARKER_CONTINUATION` for the opponent's depth-limit **continuation
    /// choice** (finding #1: `player` is the fixed chooser, `children[i]` a
    /// `RunoutShowdown` at the `i`-th continuation's inflated pot, matching the
    /// explicit oracle's continuation info set); `MARKER_GADGET` for the
    /// re-solving gadget's per-hand Follow/Terminate choice (never emitted).
    Decision {
        player: usize,
        store: usize,
        children: Vec<usize>,
        board: [u8; 5],
        history: Vec<u8>,
        marker: u8,
    },
}

/// Per-decision-node regret and strategy-sum, one row of `num_actions` per hand.
pub(super) struct NodeStore {
    pub(super) num_actions: usize,
    pub(super) regret: Vec<f64>,      // NUM_COMBOS × num_actions
    pub(super) strategy_sum: Vec<f64>, // NUM_COMBOS × num_actions
}

impl NodeStore {
    pub(super) fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            regret: vec![0.0; NUM_COMBOS * num_actions],
            strategy_sum: vec![0.0; NUM_COMBOS * num_actions],
        }
    }

    /// Regret-matched current strategy, all hands, row-major `[hand][action]`.
    pub(super) fn strategy(&self) -> Vec<f64> {
        let a = self.num_actions;
        let mut out = vec![0.0; NUM_COMBOS * a];
        for h in 0..NUM_COMBOS {
            let row = &self.regret[h * a..h * a + a];
            let pos: f64 = row.iter().map(|&r| r.max(0.0)).sum();
            let dst = &mut out[h * a..h * a + a];
            if pos > 0.0 {
                for (o, &r) in dst.iter_mut().zip(row) {
                    *o = r.max(0.0) / pos;
                }
            } else {
                dst.fill(1.0 / a as f64);
            }
        }
        out
    }

    /// The linear-averaged strategy (normalized `strategy_sum`, uniform where
    /// no mass accumulated — matching what `into_resolved` emits), row-major
    /// `[hand][action]`.  Used by the CFV-extraction evaluation pass.
    pub(super) fn average(&self) -> Vec<f64> {
        let a = self.num_actions;
        let mut out = vec![0.0; NUM_COMBOS * a];
        for h in 0..NUM_COMBOS {
            let row = &self.strategy_sum[h * a..h * a + a];
            let total: f64 = row.iter().sum();
            let dst = &mut out[h * a..h * a + a];
            if total > 0.0 {
                for (o, &s) in dst.iter_mut().zip(row) {
                    *o = s / total;
                }
            } else {
                dst.fill(1.0 / a as f64);
            }
        }
        out
    }

    /// CFR⁺ / RM⁺ update for this node's player: add the instantaneous
    /// counterfactual regret `child_p[a] − v_p` and **floor the accumulated regret
    /// at 0**, then accumulate the linearly-weighted (`weight = t`) reach-weighted
    /// strategy.  This mirrors [`PredictiveSolver`](crate::solver::predictive),
    /// which the resolver defaults to: RM⁺'s non-negativity keeps low-reach
    /// information sets responsive (DCFR's signed, discounted regret froze them at
    /// uniform once the opponent's strategy went pure — exploitable off-path).
    pub(super) fn update(&mut self, sigma: &[f64], child_p: &[Vec<f64>], v_p: &[f64], reach_p: &[f64; NUM_COMBOS], t: u64) {
        let a = self.num_actions;
        let weight = t as f64; // linear averaging
        for h in 0..NUM_COMBOS {
            let rp = reach_p[h];
            for (ai, cp) in child_p.iter().enumerate().take(a) {
                let idx = h * a + ai;
                let r = &mut self.regret[idx];
                *r = (*r + cp[h] - v_p[h]).max(0.0);
                self.strategy_sum[idx] += weight * rp * sigma[idx];
            }
        }
    }
}

/// The read-only traversal environment — everything `cfr` needs besides the
/// mutable stores, bundled so the recursion's signature stays sane.
pub(super) struct Env<'a> {
    pub(super) kinds: &'a [NodeKind],
    pub(super) board: &'a [u8; 5],
    pub(super) runout: Option<&'a PreparedRunout>,
    pub(super) prepared: &'a [PreparedShowdown],
    pub(super) cards: &'a [[u8; 2]],
    /// Carried opponent CFVs when a gadget wraps the root (`CfvTerminal`).
    pub(super) carried: Option<&'a [f64; NUM_COMBOS]>,
    /// The constrained opponent (the gadget's owner).
    pub(super) chooser: usize,
}

/// Values at the gadget's Terminate terminal, for `player`'s hands weighted by
/// the other seat's reach: the owner banks its carried per-hand CFV; the other
/// seat pays it (blocker-corrected inclusion–exclusion over the weighted
/// opponent reach, the same sums as a fold terminal).
pub(super) fn cfv_terminal_values(env: &Env<'_>, reach_other: &[f64; NUM_COMBOS], player: usize) -> Vec<f64> {
    let cfvs = env.carried.expect("CfvTerminal requires carried CFVs");
    if player == env.chooser {
        let vr = valid_reach(env.board, reach_other);
        cfvs.iter().zip(vr.iter()).map(|(&c, &r)| c * r).collect()
    } else {
        let mut weighted = [0.0f64; NUM_COMBOS];
        for ((w, &r), &c) in weighted.iter_mut().zip(reach_other.iter()).zip(cfvs.iter()) {
            *w = r * c;
        }
        valid_reach(env.board, &weighted).iter().map(|&w| -w).collect()
    }
}

/// Blocker-corrected opponent reach per hero hand: `total − card[a] − card[b] +
/// reach[h]`, zero for hero hands using a board card.  This is the reach mass of
/// opponents that do **not** share a card with the hero (or the board).
pub(super) fn valid_reach(board: &[u8; 5], reach: &[f64; NUM_COMBOS]) -> [f64; NUM_COMBOS] {
    let mut board_mask = 0u64;
    for &c in board {
        if c != NO_CARD {
            board_mask |= 1 << c;
        }
    }
    let mut total = 0.0;
    let mut card = [0.0; 52];
    for (i, &r) in reach.iter().enumerate() {
        if r == 0.0 {
            continue;
        }
        let [a, b] = combo_cards(i);
        if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
            continue;
        }
        total += r;
        card[a as usize] += r;
        card[b as usize] += r;
    }
    let mut out = [0.0; NUM_COMBOS];
    for (i, slot) in out.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
            continue;
        }
        *slot = total - card[a as usize] - card[b as usize] + reach[i];
    }
    out
}
