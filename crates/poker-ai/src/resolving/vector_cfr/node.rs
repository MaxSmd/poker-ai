//! Public-tree node kinds and per-node regret storage.
//!
//! The tree is a flat arena: [`NodeKind`] values indexed by `usize`, with
//! decision nodes pointing at a [`NodeStore`] row block.  Terminal *valuation*
//! (showdown, fold, gadget carry) lives here too, since it is a property of the
//! node kind rather than of the traversal.
//!
//! ## Layout: action-major, `f32`
//!
//! Every hot loop in [`solve`](super::solve) has the shape
//! `v[h] += sigma[a][h] · child[a][h]` — an axpy over the 1326-long hand
//! dimension with the action held fixed.  The stores are therefore
//! **action-major** (`[action][hand]`, index `ai * NUM_COMBOS + h`) so that
//! dimension is contiguous and the loops auto-vectorize; a hand-major layout
//! strides those reads by `num_actions` and defeats it entirely.
//!
//! The accumulators are `f32`.  They are the resident data (two arrays of
//! `1326 × num_actions` per decision node) and `update` sweeps both in full
//! every iteration, so this path is bandwidth-bound and halving the traffic is
//! the win — unlike the *blueprint* store, where the same idea needed the care
//! documented in [`lean_table`](crate::solver::lean_table).  Precision is not
//! at risk here for a structural reason: a resolve runs ~10³ iterations, so
//! with linear averaging the smallest meaningful increment is ~`2/t` of the
//! running sum — five orders of magnitude above `f32` epsilon.  The traversal
//! itself stays `f64`; only the stored accumulators narrow.
//!
//! ## Rejected: skipping dead hands by index
//!
//! A natural companion is to iterate only the combos that can carry reach (a
//! complete board blocks 245 of the 1326, before any range narrowing) via a
//! precomputed index list.  **Measured: it is slower.**  On the river arm of
//! `bench_resolve_cost` the gathered form ran 12.2 ms/iter against 11.6 for the
//! dense one — the indexed loads defeat the vectorization the layout above
//! exists to enable, and 18 % fewer elements does not pay for it.  Skipping is
//! only worth revisiting as a *dense compaction* (renumbering the hand
//! dimension so the live combos stay contiguous), which means teaching
//! `PreparedShowdown`/`PreparedRunout` the compacted space too.

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
    /// ([`VectorCfr::carried`](super::VectorCfr), bb).  The resolver's seat
    /// receives the negation.  Constraining the resolve so Follow can never beat
    /// the carry is what makes re-solving *safe*: the opponent cannot profit
    /// from our strategy having been recomputed since the values were extracted.
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

/// Per-decision-node regret and strategy-sum, **action-major**: `num_actions`
/// columns of `NUM_COMBOS` (see the module header for why).
pub(super) struct NodeStore {
    pub(super) num_actions: usize,
    /// `regret[ai * NUM_COMBOS + h]`
    pub(super) regret: Vec<f32>,
    /// `strategy_sum[ai * NUM_COMBOS + h]`
    pub(super) strategy_sum: Vec<f32>,
}

impl NodeStore {
    pub(super) fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            regret: vec![0.0; NUM_COMBOS * num_actions],
            strategy_sum: vec![0.0; NUM_COMBOS * num_actions],
        }
    }

    /// The `ai`-th action's strategy-sum column (used when emitting).
    pub(super) fn strategy_col(&self, ai: usize) -> &[f32] {
        &self.strategy_sum[ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS]
    }

    /// Regret-matched current strategy for all hands into `out` (action-major,
    /// `num_actions × NUM_COMBOS`), using `total` as scratch for the per-hand
    /// normalizer.  Two unit-stride passes: accumulate the positive-regret mass
    /// per hand, then divide.
    pub(super) fn strategy_into(&self, out: &mut [f64], total: &mut [f64]) {
        self.matched_into(&self.regret, out, total, true);
    }

    /// The linear-averaged strategy (normalized `strategy_sum`, uniform where no
    /// mass accumulated — matching what `into_resolved` emits), action-major.
    /// Used by the CFV-extraction evaluation pass.
    pub(super) fn average_into(&self, out: &mut [f64], total: &mut [f64]) {
        self.matched_into(&self.strategy_sum, out, total, false);
    }

    /// Shared kernel of [`strategy_into`](Self::strategy_into) and
    /// [`average_into`](Self::average_into): normalize `src` per hand, falling
    /// back to uniform where the mass is zero.  `floor_negative` applies
    /// regret-matching's `max(·, 0)` (the strategy-sum array is already
    /// non-negative, so the average pass skips it).
    fn matched_into(
        &self,
        src: &[f32],
        out: &mut [f64],
        total: &mut [f64],
        floor_negative: bool,
    ) {
        let a = self.num_actions;
        debug_assert_eq!(out.len(), a * NUM_COMBOS);
        debug_assert_eq!(total.len(), NUM_COMBOS);
        total.fill(0.0);
        for ai in 0..a {
            let col = &src[ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS];
            for (t, &r) in total.iter_mut().zip(col) {
                *t += if floor_negative { (r as f64).max(0.0) } else { r as f64 };
            }
        }
        let uniform = 1.0 / a as f64;
        for ai in 0..a {
            let col = &src[ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS];
            let dst = &mut out[ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS];
            for ((o, &r), &t) in dst.iter_mut().zip(col).zip(total.iter()) {
                let r = if floor_negative { (r as f64).max(0.0) } else { r as f64 };
                *o = if t > 0.0 { r / t } else { uniform };
            }
        }
    }

    /// CFR⁺ / RM⁺ update for this node's player: add the instantaneous
    /// counterfactual regret `child_v[a] − v` and **floor the accumulated regret
    /// at 0**, then accumulate the linearly-weighted (`weight = t`) reach-weighted
    /// strategy.  This mirrors [`PredictiveSolver`](crate::validation::solver::predictive),
    /// which the resolver defaults to: RM⁺'s non-negativity keeps low-reach
    /// information sets responsive (DCFR's signed, discounted regret froze them at
    /// uniform once the opponent's strategy went pure — exploitable off-path).
    ///
    /// `sigma` and `child_v` are action-major like the stores, so both arrays are
    /// swept in unit stride.
    pub(super) fn update(
        &mut self,
        sigma: &[f64],
        child_v: &[f64],
        v: &[f64],
        reach_p: &[f64; NUM_COMBOS],
        t: u64,
    ) {
        let a = self.num_actions;
        debug_assert_eq!(sigma.len(), a * NUM_COMBOS);
        debug_assert_eq!(child_v.len(), a * NUM_COMBOS);
        let weight = t as f64; // linear averaging
        for ai in 0..a {
            let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
            let regret = &mut self.regret[span.clone()];
            let strat = &mut self.strategy_sum[span.clone()];
            let cv = &child_v[span.clone()];
            let sg = &sigma[span];
            for h in 0..NUM_COMBOS {
                regret[h] = ((regret[h] as f64 + cv[h] - v[h]).max(0.0)) as f32;
                strat[h] += (weight * reach_p[h] * sg[h]) as f32;
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
/// the other seat's reach, written into `out`: the owner banks its carried
/// per-hand CFV; the other seat pays it (blocker-corrected inclusion–exclusion
/// over the weighted opponent reach, the same sums as a fold terminal).
pub(super) fn cfv_terminal_values(
    env: &Env<'_>,
    reach_other: &[f64; NUM_COMBOS],
    player: usize,
    out: &mut [f64],
) {
    let cfvs = env.carried.expect("CfvTerminal requires carried CFVs");
    if player == env.chooser {
        let mut vr = [0.0; NUM_COMBOS];
        valid_reach_into(env.board, reach_other, &mut vr);
        for ((o, &c), &r) in out.iter_mut().zip(cfvs.iter()).zip(vr.iter()) {
            *o = c * r;
        }
    } else {
        let mut weighted = [0.0f64; NUM_COMBOS];
        for ((w, &r), &c) in weighted.iter_mut().zip(reach_other.iter()).zip(cfvs.iter()) {
            *w = r * c;
        }
        let mut vr = [0.0; NUM_COMBOS];
        valid_reach_into(env.board, &weighted, &mut vr);
        for (o, &w) in out.iter_mut().zip(vr.iter()) {
            *o = -w;
        }
    }
}

/// Blocker-corrected opponent reach per hero hand: `total − card[a] − card[b] +
/// reach[h]`, zero for hero hands using a board card.  This is the reach mass of
/// opponents that do **not** share a card with the hero (or the board).
pub(super) fn valid_reach_into(board: &[u8; 5], reach: &[f64; NUM_COMBOS], out: &mut [f64]) {
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
    for (i, slot) in out.iter_mut().enumerate() {
        let [a, b] = combo_cards(i);
        if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
            *slot = 0.0;
            continue;
        }
        *slot = total - card[a as usize] - card[b as usize] + reach[i];
    }
}


/// Owned-array form of [`valid_reach_into`] — the shape the tests assert on.
#[cfg(test)]
pub(super) fn valid_reach(board: &[u8; 5], reach: &[f64; NUM_COMBOS]) -> [f64; NUM_COMBOS] {
    let mut out = [0.0; NUM_COMBOS];
    valid_reach_into(board, reach, &mut out);
    out
}
