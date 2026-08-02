//! The CFR⁺ traversal, the evaluation pass, and strategy emission.
//!
//! [`VectorCfr::cfr`] is the training recursion (regrets updated at the
//! traverser's own decision nodes); [`VectorCfr::eval_average`] is its
//! no-update twin used for CFV extraction; [`VectorCfr::into_resolved`] emits
//! the deployable average keyed by the oracle's info key.
//!
//! ## Buffers
//!
//! Both recursions write their per-hand value vector into a caller-supplied
//! `out` rather than returning a fresh `Vec`.  A returning form allocates at
//! *every* node of *every* iteration — a 1326-long `f64` vector is 10.6 kB, and
//! a decision node also wanted one `a × NUM_COMBOS` strategy block (32 kB at
//! `a = 3`) plus one child-value vector per action.  On a 600-node river tree at
//! 1500 iterations that is millions of multi-kilobyte allocations, which cost
//! more than the arithmetic they carry.  [`Scratch`] recycles them instead, so
//! a solve allocates O(depth × actions) buffers in total and then reuses them
//! for the whole run.

use std::collections::HashMap;

use super::keys::{info_key, MARKER_GADGET};
use super::node::{cfv_terminal_values, valid_reach_into, Env, NodeKind, NodeStore};
use super::{VectorCfr, VectorResolved};
use crate::abstraction::features::combo_cards;
use crate::resolving::belief_state::NUM_COMBOS;

/// A pool of reusable per-hand buffers for the traversal.
///
/// Buffers are handed out zero-filled and returned when a stack frame is done
/// with them, so the recursion's working set is allocated once and then
/// recycled.  Sizes vary (`NUM_COMBOS` for a value or reach vector,
/// `num_actions × NUM_COMBOS` for a strategy or child-value block); the pool is
/// size-agnostic and simply resizes on hand-out, converging on the largest
/// capacity the tree needs.
#[derive(Default)]
pub(super) struct Scratch {
    pool: Vec<Vec<f64>>,
}

impl Scratch {
    /// A zero-filled buffer of length `len`.
    fn take(&mut self, len: usize) -> Vec<f64> {
        match self.pool.pop() {
            Some(mut v) => {
                v.clear();
                v.resize(len, 0.0);
                v
            }
            None => vec![0.0; len],
        }
    }

    /// Return a buffer for reuse.
    fn give(&mut self, v: Vec<f64>) {
        self.pool.push(v);
    }
}

impl VectorCfr {
    /// The opponent's per-hand counterfactual values at the (inner) resolve
    /// root under the emitted **average** profile, conditional on each hand
    /// (bb; zero where its prior reach has no mass) — the carry for the next
    /// continual resolve.  Evaluated below any gadget wrap, i.e. assuming
    /// Follow, matching how the explicit `ContinualResolver` refreshes.
    pub fn opponent_cfvs(&mut self) -> [f64; NUM_COMBOS] {
        let opp = self.chooser;
        let (reach_opp, reach_me) =
            if opp == 0 { (self.reach0, self.reach1) } else { (self.reach1, self.reach0) };
        let env = Env {
            kinds: &self.kinds,
            board: &self.board,
            runout: self.runout.as_ref(),
            prepared: &self.prepared,
            cards: &self.cards,
            carried: self.carried.as_deref(),
            chooser: self.chooser,
        };
        let mut raw = vec![0.0; NUM_COMBOS];
        Self::eval_average(
            &env,
            &self.stores,
            &mut self.scratch,
            self.inner_root,
            &reach_me,
            opp,
            &mut raw,
        );
        let mut mass = [0.0; NUM_COMBOS];
        valid_reach_into(&self.board, &reach_me, &mut mass);
        let mut out = [0.0; NUM_COMBOS];
        for (o, (&v, (&m, &prior))) in
            out.iter_mut().zip(raw.iter().zip(mass.iter().zip(reach_opp.iter())))
        {
            if m > 0.0 && prior > 0.0 {
                *o = v / m;
            }
        }
        out
    }

    /// Expected value per `player` hand under the stored **average** strategy
    /// (both seats), weighted by the other seat's reach, into `out` — the
    /// evaluation (no-update) counterpart of [`cfr`](Self::cfr) used for CFV
    /// extraction.
    fn eval_average(
        env: &Env<'_>,
        stores: &[NodeStore],
        scratch: &mut Scratch,
        id: usize,
        reach_other: &[f64; NUM_COMBOS],
        player: usize,
        out: &mut [f64],
    ) {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                let o: &mut [f64; NUM_COMBOS] = out.try_into().expect("value buffer is NUM_COMBOS");
                o.fill(0.0);
                env.prepared[*prep].accumulate(reach_other, *half_pot, o);
            }
            NodeKind::RunoutShowdown { half_pot } => {
                let o: &mut [f64; NUM_COMBOS] = out.try_into().expect("value buffer is NUM_COMBOS");
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_other, *half_pot, o);
            }
            NodeKind::Fold { payoffs } => {
                let mut vr = [0.0; NUM_COMBOS];
                valid_reach_into(env.board, reach_other, &mut vr);
                let p = payoffs[player];
                for (o, &r) in out.iter_mut().zip(vr.iter()) {
                    *o = p * r;
                }
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_other, player, out),
            NodeKind::Chance { children } => {
                out.fill(0.0);
                let mut cv = scratch.take(NUM_COMBOS);
                for &(c, child) in children {
                    let mut ro = *reach_other;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            ro[h] = 0.0;
                        }
                    }
                    Self::eval_average(env, stores, scratch, child, &ro, player, &mut cv);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            out[h] += cv[h];
                        }
                    }
                }
                scratch.give(cv);
                for x in out.iter_mut() {
                    *x /= 44.0;
                }
            }
            NodeKind::Decision { player: p, store, children, .. } => {
                let a = children.len();
                let mut sigma = scratch.take(a * NUM_COMBOS);
                let mut total = scratch.take(NUM_COMBOS);
                stores[*store].average_into(&mut sigma, &mut total);
                out.fill(0.0);
                let mut cv = scratch.take(NUM_COMBOS);
                for (ai, &child) in children.iter().enumerate() {
                    let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                    if *p == player {
                        Self::eval_average(env, stores, scratch, child, reach_other, player, &mut cv);
                        for ((o, &s), &c) in out.iter_mut().zip(&sigma[span]).zip(cv.iter()) {
                            *o += s * c;
                        }
                    } else {
                        let mut ro = *reach_other;
                        for (r, &s) in ro.iter_mut().zip(&sigma[span]) {
                            *r *= s;
                        }
                        Self::eval_average(env, stores, scratch, child, &ro, player, &mut cv);
                        for (o, &c) in out.iter_mut().zip(cv.iter()) {
                            *o += c;
                        }
                    }
                }
                scratch.give(cv);
                scratch.give(total);
                scratch.give(sigma);
            }
        }
    }

    /// Total nodes in the built public tree (decisions + chance + terminals) —
    /// the per-iteration work unit, so it predicts resolve cost directly.
    pub fn public_node_count(&self) -> usize {
        self.kinds.len()
    }

    /// Run `iters` vectorized CFR iterations (DCFR, **alternating** traverser:
    /// each iteration updates one player's regrets while the other plays its
    /// current strategy — the standard, robustly-converging scheme).
    pub fn run(&mut self, iters: u64) {
        let mut root_value = vec![0.0; NUM_COMBOS];
        for _ in 0..iters {
            self.t += 1;
            // Player 0 first (t=1): with every node still uniform, the traverser
            // trains its responses against an opponent that reaches *all* nodes,
            // and RM⁺ locks those regrets in — the off-path robustness CFR⁺ needs.
            let traverser = ((self.t - 1) % 2) as usize;
            let (reach0, reach1) = (self.reach0, self.reach1);
            let (reach_tr, reach_op) = if traverser == 0 { (reach0, reach1) } else { (reach1, reach0) };
            let env = Env {
                kinds: &self.kinds,
                board: &self.board,
                runout: self.runout.as_ref(),
                prepared: &self.prepared,
                cards: &self.cards,
                    carried: self.carried.as_deref(),
                chooser: self.chooser,
            };
            Self::cfr(
                &env,
                &mut self.stores,
                &mut self.scratch,
                self.root,
                &reach_tr,
                &reach_op,
                traverser,
                self.t,
                &mut root_value,
            );
        }
    }

    /// Counterfactual value vector for `traverser` (per traverser hand) into
    /// `out`, given the traverser's reach `reach_tr` and the opponent's reach
    /// `reach_op`.  Regrets/strategy are updated only at the traverser's own
    /// decision nodes.
    ///
    /// `out` is fully overwritten; callers need not pre-zero it.
    #[allow(clippy::too_many_arguments)]
    fn cfr(
        env: &Env<'_>,
        stores: &mut [NodeStore],
        scratch: &mut Scratch,
        id: usize,
        reach_tr: &[f64; NUM_COMBOS],
        reach_op: &[f64; NUM_COMBOS],
        traverser: usize,
        t: u64,
        out: &mut [f64],
    ) {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                // Traverser's value = reach-weighted showdown over the opponent,
                // on this leaf's pre-sorted complete board.
                let o: &mut [f64; NUM_COMBOS] = out.try_into().expect("value buffer is NUM_COMBOS");
                o.fill(0.0);
                env.prepared[*prep].accumulate(reach_op, *half_pot, o);
            }
            NodeKind::RunoutShowdown { half_pot } => {
                // Turn leaf: the same reach-weighted showdown, averaged over the
                // undealt river (check-down continuation), via the pre-sorted
                // runout table built once for this board.  `evaluate` zeroes.
                let o: &mut [f64; NUM_COMBOS] = out.try_into().expect("value buffer is NUM_COMBOS");
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_op, *half_pot, o);
            }
            NodeKind::Fold { payoffs } => {
                let mut vr = [0.0; NUM_COMBOS];
                valid_reach_into(env.board, reach_op, &mut vr);
                let p = payoffs[traverser];
                for (o, &r) in out.iter_mut().zip(vr.iter()) {
                    *o = p * r;
                }
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_op, traverser, out),
            NodeKind::Chance { children } => {
                // River reveal: mask both reaches per branch, sum, divide by
                // the per-pair-consistent count (44) — see `NodeKind::Chance`.
                out.fill(0.0);
                let mut cv = scratch.take(NUM_COMBOS);
                for &(c, child) in children {
                    let mut rt = *reach_tr;
                    let mut ro = *reach_op;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            rt[h] = 0.0;
                            ro[h] = 0.0;
                        }
                    }
                    Self::cfr(env, stores, scratch, child, &rt, &ro, traverser, t, &mut cv);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            out[h] += cv[h];
                        }
                    }
                }
                scratch.give(cv);
                for x in out.iter_mut() {
                    *x /= 44.0;
                }
            }
            NodeKind::Decision { player, store, children, .. } => {
                let a = children.len();
                let mut sigma = scratch.take(a * NUM_COMBOS);
                let mut total = scratch.take(NUM_COMBOS);
                stores[*store].strategy_into(&mut sigma, &mut total);
                out.fill(0.0);

                if *player == traverser {
                    // Push the traverser's own reach by σ; collect per-action
                    // counterfactual values (action-major, matching the store)
                    // to form regrets.
                    let mut child_v = scratch.take(a * NUM_COMBOS);
                    for (ai, &child) in children.iter().enumerate() {
                        let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                        let mut rt = *reach_tr;
                        for (r, &s) in rt.iter_mut().zip(&sigma[span.clone()]) {
                            *r *= s;
                        }
                        Self::cfr(
                            env,
                            stores,
                            scratch,
                            child,
                            &rt,
                            reach_op,
                            traverser,
                            t,
                            &mut child_v[span.clone()],
                        );
                        for ((o, &s), &c) in
                            out.iter_mut().zip(&sigma[span.clone()]).zip(&child_v[span])
                        {
                            *o += s * c;
                        }
                    }
                    stores[*store].update(&sigma, &child_v, out, reach_tr, t);
                    scratch.give(child_v);
                } else {
                    // Opponent node: push the opponent's reach by σ (folding it
                    // into the counterfactual weight) and sum over actions.
                    let mut cv = scratch.take(NUM_COMBOS);
                    for (ai, &child) in children.iter().enumerate() {
                        let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                        let mut ro = *reach_op;
                        for (r, &s) in ro.iter_mut().zip(&sigma[span]) {
                            *r *= s;
                        }
                        Self::cfr(env, stores, scratch, child, reach_tr, &ro, traverser, t, &mut cv);
                        for (o, &c) in out.iter_mut().zip(cv.iter()) {
                            *o += c;
                        }
                    }
                    scratch.give(cv);
                }
                scratch.give(total);
                scratch.give(sigma);
            }
        }
    }

    /// Emit the deployable average strategy keyed by the explicit `info_key`.
    pub fn into_resolved(self) -> VectorResolved {
        let mut strategy = HashMap::new();
        let mut public_nodes = 0;
        for kind in &self.kinds {
            let NodeKind::Decision { player, store, board, history, children, marker } = kind
            else {
                continue;
            };
            if *marker == MARKER_GADGET {
                // The gadget's Follow/Terminate mix is a solving device, not a
                // deployable strategy — never emitted.
                continue;
            }
            public_nodes += 1;
            let a = children.len();
            let s = &self.stores[*store];
            for h in 0..NUM_COMBOS {
                let total: f64 = (0..a).map(|ai| s.strategy_col(ai)[h] as f64).sum();
                if total <= 0.0 {
                    continue; // unreached hand: defaults to uniform in the oracle
                }
                let key = info_key(*player, combo_cards(h), board, history, *marker);
                strategy.insert(key, (0..a).map(|ai| s.strategy_col(ai)[h] as f64 / total).collect());
            }
        }
        let info_sets = strategy.len();
        VectorResolved { strategy, info_sets, public_nodes }
    }
}
