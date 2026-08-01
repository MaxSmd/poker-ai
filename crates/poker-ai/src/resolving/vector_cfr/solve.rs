//! The CFR⁺ traversal, the evaluation pass, and strategy emission.
//!
//! [`VectorCfr::cfr`] is the training recursion (regrets updated at the
//! traverser's own decision nodes); [`VectorCfr::eval_average`] is its
//! no-update twin used for CFV extraction; [`VectorCfr::into_resolved`] emits
//! the deployable average keyed by the oracle's info key.

use std::collections::HashMap;

use super::keys::{info_key, MARKER_GADGET};
use super::node::{cfv_terminal_values, valid_reach, Env, NodeKind, NodeStore};
use super::{VectorCfr, VectorResolved};
use crate::abstraction::features::combo_cards;
use crate::resolving::belief_state::NUM_COMBOS;

impl VectorCfr {
    /// The opponent's per-hand counterfactual values at the (inner) resolve
    /// root under the emitted **average** profile, conditional on each hand
    /// (bb; zero where its prior reach has no mass) — the carry for the next
    /// continual resolve.  Evaluated below any gadget wrap, i.e. assuming
    /// Follow, matching how the explicit `ContinualResolver` refreshes.
    pub fn opponent_cfvs(&self) -> [f64; NUM_COMBOS] {
        let opp = self.chooser;
        let (reach_opp, reach_me) =
            if opp == 0 { (&self.reach0, &self.reach1) } else { (&self.reach1, &self.reach0) };
        let env = Env {
            kinds: &self.kinds,
            board: &self.board,
            runout: self.runout.as_ref(),
            prepared: &self.prepared,
            cards: &self.cards,
            carried: self.carried.as_deref(),
            chooser: self.chooser,
        };
        let raw = Self::eval_average(&env, &self.stores, self.inner_root, reach_me, opp);
        let mass = valid_reach(&self.board, reach_me);
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
    /// (both seats), weighted by the other seat's reach — the evaluation
    /// (no-update) counterpart of [`cfr`](Self::cfr) used for CFV extraction.
    fn eval_average(
        env: &Env<'_>,
        stores: &[NodeStore],
        id: usize,
        reach_other: &[f64; NUM_COMBOS],
        player: usize,
    ) -> Vec<f64> {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                let mut v = [0.0; NUM_COMBOS];
                env.prepared[*prep].accumulate(reach_other, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::RunoutShowdown { half_pot } => {
                let mut v = [0.0; NUM_COMBOS];
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_other, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::Fold { payoffs } => {
                let vr = valid_reach(env.board, reach_other);
                vr.iter().map(|&r| payoffs[player] * r).collect()
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_other, player),
            NodeKind::Chance { children } => {
                let mut v = vec![0.0; NUM_COMBOS];
                for &(c, child) in children {
                    let mut ro = *reach_other;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            ro[h] = 0.0;
                        }
                    }
                    let cv = Self::eval_average(env, stores, child, &ro, player);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            v[h] += cv[h];
                        }
                    }
                }
                for x in &mut v {
                    *x /= 44.0;
                }
                v
            }
            NodeKind::Decision { player: p, store, children, .. } => {
                let a = children.len();
                let sigma = stores[*store].average();
                let mut v = vec![0.0; NUM_COMBOS];
                for (ai, &child) in children.iter().enumerate() {
                    if *p == player {
                        let cv = Self::eval_average(env, stores, child, reach_other, player);
                        for h in 0..NUM_COMBOS {
                            v[h] += sigma[h * a + ai] * cv[h];
                        }
                    } else {
                        let mut ro = *reach_other;
                        for h in 0..NUM_COMBOS {
                            ro[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::eval_average(env, stores, child, &ro, player);
                        for h in 0..NUM_COMBOS {
                            v[h] += cv[h];
                        }
                    }
                }
                v
            }
        }
    }

    /// Run `iters` vectorized CFR iterations (DCFR, **alternating** traverser:
    /// each iteration updates one player's regrets while the other plays its
    /// current strategy — the standard, robustly-converging scheme).
    /// Total nodes in the built public tree (decisions + chance + terminals) —
    /// the per-iteration work unit, so it predicts resolve cost directly.
    pub fn public_node_count(&self) -> usize {
        self.kinds.len()
    }

    pub fn run(&mut self, iters: u64) {
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
            Self::cfr(&env, &mut self.stores, self.root, &reach_tr, &reach_op, traverser, self.t);
        }
    }

    /// Counterfactual value vector for `traverser` (per traverser hand), given
    /// the traverser's reach `reach_tr` and the opponent's reach `reach_op`.
    /// Regrets/strategy are updated only at the traverser's own decision nodes.
    fn cfr(
        env: &Env<'_>,
        stores: &mut [NodeStore],
        id: usize,
        reach_tr: &[f64; NUM_COMBOS],
        reach_op: &[f64; NUM_COMBOS],
        traverser: usize,
        t: u64,
    ) -> Vec<f64> {
        match &env.kinds[id] {
            NodeKind::Showdown { half_pot, prep } => {
                // Traverser's value = reach-weighted showdown over the opponent,
                // on this leaf's pre-sorted complete board.
                let mut v = [0.0; NUM_COMBOS];
                env.prepared[*prep].accumulate(reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::RunoutShowdown { half_pot } => {
                // Turn leaf: the same reach-weighted showdown, averaged over the
                // undealt river (check-down continuation), via the pre-sorted
                // runout table built once for this board.
                let mut v = [0.0; NUM_COMBOS];
                env.runout
                    .expect("turn resolve must build a runout table for its leaves")
                    .evaluate(reach_op, *half_pot, &mut v);
                v.to_vec()
            }
            NodeKind::Fold { payoffs } => {
                let vr = valid_reach(env.board, reach_op);
                vr.iter().map(|&r| payoffs[traverser] * r).collect()
            }
            NodeKind::CfvTerminal => cfv_terminal_values(env, reach_op, traverser),
            NodeKind::Chance { children } => {
                // River reveal: mask both reaches per branch, sum, divide by
                // the per-pair-consistent count (44) — see `NodeKind::Chance`.
                let mut v = vec![0.0; NUM_COMBOS];
                for &(c, child) in children {
                    let mut rt = *reach_tr;
                    let mut ro = *reach_op;
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] == c || cards[1] == c {
                            rt[h] = 0.0;
                            ro[h] = 0.0;
                        }
                    }
                    let cv = Self::cfr(env, stores, child, &rt, &ro, traverser, t);
                    for (h, cards) in env.cards.iter().enumerate() {
                        if cards[0] != c && cards[1] != c {
                            v[h] += cv[h];
                        }
                    }
                }
                for x in &mut v {
                    *x /= 44.0;
                }
                v
            }
            NodeKind::Decision { player, store, children, .. } => {
                let a = children.len();
                let sigma = stores[*store].strategy();
                let mut v = vec![0.0; NUM_COMBOS];

                if *player == traverser {
                    // Push the traverser's own reach by σ; collect per-action
                    // counterfactual values to form regrets.
                    let mut child_v: Vec<Vec<f64>> = Vec::with_capacity(a);
                    for (ai, &child) in children.iter().enumerate() {
                        let mut rt = *reach_tr;
                        for h in 0..NUM_COMBOS {
                            rt[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::cfr(env, stores, child, &rt, reach_op, traverser, t);
                        for h in 0..NUM_COMBOS {
                            v[h] += sigma[h * a + ai] * cv[h];
                        }
                        child_v.push(cv);
                    }
                    stores[*store].update(&sigma, &child_v, &v, reach_tr, t);
                } else {
                    // Opponent node: push the opponent's reach by σ (folding it
                    // into the counterfactual weight) and sum over actions.
                    for (ai, &child) in children.iter().enumerate() {
                        let mut ro = *reach_op;
                        for h in 0..NUM_COMBOS {
                            ro[h] *= sigma[h * a + ai];
                        }
                        let cv = Self::cfr(env, stores, child, reach_tr, &ro, traverser, t);
                        for h in 0..NUM_COMBOS {
                            v[h] += cv[h];
                        }
                    }
                }
                v
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
                let row = &s.strategy_sum[h * a..h * a + a];
                let total: f64 = row.iter().sum();
                if total <= 0.0 {
                    continue; // unreached hand: defaults to uniform in the oracle
                }
                let key = info_key(*player, combo_cards(h), board, history, *marker);
                strategy.insert(key, row.iter().map(|&x| x / total).collect());
            }
        }
        let info_sets = strategy.len();
        VectorResolved { strategy, info_sets, public_nodes }
    }
}
