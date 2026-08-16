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
use std::sync::Mutex;

use rayon::prelude::*;

use super::keys::{info_key, MARKER_GADGET};
use super::node::{cfv_terminal_values, valid_reach_into, Env, NodeKind, NodeStore};
use super::{VectorCfr, VectorResolved};
use crate::abstraction::features::combo_cards;
use crate::resolving::belief_state::NUM_COMBOS;

/// Recursion depth below which sibling subtrees are traversed concurrently.
///
/// Two levels of a betting tree is O(10–40) independent tasks — enough to fill
/// a large box while every task still carries a whole subtree.  Going deeper
/// multiplies the task count into the thousands, where scheduling costs more
/// than the arithmetic; the runout sweep inside a leaf has its own, much wider,
/// parallel split (`PreparedRunout::evaluate`), so the leaves are already busy.
///
/// **That reasoning predates any measurement.**  A `RAYON_NUM_THREADS` sweep on
/// a 128-core box showed the river resolve saturating at ~4–16 threads (10.1
/// ms/iter serial → 3.5 at 128, i.e. 2.3% parallel efficiency), which is the
/// signature of too few tasks *or* of the per-task allocation in the parallel
/// branches — this depth is what distinguishes them.  Override with
/// `POKER_AI_PAR_DEPTH` to sweep it without a rebuild; the value only changes
/// *where* work runs, never the result (sibling subtrees touch disjoint stores
/// and are summed back in fixed action order).
const DEFAULT_PAR_DEPTH: usize = 2;

/// [`DEFAULT_PAR_DEPTH`], or `POKER_AI_PAR_DEPTH` when set.  Read once — the
/// resolve calls this per node, and a `getenv` in that path would itself show
/// up in a profile.
fn par_depth() -> usize {
    use std::sync::OnceLock;
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("POKER_AI_PAR_DEPTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_PAR_DEPTH)
    })
}

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
        stores: &[Mutex<NodeStore>],
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
                Self::store(stores, *store).average_into(&mut sigma, &mut total);
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
                &self.stores,
                &mut self.scratch,
                self.root,
                &reach_tr,
                &reach_op,
                traverser,
                self.t,
                0,
                &mut root_value,
            );
        }
    }

    /// The regret block for a decision node.
    ///
    /// Each store is written by exactly one node, so this lock never actually
    /// contends: it exists so the traversal can hand *disjoint* stores to
    /// concurrent subtree tasks without `unsafe` or index rebasing.  An
    /// uncontended acquire is tens of nanoseconds against a per-node cost in
    /// the tens of microseconds.
    fn store(stores: &[Mutex<NodeStore>], id: usize) -> std::sync::MutexGuard<'_, NodeStore> {
        stores[id].lock().expect("node store mutex poisoned")
    }

    /// Whether every child of a decision node is a depth-cut runout leaf — the
    /// shape [`build_continuation_chooser`](super::VectorCfr) gives a
    /// continuation chooser.  Recognised structurally rather than by
    /// `MARKER_CONTINUATION` so it cannot drift from what the builder emits,
    /// and so any future node with the same shape gets the same fast path.
    fn all_runout_leaves(env: &Env<'_>, children: &[usize]) -> bool {
        children.iter().all(|&c| matches!(env.kinds[c], NodeKind::RunoutShowdown { .. }))
    }

    /// Counterfactual value vector for `traverser` (per traverser hand) into
    /// `out`, given the traverser's reach `reach_tr` and the opponent's reach
    /// `reach_op`.  Regrets/strategy are updated only at the traverser's own
    /// decision nodes.
    ///
    /// `out` is fully overwritten; callers need not pre-zero it.
    ///
    /// ## Parallelism
    ///
    /// Sibling subtrees are independent — they touch disjoint stores — so the
    /// child loops run concurrently while `depth < par_depth()`.  The cutoff
    /// keeps tasks coarse: the root's few actions each carry a large subtree,
    /// whereas parallelising near the leaves would spend more on scheduling
    /// than on arithmetic.
    ///
    /// The result is **bit-identical** to a serial traversal.  Each child
    /// writes only its own slot, and every combination back into `out` runs
    /// serially in child-index order afterwards, so no float addition is ever
    /// re-associated by the thread schedule.
    #[allow(clippy::too_many_arguments)]
    fn cfr(
        env: &Env<'_>,
        stores: &[Mutex<NodeStore>],
        scratch: &mut Scratch,
        id: usize,
        reach_tr: &[f64; NUM_COMBOS],
        reach_op: &[f64; NUM_COMBOS],
        traverser: usize,
        t: u64,
        depth: usize,
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
                if depth < par_depth() {
                    // One independent subtree per live river card — the widest
                    // and most even split in the whole tree, and the reason a
                    // full-river turn resolve parallelises well.  Each branch
                    // zeroes its own blocked combos before returning, so the
                    // fold below is a plain ordered sum.
                    let partials: Vec<Vec<f64>> = children
                        .par_iter()
                        .map(|&(c, child)| {
                            let mut rt = *reach_tr;
                            let mut ro = *reach_op;
                            for (h, cards) in env.cards.iter().enumerate() {
                                if cards[0] == c || cards[1] == c {
                                    rt[h] = 0.0;
                                    ro[h] = 0.0;
                                }
                            }
                            let mut cv = vec![0.0; NUM_COMBOS];
                            let mut local = Scratch::default();
                            Self::cfr(
                                env,
                                stores,
                                &mut local,
                                child,
                                &rt,
                                &ro,
                                traverser,
                                t,
                                depth + 1,
                                &mut cv,
                            );
                            for (h, cards) in env.cards.iter().enumerate() {
                                if cards[0] == c || cards[1] == c {
                                    cv[h] = 0.0;
                                }
                            }
                            cv
                        })
                        .collect();
                    out.fill(0.0);
                    for cv in &partials {
                        for (o, &v) in out.iter_mut().zip(cv.iter()) {
                            *o += v;
                        }
                    }
                } else {
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
                        Self::cfr(
                            env,
                            stores,
                            scratch,
                            child,
                            &rt,
                            &ro,
                            traverser,
                            t,
                            depth + 1,
                            &mut cv,
                        );
                        for (h, cards) in env.cards.iter().enumerate() {
                            if cards[0] != c && cards[1] != c {
                                out[h] += cv[h];
                            }
                        }
                    }
                    scratch.give(cv);
                }
                for x in out.iter_mut() {
                    *x /= 44.0;
                }
            }
            NodeKind::Decision { player, store, children, .. } => {
                let a = children.len();
                let mut sigma = scratch.take(a * NUM_COMBOS);
                let mut total = scratch.take(NUM_COMBOS);
                Self::store(stores, *store).strategy_into(&mut sigma, &mut total);
                out.fill(0.0);

                if *player == traverser {
                    // Push the traverser's own reach by σ; collect per-action
                    // counterfactual values (action-major, matching the store)
                    // to form regrets.
                    let mut child_v = scratch.take(a * NUM_COMBOS);
                    if Self::all_runout_leaves(env, children) {
                        // Continuation chooser, traverser side.  Every child is
                        // a `RunoutShowdown` on this node's board, differing
                        // only by its inflated pot — and this branch passes
                        // `reach_op` through UNCHANGED (only the traverser's own
                        // reach is pushed by σ, and a runout leaf never reads
                        // it).  Since `evaluate` is linear in `half_pot`, the K
                        // children are one sweep and K scalings, not K sweeps.
                        // At K=4 that is the dominant cost of a turn/flop
                        // iteration divided by four, exactly.
                        let mut unit = scratch.take(NUM_COMBOS);
                        {
                            let u: &mut [f64; NUM_COMBOS] = unit
                                .as_mut_slice()
                                .try_into()
                                .expect("value buffer is NUM_COMBOS");
                            env.runout
                                .expect("a runout leaf requires the resolve's runout table")
                                .evaluate(reach_op, 1.0, u);
                        }
                        for (ai, &child) in children.iter().enumerate() {
                            let NodeKind::RunoutShowdown { half_pot } = &env.kinds[child] else {
                                unreachable!("all_runout_leaves checked every child")
                            };
                            let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                            for (c, &v) in child_v[span.clone()].iter_mut().zip(unit.iter()) {
                                *c = half_pot * v;
                            }
                            for ((o, &s), &c) in
                                out.iter_mut().zip(&sigma[span.clone()]).zip(&child_v[span])
                            {
                                *o += s * c;
                            }
                        }
                        scratch.give(unit);
                    } else if depth < par_depth() && a > 1 {
                        // Each action's subtree writes only its own slot of the
                        // action-major `child_v`, and touches only stores that
                        // no sibling can reach.
                        child_v
                            .par_chunks_mut(NUM_COMBOS)
                            .zip(children.par_iter())
                            .zip(sigma.par_chunks(NUM_COMBOS))
                            .for_each(|((cv, &child), sg)| {
                                let mut rt = *reach_tr;
                                for (r, &s) in rt.iter_mut().zip(sg) {
                                    *r *= s;
                                }
                                let mut local = Scratch::default();
                                Self::cfr(
                                    env,
                                    stores,
                                    &mut local,
                                    child,
                                    &rt,
                                    reach_op,
                                    traverser,
                                    t,
                                    depth + 1,
                                    cv,
                                );
                            });
                        for ai in 0..a {
                            let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                            for ((o, &s), &c) in
                                out.iter_mut().zip(&sigma[span.clone()]).zip(&child_v[span])
                            {
                                *o += s * c;
                            }
                        }
                    } else {
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
                                depth + 1,
                                &mut child_v[span.clone()],
                            );
                            for ((o, &s), &c) in
                                out.iter_mut().zip(&sigma[span.clone()]).zip(&child_v[span])
                            {
                                *o += s * c;
                            }
                        }
                    }
                    Self::store(stores, *store).update(&sigma, &child_v, out, reach_tr, t);
                    scratch.give(child_v);
                } else if Self::all_runout_leaves(env, children) {
                    // Continuation chooser, opponent side.  A runout sweep is
                    // linear in the opponent reach as well as in the pot — every
                    // hero value is a fixed blocker/rank-weighted combination of
                    // `opp_reach` entries — and all K children share this node's
                    // board, hence the same weights `M`.  So the sum over
                    // children telescopes into ONE sweep:
                    //
                    //   Σᵢ potᵢ · M · (r_op ∘ σᵢ)  =  M · (r_op ∘ Σᵢ potᵢ σᵢ)
                    //
                    // The traverser-side branch above collapses the same K
                    // sweeps by the pot scalar alone (its reach is common to
                    // every child); this is that argument carried through the
                    // per-action reach push, so a chooser costs one sweep on
                    // BOTH sides of the alternation.
                    let mut blend = [0.0; NUM_COMBOS];
                    for (ai, &child) in children.iter().enumerate() {
                        let NodeKind::RunoutShowdown { half_pot } = &env.kinds[child] else {
                            unreachable!("all_runout_leaves checked every child")
                        };
                        let span = ai * NUM_COMBOS..(ai + 1) * NUM_COMBOS;
                        for (b, &s) in blend.iter_mut().zip(&sigma[span]) {
                            *b += half_pot * s;
                        }
                    }
                    for (b, &r) in blend.iter_mut().zip(reach_op.iter()) {
                        *b *= r;
                    }
                    let o: &mut [f64; NUM_COMBOS] =
                        (&mut *out).try_into().expect("value buffer is NUM_COMBOS");
                    env.runout
                        .expect("a runout leaf requires the resolve's runout table")
                        .evaluate(&blend, 1.0, o);
                } else if depth < par_depth() && a > 1 {
                    // Opponent node, parallel: same reach push per action, but
                    // each subtree returns its own vector so the sum below can
                    // stay in action order.
                    let partials: Vec<Vec<f64>> = children
                        .par_iter()
                        .zip(sigma.par_chunks(NUM_COMBOS))
                        .map(|(&child, sg)| {
                            let mut ro = *reach_op;
                            for (r, &s) in ro.iter_mut().zip(sg) {
                                *r *= s;
                            }
                            let mut cv = vec![0.0; NUM_COMBOS];
                            let mut local = Scratch::default();
                            Self::cfr(
                                env,
                                stores,
                                &mut local,
                                child,
                                reach_tr,
                                &ro,
                                traverser,
                                t,
                                depth + 1,
                                &mut cv,
                            );
                            cv
                        })
                        .collect();
                    for cv in &partials {
                        for (o, &c) in out.iter_mut().zip(cv.iter()) {
                            *o += c;
                        }
                    }
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
                        Self::cfr(
                            env,
                            stores,
                            scratch,
                            child,
                            reach_tr,
                            &ro,
                            traverser,
                            t,
                            depth + 1,
                            &mut cv,
                        );
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
            let s = self.stores[*store].lock().expect("node store mutex poisoned");
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
