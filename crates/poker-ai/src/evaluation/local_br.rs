//! Sampled best response — exploitability for games whose chance space is too
//! large to enumerate (Lisý & Bowling, the LBR family).
//!
//! Exact best response ([`crate::solver::best_response`]) enumerates the whole
//! tree, including every deal — impossible once chance is a full 52-card deal.
//! This estimator samples chance but is careful about the subtlety that makes
//! best response under imperfect information non-trivial: the responder must
//! commit **one action per information set**, not pick a different action for
//! each sampled deal (that would be a *clairvoyant* responder that sees the
//! opponent's private cards — an over-estimate).
//!
//! It therefore runs in two phases:
//!  1. **Build a greedy policy.** Sample many deals; at each responder info set
//!     accumulate the value of every action (opponent fixed to its blueprint,
//!     responder playing greedily below).  The best action per info set is the
//!     greedy policy `g`.
//!  2. **Evaluate `g`.**  `g` is a single valid per-info-set policy, so its value
//!     against the blueprint is a sound **lower bound** on exploitability; it
//!     approaches the exact best response as phase 1 converges.  When chance is
//!     enumerable (Kuhn, Leduc) phase 2 is an exact tree walk, so the estimate
//!     matches `solver::best_response` — the validation anchor.

use std::collections::HashMap;

use crate::games::Game;

fn next_unit(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
}

/// Running mean of each action's value at a responder info set.
#[derive(Clone)]
struct Q {
    sum: Vec<f64>,
    count: f64,
}

impl Q {
    fn greedy(&self) -> usize {
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for (a, &s) in self.sum.iter().enumerate() {
            if s > best_v {
                best_v = s;
                best = a;
            }
        }
        best
    }
}

/// Phase 1: accumulate action values at the responder's info sets and return the
/// value to the responder of playing greedily here and below.
fn accumulate<G: Game>(
    game: &G,
    state: &G::State,
    traverser: usize,
    strategy: &HashMap<u64, Vec<f64>>,
    q: &mut HashMap<u64, Q>,
    rng: &mut u64,
) -> f64 {
    if game.is_terminal(state) {
        return game.utility(state, traverser);
    }
    if game.is_chance(state) {
        let child = game.sample_chance(state, || next_unit(rng));
        return accumulate(game, &child, traverser, strategy, q, rng);
    }

    let player = game.current_player(state);
    let num_actions = game.num_actions(state);
    if player == traverser {
        let key = game.info_key(state);
        let mut vals = vec![0.0; num_actions];
        for a in 0..num_actions {
            vals[a] = accumulate(game, &game.apply(state, a), traverser, strategy, q, rng);
        }
        let entry = q.entry(key).or_insert_with(|| Q { sum: vec![0.0; num_actions], count: 0.0 });
        for a in 0..num_actions {
            entry.sum[a] += vals[a];
        }
        entry.count += 1.0;
        // Propagate the value of the action that is greedy under accumulated means.
        vals[entry.greedy()]
    } else {
        let key = game.info_key(state);
        let probs = strategy
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);
        let a = sample(&probs, rng);
        accumulate(game, &game.apply(state, a), traverser, strategy, q, rng)
    }
}

fn sample(probs: &[f64], rng: &mut u64) -> usize {
    let r = next_unit(rng);
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Phase 2: exact value to `traverser` of the committed greedy policy `g`
/// against `strategy`.  Chance is enumerated when possible, else sampled.
fn eval_greedy<G: Game>(
    game: &G,
    state: &G::State,
    traverser: usize,
    strategy: &HashMap<u64, Vec<f64>>,
    g: &HashMap<u64, usize>,
    rng: &mut u64,
) -> f64 {
    if game.is_terminal(state) {
        return game.utility(state, traverser);
    }
    if game.is_chance(state) {
        if game.is_chance_enumerable(state) {
            return game
                .chance_outcomes(state)
                .iter()
                .map(|(c, p)| p * eval_greedy(game, c, traverser, strategy, g, rng))
                .sum();
        }
        let child = game.sample_chance(state, || next_unit(rng));
        return eval_greedy(game, &child, traverser, strategy, g, rng);
    }

    let player = game.current_player(state);
    let num_actions = game.num_actions(state);
    if player == traverser {
        let a = g.get(&game.info_key(state)).copied().unwrap_or(0).min(num_actions - 1);
        eval_greedy(game, &game.apply(state, a), traverser, strategy, g, rng)
    } else {
        let key = game.info_key(state);
        let probs = strategy
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);
        (0..num_actions)
            .map(|a| probs[a] * eval_greedy(game, &game.apply(state, a), traverser, strategy, g, rng))
            .sum()
    }
}

/// Best-response value for `traverser` against `strategy`, estimated with
/// `build_iters` greedy-building samples and `eval_iters` evaluation samples
/// (the latter may be 1 when chance is enumerable).
pub fn best_response_value<G: Game>(
    game: &G,
    strategy: &HashMap<u64, Vec<f64>>,
    traverser: usize,
    build_iters: u64,
    eval_iters: u64,
    seed: u64,
) -> f64 {
    let mut rng = seed | 1;
    let root = game.root();

    let mut q: HashMap<u64, Q> = HashMap::new();
    for _ in 0..build_iters {
        accumulate(game, &root, traverser, strategy, &mut q, &mut rng);
    }
    let g: HashMap<u64, usize> = q.iter().map(|(&k, qa)| (k, qa.greedy())).collect();

    let mut total = 0.0;
    for _ in 0..eval_iters {
        total += eval_greedy(game, &root, traverser, strategy, &g, &mut rng);
    }
    total / eval_iters as f64
}

/// Sampled exploitability of `strategy`, `(BR₀ + BR₁) / 2` (NashConv / 2), in the
/// game's utility units.  A lower bound that approaches exact exploitability as
/// `build_iters` grows; equal to exact (to numerical precision) when chance is
/// enumerable and the greedy policy is fully resolved.
pub fn sampled_exploitability<G: Game>(
    game: &G,
    strategy: &HashMap<u64, Vec<f64>>,
    build_iters: u64,
    eval_iters: u64,
    seed: u64,
) -> f64 {
    let br0 = best_response_value(game, strategy, 0, build_iters, eval_iters, seed);
    let br1 =
        best_response_value(game, strategy, 1, build_iters, eval_iters, seed ^ 0x9E37_79B9_7F4A_7C15);
    (br0 + br1) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::leduc::Leduc;
    use crate::solver::best_response::exploitability as exact_exploitability;
    use crate::solver::cfr::{Cfr, Variant};
    use crate::solver::dcfr::Discount;

    #[test]
    fn lower_bounds_and_approaches_exact_on_leduc() {
        // Against a partly-trained (so non-trivially exploitable) Leduc strategy,
        // the sampled BR must be a lower bound that lands close to the exact
        // best response computed by the enumerative solver.
        let mut cfr = Cfr::new(Leduc, Variant::Vanilla);
        cfr.train(150);
        let strat = cfr.average_strategy();

        let exact = exact_exploitability(&Leduc, &strat);
        // Enumerable chance ⇒ eval phase is exact; one eval pass suffices.
        let sampled = sampled_exploitability(&Leduc, &strat, 400_000, 1, 1);

        // The committed greedy policy's value can't beat the true best response,
        // and with enough build samples it lands close to it.
        assert!(exact > 0.02, "fixture should be exploitable, got {exact}");
        assert!(sampled <= exact + 1e-6, "BR of a fixed policy can't exceed exact: {sampled} > {exact}");
        assert!(
            sampled > exact - 0.02,
            "sampled BR {sampled} should approach exact {exact}"
        );
    }

    #[test]
    fn converged_strategy_is_near_zero_exploitable() {
        let mut cfr = Cfr::new(Leduc, Variant::Dcfr(Discount::RECOMMENDED));
        cfr.train(20_000);
        let expl = sampled_exploitability(&Leduc, &cfr.average_strategy(), 200_000, 1, 1);
        assert!(expl < 0.06, "converged Leduc should be near zero, got {expl}");
    }
}
