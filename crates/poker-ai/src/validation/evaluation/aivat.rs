//! AIVAT — variance-reduced value estimation (Burch, Schmid, Moravčík, Bowling
//! 2018).
//!
//! Measuring how a strategy actually performs by averaging raw chip outcomes is
//! correct but *noisy*: poker outcomes have huge per-hand variance, so a fair
//! comparison needs an impractical number of hands.  AIVAT subtracts a **control
//! variate** at every chance event and at every decision of the *evaluated*
//! player: the difference between the realized continuation value and its
//! expectation under the known distribution.  Each correction has zero
//! expectation, so the estimator stays **unbiased**, while a good value function
//! cancels most of the variance.
//!
//! ## What this implementation is for
//!
//! In the toy validation games (Kuhn, Leduc) the chance is enumerable, so we can
//! compute the **exact** value function under the profile — `ev` below — and use
//! it as the AIVAT baseline.  With the exact baseline the chance- and
//! evaluated-player-corrections cancel *all* of their variance, leaving only the
//! variance from the *other* player's choices; this is the cleanest possible
//! demonstration that the control variate is wired correctly (same mean as raw
//! sampling, far lower variance).
//!
//! For the real game the value function would instead be the blueprint's own
//! values (an approximate baseline behind the same structure) — the estimator
//! stays unbiased with any baseline; only the variance reduction depends on
//! baseline quality.  That extension is left for when a real blueprint exists.
//!
//! No production path calls this module: it is kept as the **conceptual oracle**
//! for [`crate::play::luck`], which is exactly this estimator restricted to
//! chance-node corrections with the check-down value function — the restriction
//! that makes it computable in a live match.  Read this file to see what the
//! luck adjustment approximates (and what its skipped terms would add).

use crate::games::Game;
use crate::solver::best_response::Strategy;
use crate::util::rng::{sample_index as sample, xorshift_next_unit as next_unit};

/// Strategy probabilities at a state (uniform if the info set is unseen), the
/// same fallback convention as the best-response code.
fn strat_at<G: Game>(game: &G, state: &G::State, profile: &Strategy) -> Vec<f64> {
    let n = game.num_actions(state);
    let key = game.info_key(state);
    profile.get(&key).cloned().unwrap_or_else(|| vec![1.0 / n as f64; n])
}

/// Exact expected utility to **player 0** at `state` when both players follow
/// `profile` — the AIVAT baseline value function.  Requires enumerable chance.
pub fn ev<G: Game>(game: &G, state: &G::State, profile: &Strategy) -> f64 {
    if game.is_terminal(state) {
        return game.utility(state, 0);
    }
    if game.is_chance(state) {
        return game
            .chance_outcomes(state)
            .into_iter()
            .map(|(child, p)| p * ev(game, &child, profile))
            .sum();
    }
    let strat = strat_at(game, state, profile);
    (0..game.num_actions(state)).map(|a| strat[a] * ev(game, &game.apply(state, a), profile)).sum()
}

/// One AIVAT sample (player-0 perspective): play a hand following `profile`,
/// subtracting the control variate at every chance node and every node of
/// `evaluated_player`.  Its expectation equals the true profile value.
fn aivat_sample<G: Game>(game: &G, profile: &Strategy, evaluated: usize, rng: &mut u64) -> f64 {
    fn walk<G: Game>(
        game: &G,
        state: &G::State,
        profile: &Strategy,
        evaluated: usize,
        rng: &mut u64,
        correction: &mut f64,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.utility(state, 0);
        }
        if game.is_chance(state) {
            let outcomes = game.chance_outcomes(state);
            let expected: f64 = outcomes.iter().map(|(c, p)| p * ev(game, c, profile)).sum();
            let idx = sample(outcomes.iter().map(|&(_, p)| p), next_unit(rng));
            // Control variate: realized child value minus its expectation (E = 0).
            *correction += ev(game, &outcomes[idx].0, profile) - expected;
            return walk(game, &outcomes[idx].0, profile, evaluated, rng, correction);
        }
        let strat = strat_at(game, state, profile);
        let player = game.current_player(state);
        let idx = sample(strat.iter().copied(), next_unit(rng));
        if player == evaluated {
            let expected: f64 =
                (0..strat.len()).map(|a| strat[a] * ev(game, &game.apply(state, a), profile)).sum();
            *correction += ev(game, &game.apply(state, idx), profile) - expected;
        }
        walk(game, &game.apply(state, idx), profile, evaluated, rng, correction)
    }

    let mut correction = 0.0;
    let u = walk(game, &game.root(), profile, evaluated, rng, &mut correction);
    u - correction
}

/// One raw (un-corrected) sample of player 0's utility — the naive Monte-Carlo
/// estimator AIVAT improves on.
fn raw_sample<G: Game>(game: &G, profile: &Strategy, rng: &mut u64) -> f64 {
    let mut state = game.root();
    loop {
        if game.is_terminal(&state) {
            return game.utility(&state, 0);
        }
        if game.is_chance(&state) {
            let outcomes = game.chance_outcomes(&state);
            let i = sample(outcomes.iter().map(|&(_, p)| p), next_unit(rng));
            state = outcomes.into_iter().nth(i).unwrap().0;
        } else {
            let strat = strat_at(game, &state, profile);
            let a = sample(strat.iter().copied(), next_unit(rng));
            state = game.apply(&state, a);
        }
    }
}

/// Mean and (population) variance of a per-sample estimator.
fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var)
}

/// Result of an AIVAT evaluation run.
#[derive(Clone, Copy, Debug)]
pub struct AivatEstimate {
    /// AIVAT value estimate (player-0 perspective), in game utility units.
    pub mean: f64,
    /// Standard error of the AIVAT mean — the tightened error bar.
    pub stderr: f64,
    /// Standard error a raw outcome average would have had at the same sample
    /// count, for reference (how much AIVAT tightened the estimate).
    pub raw_stderr: f64,
}

/// Estimate the value of `profile` to player 0 with AIVAT, correcting chance and
/// `evaluated_player`'s decisions, over `samples` hands from `seed`.  Returns the
/// AIVAT mean with its standard error alongside the raw-sampling standard error.
pub fn aivat_value<G: Game>(
    game: &G,
    profile: &Strategy,
    evaluated_player: usize,
    samples: usize,
    seed: u64,
) -> AivatEstimate {
    let mut rng = seed | 1;
    let mut a = Vec::with_capacity(samples);
    let mut r = Vec::with_capacity(samples);
    for _ in 0..samples {
        a.push(aivat_sample(game, profile, evaluated_player, &mut rng));
        r.push(raw_sample(game, profile, &mut rng));
    }
    let (mean, var) = mean_var(&a);
    let (_, raw_var) = mean_var(&r);
    let n = samples as f64;
    AivatEstimate { mean, stderr: (var / n).sqrt(), raw_stderr: (raw_var / n).sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::kuhn::Kuhn;
    use crate::games::leduc::Leduc;
    use crate::solver::best_response::profile_value;
    use crate::solver::cfr::{Cfr, Variant};

    fn kuhn_equilibrium() -> Strategy {
        let mut s = Cfr::new(Kuhn, Variant::Vanilla);
        s.train(20_000);
        s.average_strategy()
    }

    #[test]
    fn ev_matches_profile_value() {
        // The AIVAT baseline value function must equal the exact on-strategy
        // value (the same number `profile_value` computes).
        let profile = kuhn_equilibrium();
        let exact = profile_value(&Kuhn, &profile, 0);
        let v = ev(&Kuhn, &Kuhn.root(), &profile);
        assert!((v - exact).abs() < 1e-12, "ev {v} should equal profile_value {exact}");
    }

    #[test]
    fn aivat_is_unbiased_and_lower_variance_on_kuhn() {
        let profile = kuhn_equilibrium();
        let exact = profile_value(&Kuhn, &profile, 0);

        let est = aivat_value(&Kuhn, &profile, 0, 20_000, 12345);
        // Unbiased: AIVAT mean tracks the exact value within a few standard
        // errors (a proper statistical bound, not an arbitrary constant).
        assert!(
            (est.mean - exact).abs() < 4.0 * est.stderr,
            "AIVAT mean {} vs exact {exact} (stderr {})",
            est.mean,
            est.stderr
        );
        // The whole point: a tighter error bar than raw outcome sampling.
        assert!(
            est.stderr < est.raw_stderr,
            "AIVAT stderr {} should beat raw {}",
            est.stderr,
            est.raw_stderr
        );
    }

    #[test]
    fn aivat_is_unbiased_with_a_non_equilibrium_profile() {
        // Unbiasedness must not depend on the profile being optimal: a uniform
        // random profile is still estimated correctly, just with more residual
        // variance.
        let uniform: Strategy = {
            let mut s = Cfr::new(Kuhn, Variant::Vanilla);
            s.train(1);
            s.average_strategy()
                .into_iter()
                .map(|(k, v)| (k, vec![1.0 / v.len() as f64; v.len()]))
                .collect()
        };
        let exact = profile_value(&Kuhn, &uniform, 0);
        let est = aivat_value(&Kuhn, &uniform, 0, 20_000, 7);
        assert!(
            (est.mean - exact).abs() < 4.0 * est.stderr,
            "AIVAT mean {} vs exact {exact} (stderr {})",
            est.mean,
            est.stderr
        );
        assert!(est.stderr <= est.raw_stderr, "AIVAT should not be worse than raw");
    }

    /// Same guarantees on Leduc (bigger tree, slower) — on demand:
    ///   cargo test -p poker-ai --release -- --ignored aivat
    #[test]
    #[ignore]
    fn aivat_is_unbiased_and_lower_variance_on_leduc() {
        let mut s = Cfr::new(Leduc, Variant::Dcfr(crate::solver::dcfr::Discount::RECOMMENDED));
        s.train(2_000);
        let profile = s.average_strategy();
        let exact = profile_value(&Leduc, &profile, 0);

        let est = aivat_value(&Leduc, &profile, 0, 20_000, 99);
        println!(
            "Leduc AIVAT mean={:.5} stderr={:.5} raw_stderr={:.5} (exact {exact:.5})",
            est.mean, est.stderr, est.raw_stderr
        );
        assert!((est.mean - exact).abs() < 0.01, "AIVAT mean {} vs exact {exact}", est.mean);
        assert!(est.stderr < est.raw_stderr, "AIVAT stderr {} < raw {}", est.stderr, est.raw_stderr);
    }
}
