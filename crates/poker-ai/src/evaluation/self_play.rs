//! Head-to-head match runner for comparing two strategies.
//!
//! Self-play win rate is a weak signal on its own — a strategy can beat a copy
//! of itself and still be exploitable — but a head-to-head match between two
//! *different* strategies is the natural way to ask "is B actually better than
//! A?".  This runs `A` against `B` over many sampled hands and reports A's net
//! win rate with a confidence interval.
//!
//! **Seat alternation.** Position is worth a lot in poker, so a match that left
//! A in one seat would mostly measure positional value, not skill.  Every hand
//! therefore alternates which seat A occupies, and A's payoff is read from
//! whichever seat it sat in.  Averaged over the two seats the positional EV
//! cancels (it sums to zero in a zero-sum game), so the reported number is the
//! skill difference: `0` for equal strategies, `> 0` when A is stronger.
//!
//! The estimate is the raw outcome average; [`super::aivat`] provides the
//! variance-reduced version of the same quantity when a value function exists.

use crate::games::Game;
use crate::solver::best_response::Strategy;
use crate::util::rng::{sample_index, xorshift_next_unit as next_unit};

/// Play one hand where `seat[p]` is the strategy controlling player `p`; return
/// every player's payoff.  Chance and both players are sampled.
fn play_hand<G: Game>(game: &G, seat: [&Strategy; 2], rng: &mut u64) -> [f64; 2] {
    let mut state = game.root();
    loop {
        if game.is_terminal(&state) {
            return [game.utility(&state, 0), game.utility(&state, 1)];
        }
        if game.is_chance(&state) {
            if game.is_chance_enumerable(&state) {
                let outcomes = game.chance_outcomes(&state);
                let probs: Vec<f64> = outcomes.iter().map(|&(_, p)| p).collect();
                let i = sample_index(probs.iter().copied(), next_unit(rng));
                state = outcomes.into_iter().nth(i).unwrap().0;
            } else {
                state = game.sample_chance(&state, || next_unit(rng));
            }
            continue;
        }
        let player = game.current_player(&state);
        let n = game.num_actions(&state);
        let key = game.info_key(&state);
        let strat = seat[player].get(&key).cloned().unwrap_or_else(|| vec![1.0 / n as f64; n]);
        let a = sample_index(strat.iter().copied(), next_unit(rng));
        state = game.apply(&state, a);
    }
}

/// Outcome of a head-to-head match, from strategy A's perspective.
#[derive(Clone, Copy, Debug)]
pub struct MatchResult {
    /// A's mean net payoff per hand (positive ⇒ A beats B), in game utility
    /// units, with positional EV cancelled by seat alternation.
    pub mean: f64,
    /// Standard error of the mean — the ± on the win rate.
    pub stderr: f64,
    /// Hands played.
    pub hands: usize,
}

impl MatchResult {
    /// True when A's win rate is positive beyond `z` standard errors (a `z` of
    /// ~2 is roughly 95% confidence).
    pub fn a_wins(&self, z: f64) -> bool {
        self.mean - z * self.stderr > 0.0
    }
}

/// Play `hands` hands of `strat_a` vs `strat_b`, alternating seats, and report
/// A's win rate with its standard error.  Deterministic for a fixed `seed`.
pub fn run_match<G: Game>(
    game: &G,
    strat_a: &Strategy,
    strat_b: &Strategy,
    hands: usize,
    seed: u64,
) -> MatchResult {
    let mut rng = seed | 1;
    let mut payoffs = Vec::with_capacity(hands);
    for h in 0..hands {
        // Alternate A's seat every hand so positional value cancels.
        let a_seat = h & 1;
        let seat = if a_seat == 0 { [strat_a, strat_b] } else { [strat_b, strat_a] };
        let result = play_hand(game, seat, &mut rng);
        payoffs.push(result[a_seat]);
    }
    let n = hands as f64;
    let mean = payoffs.iter().sum::<f64>() / n;
    let var = payoffs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    MatchResult { mean, stderr: (var / n).sqrt(), hands }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::kuhn::Kuhn;
    use crate::solver::best_response::profile_value;
    use crate::solver::cfr::{Cfr, Variant};

    fn kuhn_equilibrium() -> Strategy {
        let mut s = Cfr::new(Kuhn, Variant::Vanilla);
        s.train(20_000);
        s.average_strategy()
    }

    fn kuhn_uniform() -> Strategy {
        let mut s = Cfr::new(Kuhn, Variant::Vanilla);
        s.train(1);
        s.average_strategy()
            .into_iter()
            .map(|(k, v)| (k, vec![1.0 / v.len() as f64; v.len()]))
            .collect()
    }

    #[test]
    fn equal_strategies_break_even() {
        // A strategy against a copy of itself, with seats alternated, nets ~0.
        let eq = kuhn_equilibrium();
        let r = run_match(&Kuhn, &eq, &eq, 200_000, 1);
        assert!(r.mean.abs() < 3.0 * r.stderr, "self-play should net ~0: {} ± {}", r.mean, r.stderr);
    }

    #[test]
    fn equilibrium_does_not_lose_to_a_weak_opponent() {
        // A Nash strategy is guaranteed ≥ the game value against any opponent, so
        // averaged over seats it cannot lose to uniform-random play.
        let eq = kuhn_equilibrium();
        let unif = kuhn_uniform();
        let r = run_match(&Kuhn, &eq, &unif, 200_000, 2);
        assert!(r.mean > -3.0 * r.stderr, "equilibrium must not lose to uniform: {} ± {}", r.mean, r.stderr);
    }

    #[test]
    fn match_estimate_tracks_the_exact_value() {
        // With A in seat 0 only (no alternation here, computed directly), the
        // sampled mean should track the exact profile value — a sanity check that
        // play_hand samples the tree correctly.
        let eq = kuhn_equilibrium();
        let exact_p0 = profile_value(&Kuhn, &eq, 0);
        let mut rng = 9u64 | 1;
        let mut sum = 0.0;
        let hands = 200_000;
        for _ in 0..hands {
            sum += play_hand(&Kuhn, [&eq, &eq], &mut rng)[0];
        }
        let mean = sum / hands as f64;
        assert!((mean - exact_p0).abs() < 0.01, "sampled mean {mean} vs exact {exact_p0}");
    }
}
