//! Counterfactual-value extraction (continual re-solving, finding #4).
//!
//! Continual re-solving (DeepStack) carries the opponent's **counterfactual
//! values** (CFVs) forward from one resolve to the next: they become the
//! constraints the [re-solving gadget](crate::resolving::gadget) holds the
//! opponent to, and the quantity the next street is warm-started from.  This
//! module computes them from a resolved [`Subgame`].
//!
//! The CFV of an opponent hand `h` is the value the opponent achieves **with
//! that hand**, averaged over the *other* player's range — i.e. conditioned on
//! holding `h` but not weighted by the opponent's own probability of holding it
//! (the "counterfactual": what `h` is worth, were the opponent to hold it).  In
//! the explicit range-vs-range subgame the per-deal chance children already
//! carry `P(h_me)·P(h_opp)`, so grouping the deal values by the opponent's hand
//! and dividing by that hand's marginal yields exactly this conditional value.

use std::collections::HashMap;

use crate::abstraction::features::combo_index;
use crate::games::Game;
use crate::resolving::belief_state::NUM_COMBOS;
use crate::resolving::subgame::Subgame;

/// Expected utility of `player` from `node` when both players follow `strategy`
/// (uniform at any information set `strategy` does not cover).  The per-node
/// generalization of [`crate::solver::best_response::profile_value`] — usable
/// from any subtree root, over any [`Game`] (the plain subgame *or* the gadget).
pub(crate) fn node_value<G: Game>(
    game: &G,
    node: &G::State,
    strategy: &HashMap<u64, Vec<f64>>,
    player: usize,
) -> f64 {
    if game.is_terminal(node) {
        return game.utility(node, player);
    }
    if game.is_chance(node) {
        return game
            .chance_outcomes(node)
            .into_iter()
            .map(|(child, prob)| prob * node_value(game, &child, strategy, player))
            .sum();
    }
    let n = game.num_actions(node);
    let key = game.info_key(node);
    let sigma = match strategy.get(&key) {
        Some(s) if s.len() == n => s.clone(),
        _ => vec![1.0 / n as f64; n],
    };
    (0..n)
        .filter(|&a| sigma[a] != 0.0)
        .map(|a| sigma[a] * node_value(game, &game.apply(node, a), strategy, player))
        .sum()
}

/// Per-hand counterfactual values for `opp` in a resolved `subgame` under
/// `strategy`, indexed by [`combo_index`].  Entry `i` is `opp`'s value
/// conditioned on holding combo `i`, averaged over the other player's
/// (card-removal-restricted) range; `NaN` for hands the range never deals.
/// Values are in big blinds (the subgame's utility unit).
///
/// Carry these forward to constrain the next resolve's gadget and to warm-start
/// it (see [`crate::resolving::continual`]).
pub fn opponent_cfvs(
    subgame: &Subgame,
    strategy: &HashMap<u64, Vec<f64>>,
    opp: usize,
) -> Vec<f64> {
    let mut num = vec![0.0f64; NUM_COMBOS];
    let mut den = vec![0.0f64; NUM_COMBOS];
    for (node, prob) in subgame.outcomes() {
        let h = node.hole_cards(opp).expect("deal-rooted node has hole cards");
        let ci = combo_index(h[0], h[1]);
        num[ci] += prob * node_value(subgame, node, strategy, opp);
        den[ci] += prob;
    }
    (0..NUM_COMBOS).map(|i| if den[i] > 0.0 { num[i] / den[i] } else { f64::NAN }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolving::belief_state::BeliefState;
    use crate::resolving::leaf_eval::CheckdownLeafEval;
    use crate::resolving::subgame::{SubgameSolver, Subgame};
    use crate::solver::best_response::profile_value;
    use poker_core::make_card;
    use poker_core::legal_actions;
    use poker_core::action::Action;
    use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

    fn river_board() -> [u8; 5] {
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), make_card(0, 0)]
    }

    fn public_root(board: [u8; 5], stack: u32, target_street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        let mut used = 0u64;
        for &c in &board {
            if c != NO_CARD {
                used |= 1 << c;
            }
        }
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        holes[0] = [spare.next().unwrap(), spare.next().unwrap()];
        holes[1] = [spare.next().unwrap(), spare.next().unwrap()];
        let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
        while gs.street < target_street && !gs.is_terminal() {
            let acts = legal_actions(&gs);
            let act = if acts.contains(&Action::Check) { Action::Check } else { Action::Call };
            gs.apply_action(act);
        }
        gs
    }

    fn duel_ranges() -> (BeliefState, BeliefState) {
        let b0 = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trips (nuts-ish)
            [make_card(6, 0), make_card(5, 0)],   // air
        ]);
        let b1 = BeliefState::from_hands(&[
            [make_card(8, 0), make_card(8, 1)],   // bluff-catcher
            [make_card(10, 0), make_card(9, 1)],  // weak
        ]);
        (b0, b1)
    }

    #[test]
    fn cfvs_are_consistent_and_zero_sum() {
        // The opponent's range-reach-weighted mean CFV equals its overall value
        // under the profile, which is the negative of our value (zero-sum).
        let (b0, b1) = duel_ranges();
        let leaf = CheckdownLeafEval::new();
        let resolved = SubgameSolver::new(1, 0)
            .solve_for_iters(&public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf, 3_000);

        let sg = Subgame::new(public_root(river_board(), 20, 3), &[b0.clone(), b1.clone()], &leaf);
        let cfv1 = opponent_cfvs(&sg, &resolved.strategy, 1);

        // Reconstruct the marginal of each opp hand from the deals to weight CFVs.
        let mut marg = vec![0.0f64; NUM_COMBOS];
        let mut total = 0.0;
        for (node, prob) in sg.outcomes() {
            let h = node.hole_cards(1).unwrap();
            marg[combo_index(h[0], h[1])] += prob;
            total += prob;
        }
        let weighted: f64 = (0..NUM_COMBOS)
            .filter(|&i| !cfv1[i].is_nan())
            .map(|i| marg[i] / total * cfv1[i])
            .sum();

        let v_opp = profile_value(&sg, &resolved.strategy, 1);
        let v_me = profile_value(&sg, &resolved.strategy, 0);
        assert!((weighted - v_opp).abs() < 1e-9, "Σ marg·CFV {weighted} should equal opp value {v_opp}");
        assert!((v_opp + v_me).abs() < 1e-9, "values must be zero-sum");
    }

    #[test]
    fn stronger_opponent_hand_has_higher_cfv() {
        // On the river, a strong opponent hand is worth more (higher CFV) than a
        // weak one against the same range.
        let me = BeliefState::from_hands(&[
            [make_card(10, 0), make_card(9, 0)], // medium
            [make_card(6, 1), make_card(5, 1)],  // air
        ]);
        let opp = BeliefState::from_hands(&[
            [make_card(12, 1), make_card(12, 2)], // trip aces (strong)
            [make_card(6, 0), make_card(4, 0)],   // air (weak)
        ]);
        let leaf = CheckdownLeafEval::new();
        let resolved = SubgameSolver::new(1, 0)
            .solve_for_iters(&public_root(river_board(), 20, 3), &[me.clone(), opp.clone()], &leaf, 3_000);
        let sg = Subgame::new(public_root(river_board(), 20, 3), &[me, opp], &leaf);
        let cfv = opponent_cfvs(&sg, &resolved.strategy, 1);

        let strong = cfv[combo_index(make_card(12, 1), make_card(12, 2))];
        let weak = cfv[combo_index(make_card(6, 0), make_card(4, 0))];
        assert!(!strong.is_nan() && !weak.is_nan());
        assert!(strong > weak, "trip aces CFV {strong} should exceed air CFV {weak}");
    }
}
