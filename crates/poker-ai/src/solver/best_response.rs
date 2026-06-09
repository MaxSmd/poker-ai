//! Exact best response and exploitability for small games.
//!
//! In a two-player zero-sum game the *exploitability* of a strategy profile is
//! the average amount each player can gain by deviating to a best response.
//! Unlike Local Best Response (Phase 4, a sampled lower bound), this is the
//! exact value — feasible only because the validation games are tiny — and it
//! is what proves the solver has actually reached the known equilibrium.
//!
//! ## Why this is not a per-node maximization
//!
//! A best response must commit a single action per *information set*, not per
//! history.  Taking the max action independently at each history would give the
//! responder clairvoyant power it does not have (seeing the opponent's private
//! card) and overstate exploitability.  So the responder's action at an info
//! set `I` is the argmax of the counterfactual value summed over every history
//! in `I`, weighted by the opponent-and-chance reach probability of that
//! history.  Because the games have perfect recall, an info set's optimal
//! action depends only on *deeper* info sets, so the decisions resolve
//! recursively with memoization and no fixpoint iteration.

use std::collections::HashMap;

use crate::games::Game;

/// Average strategy: information-set key → action probabilities.
pub type Strategy = HashMap<u64, Vec<f64>>;

/// Exploitability of `strategy` in `game`, in the same units as the game's
/// utilities (per player: `NashConv / 2`).  Zero at a Nash equilibrium.
pub fn exploitability<G: Game>(game: &G, strategy: &Strategy) -> f64 {
    let v0 = best_response_value(game, 0, strategy);
    let v1 = best_response_value(game, 1, strategy);
    // On the equilibrium profile both best-response values equal the players'
    // equilibrium payoffs, which sum to zero (zero-sum).  Off equilibrium their
    // sum is NashConv ≥ 0; halving gives the per-player exploitability.
    (v0 + v1) / 2.0
}

/// Expected utility for `player` when **both** players follow `strategy`
/// (the on-strategy game value).  Used to check the converged profile against a
/// game's known value, e.g. Kuhn's −1/18.
pub fn profile_value<G: Game>(game: &G, strategy: &Strategy, player: usize) -> f64 {
    fn rec<G: Game>(game: &G, state: &G::State, strategy: &Strategy, player: usize) -> f64 {
        if game.is_terminal(state) {
            return game.utility(state, player);
        }
        if game.is_chance(state) {
            return game
                .chance_outcomes(state)
                .into_iter()
                .map(|(child, prob)| prob * rec(game, &child, strategy, player))
                .sum();
        }
        let num_actions = game.num_actions(state);
        let key = game.info_key(state);
        let strat = strategy
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![1.0 / num_actions as f64; num_actions]);
        (0..num_actions)
            .map(|a| strat[a] * rec(game, &game.apply(state, a), strategy, player))
            .sum()
    }
    rec(game, &game.root(), strategy, player)
}

/// The expected utility `br_player` achieves by playing an exact best response
/// while the opponent follows `strategy`.
pub fn best_response_value<G: Game>(game: &G, br_player: usize, strategy: &Strategy) -> f64 {
    let mut br = BestResponse { game, br_player, strategy, members: HashMap::new(), chosen: HashMap::new() };
    let root = game.root();
    br.collect_members(&root, 1.0);
    br.value_below(&root)
}

struct BestResponse<'a, G: Game> {
    game: &'a G,
    br_player: usize,
    strategy: &'a Strategy,
    /// For each best-response info set: the histories in it, each paired with
    /// the opponent-and-chance reach probability of reaching that history.
    members: HashMap<u64, Vec<(G::State, f64)>>,
    /// Memoized best-response action per info set.
    chosen: HashMap<u64, usize>,
}

impl<'a, G: Game> BestResponse<'a, G> {
    /// Action probabilities the opponent plays at `key` (uniform if unseen).
    fn opp_strategy(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.strategy.get(&key) {
            Some(p) => p.clone(),
            None => vec![1.0 / num_actions as f64; num_actions],
        }
    }

    /// Forward pass: enumerate every best-response decision node, grouped by
    /// info set, recording the opponent-and-chance reach to each.  The reach
    /// deliberately excludes the best-response player's own action probabilities
    /// (counterfactual reach).
    fn collect_members(&mut self, state: &G::State, reach: f64) {
        if self.game.is_terminal(state) {
            return;
        }
        if self.game.is_chance(state) {
            for (child, prob) in self.game.chance_outcomes(state) {
                self.collect_members(&child, reach * prob);
            }
            return;
        }
        let num_actions = self.game.num_actions(state);
        if self.game.current_player(state) == self.br_player {
            let key = self.game.info_key(state);
            self.members.entry(key).or_default().push((state.clone(), reach));
            for a in 0..num_actions {
                let child = self.game.apply(state, a);
                self.collect_members(&child, reach);
            }
        } else {
            let key = self.game.info_key(state);
            let strat = self.opp_strategy(key, num_actions);
            for a in 0..num_actions {
                let child = self.game.apply(state, a);
                self.collect_members(&child, reach * strat[a]);
            }
        }
    }

    /// Expected utility for the best-response player from `state`, with the
    /// responder playing its (recursively determined) best response and the
    /// opponent following `strategy`.
    fn value_below(&mut self, state: &G::State) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, self.br_player);
        }
        if self.game.is_chance(state) {
            let mut value = 0.0;
            for (child, prob) in self.game.chance_outcomes(state) {
                value += prob * self.value_below(&child);
            }
            return value;
        }
        let num_actions = self.game.num_actions(state);
        let key = self.game.info_key(state);
        if self.game.current_player(state) == self.br_player {
            let action = self.decide(key);
            let child = self.game.apply(state, action);
            self.value_below(&child)
        } else {
            let strat = self.opp_strategy(key, num_actions);
            let mut value = 0.0;
            for a in 0..num_actions {
                let child = self.game.apply(state, a);
                value += strat[a] * self.value_below(&child);
            }
            value
        }
    }

    /// Determine (and memoize) the best-response action at info set `key` by
    /// maximizing the reach-weighted counterfactual value over all histories in
    /// the set.  Recurses only into deeper info sets, so there is no cycle.
    fn decide(&mut self, key: u64) -> usize {
        if let Some(&a) = self.chosen.get(&key) {
            return a;
        }
        // Clone the membership list so we can call `&mut self` methods while
        // iterating (the list is not mutated during the decision).
        let members = self.members.get(&key).cloned().unwrap_or_default();
        let num_actions = members
            .first()
            .map(|(s, _)| self.game.num_actions(s))
            .unwrap_or(0);
        let mut action_value = vec![0.0; num_actions];
        for (state, reach) in &members {
            for a in 0..num_actions {
                let child = self.game.apply(state, a);
                action_value[a] += reach * self.value_below(&child);
            }
        }
        let best = (0..num_actions)
            .max_by(|&a, &b| action_value[a].partial_cmp(&action_value[b]).unwrap())
            .unwrap_or(0);
        self.chosen.insert(key, best);
        best
    }
}
