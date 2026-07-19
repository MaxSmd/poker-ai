//! Per-opponent marginal hand distributions (Phase 5 resolving).
//!
//! Depth-limited resolving needs a belief over each opponent's hole cards.  The
//! joint over all opponents is intractable, so we keep **independent marginals**
//! — one distribution per opponent — the standard tractable approximation (it
//! introduces some correlation error, e.g. two opponents' marginals can both put
//! mass on the same card, but is what makes resolving feasible).
//!
//! A belief is a distribution over the 1326 two-card combinations.  It is built
//! from the blueprint's range at the resolve root, narrowed by **card removal**
//! (hands sharing a card with the board are impossible), and updated by **Bayes'
//! rule** as the opponent acts: after an observed action, multiply each hand's
//! probability by the likelihood that the opponent would take that action with
//! that hand (read from the blueprint), then renormalize.

// The canonical combo bijection (this module once carried its own first-card-
// major ordering; that duplicate caused the silent misindexing bug recorded in
// `util::combos` and is gone — `BeliefState.probs` is indexed by the one
// crate-wide ordering).
pub use crate::util::combos::{combo_cards, combo_index, NUM_COMBOS};

/// Belief distribution over one opponent's hole cards.
#[derive(Clone, Debug)]
pub struct BeliefState {
    /// Probability of each of the 1326 combinations (sums to 1 when non-empty).
    pub probs: Vec<f64>,
}

impl BeliefState {
    /// Uniform prior over all 1326 combinations.
    pub fn uniform() -> Self {
        Self { probs: vec![1.0 / NUM_COMBOS as f64; NUM_COMBOS] }
    }

    /// A uniform distribution over an explicit list of hands (the rest get zero)
    /// — the usual way to seed a small, tractable resolving range.
    pub fn from_hands(hands: &[[u8; 2]]) -> Self {
        let mut probs = vec![0.0; NUM_COMBOS];
        for &[a, b] in hands {
            probs[combo_index(a, b)] = 1.0;
        }
        let mut s = Self { probs };
        s.normalize();
        s
    }

    /// Zero out every hand that shares a card with `board` (card removal) and
    /// renormalize.  `board` cards are the community cards visible at the resolve
    /// root.
    pub fn remove_board(&mut self, board: &[u8]) {
        let mut used = 0u64;
        for &c in board {
            if c < 52 {
                used |= 1 << c;
            }
        }
        for i in 0..NUM_COMBOS {
            let [a, b] = combo_cards(i);
            if used & (1 << a) != 0 || used & (1 << b) != 0 {
                self.probs[i] = 0.0;
            }
        }
        self.normalize();
    }

    /// Bayesian update from an observed opponent action: `likelihood[i]` is
    /// `P(observed action | opponent holds combo i)` under the blueprint.  The
    /// posterior is `prior · likelihood`, renormalized.
    pub fn update(&mut self, likelihood: &[f64]) {
        assert_eq!(likelihood.len(), NUM_COMBOS, "likelihood must cover all combos");
        for (p, &l) in self.probs.iter_mut().zip(likelihood) {
            *p *= l;
        }
        self.normalize();
    }

    /// Probability assigned to a specific hand.
    pub fn prob(&self, c0: u8, c1: u8) -> f64 {
        self.probs[combo_index(c0, c1)]
    }

    /// Iterate `(hand, probability)` over hands with non-zero mass — the support
    /// the resolver enumerates.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = ([u8; 2], f64)> + '_ {
        self.probs
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p > 0.0)
            .map(|(i, &p)| (combo_cards(i), p))
    }

    /// Number of hands with non-zero probability.
    pub fn support_size(&self) -> usize {
        self.probs.iter().filter(|&&p| p > 0.0).count()
    }

    /// Renormalize to sum 1 (no-op if the distribution is empty).
    fn normalize(&mut self) {
        let total: f64 = self.probs.iter().sum();
        if total > 0.0 {
            for p in &mut self.probs {
                *p /= total;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sums_to_one() {
        let b = BeliefState::uniform();
        assert!((b.probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(b.support_size(), NUM_COMBOS);
    }

    #[test]
    fn card_removal_zeros_conflicting_hands() {
        let mut b = BeliefState::uniform();
        let board = [0u8, 1, 2]; // remove every hand containing card 0, 1, or 2
        b.remove_board(&board);
        assert!((b.probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(b.prob(0, 5), 0.0, "hand with a board card is impossible");
        assert!(b.prob(10, 11) > 0.0, "non-conflicting hand survives");
        // Remaining combos = C(49, 2).
        assert_eq!(b.support_size(), 49 * 48 / 2);
    }

    #[test]
    fn bayes_update_concentrates_mass() {
        // Two hands; the action is twice as likely with the first.
        let mut b = BeliefState::from_hands(&[[10, 11], [20, 21]]);
        assert!((b.prob(10, 11) - 0.5).abs() < 1e-12);

        let mut likelihood = vec![0.0; NUM_COMBOS];
        likelihood[combo_index(10, 11)] = 0.8;
        likelihood[combo_index(20, 21)] = 0.4;
        b.update(&likelihood);

        // Posterior ∝ (0.5·0.8, 0.5·0.4) = (0.4, 0.2) → (2/3, 1/3).
        assert!((b.prob(10, 11) - 2.0 / 3.0).abs() < 1e-9);
        assert!((b.prob(20, 21) - 1.0 / 3.0).abs() < 1e-9);
    }
}
