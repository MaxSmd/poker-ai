//! Depth-limited search with blueprint continuation strategies at the leaves
//! — the method of Modicum (Brown, Sandholm & Amos, *Depth-Limited Solving for
//! Imperfect-Information Games*, 2018) and Pluribus (2019).
//!
//! ## Why this exists
//!
//! The vectorized resolver ([`crate::resolving::vector_cfr`]) cuts flop and
//! turn resolves at the next card and values the rest of the hand with a
//! stand-in: the opponent picks the pot size per hand.  Replaying 792 logged
//! turn spots showed what that costs — 81% folds to a turn lead where
//! Slumbot folds 43% — because no real player can grow the pot only when
//! ahead.  Here the rest of the hand is instead **played out by the
//! blueprint**: at each leaf both players choose one of [`CONTINUATIONS`]
//! strategies (the blueprint, and the blueprint biased toward folding,
//! calling or raising), and the leaf's value is a rollout under that pair on
//! a sampled runout.  A player cannot exploit the search by switching style
//! after the leaf, because the search already lets it choose among styles.
//!
//! ## Shape
//!
//! * The search runs in the **blueprint's abstract game**, rooted at the
//!   agent's mirror of the real hand, so every leaf is an exact blueprint
//!   node to continue from.  It covers the rest of the current betting round;
//!   a round that closes (or an all-in) is a leaf.
//! * External-sampling Monte Carlo CFR with linear weighting.  Each iteration
//!   samples a hand for each player from its range.  Info sets are per
//!   `(node, exact hole cards)`, so the search refines the blueprint's card
//!   buckets on the current street.
//! * Our own hand is drawn with probability `own_share` (otherwise from our
//!   range), so the hand we must play is trained every other iteration.  The
//!   bias is undone with an importance weight `p(hand) / q(hand)` on every
//!   update, which keeps all regret and average-strategy estimates unbiased —
//!   in particular the opponent still trains against our whole range.
//!
//! * Every decision starts at the **blueprint's strategy** (regrets and the
//!   average seeded from it, scaled by `warm`).  With full ranges most hands
//!   are sampled only a few times per search; seeded, a rarely-sampled hand
//!   plays like the blueprint instead of uniformly at random, and the search
//!   moves away from it only where it has evidence to.
//!
//! Correctness is gated against the explicit-deal [`Subgame`] oracle: with a
//! check-down continuation the search solves the same game as the oracle's
//! check-down leaf, and its strategy must be near-unexploitable there.
//!
//! [`Subgame`]: crate::validation::resolving::subgame::Subgame

use std::collections::HashMap;

use poker_core::action::Action;
use poker_core::state::{GameState, NO_CARD};

use crate::games::blueprint::BlueprintHoldem;
use crate::play::policy::CompactPolicy;
use crate::resolving::belief_state::{combo_cards, combo_index, BeliefState, NUM_COMBOS};
use crate::resolving::vector_cfr::subgame_info_key;
use crate::util::rng::xorshift_next_unit;

/// Continuation strategies offered at a leaf: the blueprint, then the
/// blueprint biased toward folding, toward checking/calling, and toward
/// betting/raising (Pluribus's four).
pub const CONTINUATIONS: usize = 4;

/// Where the rollouts read the blueprint.  [`CompactPolicy`] in play; tests
/// supply fixed strategies.
pub trait ContinuationPolicy: Sync {
    /// The strategy at info set `key` over `actions`.
    fn probs(&self, key: u64, actions: &[Action]) -> Vec<f64>;
}

impl ContinuationPolicy for CompactPolicy {
    fn probs(&self, key: u64, actions: &[Action]) -> Vec<f64> {
        // `get`, not `probs_or_uniform`: rollouts reach rare lines by design,
        // and must not drag down the hit rate that diagnoses the live bot.
        match self.get(key) {
            Some(p) if p.len() == actions.len() => p.iter().map(|&x| x as f64).collect(),
            _ => vec![1.0 / actions.len() as f64; actions.len()],
        }
    }
}

/// Search tunables.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// MCCFR iterations (each runs one traversal per player).
    pub iterations: u64,
    /// Continuation strategies per player at a leaf: `CONTINUATIONS`, or `1`
    /// for the blueprint alone.
    pub continuations: usize,
    /// Factor applied to the favoured actions of a biased continuation before
    /// renormalizing (Pluribus: 5).
    pub bias: f64,
    /// Probability of dealing our actual hand instead of sampling our range.
    pub own_share: f64,
    /// Strength of the blueprint warm start, in pots: each decision's
    /// regrets start at `warm · pot · blueprint probability` (and the average
    /// at the blueprint), so that much evidence must accumulate before the
    /// search departs from the blueprint.  `0` starts uniform.
    pub warm: f64,
    /// Whether the searcher also chooses its continuation at a leaf
    /// (Pluribus) or plays the plain blueprint there while only the opponent
    /// chooses (Modicum's original formulation).  Letting the searcher choose
    /// lets it "plan" to keep raising, which makes aggression now look better.
    pub searcher_chooses: bool,
    pub seed: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            iterations: 2_000,
            continuations: CONTINUATIONS,
            bias: 5.0,
            own_share: 0.5,
            warm: 1.0,
            searcher_chooses: true,
            seed: 1,
        }
    }
}

/// The spot to search: an abstract-game state where the searcher acts.
pub struct SearchRoot<'a> {
    pub game: &'a BlueprintHoldem,
    /// Abstract state at the root.  Unrevealed board cards are `NO_CARD`;
    /// hole cards are ignored (each iteration deals its own).
    pub root: GameState,
    /// Raises already made on the root street (the blueprint's cap counter).
    pub street_raises: u8,
    /// Abstract action history from the start of the hand to the root — the
    /// prefix of every blueprint key a rollout looks up.
    pub history: Vec<u8>,
    /// Both ranges, engine-seat indexed, with board cards already removed.
    pub ranges: [BeliefState; 2],
    /// The searcher's actual hole cards.
    pub hero_hole: [u8; 2],
}

/// What a search returns.
pub struct SearchResult {
    /// The searcher's average strategy at the root for its actual hand.
    pub hero: Vec<f64>,
    /// The same for every hand the search trained (`None` = never sampled),
    /// indexed by combo — the likelihood for updating our own range.
    pub root_strategies: Vec<Option<Vec<f64>>>,
    /// Every trained decision strategy, keyed like the explicit `Subgame`
    /// oracle (`subgame_info_key` with the in-search action path).
    pub strategy: HashMap<u64, Vec<f64>>,
}

enum Kind {
    Decision { player: usize, children: Vec<usize>, width: usize, offset: usize, actions: Vec<Action>, history: Vec<u8> },
    Fold { value: [f64; 2] },
    /// The rest of the hand is a rollout from `gs`.  `offset[p]` locates
    /// player `p`'s continuation-choice regrets (unused when K = 1).
    Leaf { gs: GameState, history: Vec<u8>, street_raises: u8, offset: [usize; 2] },
}

struct Node {
    kind: Kind,
    /// Action indices from the search root.
    path: Vec<u8>,
}

struct Search<'a, P: ContinuationPolicy> {
    game: &'a BlueprintHoldem,
    policy: &'a P,
    cfg: SearchConfig,
    nodes: Vec<Node>,
    regrets: Vec<f64>,
    strat: Vec<f64>,
    board5: [u8; 5],
    big_blind: f64,
    rng: u64,
    /// The searching player's seat.
    hero: usize,
}

/// Run the search and return the searcher's root strategy.
pub fn search<P: ContinuationPolicy>(root: &SearchRoot<'_>, policy: &P, cfg: &SearchConfig) -> SearchResult {
    let hero = root.root.current_player();
    let mut s = Search {
        game: root.game,
        policy,
        cfg: cfg.clone(),
        nodes: Vec::new(),
        regrets: Vec::new(),
        strat: Vec::new(),
        board5: root.root.board,
        big_blind: root.game.big_blind_chips() as f64,
        rng: cfg.seed | 1,
        hero,
    };
    let street = root.root.street;
    s.build(root.root.clone(), root.street_raises, root.history.clone(), Vec::new(), street);
    s.regrets = vec![0.0; s.size()];
    s.strat = vec![0.0; s.size()];
    s.warm_start(&root.ranges, root.root.pot as f64 / s.big_blind);

    let sampler = Sampler::new(&root.ranges, hero, root.hero_hole, cfg.own_share);
    for t in 1..=cfg.iterations {
        for traverser in 0..2 {
            let Some((hands, w)) = sampler.sample(&mut s.rng) else { continue };
            s.walk(0, traverser, hands, w * t as f64);
        }
    }
    s.result(root.hero_hole)
}

impl<P: ContinuationPolicy> Search<'_, P> {
    fn size(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| match &n.kind {
                Kind::Decision { width, offset, .. } => offset + NUM_COMBOS * width,
                Kind::Leaf { offset, .. } => offset[1] + NUM_COMBOS * self.cfg.continuations,
                Kind::Fold { .. } => 0,
            })
            .max()
            .unwrap_or(0)
    }

    fn next_offset(&self) -> usize {
        self.size()
    }

    /// Build the rest of the root street: decisions until a fold (terminal)
    /// or the round closing / an all-in (leaf).
    fn build(&mut self, gs: GameState, sr: u8, history: Vec<u8>, path: Vec<u8>, street: u8) -> usize {
        let id = self.nodes.len();
        if gs.is_terminal() && gs.folded != 0 {
            let p = gs.terminal_payoffs();
            let value = [p[0] as f64 / self.big_blind, p[1] as f64 / self.big_blind];
            self.nodes.push(Node { kind: Kind::Fold { value }, path });
            return id;
        }
        if gs.is_terminal() || gs.street != street {
            let base = self.next_offset();
            let k = self.cfg.continuations;
            let offset = [base, base + NUM_COMBOS * k];
            self.nodes.push(Node { kind: Kind::Leaf { gs, history, street_raises: sr, offset }, path });
            return id;
        }
        let acts = self.game.capped_actions(&gs, sr);
        let width = acts.len();
        let offset = self.next_offset();
        self.nodes.push(Node {
            kind: Kind::Decision {
                player: gs.current_player(),
                children: Vec::new(),
                width,
                offset,
                actions: acts.to_vec(),
                history: history.clone(),
            },
            path: path.clone(),
        });
        let mut children = Vec::with_capacity(width);
        for (i, &a) in acts.iter().enumerate() {
            let mut child = gs.clone();
            let (os, ob) = (child.street, child.current_bet);
            child.apply_action(a);
            let csr = self.game.raises_after(sr, os, ob, &child);
            let mut h = history.clone();
            h.push(i as u8);
            let mut p = path.clone();
            p.push(i as u8);
            children.push(self.build(child, csr, h, p, street));
        }
        if let Kind::Decision { children: c, .. } = &mut self.nodes[id].kind {
            *c = children;
        }
        id
    }

    /// Seed every in-range hand's decisions at the blueprint (see
    /// [`SearchConfig::warm`]).
    fn warm_start(&mut self, ranges: &[BeliefState; 2], pot_bb: f64) {
        if self.cfg.warm <= 0.0 {
            return;
        }
        let visible: Vec<u8> = self.board5.iter().copied().filter(|&c| c != NO_CARD).collect();
        let scale = self.cfg.warm * pot_bb;
        for node in &self.nodes {
            let Kind::Decision { player, width, offset, actions, history, .. } = &node.kind else { continue };
            for (h, &mass) in ranges[*player].probs.iter().enumerate() {
                if mass <= 0.0 {
                    continue;
                }
                let key = self.game.key_for(*player, combo_cards(h), &visible, history);
                let probs = self.policy.probs(key, actions);
                let at = offset + h * width;
                for (a, &p) in probs.iter().enumerate() {
                    self.regrets[at + a] = scale * p;
                    self.strat[at + a] = p;
                }
            }
        }
    }

    fn unit(&mut self) -> f64 {
        xorshift_next_unit(&mut self.rng)
    }

    fn draw(&mut self, dist: &[f64]) -> usize {
        let u = self.unit();
        let mut acc = 0.0;
        for (i, &p) in dist.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        dist.len() - 1
    }

    /// Regret matching over `width` regrets at `at`.
    fn current(&self, at: usize, width: usize) -> Vec<f64> {
        let r = &self.regrets[at..at + width];
        let pos: f64 = r.iter().map(|&x| x.max(0.0)).sum();
        if pos > 0.0 {
            r.iter().map(|&x| x.max(0.0) / pos).collect()
        } else {
            vec![1.0 / width as f64; width]
        }
    }

    /// One external-sampling traversal; returns the traverser's value (bb).
    /// `w` is the importance weight times the linear-CFR iteration weight.
    fn walk(&mut self, id: usize, p: usize, hands: [usize; 2], w: f64) -> f64 {
        match &self.nodes[id].kind {
            Kind::Fold { value } => value[p],
            Kind::Decision { player, children, width, offset, .. } => {
                let (player, width) = (*player, *width);
                let children = children.clone();
                let at = offset + hands[player] * width;
                let sigma = self.current(at, width);
                if player == p {
                    let values: Vec<f64> = children.iter().map(|&c| self.walk(c, p, hands, w)).collect();
                    let v: f64 = sigma.iter().zip(&values).map(|(s, x)| s * x).sum();
                    for (r, x) in self.regrets[at..at + width].iter_mut().zip(&values) {
                        *r += w * (x - v);
                    }
                    v
                } else {
                    for (s, x) in self.strat[at..at + width].iter_mut().zip(&sigma) {
                        *s += w * x;
                    }
                    let a = self.draw(&sigma);
                    self.walk(children[a], p, hands, w)
                }
            }
            Kind::Leaf { offset, .. } => {
                let offset = *offset;
                self.leaf(id, offset, p, hands, w)
            }
        }
    }

    fn leaf(&mut self, id: usize, offset: [usize; 2], p: usize, hands: [usize; 2], w: f64) -> f64 {
        let cards = [combo_cards(hands[0]), combo_cards(hands[1])];
        let board = self.runout(cards);
        let k = self.cfg.continuations;
        let (hero, searcher_chooses) = (self.hero, self.cfg.searcher_chooses);
        let chooses = |pl: usize| k > 1 && (pl != hero || searcher_chooses);
        let o = 1 - p;
        let c_o = if chooses(o) {
            let at_o = offset[o] + hands[o] * k;
            let sigma_o = self.current(at_o, k);
            for (s, x) in self.strat[at_o..at_o + k].iter_mut().zip(&sigma_o) {
                *s += w * x;
            }
            self.draw(&sigma_o) as u8
        } else {
            0
        };
        if !chooses(p) {
            let mut conts = [0u8; 2];
            conts[o] = c_o;
            return self.rollout(id, cards, board, conts, p);
        }
        let at_p = offset[p] + hands[p] * k;
        let sigma_p = self.current(at_p, k);
        let mut values = [0.0; CONTINUATIONS];
        for (c, v) in values.iter_mut().enumerate().take(k) {
            let mut conts = [0u8; 2];
            conts[p] = c as u8;
            conts[o] = c_o;
            // Common random numbers: every continuation sees the same runout.
            *v = self.rollout(id, cards, board, conts, p);
        }
        let v: f64 = sigma_p.iter().zip(&values).map(|(s, x)| s * x).sum();
        for (r, x) in self.regrets[at_p..at_p + k].iter_mut().zip(&values) {
            *r += w * (x - v);
        }
        v
    }

    /// The root board with every unrevealed card dealt from what `cards`
    /// and the board leave.
    fn runout(&mut self, cards: [[u8; 2]; 2]) -> [u8; 5] {
        let mut used = 0u64;
        for c in cards.iter().flatten().chain(self.board5.iter()) {
            if *c != NO_CARD {
                used |= 1 << c;
            }
        }
        let mut board = self.board5;
        for slot in board.iter_mut().filter(|c| **c == NO_CARD) {
            loop {
                let c = (self.unit() * 52.0) as u8;
                if c < 52 && used & (1 << c) == 0 {
                    used |= 1 << c;
                    *slot = c;
                    break;
                }
            }
        }
        board
    }

    /// Play the hand out from leaf `id` with both players following their
    /// continuation strategies; player `p`'s payoff in bb.
    fn rollout(&mut self, id: usize, cards: [[u8; 2]; 2], board: [u8; 5], conts: [u8; 2], p: usize) -> f64 {
        let Kind::Leaf { gs, history, street_raises, .. } = &self.nodes[id].kind else {
            unreachable!("rollout from a non-leaf");
        };
        let mut gs = gs.clone();
        let mut history = history.clone();
        let mut sr = *street_raises;
        gs.hole_cards[0] = cards[0];
        gs.hole_cards[1] = cards[1];
        gs.board = board;
        while !gs.is_terminal() {
            let pl = gs.current_player();
            let acts = self.game.capped_actions(&gs, sr);
            let visible = gs.board_cards_count();
            let key = self.game.key_for(pl, gs.hole_cards[pl], &gs.board[..visible], &history);
            let mut probs = self.policy.probs(key, &acts);
            bias(&mut probs, &acts, conts[pl], self.cfg.bias);
            let i = self.draw(&probs);
            let (os, ob) = (gs.street, gs.current_bet);
            gs.apply_action(acts[i]);
            sr = self.game.raises_after(sr, os, ob, &gs);
            history.push(i as u8);
        }
        gs.terminal_payoffs()[p] as f64 / self.big_blind
    }

    fn result(&self, hero_hole: [u8; 2]) -> SearchResult {
        let avg = |at: usize, width: usize| -> Option<Vec<f64>> {
            let s = &self.strat[at..at + width];
            let total: f64 = s.iter().sum();
            (total > 0.0).then(|| s.iter().map(|x| x / total).collect())
        };
        let mut strategy = HashMap::new();
        for node in &self.nodes {
            let Kind::Decision { player, width, offset, .. } = &node.kind else { continue };
            for h in 0..NUM_COMBOS {
                if let Some(s) = avg(offset + h * width, *width) {
                    strategy.insert(subgame_info_key(*player, combo_cards(h), &self.board5, &node.path), s);
                }
            }
        }
        let Kind::Decision { width, offset, .. } = &self.nodes[0].kind else {
            panic!("search root is not a decision");
        };
        let root_strategies: Vec<Option<Vec<f64>>> = (0..NUM_COMBOS).map(|h| avg(offset + h * width, *width)).collect();
        let mine = combo_index(hero_hole[0], hero_hole[1]);
        let hero_strategy = root_strategies[mine].clone().unwrap_or_else(|| vec![1.0 / *width as f64; *width]);
        SearchResult { hero: hero_strategy, root_strategies, strategy }
    }
}

/// Bias `probs` (over `actions`) toward continuation `c`'s style: `0` leaves
/// the blueprint as is; `1` folds, `2` checks/calls and `3` bets/raises
/// `factor` times more often, renormalized.  A style with no matching legal
/// action leaves the distribution unchanged.
fn bias(probs: &mut [f64], actions: &[Action], c: u8, factor: f64) {
    if c == 0 {
        return;
    }
    for (p, a) in probs.iter_mut().zip(actions) {
        let favoured = match c {
            1 => matches!(a, Action::Fold),
            2 => matches!(a, Action::Check | Action::Call),
            _ => matches!(a, Action::Raise(_) | Action::AllIn),
        };
        if favoured {
            *p *= factor;
        }
    }
    let total: f64 = probs.iter().sum();
    if total > 0.0 {
        for p in probs.iter_mut() {
            *p /= total;
        }
    }
}

/// Deals both hands for an iteration, and the importance weight that undoes
/// dealing our own hand more often than our range would.
struct Sampler {
    hero: usize,
    /// Cumulative distribution of each seat's range over `hands[seat]`.
    hands: [Vec<usize>; 2],
    cdf: [Vec<f64>; 2],
    hero_hand: usize,
    own_share: f64,
    /// `p(h) / q(h)` for each of our hands: the true marginal of our hand in
    /// a card-compatible deal over the probability this sampler draws it.
    weight: Vec<f64>,
}

impl Sampler {
    fn new(ranges: &[BeliefState; 2], hero: usize, hero_hole: [u8; 2], own_share: f64) -> Self {
        let support = |r: &BeliefState| -> (Vec<usize>, Vec<f64>) {
            let mut hands = Vec::new();
            let mut cdf = Vec::new();
            let mut acc = 0.0;
            for (h, &p) in r.probs.iter().enumerate() {
                if p > 0.0 {
                    acc += p;
                    hands.push(h);
                    cdf.push(acc);
                }
            }
            for c in &mut cdf {
                *c /= acc.max(f64::MIN_POSITIVE);
            }
            (hands, cdf)
        };
        let (h0, c0) = support(&ranges[0]);
        let (h1, c1) = support(&ranges[1]);
        let hero_hand = combo_index(hero_hole[0], hero_hole[1]);
        let villain = 1 - hero;

        // C(h): opponent mass compatible with our hand h; true marginal
        // p(h) ∝ r(h)·C(h).
        let r_hero = &ranges[hero].probs;
        let r_vill = &ranges[villain].probs;
        let total_vill: f64 = r_vill.iter().sum();
        let mut card_mass = [0.0; 52];
        for (o, &q) in r_vill.iter().enumerate() {
            if q > 0.0 {
                let [a, b] = combo_cards(o);
                card_mass[a as usize] += q;
                card_mass[b as usize] += q;
            }
        }
        let mut true_p = vec![0.0; NUM_COMBOS];
        for (h, &r) in r_hero.iter().enumerate() {
            if r > 0.0 {
                let [a, b] = combo_cards(h);
                // Inclusion–exclusion: hands sharing a card with ours; the
                // one hand sharing both is our own (counted twice).
                let shared = card_mass[a as usize] + card_mass[b as usize] - r_vill[h];
                true_p[h] = r * (total_vill - shared).max(0.0);
            }
        }
        let z: f64 = true_p.iter().sum();
        let r_total: f64 = r_hero.iter().sum();
        let weight = (0..NUM_COMBOS)
            .map(|h| {
                let q = own_share * f64::from(u8::from(h == hero_hand)) + (1.0 - own_share) * r_hero[h] / r_total;
                if q > 0.0 && z > 0.0 { true_p[h] / z / q } else { 0.0 }
            })
            .collect();
        Self { hero, hands: [h0, h1], cdf: [c0, c1], hero_hand, own_share, weight }
    }

    fn pick(&self, seat: usize, u: f64) -> Option<usize> {
        let cdf = &self.cdf[seat];
        let i = cdf.partition_point(|&c| c < u).min(cdf.len().checked_sub(1)?);
        Some(self.hands[seat][i])
    }

    /// `(hands by seat, importance weight)`, or `None` if no compatible deal
    /// turned up (ranges that share every card).
    fn sample(&self, rng: &mut u64) -> Option<([usize; 2], f64)> {
        let hero_hand = if xorshift_next_unit(rng) < self.own_share {
            self.hero_hand
        } else {
            self.pick(self.hero, xorshift_next_unit(rng))?
        };
        let [a, b] = combo_cards(hero_hand);
        for _ in 0..1000 {
            let o = self.pick(1 - self.hero, xorshift_next_unit(rng))?;
            let [c, d] = combo_cards(o);
            if c != a && c != b && d != a && d != b {
                let mut hands = [0; 2];
                hands[self.hero] = hero_hand;
                hands[1 - self.hero] = o;
                return Some((hands, self.weight[hero_hand]));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::resolving::leaf_eval::CheckdownLeafEval;
    use crate::validation::resolving::subgame::Subgame;
    use crate::validation::solver::best_response::exploitability;
    use poker_core::make_card;
    use poker_core::state::MAX_PLAYERS;

    /// Always check, else call: the rollout of a check-down.
    struct Passive;
    impl ContinuationPolicy for Passive {
        fn probs(&self, _key: u64, actions: &[Action]) -> Vec<f64> {
            let i = actions
                .iter()
                .position(|a| matches!(a, Action::Check))
                .or_else(|| actions.iter().position(|a| matches!(a, Action::Call)))
                .unwrap_or(0);
            let mut p = vec![0.0; actions.len()];
            p[i] = 1.0;
            p
        }
    }

    /// Uniform over the menu.
    struct Uniform;
    impl ContinuationPolicy for Uniform {
        fn probs(&self, _key: u64, actions: &[Action]) -> Vec<f64> {
            vec![1.0 / actions.len() as f64; actions.len()]
        }
    }

    fn turn_board() -> [u8; 5] {
        // A♣ K♦ 9♥ 4♠, river undealt.
        [make_card(12, 0), make_card(11, 1), make_card(7, 2), make_card(2, 3), NO_CARD]
    }

    /// Heads-up public state checked down to `street` with tiny stacks.
    fn public_root(board: [u8; 5], stack: u32, street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [make_card(0, 1), make_card(1, 1)];
        holes[1] = [make_card(0, 2), make_card(1, 2)];
        let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
        while gs.street < street && !gs.is_terminal() {
            let acts = poker_core::legal_actions(&gs);
            gs.apply_action(if acts.contains(&Action::Check) { Action::Check } else { Action::Call });
        }
        gs
    }

    fn duel() -> [BeliefState; 2] {
        [
            BeliefState::from_hands(&[[make_card(12, 1), make_card(12, 2)], [make_card(6, 0), make_card(5, 0)]]),
            BeliefState::from_hands(&[[make_card(8, 0), make_card(8, 1)], [make_card(10, 0), make_card(9, 1)]]),
        ]
    }

    #[test]
    fn bias_scales_the_favoured_actions_and_renormalizes() {
        let acts = [Action::Fold, Action::Call, Action::Raise(10)];
        let mut p = vec![1.0 / 3.0; 3];
        bias(&mut p, &acts, 1, 5.0);
        assert!((p[0] - 5.0 / 7.0).abs() < 1e-12 && (p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let mut p = vec![0.5, 0.5, 0.0];
        bias(&mut p, &acts, 3, 5.0);
        assert_eq!(p, vec![0.5, 0.5, 0.0], "a never-taken action stays at zero");
        let mut p = vec![0.2, 0.3, 0.5];
        bias(&mut p, &acts, 0, 5.0);
        assert_eq!(p, vec![0.2, 0.3, 0.5], "continuation 0 is the blueprint");
    }

    /// The correctness gate.  With a check-down continuation the search's
    /// leaves are worth exactly the oracle's check-down leaf, so both solve
    /// the same turn game — and the search's strategy, scored in the oracle
    /// by exact best response, must be near-optimal.  Our actual hand is
    /// heavily over-sampled (own_share 0.9), so this also gates the
    /// importance weights.  Measured: with them the error shrinks with
    /// iterations (0.075 / 0.010 / 0.0025 bb at 40k / 160k / 640k); without
    /// them it stays at 0.150 bb however long the search runs — the opponent
    /// trains against the wrong range.
    #[test]
    fn search_with_a_checkdown_continuation_matches_the_oracle() {
        let beliefs = duel();
        let root = public_root(turn_board(), 20, 2);
        let me = root.current_player();
        let game = BlueprintHoldem::new(20, 2, 1, 0).with_raise_cap(10);
        let hero_hole = beliefs[me].iter_nonzero().next().unwrap().0;
        let input = SearchRoot {
            game: &game,
            root: root.clone(),
            street_raises: 0,
            history: Vec::new(),
            ranges: beliefs.clone(),
            hero_hole,
        };
        let cfg = SearchConfig { iterations: 160_000, continuations: 1, own_share: 0.9, warm: 0.0, ..Default::default() };
        let res = search(&input, &Passive, &cfg);

        let leaf = CheckdownLeafEval::new();
        let oracle = Subgame::new(root, &beliefs, &leaf);
        let expl = exploitability(&oracle, &res.strategy);
        println!("depth-limited search, check-down continuation: {expl:.4} bb exploitable in the oracle");
        assert!(expl < 0.05, "search should be near-optimal in the oracle game, got {expl}");
        assert!((res.hero.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    /// Full K = 4 search on a deep flop: every trained root strategy is a
    /// distribution, our hand is trained, and the continuation choosers do
    /// not break the tree.
    #[test]
    fn four_continuation_flop_search_emits_valid_strategies() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD];
        let root = public_root(board, 400, 1);
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(2);
        let mut ranges = [BeliefState::uniform(), BeliefState::uniform()];
        for r in &mut ranges {
            r.remove_board(&board[..3]);
        }
        let hero_hole = [make_card(10, 3), make_card(10, 2)];
        let input = SearchRoot { game: &game, root, street_raises: 0, history: Vec::new(), ranges, hero_hole };
        let res = search(&input, &Uniform, &SearchConfig { iterations: 300, ..Default::default() });
        assert!((res.hero.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let trained = res.root_strategies.iter().flatten().count();
        assert!(trained > 50, "a broad range is sampled, got {trained}");
        assert!(res.root_strategies[combo_index(hero_hole[0], hero_hole[1])].is_some(), "our hand is trained");
        for s in res.root_strategies.iter().flatten() {
            assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    /// Warm start: with no iterations, every in-range hand plays exactly the
    /// blueprint; with iterations, a hand moves only as far as evidence
    /// takes it.  The oracle gate also passes from a blueprint start.
    #[test]
    fn warm_start_begins_at_the_blueprint() {
        let beliefs = duel();
        let root = public_root(turn_board(), 20, 2);
        let me = root.current_player();
        let game = BlueprintHoldem::new(20, 2, 1, 0).with_raise_cap(10);
        let hero_hole = beliefs[me].iter_nonzero().next().unwrap().0;
        let input = SearchRoot { game: &game, root: root.clone(), street_raises: 0, history: Vec::new(), ranges: beliefs.clone(), hero_hole };
        let res = search(&input, &Passive, &SearchConfig { iterations: 0, continuations: 1, ..Default::default() });
        let blueprint = Passive.probs(0, &game.capped_actions(&root, 0));
        assert_eq!(res.hero, blueprint, "zero iterations = the blueprint");

        let res = search(&input, &Passive, &SearchConfig { iterations: 160_000, continuations: 1, ..Default::default() });
        let leaf = CheckdownLeafEval::new();
        let expl = exploitability(&Subgame::new(root, &beliefs, &leaf), &res.strategy);
        println!("warm-started search: {expl:.4} bb exploitable in the oracle");
        assert!(expl < 0.05, "a blueprint start must still converge, got {expl}");
    }

    /// With only the opponent choosing, the searcher's continuation is the
    /// blueprint: its choice regrets stay untouched and the search still
    /// emits valid strategies.
    #[test]
    fn opponent_only_continuations_leave_the_searcher_on_the_blueprint() {
        let board = [make_card(12, 0), make_card(11, 1), make_card(7, 2), NO_CARD, NO_CARD];
        let root = public_root(board, 400, 1);
        let hero = root.current_player();
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(2);
        let mut ranges = [BeliefState::uniform(), BeliefState::uniform()];
        for r in &mut ranges {
            r.remove_board(&board[..3]);
        }
        let hero_hole = [make_card(10, 3), make_card(10, 2)];
        let input = SearchRoot { game: &game, root, street_raises: 0, history: Vec::new(), ranges, hero_hole };
        let cfg = SearchConfig { iterations: 200, searcher_chooses: false, ..Default::default() };
        let mut s = Search {
            game: &game,
            policy: &Uniform,
            cfg: cfg.clone(),
            nodes: Vec::new(),
            regrets: Vec::new(),
            strat: Vec::new(),
            board5: input.root.board,
            big_blind: 2.0,
            rng: 1,
            hero,
        };
        s.build(input.root.clone(), 0, Vec::new(), Vec::new(), 1);
        s.regrets = vec![0.0; s.size()];
        s.strat = vec![0.0; s.size()];
        let sampler = Sampler::new(&input.ranges, hero, hero_hole, 0.5);
        for t in 1..=200u64 {
            for p in 0..2 {
                let (hands, w) = sampler.sample(&mut s.rng).unwrap();
                s.walk(0, p, hands, w * t as f64);
            }
        }
        let k = CONTINUATIONS;
        let mut opp_trained = false;
        for node in &s.nodes {
            if let Kind::Leaf { offset, .. } = node.kind {
                let mine = &s.regrets[offset[hero]..offset[hero] + NUM_COMBOS * k];
                assert!(mine.iter().all(|&r| r == 0.0), "searcher's leaf choices must stay untouched");
                opp_trained |= s.regrets[offset[1 - hero]..offset[1 - hero] + NUM_COMBOS * k].iter().any(|&r| r != 0.0);
            }
        }
        assert!(opp_trained, "the opponent still chooses");
        let res = search(&input, &Uniform, &cfg);
        assert!((res.hero.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    /// The importance weights are p/q for the dealing scheme: weighting each
    /// dealt hand by them recovers the true marginal of our hand.
    #[test]
    fn sampler_weights_recover_the_true_hand_distribution() {
        let ranges = duel();
        let hero = 0;
        let hole = [make_card(6, 0), make_card(5, 0)];
        let s = Sampler::new(&ranges, hero, hole, 0.5);
        let mut rng = 7u64;
        let mut seen = HashMap::new();
        let n = 200_000;
        for _ in 0..n {
            let (hands, w) = s.sample(&mut rng).unwrap();
            *seen.entry(hands[hero]).or_insert(0.0) += w / n as f64;
        }
        // No card conflicts between the two ranges, so the true marginal is
        // the range itself: 1/2 each.
        for (_, &mass) in seen.iter() {
            assert!((mass - 0.5).abs() < 0.01, "weighted mass {mass}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::tests_support::*;
    use super::*;

    /// Not a gate: prints the cost of a realistic search (200 bb, full
    /// ranges, K = 4) so the bot's iteration budget can be sized.
    #[test]
    #[ignore = "timing only"]
    fn time_a_full_range_flop_and_turn_search() {
        for (name, board, street) in [
            ("flop", [card(12, 0), card(11, 1), card(7, 2), NO_CARD, NO_CARD], 1),
            ("turn", [card(12, 0), card(11, 1), card(7, 2), card(2, 3), NO_CARD], 2),
        ] {
            let root = checked_down(board, 400, street);
            let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(2);
            let visible = &board[..if street == 1 { 3 } else { 4 }];
            let mut ranges = [BeliefState::uniform(), BeliefState::uniform()];
            for r in &mut ranges {
                r.remove_board(visible);
            }
            let hero_hole = [card(10, 3), card(10, 2)];
            let input = SearchRoot { game: &game, root, street_raises: 0, history: Vec::new(), ranges, hero_hole };
            let t = std::time::Instant::now();
            let iters = 2_000;
            search(&input, &UniformPolicy, &SearchConfig { iterations: iters, ..Default::default() });
            let s = t.elapsed().as_secs_f64();
            println!("{name}: {iters} iterations in {s:.2}s = {:.0} us/iteration", s / iters as f64 * 1e6);
        }
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    use poker_core::state::MAX_PLAYERS;

    pub fn card(rank: u8, suit: u8) -> u8 {
        poker_core::make_card(rank, suit)
    }

    pub fn checked_down(board: [u8; 5], stack: u32, street: u8) -> GameState {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [card(0, 1), card(1, 1)];
        holes[1] = [card(0, 2), card(1, 2)];
        let mut gs = GameState::new(2, 2, 1, [stack; MAX_PLAYERS], holes, board, 0);
        while gs.street < street && !gs.is_terminal() {
            let acts = poker_core::legal_actions(&gs);
            gs.apply_action(if acts.contains(&Action::Check) { Action::Check } else { Action::Call });
        }
        gs
    }

    pub struct UniformPolicy;
    impl ContinuationPolicy for UniformPolicy {
        fn probs(&self, _key: u64, actions: &[Action]) -> Vec<f64> {
            vec![1.0 / actions.len() as f64; actions.len()]
        }
    }
}
