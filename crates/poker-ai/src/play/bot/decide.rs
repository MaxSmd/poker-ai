//! Where the agent actually chooses a move.
//!
//! Three regimes, dispatched by `should_resolve` and the tracker's sync state:
//! the **blueprint** lookup (preflop/flop/turn by default), a full-range
//! **vectorized re-solve** of the real state (river by default, optionally
//! turn/flop), and the **desynced** fallback for spots the abstraction has no
//! node for (an opponent raise past the cap), where hand-vs-range equity
//! against the frozen range is the only well-defined signal left.

use poker_core::action::Action;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::{Bot, DecisionPath, DecisionTrace, HandState};
use crate::games::blueprint::BlueprintHoldem;
use crate::play::equity::equity_vs_range;
use crate::play::protocol::{EventKind, Parsed, BIG_BLIND, SMALL_BLIND, STACK_SIZE};
use crate::play::tracker::{MapOutcome, RealMove};
use super::hole_mask;
use crate::resolving::belief_state::{combo_cards, combo_index, BeliefState, NUM_COMBOS};
use crate::resolving::depth_limited::{search, SearchConfig, SearchRoot};
use crate::resolving::vector_cfr::{
    capped_root_actions, subgame_info_key, ResolveConfig, VectorCfr, VectorResolved,
};
use crate::util::rng::xorshift_next_unit;

impl Bot {

    /// Blueprint decision (preflop / flop / turn, and the river fallback).
    pub(super) fn decide_blueprint(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let facing = parsed.last_bet_size > 0;
        if !hs.hand.expects(&self.game, hs.my_seat, parsed.street) {
            // No abstract node for this spot: the opponent raised past the
            // blueprint's cap.  (Counted in `act`, which sees every decision.)
            hs.pending_self = Some(None);
            return self.decide_desynced(hs, parsed, board);
        }

        let acts = hs.hand.actions(&self.game);
        let key = hs.hand.info_key(&self.game);
        let probs = self.policy.probs_or_uniform(key, acts.len());

        let moves = self.abstract_moves(hs, parsed, &acts, facing);
        let played = self.play_distribution(&probs);
        let idx = self.choose(DecisionPath::Blueprint, parsed.street, &played, &moves);
        let mv = moves[idx];

        self.update_range(hs, hs.my_seat, idx as u8);
        hs.hand.apply(&self.game, idx as u8);
        hs.pending_self = Some(Some(idx as u8));
        mv
    }

    /// Each abstract action at the mirror's node as a real move.
    fn abstract_moves(&self, hs: &HandState, parsed: &Parsed, acts: &[Action], facing: bool) -> Vec<RealMove> {
        let real_min = parsed.last_bet_size.max(BIG_BLIND);
        let remaining = STACK_SIZE - parsed.total_last_bet_to;
        acts.iter()
            .map(|&a| {
                hs.hand.abstract_to_real(
                    &self.game,
                    a,
                    parsed.pot() as f64,
                    parsed.street_last_bet_to,
                    facing,
                    real_min.min(remaining),
                    remaining,
                )
            })
            .collect()
    }

    /// Flop/turn decision by depth-limited search in the blueprint's own game
    /// (see [`crate::resolving::depth_limited`]).  Called only when the mirror
    /// is in sync, so the root is an exact blueprint node.
    pub(super) fn decide_search(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let acts: Vec<Action> = hs.hand.actions(&self.game).iter().copied().collect();
        let seed = (self.unit() * u64::MAX as f64) as u64;
        let state = hs.hand.state();
        let input = SearchRoot {
            game: &self.game,
            root: self.game.game_state(state).clone(),
            street_raises: state.street_raises(),
            history: state.history().to_vec(),
            ranges: resolve_beliefs(hs, board),
            hero_hole: hs.my_hole,
        };
        let cfg = SearchConfig {
            iterations: self.cfg.search_iters,
            continuations: self.cfg.search_continuations,
            bias: self.cfg.search_bias,
            searcher_chooses: self.cfg.search_searcher_chooses,
            seed,
            ..Default::default()
        };
        let result = search(&input, &self.policy, &cfg);

        let moves = self.abstract_moves(hs, parsed, &acts, parsed.last_bet_size > 0);
        let played = self.play_distribution(&result.hero);
        let idx = self.choose(DecisionPath::Search, parsed.street, &played, &moves);

        // Our range follows the strategy the search played for each hand;
        // a hand it never sampled keeps the blueprint's likelihood.
        let gs = hs.hand.gs(&self.game);
        let mut board_mask = 0u64;
        for &c in &gs.board[..gs.board_cards_count()] {
            board_mask |= 1 << c;
        }
        let mut likelihood = vec![0.0; NUM_COMBOS];
        for (h, l) in likelihood.iter_mut().enumerate() {
            if hs.ranges[hs.my_seat].probs[h] <= 0.0 {
                continue;
            }
            let [a, b] = combo_cards(h);
            if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
                continue;
            }
            *l = match &result.root_strategies[h] {
                Some(s) => self.play_distribution(s)[idx],
                None => {
                    let key = hs.hand.key_with_hole(&self.game, [a, b]);
                    self.play_distribution(&self.policy.probs_or_uniform(key, acts.len()))[idx]
                }
            };
        }
        let mut updated = hs.ranges[hs.my_seat].clone();
        updated.update(&likelihood);
        if updated.probs.iter().any(|&p| p > 0.0) {
            hs.ranges[hs.my_seat] = updated;
        }

        // A search carries no CFVs: the next re-solve this hand bootstraps.
        hs.carried_cfvs = None;
        hs.hand.apply(&self.game, idx as u8);
        hs.pending_self = Some(Some(idx as u8));
        moves[idx]
    }

    /// Decision when the abstract game has no node for this spot.
    ///
    /// This happens when the opponent raises past the blueprint's cap: the
    /// abstraction offers no aggressive action, the event cannot be mapped, and
    /// the tracker stops advancing for the rest of the hand.  The blueprint is
    /// unusable here, so fall back on the one thing that is still well defined
    /// — our hand's equity against the opponent's belief range, weighed against
    /// the price we are being laid.
    ///
    /// The range is frozen at the last node we could translate, which is stale
    /// but blueprint-consistent up to that point; the alternative is no
    /// information at all.  We never raise from a desynced state.
    fn decide_desynced(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let me = hs.my_pos as usize;
        let stack = STACK_SIZE - parsed.total_committed[me];
        let to_call = (parsed.street_last_bet_to - parsed.street_committed[me]).min(stack);
        if to_call == 0 {
            self.forced = None;
            self.last_trace = Some(DecisionTrace {
                street: parsed.street,
                path: DecisionPath::Desynced,
                moves: vec![(RealMove::Check, 1.0)],
            });
            return RealMove::Check;
        }

        let opp = &hs.ranges[1 - hs.my_seat];
        let mut rng = self.rng;
        let equity = equity_vs_range(hs.my_hole, board, opp, &mut rng);
        self.rng = rng;

        // Price of the call: chips in versus the pot we would be contesting.
        let odds = to_call as f64 / (parsed.pot() + to_call) as f64;
        let mv = if equity >= odds { RealMove::Call } else { RealMove::Fold };
        // Deterministic, and nothing downstream depends on which move was
        // played, so a forced move is recorded but not needed.
        self.forced = None;
        self.last_trace = Some(DecisionTrace { street: parsed.street, path: DecisionPath::Desynced, moves: vec![(mv, 1.0)] });
        mv
    }

    /// Postflop decision by full-range vectorized re-solve of the real state —
    /// exact to showdown on the river, cut at the undealt street on turn/flop
    /// (runout leaves with K continuations).  Dispatched by [`Self::should_resolve`].
    pub(super) fn decide_resolve(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let root = self.resolve_root(hs, parsed, board);
        let acts = capped_root_actions(&root, self.cfg.river_cap);
        if acts.is_empty() {
            return self.decide_blueprint(hs, parsed, board);
        }

        let beliefs = resolve_beliefs(hs, board);
        let (cfg, iters) = self.resolve_config(board.len());
        let mut solver = VectorCfr::new(&root, &beliefs, &cfg);
        // Continual re-solving: constrain by the opponent CFVs carried from
        // our previous resolve this hand (the first resolve bootstraps).  The
        // carry is refreshed below, once our action is known.
        if self.cfg.continual {
            if let Some(cfvs) = &hs.carried_cfvs {
                solver = solver.with_opponent_gadget(**cfvs);
            }
        }
        solver.run(iters);
        let resolved = solver.resolved();

        let mut hole = hs.my_hole;
        hole.sort_unstable();
        // The root key filters NO_CARD, so a padded board yields the visible
        // cards only — matching the solver's own root emission.
        let mut board5 = [NO_CARD; 5];
        board5[..board.len()].copy_from_slice(board);
        let key = subgame_info_key(hs.my_seat, hole, &board5, &[]);
        let Some(probs) = resolved.strategy.get(&key).cloned() else {
            // Unreached in the resolve (shouldn't happen with the floor):
            // degrade to the blueprint.  Its action is not this resolve's, so
            // no carry describes it — the next resolve bootstraps.
            hs.carried_cfvs = None;
            return self.decide_blueprint(hs, parsed, board);
        };

        let played = self.play_distribution(&probs);
        let facing = parsed.last_bet_size > 0;
        let moves: Vec<RealMove> = acts.iter().map(|&a| root_move(&root, hs.my_seat, a, facing)).collect();
        // AIVAT: value each root action for our hand BEFORE the draw, so the
        // term below is a pure function of the pre-draw state and the draw.
        let values = solver.root_action_values(combo_index(hs.my_hole[0], hs.my_hole[1]));
        let idx = self.choose(DecisionPath::Resolve, parsed.street, &played, &moves);
        if let Some(v) = values {
            let expected: f64 = played.iter().zip(&v).map(|(p, x)| p * x).sum();
            hs.action_luck_bb += v[idx] - expected;
        }

        self.condition_on_resolve(hs, &mut solver, &resolved, &board5, idx);
        let mv = moves[idx];
        self.mirror_own_move(hs, parsed, mv);
        mv
    }

    /// The resolve's tree shape and iteration budget for a board of
    /// `board_len` cards.
    ///
    /// River: exact showdown terminals.  Turn (full-river mode): the river is
    /// dealt as a chance node and the real river betting solved below — exact
    /// to showdown, no leaf model.  Otherwise turn/flop cut at the undealt
    /// street with the K-continuation check-down leaf.
    fn resolve_config(&self, board_len: usize) -> (ResolveConfig, u64) {
        let capped = ResolveConfig::capped(self.cfg.river_cap);
        if board_len == 5 {
            (capped, self.cfg.river_iters)
        } else if board_len == 4 && self.cfg.turn_full_river {
            (ResolveConfig { full_river: true, ..capped }, self.cfg.turn_iters)
        } else {
            // Turn and flop cut at the undealt street, so every leaf pays for a
            // runout sweep once per iteration — 48 completions on the turn and
            // 1176 on the flop.  `runout_sample` is what keeps that inside a
            // live clock; the river arm above never reaches here (a complete
            // board has no runout to sample).
            let cfg = ResolveConfig {
                continuations: self.cfg.continuations.clone(),
                runout_sample: self.cfg.runout_sample,
                ..capped
            };
            (cfg, self.cfg.turn_iters)
        }
    }

    /// Condition everything carried forward on root action `idx`, which we
    /// just played from `resolved`.
    ///
    /// Every later resolve this hand starts from our range, so it must be
    /// conditioned on the strategy we actually played — this resolve's,
    /// purified as sampled — not on the blueprint's likelihood of the nearest
    /// abstract action, which is a different strategy on every re-solved
    /// street.  The carry is taken at the child we chose, with the same
    /// likelihood, so the two describe one strategy.
    fn condition_on_resolve(
        &self,
        hs: &mut HandState,
        solver: &mut VectorCfr,
        resolved: &VectorResolved,
        board5: &[u8; 5],
        idx: usize,
    ) {
        let likelihood = self.resolve_likelihood(resolved, hs.my_seat, board5, idx);
        if self.cfg.continual {
            hs.carried_cfvs = solver.opponent_cfvs_after(idx, &likelihood).map(Box::new);
        }
        let mut updated = hs.ranges[hs.my_seat].clone();
        updated.update(&likelihood);
        if updated.probs.iter().any(|&p| p > 0.0) {
            hs.ranges[hs.my_seat] = updated;
        }
    }

    /// Keep the abstract tracker coherent with what we do: map our real move
    /// back into the abstract game as if observed.  Tracker only — our range
    /// was already conditioned on the resolve.
    fn mirror_own_move(&mut self, hs: &mut HandState, parsed: &Parsed, mv: RealMove) {
        if !hs.hand.expects(&self.game, hs.my_seat, parsed.street) {
            hs.pending_self = Some(None);
            return;
        }
        let kind = match mv {
            RealMove::Fold => EventKind::Fold,
            RealMove::Check => EventKind::Check,
            RealMove::Call => EventKind::Call,
            RealMove::BetTo(n) => EventKind::BetTo(n),
        };
        let mut rng = self.rng;
        let mapped = hs.hand.map_real(
            &self.game,
            kind,
            parsed.pot() as f64,
            parsed.street_last_bet_to as f64,
            &mut || xorshift_next_unit(&mut rng),
        );
        self.rng = rng;
        if let MapOutcome::Index(i) = mapped {
            hs.hand.apply(&self.game, i);
            hs.pending_self = Some(Some(i));
        } else {
            hs.pending_self = Some(None);
        }
    }

    /// Synthesize the real postflop public state (Slumbot chips) as an engine
    /// `GameState` at `parsed.street` — the resolve root.  The board is padded
    /// with `NO_CARD` past the revealed cards (turn/flop), which the solver
    /// reads as the depth cut.
    fn resolve_root(&self, hs: &HandState, parsed: &Parsed, board: &[u8]) -> GameState {
        let mut board5 = [NO_CARD; 5];
        board5[..board.len()].copy_from_slice(board);
        // Placeholder opponent cards (never read by the public-tree solver).
        let mut used = 0u64;
        for &c in board {
            used |= 1 << c;
        }
        used |= 1 << hs.my_hole[0];
        used |= 1 << hs.my_hole[1];
        let mut spare = (0u8..52).filter(|&c| used & (1 << c) == 0);
        let opp_cards = [spare.next().unwrap(), spare.next().unwrap()];
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[hs.my_seat] = hs.my_hole;
        holes[1 - hs.my_seat] = opp_cards;

        let mut gs = GameState::new(2, BIG_BLIND, SMALL_BLIND, [STACK_SIZE; MAX_PLAYERS], holes, board5, 0);
        gs.street = parsed.street;
        for pos in 0..2usize {
            let seat = 1 - pos;
            gs.total_committed[seat] = parsed.total_committed[pos];
            gs.stacks[seat] = STACK_SIZE - parsed.total_committed[pos];
            gs.street_bets[seat] = parsed.street_committed[pos];
        }
        gs.pot = parsed.pot();
        gs.current_bet = parsed.street_committed[0].max(parsed.street_committed[1]);
        gs.min_raise = parsed.last_bet_size.max(BIG_BLIND);
        gs.to_act = hs.my_seat as u8;
        gs.folded = 0;
        gs.allin = 0;
        for seat in 0..2 {
            if gs.stacks[seat] == 0 {
                gs.allin |= 1 << seat;
            }
        }
        let street_fresh = !parsed.events.iter().any(|e| e.street == parsed.street);
        gs.players_to_act = if street_fresh && gs.current_bet == 0 { 2 } else { 1 };
        gs.last_aggressor = if parsed.last_bettor >= 0 {
            (1 - parsed.last_bettor) as u8
        } else {
            gs.to_act
        };
        gs
    }

    /// Pick the root action to play from the purified distribution `played`
    /// (one entry per action, `moves[i]` its real move): the forced move when
    /// [`Bot::act_forced`] set one, otherwise a draw.  Records the trace either
    /// way.
    fn choose(&mut self, path: DecisionPath, street: u8, played: &[f64], moves: &[RealMove]) -> usize {
        self.last_trace = Some(DecisionTrace {
            street,
            path,
            moves: moves.iter().copied().zip(played.iter().copied()).collect(),
        });
        let Some(forced) = self.forced.take() else {
            return self.draw(played);
        };
        let gap = |m: RealMove| match (m, forced) {
            (RealMove::BetTo(a), RealMove::BetTo(b)) => Some(a.abs_diff(b)),
            _ if m == forced => Some(0),
            _ => None,
        };
        // Nearest matching move; among equals, the one we were likelier to play.
        let best = (0..moves.len())
            .filter_map(|i| gap(moves[i]).map(|g| (g, i)))
            .min_by(|a, b| a.0.cmp(&b.0).then(played[b.1].partial_cmp(&played[a.1]).unwrap()));
        match best {
            Some((_, i)) => i,
            None => {
                self.forced_mismatches += 1;
                (0..played.len()).max_by(|&a, &b| played[a].partial_cmp(&played[b]).unwrap()).unwrap_or(0)
            }
        }
    }

    /// The distribution a decision actually draws from:
    /// actions below `purify` dropped and the rest renormalized, or the
    /// argmax alone when purification drops everything.  Exposed separately
    /// because range updates must use what was played, not what was solved.
    pub(super) fn play_distribution(&self, probs: &[f64]) -> Vec<f64> {
        let mut kept: Vec<f64> = probs.iter().map(|&p| if p < self.cfg.purify { 0.0 } else { p }).collect();
        let total: f64 = kept.iter().sum();
        if total <= 0.0 {
            let argmax = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut one_hot = vec![0.0; probs.len()];
            if let Some(p) = one_hot.get_mut(argmax) {
                *p = 1.0;
            }
            return one_hot;
        }
        for p in &mut kept {
            *p /= total;
        }
        kept
    }

    /// Draw an index from an already-normalized distribution.
    fn draw(&mut self, dist: &[f64]) -> usize {
        let u = self.unit();
        let mut acc = 0.0;
        for (i, &p) in dist.iter().enumerate() {
            acc += p;
            if u < acc {
                return i;
            }
        }
        dist.len().saturating_sub(1)
    }

    /// Each of our hands' probability of playing root action `idx` under a
    /// resolve's strategy, purified as it would be played —
    /// the Bayes likelihood for our own range after a re-solved decision.
    /// Hands the resolve never reached get 0: they carry no range mass.
    fn resolve_likelihood(
        &self,
        resolved: &VectorResolved,
        seat: usize,
        board5: &[u8; 5],
        idx: usize,
    ) -> [f64; NUM_COMBOS] {
        let mut out = [0.0; NUM_COMBOS];
        for (h, l) in out.iter_mut().enumerate() {
            let key = subgame_info_key(seat, combo_cards(h), board5, &[]);
            if let Some(probs) = resolved.strategy.get(&key) {
                *l = self.play_distribution(probs).get(idx).copied().unwrap_or(0.0);
            }
        }
        out
    }

    pub fn game(&self) -> &BlueprintHoldem {
        &self.game
    }
}

/// Both ranges as a resolve needs them: board cards removed, the opponent's
/// additionally filtered by our hole cards, and our actual hand floored to
/// nonzero mass so the resolve always covers it.
fn resolve_beliefs(hs: &HandState, board: &[u8]) -> [BeliefState; 2] {
    let mut beliefs = [hs.ranges[0].clone(), hs.ranges[1].clone()];
    for b in &mut beliefs {
        b.remove_board(board);
    }
    beliefs[1 - hs.my_seat].update(&hole_mask(hs.my_hole));
    let mine = &mut beliefs[hs.my_seat];
    let me_idx = combo_index(hs.my_hole[0], hs.my_hole[1]);
    let floor = mine.probs.iter().cloned().fold(0.0, f64::max) * 0.02;
    if mine.probs[me_idx] < floor {
        mine.probs[me_idx] = floor.max(1e-6);
        mine.update(&vec![1.0; NUM_COMBOS]); // renormalize
    }
    beliefs
}

/// Our root `action` in Slumbot's terms.  An engine `Raise(level)` is already
/// the actor's new street-bet level in real chips — identical to Slumbot's
/// `b<level>` semantics — while an all-in that does not exceed the current
/// bet is really a call (or a check when nothing is `facing` us).
fn root_move(root: &GameState, seat: usize, action: Action, facing: bool) -> RealMove {
    match action {
        Action::Fold => RealMove::Fold,
        Action::Check => RealMove::Check,
        Action::Call => RealMove::Call,
        Action::Raise(level) => RealMove::BetTo(level),
        Action::AllIn => {
            let level = root.street_bets[seat] + root.stacks[seat];
            if level > root.current_bet {
                RealMove::BetTo(level)
            } else if facing {
                RealMove::Call
            } else {
                RealMove::Check
            }
        }
    }
}
