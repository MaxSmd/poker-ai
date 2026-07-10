//! The playing agent: blueprint policy + belief tracking + river re-solving.
//!
//! Decision architecture (Pluribus-style split, sized to what a single machine
//! can compute in real time at 200 bb):
//!
//! * **Preflop / flop / turn** — play directly from the trained blueprint:
//!   translate the real hand into the abstract game ([`AbstractHand`]), look up
//!   the average strategy at the info key, purify + sample.
//! * **River** — full-range re-solve of the *actual* public state with the
//!   vectorized public-tree CFR ([`solve_vectorized_capped`]): the resolve root
//!   carries the real pot and stacks (off-tree bets included, so translation
//!   distortion vanishes exactly where the money is deepest), our range and
//!   the opponent's range are carried Bayes updates of the blueprint
//!   (`P(observed abstract action | hand)` at every prior decision), and the
//!   resolved root distribution is played directly in real chips.
//!
//! Both ranges are maintained per hand; the opponent's is additionally
//! filtered by card removal (board + our own hole cards).

use poker_core::action::Action;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use crate::games::blueprint::BlueprintHoldem;
use crate::play::equity::equity_vs_range;
use crate::play::policy::CompactPolicy;
use crate::play::protocol::{parse_action, Event, EventKind, Parsed, BIG_BLIND, SMALL_BLIND, STACK_SIZE};
use crate::play::tracker::{AbstractHand, MapOutcome, RealMove};
use crate::resolving::belief_state::{combo_cards, combo_index, BeliefState, NUM_COMBOS};
use crate::resolving::vector_cfr::{capped_root_actions, solve_vectorized_capped, subgame_info_key};

/// Tunables for the playing agent.
#[derive(Clone, Debug)]
pub struct BotConfig {
    /// Re-solve river decisions (recommended); otherwise blueprint throughout.
    pub resolve_river: bool,
    /// CFR⁺ iterations per river resolve.
    pub river_iters: u64,
    /// Raise cap inside a river resolve (bounds the public tree).
    pub river_cap: u32,
    /// Purification: drop actions below this probability, renormalize, then
    /// sample (`0.0` = sample the raw mixed strategy).
    pub purify: f64,
    /// Seed for the agent's action sampling and bet-mapping randomization.
    pub seed: u64,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self { resolve_river: true, river_iters: 1_500, river_cap: 3, purify: 0.1, seed: 1 }
    }
}

/// Per-hand state.
pub struct HandState {
    /// Slumbot position (0 = big blind, 1 = small blind).
    my_pos: u8,
    /// Engine seat (0 = small blind / button, 1 = big blind) = `1 − my_pos`.
    my_seat: usize,
    my_hole: [u8; 2],
    hand: AbstractHand,
    /// Blueprint-consistent hand distributions, engine-seat indexed.
    ranges: [BeliefState; 2],
    /// Events already consumed from the (cumulative) action string.
    processed: usize,
    /// Board cards already applied to the abstract state.
    board_seen: usize,
    /// Our next echoed event was already applied at decision time:
    /// `Some(Some(i))` = applied index `i`; `Some(None)` = deliberately skipped
    /// (no abstract node existed).  `None` = nothing pending.
    pending_self: Option<Option<u8>>,
}

/// The playing agent. One instance plays many hands (per-hand state lives in
/// [`HandState`]); it owns the abstract game and the blueprint policy.
pub struct Bot {
    game: BlueprintHoldem,
    policy: CompactPolicy,
    cfg: BotConfig,
    rng: u64,
}

impl Bot {
    pub fn new(game: BlueprintHoldem, policy: CompactPolicy, cfg: BotConfig) -> Self {
        let rng = cfg.seed | 1;
        Self { game, policy, cfg, rng }
    }

    /// xorshift64* uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Begin a hand as `client_pos` (Slumbot convention) holding `hole`.
    pub fn start_hand(&mut self, client_pos: u8, hole: [u8; 2]) -> HandState {
        let my_seat = 1 - client_pos as usize;
        let mut ranges = [BeliefState::uniform(), BeliefState::uniform()];
        // The opponent can never hold our cards — filter them out up front.
        let mut mask = vec![1.0; NUM_COMBOS];
        for (i, m) in mask.iter_mut().enumerate() {
            let [a, b] = combo_cards(i);
            if a == hole[0] || a == hole[1] || b == hole[0] || b == hole[1] {
                *m = 0.0;
            }
        }
        ranges[1 - my_seat].update(&mask);
        HandState {
            my_pos: client_pos,
            my_seat,
            my_hole: hole,
            hand: AbstractHand::new(&self.game, my_seat, hole),
            ranges,
            processed: 0,
            board_seen: 0,
            pending_self: None,
        }
    }

    /// Consume the server's cumulative view (`action` string + `board` in
    /// engine encoding) and produce our next move as a wire increment
    /// (`"k" | "c" | "f" | "b<N>"`).  Call only when it is our turn.
    pub fn act(&mut self, hs: &mut HandState, action_str: &str, board: &[u8]) -> Result<String, String> {
        let parsed = parse_action(action_str)?;
        self.sync(hs, &parsed, board);

        if parsed.next_pos != hs.my_pos as i8 {
            return Err(format!(
                "act() called but next to act is {} (we are {})",
                parsed.next_pos, hs.my_pos
            ));
        }

        let mv = if parsed.street == 3 && self.cfg.resolve_river && board.len() == 5 {
            self.decide_river(hs, &parsed, board)
        } else {
            self.decide_blueprint(hs, &parsed, board)
        };
        Ok(mv.to_incr())
    }

    /// Bring the abstract state, board, and ranges up to date with the
    /// server's cumulative view (also used at hand end to observe the final
    /// actions, though ranges then no longer matter).
    pub fn sync(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) {
        if board.len() != hs.board_seen {
            hs.hand.set_board(&self.game, board, hs.my_seat);
            hs.board_seen = board.len();
            // Card removal: hands overlapping the revealed board are dead.
            // Doing this at every reveal (not just at the river resolve) is
            // load-bearing — the likelihood loop must never compute a key for
            // a combo that shares a card with the board.
            for r in &mut hs.ranges {
                r.remove_board(board);
            }
        }
        let events: Vec<Event> = parsed.events[hs.processed..].to_vec();
        for ev in events {
            hs.processed += 1;
            self.consume(hs, ev);
        }
    }

    /// Fold one observed event into the abstract state and the actor's range.
    fn consume(&mut self, hs: &mut HandState, ev: Event) {
        let seat = 1 - ev.pos as usize;

        // Our own echoed action: already applied (or deliberately skipped) at
        // decision time.
        if ev.pos == hs.my_pos {
            if let Some(pending) = hs.pending_self.take() {
                let _ = pending; // applied (or skipped) when we decided
                return;
            }
        }

        // Desync guard: only translate events the abstract game has a node for.
        if !hs.hand.expects(&self.game, seat, ev.street) {
            return;
        }

        let mut rng = self.rng;
        let mut unit = || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mapped = hs.hand.map_real(
            &self.game,
            ev.kind,
            ev.pot_before as f64,
            ev.bet_before as f64,
            &mut unit,
        );
        self.rng = rng;

        if let MapOutcome::Index(idx) = mapped {
            self.update_range(hs, seat, idx);
            hs.hand.apply(&self.game, idx);
        }
    }

    /// Bayes update of `seat`'s range from its observed abstract action at the
    /// *current* (pre-action) abstract node: multiply each hand's probability
    /// by the blueprint's likelihood of the action with that hand.
    fn update_range(&self, hs: &mut HandState, seat: usize, action_index: u8) {
        let n = hs.hand.actions(&self.game).len();
        // Combos sharing a card with the visible board must never reach the
        // hand indexer (a duplicated card yields a garbage canonical index).
        // They carry zero range mass after card removal; this mask is the
        // defensive second line.
        let gs = hs.hand.gs(&self.game);
        let mut board_mask = 0u64;
        for &c in &gs.board[..gs.board_cards_count()] {
            board_mask |= 1 << c;
        }
        let mut likelihood = vec![1.0; NUM_COMBOS];
        for (i, l) in likelihood.iter_mut().enumerate() {
            if hs.ranges[seat].probs[i] <= 0.0 {
                continue; // dead hand: skip the (costly) key computation
            }
            let [a, b] = combo_cards(i);
            if board_mask & (1 << a) != 0 || board_mask & (1 << b) != 0 {
                *l = 0.0;
                continue;
            }
            let key = hs.hand.key_with_hole(&self.game, [a, b]);
            *l = self.policy.probs_or_uniform(key, n)[action_index as usize];
        }
        hs.ranges[seat].update(&likelihood);
    }

    /// Blueprint decision (preflop / flop / turn, and the river fallback).
    fn decide_blueprint(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let facing = parsed.last_bet_size > 0;
        if !hs.hand.expects(&self.game, hs.my_seat, parsed.street) {
            hs.pending_self = Some(None);
            return self.decide_desynced(hs, parsed, board);
        }

        let acts = hs.hand.actions(&self.game);
        let key = hs.hand.info_key(&self.game);
        let probs = self.policy.probs_or_uniform(key, acts.len());
        let idx = self.sample(&probs);

        let real_min = parsed.last_bet_size.max(BIG_BLIND);
        let remaining = STACK_SIZE - parsed.total_last_bet_to;
        let mv = hs.hand.abstract_to_real(
            &self.game,
            acts[idx],
            parsed.pot() as f64,
            parsed.street_last_bet_to,
            facing,
            real_min.min(remaining),
            remaining,
        );

        self.update_range(hs, hs.my_seat, idx as u8);
        hs.hand.apply(&self.game, idx as u8);
        hs.pending_self = Some(Some(idx as u8));
        mv
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
            return RealMove::Check;
        }

        let opp = &hs.ranges[1 - hs.my_seat];
        let mut rng = self.rng;
        let equity = equity_vs_range(hs.my_hole, board, opp, &mut rng);
        self.rng = rng;

        // Price of the call: chips in versus the pot we would be contesting.
        let odds = to_call as f64 / (parsed.pot() + to_call) as f64;
        if equity >= odds {
            RealMove::Call
        } else {
            RealMove::Fold
        }
    }

    /// River decision by full-range vectorized re-solve of the real state.
    fn decide_river(&mut self, hs: &mut HandState, parsed: &Parsed, board: &[u8]) -> RealMove {
        let root = self.river_root(hs, parsed, board);
        let acts = capped_root_actions(&root, self.cfg.river_cap);
        if acts.is_empty() {
            return self.decide_blueprint(hs, parsed, board);
        }

        // Ranges: card removal, opponent additionally filtered by our cards,
        // and our actual hand floored to nonzero mass so the resolve always
        // covers it.
        let opp_seat = 1 - hs.my_seat;
        let mut beliefs = [hs.ranges[0].clone(), hs.ranges[1].clone()];
        for b in &mut beliefs {
            b.remove_board(board);
        }
        let mut mask = vec![1.0; NUM_COMBOS];
        for (i, m) in mask.iter_mut().enumerate() {
            let [a, b] = crate::resolving::belief_state::combo_cards(i);
            if a == hs.my_hole[0] || a == hs.my_hole[1] || b == hs.my_hole[0] || b == hs.my_hole[1] {
                *m = 0.0;
            }
        }
        beliefs[opp_seat].update(&mask);
        let me_idx = combo_index(hs.my_hole[0], hs.my_hole[1]);
        let floor = beliefs[hs.my_seat].probs.iter().cloned().fold(0.0, f64::max) * 0.02;
        if beliefs[hs.my_seat].probs[me_idx] < floor {
            beliefs[hs.my_seat].probs[me_idx] = floor.max(1e-6);
            let ones = vec![1.0; NUM_COMBOS];
            beliefs[hs.my_seat].update(&ones); // renormalize
        }

        let resolved = solve_vectorized_capped(&root, &beliefs, self.cfg.river_iters, self.cfg.river_cap);

        let mut hole = hs.my_hole;
        hole.sort_unstable();
        let mut board5 = [NO_CARD; 5];
        board5.copy_from_slice(&board[..5]);
        let key = subgame_info_key(hs.my_seat, hole, &board5, &[]);
        let Some(probs) = resolved.strategy.get(&key).cloned() else {
            // Unreached in the resolve (shouldn't happen with the floor):
            // degrade to the blueprint.
            return self.decide_blueprint(hs, parsed, board);
        };

        let idx = self.sample(&probs);
        let action = acts[idx];

        // Keep the abstract tracker and our range coherent with what we do:
        // map the real move back into the abstract game as if observed.
        let facing = parsed.last_bet_size > 0;
        let mv = match action {
            Action::Fold => RealMove::Fold,
            Action::Check => RealMove::Check,
            Action::Call => RealMove::Call,
            Action::Raise(level) => RealMove::BetTo(level_to_pos_level(&root, hs.my_seat, level)),
            Action::AllIn => {
                let gs = &root;
                let level = gs.street_bets[hs.my_seat] + gs.stacks[hs.my_seat];
                if level <= gs.current_bet {
                    if facing {
                        RealMove::Call
                    } else {
                        RealMove::Check
                    }
                } else {
                    RealMove::BetTo(level)
                }
            }
        };
        let kind = match mv {
            RealMove::Fold => EventKind::Fold,
            RealMove::Check => EventKind::Check,
            RealMove::Call => EventKind::Call,
            RealMove::BetTo(n) => EventKind::BetTo(n),
        };
        if hs.hand.expects(&self.game, hs.my_seat, 3) {
            let mut rng = self.rng;
            let mut unit = || {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
            };
            let mapped = hs.hand.map_real(
                &self.game,
                kind,
                parsed.pot() as f64,
                parsed.street_last_bet_to as f64,
                &mut unit,
            );
            self.rng = rng;
            if let MapOutcome::Index(i) = mapped {
                self.update_range(hs, hs.my_seat, i);
                hs.hand.apply(&self.game, i);
                hs.pending_self = Some(Some(i));
            } else {
                hs.pending_self = Some(None);
            }
        } else {
            hs.pending_self = Some(None);
        }
        mv
    }

    /// Synthesize the real river public state (Slumbot chips) as an engine
    /// `GameState` — the resolve root.
    fn river_root(&self, hs: &HandState, parsed: &Parsed, board: &[u8]) -> GameState {
        let mut board5 = [NO_CARD; 5];
        board5.copy_from_slice(&board[..5]);
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
        gs.street = 3;
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
        let river_fresh = !parsed.events.iter().any(|e| e.street == 3);
        gs.players_to_act = if river_fresh && gs.current_bet == 0 { 2 } else { 1 };
        gs.last_aggressor = if parsed.last_bettor >= 0 {
            (1 - parsed.last_bettor) as u8
        } else {
            gs.to_act
        };
        gs
    }

    /// Purify + sample an action index from a distribution.
    fn sample(&mut self, probs: &[f64]) -> usize {
        let mut kept: Vec<f64> = probs.iter().map(|&p| if p < self.cfg.purify { 0.0 } else { p }).collect();
        let total: f64 = kept.iter().sum();
        if total <= 0.0 {
            // Everything purified away: play the argmax.
            return probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        for p in &mut kept {
            *p /= total;
        }
        let draw = self.unit();
        let mut acc = 0.0;
        for (i, &p) in kept.iter().enumerate() {
            acc += p;
            if draw < acc {
                return i;
            }
        }
        kept.len() - 1
    }

    pub fn game(&self) -> &BlueprintHoldem {
        &self.game
    }
}

/// An engine `Raise(level)` at the resolve root is already the actor's new
/// street-bet level in real chips — identical to Slumbot's `b<level>`
/// semantics.  Kept as a named function to make that unit statement explicit.
fn level_to_pos_level(_root: &GameState, _seat: usize, level: u32) -> u32 {
    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play::cards::parse_card;

    fn bot(resolve: bool) -> Bot {
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
        let policy = CompactPolicy::from_entries(vec![]); // uniform everywhere
        Bot::new(
            game,
            policy,
            BotConfig { resolve_river: resolve, river_iters: 120, river_cap: 2, purify: 0.0, seed: 42 },
        )
    }

    fn cards(list: &[&str]) -> Vec<u8> {
        list.iter().map(|s| parse_card(s).unwrap()).collect()
    }

    /// Drive the bot through whole hands against a scripted/random opponent,
    /// verifying every emitted increment is legal per the protocol parser.
    #[test]
    fn emits_legal_increments_over_random_hands() {
        let mut b = bot(false);
        let mut rng = 0xBADC_0FFEu64;
        let mut unit = move || {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };

        let full_board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
        for hand_no in 0..40u32 {
            let client_pos = (hand_no % 2) as u8;
            let hole = if hand_no % 3 == 0 {
                [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()]
            } else {
                [parse_card("7h").unwrap(), parse_card("2c").unwrap()]
            };
            let mut hs = b.start_hand(client_pos, hole);
            let mut action = String::new();

            loop {
                let parsed = parse_action(&action).expect("running action string stays legal");
                if parsed.next_pos < 0 {
                    break; // hand over
                }
                let street = parsed.street as usize;
                let board = &full_board[..[0usize, 3, 4, 5][street]];
                let before_street = parsed.street;
                if parsed.next_pos == hs.my_pos as i8 {
                    let incr = b.act(&mut hs, &action, board).expect("bot acts");
                    action.push_str(&incr);
                } else {
                    // Random legal opponent move.
                    let facing = parsed.last_bet_size > 0;
                    let remaining = STACK_SIZE - parsed.total_last_bet_to;
                    let choice = unit();
                    let mv = if facing {
                        if choice < 0.55 {
                            "c".to_string()
                        } else if choice < 0.7 && remaining > 0 {
                            let min = parsed.last_bet_size.max(BIG_BLIND).min(remaining);
                            let to = parsed.street_last_bet_to + min.max((remaining as f64 * unit() * 0.4) as u32).min(remaining);
                            format!("b{to}")
                        } else {
                            "f".to_string()
                        }
                    } else if choice < 0.6 || remaining == 0 {
                        "k".to_string()
                    } else {
                        let min = BIG_BLIND.min(remaining);
                        let to = parsed.street_last_bet_to
                            + min.max((remaining as f64 * unit() * 0.3) as u32).min(remaining);
                        format!("b{to}")
                    };
                    action.push_str(&mv);
                }
                // Close a finished pre-river street with the slash the server
                // inserts (mid-string separators are mandatory).
                let reparsed = parse_action(&action).expect("every appended move stays legal");
                if reparsed.next_pos >= 0 && reparsed.street > before_street && !action.ends_with('/')
                {
                    action.push('/');
                }
            }
        }
    }

    #[test]
    fn bucketed_flop_updates_never_index_out_of_bounds() {
        // Regression: with a real (bounds-checked) bucket map loaded, the
        // belief-update loop used to feed board-overlapping combos into the
        // hand indexer — a duplicated card yields an index past the canonical
        // table and panicked in BucketMap::bucket (seen live on the first
        // flop decision against Slumbot).  Card removal at board reveal plus
        // the defensive mask must keep every queried combo valid.
        use crate::abstraction::bucket_map::BucketMap;
        let game = BlueprintHoldem::new(400, 2, 1, 0)
            .with_raise_cap(3)
            .with_street_bucket(0, BucketMap::full_coverage_mod(&[2, 3], 40));
        let policy = CompactPolicy::from_entries(vec![]);
        let mut b = Bot::new(
            game,
            policy,
            BotConfig { resolve_river: false, river_iters: 0, river_cap: 2, purify: 0.0, seed: 7 },
        );
        // We are the BB (first to act postflop): SB opens, we call, flop comes,
        // our decision triggers a bucketed range update over the full board.
        let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
        let mut hs = b.start_hand(0, hole);
        let board = cards(&["Qs", "4s", "3h"]);
        let incr = b.act(&mut hs, "b300c/", &board).expect("flop decision with bucket map");
        assert!(parse_action(&format!("b300c/{incr}")).is_ok(), "legal move, got {incr:?}");

        // Board-overlapping combos carry zero mass in both ranges.
        for r in &hs.ranges {
            for (i, &p) in r.probs.iter().enumerate() {
                let [a, c] = combo_cards(i);
                if board.contains(&a) || board.contains(&c) {
                    assert_eq!(p, 0.0, "board-overlap combo must be dead");
                }
            }
            assert!((r.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // And the opponent can never hold our exact cards.
        let opp = &hs.ranges[1 - hs.my_seat];
        assert_eq!(opp.prob(hole[0], hole[1]), 0.0);
    }

    #[test]
    fn river_resolve_returns_a_root_action() {
        let mut b = bot(true);
        let hole = [parse_card("Ac").unwrap(), parse_card("Kd").unwrap()];
        let mut hs = b.start_hand(0, hole); // we are BB, first to act postflop
        // SB open to 200, we call; flop checks; turn checks; river to us.
        let action = "b200c/kk/kk/";
        let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
        let incr = b.act(&mut hs, action, &board).expect("river decision");
        assert!(
            incr == "k" || incr.starts_with('b'),
            "unopened river allows check or bet, got {incr:?}"
        );
        if let Some(n) = incr.strip_prefix('b') {
            let to: u32 = n.parse().unwrap();
            assert!((BIG_BLIND..=19_800).contains(&to), "legal river bet size, got {to}");
        }
        // The full string with our move must still parse.
        assert!(parse_action(&format!("{action}{incr}")).is_ok());
    }

    /// A raise war past the cap leaves the abstract tracker with no node, and
    /// the agent used to answer every such spot with an unconditional call —
    /// which is how it called off 200bb with bottom pair against Slumbot.
    #[test]
    fn a_desynced_bot_folds_trash_and_calls_the_nuts() {
        // SB opens 250, BB 3-bets 750, SB 4-bets 1657, BB 5-bets 12000.  That
        // fifth bet is the fourth raise: past a cap of 3, so it cannot be
        // translated.  Calling costs 10343 into a 13657 pot -- 0.43 equity.
        let action = "b250b750b1657b12000";
        for (hole, expected) in [(["7h", "2c"], "f"), (["Ac", "Ad"], "c")] {
            let mut b = bot(false);
            let h = [parse_card(hole[0]).unwrap(), parse_card(hole[1]).unwrap()];
            let mut hs = b.start_hand(1, h);

            let parsed = parse_action(action).expect("legal action string");
            b.sync(&mut hs, &parsed, &[]);
            assert!(
                !hs.hand.expects(&b.game, hs.my_seat, 0),
                "the post-cap raise must desync the tracker, else this tests nothing"
            );

            let incr = b.act(&mut hs, action, &[]).expect("bot acts");
            assert_eq!(incr, expected, "holding {hole:?} facing a 5-bet");
        }
    }

    #[test]
    fn facing_a_river_shove_resolves_to_a_legal_response() {
        let mut b = bot(true);
        let hole = [parse_card("Ac").unwrap(), parse_card("Ad").unwrap()];
        let mut hs = b.start_hand(0, hole);
        let action = "b200c/kk/kk/kb19800";
        let board = cards(&["Qs", "4s", "3h", "Th", "8c"]);
        let incr = b.act(&mut hs, action, &board).expect("shove response");
        assert!(incr == "c" || incr == "f", "call or fold vs a shove, got {incr:?}");
        assert!(parse_action(&format!("{action}{incr}")).is_ok());
    }
}
