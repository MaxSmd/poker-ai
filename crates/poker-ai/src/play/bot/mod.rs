//! The playing agent: blueprint policy + belief tracking + postflop re-solving.
//!
//! Decision architecture (Pluribus-style split, sized to what a single machine
//! can compute in real time at 200 bb):
//!
//! * **Preflop** — play directly from the trained blueprint: translate the real
//!   hand into the abstract game ([`AbstractHand`]), look up the average
//!   strategy at the info key, purify + sample.
//! * **Flop / turn / river** — full-range re-solve of the *actual* public state
//!   with the vectorized public-tree CFR: the resolve root carries the real pot
//!   and stacks (off-tree bets included, so translation distortion vanishes
//!   exactly where the money is deepest), our range and the opponent's range are
//!   carried Bayes updates of the blueprint (`P(observed abstract action | hand)`
//!   at every prior decision), and the resolved root distribution is played
//!   directly in real chips.  A **river** resolve is exact to showdown; a
//!   **turn** resolve cuts at the river reveal and a **flop** resolve at the
//!   turn reveal, with the opponent choosing among K continuations at the leaf,
//!   so the resolve is robust to post-leaf betting a plain check-down ignores.
//!   By default only the **river** re-solves (exact to showdown, no leaf
//!   model); the flop and turn play the blueprint — their leaf-cut re-solves
//!   over-fold, measured in `BotConfig::resolve_turn` and `resolve_flop`;
//!   `runout_sample` is what makes the turn and flop affordable, and
//!   `turn_full_river` is an offline reference mode at 798 s/decision.
//! * **Continual re-solving** (DeepStack-style, on by default) — every resolve
//!   extracts the opponent's per-hand counterfactual values under the emitted
//!   strategy; the next resolve in the same hand is constrained by the CFV
//!   gadget (per-hand Follow/Terminate), so the opponent cannot profit from our
//!   strategy being recomputed between decisions.
//!
//! Both ranges are maintained per hand; the opponent's is additionally
//! filtered by card removal (board + our own hole cards).

mod decide;
#[cfg(test)]
mod tests;

use crate::games::blueprint::BlueprintHoldem;
use crate::play::policy::CompactPolicy;
use crate::play::protocol::{parse_action, Event, Parsed};
use crate::play::tracker::{AbstractHand, MapOutcome, RealMove};
use crate::resolving::belief_state::{combo_cards, BeliefState, NUM_COMBOS};
use crate::util::rng::xorshift_next_unit;

/// Tunables for the playing agent.
#[derive(Clone, Debug)]
pub struct BotConfig {
    /// Re-solve river decisions (recommended); otherwise blueprint throughout.
    ///
    /// 1.9 s per decision at `river_iters` (`bench_resolve_cost`, 16 threads).
    /// A complete board has no runout, so [`Self::runout_sample`] does not
    /// apply here — the river is always exact to showdown.
    pub resolve_river: bool,
    /// Re-solve turn decisions.  **Off by default** — the blueprint plays the
    /// turn.
    ///
    /// Measured by replaying the 792 logged spots where Slumbot leads the turn
    /// into our small blind (`scripts/replay_turn_folds.sh`, 2026-10): with
    /// the K-continuation leaf the resolve folds 81% there (top pair or better
    /// 13%), against Slumbot's 43% in the same spot; the blueprint folds 40%
    /// (top pair or better 2%).  The leaf lets the opponent pick the river pot
    /// size per hand — large when ahead, small when behind — which prices
    /// every marginal call as a loser.  A plain check-down leaf errs the other
    /// way (20%, calling with nothing 61%), and flooring the opponent's range
    /// updates only reached 64%.  Turn on with `--resolve-turn` once the leaf
    /// model is fixed.
    ///
    /// 1.9 s per decision. A turn board has only 48 completions, so at the
    /// default [`Self::runout_sample`] of 64 the turn sweep clamps to exact —
    /// sampling buys nothing here and is not used.
    ///
    /// **[`Self::turn_full_river`] must stay off.** It replaces the depth cut
    /// with the real river betting and costs 798 s per decision.
    pub resolve_turn: bool,
    /// Re-solve flop decisions.  **Off by default** — the blueprint plays the
    /// flop, as it does the turn.
    ///
    /// Two 10k-hand Slumbot logs with this on show where the money went:
    /// facing a flop bet the small blind folded 66% (Slumbot 33-37% in the
    /// same seat) and the big blind 56-57% (Slumbot 46-49%), together about
    /// -20 bb/100 — the whole loss.  It is the same continuation leaf that
    /// made the turn re-solve fold 81% to leads (see `resolve_turn`), and the
    /// match history agrees: blueprint flop with river re-solving scored
    /// +7.9 ± 26 (July) and -5.9 ± 33 (August); re-solving the flop, -23.7
    /// and -19.6.  `--resolve-flop` turns it back on.
    ///
    /// 2.1 s per decision — down from **24.6 s** before [`Self::runout_sample`]
    /// existed. Each depth-cut leaf enumerates C(49,2)=1176 turn+river
    /// completions against 48 on the turn, so the flop is the one street whose
    /// cost is set almost entirely by the runout sweep rather than by the tree:
    /// identical 735-node trees, 12× the per-iteration cost.
    pub resolve_flop: bool,
    /// CFR⁺ iterations per river resolve.
    pub river_iters: u64,
    /// CFR⁺ iterations per turn/flop resolve — kept lower than `river_iters`
    /// because runout leaves make each iteration far costlier.
    pub turn_iters: u64,
    /// Raise cap inside a resolve (bounds the public tree; used at every street).
    pub river_cap: u32,
    /// Rest-of-hand pot scales for the turn/flop depth-limit continuation choice
    /// the opponent picks among these at each runout leaf, so the
    /// resolve is robust to the post-leaf betting a plain check-down ignores.
    /// `scales[0]` should be `0.0`; `[0.0]` is a single check-down (no chooser).
    pub continuations: Vec<f64>,
    /// Turn resolves deal the river as an explicit chance node and solve the
    /// **real river betting** — exact to showdown, no leaf model at all
    /// (default).  Off = cut at the reveal with the K-continuation check-down
    /// leaf (`continuations`); flop resolves always use the continuation cut
    /// either way.
    ///
    /// The cost difference is not a tuning knob, it is a change of regime.
    /// Measured at `river_cap = 3` (`bench_resolve_cost`, 16 threads):
    ///
    /// ```text
    ///   full-river          1 356 939 nodes   1580.7 ms/iter   798 s/decision
    ///   continuation cut          735 nodes      3.9 ms/iter     1.9 s/decision
    ///   full-river, cap 1      11 115 nodes     16.3 ms/iter     8.2 s/decision
    /// ```
    ///
    /// That is 439×, not the "~48×" this comment claimed before anyone
    /// measured it.  It used to be the default, which made turning
    /// [`Self::resolve_turn`] on a 13-minute decision; it is now **off**, and
    /// exists as an offline reference for checking what the continuation cut
    /// costs in strategy.  Sampling does not rescue it — its cost is the
    /// 1.36 M-node tree, not the runout — and neither does `river_cap = 1`.
    pub turn_full_river: bool,
    /// Runout completions evaluated per iteration at each depth-cut leaf
    /// (`0` = exact).  The lever that makes turn and flop re-solving affordable
    /// at all: a flop leaf has 1176 completions and a turn leaf 48, and the
    /// resolve pays that *per iteration*.  Sampling trades per-iteration
    /// accuracy — which CFR averages away across iterations — for latency,
    /// which it cannot.
    ///
    /// Seconds per decision at 500 iterations (`bench_resolve_cost`, 16
    /// threads):
    ///
    /// ```text
    ///   sample     turn    flop
    ///   exact       2.1    24.6
    ///       16      0.9     1.1
    ///       32      1.4     1.6
    ///       64      1.9     2.1   <-- default
    ///      128      1.9     3.3
    ///      256      1.8     6.1
    /// ```
    ///
    /// 64 is chosen so the **turn stays exact** — it has only 48 completions,
    /// so anything at or above that clamps to the full sweep — while the flop
    /// samples 64 of 1176 and lands beside the river's 1.9 s. Below 64 the turn
    /// starts being sampled to save time it does not need to save; above it,
    /// only the flop gets slower. More completions is strictly better strategy
    /// at equal latency, so raise this on a machine with headroom.
    ///
    /// Ignored on the river (a complete board has no runout) and by CFV
    /// extraction, which runs once per resolve and stays exact because its
    /// output is the safety guarantee the next continual resolve carries.
    pub runout_sample: usize,
    /// Continual re-solving (DeepStack-style): carry the opponent's
    /// counterfactual values from each resolve and constrain the next one
    /// with the CFV gadget, so the opponent cannot profit from our strategy
    /// being recomputed between decisions.  The hand's first resolve is the
    /// unconstrained bootstrap.
    pub continual: bool,
    /// Re-solve any postflop decision the abstract mirror has lost track of,
    /// even on a street whose re-solving is off.  Such a spot has no blueprint
    /// node, and the alternative is the desynced fallback's pot odds against
    /// a frozen range, which never raises.  Off only for the blueprint-only
    /// ablation (`--no-resolve`), which must not re-solve anything.
    pub resolve_desynced: bool,
    /// Purification: drop actions below this probability, renormalize, then
    /// sample (`0.0` = sample the raw mixed strategy).
    pub purify: f64,
    /// Floor on the likelihood of the opponent's **postflop** actions in its
    /// range update: each hand's blueprint probability `L` of the observed
    /// action becomes `floor + (1 − floor)·L`.  `0.0` is the plain Bayes
    /// update; `1.0` ignores postflop actions entirely.  A floor keeps an
    /// action the blueprint rarely takes (say, a turn lead) from narrowing the
    /// opponent's range to the few hands that take it — the right inference
    /// against the blueprint itself, the wrong one against an opponent whose
    /// frequencies differ.  Preflop updates and our own range are unaffected.
    pub opp_range_floor: f64,
    /// Play flop and turn decisions by depth-limited search with blueprint
    /// continuation strategies at the leaves
    /// ([`crate::resolving::depth_limited`]) instead of the vectorized
    /// resolve or the blueprint.  Only where the abstract mirror is in sync —
    /// the search runs in the blueprint's own game; anything else falls back
    /// to the usual path.  **Off by default** (`--search` turns it on): it
    /// fixed the four spots it was replayed on but turned the big blind
    /// hyper-aggressive in play (flop lead 53%, check-raise 51%; Slumbot 16%
    /// and 7%), and more iterations did not cure that.
    ///
    /// Measured by replaying 600 logged spots each from a 10k-hand Slumbot
    /// log (`scripts/replay_search.sh`, 2026-10), against Slumbot's own
    /// frequencies in the same spots:
    ///
    /// ```text
    ///                               before   search   Slumbot
    ///   flop c-bet (we bet)          0.28     0.54     0.58
    ///   flop vs a bet (we fold)      0.56     0.45     0.49
    ///   turn lead (we bet)           0.19     0.42     0.33
    ///   turn vs a lead (we fold)     0.39     0.43     0.35
    /// ```
    ///
    /// at 0.12 s per decision with the 16 GB blueprint at 2000 iterations.
    pub search: bool,
    /// MCCFR iterations per search.
    pub search_iters: u64,
    /// Continuation strategies per player at a search leaf (4, or 1 for the
    /// blueprint alone), how strongly the biased ones lean, and whether we
    /// choose ours too — see `resolving::depth_limited::SearchConfig`.
    pub search_continuations: usize,
    pub search_bias: f64,
    pub search_searcher_chooses: bool,
    /// Seed for the agent's action sampling and bet-mapping randomization.
    pub seed: u64,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            resolve_river: true,
            resolve_turn: false,
            resolve_flop: false,
            river_iters: 1_500,
            turn_iters: 500,
            river_cap: 3,
            continuations: vec![0.0, 0.75, 1.5, 3.0],
            turn_full_river: false,
            runout_sample: 64,
            continual: true,
            resolve_desynced: true,
            purify: 0.1,
            opp_range_floor: 0.0,
            search: false,
            search_iters: 2_000,
            search_continuations: crate::resolving::depth_limited::CONTINUATIONS,
            search_bias: 5.0,
            search_searcher_chooses: true,
            seed: 1,
        }
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
    /// Continual re-solving: the opponent's per-hand counterfactual values
    /// extracted from our previous resolve this hand (bb, `features`
    /// combo order).  The next resolve is gadget-constrained by these, so the
    /// opponent cannot profit from our strategy being recomputed between
    /// decisions; refreshed after every resolve.  `None` until a street's
    /// first (bootstrap) resolve: it is dropped at every board reveal.
    carried_cfvs: Option<Box<[f64; NUM_COMBOS]>>,
    /// AIVAT action-luck term for our own decisions this hand (bb): the sum
    /// over re-solved decisions of `v(chosen) − Σ_a π(a)·v(a)`, with `π` the
    /// distribution the action was drawn from and `v` the resolve's value of
    /// each root action for our actual hand.  Zero-mean per decision given
    /// everything before it, so subtracting it from winnings stays unbiased
    /// while removing the variance of our own mixing.  Blueprint decisions
    /// have no per-action values and contribute nothing (also unbiased).
    action_luck_bb: f64,
}

/// Which regime decided a move (see `decide`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionPath {
    Blueprint,
    Resolve,
    Search,
    Desynced,
}

/// The distribution one decision was drawn from, in real moves: one entry
/// per root action, so a move can appear more than once (an abstract fold
/// with nothing to call is a check).  Recorded for every decision; read by
/// `play replay` to measure what the agent *would* do in logged spots.
#[derive(Clone, Debug)]
pub struct DecisionTrace {
    pub street: u8,
    pub path: DecisionPath,
    pub moves: Vec<(RealMove, f64)>,
}

impl DecisionTrace {
    /// Total probability of moves matching `pred`.
    pub fn prob(&self, pred: impl Fn(RealMove) -> bool) -> f64 {
        // `+ 0.0`: an empty sum is −0.0, which would print as "-0".
        self.moves.iter().filter(|(m, _)| pred(*m)).map(|(_, p)| p).sum::<f64>() + 0.0
    }
}

impl HandState {
    /// The AIVAT action-luck term for this hand in Slumbot chips (see
    /// `action_luck_bb`).  Adjusted winnings = raw − card luck − this.
    pub fn action_luck_chips(&self) -> f64 {
        self.action_luck_bb * crate::play::protocol::BIG_BLIND as f64
    }
}

/// The playing agent. One instance plays many hands (per-hand state lives in
/// [`HandState`]); it owns the abstract game and the blueprint policy.
pub struct Bot {
    game: BlueprintHoldem,
    policy: CompactPolicy,
    cfg: BotConfig,
    rng: u64,
    tracking: TrackingCounts,
    /// Set by [`Bot::act_forced`]: play this move instead of sampling.
    forced: Option<RealMove>,
    /// Forced moves that matched none of the decision's root actions.
    forced_mismatches: u64,
    last_trace: Option<DecisionTrace>,
}

/// How often the agent's abstract mirror of the real hand broke down.
///
/// The blueprint's betting abstraction is capped, so an opponent who raises
/// past the cap produces a real event with **no abstract node**.  Two things
/// then happen, and neither announces itself:
///
/// * the event is skipped, so the opponent's range stops being updated — every
///   later belief in the hand is conditioned on a strictly incomplete history;
/// * our own decisions have no blueprint node: postflop they are re-solved
///   from the real state (unless `BotConfig::resolve_desynced` is off), and
///   otherwise fall through to `decide_desynced`, which knows only pot odds
///   against a frozen range and never raises.
///
/// Both degrade play in ways that look exactly like "re-solving is bad", so the
/// rate has to be measured before any conclusion is drawn about the resolver.
/// Against a near-equilibrium opponent this should be rare; against one that
/// raises constantly it need not be.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrackingCounts {
    /// Opponent (or our own echoed) events the abstraction could not map,
    /// including every event skipped because the mirror had already lost the
    /// hand.
    pub unmapped_events: u64,
    /// Total events fed through the mirror.
    pub events: u64,
    /// Our decisions made while the abstract mirror had no node for the spot,
    /// whichever path then decided them (re-solve or the pot-odds fallback).
    pub desynced_decisions: u64,
    /// Total decisions we were asked for.
    pub decisions: u64,
}

impl TrackingCounts {
    pub fn desync_rate(&self) -> f64 {
        match self.decisions {
            0 => 0.0,
            n => self.desynced_decisions as f64 / n as f64,
        }
    }

    pub fn unmapped_rate(&self) -> f64 {
        match self.events {
            0 => 0.0,
            n => self.unmapped_events as f64 / n as f64,
        }
    }
}

impl std::fmt::Display for TrackingCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} decisions desynced ({:.2}%), {}/{} events unmappable ({:.2}%)",
            self.desynced_decisions,
            self.decisions,
            100.0 * self.desync_rate(),
            self.unmapped_events,
            self.events,
            100.0 * self.unmapped_rate()
        )
    }
}

impl Bot {
    pub fn new(game: BlueprintHoldem, policy: CompactPolicy, cfg: BotConfig) -> Self {
        let rng = cfg.seed | 1;
        Self {
            game,
            policy,
            cfg,
            rng,
            tracking: TrackingCounts::default(),
            forced: None,
            forced_mismatches: 0,
            last_trace: None,
        }
    }

    /// The distribution behind the most recent decision (see [`DecisionTrace`]).
    pub fn last_trace(&self) -> Option<&DecisionTrace> {
        self.last_trace.as_ref()
    }

    /// Forced moves so far that no root action matched (see [`Self::act_forced`]).
    pub fn forced_mismatches(&self) -> u64 {
        self.forced_mismatches
    }

    /// [`act`](Self::act), but play `forced` (a wire increment) instead of
    /// sampling — everything else, including our range and the carried
    /// values, is updated exactly as if we had drawn it.  The replay of a
    /// logged hand: it follows the hand's real history while still recording
    /// what the agent would have done ([`Self::last_trace`]).  A bet is matched
    /// to the root action with the nearest amount; a move no root action
    /// matches plays the most likely action and is counted.
    pub fn act_forced(
        &mut self,
        hs: &mut HandState,
        action_str: &str,
        board: &[u8],
        forced: &str,
    ) -> Result<String, String> {
        let mv = RealMove::from_incr(forced).ok_or_else(|| format!("bad forced move {forced:?}"))?;
        self.forced = Some(mv);
        let out = self.act(hs, action_str, board);
        self.forced = None;
        out
    }

    /// How the blueprint lookups behind this bot's decisions have resolved.
    ///
    /// The number to watch during a match: a low hit rate means the bot is
    /// playing uniform-random in the abstract game *and* feeding those fake
    /// action likelihoods to the range tracker, so every re-solve below it
    /// starts from beliefs the blueprint never informed.  See
    /// [`crate::play::policy::LookupStats`].
    pub fn lookup_counts(&self) -> crate::play::policy::LookupCounts {
        self.policy.lookup_counts()
    }

    /// How often the abstract mirror broke down — see [`TrackingCounts`].
    pub fn tracking_counts(&self) -> TrackingCounts {
        self.tracking
    }

    /// xorshift64* uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        xorshift_next_unit(&mut self.rng)
    }

    /// Begin a hand as `client_pos` (Slumbot convention) holding `hole`.
    pub fn start_hand(&mut self, client_pos: u8, hole: [u8; 2]) -> HandState {
        let my_seat = 1 - client_pos as usize;
        let mut ranges = [BeliefState::uniform(), BeliefState::uniform()];
        // The opponent can never hold our cards — filter them out up front.
        ranges[1 - my_seat].update(&hole_mask(hole));
        HandState {
            my_pos: client_pos,
            my_seat,
            my_hole: hole,
            hand: AbstractHand::new(&self.game, my_seat, hole),
            ranges,
            processed: 0,
            board_seen: 0,
            pending_self: None,
            carried_cfvs: None,
            action_luck_bb: 0.0,
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

        self.last_trace = None;
        // Counted here rather than in the decision functions: `decide_resolve`
        // can fall through to `decide_blueprint`, and this is the one entry
        // point every decision passes through exactly once.
        self.tracking.decisions += 1;
        let desynced = !hs.hand.expects(&self.game, hs.my_seat, parsed.street);
        if desynced {
            self.tracking.desynced_decisions += 1;
        }
        let resolve_desynced =
            desynced && self.cfg.resolve_desynced && parsed.street > 0 && board_matches(parsed.street, board);
        let search = self.cfg.search && !desynced && matches!(parsed.street, 1 | 2) && board_matches(parsed.street, board);
        let mv = if search {
            self.decide_search(hs, &parsed, board)
        } else if self.should_resolve(parsed.street, board) || resolve_desynced {
            self.decide_resolve(hs, &parsed, board)
        } else {
            self.decide_blueprint(hs, &parsed, board)
        };
        Ok(mv.to_incr())
    }

    /// Whether to re-solve this street, given the enabled flags and a board with
    /// the right number of revealed cards for the street (a defensive guard —
    /// the resolve root synthesis assumes the board matches the street).
    fn should_resolve(&self, street: u8, board: &[u8]) -> bool {
        let enabled = match street {
            3 => self.cfg.resolve_river,
            2 => self.cfg.resolve_turn,
            1 => self.cfg.resolve_flop,
            _ => false, // preflop is always blueprint
        };
        enabled && board_matches(street, board)
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
            // The carry was taken before this card and averages over it, so
            // it constrains no specific card: too tight where the card helps
            // an opponent hand, too loose where it hurts.  The street's first
            // resolve bootstraps instead (see `VectorCfr::opponent_cfvs_after`).
            hs.carried_cfvs = None;
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
        // A skipped event is an opponent action the range never sees, so it is
        // counted as unmapped — the typical desync (a raise past the cap mapped
        // to all-in and called, abstract hand over, real hand still running)
        // produces nothing *but* skipped events, and used to read as 0%.
        if !hs.hand.expects(&self.game, seat, ev.street) {
            self.tracking.events += 1;
            self.tracking.unmapped_events += 1;
            return;
        }

        let mut rng = self.rng;
        let mapped = hs.hand.map_real(
            &self.game,
            ev.kind,
            ev.pot_before as f64,
            ev.bet_before as f64,
            &mut || xorshift_next_unit(&mut rng),
        );
        self.rng = rng;

        self.tracking.events += 1;
        if let MapOutcome::Index(idx) = mapped {
            self.update_range(hs, seat, idx);
            hs.hand.apply(&self.game, idx);
        } else {
            // Post-cap aggression: no abstract node, so the range update for
            // this event never happens.  Counted, because a high rate here
            // means the beliefs every resolve starts from are conditioned on a
            // truncated history.
            self.tracking.unmapped_events += 1;
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
        let floor = if seat != hs.my_seat && gs.street >= 1 { self.cfg.opp_range_floor } else { 0.0 };
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
            *l = floor + (1.0 - floor) * self.policy.probs_or_uniform(key, n)[action_index as usize];
        }
        hs.ranges[seat].update(&likelihood);
    }
}

/// Whether `board` holds exactly the cards `street` has revealed — the guard
/// every re-solve needs, since the root synthesis assumes they agree.
/// Per-combo mask that zeroes every hand sharing a card with `hole` — the
/// hands an opponent can never hold against us.
fn hole_mask(hole: [u8; 2]) -> Vec<f64> {
    (0..NUM_COMBOS)
        .map(|i| {
            let [a, b] = combo_cards(i);
            if a == hole[0] || a == hole[1] || b == hole[0] || b == hole[1] {
                0.0
            } else {
                1.0
            }
        })
        .collect()
}

fn board_matches(street: u8, board: &[u8]) -> bool {
    match street {
        1 => board.len() == 3,
        2 => board.len() == 4,
        3 => board.len() == 5,
        _ => false,
    }
}
