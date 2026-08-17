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
//!   All three streets re-solve by default, each inside ~2 s per decision;
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
use crate::play::tracker::{AbstractHand, MapOutcome};
use crate::resolving::belief_state::{combo_cards, BeliefState, NUM_COMBOS};

/// Tunables for the playing agent.
#[derive(Clone, Debug)]
pub struct BotConfig {
    /// Re-solve river decisions (recommended); otherwise blueprint throughout.
    ///
    /// 1.9 s per decision at `river_iters` (`bench_resolve_cost`, 16 threads).
    /// A complete board has no runout, so [`Self::runout_sample`] does not
    /// apply here — the river is always exact to showdown.
    pub resolve_river: bool,
    /// Re-solve turn decisions.  On by default.
    ///
    /// 1.9 s per decision. A turn board has only 48 completions, so at the
    /// default [`Self::runout_sample`] of 64 the turn sweep clamps to exact —
    /// sampling buys nothing here and is not used.
    ///
    /// **[`Self::turn_full_river`] must stay off.** It replaces the depth cut
    /// with the real river betting and costs 798 s per decision.
    pub resolve_turn: bool,
    /// Re-solve flop decisions.  On by default.
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
    /// Purification: drop actions below this probability, renormalize, then
    /// sample (`0.0` = sample the raw mixed strategy).
    pub purify: f64,
    /// Seed for the agent's action sampling and bet-mapping randomization.
    pub seed: u64,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            resolve_river: true,
            resolve_turn: true,
            resolve_flop: true,
            river_iters: 1_500,
            turn_iters: 500,
            river_cap: 3,
            continuations: vec![0.0, 0.75, 1.5, 3.0],
            turn_full_river: false,
            runout_sample: 64,
            continual: true,
            purify: 0.1,
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
    /// decisions; refreshed after every resolve.  `None` until the hand's
    /// first (bootstrap) resolve.
    carried_cfvs: Option<Box<[f64; NUM_COMBOS]>>,
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
            carried_cfvs: None,
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

        let mv = if self.should_resolve(parsed.street, board) {
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
        match street {
            3 => self.cfg.resolve_river && board.len() == 5,
            2 => self.cfg.resolve_turn && board.len() == 4,
            1 => self.cfg.resolve_flop && board.len() == 3,
            _ => false, // preflop is always blueprint
        }
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
}
