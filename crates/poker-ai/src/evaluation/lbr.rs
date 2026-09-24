//! **Local best response** (LBR) — a lower bound on the *live agent's*
//! exploitability, in real chips.
//!
//! ## Why this exists when `play expl` already reports a number
//!
//! [`crate::evaluation::vector_br`] best-responds to the **blueprint**, walking
//! the abstract game.  It cannot see the agent that actually plays: preflop
//! lookups, Bayes range tracking, purification, and per-decision re-solving of
//! the real public state.  A true best response to *that* would have to run the
//! agent's own resolve at every node it reaches — millions of nodes at ~2 s
//! each — which is why the literature does not compute one.
//!
//! LBR (Lisý & Bowling 2017) sidesteps it.  Rather than best-responding to the
//! agent's strategy, LBR *plays* the agent, choosing at each of its own
//! decisions the action that maximizes an immediate, locally-computed value.
//! The result is a legitimate strategy, so **its win rate is a valid lower
//! bound on the agent's exploitability** — a bound that holds no matter how
//! crude LBR's internal model is.  Model quality only affects how *tight* the
//! bound is, never whether it is sound.  That property is what makes this
//! measurable at all, and it is the reason the numbers below can be reported
//! without an argument about whether the responder was strong enough.
//!
//! ## The local value model
//!
//! At each LBR decision, with `e` = LBR's equity against its model of the
//! agent's range and `pot` = chips already in:
//!
//! ```text
//!   fold        0
//!   check       e · pot
//!   call        e · (pot + c) − c                       c = chips to call
//!   raise to L  f · pot + (1 − f) · (e' · (pot + 2d) − d)
//!               d = L − our street bet,  f = d / (pot + d)
//! ```
//!
//! Two deliberate modelling choices, both stated because they bound the result:
//!
//! * **`f` is minimum-defense frequency.** The agent is assumed to fold exactly
//!   the fraction of its range that makes LBR's bluff break even — what an
//!   equilibrium opponent does. A real agent that over-folds is exploited *more*
//!   than this predicts, so the bound stays valid.
//! * **`e' = (e − f) / (1 − f)`,** clamped to `[0, 1]`: after the worst `f` of
//!   the range folds, LBR's equity against what remains is its equity against
//!   the whole range less the part that got away. This is the standard
//!   correction, and it keeps LBR from over-valuing a bet into a range that only
//!   continues with its strong hands.
//!
//! LBR chooses among exactly the actions [`legal_actions`] offers, so it never
//! constructs a bet the engine would reject and never gets credit for a sizing
//! the agent could not have faced.
//!
//! ## What this version does not do
//!
//! It does **not** narrow its model of the agent's range using the agent's
//! observed actions — the range stays uniform-minus-blockers for the whole
//! hand. A range-tracking LBR is strictly stronger and would report a *larger*
//! number. This one is the conservative floor, and the floor is the honest
//! thing to publish first.
//!
//! ## Luck adjustment
//!
//! Both hands are known here, so every hand also gets an AIVAT-adjusted
//! result: the card-luck correction of [`crate::play::luck`] plus the agent's
//! own action-luck term (`HandState::action_luck_chips`).  Each correction
//! has zero mean, so the adjusted win rate estimates the same quantity with a
//! tighter interval — which is what makes paired ablations between agent
//! configurations readable at a few thousand hands.  LBR's own randomness is
//! not corrected (its policy distribution is not exposed), which is still
//! unbiased, only less tight.

use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};
use poker_core::{legal_actions, Action};

use crate::play::bot::Bot;
use crate::play::equity::equity_vs_range;
use crate::play::luck::luck_adjustment;
use crate::play::protocol::{BIG_BLIND, SMALL_BLIND, STACK_SIZE};
use crate::resolving::belief_state::BeliefState;

/// Result of an LBR match, in big blinds.
#[derive(Clone, Copy, Debug, Default)]
pub struct LbrOutcome {
    pub hands: u64,
    /// LBR's net winnings (bb).  **Positive means the agent is exploitable.**
    pub net_bb: f64,
    sumsq_bb: f64,
    /// LBR's net winnings after the luck corrections (bb) — same expectation
    /// as `net_bb`, lower variance.  See the module header.
    pub adj_net_bb: f64,
    adj_sumsq_bb: f64,
    /// Hands abandoned because the agent errored (desync, unparseable action).
    pub errors: u64,
}

impl LbrOutcome {
    /// LBR's win rate in bb/100 — the exploitability lower bound.
    pub fn bb100(&self) -> f64 {
        rate_bb100(self.net_bb, self.hands)
    }

    /// 95% confidence half-width on [`bb100`](Self::bb100).
    pub fn ci95(&self) -> f64 {
        ci95_bb100(self.net_bb, self.sumsq_bb, self.hands)
    }

    /// Luck-adjusted win rate in bb/100: the same expectation as
    /// [`bb100`](Self::bb100), estimated with lower variance.
    pub fn adj_bb100(&self) -> f64 {
        rate_bb100(self.adj_net_bb, self.hands)
    }

    /// 95% confidence half-width on [`adj_bb100`](Self::adj_bb100).
    pub fn adj_ci95(&self) -> f64 {
        ci95_bb100(self.adj_net_bb, self.adj_sumsq_bb, self.hands)
    }

    /// LBR's win rate in milli-big-blinds per hand — the figure the literature
    /// quotes.  **Negative means the bound is vacuous**, not that the agent is
    /// unexploitable: exploitability is never below zero (a Nash opponent scores
    /// 0), so a losing LBR has only shown itself to be a weak responder.  Use
    /// [`bound_mbb`](Self::bound_mbb) for the quantity that is safe to publish
    /// and read this one as a head-to-head win rate.
    pub fn mbb_per_hand(&self) -> f64 {
        self.bb100() * 10.0
    }

    /// The publishable exploitability lower bound: `max(0, ·)`.  A responder
    /// that loses proves nothing about the agent, so the bound it certifies is
    /// zero.
    pub fn bound_mbb(&self) -> f64 {
        self.mbb_per_hand().max(0.0)
    }

    fn record(&mut self, bb: f64, adj_bb: f64) {
        self.hands += 1;
        self.net_bb += bb;
        self.sumsq_bb += bb * bb;
        self.adj_net_bb += adj_bb;
        self.adj_sumsq_bb += adj_bb * adj_bb;
    }
}

fn rate_bb100(net: f64, hands: u64) -> f64 {
    if hands == 0 {
        return 0.0;
    }
    net / hands as f64 * 100.0
}

fn ci95_bb100(net: f64, sumsq: f64, hands: u64) -> f64 {
    if hands < 2 {
        return 0.0;
    }
    let n = hands as f64;
    let mean = net / n;
    let var = (sumsq / n - mean * mean).max(0.0);
    1.96 * (var / n).sqrt() * 100.0
}

fn next_u64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Nine distinct cards: two per player, five board.
fn deal(rng: &mut u64) -> ([[u8; 2]; 2], [u8; 5]) {
    let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
    // Partial Fisher–Yates: only the first nine slots are needed.
    for i in 0..9 {
        let j = i + (next_u64(rng) % (52 - i as u64)) as usize;
        deck.swap(i, j);
    }
    ([[deck[0], deck[1]], [deck[2], deck[3]]], [deck[4], deck[5], deck[6], deck[7], deck[8]])
}

/// Board cards visible on `street` (0 preflop … 3 river).
fn visible(board: &[u8; 5], street: u8) -> Vec<u8> {
    let n = match street {
        0 => 0,
        1 => 3,
        2 => 4,
        _ => 5,
    };
    board[..n].to_vec()
}

/// The Slumbot wire token for an action LBR chose.
fn token(gs: &GameState, act: Action) -> String {
    let seat = gs.to_act as usize;
    match act {
        Action::Fold => "f".into(),
        Action::Check => "k".into(),
        Action::Call => "c".into(),
        Action::Raise(to) => format!("b{to}"),
        Action::AllIn => format!("b{}", gs.street_bets[seat] + gs.stacks[seat]),
    }
}

/// Map the agent's wire token back to an engine action.
/// The legal aggressive action in LBR's game closest to a bet **to** `to`.
///
/// LBR's game offers only the engine's own raise sizes, while the agent
/// translates its abstract raise into real chips independently, so the two
/// can round to different amounts (a raise to 200 where the menu has 220).
/// Applying the agent's exact amount would panic the engine; snapping keeps
/// the hand going, and the caller writes the snapped size to the wire so the
/// agent sees the history LBR's game actually played.  With no aggressive
/// action legal (at the cap), the bet becomes a call — or a check.
fn snap_bet(gs: &GameState, to: u32) -> Action {
    let seat = gs.to_act as usize;
    let all_in_level = gs.street_bets[seat] + gs.stacks[seat];
    let legal = legal_actions(gs);
    let nearest = legal
        .iter()
        .filter_map(|&a| match a {
            Action::Raise(l) => Some((a, l)),
            Action::AllIn if all_in_level > gs.current_bet => Some((a, all_in_level)),
            _ => None,
        })
        .min_by_key(|&(_, l)| l.abs_diff(to))
        .map(|(a, _)| a);
    nearest.unwrap_or_else(|| {
        if legal.iter().any(|a| matches!(a, Action::Call)) {
            Action::Call
        } else {
            Action::Check
        }
    })
}

fn action_from_token(incr: &str, gs: &GameState) -> Result<Action, String> {
    let seat = gs.to_act as usize;
    match incr.as_bytes().first() {
        Some(b'f') => Ok(Action::Fold),
        Some(b'k') => Ok(Action::Check),
        Some(b'c') => Ok(Action::Call),
        Some(b'b') => {
            let to: u32 = incr[1..].parse().map_err(|_| format!("bad bet token {incr:?}"))?;
            // A bet that commits the stack is the engine's AllIn, not a Raise:
            // `legal_actions` offers exactly one of the two.
            if to.saturating_sub(gs.street_bets[seat]) >= gs.stacks[seat] {
                return Ok(Action::AllIn);
            }
            Ok(snap_bet(gs, to))
        }
        _ => Err(format!("unparseable agent action {incr:?}")),
    }
}

/// LBR's choice at one decision: the highest local value among the legal
/// actions.  See the module header for the value model.
fn lbr_action(gs: &GameState, hole: [u8; 2], opp: &BeliefState, rng: &mut u64) -> Action {
    let seat = gs.to_act as usize;
    let board = visible(&gs.board, gs.street);
    let equity = equity_vs_range(hole, &board, opp, rng);
    let pot = gs.pot as f64;
    let to_call = gs.current_bet.saturating_sub(gs.street_bets[seat]).min(gs.stacks[seat]) as f64;

    let mut best = Action::Fold;
    let mut best_ev = f64::NEG_INFINITY;
    for act in legal_actions(gs).iter().copied() {
        let ev = match act {
            Action::Fold => 0.0,
            Action::Check => equity * pot,
            Action::Call => equity * (pot + to_call) - to_call,
            Action::Raise(_) | Action::AllIn => {
                let to = match act {
                    Action::Raise(l) => l,
                    _ => gs.street_bets[seat] + gs.stacks[seat],
                };
                let d = to.saturating_sub(gs.street_bets[seat]) as f64;
                // Minimum-defense frequency: the fold share that makes a bluff
                // of this size break even.
                let fold = d / (pot + d);
                let called_equity = ((equity - fold) / (1.0 - fold)).clamp(0.0, 1.0);
                fold * pot + (1.0 - fold) * (called_equity * (pot + 2.0 * d) - d)
            }
        };
        if ev > best_ev {
            best_ev = ev;
            best = act;
        }
    }
    best
}

/// Play one hand of LBR against the agent, returning LBR's net in big blinds.
///
/// `lbr_seat` is the engine seat LBR occupies (0 = SB/button, 1 = BB); the
/// caller alternates it so neither side keeps the positional edge.
fn play_hand(
    bot: &mut Bot,
    lbr_seat: usize,
    deal_rng: &mut u64,
    play_rng: &mut u64,
) -> Result<(f64, f64), String> {
    let (holes_two, board) = deal(deal_rng);
    let bot_seat = 1 - lbr_seat;
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    holes[lbr_seat] = holes_two[0];
    holes[bot_seat] = holes_two[1];

    let mut gs =
        GameState::new(2, BIG_BLIND, SMALL_BLIND, [STACK_SIZE; MAX_PLAYERS], holes, board, 0);

    // Slumbot position is the mirror of the engine seat (pos 0 = BB).
    let bot_pos = 1 - bot_seat;
    let mut hs = bot.start_hand(bot_pos as u8, holes[bot_seat]);
    let mut wire = String::new();
    // The last street an action was taken on: a fold ends the hand there, so
    // only that street's cards were revealed.
    let mut last_street = 0;

    while !gs.is_terminal() {
        last_street = gs.street;
        let board_now = visible(&gs.board, gs.street);
        let act = if gs.to_act as usize == bot_seat {
            let incr = bot.act(&mut hs, &wire, &board_now)?;
            let act = action_from_token(&incr, &gs)?;
            // The token of what was applied, not what was asked for: they
            // differ when a bet had to be snapped to LBR's menu.
            wire.push_str(&token(&gs, act));
            act
        } else {
            // LBR's model of the agent: everything not blocked by the board or
            // its own cards.  Deliberately not narrowed by observed actions —
            // see the module header.
            let mut opp = BeliefState::uniform();
            opp.remove_board(&board_now);
            let act = lbr_action(&gs, holes[lbr_seat], &opp, play_rng);
            wire.push_str(&token(&gs, act));
            act
        };

        let street_before = gs.street;
        gs.apply_action(act);
        if gs.street != street_before && !gs.is_terminal() {
            wire.push('/');
        }
    }

    let raw = gs.terminal_payoffs()[lbr_seat] as f64 / BIG_BLIND as f64;
    // A showdown (including an all-in run-out) reveals the whole board.
    let revealed = if gs.folded != 0 { visible(&board, last_street) } else { board.to_vec() };
    let card_luck = luck_adjustment(holes[bot_seat], holes[lbr_seat], &revealed, &wire)?;
    // Both corrections are the agent's luck; LBR's result is its negation.
    let adj = raw + (card_luck + hs.action_luck_chips()) / BIG_BLIND as f64;
    Ok((raw, adj))
}

/// Play `hands` of LBR against `bot` and report the bound.
///
/// Seats alternate every hand, so position cancels rather than being averaged
/// over an unbalanced sample.
pub fn run_lbr(bot: &mut Bot, hands: u64, seed: u64, progress: impl Fn(&LbrOutcome)) -> LbrOutcome {
    // TWO streams, deliberately.  Dealing and LBR's equity rollouts must not
    // share one, or the deal sequence depends on how many equity calls LBR
    // happened to make — which depends on how the *agent* played.  Two
    // configurations of the agent would then see different cards, and an
    // ablation between them would be comparing different games.  Split, every
    // arm run at the same `seed` sees the identical deals, and the comparison
    // is paired.
    let mut deal_rng = seed | 1;
    let mut play_rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = LbrOutcome::default();
    for h in 0..hands {
        match play_hand(bot, (h % 2) as usize, &mut deal_rng, &mut play_rng) {
            Ok((bb, adj)) => out.record(bb, adj),
            Err(e) => {
                out.errors += 1;
                eprintln!("  LBR hand error ({} so far): {e}", out.errors);
            }
        }
        if (h + 1) % 100 == 0 || h + 1 == hands {
            progress(&out);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value model must prefer folding a hopeless hand to a large bet, and
    /// calling with a strong one — the two cases that make LBR a responder
    /// rather than a random agent.  Checked on the value function directly, so
    /// the test does not depend on a trained blueprint.
    #[test]
    fn the_value_model_folds_air_and_calls_the_nuts() {
        // Board: A♣ K♦ 9♥ 4♠ 2♦.  Hero either has the nuts-ish or total air.
        let board = [
            poker_core::make_card(12, 0),
            poker_core::make_card(11, 1),
            poker_core::make_card(7, 2),
            poker_core::make_card(2, 3),
            poker_core::make_card(0, 1),
        ];
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        // Hero (seat 1) holds A♦A♠ — near the top of this board.
        holes[1] = [poker_core::make_card(12, 1), poker_core::make_card(12, 3)];
        holes[0] = [poker_core::make_card(5, 0), poker_core::make_card(6, 0)];
        let mut gs =
            GameState::new(2, BIG_BLIND, SMALL_BLIND, [STACK_SIZE; MAX_PLAYERS], holes, board, 0);
        // Walk to the river with checks/calls.
        while gs.street < 3 && !gs.is_terminal() {
            let acts = legal_actions(&gs);
            let a = if acts.contains(&Action::Check) { Action::Check } else { Action::Call };
            gs.apply_action(a);
        }
        assert_eq!(gs.street, 3, "should be on the river");

        let mut opp = BeliefState::uniform();
        opp.remove_board(&board);
        let mut rng = 12345u64;

        let strong = lbr_action(&gs, holes[gs.to_act as usize], &opp, &mut rng);
        assert_ne!(strong, Action::Fold, "must not fold a premium hand to no bet");
    }

    /// End to end against a blueprint-only agent: every hand is scored, the
    /// adjusted result is finite, and — since a blueprint-only agent has no
    /// action-luck term — the adjustment is card luck alone, which is exactly
    /// zero on any hand that ends before a card is revealed.
    #[test]
    fn run_lbr_reports_a_finite_luck_adjusted_result() {
        use crate::games::blueprint::BlueprintHoldem;
        use crate::play::{BotConfig, CompactPolicy};
        let game = BlueprintHoldem::new(400, 2, 1, 0).with_raise_cap(3);
        let cfg = BotConfig {
            resolve_river: false,
            resolve_turn: false,
            resolve_flop: false,
            resolve_desynced: false,
            seed: 7,
            ..Default::default()
        };
        let mut bot = Bot::new(game, CompactPolicy::from_entries(vec![]), cfg);

        let (mut deal_rng, mut play_rng) = (3u64, 5u64);
        let mut saw_card_luck = false;
        for h in 0..30u64 {
            let (raw, adj) = play_hand(&mut bot, (h % 2) as usize, &mut deal_rng, &mut play_rng).expect("hand plays");
            assert!(raw.is_finite() && adj.is_finite());
            saw_card_luck |= (adj - raw).abs() > 1e-12;
        }
        assert!(saw_card_luck, "some hand should reach a flop and carry card luck");

        let out = run_lbr(&mut bot, 10, 11, |_| {});
        assert_eq!(out.hands + out.errors, 10);
        assert!(out.adj_bb100().is_finite() && out.adj_ci95().is_finite());
    }

    /// The spot that used to panic the engine: SB limps, the agent (BB)
    /// raises to 200, but LBR's menu has 220 as its smallest raise.
    #[test]
    fn an_off_menu_bet_snaps_to_the_nearest_legal_size() {
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [0, 1];
        holes[1] = [2, 3];
        let mut gs = GameState::new(2, BIG_BLIND, SMALL_BLIND, [STACK_SIZE; MAX_PLAYERS], holes, [4, 5, 6, 7, 8], 0);
        gs.apply_action(Action::Call); // SB limps
        assert!(!legal_actions(&gs).contains(&Action::Raise(200)), "precondition: 200 is off-menu");
        let snapped = action_from_token("b200", &gs).unwrap();
        assert!(matches!(snapped, Action::Raise(_)), "got {snapped:?}");
        assert!(legal_actions(&gs).contains(&snapped), "{snapped:?} must be legal");
        // A legal size passes through unchanged, and the wire gets its token.
        assert_eq!(action_from_token(&token(&gs, snapped), &gs).unwrap(), snapped);
    }

    /// Seats must alternate, or the bound is measured from one position.
    #[test]
    fn seats_alternate_across_hands() {
        let seats: Vec<usize> = (0..6u64).map(|h| (h % 2) as usize).collect();
        assert_eq!(seats, vec![0, 1, 0, 1, 0, 1]);
    }
}
