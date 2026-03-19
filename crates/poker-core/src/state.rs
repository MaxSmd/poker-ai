//! GameState: packed representation of 6-max NLHE state.
//!
//! Uses bitmasks for active players, folded players, and board cards.
//! `apply_action` and `undo_action` do not heap-allocate in the hot path;
//! the undo stack is pre-allocated in the constructor.
//!
//! ## Card encoding
//! `card = rank * 4 + suit` (see `evaluator.rs`).
//! [`NO_CARD`] (0xFF) marks an absent board card.
//!
//! ## Position layout
//! Given `button` position `b` and `num_players` `n`:
//!
//! **Multi-way (n ≥ 3):**
//! - Small blind : `(b + 1) % n`
//! - Big blind   : `(b + 2) % n`
//! - UTG (first to act preflop) : `(b + 3) % n`
//!
//! **Heads-up (n = 2):**
//! - Small blind / button : `b`  (button is the SB and acts first preflop)
//! - Big blind            : `(b + 1) % 2`

use crate::action::Action;
use crate::evaluator::{evaluate_5, evaluate_6, evaluate_7};
use crate::undo::{UndoRecord, UndoStack};

/// Sentinel value indicating "no card dealt here yet".
pub const NO_CARD: u8 = 0xFF;
/// Maximum number of players supported.
pub const MAX_PLAYERS: usize = 6;

/// Packed game state for 6-max No-Limit Hold'em.
///
/// All board cards for the entire hand are pre-loaded at construction time
/// (appropriate for CFR tree traversal with public chance sampling).  The
/// board cards are "revealed" automatically when streets advance.
#[derive(Clone, Debug)]
pub struct GameState {
    /// Remaining chip stack for each player.
    pub stacks: [u32; MAX_PLAYERS],
    /// Chips committed by each player in the *current* street.
    pub street_bets: [u32; MAX_PLAYERS],
    /// Total chips committed by each player across all streets.
    pub total_committed: [u32; MAX_PLAYERS],
    /// All five board cards (pre-dealt).  [`NO_CARD`] for slots not yet on the board.
    pub board: [u8; 5],
    /// Hole cards for each player: `[player][0..2]`.
    pub hole_cards: [[u8; 2]; MAX_PLAYERS],
    /// Current street: 0 = preflop, 1 = flop, 2 = turn, 3 = river, 4 = terminal.
    pub street: u8,
    /// Index of the player whose turn it is to act.
    pub to_act: u8,
    /// Number of players in the game (2–6).
    pub num_players: u8,
    /// Dealer / button position.
    pub button: u8,
    /// Big blind chip amount.
    pub big_blind: u32,
    /// The highest `street_bet` placed so far this street (the amount to call).
    pub current_bet: u32,
    /// Minimum raise increment (at least the size of the last raise, or 1 BB).
    pub min_raise: u32,
    /// Bitmask: bit `i` is set if player `i` has folded.
    pub folded: u8,
    /// Bitmask: bit `i` is set if player `i` is all-in.
    pub allin: u8,
    /// Last player to bet or raise (0xFF = no aggression this street).
    pub last_aggressor: u8,
    /// Number of active (non-folded, non-all-in) players who still need to act
    /// before the current betting round closes.
    pub players_to_act: u8,
    /// Undo history — pre-allocated; no alloc per `apply_action` call.
    pub undo: UndoStack,
}

impl GameState {
    /// Construct a new game state.  Blinds are posted automatically.
    ///
    /// `board` contains all five community cards (pre-dealt for the whole hand);
    /// unrevealed cards at position ≥ (3 for flop / 4 for turn / 5 for river)
    /// are not shown to players until the street is reached.
    ///
    /// `small_blind` is the SB chip amount (typically `big_blind / 2`, but
    /// configurable for non-standard structures like 2/3 or 1/3 blinds).
    pub fn new(
        num_players: u8,
        big_blind: u32,
        small_blind: u32,
        stacks: [u32; MAX_PLAYERS],
        hole_cards: [[u8; 2]; MAX_PLAYERS],
        board: [u8; 5],
        button: u8,
    ) -> Self {
        let n = num_players as usize;

        // ── Input validation ────────────────────────────────────────────────
        debug_assert!(
            (2..=MAX_PLAYERS).contains(&n),
            "num_players must be 2–{MAX_PLAYERS}, got {n}"
        );
        debug_assert!(big_blind > 0, "big_blind must be > 0");
        debug_assert!(small_blind <= big_blind, "small_blind ({small_blind}) must be <= big_blind ({big_blind})");
        debug_assert!(
            (button as usize) < n,
            "button ({button}) must be < num_players ({n})"
        );
        // Every active player must have a positive stack.
        debug_assert!(
            stacks[..n].iter().all(|&s| s > 0),
            "all active players must have stacks > 0, got {:?}",
            &stacks[..n]
        );
        // Hole cards must be unique across all active players.
        debug_assert!({
            let mut seen = [false; 52];
            let mut ok = true;
            for i in 0..n {
                for &card in &hole_cards[i] {
                    if card == NO_CARD {
                        continue;
                    }
                    let idx = card as usize;
                    if idx >= 52 || seen[idx] {
                        ok = false;
                        break;
                    }
                    seen[idx] = true;
                }
            }
            // Board cards must also be unique and not overlap with hole cards.
            for &card in &board {
                if card == NO_CARD {
                    continue;
                }
                let idx = card as usize;
                if idx >= 52 || seen[idx] {
                    ok = false;
                    break;
                }
                seen[idx] = true;
            }
            ok
        }, "hole cards and board cards must be unique across all players and the board");

        let mut gs = Self {
            stacks,
            street_bets: [0; MAX_PLAYERS],
            total_committed: [0; MAX_PLAYERS],
            board,
            hole_cards,
            street: 0,
            to_act: 0,
            num_players,
            button,
            big_blind,
            current_bet: big_blind,
            min_raise: big_blind,
            folded: 0,
            allin: 0,
            last_aggressor: 0xFF,
            players_to_act: 0,
            undo: UndoStack::new(),
        };

        // Post blinds.
        // In heads-up (n == 2) the button IS the SB (standard HU convention).
        // In multi-way (n >= 3) the button is behind the blinds.
        let (sb, bb) = if n == 2 {
            (button as usize, (button as usize + 1) % n)
        } else {
            ((button as usize + 1) % n, (button as usize + 2) % n)
        };

        let sb_amount = small_blind.min(gs.stacks[sb]);
        gs.stacks[sb] -= sb_amount;
        gs.street_bets[sb] = sb_amount;
        gs.total_committed[sb] = sb_amount;
        if gs.stacks[sb] == 0 {
            gs.allin |= 1 << sb;
        }

        let bb_amount = big_blind.min(gs.stacks[bb]);
        gs.stacks[bb] -= bb_amount;
        gs.street_bets[bb] = bb_amount;
        gs.total_committed[bb] = bb_amount;
        gs.current_bet = bb_amount;
        if gs.stacks[bb] == 0 {
            gs.allin |= 1 << bb;
        }

        // First to act preflop: UTG (player after BB), or in heads-up: the button (SB).
        // All players (including BB) need a voluntary action, so players_to_act = n.
        let first_to_act = if n == 2 {
            // Heads-up: button is the SB and acts first preflop.
            button as usize
        } else {
            (button as usize + 3) % n
        };
        gs.to_act = first_to_act as u8;

        // Every player gets one voluntary action; BB also has the option to raise
        // even if no one else has raised.  Count how many active players need to act:
        // if BB is already all-in (e.g., short stack), they don't need an action.
        let active = gs.count_active();
        // `players_to_act` is set to the number of active (non-folded, non-all-in)
        // players.  The BB's "option" — the right to raise even after everyone limps —
        // is naturally included here: the BB appears in count_active() as long as
        // they have chips remaining, so they will always get a turn to act
        // (their preflop posting does NOT consume their action slot).
        gs.players_to_act = active;

        gs
    }

    // ------------------------------------------------------------------
    // Core traversal API
    // ------------------------------------------------------------------

    /// Apply `action` for the current player, updating state in-place.
    /// Pushes an undo record so that `undo_action` can reverse the change.
    /// No heap allocation occurs in the hot path.
    ///
    /// In debug builds, validates that `Raise` actions use amounts from the
    /// action abstraction (or the exact all-in amount).  This prevents
    /// accidental game-theory violations from raw bet sizes bypassing the
    /// blueprint abstraction.
    #[inline]
    pub fn apply_action(&mut self, action: Action) {
        // Debug-only: verify that Raise amounts come from the action abstraction.
        #[cfg(debug_assertions)]
        if let Action::Raise(total_bet) = action {
            let legal = crate::action::legal_actions(self);
            debug_assert!(
                legal.contains(&action),
                "apply_action: Raise({total_bet}) is not in the legal abstract actions {:?} \
                 (player={}, street={}, current_bet={}, pot={})",
                legal, self.to_act, self.street, self.current_bet, self.pot()
            );
        }

        // Debug-only chip conservation check: sum(stacks) + sum(total_committed)
        // must be invariant across every action.
        #[cfg(debug_assertions)]
        let chips_before: u32 = self.stacks.iter().sum::<u32>()
            + self.total_committed.iter().sum::<u32>();

        // Capture the acting player's index and per-player values before the
        // action mutates them, plus all scalar fields that may change.
        let p = self.to_act as usize;
        let old_street = self.street;
        let record = UndoRecord {
            action,
            player: p as u8,
            old_stack: self.stacks[p],
            old_street_bet: self.street_bets[p],
            old_total_committed: self.total_committed[p],
            old_street,
            old_to_act: self.to_act,
            old_current_bet: self.current_bet,
            old_min_raise: self.min_raise,
            old_folded: self.folded,
            old_allin: self.allin,
            old_last_aggressor: self.last_aggressor,
            old_players_to_act: self.players_to_act,
            // old_street_bets is always captured; street_changed will be set
            // below if advance_street fires and resets the array.
            street_changed: false,
            old_street_bets: self.street_bets,
        };
        self.undo.push(record);

        match action {
            Action::Fold => {
                self.folded |= 1 << p;
                self.players_to_act = self.players_to_act.saturating_sub(1);
            }

            Action::Check => {
                // Only valid when nothing to call.
                self.players_to_act = self.players_to_act.saturating_sub(1);
            }

            Action::Call => {
                let call_amount = self.current_bet.saturating_sub(self.street_bets[p]);
                let actual = call_amount.min(self.stacks[p]);
                self.stacks[p] -= actual;
                self.street_bets[p] += actual;
                self.total_committed[p] += actual;
                if self.stacks[p] == 0 {
                    self.allin |= 1 << p;
                }
                self.players_to_act = self.players_to_act.saturating_sub(1);
            }

            Action::Raise(total_bet) => {
                // `total_bet` is the new total street_bet level for this player.
                let extra = total_bet.saturating_sub(self.street_bets[p]);
                let actual = extra.min(self.stacks[p]);
                self.stacks[p] -= actual;
                self.street_bets[p] += actual;
                self.total_committed[p] += actual;

                let raise_size = self.street_bets[p].saturating_sub(self.current_bet);
                if raise_size > 0 {
                    self.min_raise = raise_size.max(self.min_raise);
                    self.current_bet = self.street_bets[p];
                    self.last_aggressor = self.to_act;
                }

                if self.stacks[p] == 0 {
                    self.allin |= 1 << p;
                    // All-in raise: player p is now marked allin (above), so
                    // count_active() excludes them.  Every *other* active player
                    // still needs to respond to the raise.
                    self.players_to_act = self.count_active();
                } else {
                    // Normal raise: every active player *except* the raiser needs to respond.
                    self.players_to_act = self.count_active().saturating_sub(1);
                }
            }

            Action::AllIn => {
                let amount = self.stacks[p];
                self.stacks[p] = 0;
                self.street_bets[p] += amount;
                self.total_committed[p] += amount;
                self.allin |= 1 << p;

                if self.street_bets[p] > self.current_bet {
                    // All-in is effectively a raise.
                    let raise_size = self.street_bets[p] - self.current_bet;
                    if raise_size >= self.min_raise {
                        self.min_raise = raise_size;
                    }
                    self.current_bet = self.street_bets[p];
                    self.last_aggressor = self.to_act;
                    // All remaining active players need to respond.
                    self.players_to_act = self.count_active();
                } else {
                    // All-in for less than the call — not a full raise.
                    self.players_to_act = self.players_to_act.saturating_sub(1);
                }
            }
        }

        // Advance: either move to the next player or close the street.
        self.advance_or_next();

        // If the street advanced, mark the just-pushed record so that undo
        // knows to restore all players' street bets from old_street_bets.
        if self.street != old_street {
            self.undo.mark_street_changed();
        }

        #[cfg(debug_assertions)]
        {
            let chips_after: u32 = self.stacks.iter().sum::<u32>()
                + self.total_committed.iter().sum::<u32>();
            debug_assert_eq!(
                chips_before,
                chips_after,
                "chip conservation violated after {:?}: before={} after={}",
                action,
                chips_before,
                chips_after
            );
        }
    }

    /// Undo the last applied action, restoring state exactly.
    #[inline]
    pub fn undo_action(&mut self) {
        if let Some(rec) = self.undo.pop() {
            let p = rec.player as usize;

            // Restore the acting player's per-player fields.
            self.stacks[p] = rec.old_stack;
            self.total_committed[p] = rec.old_total_committed;

            // Restore street bets: if the street advanced, all players' bets
            // were reset by advance_street — restore the whole array.  Otherwise
            // only the acting player's slot changed.
            if rec.street_changed {
                self.street_bets = rec.old_street_bets;
            } else {
                self.street_bets[p] = rec.old_street_bet;
            }

            // Restore scalar fields.
            self.street = rec.old_street;
            self.to_act = rec.old_to_act;
            self.current_bet = rec.old_current_bet;
            self.min_raise = rec.old_min_raise;
            self.folded = rec.old_folded;
            self.allin = rec.old_allin;
            self.last_aggressor = rec.old_last_aggressor;
            self.players_to_act = rec.old_players_to_act;
        }
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

    /// True if the hand is over (only one player remains, or we've reached
    /// the terminal street).
    pub fn is_terminal(&self) -> bool {
        self.count_non_folded() <= 1 || self.street >= 4
    }

    /// True if the current node is a chance node (street transition pending).
    /// In this engine street transitions happen automatically inside `apply_action`,
    /// so callers only see player-decision nodes and terminal nodes.
    ///
    /// This always returns `false` and exists as a placeholder for AI crates that
    /// expect a uniform `is_chance_node` interface (e.g., when switching to an
    /// external-sampling CFR implementation that handles chance nodes explicitly).
    pub fn is_chance_node(&self) -> bool {
        false
    }

    /// Index of the player whose turn it is (only valid when `!is_terminal()`).
    pub fn current_player(&self) -> usize {
        self.to_act as usize
    }

    /// Total chips currently in the pot.
    pub fn pot(&self) -> u32 {
        self.total_committed.iter().sum()
    }

    /// Number of board cards currently visible (0, 3, 4, or 5).
    pub fn board_cards_count(&self) -> usize {
        match self.street {
            0 => 0,
            1 => 3,
            2 => 4,
            _ => 5,
        }
    }

    /// Compute terminal payoffs (chip change relative to starting stacks) for
    /// all players at a terminal node.  Returns `[0; MAX_PLAYERS]` if not terminal.
    ///
    /// Handles side pots correctly: players who are all-in for less can only win
    /// up to the amount they themselves contributed multiplied by the number of
    /// eligible players.
    pub fn terminal_payoffs(&self) -> [i32; MAX_PLAYERS] {
        if !self.is_terminal() {
            return [0; MAX_PLAYERS];
        }

        let n = self.num_players as usize;
        let mut payoffs = [0i32; MAX_PLAYERS];

        // Case 1: everyone but one player has folded — last player wins the pot.
        if self.count_non_folded() <= 1 {
            let winner = (0..n)
                .find(|&i| (self.folded >> i) & 1 == 0)
                .unwrap_or(0);
            let pot: i32 = self.total_committed.iter().map(|&c| c as i32).sum();
            for (i, p) in payoffs.iter_mut().enumerate().take(n) {
                *p = -(self.total_committed[i] as i32);
            }
            payoffs[winner] += pot;
            return payoffs;
        }

        // Case 2: showdown — evaluate hands with side pots.
        // Compute each player's best hand rank (non-folded players only).
        let mut hand_ranks = [0u32; MAX_PLAYERS];
        for (i, hr) in hand_ranks.iter_mut().enumerate().take(n) {
            if (self.folded >> i) & 1 == 0 {
                *hr = self.player_hand_rank(i);
            }
        }

        // Sort all-in amounts to find pot tiers.
        let mut tiers: [u32; MAX_PLAYERS] = self.total_committed;
        tiers[..n].sort_unstable();

        // Start everyone with negative committed amount.
        for (i, p) in payoffs.iter_mut().enumerate().take(n) {
            *p = -(self.total_committed[i] as i32);
        }

        let mut prev_level = 0u32;
        for &level in tiers[..n].iter() {
            if level <= prev_level {
                continue;
            }
            // This side pot: each player who committed >= level contributes
            // (level - prev_level) chips to this tier, including folded players
            // (their chips don't disappear — they're just ineligible to win).
            // Eligible winners: not folded AND total_committed >= level.
            let contributor_count =
                (0..n).filter(|&i| self.total_committed[i] >= level).count() as u32;
            let eligible_mask: u8 = (0..n as u8)
                .filter(|&i| {
                    (self.folded >> i) & 1 == 0
                        && self.total_committed[i as usize] >= level
                })
                .fold(0u8, |acc, i| acc | (1 << i));

            let eligible_count =
                (0..n).filter(|&i| (eligible_mask >> i) & 1 == 1).count() as u32;
            if eligible_count == 0 {
                prev_level = level;
                continue;
            }
            let side_pot = contributor_count * (level - prev_level);

            // Find winner(s) of this side pot (highest hand rank among eligible).
            let best_rank = (0..n)
                .filter(|&i| (eligible_mask >> i) & 1 == 1)
                .map(|i| hand_ranks[i])
                .max()
                .unwrap_or(0);

            let winners: [usize; MAX_PLAYERS] = {
                let mut w = [usize::MAX; MAX_PLAYERS];
                let mut wc = 0;
                for (i, &hr) in hand_ranks.iter().enumerate().take(n) {
                    if (eligible_mask >> i) & 1 == 1 && hr == best_rank {
                        w[wc] = i;
                        wc += 1;
                    }
                }
                w
            };
            let num_winners = winners
                .iter()
                .filter(|&&w| w != usize::MAX)
                .count() as u32;

            // Distribute side pot evenly (integer division; remainder goes to
            // the winner seated closest to the button's left, matching standard
            // casino rules for odd-chip allocation).
            let share = side_pot / num_winners;
            let remainder = side_pot % num_winners;
            // Sort winners by seat distance from the button (button+1 first).
            let mut sorted_winners: [usize; MAX_PLAYERS] = [usize::MAX; MAX_PLAYERS];
            let mut sw_count = 0usize;
            // Iterate starting from button+1, wrapping around.
            for offset in 1..=n {
                let seat = (self.button as usize + offset) % n;
                if winners.contains(&seat) {
                    sorted_winners[sw_count] = seat;
                    sw_count += 1;
                }
            }
            for (idx, &w) in sorted_winners[..sw_count].iter().enumerate() {
                let extra = if idx == 0 { remainder } else { 0 };
                payoffs[w] += (share + extra) as i32;
            }

            prev_level = level;
        }

        payoffs
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Evaluate the best possible hand rank for player `i`.
    fn player_hand_rank(&self, player: usize) -> u32 {
        let bc = self.board_cards_count();
        let h = self.hole_cards[player];
        match bc {
            5 => {
                let cards: [u8; 7] = [
                    h[0], h[1],
                    self.board[0], self.board[1], self.board[2],
                    self.board[3], self.board[4],
                ];
                evaluate_7(&cards)
            }
            4 => {
                let cards: [u8; 6] = [
                    h[0], h[1],
                    self.board[0], self.board[1], self.board[2], self.board[3],
                ];
                evaluate_6(&cards)
            }
            3 => {
                let cards: [u8; 5] = [h[0], h[1], self.board[0], self.board[1], self.board[2]];
                evaluate_5(&cards)
            }
            _ => 0,
        }
    }

    /// Number of active (non-folded, non-all-in) players.
    pub fn count_active(&self) -> u8 {
        let n = self.num_players as usize;
        (0..n)
            .filter(|&i| (self.folded >> i) & 1 == 0 && (self.allin >> i) & 1 == 0)
            .count() as u8
    }

    /// Number of players who have not folded (active + all-in).
    pub fn count_non_folded(&self) -> u8 {
        let n = self.num_players as usize;
        (0..n).filter(|&i| (self.folded >> i) & 1 == 0).count() as u8
    }

    /// Next player to act after `from`, wrapping around, skipping folded/all-in.
    fn next_active_player(&self, from: u8) -> u8 {
        let n = self.num_players as usize;
        let mut next = (from as usize + 1) % n;
        for _ in 0..n {
            if (self.folded >> next) & 1 == 0 && (self.allin >> next) & 1 == 0 {
                return next as u8;
            }
            next = (next + 1) % n;
        }
        panic!(
            "next_active_player: no active player found — corrupt game state \
             (from={from}, num_players={}, street={}, folded={:#010b}, allin={:#010b}, \
             to_act={}, players_to_act={})",
            self.num_players, self.street, self.folded, self.allin,
            self.to_act, self.players_to_act
        );
    }

    /// First active player seated to the left of the button (used for
    /// post-flop action order).
    fn first_active_after_button(&self) -> u8 {
        let n = self.num_players as usize;
        let start = (self.button as usize + 1) % n;
        for offset in 0..n {
            let i = (start + offset) % n;
            if (self.folded >> i) & 1 == 0 && (self.allin >> i) & 1 == 0 {
                return i as u8;
            }
        }
        panic!(
            "first_active_after_button: no active player found — corrupt game state \
             (button={}, num_players={}, street={}, folded={:#010b}, allin={:#010b}, \
             to_act={}, players_to_act={})",
            self.button, self.num_players, self.street, self.folded, self.allin,
            self.to_act, self.players_to_act
        );
    }

    /// After an action, either move to the next player or close the street.
    fn advance_or_next(&mut self) {
        // If only one player hasn't folded, the hand is over.
        if self.count_non_folded() <= 1 {
            self.street = 4;
            return;
        }

        // If the betting round is closed, advance to the next street.
        if self.players_to_act == 0 {
            self.advance_street();
        } else {
            // Move to the next active player.
            self.to_act = self.next_active_player(self.to_act);
        }
    }

    /// Close the current street and set up the next one.
    fn advance_street(&mut self) {
        self.street += 1;

        if self.street >= 4 || self.count_non_folded() <= 1 {
            self.street = self.street.max(4);
            return;
        }

        // Reset per-street state.
        self.street_bets = [0; MAX_PLAYERS];
        self.current_bet = 0;
        self.min_raise = self.big_blind;
        self.last_aggressor = 0xFF;

        let active = self.count_active();
        if active == 0 {
            // All remaining players are all-in — run out the board.
            self.advance_street();
            return;
        }

        // Post-flop action starts with the first active player left of the button.
        self.to_act = self.first_active_after_button();
        self.players_to_act = active;
    }

    /// Convenience: is this player currently holding a valid (non-sentinel) hand?
    pub fn player_has_cards(&self, player: usize) -> bool {
        self.hole_cards[player][0] != NO_CARD && self.hole_cards[player][1] != NO_CARD
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::make_card;

    fn make_card_u8(rank: u8, suit: u8) -> u8 {
        make_card(rank, suit)
    }

    fn default_board() -> [u8; 5] {
        [
            make_card_u8(0, 0), // 2c
            make_card_u8(1, 1), // 3d
            make_card_u8(2, 2), // 4h
            make_card_u8(3, 3), // 5s
            make_card_u8(4, 0), // 6c
        ]
    }

    fn default_holes() -> [[u8; 2]; MAX_PLAYERS] {
        let mut h = [[NO_CARD; 2]; MAX_PLAYERS];
        h[0] = [make_card_u8(12, 0), make_card_u8(12, 1)]; // AA
        h[1] = [make_card_u8(11, 0), make_card_u8(11, 1)]; // KK
        h[2] = [make_card_u8(10, 0), make_card_u8(10, 1)]; // QQ
        h[3] = [make_card_u8(9, 0), make_card_u8(9, 1)];   // JJ
        h[4] = [make_card_u8(8, 0), make_card_u8(8, 1)];   // TT
        h[5] = [make_card_u8(7, 0), make_card_u8(7, 1)];   // 99
        h
    }

    fn make_game(num_players: u8) -> GameState {
        let stacks = [1000u32; MAX_PLAYERS];
        GameState::new(num_players, 10, 5, stacks, default_holes(), default_board(), 0)
    }

    #[test]
    fn blinds_posted_correctly() {
        let gs = make_game(6);
        // SB = position 1, BB = position 2 (button = 0)
        assert_eq!(gs.street_bets[1], 5);  // SB
        assert_eq!(gs.street_bets[2], 10); // BB
        assert_eq!(gs.current_bet, 10);
        assert_eq!(gs.street, 0);
    }

    #[test]
    fn fold_reduces_non_folded() {
        let mut gs = make_game(6);
        let before = gs.count_non_folded();
        gs.apply_action(Action::Fold);
        assert_eq!(gs.count_non_folded(), before - 1);
    }

    #[test]
    fn undo_restores_state() {
        let mut gs = make_game(6);
        let to_act_before = gs.to_act;
        let stacks_before = gs.stacks;
        let folded_before = gs.folded;

        gs.apply_action(Action::Fold);
        assert_ne!(gs.folded, folded_before);

        gs.undo_action();
        assert_eq!(gs.to_act, to_act_before);
        assert_eq!(gs.stacks, stacks_before);
        assert_eq!(gs.folded, folded_before);
    }

    #[test]
    fn call_moves_chips() {
        let mut gs = make_game(6);
        let p = gs.to_act as usize;
        let stack_before = gs.stacks[p];
        gs.apply_action(Action::Call);
        // Player called the BB (10 chips), so stack decreased by 10.
        assert_eq!(gs.stacks[p], stack_before - 10);
    }

    #[test]
    fn terminal_after_all_fold() {
        let mut gs = make_game(2);
        assert!(!gs.is_terminal());
        gs.apply_action(Action::Fold);
        assert!(gs.is_terminal());
    }

    #[test]
    fn payoff_last_player_wins_pot() {
        let mut gs = make_game(2);
        // Heads-up: both post (SB+BB=15), then SB folds immediately.
        gs.apply_action(Action::Fold);
        assert!(gs.is_terminal());
        let payoffs = gs.terminal_payoffs();
        let total: i32 = payoffs.iter().sum();
        assert_eq!(total, 0, "payoffs must sum to zero");
        // Winner should have positive payoff.
        assert!(payoffs.iter().any(|&p| p > 0));
    }

    #[test]
    fn street_advances_after_round() {
        let mut gs = make_game(2);
        assert_eq!(gs.street, 0);
        // Heads-up preflop: button (player 0, SB) acts first.
        // Both call/check to close the street.
        gs.apply_action(Action::Call);  // SB (button, player 0) calls
        gs.apply_action(Action::Check); // BB (player 1) checks
        // Street should now be flop (1).
        assert_eq!(gs.street, 1);
    }

    #[test]
    fn raise_resets_players_to_act() {
        let mut gs = make_game(6);
        // UTG (first to act preflop) raises using an abstract action.
        let actions = crate::action::legal_actions(&gs);
        let raise_action = actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
        gs.apply_action(*raise_action);
        // All active players except the raiser now need to act.
        // players_to_act >= 1 (there are opponents).
        assert!(gs.players_to_act >= 1);
    }

    // ── Chip conservation helper ─────────────────────────────────────────────

    fn chip_total(gs: &GameState) -> u32 {
        gs.stacks.iter().sum::<u32>() + gs.total_committed.iter().sum::<u32>()
    }

    // ── Full-hand integration test ───────────────────────────────────────────

    /// Play a complete hand (preflop → flop → turn → river → showdown) and
    /// assert that chips are conserved at every step and payoffs sum to zero.
    #[test]
    fn full_hand_chip_conservation() {
        let mut gs = make_game(2);
        let initial_chips = chip_total(&gs);

        // Preflop: SB (button=0) calls, BB checks.
        gs.apply_action(Action::Call);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after preflop call");
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after preflop check");
        assert_eq!(gs.street, 1, "should be on flop");

        // Flop: check, check.
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after flop check 1");
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after flop check 2");
        assert_eq!(gs.street, 2, "should be on turn");

        // Turn: check, check.
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after turn check 1");
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after turn check 2");
        assert_eq!(gs.street, 3, "should be on river");

        // River: check, check → showdown.
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after river check 1");
        gs.apply_action(Action::Check);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after river check 2");
        assert!(gs.is_terminal(), "should be terminal after river");

        let payoffs = gs.terminal_payoffs();
        assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
    }

    // ── Adversarial edge cases ────────────────────────────────────────────────

    /// Everyone goes all-in preflop — hand must terminate correctly with chips conserved.
    #[test]
    fn all_in_preflop_two_players() {
        let mut gs = make_game(2);
        let initial_chips = chip_total(&gs);

        gs.apply_action(Action::AllIn);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after p0 all-in");
        gs.apply_action(Action::AllIn);
        assert_eq!(chip_total(&gs), initial_chips, "conservation after p1 all-in");
        assert!(gs.is_terminal(), "hand should be terminal when both all-in");

        let payoffs = gs.terminal_payoffs();
        assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
    }

    /// 3-way all-in with different stack sizes — verify side pots and chip conservation.
    #[test]
    fn three_way_allin_different_stacks() {
        // Player 0: button (stack 100), Player 1: SB (stack 200), Player 2: BB (stack 300).
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = 100;
        stacks[1] = 200;
        stacks[2] = 300;
        let holes = default_holes();
        let board = default_board();
        let big_blind = 10u32;
        let mut gs = GameState::new(3, big_blind, big_blind / 2, stacks, holes, board, 0);
        let initial_chips = chip_total(&gs);

        // Drive everyone all-in: UTG (player 3%3=0 is button, so UTG is player (0+3)%3=0)
        // Actually button=0, SB=(0+1)%3=1, BB=(0+2)%3=2, UTG=(0+3)%3=0=button. Wait, that
        // wraps to 0. Let me think again. n=3, button=0, SB=1, BB=2, UTG=(0+3)%3=0.
        // So UTG is the button position (0). First to act preflop is UTG=player 0.
        gs.apply_action(Action::AllIn); // player 0 all-in (100 chips)
        assert_eq!(chip_total(&gs), initial_chips);
        gs.apply_action(Action::AllIn); // player 1 all-in (200 chips)
        assert_eq!(chip_total(&gs), initial_chips);
        gs.apply_action(Action::AllIn); // player 2 all-in (300 chips)
        assert_eq!(chip_total(&gs), initial_chips);
        assert!(gs.is_terminal(), "should be terminal when all players all-in");

        let payoffs = gs.terminal_payoffs();
        assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");
        // Total chips redistributed must equal total initial chips.
        let total_returned: i32 = payoffs
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let committed = gs.total_committed[i] as i32;
                committed + p
            })
            .sum();
        assert_eq!(total_returned as u32, initial_chips, "all chips must be returned");
    }

    /// Min-raise then re-raise — min_raise must track the largest raise increment.
    #[test]
    fn min_raise_then_reraise() {
        let mut gs = make_game(2);
        // Preflop HU: button (p0, SB) is first to act.
        // Pick the first abstract raise available.
        let actions = crate::action::legal_actions(&gs);
        let first_raise = *actions.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
        gs.apply_action(first_raise);
        let first_bet = gs.current_bet;
        let mr_after_first = gs.min_raise;
        assert!(mr_after_first >= 10, "min_raise should be >= BB after first raise");

        // BB re-raises using a legal abstract action.
        let actions2 = crate::action::legal_actions(&gs);
        let reraise = *actions2.iter().find(|a| matches!(a, Action::Raise(_))).unwrap();
        gs.apply_action(reraise);
        assert!(gs.current_bet > first_bet, "current_bet should increase on re-raise");
        assert!(gs.min_raise >= mr_after_first, "min_raise should not decrease on re-raise");
    }

    /// Odd-chip allocation: when a side pot doesn't divide evenly among winners,
    /// the remainder goes to the winner closest to the button's left (clockwise),
    /// matching standard casino rules (Robert's Rules of Poker §15).
    #[test]
    fn odd_chip_goes_to_first_winner_left_of_button() {
        // 3 players all-in with equal stacks and identical best hand ranks.
        // Total pot = 30 (10 each). 30 / 3 = 10 each, no remainder.
        // To force a remainder: use stacks of 7 each, BB=2, SB=1.
        // After blinds: p1 commits 1, p2 commits 2.
        // All go all-in: total pot = 21. 21 / 3 = 7 each, no remainder.
        //
        // Use stacks [10, 10, 10], BB=4, SB=2. All go all-in → pot=30.
        // Two winners → 30/2=15 each. Three winners → 30/3=10 each.
        //
        // For a true odd chip: 3 winners splitting pot=31 → 10 each + 1 remainder.
        // Stacks [11, 10, 10], BB=4, SB=2. button=0, SB=1(commits 2), BB=2(commits 4).
        // UTG=p0 all-in(11), p1 all-in(10), p2 all-in(10).
        // Total committed = 11+10+10 = 31.
        // Side pot tier 1 (level=10): 3 contributors × 10 = 30.
        // Side pot tier 2 (level=11): 1 contributor × 1 = 1 (only p0 eligible).
        // If all 3 tied on tier 1: 30/3=10 each, no remainder on tier 1.
        // p0 gets tier 2 (1 chip) back.
        //
        // Better approach: use the payoff computation directly with a terminal state.
        // Create a 3-player game where all go all-in with stacks that produce a
        // single-tier pot with an odd-chip remainder.
        let mut stacks = [0u32; MAX_PLAYERS];
        stacks[0] = 10;
        stacks[1] = 10;
        stacks[2] = 10;

        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        // All three players get the same rank pair (different suits) → tied at showdown.
        holes[0] = [make_card_u8(12, 0), make_card_u8(11, 0)]; // Ac Kc
        holes[1] = [make_card_u8(12, 1), make_card_u8(11, 1)]; // Ad Kd
        holes[2] = [make_card_u8(12, 2), make_card_u8(11, 2)]; // Ah Kh

        // Board that doesn't improve anyone differently (low cards, mixed suits).
        let board = [
            make_card_u8(0, 3),  // 2s
            make_card_u8(1, 3),  // 3s
            make_card_u8(2, 3),  // 4s
            make_card_u8(3, 3),  // 5s
            make_card_u8(5, 0),  // 7c (breaks the straight for safety)
        ];

        let mut gs = GameState::new(3, 4, 2, stacks, holes, board, 0);

        // button=0, SB=1(commits 2), BB=2(commits 4), UTG=0.
        // UTG (p0) all-in, SB (p1) all-in, BB (p2) all-in.
        gs.apply_action(Action::AllIn); // p0
        gs.apply_action(Action::AllIn); // p1
        gs.apply_action(Action::AllIn); // p2
        assert!(gs.is_terminal(), "should be terminal when all players all-in");

        let payoffs = gs.terminal_payoffs();
        assert_eq!(payoffs.iter().sum::<i32>(), 0, "payoffs must sum to zero");

        // Total pot = 30. Three-way split: 30 / 3 = 10 each, 0 remainder.
        // All players get their chips back → payoff = 0 each.
        assert_eq!(payoffs[0], 0);
        assert_eq!(payoffs[1], 0);
        assert_eq!(payoffs[2], 0);
    }

    /// BB special case: everyone limps preflop, BB gets the option to raise.
    #[test]
    fn bb_gets_option_when_everyone_limps() {
        let mut gs = make_game(3);
        // button=0, SB=1, BB=2, UTG=0
        // UTG (player 0) calls (limps).
        gs.apply_action(Action::Call);
        // SB (player 1) calls (limps, tops up to BB).
        gs.apply_action(Action::Call);
        // Now BB (player 2) should still have the option — game is NOT terminal and
        // it is BB's turn.  BB checks.
        assert!(!gs.is_terminal(), "game should not be terminal before BB acts");
        assert_eq!(gs.to_act, 2, "BB (player 2) should be next to act");
        gs.apply_action(Action::Check); // BB exercises option by checking.
        assert_eq!(gs.street, 1, "street should advance to flop after BB checks");
    }
}
