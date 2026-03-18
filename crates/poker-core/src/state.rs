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
//! - Small blind : `(b + 1) % n`
//! - Big blind   : `(b + 2) % n`
//! - UTG (first to act preflop) : `(b + 3) % n`

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
    pub fn new(
        num_players: u8,
        big_blind: u32,
        stacks: [u32; MAX_PLAYERS],
        hole_cards: [[u8; 2]; MAX_PLAYERS],
        board: [u8; 5],
        button: u8,
    ) -> Self {
        let n = num_players as usize;
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
        let sb = (button as usize + 1) % n;
        let bb = (button as usize + 2) % n;

        let sb_amount = (big_blind / 2).min(gs.stacks[sb]);
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
            // Heads-up: button (who is also SB) acts first preflop.
            (button as usize + 1) % n
        } else {
            (button as usize + 3) % n
        };
        gs.to_act = first_to_act as u8;

        // Every player gets one voluntary action; BB also has the option to raise
        // even if no one else has raised.  Count how many active players need to act:
        // if BB is already all-in (e.g., short stack), they don't need an action.
        let active = gs.count_active();
        // Number of players who still need to voluntarily act = all active players,
        // but BB counts even though they posted (they get the "option").
        // We represent this by setting players_to_act = active (which includes BB when
        // not all-in).  The extra "+1 for BB option" is already embedded because BB is
        // counted in count_active() as long as they have chips remaining.
        gs.players_to_act = active;

        gs
    }

    // ------------------------------------------------------------------
    // Core traversal API
    // ------------------------------------------------------------------

    /// Apply `action` for the current player, updating state in-place.
    /// Pushes an undo record so that `undo_action` can reverse the change.
    /// No heap allocation occurs in the hot path.
    pub fn apply_action(&mut self, action: Action) {
        // Save snapshot for undo.
        let record = UndoRecord {
            action,
            stacks: self.stacks,
            street_bets: self.street_bets,
            total_committed: self.total_committed,
            street: self.street,
            to_act: self.to_act,
            current_bet: self.current_bet,
            min_raise: self.min_raise,
            folded: self.folded,
            allin: self.allin,
            last_aggressor: self.last_aggressor,
            players_to_act: self.players_to_act,
        };
        self.undo.push(record);

        let p = self.to_act as usize;

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
                    // All-in raise: all currently active players need to respond.
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
    }

    /// Undo the last applied action, restoring state exactly.
    pub fn undo_action(&mut self) {
        if let Some(rec) = self.undo.pop() {
            self.stacks = rec.stacks;
            self.street_bets = rec.street_bets;
            self.total_committed = rec.total_committed;
            self.street = rec.street;
            self.to_act = rec.to_act;
            self.current_bet = rec.current_bet;
            self.min_raise = rec.min_raise;
            self.folded = rec.folded;
            self.allin = rec.allin;
            self.last_aggressor = rec.last_aggressor;
            self.players_to_act = rec.players_to_act;
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
            // This side pot: each eligible player contributed (level - prev_level) chips.
            // Eligible: not folded AND total_committed >= level.
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
            let side_pot = eligible_count * (level - prev_level);

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

            // Distribute side pot evenly (integer division; remainder goes to first winner).
            let share = side_pot / num_winners;
            let remainder = side_pot % num_winners;
            let mut first = true;
            for &w in winners.iter().filter(|&&w| w != usize::MAX) {
                let extra = if first { remainder } else { 0 };
                payoffs[w] += (share + extra) as i32;
                first = false;
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
        debug_assert!(false, "next_active_player: no active player found — invalid game state");
        from // fallback: return same player to avoid out-of-bounds
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
        debug_assert!(false, "first_active_after_button: no active player found — invalid game state");
        self.button // fallback
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

// ------------------------------------------------------------------
// Suit / rank convenience re-exports so callers don't need to import
// the evaluator module directly.
// ------------------------------------------------------------------

/// Decoded suit of a card byte.
pub use crate::evaluator::suit_of as card_suit;
/// Decoded rank of a card byte.
pub use crate::evaluator::rank_of as card_rank;

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
        GameState::new(num_players, 10, stacks, default_holes(), default_board(), 0)
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
        // Heads-up preflop: player 1 (SB/button) acts first.
        // Both call/check to close the street.
        gs.apply_action(Action::Call);  // SB calls
        gs.apply_action(Action::Check); // BB checks
        // Street should now be flop (1).
        assert_eq!(gs.street, 1);
    }

    #[test]
    fn raise_resets_players_to_act() {
        let mut gs = make_game(6);
        // UTG (first to act preflop) raises.
        gs.apply_action(Action::Raise(30));
        // All active players except the raiser now need to act.
        // players_to_act >= 1 (there are opponents).
        assert!(gs.players_to_act >= 1);
    }
}
