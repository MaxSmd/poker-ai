//! Terminal payoffs and hand evaluation.
//!
//! [`GameState::terminal_payoffs`] is the money-correct half of the engine:
//! side pots, all-in-for-less eligibility, and odd-chip assignment.  Chip
//! conservation across every path is gated by the tests.

use crate::lut_eval::{evaluate_5_lut, evaluate_6_lut, evaluate_7_lut};
use crate::state::{GameState, MAX_PLAYERS};

impl GameState {
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
                evaluate_7_lut(&cards)
            }
            4 => {
                let cards: [u8; 6] = [
                    h[0], h[1],
                    self.board[0], self.board[1], self.board[2], self.board[3],
                ];
                evaluate_6_lut(&cards)
            }
            3 => {
                let cards: [u8; 5] = [h[0], h[1], self.board[0], self.board[1], self.board[2]];
                evaluate_5_lut(&cards)
            }
            _ => 0,
        }
    }
}
