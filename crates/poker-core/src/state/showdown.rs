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
        if self.is_showdown() {
            self.showdown_payoffs(&self.showdown_ranks())
        } else {
            self.fold_payoffs()
        }
    }

    /// Whether this terminal node is decided by comparing hands (two or more
    /// players never folded) rather than by everyone else folding.
    pub fn is_showdown(&self) -> bool {
        self.is_terminal() && self.count_non_folded() > 1
    }

    /// Each non-folded player's hand rank on the visible board (`0` for a
    /// folded player) — the hand-evaluation half of a showdown.
    ///
    /// Split from [`showdown_payoffs`](Self::showdown_payoffs) because the two
    /// have different lifetimes in a tree walk: the ranks depend only on the
    /// cards, the payoffs on the betting line.  A sampled CFR traversal reaches
    /// hundreds of showdowns per deal, all with the same cards, and can
    /// evaluate the hands once.
    pub fn showdown_ranks(&self) -> [u32; MAX_PLAYERS] {
        let mut hand_ranks = [0u32; MAX_PLAYERS];
        for (i, hr) in hand_ranks.iter_mut().enumerate().take(self.num_players as usize) {
            if (self.folded >> i) & 1 == 0 {
                *hr = self.player_hand_rank(i);
            }
        }
        hand_ranks
    }

    /// Payoffs at a showdown terminal given the players' `hand_ranks` (from
    /// [`showdown_ranks`](Self::showdown_ranks) at this node, or at any other
    /// showdown of the same deal with the same players still in).
    pub fn showdown_payoffs(&self, hand_ranks: &[u32; MAX_PLAYERS]) -> [i32; MAX_PLAYERS] {
        debug_assert!(self.is_showdown(), "showdown_payoffs at a non-showdown node");
        if self.num_players == 2 {
            return self.heads_up_showdown_payoffs(hand_ranks);
        }
        self.side_pot_payoffs(hand_ranks)
    }

    /// Everyone but one player folded: the last player standing wins the pot.
    fn fold_payoffs(&self) -> [i32; MAX_PLAYERS] {
        let n = self.num_players as usize;
        let mut payoffs = [0i32; MAX_PLAYERS];
        let winner = (0..n).find(|&i| (self.folded >> i) & 1 == 0).unwrap_or(0);
        let pot: i32 = self.total_committed.iter().map(|&c| c as i32).sum();
        for (i, p) in payoffs.iter_mut().enumerate().take(n) {
            *p = -(self.total_committed[i] as i32);
        }
        payoffs[winner] += pot;
        payoffs
    }

    /// The two-player showdown, without the side-pot machinery.
    ///
    /// Heads-up there is one contested pot — twice the smaller commitment —
    /// and anything a player put in above that comes straight back.  So the
    /// winner gains the smaller commitment, the loser pays it, and a tie
    /// returns everyone's chips (the contested pot is even, so there is no odd
    /// chip to assign).  This is what [`side_pot_payoffs`](Self::side_pot_payoffs)
    /// computes for two players, and the tests hold the two to each other.
    fn heads_up_showdown_payoffs(&self, hand_ranks: &[u32; MAX_PLAYERS]) -> [i32; MAX_PLAYERS] {
        let contested = self.total_committed[0].min(self.total_committed[1]) as i32;
        let mut payoffs = [0i32; MAX_PLAYERS];
        match hand_ranks[0].cmp(&hand_ranks[1]) {
            std::cmp::Ordering::Greater => (payoffs[0], payoffs[1]) = (contested, -contested),
            std::cmp::Ordering::Less => (payoffs[0], payoffs[1]) = (-contested, contested),
            std::cmp::Ordering::Equal => {}
        }
        payoffs
    }

    /// The general showdown: split the pot into tiers by commitment and award
    /// each tier to the best hand eligible for it.
    pub(super) fn side_pot_payoffs(&self, hand_ranks: &[u32; MAX_PLAYERS]) -> [i32; MAX_PLAYERS] {
        let n = self.num_players as usize;
        let mut payoffs = [0i32; MAX_PLAYERS];

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
