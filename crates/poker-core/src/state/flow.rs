//! Action application, undo, and turn/street flow.
//!
//! `apply_action`/`undo_action` are the mutate-and-restore pair the CFR hot
//! path walks the tree with: no heap allocation, every change recorded on the
//! pre-allocated undo stack.  The private walkers below own the "whose turn is
//! it, and has the street ended" logic they depend on.

use crate::action::Action;
use crate::state::{GameState, MAX_PLAYERS};
use crate::undo::UndoRecord;

impl GameState {
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
                legal, self.to_act, self.street, self.current_bet, self.pot
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
            old_pot: self.pot,
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
                self.pot += actual;
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
                self.pot += actual;

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
                self.pot += amount;
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
            self.pot = rec.old_pot;
        }
    }

    // ------------------------------------------------------------------
    // Query helpers
    // ------------------------------------------------------------------

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
}
