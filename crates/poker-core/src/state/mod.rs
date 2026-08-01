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


mod flow;
mod showdown;
#[cfg(test)]
mod tests;

use crate::undo::UndoStack;

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
    /// Cached total chips in the pot (== `total_committed.iter().sum()`).
    /// Maintained incrementally to avoid an O(n) sum on every action.
    pub pot: u32,
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
        // Hole cards must be unique across all active players (enforced in
        // release builds to prevent silent wrong evaluations from duplicates).
        assert!({
            let mut seen = [false; 52];
            let mut ok = true;
            for cards in &hole_cards[..n] {
                for &card in cards {
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
            pot: 0,
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

        gs.pot = sb_amount + bb_amount;

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

    /// Total chips currently in the pot (O(1) — backed by a cached field).
    pub fn pot(&self) -> u32 {
        self.pot
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

    /// Number of active (non-folded, non-all-in) players.
    #[inline]
    pub fn count_active(&self) -> u8 {
        // Build a mask of the num_players low bits, then count those that are
        // neither folded nor all-in.  Uses a single hardware popcount instruction.
        let player_mask = (1u8 << self.num_players) - 1;
        (player_mask & !(self.folded | self.allin)).count_ones() as u8
    }

    /// Number of players who have not folded (active + all-in).
    #[inline]
    pub fn count_non_folded(&self) -> u8 {
        let player_mask = (1u8 << self.num_players) - 1;
        (player_mask & !self.folded).count_ones() as u8
    }

    /// Convenience: is this player currently holding a valid (non-sentinel) hand?
    pub fn player_has_cards(&self, player: usize) -> bool {
        self.hole_cards[player][0] != NO_CARD && self.hole_cards[player][1] != NO_CARD
    }
}
