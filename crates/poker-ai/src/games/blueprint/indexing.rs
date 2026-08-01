//! The dense info-set index: the abstract betting tree, enumerated once.
//!
//! Legal actions depend only on *public* chip state, never on card identities,
//! so a single skeleton deal enumerates every reachable betting sequence.  Each
//! decision node gets a contiguous block of `buckets_for(street)` slots, giving
//! the bijection `(sequence, bucket) → dense index` that the flat SoA regret
//! store needs — and [`BlueprintHoldem::info_key_at`] inverts it, so an
//! SoA-trained strategy exports to exactly the HashMap path's artifact.

use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

use super::{BlueprintCursor, BlueprintHoldem};
use crate::abstraction::canonical::preflop_index;
use crate::util::hash::Fnv1a;

/// A deal-independent enumeration of the abstract betting tree under the raise
/// cap, mapping every reachable decision **history** to a dense sequence id — the
/// backbone of the flat SoA info-set index.
///
/// The key fact that makes this exact: legal actions depend only on the *public*
/// chip state (pot / current bet / stacks / street), never on card identities.
/// So one skeleton deal enumerates every betting sequence the solver can ever
/// reach, and a dense info-set index is simply `sequence_offset + card_bucket`,
/// where the card bucket ranges over `0..buckets_for(street)`.  This partitions
/// information sets **identically** to the `HashMap` key
/// [`info_key_for`](BlueprintHoldem::info_key_for) (which keys on
/// `player + visible + bucket + history`, and player/visible are themselves pure
/// functions of the history) — proven by `indexed_partition_matches_info_key`.
pub(super) struct Indexing {
    /// Per decision node: child node id for each action index (`-1` = the action
    /// leads to a terminal, i.e. no child decision node).
    children: Vec<[i32; 8]>,
    /// Per node: board cards visible (`0` pre-flop / `3` flop / `4` turn /
    /// `5` river) → which street's bucket count this node draws from.
    visible: Vec<u8>,
    /// Per node: the player to act (only used to reconstruct the info key on
    /// export — [`info_key_at`](BlueprintHoldem::info_key_at)).
    to_act: Vec<u8>,
    /// Per node: number of (capped) legal actions — the flat table's width.
    num_actions: Vec<u8>,
    /// Per node: parent node id (`-1` at the root) and the action index taken
    /// from the parent, to rebuild the perfect-recall history on export.
    parent: Vec<i32>,
    in_action: Vec<u8>,
    /// Per node: base dense index of its `buckets_for(visible)` bucket block.
    seq_offset: Vec<u32>,
    /// Number of legal actions per dense info-set index (drives the table layout).
    actions_by_index: Vec<u8>,
    /// Total info sets = `Σ buckets_for(visible[node])`.
    capacity: usize,
}


impl BlueprintHoldem {
    fn indexing(&self) -> &Indexing {
        self.indexing.as_ref().expect("call with_indexing() before SoA training")
    }

    /// Number of card buckets a street contributes: 169 pre-flop classes, else
    /// the loaded bucket map's count.
    fn buckets_for_visible(&self, visible: u8) -> u32 {
        match visible {
            0 => 169,
            3..=5 => {
                self.street_buckets[visible as usize - 3].as_ref().expect("map present").num_buckets()
            }
            other => unreachable!("impossible board-card count {other}"),
        }
    }

    /// Enumerate the abstract betting tree from a skeleton deal (card identities
    /// are irrelevant to betting legality), then lay out the dense info-set
    /// index by giving each decision node a contiguous `buckets_for(street)` block.
    pub(super) fn build_indexing(&self) -> Indexing {
        // Skeleton deal: any nine distinct real cards drive the public tree.
        let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
        holes[0] = [0, 1];
        holes[1] = [2, 3];
        let board = [4, 5, 6, 7, 8];
        let gs = GameState::new(2, self.big_blind, self.small_blind, self.stacks, holes, board, self.button);

        let mut idx = Indexing {
            children: Vec::new(),
            visible: Vec::new(),
            to_act: Vec::new(),
            num_actions: Vec::new(),
            parent: Vec::new(),
            in_action: Vec::new(),
            seq_offset: Vec::new(),
            actions_by_index: Vec::new(),
            capacity: 0,
        };
        self.walk_tree(&gs, 0, -1, 0, &mut idx);

        let mut cap = 0u32;
        for node in 0..idx.visible.len() {
            idx.seq_offset.push(cap);
            let nb = self.buckets_for_visible(idx.visible[node]);
            for _ in 0..nb {
                idx.actions_by_index.push(idx.num_actions[node]);
            }
            cap += nb;
        }
        idx.capacity = cap as usize;
        idx
    }

    /// Depth-first enumeration: allocate a node for the decision at `gs`, then
    /// recurse over its capped legal actions.  Returns the node id, or `-1` if
    /// `gs` is terminal (so the parent records "no child here").
    fn walk_tree(&self, gs: &GameState, street_raises: u8, parent: i32, in_action: u8, idx: &mut Indexing) -> i32 {
        if gs.is_terminal() {
            return -1;
        }
        let acts = self.capped_legal(gs, street_raises);
        let id = idx.children.len();
        idx.children.push([-1; 8]);
        idx.visible.push(gs.board_cards_count() as u8);
        idx.to_act.push(gs.current_player() as u8);
        idx.num_actions.push(acts.len() as u8);
        idx.parent.push(parent);
        idx.in_action.push(in_action);
        for a in 0..acts.len() {
            let (old_street, old_bet) = (gs.street, gs.current_bet);
            let mut child = gs.clone();
            child.apply_action(acts[a]);
            let sr = Self::next_raises(street_raises, old_street, old_bet, &child);
            let child_id = self.walk_tree(&child, sr, id as i32, a as u8, idx);
            idx.children[id][a] = child_id;
        }
        id as i32
    }

    /// The current player's card bucket, in `0..buckets_for(visible)`.  Mirrors
    /// [`situation_bucket`](BlueprintHoldem::situation_bucket) under full
    /// coverage (the precondition of [`with_indexing`]); an out-of-set situation
    /// — which cannot occur with a full-coverage map — falls back to bucket `0`.
    fn dense_bucket(&self, hole: &[u8; 2], board: &[u8]) -> usize {
        let visible = board.len();
        if visible == 0 {
            return preflop_index(hole) as usize;
        }
        self.street_buckets[visible - 3]
            .as_ref()
            .expect("map present")
            .bucket(hole, board)
            .map(|b| b as usize)
            .unwrap_or(0)
    }

    /// Reconstruct the `HashMap` info key for a dense index, so an SoA-trained
    /// strategy exports to the **same** `HashMap<u64, _>` artifact as the
    /// `HashMap`-solver path (identical bytes — see
    /// [`info_key_for`](BlueprintHoldem::info_key_for)).
    pub fn info_key_at(&self, index: usize) -> u64 {
        let idx = self.indexing();
        // The node owning this index is the last whose block starts at/below it.
        let node = idx.seq_offset.partition_point(|&o| o as usize <= index) - 1;
        let bucket = (index - idx.seq_offset[node] as usize) as u64;

        // Rebuild the perfect-recall history from the parent chain.
        let mut history = Vec::new();
        let mut n = node as i32;
        while idx.parent[n as usize] >= 0 {
            history.push(idx.in_action[n as usize]);
            n = idx.parent[n as usize];
        }
        history.reverse();

        let mut h = Fnv1a::new();
        h.write(idx.to_act[node]);
        h.write(idx.visible[node]);
        h.write_all(&bucket.to_le_bytes());
        h.write(0xFF);
        h.write_all(&history);
        h.finish()
    }
}

impl crate::games::IndexedGame for BlueprintHoldem {
    fn info_set_capacity(&self) -> usize {
        self.indexing().capacity
    }

    /// `sequence_offset + card_bucket`.  The sequence is found by walking the
    /// enumerated betting tree along the cursor's inline history (O(depth), no
    /// allocation); the bucket is the acting player's current-street card bucket.
    fn info_set_index(&self, c: &BlueprintCursor) -> usize {
        let idx = self.indexing();
        let gs = c.gs.as_ref().expect("info_set_index at a play node");

        let mut node = 0usize;
        for &a in &c.history[..c.depth] {
            node = idx.children[node][a as usize] as usize;
        }
        debug_assert_eq!(idx.to_act[node] as usize, gs.current_player(), "tree node player matches");

        let player = gs.current_player();
        let mut hole = gs.hole_cards[player];
        hole.sort_unstable();
        let visible = gs.board_cards_count();
        debug_assert_eq!(idx.visible[node] as usize, visible, "tree node street matches");
        let bucket = self.dense_bucket(&hole, &gs.board[..visible]);

        idx.seq_offset[node] as usize + bucket
    }

    fn actions_at(&self, index: usize) -> usize {
        self.indexing().actions_by_index[index] as usize
    }
}
