//! Measure the blueprint's **exact** information-set count and SoA memory
//! footprint for 2-player and 6-max, over a matrix of (stack depth × raise
//! cap), with the current card-bucket counts.
//!
//! ## Why this is a measurement, not an estimate
//!
//! An early version of this tool *guessed* the footprint from a generic
//! betting-tree model and was off by ~400× (it ignored stack depth). The only
//! honest number comes from enumerating the **actual** abstract betting tree —
//! the same recursion as `BlueprintHoldem::walk_tree` (same skeleton deal, same
//! `capped_legal` filter, same `next_raises` bookkeeping), driven by the real
//! `poker-core` engine.
//!
//! Heads-up, `BlueprintHoldem::with_indexing` builds that tree explicitly. For
//! 6-max no game implementation exists yet, but the tree is still exactly
//! enumerable: legal actions depend only on the **public** chip state (stacks /
//! bets / pot / street / masks), never on card identities. Two histories that
//! reach the same public state therefore root *identical* subtrees, so the
//! walker memoizes per-street node/slot counts on the public state and folds
//! the sequence tree into its public-state DAG — exact per-sequence counts at
//! DAG cost. Info sets = Σ over streets (decision nodes × card buckets).
//!
//! **Correctness gate:** the walker must reproduce, to the exact info set, the
//! number the real `with_indexing` produced on the server (heads-up 20 bb /
//! cap-2 / 169·500·500·800 buckets = 4,631,870) — asserted in a unit test and
//! cross-checked at runtime whenever `data/{flop,turn,river}_buckets.bin` are
//! present.
//!
//! ## Footprints reported
//!
//! * **f32 SoA** (the current store): f32 regret + f64 strategy-sum + f32
//!   baseline per (info set, action) slot, plus a 5-byte per-info-set index.
//! * **lean** (the quantized `LeanTable` store, validated on push/fold): `i16`
//!   regret + `u16` strategy-sum + `i16` baseline per slot, paired with Linear
//!   CFR — the regime where integer quantization works (DCFR+bf16 was tried
//!   and rejected; see `lean_table.rs`).  Half the accumulator bytes at equal
//!   measured convergence.
//!
//! Usage (from the repo root):
//!   memory_estimate [flop_buckets] [turn_buckets] [river_buckets]
//! Defaults: 500 500 800 (the current cluster build). Pre-flop is always the
//! 169 canonical classes. `POKER_AI_ESTIMATE_STATES` caps the memo (default
//! 15M states ≈ a few GB); a config that exceeds it is reported as truncated.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use poker_ai::abstraction::bucket_map::BucketMap;
use poker_ai::games::blueprint::BlueprintHoldem;
use poker_ai::games::IndexedGame;
use poker_core::action::{Action, ActionList};
use poker_core::legal_actions;
use poker_core::state::{GameState, MAX_PLAYERS, NO_CARD};

/// Match the trainer's blinds ([`crate::bin::train`]); stack(chips) = bb × BIG_BLIND.
const BIG_BLIND: u32 = 2;
const SMALL_BLIND: u32 = 1;
/// Current store: f32 regret + f64 strategy-sum + f32 baseline per slot
/// (the sum is f64 so long-run averaging cannot freeze below f32 precision).
const BYTES_PER_SLOT: usize = 4 + 8 + 4;
/// Per-info-set index overhead: `offsets` (u32) + `num_actions` (u8).
const INDEX_BYTES_PER_INFOSET: usize = 4 + 1;
/// Lean store (`LeanTable`, validated): `i16` regret + `u16` strategy-sum +
/// `i16` baseline per slot (a per-info-set scalar baseline was considered and
/// is unsound — a constant control variate cancels out of the correction)…
const LEAN_BYTES_PER_SLOT: usize = 3 * 2;
/// …plus the same per-info-set index as the f32 store.
const LEAN_BYTES_PER_INFOSET: usize = INDEX_BYTES_PER_INFOSET;
/// Card-bucket maps loaded at train time, constant across the whole matrix:
/// flop 1.29M + turn 13.96M + river 123.16M situations × 2 bytes (u16).
const BUCKET_MAP_BYTES: usize = (1_286_792 + 13_960_050 + 123_156_254) * 2;

// ---------------------------------------------------------------------------
// The abstract betting-tree walker (mirrors BlueprintHoldem::walk_tree).
// ---------------------------------------------------------------------------

/// Per-street decision-node and action-slot counts of a (sub)tree, indexed by
/// street 0..=3 (pre-flop / flop / turn / river).
#[derive(Clone, Copy, Default)]
struct Counts {
    nodes: [u64; 4],
    slots: [u64; 4],
}

impl Counts {
    fn add(&mut self, o: &Counts) {
        for s in 0..4 {
            self.nodes[s] = self.nodes[s].saturating_add(o.nodes[s]);
            self.slots[s] = self.slots[s].saturating_add(o.slots[s]);
        }
    }
}

/// Everything the engine's legal-action generator and street bookkeeping can
/// read — card identities excluded (they never affect betting legality). Two
/// nodes with equal keys root identical subtrees.
#[derive(Hash, PartialEq, Eq)]
struct PublicKey {
    street: u8,
    to_act: u8,
    players_to_act: u8,
    folded: u8,
    allin: u8,
    street_raises: u8,
    current_bet: u32,
    min_raise: u32,
    pot: u32,
    stacks: [u32; MAX_PLAYERS],
    street_bets: [u32; MAX_PLAYERS],
}

impl PublicKey {
    fn of(gs: &GameState, street_raises: u8) -> Self {
        Self {
            street: gs.street,
            to_act: gs.to_act,
            players_to_act: gs.players_to_act,
            folded: gs.folded,
            allin: gs.allin,
            street_raises,
            current_bet: gs.current_bet,
            min_raise: gs.min_raise,
            pot: gs.pot,
            stacks: gs.stacks,
            street_bets: gs.street_bets,
        }
    }
}

/// `POKER_AI_ESTIMATE_ALLIN_AT_CAP=1` sizes the *proposed* tree, in which
/// `AllIn` stays legal at the cap so a raise war always has a terminating
/// action.  Under the current rule, aggression past the cap has no abstract
/// node at all and the agent cannot translate an opponent's shove.
fn allin_at_cap() -> bool {
    std::env::var("POKER_AI_ESTIMATE_ALLIN_AT_CAP").is_ok_and(|v| v == "1")
}

/// Mirror of `BlueprintHoldem::capped_legal` (private there; the HU data-point
/// test gates the two staying in lock-step): at/over the cap drop all `Raise`s
/// and any *voluntary* `AllIn`, but keep a forced all-in call.
fn capped_legal(gs: &GameState, street_raises: u8, raise_cap: u32) -> ActionList {
    let full = legal_actions(gs);
    if (street_raises as u32) < raise_cap {
        return full;
    }
    let keep_allin = allin_at_cap();
    let has_passive = full.iter().any(|a| matches!(a, Action::Check | Action::Call));
    let mut buf = [Action::Fold; 8];
    let mut n = 0;
    for &a in full.iter() {
        let drop = matches!(a, Action::Raise(_))
            || (matches!(a, Action::AllIn) && has_passive && !keep_allin);
        if !drop {
            buf[n] = a;
            n += 1;
        }
    }
    ActionList::from_actions(&buf[..n])
}

/// Mirror of `BlueprintHoldem::next_raises`: reset on a street change, +1 when
/// the bet level rose, unchanged otherwise.
fn next_raises(prev: u8, old_street: u8, old_bet: u32, gs: &GameState) -> u8 {
    if gs.street != old_street {
        0
    } else if gs.current_bet > old_bet {
        prev.saturating_add(1)
    } else {
        prev
    }
}

struct Walker {
    raise_cap: u32,
    memo: HashMap<PublicKey, Counts>,
    limit: usize,
    truncated: bool,
}

impl Walker {
    /// Exact per-sequence counts of the subtree at `gs`, folded over the
    /// public-state DAG. Walks with the engine's own apply/undo (zero clones).
    fn count(&mut self, gs: &mut GameState, street_raises: u8) -> Counts {
        if gs.is_terminal() {
            return Counts::default();
        }
        let key = PublicKey::of(gs, street_raises);
        if let Some(c) = self.memo.get(&key) {
            return *c;
        }
        if self.truncated || self.memo.len() >= self.limit {
            self.truncated = true;
            return Counts::default();
        }
        let acts = capped_legal(gs, street_raises, self.raise_cap);
        let s = gs.street as usize;
        let mut c = Counts::default();
        c.nodes[s] = 1;
        c.slots[s] = acts.len() as u64;
        for i in 0..acts.len() {
            let (old_street, old_bet) = (gs.street, gs.current_bet);
            gs.apply_action(acts[i]);
            let sr = next_raises(street_raises, old_street, old_bet, gs);
            let child = self.count(gs, sr);
            gs.undo_action();
            c.add(&child);
        }
        // A truncated expansion holds partial counts — don't poison the memo.
        if !self.truncated {
            self.memo.insert(key, c);
        }
        c
    }
}

/// Skeleton deal: any distinct real cards drive the public tree (identities are
/// irrelevant to betting legality) — same trick as `BlueprintHoldem`.
fn skeleton(num_players: u8, stack: u32) -> GameState {
    let n = num_players as usize;
    let mut holes = [[NO_CARD; 2]; MAX_PLAYERS];
    for (p, h) in holes.iter_mut().take(n).enumerate() {
        *h = [(2 * p) as u8, (2 * p + 1) as u8];
    }
    let board: [u8; 5] = std::array::from_fn(|i| (2 * n + i) as u8);
    let mut stacks = [0u32; MAX_PLAYERS];
    stacks[..n].fill(stack);
    GameState::new(num_players, BIG_BLIND, SMALL_BLIND, stacks, holes, board, 0)
}

struct Measurement {
    counts: Counts,
    states: usize,
    truncated: bool,
    secs: f64,
}

fn enumerate_tree(num_players: u8, stack_bb: u32, cap: u32, limit: usize) -> Measurement {
    let mut gs = skeleton(num_players, stack_bb * BIG_BLIND);
    let mut w = Walker { raise_cap: cap.max(1), memo: HashMap::new(), limit, truncated: false };
    let t = Instant::now();
    let counts = w.count(&mut gs, 0);
    Measurement { counts, states: w.memo.len(), truncated: w.truncated, secs: t.elapsed().as_secs_f64() }
}

/// `(info sets, action slots)` for per-street betting counts × card buckets.
fn table_size(counts: &Counts, buckets: [u64; 4]) -> (u64, u64) {
    let info_sets = (0..4).map(|s| counts.nodes[s] * buckets[s]).sum();
    let slots = (0..4).map(|s| counts.slots[s] * buckets[s]).sum();
    (info_sets, slots)
}

fn f32_bytes(info_sets: u64, slots: u64) -> u64 {
    slots * BYTES_PER_SLOT as u64 + info_sets * INDEX_BYTES_PER_INFOSET as u64
}

fn lean_bytes(info_sets: u64, slots: u64) -> u64 {
    slots * LEAN_BYTES_PER_SLOT as u64 + info_sets * LEAN_BYTES_PER_INFOSET as u64
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

fn fmt_count(x: u64) -> String {
    let x = x as f64;
    if x < 1e3 {
        format!("{x:.0}")
    } else if x < 1e6 {
        format!("{:.1}K", x / 1e3)
    } else if x < 1e9 {
        format!("{:.2}M", x / 1e6)
    } else if x < 1e12 {
        format!("{:.2}B", x / 1e9)
    } else {
        format!("{:.2}T", x / 1e12)
    }
}

fn fmt_bytes(b: u64) -> String {
    let b = b as f64;
    if b < 1e9 {
        format!("{:.0} MB", b / 1e6)
    } else if b < 1e12 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.2} TB", b / 1e12)
    }
}

// ---------------------------------------------------------------------------
// Optional runtime cross-check against the real dense index (needs full maps).
// ---------------------------------------------------------------------------

const STREETS: [(usize, &str); 3] = [(0, "flop"), (1, "turn"), (2, "river")];

fn cross_check_hu(dir: &Path, limit: usize) {
    if !STREETS.iter().all(|(_, name)| dir.join(format!("{name}_buckets.bin")).exists()) {
        println!(
            "(cross-check vs the real BlueprintHoldem index skipped — \
             data/*_buckets.bin incomplete on this machine; the unit-test gate \
             against the recorded server measurement still applies)\n"
        );
        return;
    }
    let mut buckets = [169u64, 0, 0, 0];
    let mut game = BlueprintHoldem::new(20 * BIG_BLIND, BIG_BLIND, SMALL_BLIND, 0).with_raise_cap(2);
    for (street, name) in STREETS {
        let map = BucketMap::load(dir.join(format!("{name}_buckets.bin")))
            .unwrap_or_else(|e| panic!("load {name}_buckets.bin: {e}"));
        buckets[street + 1] = map.num_buckets() as u64;
        game = game.with_street_bucket(street, map);
    }
    let game = game.with_indexing();
    let real = game.info_set_capacity() as u64;
    let (walked, _) = table_size(&enumerate_tree(2, 20, 2, limit).counts, buckets);
    assert_eq!(walked, real, "walker disagrees with the real dense index — investigate before trusting the matrix");
    println!("Cross-check vs the real BlueprintHoldem dense index (HU 20bb cap-2): {} info sets — exact match.\n", fmt_count(real));
}

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let buckets = [
        169u64,
        args.first().copied().unwrap_or(500),
        args.get(1).copied().unwrap_or(500),
        args.get(2).copied().unwrap_or(800),
    ];
    let limit: usize = std::env::var("POKER_AI_ESTIMATE_STATES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15_000_000);
    let env_list = |name: &str, default: Vec<u32>| -> Vec<u32> {
        std::env::var(name)
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .filter(|l: &Vec<u32>| !l.is_empty())
            .unwrap_or(default)
    };
    // Comma lists, e.g. POKER_AI_ESTIMATE_STACKS=20,100,200 POKER_AI_ESTIMATE_PLAYERS=2
    let stacks = env_list("POKER_AI_ESTIMATE_STACKS", vec![20, 50, 100]);
    let players_list = env_list("POKER_AI_ESTIMATE_PLAYERS", vec![2, 6]);

    println!("Blueprint memory — EXACT abstract betting-tree enumeration (real poker-core engine).");
    println!(
        "Card buckets: preflop=169 flop={} turn={} river={}  |  memo limit {} public states",
        buckets[1], buckets[2], buckets[3], fmt_count(limit as u64)
    );
    println!(
        "Stores: f32 SoA = {BYTES_PER_SLOT} B/slot + {INDEX_BYTES_PER_INFOSET} B/info-set; \
         lean = {LEAN_BYTES_PER_SLOT} B/slot + {LEAN_BYTES_PER_INFOSET} B/info-set \
         (i16 regret + u16 strat-sum + f32 baseline/info-set, Linear-MCCFR regime).\n\
         Train RAM ≈ table + {} bucket maps. Resident RAM is lower still: untouched\n\
         slots (unreached sequence×bucket combos) never commit a page.\n",
        fmt_bytes(BUCKET_MAP_BYTES as u64)
    );

    cross_check_hu(Path::new("data"), limit);

    for &players in &players_list {
        let players = players as u8;
        println!("── {players}-player ─────────────────────────────────────────────────────────────────────────");
        println!(
            "{:>7} {:>4}  {:>26}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>7}",
            "stack", "cap", "betting nodes pf/f/t/r", "info sets", "slots", "f32 RAM", "lean RAM", "DAG", "walk"
        );
        for &stack_bb in &stacks {
            for cap in 1..=3u32 {
                let m = enumerate_tree(players, stack_bb, cap, limit);
                if m.truncated {
                    println!(
                        "{:>5}bb {:>4}  aborted: > {} public states (raise POKER_AI_ESTIMATE_STATES)",
                        stack_bb,
                        cap,
                        fmt_count(limit as u64)
                    );
                    continue;
                }
                let (info_sets, slots) = table_size(&m.counts, buckets);
                let nodes = m
                    .counts
                    .nodes
                    .iter()
                    .map(|&n| fmt_count(n))
                    .collect::<Vec<_>>()
                    .join("/");
                println!(
                    "{:>5}bb {:>4}  {:>26}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>6.1}s",
                    stack_bb,
                    cap,
                    nodes,
                    fmt_count(info_sets),
                    fmt_count(slots),
                    fmt_bytes(f32_bytes(info_sets, slots)),
                    fmt_bytes(lean_bytes(info_sets, slots)),
                    fmt_count(m.states as u64),
                    m.secs,
                );
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE gate: the walker must reproduce, exactly, the info-set count the
    /// real `BlueprintHoldem::with_indexing` reported on the server run
    /// (heads-up, 20 bb, cap-2, buckets 169/500/500/800 → 4,631,870).
    #[test]
    fn walker_reproduces_the_measured_hu_blueprint() {
        let m = enumerate_tree(2, 20, 2, usize::MAX);
        assert!(!m.truncated);
        let (info_sets, slots) = table_size(&m.counts, [169, 500, 500, 800]);
        assert_eq!(info_sets, 4_631_870);
        // ~2.5 slots/info set exactly counted; 16 B/slot (f64 strategy sums)
        // + 5 B index ≈ 0.21 GB.
        let bytes = f32_bytes(info_sets, slots) as f64;
        assert!((0.19e9..0.23e9).contains(&bytes), "got {}", fmt_bytes(bytes as u64));
    }

    /// 6-max plumbing smoke test: micro stacks collapse the tree to a
    /// shove-fest that must terminate quickly with a non-empty pre-flop street.
    #[test]
    fn six_max_walker_terminates_on_micro_stacks() {
        let m = enumerate_tree(6, 2, 1, usize::MAX);
        assert!(!m.truncated);
        assert!(m.counts.nodes[0] > 0);
    }

    #[test]
    fn truncation_is_flagged_not_silent() {
        let m = enumerate_tree(2, 20, 2, 10);
        assert!(m.truncated);
    }

    #[test]
    fn fmt_helpers() {
        assert_eq!(fmt_count(4_631_870), "4.63M");
        assert_eq!(fmt_bytes(130_000_000), "130 MB");
        assert_eq!(fmt_bytes(53_000_000_000), "53.00 GB");
    }
}
