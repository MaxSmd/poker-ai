//! Estimate the blueprint's information-set count and memory footprint for a
//! given card- and action-abstraction, and compare it against common RAM
//! budgets.  The plan (Memory Budget — Do This First) is explicit: *run this
//! before buying RAM or finalizing bucket counts.*
//!
//! ## What dominates, and how this models it
//!
//! The footprint is `info_sets × actions_per_set × 3 × 4 bytes`.  The `3` is the
//! per-(info set, action) accumulator count in v3 — regret, strategy-sum, and
//! the **baseline** value for VR-MCCFR (v2 had only 2).  The hard part is the
//! **betting-sequence multiplier**: how many distinct public action histories
//! reach a decision on each street.  Rather than guess one, this tool
//! *enumerates the abstract betting tree* with a configurable per-street raise
//! cap, and — crucially — distinguishes the number of *opening-bet* sizes from
//! the number of *re-raise* sizes, because giving every re-raise the full size
//! menu is what makes a naive count explode by orders of magnitude.
//!
//! ## The dominant lever
//!
//! The betting tree compounds multiplicatively across the four streets, so the
//! **raise cap** is the single biggest driver of feasibility.  The report prints
//! a sensitivity row over caps 1–3 so the explosion is visible, not hidden in a
//! single number.
//!
//! ## Modeling caveats (kept honest)
//!
//! * Folds are tracked across streets, so active-player subsets and position
//!   fall out of the enumeration naturally.
//! * The actor at each node is the next player still owing action in seat
//!   order; exact blind/rotation detail is simplified — affects ordering, not
//!   counts materially.
//! * All-in is one extra aggressive branch under the cap, not exact stack math.
//! * Card buckets per street and the bet/raise size counts are inputs (the
//!   plan's targets), not computed.
//!
//! Treat the output as a well-grounded order-of-magnitude figure for the
//! *when do I need a server* decision, not a to-the-byte count.

use std::collections::HashMap;

use poker_core::betting::{FLOP_BET_FRACS, PREFLOP_BET_FRACS, RIVER_BET_FRACS, TURN_BET_FRACS};

const BYTES_PER_ENTRY: f64 = 3.0 * 4.0; // regret + strategy_sum + baseline, f32 each
const STREETS: usize = 4;
const STREET_NAMES: [&str; STREETS] = ["preflop", "flop", "turn", "river"];

/// Abstraction parameters to evaluate.
#[derive(Clone)]
struct Config {
    label: &'static str,
    players: usize,
    /// Card buckets per street (preflop, flop, turn, river).
    buckets: [f64; STREETS],
    /// Number of abstract *opening-bet* sizes per street.
    bet_sizes: [usize; STREETS],
    /// Number of abstract *re-raise* sizes per street (usually fewer).
    raise_sizes: [usize; STREETS],
    /// Whether all-in adds an extra aggressive branch.
    include_allin: bool,
    /// Maximum number of aggressive actions (bets + raises) **per street** — real
    /// systems cap raises lower on deeper streets (and lean on search there).
    max_raises: [u8; STREETS],
    /// Fraction of the naive (bucket × betting-sequence) cross product that is
    /// actually *stored*: `1.0` is the upper-bound enumeration; a value below 1
    /// models that most combinations are unreachable and that imperfect-recall
    /// abstraction collapses many situations into one — the difference between a
    /// textbook count and what a system like Pluribus held in RAM.
    reachable_fraction: f64,
}

/// Per-street tallies produced by the betting-tree enumeration.
#[derive(Clone, Copy, Default)]
struct StreetTally {
    decision_nodes: f64,
    action_slots: f64,
}

/// Key identifying a betting-subtree shape for memoization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StateKey {
    street: u8,
    in_hand: u8,
    need: u8,
    to_act: u8,
    raises: u8,
    has_bet: bool,
}

struct Enumerator {
    players: usize,
    bet_sizes: [usize; STREETS],
    raise_sizes: [usize; STREETS],
    include_allin: bool,
    max_raises: [u8; STREETS],
    memo: HashMap<StateKey, [StreetTally; STREETS]>,
}

impl Enumerator {
    /// Aggressive branches available at a node: opening-bet sizes when there is
    /// no live bet, re-raise sizes when facing one, plus an optional all-in.
    fn aggressive_branches(&self, street: usize, has_bet: bool) -> usize {
        let sizes = if has_bet { self.raise_sizes[street] } else { self.bet_sizes[street] };
        sizes + if self.include_allin { 1 } else { 0 }
    }

    /// Next player still owing action, scanning seats in order from `from`.
    fn next_actor(&self, need: u8, from: u8) -> Option<u8> {
        for off in 0..self.players as u8 {
            let p = (from + off) % self.players as u8;
            if need & (1 << p) != 0 {
                return Some(p);
            }
        }
        None
    }

    /// Enumerate the subtree from a fresh betting round on `street`.
    fn round(&mut self, street: usize, in_hand: u8) -> [StreetTally; STREETS] {
        let need = in_hand;
        let to_act = self.next_actor(need, 0).unwrap_or(0);
        self.node(StateKey { street: street as u8, in_hand, need, to_act, raises: 0, has_bet: false })
    }

    /// Enumerate the subtree rooted at one decision node.
    fn node(&mut self, key: StateKey) -> [StreetTally; STREETS] {
        if let Some(&cached) = self.memo.get(&key) {
            return cached;
        }
        let street = key.street as usize;
        let mut tally = [StreetTally::default(); STREETS];
        let mut actions = 0.0;
        let actor_bit = 1u8 << key.to_act;
        let others = key.in_hand & !actor_bit;

        // Fold (only when facing a bet).
        if key.has_bet {
            actions += 1.0;
            if others.count_ones() >= 2 {
                let need = key.need & !actor_bit;
                add(&mut tally, self.after(street, others, need, key.to_act, key.raises, key.has_bet));
            }
        }

        // Check or call (always legal).
        actions += 1.0;
        {
            let need = key.need & !actor_bit;
            add(&mut tally, self.after(street, key.in_hand, need, key.to_act, key.raises, key.has_bet));
        }

        // Bet or raise (under this street's cap).
        if (key.raises as usize) < self.max_raises[street] as usize {
            for _ in 0..self.aggressive_branches(street, key.has_bet) {
                actions += 1.0;
                let need = key.in_hand & !actor_bit;
                add(&mut tally, self.after(street, key.in_hand, need, key.to_act, key.raises + 1, true));
            }
        }

        tally[street].decision_nodes += 1.0;
        tally[street].action_slots += actions;
        self.memo.insert(key, tally);
        tally
    }

    /// Advance after an action: continue the round, close the street, or end.
    fn after(
        &mut self,
        street: usize,
        in_hand: u8,
        need: u8,
        last_actor: u8,
        raises: u8,
        has_bet: bool,
    ) -> [StreetTally; STREETS] {
        if let Some(next) = self.next_actor(need, (last_actor + 1) % self.players as u8) {
            self.node(StateKey { street: street as u8, in_hand, need, to_act: next, raises, has_bet })
        } else if street + 1 < STREETS {
            self.round(street + 1, in_hand)
        } else {
            [StreetTally::default(); STREETS]
        }
    }
}

fn add(dst: &mut [StreetTally; STREETS], src: [StreetTally; STREETS]) {
    for s in 0..STREETS {
        dst[s].decision_nodes += src[s].decision_nodes;
        dst[s].action_slots += src[s].action_slots;
    }
}

fn enumerate(cfg: &Config, max_raises: [u8; STREETS]) -> [StreetTally; STREETS] {
    let mut e = Enumerator {
        players: cfg.players,
        bet_sizes: cfg.bet_sizes,
        raise_sizes: cfg.raise_sizes,
        include_allin: cfg.include_allin,
        max_raises,
        memo: HashMap::new(),
    };
    let all_in = (1u8 << cfg.players) - 1;
    e.round(0, all_in)
}

/// Total footprint in bytes for a given enumeration, after applying the
/// reachability / imperfect-recall collapse factor.
fn footprint(cfg: &Config, tally: &[StreetTally; STREETS]) -> (f64, f64) {
    let mut info_sets = 0.0;
    let mut bytes = 0.0;
    for s in 0..STREETS {
        info_sets += tally[s].decision_nodes * cfg.buckets[s];
        bytes += cfg.buckets[s] * tally[s].action_slots * BYTES_PER_ENTRY;
    }
    (info_sets * cfg.reachable_fraction, bytes * cfg.reachable_fraction)
}

fn report(cfg: &Config) {
    println!("\n══════════════════════════════════════════════════════════════");
    println!(" {}  ({} players, all-in {})", cfg.label, cfg.players, if cfg.include_allin { "on" } else { "off" });
    println!("══════════════════════════════════════════════════════════════");

    // Per-street detail at the configured per-street caps.
    let tally = enumerate(cfg, cfg.max_raises);
    println!(
        "Per-street raise caps {:?}, bet sizes {:?}, reraises {:?}, store fraction {}:",
        cfg.max_raises, cfg.bet_sizes, cfg.raise_sizes, cfg.reachable_fraction
    );
    println!("{:<8}  {:>14}  {:>8}  {:>16}", "street", "betting seqs", "buckets", "info sets");
    for s in 0..STREETS {
        println!(
            "{:<8}  {:>14}  {:>8}  {:>16}",
            STREET_NAMES[s],
            fmt_count(tally[s].decision_nodes),
            cfg.buckets[s] as u64,
            fmt_count(tally[s].decision_nodes * cfg.buckets[s] * cfg.reachable_fraction),
        );
    }
    let (info_sets, bytes) = footprint(cfg, &tally);
    println!("total info sets : {}", fmt_count(info_sets));
    println!(
        "total memory    : {}   8GB:{:<4} 64GB:{:<4} 128GB:{:<4} 512GB:{}",
        fmt_bytes(bytes),
        fits(bytes, 8.0 * 0.5),
        fits(bytes, 64.0 * 0.8),
        fits(bytes, 128.0 * 0.8),
        fits(bytes, 512.0 * 0.8),
    );

    // Sensitivity over a uniform raise cap — the dominant feasibility lever.
    println!("\nSensitivity to a uniform raise cap (total memory):");
    for cap in 1..=3u8 {
        let t = enumerate(cfg, [cap; STREETS]);
        let (is, b) = footprint(cfg, &t);
        println!(
            "  cap {cap}: {:>12}   info sets {:>10}   8GB:{:<4} 64GB:{:<4} 128GB:{:<4} 512GB:{}",
            fmt_bytes(b),
            fmt_count(is),
            fits(b, 8.0 * 0.5),
            fits(b, 64.0 * 0.8),
            fits(b, 128.0 * 0.8),
            fits(b, 512.0 * 0.8),
        );
    }
}

fn fits(bytes: f64, usable_gb: f64) -> &'static str {
    if bytes <= usable_gb * 1e9 { "yes" } else { "NO" }
}

fn fmt_count(x: f64) -> String {
    if x < 1e3 {
        format!("{:.0}", x)
    } else if x < 1e6 {
        format!("{:.1}K", x / 1e3)
    } else if x < 1e9 {
        format!("{:.1}M", x / 1e6)
    } else if x < 1e12 {
        format!("{:.1}B", x / 1e9)
    } else {
        format!("{:.1}T", x / 1e12)
    }
}

fn fmt_bytes(b: f64) -> String {
    if b < 1e9 {
        format!("{:.1} MB", b / 1e6)
    } else if b < 1e12 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.1} TB", b / 1e12)
    }
}

fn main() {
    println!("Memory estimate — betting-tree enumeration over the action abstraction.");
    println!(
        "Opening-bet sizes from poker_core::betting: preflop={}, flop={}, turn={}, river={}.",
        PREFLOP_BET_FRACS.len(),
        FLOP_BET_FRACS.len(),
        TURN_BET_FRACS.len(),
        RIVER_BET_FRACS.len()
    );
    if TURN_BET_FRACS.len() != 2 {
        println!(
            "NOTE: v3 calls for 2 turn bet sizes (0.5, 1.0); betting.rs defines {}. The turn \
             multiplies through turn+river, so trimming it is the cheapest memory win.",
            TURN_BET_FRACS.len()
        );
    }
    println!(
        "Re-raise size counts are modeled separately (betting.rs only defines opening sizes); \
         the plan lists raises as ~2 sizes per street."
    );

    let bet_sizes = [
        PREFLOP_BET_FRACS.len(),
        FLOP_BET_FRACS.len(),
        TURN_BET_FRACS.len(),
        RIVER_BET_FRACS.len(),
    ];
    // Per the plan's table, re-raises carry ~2 sizes on every street.
    let raise_sizes = [2usize; STREETS];

    report(&Config {
        label: "Heads-up NLHE blueprint",
        players: 2,
        buckets: [169.0, 600.0, 600.0, 1000.0],
        bet_sizes,
        raise_sizes,
        include_allin: true,
        max_raises: [3; STREETS],
        reachable_fraction: 1.0,
    });

    report(&Config {
        label: "6-max NLHE blueprint (naive full-granularity upper bound)",
        players: 6,
        buckets: [169.0 * 6.0, 600.0, 600.0, 1000.0],
        bet_sizes,
        raise_sizes,
        include_allin: true,
        max_raises: [3; STREETS],
        reachable_fraction: 1.0,
    });

    // The same 6-max game, modeled the way a search-based system like Pluribus
    // actually shrinks it: the blueprint stays *coarse on deep streets* (fewer
    // bet sizes and fewer raises on turn/river, because real-time search refills
    // the detail at play time), and only the reachable, imperfect-recall-
    // collapsed fraction of the cross product is stored.  This is what closes the
    // gap between the textbook enumeration and the ~512 GB Pluribus held in RAM.
    report(&Config {
        label: "6-max NLHE (Pluribus-style: coarse deep streets + search + IR collapse)",
        players: 6,
        buckets: [169.0 * 6.0, 600.0, 600.0, 1000.0],
        // Rich early where decisions matter; thin deep where search takes over.
        bet_sizes: [3, 2, 1, 1],
        raise_sizes: [2, 1, 1, 1],
        include_allin: true,
        // Raises capped low everywhere (Pluribus kept the blueprint shallow and
        // let search add depth); just one raise on the deep streets.
        max_raises: [2, 1, 1, 1],
        // Imperfect recall + unreachable-combo pruning collapses ~10× of the
        // naive cross product (illustrative, defensible order of magnitude).
        reachable_fraction: 0.1,
    });

    // The "can I actually do this on one machine" 6-max config: same coarsening,
    // but with *very* coarse postflop buckets (the river dominates the footprint,
    // so shrinking it is the biggest lever).  This is a weak-but-real 6-max
    // blueprint — not Pluribus-grade, but it FITS, which is the question.
    report(&Config {
        label: "6-max NLHE (shoestring: very coarse buckets — fits one box)",
        players: 6,
        buckets: [169.0 * 6.0, 100.0, 100.0, 100.0],
        bet_sizes: [3, 2, 1, 1],
        raise_sizes: [2, 1, 1, 1],
        include_allin: true,
        max_raises: [2, 1, 1, 1],
        reachable_fraction: 0.1,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(players: usize, bet_sizes: [usize; STREETS], raise_sizes: [usize; STREETS]) -> Config {
        Config {
            label: "t",
            players,
            buckets: [1.0; STREETS],
            bet_sizes,
            raise_sizes,
            include_allin: false,
            max_raises: [2; STREETS],
            reachable_fraction: 1.0,
        }
    }

    #[test]
    fn check_only_tree_has_two_checks_per_street() {
        let c = cfg(2, [0; STREETS], [0; STREETS]);
        let tally = enumerate(&c, [0; STREETS]);
        for s in 0..STREETS {
            assert_eq!(tally[s].decision_nodes, 2.0, "two checks on {}", STREET_NAMES[s]);
        }
    }

    #[test]
    fn more_bet_sizes_means_more_sequences() {
        let base: f64 = enumerate(&cfg(2, [1; STREETS], [1; STREETS]), [2; STREETS])
            .iter()
            .map(|t| t.decision_nodes)
            .sum();
        let wide: f64 = enumerate(&cfg(2, [3; STREETS], [2; STREETS]), [2; STREETS])
            .iter()
            .map(|t| t.decision_nodes)
            .sum();
        assert!(wide > base);
    }

    #[test]
    fn raise_cap_drives_size() {
        let c = cfg(2, [3; STREETS], [2; STREETS]);
        let cap1: f64 = enumerate(&c, [1; STREETS]).iter().map(|t| t.decision_nodes).sum();
        let cap3: f64 = enumerate(&c, [3; STREETS]).iter().map(|t| t.decision_nodes).sum();
        assert!(cap3 > cap1 * 2.0, "raise cap should sharply grow the tree");
    }

    #[test]
    fn deeper_street_cap_shrinks_the_tree() {
        // Capping raises lower on deep streets must reduce the count — the lever
        // the Pluribus-style config uses.
        let c = cfg(2, [3; STREETS], [2; STREETS]);
        let uniform: f64 = enumerate(&c, [3; STREETS]).iter().map(|t| t.decision_nodes).sum();
        let coarse_deep: f64 = enumerate(&c, [3, 2, 1, 1]).iter().map(|t| t.decision_nodes).sum();
        assert!(coarse_deep < uniform, "lower deep-street caps shrink the tree");
    }

    #[test]
    fn reachable_fraction_scales_footprint_linearly() {
        let mut c = cfg(2, [3; STREETS], [2; STREETS]);
        c.buckets = [10.0; STREETS];
        let full = footprint(&c, &enumerate(&c, [2; STREETS]));
        c.reachable_fraction = 0.1;
        let collapsed = footprint(&c, &enumerate(&c, [2; STREETS]));
        assert!((collapsed.1 - full.1 * 0.1).abs() < 1.0, "store fraction scales bytes");
    }

    #[test]
    fn six_max_is_larger_than_heads_up() {
        let hu: f64 = enumerate(&cfg(2, [3, 3, 2, 4], [2; STREETS]), [3; STREETS])
            .iter()
            .map(|t| t.decision_nodes)
            .sum();
        let six: f64 = enumerate(&cfg(6, [3, 3, 2, 4], [2; STREETS]), [3; STREETS])
            .iter()
            .map(|t| t.decision_nodes)
            .sum();
        assert!(six > hu);
    }
}
