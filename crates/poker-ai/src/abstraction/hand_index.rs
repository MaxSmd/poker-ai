//! Perfect dense suit-isomorphic hand index (Waugh's hand-isomorphism).
//!
//! [`super::canonical::canonical_key`] is a *correct* suit-isomorphic key, but it
//! is sparse — a `u64`/`Vec<u8>` scattered across the key space — so the cache
//! and bucket map it feeds must be hash maps.  At the plan's river scale
//! (~10^8 canonical situations) that is tens of GB of keys and pointer chasing.
//!
//! A [`HandIndexer`] instead maps every canonical `(hole, board)` *bijectively*
//! onto a **dense** integer in `0..size()`, so the abstraction becomes flat
//! arrays indexed by that integer: one index computation + one array read, no
//! hashing, and total coverage by construction.  `index` and `unindex` are
//! inverses, which lets the offline build iterate the index range directly
//! (partition `0..size()` across cores) instead of sampling boards and merging
//! per-board maps.
//!
//! # Construction (the standard hand-isomorphism algorithm)
//!
//! Cards are dealt in *rounds* (`cards_per_round`, e.g. `[2, 3]` = hole + flop).
//! Suit isomorphism is the freedom to relabel the four suits.  Two facts make a
//! dense index possible:
//!
//!  * **Suits are rank-independent.**  Each suit's cards occupy ranks `0..13`
//!    with no cross-suit conflict (same rank in two suits = two distinct cards),
//!    so a suit's "rank pattern" — which ranks it holds in each round — is ranked
//!    on its own with the [combinatorial number system](colex), giving a
//!    `suit_index` in `0..suit_size`.
//!  * **Only the multiset of suit patterns matters.**  Relabeling suits permutes
//!    them, so the canonical form sorts the suits.  Suits are grouped by their
//!    per-round *count vector* (a "configuration"); within a group the suits are
//!    interchangeable, so their `suit_index`es are combined as a **multiset**
//!    (sorted, ranked with repetition) rather than an ordered tuple.
//!
//! The index is then `config_offset + Σ_group multiset_rank(group)` with mixed
//! radixes, and `size()` is the sum of all configuration sizes.  Exact published
//! counts (`[2]` → 169, `[2,3]` → 1,286,792) gate correctness.
//!
//! [colex]: https://en.wikipedia.org/wiki/Combinatorial_number_system

use std::collections::HashMap;

use poker_core::{make_card, rank_of, suit_of};

const SUITS: usize = 4;
const RANKS: u8 = 13;

/// `n choose k`, computed incrementally in `u128` (exact, no table).  Our `n`
/// stays well under 2^32 and results fit `u64`.
fn choose(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c: u128 = 1;
    for i in 1..=k {
        c = c * (n - k + i) as u128 / i as u128;
    }
    c as u64
}

/// Ways to choose `g` items from `m` with repetition (multiset coefficient).
fn multiset_choose(m: u64, g: usize) -> u64 {
    if g == 0 {
        return 1;
    }
    choose(m + g as u64 - 1, g as u64)
}

/// Colexicographic rank of a strictly-increasing position set `{p_0<…<p_{k-1}}`.
fn combination_rank(positions: &[u64]) -> u64 {
    positions.iter().enumerate().map(|(i, &p)| choose(p, i as u64 + 1)).sum()
}

/// Inverse of [`combination_rank`]: the `k`-subset of `0..n` (sorted ascending)
/// at colex `rank`.
fn combination_unrank(mut rank: u64, k: usize) -> Vec<u64> {
    let mut ps = vec![0u64; k];
    for i in (0..k).rev() {
        let mut p = i as u64;
        while choose(p + 1, i as u64 + 1) <= rank {
            p += 1;
        }
        rank -= choose(p, i as u64 + 1);
        ps[i] = p;
    }
    ps
}

/// Inverse of the multiset rank: the non-decreasing `g`-tuple over `0..m` at
/// `rank` (via the strictly-increasing bijection `y_i = x_i + i`).
fn multiset_unrank(rank: u64, g: usize) -> Vec<u64> {
    let ys = combination_unrank(rank, g);
    ys.iter().enumerate().map(|(i, &y)| y - i as u64).collect()
}

/// Multiset rank of a non-decreasing `g`-tuple `xs` over `0..m`.
fn multiset_rank(xs: &[u64]) -> u64 {
    let ys: Vec<u64> = xs.iter().enumerate().map(|(i, &x)| x + i as u64).collect();
    combination_rank(&ys)
}

/// One group of interchangeable suits inside a configuration (suits sharing a
/// per-round count vector).
#[derive(Clone, Debug)]
struct Group {
    /// Per-round card counts shared by every suit in the group.
    vector: Vec<u8>,
    /// Number of suits with this vector.
    mult: usize,
    /// Number of distinct rank patterns one suit with this vector can take.
    suit_size: u64,
    /// `multiset_choose(suit_size, mult)` — the group's index span.
    group_size: u64,
}

/// A canonical suit configuration: the multiset of the four suits' count vectors,
/// stored as groups in a fixed (descending-vector) order.
#[derive(Clone, Debug)]
struct Config {
    groups: Vec<Group>,
    offset: u64,
    size: u64,
}

/// A perfect dense index for `(hole, board)` situations under suit isomorphism.
pub struct HandIndexer {
    cards_per_round: Vec<u8>,
    rounds: usize,
    configs: Vec<Config>,
    /// Group signature `[(vector, mult)…]` → configuration position.
    lookup: HashMap<Vec<(Vec<u8>, usize)>, usize>,
    size: u64,
}

impl HandIndexer {
    /// Build an indexer for the given per-round card counts (e.g. `[2, 3]` for a
    /// flop situation: 2 hole + 3 board).
    pub fn new(cards_per_round: &[u8]) -> Self {
        let rounds = cards_per_round.len();
        let mut indexer = Self {
            cards_per_round: cards_per_round.to_vec(),
            rounds,
            configs: Vec::new(),
            lookup: HashMap::new(),
            size: 0,
        };
        indexer.build_configs();
        indexer
    }

    /// Number of distinct canonical situations — the dense index range.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Total cards consumed across all rounds.
    pub fn total_cards(&self) -> usize {
        self.cards_per_round.iter().map(|&c| c as usize).sum()
    }

    /// The per-round card counts this indexer was built for.
    pub fn cards_per_round(&self) -> &[u8] {
        &self.cards_per_round
    }

    /// Dense canonical index for `cards`, laid out in round order
    /// (`cards_per_round[0]` cards, then `[1]`, …).  Invariant under suit
    /// relabeling; bijective onto `0..size()`.
    pub fn index(&self, cards: &[u8]) -> u64 {
        debug_assert_eq!(cards.len(), self.total_cards(), "card count must match the round structure");

        // Bucket card ranks by (suit, round).
        let mut per_suit: [Vec<Vec<u8>>; SUITS] =
            std::array::from_fn(|_| vec![Vec::new(); self.rounds]);
        let mut pos = 0;
        for (r, &n) in self.cards_per_round.iter().enumerate() {
            for _ in 0..n {
                let c = cards[pos];
                pos += 1;
                per_suit[suit_of(c) as usize][r].push(rank_of(c));
            }
        }

        // Per-suit (count vector, rank-pattern index).
        let mut suits: Vec<(Vec<u8>, u64)> = (0..SUITS)
            .map(|s| {
                let vector: Vec<u8> = per_suit[s].iter().map(|rs| rs.len() as u8).collect();
                (vector, suit_rank_index(&per_suit[s]))
            })
            .collect();

        // Canonicalize: suits sorted by count vector (desc), then by pattern
        // index (asc) so each group's indices come out already sorted.
        suits.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        // Group by vector and form the configuration signature.
        let mut signature: Vec<(Vec<u8>, usize)> = Vec::new();
        let mut group_indices: Vec<Vec<u64>> = Vec::new();
        for (vector, idx) in &suits {
            if signature.last().map(|(v, _)| v) == Some(vector) {
                let last = signature.last_mut().unwrap();
                last.1 += 1;
                group_indices.last_mut().unwrap().push(*idx);
            } else {
                signature.push((vector.clone(), 1));
                group_indices.push(vec![*idx]);
            }
        }

        let config = &self.configs[self.lookup[&signature]];
        let mut within = 0u64;
        for (group, idxs) in config.groups.iter().zip(&group_indices) {
            within = within * group.group_size + multiset_rank(idxs);
        }
        config.offset + within
    }

    /// A representative card layout (round order) for dense index `index` — the
    /// inverse of [`index`](Self::index) up to suit relabeling
    /// (`index(unindex(i)) == i`).  Drives the offline index-range build.
    pub fn unindex(&self, index: u64) -> Vec<u8> {
        debug_assert!(index < self.size, "index out of range");

        // Locate the configuration owning this index.
        let config = self
            .configs
            .iter()
            .find(|c| index >= c.offset && index < c.offset + c.size)
            .expect("index lies in some configuration");
        let mut within = index - config.offset;

        // Decode each group's multiset rank (least-significant group last).
        let mut ranks = vec![0u64; config.groups.len()];
        for (g, group) in config.groups.iter().enumerate().rev() {
            ranks[g] = within % group.group_size;
            within /= group.group_size;
        }

        // Assign physical suits 0,1,2,… in canonical group order, reconstruct
        // each suit's per-round ranks, and lay the cards out by round.
        let mut suit_round_ranks: [Vec<Vec<u8>>; SUITS] =
            std::array::from_fn(|_| vec![Vec::new(); self.rounds]);
        let mut next_suit = 0usize;
        for (group, &rank) in config.groups.iter().zip(&ranks) {
            let pattern_indices = multiset_unrank(rank, group.mult);
            for &pi in &pattern_indices {
                let per_round = suit_unrank(&group.vector, pi);
                suit_round_ranks[next_suit] = per_round;
                next_suit += 1;
            }
        }

        let mut cards = Vec::with_capacity(self.total_cards());
        for r in 0..self.rounds {
            for (s, rounds) in suit_round_ranks.iter().enumerate() {
                for &rank in &rounds[r] {
                    cards.push(make_card(rank, s as u8));
                }
            }
        }
        cards
    }

    /// Enumerate every canonical suit configuration with its offset and size.
    fn build_configs(&mut self) {
        let mut configs = Vec::new();
        let remaining = self.cards_per_round.clone();
        self.enumerate(0, &remaining, &mut Vec::new(), &mut configs);

        let mut offset = 0u64;
        for cfg in &mut configs {
            cfg.offset = offset;
            offset += cfg.size;
        }
        self.size = offset;
        for (i, cfg) in configs.iter().enumerate() {
            let sig: Vec<(Vec<u8>, usize)> =
                cfg.groups.iter().map(|g| (g.vector.clone(), g.mult)).collect();
            self.lookup.insert(sig, i);
        }
        self.configs = configs;
    }

    /// Recursively assign a count vector to each suit, keeping the four vectors
    /// in non-increasing order (canonical) and the per-round column sums exact.
    fn enumerate(&self, suit: usize, remaining: &[u8], chosen: &mut Vec<Vec<u8>>, out: &mut Vec<Config>) {
        if suit == SUITS {
            if remaining.iter().all(|&x| x == 0) {
                out.push(build_config(chosen));
            }
            return;
        }
        let prev = chosen.last().cloned();
        for v in candidate_vectors(remaining) {
            if v.iter().sum::<u8>() > RANKS {
                continue; // a suit cannot hold more than 13 distinct ranks
            }
            if let Some(p) = &prev {
                if v > *p {
                    continue; // enforce descending order ⇒ one canonical form
                }
            }
            let next: Vec<u8> = remaining.iter().zip(&v).map(|(&rem, &x)| rem - x).collect();
            chosen.push(v);
            self.enumerate(suit + 1, &next, chosen, out);
            chosen.pop();
        }
    }
}

/// Number of distinct rank patterns a single suit with per-round counts
/// `vector` can take (`∏_r C(13 − used_before_r, vector[r])`).
fn suit_size(vector: &[u8]) -> u64 {
    let mut used = 0u64;
    let mut size = 1u64;
    for &c in vector {
        size *= choose(RANKS as u64 - used, c as u64);
        used += c as u64;
    }
    size
}

/// All count vectors `v` with `0 <= v[r] <= remaining[r]` (the Cartesian
/// product over rounds).
fn candidate_vectors(remaining: &[u8]) -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new()];
    for &cap in remaining {
        let mut next = Vec::new();
        for prefix in &out {
            for x in 0..=cap {
                let mut v = prefix.clone();
                v.push(x);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// Build a [`Config`] from four count vectors already in descending order.
fn build_config(vectors: &[Vec<u8>]) -> Config {
    let mut groups: Vec<Group> = Vec::new();
    for v in vectors {
        if groups.last().map(|g| &g.vector) == Some(v) {
            groups.last_mut().unwrap().mult += 1;
        } else {
            groups.push(Group { vector: v.clone(), mult: 1, suit_size: 0, group_size: 0 });
        }
    }
    let mut size = 1u64;
    for g in &mut groups {
        g.suit_size = suit_size(&g.vector);
        g.group_size = multiset_choose(g.suit_size, g.mult);
        size *= g.group_size;
    }
    Config { groups, offset: 0, size }
}

/// Colex index of a suit's rank pattern: combine each round's chosen ranks
/// (ranked among the ranks still unused by this suit) with mixed radixes.
fn suit_rank_index(per_round_ranks: &[Vec<u8>]) -> u64 {
    let mut used = 0u16;
    let mut idx = 0u64;
    for ranks in per_round_ranks {
        let avail = RANKS as u64 - used.count_ones() as u64;
        let mut positions: Vec<u64> = ranks
            .iter()
            .map(|&rk| {
                // Position = number of still-unused ranks strictly below `rk`.
                let lower = (1u16 << rk) - 1;
                (!used & lower).count_ones() as u64
            })
            .collect();
        positions.sort_unstable();
        idx = idx * choose(avail, ranks.len() as u64) + combination_rank(&positions);
        for &rk in ranks {
            used |= 1 << rk;
        }
    }
    idx
}

/// Inverse of [`suit_rank_index`]: the per-round rank lists for pattern `idx` of
/// a suit with counts `vector`.
fn suit_unrank(vector: &[u8], mut idx: u64) -> Vec<Vec<u8>> {
    let rounds = vector.len();

    // Per-round radixes depend only on the prefix card counts, so they are known
    // up front; extract the round digits least-significant first.
    let mut used_before = 0u64;
    let mut avail = vec![0u64; rounds];
    for r in 0..rounds {
        avail[r] = RANKS as u64 - used_before;
        used_before += vector[r] as u64;
    }
    let mut subs = vec![0u64; rounds];
    for r in (0..rounds).rev() {
        let radix = choose(avail[r], vector[r] as u64);
        subs[r] = idx % radix;
        idx /= radix;
    }

    // Forward pass: map colex positions back to concrete ranks, consuming the
    // shared rank pool round by round.
    let mut used = 0u16;
    let mut out = vec![Vec::new(); rounds];
    for r in 0..rounds {
        // All positions are colex ranks over the ranks unused *before this round*
        // (the forward pass marks `used` only after the whole round), so resolve
        // them all against the round-start mask, then consume them together.
        let positions = combination_unrank(subs[r], vector[r] as usize);
        for &p in &positions {
            out[r].push(nth_unused(used, p));
        }
        for &rank in &out[r] {
            used |= 1 << rank;
        }
    }
    out
}

/// The `p`-th (0-indexed) rank not present in the `used` bitmask.
fn nth_unused(used: u16, p: u64) -> u8 {
    let mut count = 0u64;
    for rank in 0..RANKS {
        if used & (1 << rank) == 0 {
            if count == p {
                return rank;
            }
            count += 1;
        }
    }
    unreachable!("fewer than {p} unused ranks")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::canonical::canonical_key;

    /// xorshift64* — a tiny deterministic RNG for sampling cards in tests.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || crate::util::rng::xorshift_next_u64(&mut s)
    }

    /// Deal `n` distinct cards via partial Fisher–Yates.
    fn deal(n: usize, next: &mut impl FnMut() -> u64) -> Vec<u8> {
        let mut deck: [u8; 52] = std::array::from_fn(|i| i as u8);
        for i in 0..n {
            let span = 52 - i;
            let j = i + (next() as usize) % span;
            deck.swap(i, j);
        }
        deck[..n].to_vec()
    }

    #[test]
    fn preflop_has_169_classes() {
        let ix = HandIndexer::new(&[2]);
        assert_eq!(ix.size(), 169);
        // Every 2-card hand maps into 0..169 and the range is dense.
        let mut seen = vec![false; 169];
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let i = ix.index(&[a, b]);
                assert!(i < 169);
                seen[i as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "all 169 preflop classes are reachable");
    }

    #[test]
    fn index_is_suit_isomorphic() {
        // For each street, a random situation and all 24 suit relabelings must
        // share one index.
        let mut next = rng(7);
        for rounds in [vec![2u8], vec![2, 3], vec![2, 4], vec![2, 5]] {
            let ix = HandIndexer::new(&rounds);
            let total: usize = rounds.iter().map(|&c| c as usize).sum();
            for _ in 0..50 {
                let cards = deal(total, &mut next);
                let base = ix.index(&cards);
                for perm in suit_perms() {
                    let relabeled: Vec<u8> =
                        cards.iter().map(|&c| make_card(rank_of(c), perm[suit_of(c) as usize])).collect();
                    assert_eq!(ix.index(&relabeled), base, "perm {perm:?} changed the index");
                }
            }
        }
    }

    #[test]
    fn index_agrees_with_canonical_key_oracle() {
        // The dense index must induce exactly the same equivalence classes as the
        // already-validated lex-min canonical key: same key ⟺ same index.
        let mut next = rng(99);
        let ix = HandIndexer::new(&[2, 3]);
        let mut samples: Vec<(u64, u64)> = Vec::new(); // (canonical key, dense index)
        for _ in 0..3000 {
            let cards = deal(5, &mut next);
            let hole = [cards[0], cards[1]];
            let board = [cards[2], cards[3], cards[4]];
            samples.push((canonical_key(&hole, &board), ix.index(&cards)));
        }
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                assert_eq!(
                    samples[i].0 == samples[j].0,
                    samples[i].1 == samples[j].1,
                    "key/index equivalence disagree"
                );
            }
        }
    }

    #[test]
    fn unindex_round_trips() {
        // Covers the joint indexers (used as info keys) and the single-round
        // board indexers (used by the offline build to enumerate boards).
        for rounds in [vec![2u8], vec![2, 3], vec![2, 4], vec![2, 5], vec![3], vec![4], vec![5]] {
            let ix = HandIndexer::new(&rounds);
            let n = ix.size();
            // Sample indices across the whole range (and the endpoints).
            let probes: Vec<u64> = (0..200).map(|k| (k * n / 200).min(n - 1)).chain([n - 1]).collect();
            for i in probes {
                let cards = ix.unindex(i);
                assert_eq!(cards.len(), ix.total_cards());
                // Cards are distinct and real.
                let mut seen = 0u64;
                for &c in &cards {
                    assert!(c < 52);
                    assert_eq!(seen & (1 << c), 0, "duplicate card from unindex");
                    seen |= 1 << c;
                }
                assert_eq!(ix.index(&cards), i, "round-trip mismatch at {i} (rounds {rounds:?}, cards {cards:?})");
            }
        }
    }

    /// Exact count gate — enumerates all C(52,2)·C(50,3) = 25,989,600 flop deals
    /// and asserts the canonical index hits exactly 1,286,792 dense values.  Slow;
    /// run with:  cargo test -p poker-ai --release -- --ignored flop_count
    #[test]
    #[ignore]
    fn flop_count_is_exactly_1_286_792() {
        let ix = HandIndexer::new(&[2, 3]);
        assert_eq!(ix.size(), 1_286_792, "flop canonical count");

        let mut seen = vec![false; ix.size() as usize];
        for h0 in 0..52u8 {
            for h1 in (h0 + 1)..52u8 {
                for b0 in 0..52u8 {
                    if b0 == h0 || b0 == h1 {
                        continue;
                    }
                    for b1 in (b0 + 1)..52u8 {
                        if b1 == h0 || b1 == h1 {
                            continue;
                        }
                        for b2 in (b1 + 1)..52u8 {
                            if b2 == h0 || b2 == h1 {
                                continue;
                            }
                            let i = ix.index(&[h0, h1, b0, b1, b2]);
                            assert!(i < ix.size());
                            seen[i as usize] = true;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "index range is dense (every slot reached)");
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn suit_perms() -> Vec<[u8; 4]> {
        let mut out = Vec::new();
        for a in 0..4u8 {
            for b in 0..4u8 {
                if b == a {
                    continue;
                }
                for c in 0..4u8 {
                    if c == a || c == b {
                        continue;
                    }
                    out.push([a, b, c, 6 - a - b - c]);
                }
            }
        }
        out
    }
}
