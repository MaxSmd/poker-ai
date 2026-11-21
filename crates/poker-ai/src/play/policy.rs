//! Compact, read-only view of a trained blueprint strategy.
//!
//! `data/blueprint_holdem.bin` comes in two formats, told apart by the file's
//! first eight bytes; [`CompactPolicy::load`] reads either.
//!
//! **Dense** — what the SoA trainer writes.  One `u16` probability per
//! (info set, action), in the order of the game's dense info-set index, with
//! the table's run-length [`Layout`] in the header.  The key of an info set
//! is its index, so nothing is stored per info set and a lookup is one array
//! read: 2 bytes per slot, 3.2 GB for the 595 M info sets of the 200 bb
//! blueprint.  The game has to hand out those indices as its keys —
//! [`CompactPolicy::bind`] arranges that, and refuses a game whose table is
//! laid out differently from the file's.
//!
//! **Hashed** — a bincode `HashMap<u64, Vec<f32>>` (hashed info key → action
//! distribution); what the `HashMap` trainers write, and what every blueprint
//! before the dense format is in.  At Slumbot depth that is ~600 M entries:
//! deserializing it back into a `HashMap` would cost ~3× the file size in
//! allocator and table overhead, so the loader streams it into three flat
//! arrays — sorted keys, packed offset+length, and one probability pool — and
//! looks keys up by binary search.  Still 16 bytes of key and offset per info
//! set on top of 4 per probability: 16 GB for the same blueprint, five times
//! the dense form.  `play compact` converts one into the other.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::games::blueprint::BlueprintHoldem;
use crate::games::{IndexedGame, Layout, Span};

/// A key's probabilities live at `offset = packed >> 4`, `len = packed & 0xF`.
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    packed: u64,
}

/// How [`CompactPolicy::probs_or_uniform`] resolved its lookups.
///
/// **Why this is instrumented at all.** `probs_or_uniform` falls back to a
/// uniform distribution on a miss, silently — no error, no log line. A uniform
/// fallback is a *reasonable* default and a *catastrophic* steady state: the
/// bot plays every action equally often and, worse, the Bayes range tracker
/// updates both players' ranges with those fake likelihoods, so every
/// downstream re-solve starts from beliefs that were never informed by the
/// blueprint. A bot in that state looks like it is working. It plays legal
/// poker, logs no errors, and loses.
///
/// That failure has already happened once here — a `--cap` mismatch between the
/// bot and the trained blueprint changed the abstract betting tree, so every
/// info key missed. It cost roughly two thousand hands to notice. This counter
/// makes it visible in ten.
///
/// Counts are `Relaxed` atomics: they are diagnostics, never control flow, and
/// exact ordering between them is worth nothing next to keeping the lookup
/// free of synchronization.
#[derive(Default)]
pub struct LookupStats {
    hit: AtomicU64,
    /// The blueprint has no entry for this info key at all.
    miss: AtomicU64,
    /// The key resolved, but to a different number of actions than the caller
    /// offered — a key collision, or an abstraction that no longer matches the
    /// one that was trained.
    width_mismatch: AtomicU64,
}

/// A snapshot of [`LookupStats`], taken together so the ratios are consistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupCounts {
    pub hit: u64,
    pub miss: u64,
    pub width_mismatch: u64,
}

impl LookupCounts {
    pub fn total(&self) -> u64 {
        self.hit + self.miss + self.width_mismatch
    }

    /// Fraction of lookups the blueprint actually answered, `1.0` if none were
    /// made (nothing has gone wrong yet).
    pub fn hit_rate(&self) -> f64 {
        match self.total() {
            0 => 1.0,
            n => self.hit as f64 / n as f64,
        }
    }
}

impl std::fmt::Display for LookupCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} blueprint lookups hit ({:.1}%), {} missed, {} width-mismatched",
            self.hit,
            self.total(),
            100.0 * self.hit_rate(),
            self.miss,
            self.width_mismatch
        )
    }
}

/// A trained strategy: info key → action distribution.
pub struct CompactPolicy {
    store: Store,
    stats: LookupStats,
}

enum Store {
    /// Keys are hashed info keys, found by binary search.
    Hashed { entries: Vec<Entry>, probs: Vec<f32> },
    /// Keys are dense info-set indices; `probs` holds one quantized
    /// probability per slot of `layout`, an all-zero row meaning "never
    /// trained".  `visited` counts the rows that are not.
    Dense { layout: Layout, probs: Vec<u16>, visited: u64 },
}

/// First eight bytes of a dense policy file.  A hashed file starts with its
/// entry count, which no real blueprint brings anywhere near this.
const DENSE_MAGIC: u64 = u64::from_le_bytes(*b"POKRDNS1");

/// A stored probability of 1.  Every trained row's entries sum to exactly
/// this (see [`quantize`]), so dividing by it gives a distribution that sums
/// to one, each entry within `1/Q_ONE` of the value it was built from.
const Q_ONE: u16 = u16::MAX;

/// Quantize a distribution to `u16`s that sum to exactly [`Q_ONE`], by the
/// largest-remainder method: every entry is rounded down, then the units that
/// rounding lost go to the entries that lost most.  Each ends up the floor or
/// the ceiling of its exact value, so it is off by less than one unit, and an
/// exact zero stays zero.
///
/// A row without positive mass comes out all zero, which is how the file says
/// "never trained".
fn quantize(probs: &[f64], out: &mut [u16]) {
    debug_assert_eq!(probs.len(), out.len());
    let total: f64 = probs.iter().map(|&p| p.max(0.0)).sum();
    if !(total > 0.0 && total.is_finite()) {
        out.fill(0);
        return;
    }
    let exact = |p: f64| p.max(0.0) / total * Q_ONE as f64;
    let mut given = 0u32;
    for (o, &p) in out.iter_mut().zip(probs) {
        *o = exact(p).floor().min(Q_ONE as f64) as u16;
        given += *o as u32;
    }
    // At most one unit per action is left over; hand them out by remainder.
    for _ in given..Q_ONE as u32 {
        let most = (0..out.len())
            .filter(|&i| out[i] < Q_ONE)
            .max_by(|&i, &j| {
                let lost = |k: usize| exact(probs[k]) - out[k] as f64;
                // Ties go to the earlier action, so the result is deterministic.
                lost(i).total_cmp(&lost(j)).then(j.cmp(&i))
            })
            .expect("a row with mass has an entry below one");
        out[most] += 1;
    }
}

impl CompactPolicy {
    /// Load a policy file in either format (see the module header), without
    /// materializing a `HashMap` for the hashed one.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 20, file);
        let head = read_u64(&mut r)?;
        let store = if head == DENSE_MAGIC {
            Self::load_dense(&mut r, file_len)?
        } else {
            Self::load_hashed(&mut r, head as usize)?
        };
        Ok(Self { store, stats: LookupStats::default() })
    }

    /// The body of a hashed file: `n` entries of key, length, `f32`s.
    fn load_hashed(r: &mut impl Read, n: usize) -> io::Result<Store> {
        let mut entries = Vec::with_capacity(n);
        let mut probs = Vec::new();
        for _ in 0..n {
            let key = read_u64(r)?;
            let len = read_u64(r)? as usize;
            if len > 0xF {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("action count {len} exceeds the 15-action packing limit"),
                ));
            }
            let offset = probs.len() as u64;
            let mut buf = [0u8; 4];
            for _ in 0..len {
                r.read_exact(&mut buf)?;
                probs.push(f32::from_le_bytes(buf));
            }
            entries.push(Entry { key, packed: (offset << 4) | len as u64 });
        }
        entries.sort_unstable_by_key(|e| e.key);
        Ok(Store::Hashed { entries, probs })
    }

    /// The body of a dense file, after its marker: the trained-row count, the
    /// layout, then one little-endian `u16` per slot.
    fn load_dense(r: &mut BufReader<File>, file_len: u64) -> io::Result<Store> {
        let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
        let visited = read_u64(r)?;
        let layout: Layout = bincode::deserialize_from(&mut *r).map_err(io::Error::other)?;
        layout.validate().map_err(bad)?;
        // Checked against the file's length before allocating: a damaged
        // header must not ask for an arbitrary amount of memory, and a
        // truncated file must not load as a policy with a silent tail.
        let body = file_len - r.stream_position()?;
        if body != layout.total_slots() as u64 * 2 {
            return Err(bad(format!(
                "dense policy holds {body} bytes of probabilities but its layout has {} slots",
                layout.total_slots()
            )));
        }
        let mut probs = vec![0u16; layout.total_slots()];
        let mut bytes = vec![0u8; 1 << 20];
        for chunk in probs.chunks_mut(bytes.len() / 2) {
            let raw = &mut bytes[..chunk.len() * 2];
            r.read_exact(raw)?;
            for (q, pair) in chunk.iter_mut().zip(raw.chunks_exact(2)) {
                *q = u16::from_le_bytes([pair[0], pair[1]]);
            }
        }
        Ok(Store::Dense { layout, probs, visited })
    }

    /// Build a hashed policy from explicit entries (tests / small games).
    pub fn from_entries(mut list: Vec<(u64, Vec<f32>)>) -> Self {
        list.sort_unstable_by_key(|(k, _)| *k);
        let mut entries = Vec::with_capacity(list.len());
        let mut probs = Vec::new();
        for (key, p) in list {
            assert!(p.len() <= 0xF, "action count exceeds the packing limit");
            entries.push(Entry { key, packed: ((probs.len() as u64) << 4) | p.len() as u64 });
            probs.extend_from_slice(&p);
        }
        Self { store: Store::Hashed { entries, probs }, stats: LookupStats::default() }
    }

    /// Whether keys are dense info-set indices (a dense file) rather than
    /// hashed info keys.
    pub fn is_dense(&self) -> bool {
        matches!(self.store, Store::Dense { .. })
    }

    /// Ready `game` to be played from this policy, or say why it cannot be.
    ///
    /// A hashed policy takes the game as it is.  A dense one needs the game's
    /// keys to be its dense indices, and those indices to mean what they meant
    /// in training: the game gets its index built and its keys switched
    /// ([`BlueprintHoldem::with_dense_keys`]), and its table layout must equal
    /// the file's.  A different raise cap, stack depth or bucket count changes
    /// the layout — with hashed keys that only showed as a collapsed hit rate
    /// a few thousand hands in; here it does not load.
    pub fn bind(&self, game: BlueprintHoldem) -> Result<BlueprintHoldem, String> {
        let Store::Dense { layout, .. } = &self.store else {
            return Ok(game);
        };
        let game = game.try_with_indexing().map_err(|e| format!("a dense policy needs the dense index: {e}"))?;
        let game = game.with_dense_keys();
        let ours = game.layout();
        if ours != *layout {
            return Err(format!(
                "the policy was trained on a different game: it holds {} info sets / {} action slots, \
                 this game has {} / {} — check --cap, --stack-bb and that the bucket maps are the training run's",
                layout.capacity(),
                layout.total_slots(),
                ours.capacity(),
                ours.total_slots()
            ));
        }
        Ok(game)
    }

    /// Panic unless `game`'s keys are the kind this policy is stored under —
    /// for the code that takes a game and a policy separately.  Mixing the two
    /// kinds does not fail on its own: every lookup just answers for some
    /// other info set, or none.
    pub fn assert_keys_match(&self, game: &BlueprintHoldem) {
        assert_eq!(
            self.is_dense(),
            game.uses_dense_keys(),
            "policy and game disagree on what an info key is — pair them with CompactPolicy::bind"
        );
    }

    /// The stored distribution for `key`, if the blueprint visited it.
    pub fn get(&self, key: u64) -> Option<Vec<f64>> {
        match &self.store {
            Store::Hashed { entries, probs } => {
                let i = entries.binary_search_by_key(&key, |e| e.key).ok()?;
                let e = entries[i];
                let off = (e.packed >> 4) as usize;
                let len = (e.packed & 0xF) as usize;
                Some(probs[off..off + len].iter().map(|&x| x as f64).collect())
            }
            Store::Dense { layout, probs, .. } => {
                let index = usize::try_from(key).ok().filter(|&i| i < layout.capacity())?;
                let row = &probs[layout.span(index).range()];
                if row.iter().all(|&q| q == 0) {
                    return None;
                }
                Some(row.iter().map(|&q| q as f64 / Q_ONE as f64).collect())
            }
        }
    }

    /// Whether the blueprint has a strategy for `key`.
    pub fn contains(&self, key: u64) -> bool {
        match &self.store {
            Store::Hashed { entries, .. } => entries.binary_search_by_key(&key, |e| e.key).is_ok(),
            Store::Dense { .. } => self.get(key).is_some(),
        }
    }

    /// The distribution for `key` as `f64`, falling back to uniform over
    /// `num_actions` when the blueprint never visited the info set.
    ///
    /// Every call is counted — see [`LookupStats`] for why a silent uniform
    /// fallback is the most dangerous failure this bot has.  Read the tally with
    /// [`lookup_counts`](Self::lookup_counts).
    pub fn probs_or_uniform(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.get(key) {
            Some(p) if p.len() == num_actions => {
                self.stats.hit.fetch_add(1, Relaxed);
                p
            }
            // A stored width that disagrees with the queried menu means the key
            // collided or the abstraction changed — treat as unknown, but count
            // it separately: a miss means "never trained here", a mismatch means
            // "trained a different game", and they need different fixes.
            Some(_) => {
                self.stats.width_mismatch.fetch_add(1, Relaxed);
                vec![1.0 / num_actions as f64; num_actions]
            }
            None => {
                self.stats.miss.fetch_add(1, Relaxed);
                vec![1.0 / num_actions as f64; num_actions]
            }
        }
    }

    /// Snapshot of how the blueprint lookups have resolved so far.
    pub fn lookup_counts(&self) -> LookupCounts {
        LookupCounts {
            hit: self.stats.hit.load(Relaxed),
            miss: self.stats.miss.load(Relaxed),
            width_mismatch: self.stats.width_mismatch.load(Relaxed),
        }
    }

    /// Number of info sets the policy has a strategy for.
    pub fn len(&self) -> usize {
        match &self.store {
            Store::Hashed { entries, .. } => entries.len(),
            Store::Dense { visited, .. } => *visited as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes the strategy occupies in memory (keys and probabilities).
    pub fn resident_bytes(&self) -> usize {
        match &self.store {
            Store::Hashed { entries, probs } => {
                entries.len() * std::mem::size_of::<Entry>() + probs.len() * std::mem::size_of::<f32>()
            }
            Store::Dense { probs, .. } => probs.len() * std::mem::size_of::<u16>(),
        }
    }
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// **Stream** a **hashed** policy — the `HashMap<u64, Vec<f32>>` artifact —
/// to `path`, one entry at a time: the write counterpart of
/// [`CompactPolicy::load`], and the reason the two live in the same module —
/// the byte format is defined once.
///
/// ## Why this exists rather than `bincode::serialize(&map)`
///
/// The export used to build the whole `HashMap` *and* then hand it to
/// `bincode::serialize`, which returns a `Vec<u8>` of the entire file. At
/// blueprint scale (~390 M entries on the last server run, a 9.8 GB artifact)
/// that is ~23 GB of map plus ~10 GB of byte buffer, **on top of** the regret
/// table, which is still resident. The peak lands at the very end of a
/// multi-hour run, so an OOM there costs the whole run. Streaming holds one
/// entry at a time; the checkpoint path took the same medicine for the same
/// reason (see `SoaMccfr::save_checkpoint`).
///
/// The count prefix is written as a placeholder and patched after the entries
/// are known, so a caller cannot desync the header from the body by
/// miscounting. Writes to a temp file and renames, so a crash mid-export leaves
/// the previous artifact intact.
///
/// Returns the number of entries written.
pub fn write_policy<I>(path: impl AsRef<Path>, entries: I) -> io::Result<u64>
where
    I: IntoIterator<Item = (u64, Vec<f32>)>,
{
    let path = path.as_ref();
    let tmp = path.with_extension("bin.tmp");
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);

    // Placeholder for the entry count; patched below once it is known.
    w.write_all(&0u64.to_le_bytes())?;
    let mut count = 0u64;
    for (key, probs) in entries {
        if probs.len() > 0xF {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("action count {} exceeds the 15-action packing limit", probs.len()),
            ));
        }
        w.write_all(&key.to_le_bytes())?;
        w.write_all(&(probs.len() as u64).to_le_bytes())?;
        for p in probs {
            w.write_all(&p.to_le_bytes())?;
        }
        count += 1;
    }

    w.seek(SeekFrom::Start(0))?;
    w.write_all(&count.to_le_bytes())?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(count)
}

/// Write a **dense** policy to `path`: one row per info set of `layout`, in
/// index order.  `row` is asked for each info set's distribution — it fills
/// the buffer and returns `true`, or returns `false` for an info set training
/// never reached.  Returns the number of trained rows written.
///
/// Streamed, for the reason [`write_policy`] is: the caller is the trainer,
/// with its table still resident.  The trained-row count is patched into the
/// header afterwards, and the file is renamed into place only once complete.
pub fn write_dense_policy(
    path: impl AsRef<Path>,
    layout: &Layout,
    mut row: impl FnMut(Span, &mut Vec<f64>) -> bool,
) -> io::Result<u64> {
    let path = path.as_ref();
    let tmp = path.with_extension("bin.tmp");
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
    w.write_all(&DENSE_MAGIC.to_le_bytes())?;
    w.write_all(&0u64.to_le_bytes())?; // trained-row count, patched below
    bincode::serialize_into(&mut w, layout).map_err(io::Error::other)?;

    let mut visited = 0u64;
    let mut probs = Vec::new();
    let mut quantized = vec![0u16; layout.max_width()];
    for at in layout.spans() {
        let q = &mut quantized[..at.n];
        probs.clear();
        if row(at, &mut probs) {
            if probs.len() != at.n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("a row of {} probabilities for an info set of {} actions", probs.len(), at.n),
                ));
            }
            quantize(&probs, q);
        } else {
            q.fill(0);
        }
        visited += u64::from(q.iter().any(|&x| x != 0));
        for x in q.iter() {
            w.write_all(&x.to_le_bytes())?;
        }
    }

    w.seek(SeekFrom::Start(8))?;
    w.write_all(&visited.to_le_bytes())?;
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)?;
    Ok(visited)
}

/// What [`compact_policy`] wrote, and how it checked out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactReport {
    /// Info sets carried over — every entry of the hashed policy.
    pub info_sets: u64,
    /// Probabilities written (one `u16` each).
    pub slots: u64,
    /// The largest difference between a probability read back from the dense
    /// file and the one the hashed policy holds for the same info set.
    pub max_error: f64,
}

/// Rewrite a **hashed** policy as a **dense** one at `out`, and prove the
/// result: the file is loaded back and every info set of the game is looked
/// up in both, which must agree on whether it was trained and, where it was,
/// on each probability to within one `u16` step.
///
/// `game` is the game the policy was trained on, with its dense index built
/// and hashed keys (not [`with_dense_keys`](BlueprintHoldem::with_dense_keys))
/// — the index says where each info set goes, the hash finds it in `hashed`.
/// If that is the wrong game (another raise cap, stack depth or bucket count),
/// entries of the policy match no info set of it; that is an error, and so is
/// any disagreement in the read-back.  A failed conversion leaves no file.
///
/// `progress(stage, done, total)` is called now and then with info sets
/// processed — at blueprint scale each of the two passes is hundreds of
/// millions of binary searches into the hashed policy.
pub fn compact_policy(
    hashed: &CompactPolicy,
    game: &BlueprintHoldem,
    out: impl AsRef<Path>,
    mut progress: impl FnMut(&str, usize, usize),
) -> io::Result<CompactReport> {
    const TICK: usize = 1 << 24;
    let out = out.as_ref();
    let bad = |why: String| io::Error::new(io::ErrorKind::InvalidData, why);
    if hashed.is_dense() {
        return Err(bad("the policy is already dense".into()));
    }
    if game.uses_dense_keys() {
        return Err(bad("conversion needs the game's hashed keys to find each info set".into()));
    }
    let layout = game.layout();
    let total = layout.capacity();
    // What the hashed policy says about info set `index`, as a distribution
    // over exactly its `n` actions.
    let lookup = |index: usize, n: usize| hashed.get(game.info_key_at(index)).filter(|p| p.len() == n);

    let mut index = 0usize;
    let placed = write_dense_policy(out, &layout, |at, row| {
        if index.is_multiple_of(TICK) {
            progress("writing", index, total);
        }
        let found = lookup(index, at.n);
        index += 1;
        found.map(|p| *row = p).is_some()
    })?;

    let mut check = || -> io::Result<f64> {
        if placed != hashed.len() as u64 {
            return Err(bad(format!(
                "only {placed} of the policy's {} entries are info sets of this game — \
                 it was trained with another --cap, --stack-bb or set of bucket maps",
                hashed.len()
            )));
        }
        let dense = CompactPolicy::load(out)?;
        let mut max_error = 0.0f64;
        for (index, at) in layout.spans().enumerate() {
            if index.is_multiple_of(TICK) {
                progress("verifying", index, total);
            }
            match (lookup(index, at.n), dense.get(index as u64)) {
                (None, None) => {}
                (Some(want), Some(got)) => {
                    let sum: f64 = want.iter().sum();
                    for (w, g) in want.iter().zip(&got) {
                        max_error = max_error.max((w / sum - g).abs());
                    }
                }
                (want, _) => {
                    return Err(bad(format!(
                        "info set {index} is {} in the hashed policy but not in the file written from it",
                        if want.is_some() { "trained" } else { "untrained" }
                    )))
                }
            }
        }
        // One quantization step, plus slack for the hashed file's own `f32`s.
        if max_error >= 1.0 / Q_ONE as f64 + 1e-6 {
            return Err(bad(format!("a probability changed by {max_error:.2e}, more than one u16 step")));
        }
        Ok(max_error)
    };
    match check() {
        Ok(max_error) => Ok(CompactReport { info_sets: placed, slots: layout.total_slots() as u64, max_error }),
        Err(e) => {
            std::fs::remove_file(out).ok();
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// What a hashed policy returns for stored `f32`s.
    fn f64s(p: &[f32]) -> Vec<f64> {
        p.iter().map(|&x| x as f64).collect()
    }

    #[test]
    fn load_round_trips_the_trainer_artifact_format() {
        // Serialize exactly what train.rs writes and stream it back.
        let mut map: HashMap<u64, Vec<f32>> = HashMap::new();
        map.insert(7, vec![0.25, 0.75]);
        map.insert(42, vec![0.1, 0.2, 0.7]);
        map.insert(u64::MAX, vec![1.0]);
        let bytes = bincode::serialize(&map).unwrap();
        let dir = std::env::temp_dir().join("poker_ai_policy_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("policy.bin");
        std::fs::write(&path, &bytes).unwrap();

        let p = CompactPolicy::load(&path).unwrap();
        assert_eq!(p.len(), 3);
        assert!(!p.is_dense());
        assert_eq!(p.get(7).unwrap(), f64s(&[0.25, 0.75]));
        assert_eq!(p.get(42).unwrap(), f64s(&[0.1, 0.2, 0.7]));
        assert_eq!(p.get(u64::MAX).unwrap(), f64s(&[1.0]));
        assert_eq!(p.get(8), None);
        assert!(p.contains(42) && !p.contains(8));
        std::fs::remove_file(&path).ok();
    }

    /// The streamed writer must produce bytes `bincode` still reads back as the
    /// `HashMap<u64, Vec<f32>>` it replaced — the compatibility claim that lets
    /// the export skip building the map at all.  Checked *through bincode*, not
    /// against a byte snapshot, because `HashMap` iteration order is arbitrary.
    #[test]
    fn streamed_writes_stay_bincode_hashmap_compatible() {
        let dir = std::env::temp_dir().join("poker_ai_policy_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("streamed.bin");
        let entries =
            vec![(7u64, vec![0.25f32, 0.75]), (42, vec![0.1, 0.2, 0.7]), (u64::MAX, vec![1.0])];

        let n = write_policy(&path, entries.clone()).unwrap();
        assert_eq!(n, 3, "returns the count it patched into the header");

        let bytes = std::fs::read(&path).unwrap();
        let decoded: HashMap<u64, Vec<f32>> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, entries.iter().cloned().collect::<HashMap<_, _>>());

        // …and the streaming reader the bot actually uses agrees.
        let p = CompactPolicy::load(&path).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p.get(42).unwrap(), f64s(&[0.1, 0.2, 0.7]));
        std::fs::remove_file(&path).ok();
    }

    /// An empty export is a real case (a run that visited nothing) and the
    /// count patch-back must still leave a valid zero-length map.
    #[test]
    fn streamed_empty_policy_is_a_valid_empty_map() {
        let dir = std::env::temp_dir().join("poker_ai_policy_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.bin");
        assert_eq!(write_policy(&path, Vec::new()).unwrap(), 0);
        let bytes = std::fs::read(&path).unwrap();
        let decoded: HashMap<u64, Vec<f32>> = bincode::deserialize(&bytes).unwrap();
        assert!(decoded.is_empty());
        assert!(CompactPolicy::load(&path).unwrap().is_empty());
        std::fs::remove_file(&path).ok();
    }

    /// The writer rejects what the reader would reject, rather than emitting a
    /// file that only fails at load time — on the far side of a training run.
    #[test]
    fn streamed_writer_rejects_an_unpackable_action_count() {
        let dir = std::env::temp_dir().join("poker_ai_policy_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("toowide.bin");
        let err = write_policy(&path, vec![(1u64, vec![0.0625f32; 16])]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(!path.exists(), "a rejected export must not replace the artifact");
    }

    /// The counter has to separate the two failure modes, because they have
    /// different fixes: a **miss** means the blueprint never trained that info
    /// set (undertrained tail, or the wrong artifact), a **width mismatch**
    /// means the key resolved into a differently-shaped game (a `--cap` or
    /// `--stack-bb` that disagrees with training). Lumping them together would
    /// have made the `--cap` incident look like sparse coverage.
    #[test]
    fn lookup_counts_separate_hits_misses_and_width_mismatches() {
        let p = CompactPolicy::from_entries(vec![(5, vec![0.9, 0.1]), (9, vec![0.3, 0.3, 0.4])]);
        assert_eq!(p.lookup_counts().total(), 0, "nothing counted before any lookup");
        assert_eq!(p.lookup_counts().hit_rate(), 1.0, "no lookups is not a failure");

        p.probs_or_uniform(5, 2); // hit
        p.probs_or_uniform(9, 3); // hit
        p.probs_or_uniform(6, 2); // miss: no such key
        p.probs_or_uniform(5, 3); // width mismatch: key exists, wrong menu

        let c = p.lookup_counts();
        assert_eq!(c, LookupCounts { hit: 2, miss: 1, width_mismatch: 1 });
        assert_eq!(c.total(), 4);
        assert!((c.hit_rate() - 0.5).abs() < 1e-12);
    }

    /// The whole point is catching a totally-missing blueprint, so pin the
    /// degenerate case: an empty policy answers nothing and must report a 0%
    /// hit rate rather than the "no lookups yet" 1.0.
    #[test]
    fn an_empty_policy_reports_a_zero_hit_rate_once_queried() {
        let p = CompactPolicy::from_entries(vec![]);
        for k in 0..10u64 {
            p.probs_or_uniform(k, 3);
        }
        let c = p.lookup_counts();
        assert_eq!(c.hit, 0);
        assert_eq!(c.miss, 10);
        assert_eq!(c.hit_rate(), 0.0);
    }

    #[test]
    fn uniform_fallback_on_miss_and_width_mismatch() {
        let p = CompactPolicy::from_entries(vec![(5, vec![0.9, 0.1])]);
        assert_eq!(p.probs_or_uniform(5, 2), vec![0.9f32 as f64, 0.1f32 as f64]);
        assert_eq!(p.probs_or_uniform(6, 4), vec![0.25; 4], "miss is uniform");
        assert_eq!(p.probs_or_uniform(5, 3), vec![1.0 / 3.0; 3], "width mismatch is uniform");
    }

    // ── dense format ────────────────────────────────────────────────────────

    fn temp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("poker_ai_dense_policy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn quantized_rows_sum_to_one_and_stay_within_a_step() {
        let mut rng = 0x5EEDu64;
        let mut unit = || crate::util::rng::xorshift_next_unit(&mut rng);
        for case in 0..20_000 {
            let n = 1 + case % 8;
            // Mix ordinary rows with the awkward ones: exact zeros, near-ties,
            // and probabilities far below one u16 step.
            let mut p: Vec<f64> = (0..n)
                .map(|_| match (unit() * 4.0) as u32 {
                    0 => 0.0,
                    1 => unit() * 1e-7,
                    _ => unit(),
                })
                .collect();
            if p.iter().all(|&x| x == 0.0) {
                p[0] = 1.0;
            }
            let total: f64 = p.iter().sum();
            let mut q = vec![0u16; n];
            quantize(&p, &mut q);
            assert_eq!(q.iter().map(|&x| x as u32).sum::<u32>(), Q_ONE as u32, "row {p:?} -> {q:?}");
            for (&x, &want) in q.iter().zip(&p) {
                let got = x as f64 / Q_ONE as f64;
                assert!((got - want / total).abs() < 1.0 / Q_ONE as f64, "{want} of {total} stored as {got}");
                // Purification and the range updates both treat zero as "never":
                // an action the strategy does not play must not acquire mass.
                assert!(want > 0.0 || x == 0, "a zero probability stored as {x}");
            }
        }
        // No mass at all is the "never trained" row, not a division by zero.
        let mut q = [7u16; 3];
        quantize(&[0.0, 0.0, 0.0], &mut q);
        assert_eq!(q, [0, 0, 0]);
    }

    /// Info sets of widths 2, 2, 3, 3, 3, 2 — three runs — with the second and
    /// fifth never trained.
    fn sample_rows() -> (Layout, Vec<Option<Vec<f64>>>) {
        let rows = vec![
            Some(vec![0.25, 0.75]),
            None,
            Some(vec![0.1, 0.2, 0.7]),
            Some(vec![1.0, 0.0, 0.0]),
            None,
            Some(vec![1.0 / 3.0, 2.0 / 3.0]),
        ];
        let widths = [2usize, 2, 3, 3, 3, 2];
        (Layout::build(widths.len(), |i| widths[i]), rows)
    }

    fn write_sample(path: &Path) -> (Layout, Vec<Option<Vec<f64>>>) {
        let (layout, rows) = sample_rows();
        let mut next = rows.iter();
        let written = write_dense_policy(path, &layout, |_, out| match next.next().unwrap() {
            Some(p) => {
                out.extend_from_slice(p);
                true
            }
            None => false,
        })
        .unwrap();
        assert_eq!(written, 4, "the trained rows are counted");
        (layout, rows)
    }

    #[test]
    fn a_dense_policy_round_trips_by_index() {
        let path = temp("round_trip.bin");
        let (layout, rows) = write_sample(&path);
        // The whole point: 2 bytes per probability and nothing per info set.
        let header = 16 + bincode::serialized_size(&layout).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), header + 2 * layout.total_slots() as u64);

        let p = CompactPolicy::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(p.is_dense());
        assert_eq!(p.len(), 4);
        assert_eq!(p.resident_bytes(), 2 * layout.total_slots());
        for (index, row) in rows.iter().enumerate() {
            match (row, p.get(index as u64)) {
                (None, got) => assert_eq!(got, None, "info set {index} was never trained"),
                (Some(want), Some(got)) => {
                    assert_eq!(got.len(), want.len());
                    assert_eq!(got.iter().sum::<f64>(), 1.0, "a stored row sums to exactly one");
                    for (g, w) in got.iter().zip(want) {
                        assert!((g - w).abs() < 1.0 / Q_ONE as f64, "info set {index}: {g} vs {w}");
                    }
                }
                (Some(_), None) => panic!("info set {index} lost its row"),
            }
            assert_eq!(p.contains(index as u64), row.is_some());
        }
        assert_eq!(p.get(3).unwrap(), vec![1.0, 0.0, 0.0], "a pure strategy is stored exactly");
        // Keys that are no info set of the game are misses, not neighbours.
        assert_eq!(p.get(rows.len() as u64), None);
        assert_eq!(p.get(crate::games::blueprint::NO_INFO_SET), None);
    }

    #[test]
    fn dense_lookups_are_counted_like_hashed_ones() {
        let path = temp("counts.bin");
        write_sample(&path);
        let p = CompactPolicy::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(p.probs_or_uniform(0, 2), vec![16384.0 / 65535.0, 49151.0 / 65535.0]); // hit
        assert_eq!(p.probs_or_uniform(1, 2), vec![0.5; 2]); // never trained
        assert_eq!(p.probs_or_uniform(2, 2), vec![0.5; 2]); // three actions stored
        assert_eq!(p.probs_or_uniform(99, 4), vec![0.25; 4]); // no such info set
        assert_eq!(p.lookup_counts(), LookupCounts { hit: 1, miss: 2, width_mismatch: 1 });
    }

    #[test]
    fn a_damaged_dense_file_does_not_load() {
        let path = temp("damaged.bin");
        write_sample(&path);
        let good = std::fs::read(&path).unwrap();

        // Cut short: the tail of the table would silently read as untrained.
        std::fs::write(&path, &good[..good.len() - 2]).unwrap();
        assert_eq!(CompactPolicy::load(&path).err().unwrap().kind(), io::ErrorKind::InvalidData);
        // Extra bytes: not the file its header describes.
        let mut long = good.clone();
        long.extend_from_slice(&[0, 0]);
        std::fs::write(&path, &long).unwrap();
        assert_eq!(CompactPolicy::load(&path).err().unwrap().kind(), io::ErrorKind::InvalidData);
        // A layout that contradicts itself (its info-set total, the last field
        // before the probabilities, off by one).
        let mut skewed = good.clone();
        let (layout, _) = sample_rows();
        let capacity_at = 16 + bincode::serialized_size(&layout).unwrap() as usize - 16;
        skewed[capacity_at] ^= 1;
        std::fs::write(&path, &skewed).unwrap();
        assert_eq!(CompactPolicy::load(&path).err().unwrap().kind(), io::ErrorKind::InvalidData);

        std::fs::write(&path, &good).unwrap();
        assert!(CompactPolicy::load(&path).is_ok(), "the undamaged bytes still load");
        std::fs::remove_file(&path).ok();
    }

    // ── dense policy × game ─────────────────────────────────────────────────

    use crate::abstraction::bucket_map::BucketMap;
    use crate::games::CursorGame;
    use crate::solver::dcfr::Discount;
    use crate::solver::mccfr::SoaMccfr;
    use crate::solver::variant::Variant;

    /// A blueprint game small enough for the fast lane.  With the stack equal
    /// to the big blind the only decision is the opener's fold / all-in, so no
    /// post-flop node exists and the placeholder maps are never queried; a
    /// deeper stack gives the same game a different tree (and so a different
    /// table layout) without dealing a card.
    fn tiny_game(stack: u32) -> BlueprintHoldem {
        BlueprintHoldem::new(stack, 2, 1, 0)
            .with_raise_cap(1)
            .with_street_bucket(0, BucketMap::placeholder(&[2, 3], 50))
            .with_street_bucket(1, BucketMap::placeholder(&[2, 4], 50))
            .with_street_bucket(2, BucketMap::placeholder(&[2, 5], 50))
    }

    /// Train the tiny game and export its strategy both ways: the dense file
    /// straight from the table, the hashed one as the trainer used to write it.
    fn trained_tiny(dense: &Path, hashed: &Path) -> SoaMccfr<BlueprintHoldem> {
        let mut soa: SoaMccfr<BlueprintHoldem> =
            SoaMccfr::with_seed(tiny_game(2).with_indexing(), Variant::Dcfr(Discount::RECOMMENDED), 3);
        soa.train(30_000);
        write_dense_policy(dense, soa.layout(), |at, out| soa.average_strategy_into(at, out)).unwrap();
        let keys = tiny_game(2).with_indexing();
        let entries = (0..soa.capacity()).filter(|&i| soa.is_visited(i)).map(|i| {
            (keys.info_key_at(i), soa.average_strategy_at(i).into_iter().map(|x| x as f32).collect())
        });
        write_policy(hashed, entries).unwrap();
        soa
    }

    /// Every starting hand's key at the opening decision of `game`.
    fn opening_keys(game: &BlueprintHoldem) -> Vec<u64> {
        let state = game.play_state([[0, 1], [2, 3]], [poker_core::state::NO_CARD; 5]);
        let mut keys = Vec::new();
        for hi in 1..52u8 {
            for lo in 0..hi {
                keys.push(game.info_key_with_hole(&state, [lo, hi]));
            }
        }
        keys
    }

    #[test]
    fn a_dense_policy_plays_like_the_hashed_one_it_replaces() {
        let (dense_path, hashed_path) = (temp("tiny_dense.bin"), temp("tiny_hashed.bin"));
        let soa = trained_tiny(&dense_path, &hashed_path);
        let dense = CompactPolicy::load(&dense_path).unwrap();
        let hashed = CompactPolicy::load(&hashed_path).unwrap();
        assert!(dense.is_dense() && !hashed.is_dense());
        assert_eq!(dense.len(), hashed.len(), "the same info sets are trained in both");
        assert!(dense.len() >= 169, "every starting-hand class was reached");

        // Each policy readies its own game; the dense one changes what a key is.
        let dense_game = dense.bind(tiny_game(2)).unwrap();
        let hashed_game = hashed.bind(tiny_game(2)).unwrap();
        assert!(dense_game.uses_dense_keys() && !hashed_game.uses_dense_keys());
        dense.assert_keys_match(&dense_game);
        hashed.assert_keys_match(&hashed_game);

        // All 1326 hands at the one decision: the same strategy from both, to
        // a u16 step, and the dense key is the trainer's own index.
        let (dk, hk) = (opening_keys(&dense_game), opening_keys(&hashed_game));
        for (&d, &h) in dk.iter().zip(&hk) {
            assert!((d as usize) < soa.capacity(), "a dense key is an info-set index");
            let (pd, ph) = (dense.probs_or_uniform(d, 2), hashed.probs_or_uniform(h, 2));
            for ((a, b), want) in pd.iter().zip(&ph).zip(soa.average_strategy_at(d as usize)) {
                assert!((a - b).abs() < 1.0 / Q_ONE as f64 + 1e-7, "dense {a} vs hashed {b}");
                assert!((a - want).abs() < 1.0 / Q_ONE as f64, "dense {a} vs the solver's {want}");
            }
        }
        assert_eq!(dense.lookup_counts(), LookupCounts { hit: 1326, miss: 0, width_mismatch: 0 });
        assert_eq!(hashed.lookup_counts(), dense.lookup_counts());

        // The cursor's key is the same index, so training on a dense-key game
        // and playing from one agree on what a key means.
        let mut c = CursorGame::root(&dense_game);
        let mut rng = 11u64;
        CursorGame::sample_chance(&dense_game, &mut c, || crate::util::rng::xorshift_next_unit(&mut rng));
        assert_eq!(CursorGame::info_key(&dense_game, &c), dense_game.info_set_index(&c) as u64);

        std::fs::remove_file(&dense_path).ok();
        std::fs::remove_file(&hashed_path).ok();
    }

    #[test]
    fn a_dense_policy_refuses_a_game_it_was_not_trained_on() {
        let (dense_path, hashed_path) = (temp("refuse_dense.bin"), temp("refuse_hashed.bin"));
        trained_tiny(&dense_path, &hashed_path);
        let dense = CompactPolicy::load(&dense_path).unwrap();
        std::fs::remove_file(&dense_path).ok();
        std::fs::remove_file(&hashed_path).ok();

        // Deeper stacks: a different betting tree, so the file's rows are not
        // this game's info sets.  With hashed keys this loaded and missed.
        let err = dense.bind(tiny_game(8)).err().expect("a different tree must not bind");
        assert!(err.contains("different game"), "{err}");
        // A game that cannot have a dense index at all (no raise cap).
        let uncapped = BlueprintHoldem::new(2, 2, 1, 0);
        assert!(dense.bind(uncapped).err().unwrap().contains("dense index"));
    }

    #[test]
    #[should_panic(expected = "disagree on what an info key is")]
    fn mixing_key_kinds_is_caught() {
        let path = temp("mixed.bin");
        write_sample(&path);
        let dense = CompactPolicy::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        dense.assert_keys_match(&tiny_game(2)); // hashed keys
    }

    #[test]
    fn compacting_a_hashed_policy_gives_the_dense_one() {
        let (dense_path, hashed_path) = (temp("compact_dense.bin"), temp("compact_hashed.bin"));
        let out = temp("compact_out.bin");
        trained_tiny(&dense_path, &hashed_path);
        let hashed = CompactPolicy::load(&hashed_path).unwrap();
        let game = tiny_game(2).with_indexing();

        let mut stages = Vec::new();
        let report = compact_policy(&hashed, &game, &out, |stage, _, _| stages.push(stage.to_string())).unwrap();
        assert_eq!(report.info_sets, hashed.len() as u64, "every entry carried over");
        assert!(report.max_error < 1.0 / Q_ONE as f64 + 1e-6, "got {}", report.max_error);
        assert_eq!(stages, ["writing", "verifying"]);

        // The converted file answers like the one exported from the table
        // (they were quantized from f32 and f64 copies of the same numbers).
        let converted = CompactPolicy::load(&out).unwrap();
        let direct = CompactPolicy::load(&dense_path).unwrap();
        assert_eq!(converted.len(), direct.len());
        for index in 0..game.info_set_capacity() as u64 {
            match (converted.get(index), direct.get(index)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    for (x, y) in a.iter().zip(&b) {
                        assert!((x - y).abs() <= 1.0 / Q_ONE as f64, "info set {index}: {x} vs {y}");
                    }
                }
                other => panic!("info set {index} differs in being trained: {other:?}"),
            }
        }

        // The wrong game: its info sets are not the policy's.  An error, and
        // no half-right file left behind.
        std::fs::remove_file(&out).unwrap();
        let wrong = tiny_game(8).with_indexing();
        let err = compact_policy(&hashed, &wrong, &out, |_, _, _| {}).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(!out.exists(), "a failed conversion must not leave a file");
        // Nor is there anything to convert in a file that is already dense.
        assert!(compact_policy(&direct, &game, &out, |_, _, _| {}).is_err());

        for p in [&dense_path, &hashed_path] {
            std::fs::remove_file(p).ok();
        }
    }
}
