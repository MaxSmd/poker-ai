//! Compact, read-only view of a trained blueprint strategy.
//!
//! The trainer persists `data/blueprint_holdem.bin` as a bincode
//! `HashMap<u64, Vec<f32>>` (info key → action distribution).  At Slumbot depth
//! that is ~296 M entries: deserializing it back into a `HashMap` would cost
//! ~3× the file size in allocator and table overhead.  This loader streams the
//! bincode format directly into three flat arrays — sorted keys, packed
//! offset+length, and one contiguous probability pool — so the resident
//! footprint stays close to the file size and lookups are a binary search.

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// A key's probabilities live at `offset = packed >> 4`, `len = packed & 0xF`.
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    packed: u64,
}

/// Sorted flat map from info key to an action distribution.
pub struct CompactPolicy {
    entries: Vec<Entry>,
    probs: Vec<f32>,
}

impl CompactPolicy {
    /// Stream-load a bincode `HashMap<u64, Vec<f32>>` artifact
    /// (`data/blueprint_holdem.bin`) without materializing the `HashMap`.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let n = read_u64(&mut r)? as usize;
        let mut entries = Vec::with_capacity(n);
        let mut probs = Vec::new();
        for _ in 0..n {
            let key = read_u64(&mut r)?;
            let len = read_u64(&mut r)? as usize;
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
        Ok(Self { entries, probs })
    }

    /// Build from explicit entries (tests / small games).
    pub fn from_entries(mut list: Vec<(u64, Vec<f32>)>) -> Self {
        list.sort_unstable_by_key(|(k, _)| *k);
        let mut entries = Vec::with_capacity(list.len());
        let mut probs = Vec::new();
        for (key, p) in list {
            assert!(p.len() <= 0xF, "action count exceeds the packing limit");
            entries.push(Entry { key, packed: ((probs.len() as u64) << 4) | p.len() as u64 });
            probs.extend_from_slice(&p);
        }
        Self { entries, probs }
    }

    /// The stored distribution for `key`, if the blueprint visited it.
    pub fn get(&self, key: u64) -> Option<&[f32]> {
        let i = self.entries.binary_search_by_key(&key, |e| e.key).ok()?;
        let e = self.entries[i];
        let off = (e.packed >> 4) as usize;
        let len = (e.packed & 0xF) as usize;
        Some(&self.probs[off..off + len])
    }

    /// The distribution for `key` as `f64`, falling back to uniform over
    /// `num_actions` when the blueprint never visited the info set.
    pub fn probs_or_uniform(&self, key: u64, num_actions: usize) -> Vec<f64> {
        match self.get(key) {
            Some(p) if p.len() == num_actions => p.iter().map(|&x| x as f64).collect(),
            // A stored width that disagrees with the queried menu means the key
            // collided or the abstraction changed — treat as unknown.
            _ => vec![1.0 / num_actions as f64; num_actions],
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
        assert_eq!(p.get(7).unwrap(), &[0.25, 0.75]);
        assert_eq!(p.get(42).unwrap(), &[0.1, 0.2, 0.7]);
        assert_eq!(p.get(u64::MAX).unwrap(), &[1.0]);
        assert_eq!(p.get(8), None);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn uniform_fallback_on_miss_and_width_mismatch() {
        let p = CompactPolicy::from_entries(vec![(5, vec![0.9, 0.1])]);
        assert_eq!(p.probs_or_uniform(5, 2), vec![0.9f32 as f64, 0.1f32 as f64]);
        assert_eq!(p.probs_or_uniform(6, 4), vec![0.25; 4], "miss is uniform");
        assert_eq!(p.probs_or_uniform(5, 3), vec![1.0 / 3.0; 3], "width mismatch is uniform");
    }
}
