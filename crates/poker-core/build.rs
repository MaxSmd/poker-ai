//! Build script: precompute flush and non-flush hand-rank lookup tables and
//! write them to `$OUT_DIR/lut_tables.rs`.
//!
//! **Flush table** (`FLUSH_LUT`)
//!   A flat `[u32; 8192]` array.  Index = 13-bit rank bitmask.  Only entries
//!   where `popcount(index) == 5` are meaningful; all others are zero.
//!
//! **Non-flush table** (`NOFLUSH_LUT`)
//!   A hash table stored as `[(u32, u32); NOFLUSH_SIZE]` using open addressing
//!   with linear probing.  Key = product of five rank primes; value = hand rank.
//!   An entry with key == 0 is empty.  Minimum valid key is 2^5 = 32, so 0
//!   is a safe sentinel.
//!
//! The tables are keyed on the same `u32` encoding used by `make_hand` in
//! `evaluator.rs` so that `evaluate_5_lut` returns bit-for-bit identical
//! results to `evaluate_5`.

use std::io::Write;
use std::path::PathBuf;

/// Primes assigned to ranks 0..=12 (two through ace).
const RANK_PRIMES: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// The 10 possible 5-card straight rank masks and their top rank.
const STRAIGHT_MASKS: [(u16, u8); 10] = [
    (0b1_1111_0000_0000, 12),
    (0b0_1111_1000_0000, 11),
    (0b0_0111_1100_0000, 10),
    (0b0_0011_1110_0000, 9),
    (0b0_0001_1111_0000, 8),
    (0b0_0000_1111_1000, 7),
    (0b0_0000_0111_1100, 6),
    (0b0_0000_0011_1110, 5),
    (0b0_0000_0001_1111, 4),
    (0b1_0000_0000_1111, 3), // wheel A-2-3-4-5
];

// ── Helpers (mirror of evaluator.rs, used only during build) ─────────────────

fn make_hand(cat: u8, r1: u8, r2: u8, r3: u8, r4: u8, r5: u8) -> u32 {
    ((cat as u32) << 20)
        | ((r1 as u32) << 16)
        | ((r2 as u32) << 12)
        | ((r3 as u32) << 8)
        | ((r4 as u32) << 4)
        | (r5 as u32)
}

fn find_best_straight(bits: u16) -> (bool, u8) {
    for (mask, top) in STRAIGHT_MASKS {
        if bits & mask == mask {
            return (true, top);
        }
    }
    (false, 0)
}

/// Return the top `n` set bit indices (rank values) from `bits`, high → low.
fn top_bits(bits: u16, n: usize) -> [u8; 5] {
    let mut out = [0u8; 5];
    let mut i = 0;
    for rank in (0..13usize).rev() {
        if i == n {
            break;
        }
        if (bits >> rank) & 1 == 1 {
            out[i] = rank as u8;
            i += 1;
        }
    }
    out
}

/// Evaluate the non-flush hand rank for a 5-card rank-frequency vector.
/// Flush detection is handled separately via `FLUSH_LUT`.
fn eval_noflush(freq: &[u8; 13]) -> u32 {
    let mut rank_bits = 0u16;
    for r in 0..13usize {
        if freq[r] > 0 {
            rank_bits |= 1u16 << r;
        }
    }

    let mut quad_rank = 0u8;
    let mut trips_rank = 0u8;
    let mut pair1 = 0u8;
    let mut pair2 = 0u8;
    let mut num_pairs = 0u8;
    let mut has_quads = false;
    let mut has_trips = false;

    for r in (0..13u8).rev() {
        match freq[r as usize] {
            4 => {
                quad_rank = r;
                has_quads = true;
            }
            3 => {
                trips_rank = r;
                has_trips = true;
            }
            2 => {
                if num_pairs == 0 {
                    pair1 = r;
                } else {
                    pair2 = r;
                }
                num_pairs += 1;
            }
            _ => {}
        }
    }

    let (is_straight, straight_top) = if rank_bits.count_ones() == 5 {
        find_best_straight(rank_bits)
    } else {
        (false, 0)
    };

    if has_quads {
        let kicker = (0..13u8)
            .rev()
            .find(|&r| r != quad_rank && freq[r as usize] > 0)
            .unwrap_or(0);
        make_hand(7, quad_rank, kicker, 0, 0, 0)
    } else if has_trips && num_pairs >= 1 {
        make_hand(6, trips_rank, pair1, 0, 0, 0)
    } else if is_straight {
        make_hand(4, straight_top, 0, 0, 0, 0)
    } else if has_trips {
        let mut k = [0u8; 2];
        let mut ki = 0;
        for r in (0..13u8).rev() {
            if freq[r as usize] > 0 && r != trips_rank {
                k[ki] = r;
                ki += 1;
                if ki == 2 {
                    break;
                }
            }
        }
        make_hand(3, trips_rank, k[0], k[1], 0, 0)
    } else if num_pairs >= 2 {
        let kicker = (0..13u8)
            .rev()
            .find(|&r| r != pair1 && r != pair2 && freq[r as usize] > 0)
            .unwrap_or(0);
        make_hand(2, pair1, pair2, kicker, 0, 0)
    } else if num_pairs == 1 {
        let mut k = [0u8; 3];
        let mut ki = 0;
        for r in (0..13u8).rev() {
            if freq[r as usize] > 0 && r != pair1 {
                k[ki] = r;
                ki += 1;
                if ki == 3 {
                    break;
                }
            }
        }
        make_hand(1, pair1, k[0], k[1], k[2], 0)
    } else {
        let r = top_bits(rank_bits, 5);
        make_hand(0, r[0], r[1], r[2], r[3], r[4])
    }
}

/// Enumerate all multisets of `remaining` cards from ranks `start..13` where
/// each rank appears at most 4 times (maximum hand occupancy).
///
/// Uses backtracking: for each rank from `start` to 12 we increment its
/// frequency, recurse with `remaining - 1` starting at the same rank (allowing
/// repeated picks), then restore the frequency.  This produces every
/// unordered multiset without duplicates.
///
/// The total count for `remaining = 5` is C(17,5) − 13 = **6175**.
fn gen_rank_combos(freq: &mut [u8; 13], remaining: u8, start: usize, out: &mut Vec<[u8; 13]>) {
    if remaining == 0 {
        out.push(*freq);
        return;
    }
    for rank in start..13 {
        if freq[rank] < 4 {
            freq[rank] += 1;
            gen_rank_combos(freq, remaining - 1, rank, out);
            freq[rank] -= 1;
        }
    }
}

// ── Hash table constants ──────────────────────────────────────────────────────

/// Power-of-2 size for the non-flush open-addressed hash table.
/// Load factor = 6175 / 16384 ≈ 37.7%, keeping average probe length very low.
const NOFLUSH_SIZE: usize = 16384;
const NOFLUSH_MASK: u32 = NOFLUSH_SIZE as u32 - 1;

/// Fibonacci / golden-ratio multiplicative hash constant (32-bit).
const HASH_MUL: u32 = 2_654_435_761;

#[inline]
fn noflush_slot(product: u32) -> usize {
    (product.wrapping_mul(HASH_MUL) & NOFLUSH_MASK) as usize
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // ── 1. Flush LUT (8192 entries, indexed by 13-bit rank bitmask) ──────────
    let mut flush_lut = [0u32; 8192];
    for bits in 0u16..8192 {
        if bits.count_ones() == 5 {
            let (is_sf, sf_top) = find_best_straight(bits);
            if is_sf {
                flush_lut[bits as usize] = make_hand(8, sf_top, 0, 0, 0, 0);
            } else {
                let r = top_bits(bits, 5);
                flush_lut[bits as usize] = make_hand(5, r[0], r[1], r[2], r[3], r[4]);
            }
        }
    }

    // ── 2. Non-flush LUT ─────────────────────────────────────────────────────
    let mut combos: Vec<[u8; 13]> = Vec::with_capacity(6200);
    let mut freq = [0u8; 13];
    gen_rank_combos(&mut freq, 5, 0, &mut combos);
    assert_eq!(combos.len(), 6175, "expected 6175 non-flush rank multisets");

    // Compute (prime_product, hand_rank) for each multiset.
    let entries: Vec<(u32, u32)> = combos
        .iter()
        .map(|f| {
            let product: u32 = f
                .iter()
                .enumerate()
                .map(|(r, &cnt)| {
                    if cnt > 0 {
                        RANK_PRIMES[r].pow(cnt as u32)
                    } else {
                        1
                    }
                })
                .product();
            let rank = eval_noflush(f);
            (product, rank)
        })
        .collect();

    // Build the open-addressed hash table.
    let mut noflush_lut = vec![(0u32, 0u32); NOFLUSH_SIZE];
    for &(product, rank) in &entries {
        let mut idx = noflush_slot(product);
        loop {
            if noflush_lut[idx].0 == 0 {
                noflush_lut[idx] = (product, rank);
                break;
            }
            idx = (idx + 1) & (NOFLUSH_SIZE - 1);
        }
    }

    // Verify every entry is retrievable and measure max probe length.
    let max_probes = entries
        .iter()
        .map(|&(product, _)| {
            let mut idx = noflush_slot(product);
            let mut probes = 1usize;
            while noflush_lut[idx].0 != product {
                idx = (idx + 1) & (NOFLUSH_SIZE - 1);
                probes += 1;
            }
            probes
        })
        .max()
        .unwrap_or(0);

    // Emit a note so the build log shows the hash quality.
    println!("cargo:warning=NOFLUSH_LUT max probe length: {max_probes}");

    // ── 3. Write generated file ───────────────────────────────────────────────
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let path = out_dir.join("lut_tables.rs");
    let mut file = std::fs::File::create(&path).unwrap();

    // FLUSH_LUT
    write!(file, "pub static FLUSH_LUT: [u32; 8192] = [").unwrap();
    for (i, &v) in flush_lut.iter().enumerate() {
        if i > 0 {
            write!(file, ",").unwrap();
        }
        write!(file, "{v}").unwrap();
    }
    writeln!(file, "];").unwrap();

    // NOFLUSH_LUT
    writeln!(
        file,
        "pub static NOFLUSH_LUT: [(u32, u32); {NOFLUSH_SIZE}] = ["
    )
    .unwrap();
    for &(k, v) in &noflush_lut {
        writeln!(file, "({k},{v}),").unwrap();
    }
    writeln!(file, "];").unwrap();

    println!("cargo:rerun-if-changed=build.rs");
}
