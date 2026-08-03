# Memory Budget

Do not size a run from a formula on this page — run the tool:

```sh
cargo run --release --bin memory_estimate            # current bucket counts
cargo run --release --bin memory_estimate -- 200 200 200   # what-if buckets
```

`memory_estimate` enumerates the **exact** abstract betting tree with the real
`poker-core` engine and prints info sets, action slots and table RAM for a
stack × raise-cap matrix. An earlier version of that tool guessed the footprint
from a generic betting-tree model and was off by ~400× because it ignored stack
depth; the arithmetic below is a sanity check, not a substitute.

## What a slot actually costs

Per `(info set, action)` slot in the production SoA store
([`RegretTable`](../crates/poker-ai/src/solver/regret_table.rs)):

| Accumulator | Type | Bytes |
|---|---|---|
| Cumulative regret | `f32` | 4 |
| Average-strategy numerator | **`f64`** | 8 |
| VR-MCCFR baseline | `f32` | 4 |
| **Total, default layout** | | **16** |
| `prev_inst` — only with `--optimistic` | `f32` | +4 |
| `consec_below` — only with `--rbp` | `u32` | +4 |

```
total_memory ≈ num_info_sets × avg_actions_per_set × 16 bytes
```

The strategy sum is `f64`, not `f32`: the average is accumulated over ~10^15
visits, and an `f32` numerator stops resolving increments long before that.
Sizing it as a third `f32` understates RAM by a third. `RegretTable::bytes_per_info_set`
is the authority — it is what the trainer prints — so check it against any hand
arithmetic here.

The optimistic and pruning arrays are allocated only when their feature is
enabled, so a default run pays 16 B/slot and nothing more.

The quantized `LeanTable` store (`i16` / `u16` / `i16`) cuts this to 6 B/slot —
2.7× smaller — and must be paired with `Discount::LINEAR`; see
[options.md](options.md) §1–2 for why DCFR's β=0 is incompatible with 16-bit
storage.
