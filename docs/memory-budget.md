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
| Average-strategy numerator | `f32`, stochastically rounded | 4 |
| VR-MCCFR baseline | `f32` | 4 |
| **Total, default layout** | | **12** |
| `prev_inst` — only with `--optimistic` | `f32` | +4 |
| `consec_below` — only with `--rbp` | `u32` | +4 |

```
total_memory ≈ num_info_sets × avg_actions_per_set × 12 bytes
```

The strategy sum used to be `f64`, which made this 16 B/slot — half the table
in one array. `f32` genuinely did break it: iteration `t` carries weight `t^γ`,
so after `n` visits an increment is ~`1/n` of the running sum, and past
`n ≈ 2^24` it falls below half an ulp and the deployed average **silently
freezes**. That is a property of round-to-nearest, not of `f32`: the sum is now
accumulated with **stochastic rounding**, which carries a sub-ulp increment up
with probability equal to its share of the gap, so it accumulates in
expectation. Safe here specifically because this array is write-only and
monotone growing — never read back into the update — which is the distinction
that sank the `bf16` regret experiment. See `solver::regret_table` for the
argument and its gating tests. `RegretTable::bytes_per_info_set` is the
authority — it is what the trainer prints — so check it against any hand
arithmetic here.

The optimistic and pruning arrays are allocated only when their feature is
enabled, so a default run pays 12 B/slot and nothing more.

The quantized `LeanTable` store (`i16` / `u16` / `i16`) cuts this to 6 B/slot —
2× smaller, down from 2.7× now that the strategy sum is `f32` — and must be
paired with `Discount::LINEAR`; see
[options.md](options.md) §1–2 for why DCFR's β=0 is incompatible with 16-bit
storage.
