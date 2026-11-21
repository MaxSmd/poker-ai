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
| VR-MCCFR baseline — not with `--no-baseline` | `f32` | 4 |
| **Total, default layout** | | **12** |

```
total_memory = action_slots × 12 bytes        (× 8 with --no-baseline)
```

Nothing is stored per info set. Finding an info set's slots needs its first
slot and its action count; the table used to keep both for every info set
(5 bytes each — 3 GB of the 22.4 GB the 595M-info-set blueprint was sized
at). Info sets come in blocks, though — one per betting-tree node, all of that
node's width — so the same information is a run-length
[`Layout`](../crates/poker-ai/src/games/layout.rs) with one entry per node: a
few megabytes at any scale. The trainer does not even search it; its cursor
knows which node it is at.

## What the trained strategy costs

`train blueprint --soa` writes the average strategy in the same order as the
table, one `u16` probability per slot
([`play::policy`](../crates/poker-ai/src/play/policy.rs)):

```
policy_memory = action_slots × 2 bytes        (on disk and in `play`)
```

The format before it kept a hashed key and an offset for every info set and
`f32` probabilities — 16 bytes per info set plus 4 per slot, 16 GB where the
dense form is 3.2 GB. `play compact` converts an old file. `u16` resolves a
probability to 1.5e-5; a row is quantized so that it still sums to exactly
one and an action that was never played stays at exactly zero.

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

The baseline array is allocated only when control variates are on, so a
`--no-baseline` run pays 8 B/slot.

The quantized `LeanTable` store (`i16` / `u16` / `i16`) cuts this to 6 B/slot —
2× smaller, down from 2.7× now that the strategy sum is `f32` — and must be
paired with `Discount::LINEAR`; see
[options.md](options.md) §1–2 for why DCFR's β=0 is incompatible with 16-bit
storage.
