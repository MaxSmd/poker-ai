# Memory Budget

Formula:
```
total_memory = num_info_sets × avg_actions_per_set × 3 × 4 bytes
```

The `3` is the three `f32` accumulators kept per (info set, action): cumulative
regret, the average-strategy numerator, and the VR-MCCFR baseline value. (See
`RegretTable::bytes_per_info_set` in `solver/regret_table.rs`, which sizes the
flat blueprint store the same way.)

Run `cargo run --bin memory_estimate -- --help` to compute info set counts for your abstraction parameters before committing to RAM purchases.
