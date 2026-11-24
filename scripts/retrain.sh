#!/usr/bin/env bash
#
# Retrain the blueprint finer and longer, then play it.
#
#   tmux new-session -d -s retrain '~/poker-ai/scripts/retrain.sh; exec bash'
#
# Why: the cap-3,3,2,1 blueprint (data/bp_3321, −7.3 ± 28.1 bb/100 against
# Slumbot) closed the 4-bet gap, but the leaks left in its hand log — the BB
# over-folding to 2bb opens, the passive turn lead, the flop c-bet — sit in
# the blueprint itself, and nothing in it is hand-coded to fix them.  So this
# run spends the budget on getting the blueprint closer to equilibrium:
#
#   * finer cards  12000/12000/1200 buckets (1.5× bp_3321's), so hands that
#                  play differently stop sharing a strategy;
#   * longer       10B iterations (2× bp_3321's; ~11 per info set vs ~8);
#   * no baseline  8 instead of 12 bytes per slot, which is what pays for the
#                  finer buckets: 19.45 GB, the same as bp_3321 (memory_estimate);
#   * β = 0.5      DCFR 1.5,0.5,2, which bench_discount found slower to start
#                  and then far ahead — its crossover is near ~10 iterations
#                  per info set, which is about where this run ends.
#
# Same caps as bp_3321, so the betting tree is unchanged.  Everything goes to
# its own directory; nothing else in data/ is touched.  Each step is skipped
# if its output exists, and training resumes from its checkpoint, so
# re-running after an interruption continues where it stopped.  The
# checkpoint is kept (KEEP_CKPT=1), so training can be extended later with a
# larger ITERS instead of starting over.
#
#   1. cluster  buckets, OCHS river                       ~5-8 h
#   2. train    10B iterations, 32 threads                ~1.5-2 days
#   3. chart    preflop strategy → $DIR/chart.txt         seconds
#   4. match    10k hands against Slumbot, logged         ~15 h
#
# Check $DIR/chart.txt as soon as it appears: the BB's response to the 2bb
# open should fold far less than bp_3321's 23% (Slumbot folds 6%).
#
# Result: $DIR/summary.txt (match result + spot report).
set -u

cd "$(dirname "$0")/.." || exit 1
DIR=${DIR:-data/bp_fine}
CAPS=${CAPS:-3,3,2,1}
BUCKETS=${BUCKETS:-"12000 12000 1200"}
ITERS=${ITERS:-10000000000}
TRAIN_FLAGS=${TRAIN_FLAGS:-"--no-baseline --discount=1.5,0.5,2"}
KEEP_CKPT=${KEEP_CKPT:-1}
THREADS=${THREADS:-32}
BIN=./target/release

for b in cluster train play; do
    [ -x "$BIN/$b" ] || { echo "missing $BIN/$b — run: cargo build --release"; exit 1; }
done
mkdir -p "$DIR" || exit 1
step() { printf '=== %s  %s\n' "$(date '+%m-%d %H:%M')" "$*"; }

if [ ! -f "$DIR/river_buckets.bin" ]; then
    step "cluster $BUCKETS -> $DIR"
    # shellcheck disable=SC2086  # BUCKETS is a word list
    POKER_AI_CLUSTER_MEM_GB=8 POKER_AI_RIVER_OCHS=1 RAYON_NUM_THREADS=$THREADS \
        nice -n 19 "$BIN/cluster" 0 1 $BUCKETS --data="$DIR" > "$DIR/cluster.log" 2>&1 \
        || { step "cluster FAILED — see $DIR/cluster.log"; exit 1; }
fi

if [ ! -f "$DIR/blueprint_holdem.bin" ]; then
    step "train $ITERS iterations, caps $CAPS, $TRAIN_FLAGS"
    # shellcheck disable=SC2086  # TRAIN_FLAGS is a word list
    nice -n 19 "$BIN/train" blueprint "$ITERS" 200 1 --cap="$CAPS" --soa --atomic="$THREADS" \
        $TRAIN_FLAGS --chunk=50000000 --resume --data="$DIR" >> "$DIR/train.log" 2>&1 \
        || { step "train FAILED — see $DIR/train.log"; exit 1; }
    # The checkpoint is as large as the table and only needed to resume or
    # to extend the run.
    [ "$KEEP_CKPT" = 1 ] || rm -f "$DIR/blueprint_holdem_soa.ckpt"
fi

if [ ! -f "$DIR/chart.txt" ]; then
    step "preflop chart -> $DIR/chart.txt"
    "$BIN/play" chart --cap="$CAPS" --data="$DIR" > "$DIR/chart.txt" 2>&1
fi

if [ ! -f "$DIR/hands_10k.jsonl" ] || [ "$(wc -l < "$DIR/hands_10k.jsonl")" -lt 10000 ]; then
    step "slumbot 10000 hands"
    nice -n 19 "$BIN/play" slumbot 10000 --cap="$CAPS" --data="$DIR" \
        --log-hands="$DIR/hands_10k.jsonl" > "$DIR/slumbot_10k.log" 2>&1
fi

{
    echo "Blueprint caps $CAPS, buckets $BUCKETS, $ITERS iterations, $TRAIN_FLAGS ($DIR)"
    grep -E "info sets|Saved" "$DIR/train.log" | tail -2
    echo
    echo "Slumbot, 10k hands (bp_3321, same play config: luck-adjusted -7.3 ± 28.1 bb/100):"
    grep -E "^Done:|^Blueprint lookups|^Abstraction tracking" "$DIR/slumbot_10k.log"
    python3 scripts/spot_report.py "$DIR/hands_10k.jsonl"
} | tee "$DIR/summary.txt"
step "done — $DIR/summary.txt"
