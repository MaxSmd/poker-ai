#!/usr/bin/env bash
#
# Retrain the blueprint with deeper preflop and flop betting, then play it.
#
#   tmux new-session -d -s retrain '~/poker-ai/scripts/retrain.sh; exec bash'
#
# Why: with the flop and turn played from the blueprint (only the river is
# re-solved), every leak left in the spot report is the blueprint's, and the
# biggest structural one is betting depth.  The current blueprint allows two
# raises a street, so preflop it can 3-bet but never 4-bet except all-in (we
# 4-bet 0% where Slumbot 4-bets 7%), and the flop has one re-raise.  Our best
# result against Slumbot (+7.9 bb/100, July) came from a cap-3 blueprint.
#
# Caps 3,3,2,1 (preflop, flop, turn, river) restore that depth where it is
# played from the blueprint; the river gets one raise since it is re-solved
# exactly.  Buckets 8000/8000/800 keep the info-set count at the current
# blueprint's (595M vs 599M), so 5B iterations train it about as densely,
# in 19.5 GB (memory_estimate); the strategy it writes is 3.2 GB.
#
# Everything goes to its own directory; the current blueprint in data/ is
# untouched, and play uses it unless given --data.  Each step is skipped if
# its output exists, and training resumes from its checkpoint, so re-running
# after an interruption continues where it stopped.
#
#   1. cluster  8000/8000/800 buckets, OCHS river         ~1-2 h
#   2. train    5B iterations, 32 threads                 ~1 day
#   3. match    10k hands against Slumbot, logged         ~15 h
#
# Result: $DIR/summary.txt (match result + spot report).
set -u

cd "$(dirname "$0")/.." || exit 1
DIR=${DIR:-data/bp_3321}
CAPS=${CAPS:-3,3,2,1}
BUCKETS=${BUCKETS:-"8000 8000 800"}
ITERS=${ITERS:-5000000000}
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
    step "train $ITERS iterations, caps $CAPS"
    nice -n 19 "$BIN/train" blueprint "$ITERS" 200 1 --cap="$CAPS" --soa --atomic="$THREADS" \
        --chunk=50000000 --resume --data="$DIR" >> "$DIR/train.log" 2>&1 \
        || { step "train FAILED — see $DIR/train.log"; exit 1; }
    # The checkpoint is as large as the table and only needed to resume.
    rm -f "$DIR/blueprint_holdem_soa.ckpt"
fi

if [ ! -f "$DIR/hands_10k.jsonl" ] || [ "$(wc -l < "$DIR/hands_10k.jsonl")" -lt 10000 ]; then
    step "slumbot 10000 hands"
    nice -n 19 "$BIN/play" slumbot 10000 --cap="$CAPS" --data="$DIR" \
        --log-hands="$DIR/hands_10k.jsonl" > "$DIR/slumbot_10k.log" 2>&1
fi

{
    echo "Blueprint caps $CAPS, buckets $BUCKETS, $ITERS iterations ($DIR)"
    grep -E "info sets|Saved" "$DIR/train.log" | tail -2
    echo
    echo "Slumbot, 10k hands (previous blueprint, same play config: luck-adjusted -13.3 ± 24.8 bb/100):"
    grep -E "^Done:|^Blueprint lookups|^Abstraction tracking" "$DIR/slumbot_10k.log"
    python3 scripts/spot_report.py "$DIR/hands_10k.jsonl"
} | tee "$DIR/summary.txt"
step "done — $DIR/summary.txt"
