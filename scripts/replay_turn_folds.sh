#!/usr/bin/env bash
#
# One round: does any setting stop us over-folding to turn leads?
#
#   tmux new-session -d -s replay '~/poker-ai/scripts/replay_turn_folds.sh; exec bash'
#
# Replays the ~800 logged spots where Slumbot leads the turn into us (we fold
# 82%, Slumbot 43%) under each setting below, then prints one table and the
# verdict of the rule in scripts/replay_spots.py.  Strictly one process at a
# time: each loads the ~16 GB blueprint.  About 3.5 hours.
#
# Arms, cheapest-to-interpret first:
#   default         the configuration the log was played with (turn re-solve
#                   on); must reproduce the logged 82%, or the replay is unfaithful
#   leaf_checkdown  a plain check-down leaf instead of K continuations
#   floor_1         Slumbot's postflop actions do not narrow its range at all
#   floor_0.3       the same, partially — the shippable form of floor_1
#   blueprint_turn  no turn re-solve: does the blueprint itself over-fold?
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=${1:-data/hands_10k_postfix.jsonl}
BIN=./target/release/play
OUT=${OUT:-"data/replay_$(date +%Y%m%d_%H%M)"}

[ -x "$BIN" ] || { echo "missing $BIN — run: cargo build --release"; exit 1; }
mkdir -p "$OUT" || exit 1
python3 scripts/replay_spots.py select "$LOG" --out="$OUT/spots.jsonl" || exit 1

ARMS=""
run() {
    name=$1
    shift
    ARMS="$ARMS $name=$OUT/$name.jsonl"
    printf '=== %s  START  %-15s  %s\n' "$(date +%H:%M:%S)" "$name" "$*"
    nice -n 19 "$BIN" replay --spots="$OUT/spots.jsonl" --out="$OUT/$name.jsonl" --cap=2 "$@" > "$OUT/$name.log" 2>&1
    printf '=== %s  DONE   %-15s  exit %d\n' "$(date +%H:%M:%S)" "$name" "$?"
    grep -E "^Done:" "$OUT/$name.log" | sed 's/^/      /'
}

run default --no-search --resolve-turn
run leaf_checkdown --no-search --resolve-turn --continuations=0.0
run floor_1 --no-search --resolve-turn --opp-range-floor=1.0
run floor_0.3 --no-search --resolve-turn --opp-range-floor=0.3
run blueprint_turn --no-search --no-resolve-turn

# shellcheck disable=SC2086  # ARMS is a word list by construction
python3 scripts/replay_spots.py summarize "$OUT/spots.jsonl" $ARMS | tee "$OUT/summary.txt"
