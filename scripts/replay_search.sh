#!/usr/bin/env bash
#
# One round for the depth-limited search (--search): does it fix the flop and
# turn spots, at a usable speed — and if so, does it change the result
# against Slumbot?
#
#   tmux new-session -d -s search '~/poker-ai/scripts/replay_search.sh; exec bash'
#
# Waits for a running replay_flop.sh to finish first (the box fits one
# blueprint at a time), then:
#
# 1. Replays four spots from the latest match log, 600 each (fixed sample),
#    with the logged configuration (`default`) and with --search:
#      flop_cbet    SB, BB checked to us        we bet 28%, Slumbot 58%
#      flop_defend  BB, facing a 40-80% bet     we fold 57%, Slumbot 49%
#      turn_lead    BB, first to act on the turn we bet 20%, Slumbot 33%
#      turn_defend  SB, facing a 40-80% lead    we fold 41%, Slumbot 35%
# 2. --search wins only if it passes ALL FOUR by the rule in
#    scripts/replay_spots.py (within 10 points of Slumbot on either side; top
#    pair or better folded at most 10% when facing a bet), every replay was
#    faithful, and a searched decision averages at most 5 s.
# 3. If it wins: a logged 10k-hand Slumbot match with --search, and that
#    log's spot report, appended to summary.txt.
#
# Roughly 6 h of replays, plus ~25 h for the match.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=${1:-data/hands_10k_turnbp.jsonl}
BIN=./target/release/play
OUT=${OUT:-"data/replay_search_$(date +%Y%m%d_%H%M)"}
ITERS=${ITERS:-2000}

# The flop round is finished once its summary says how it ended.  (Not
# pgrep: a tmux server keeps the command line it was started with, so a
# process match can wait forever.)  WAIT=0 skips this.
flop_done() {
    [ -z "$(ls -d data/replay_flop_* 2>/dev/null)" ] && return 0
    grep -lqE "No arm passed|^Slumbot, 10k hands with" data/replay_flop_*/summary.txt 2>/dev/null
}
if [ "${WAIT:-1}" != 0 ]; then
    until flop_done; do
        echo "$(date +%H:%M) waiting for replay_flop.sh to finish ..."
        sleep 600
    done
fi

[ -x "$BIN" ] || { echo "missing $BIN — run: cargo build --release"; exit 1; }
mkdir -p "$OUT" || exit 1
sel() { python3 scripts/replay_spots.py select "$LOG" --max=600 --out="$OUT/$1.jsonl" "${@:2}" || exit 1; }
sel flop_cbet --street=flop --seat=SB --spot=unopened
sel flop_defend --street=flop --seat=BB --spot="vs bet" --size=40-80%
sel turn_lead --street=turn --seat=BB --spot=unopened
sel turn_defend --street=turn --seat=SB --spot="vs bet" --size=40-80%
SPOTS="flop_cbet flop_defend turn_lead turn_defend"

for name in default search; do
    flags="--no-search"
    [ "$name" = search ] && flags="--search --search-iters=$ITERS"
    for spot in $SPOTS; do
        printf '=== %s  START  %-8s %-12s %s\n' "$(date +%H:%M:%S)" "$name" "$spot" "$flags"
        # shellcheck disable=SC2086  # flags is a word list
        nice -n 19 "$BIN" replay --spots="$OUT/$spot.jsonl" --out="$OUT/${spot}_$name.jsonl" --cap=2 \
            $flags > "$OUT/${spot}_$name.log" 2>&1
        printf '=== %s  DONE   %-8s %-12s exit %d\n' "$(date +%H:%M:%S)" "$name" "$spot" "$?"
        grep -E "^Done:" "$OUT/${spot}_$name.log" | sed 's/^/      /'
    done
done

: > "$OUT/summary.txt"
WIN=yes
for spot in $SPOTS; do
    python3 scripts/replay_spots.py summarize "$OUT/$spot.jsonl" \
        default="$OUT/${spot}_default.jsonl" search="$OUT/${spot}_search.jsonl" > "$OUT/summary_$spot.txt"
    cat "$OUT/summary_$spot.txt" >> "$OUT/summary.txt"
    grep -q "replay faithful" "$OUT/summary_$spot.txt" || WIN=no
    grep -qE "^  search +FIXES the leak" "$OUT/summary_$spot.txt" || WIN=no
done
SECS=$(python3 -c "
import json, sys
s = [json.loads(l)['secs'] for spot in sys.argv[1:] for l in open(f'$OUT/{spot}_search.jsonl') if l.strip()]
print(f'{sum(s) / len(s):.2f}' if s else 'nan')" $SPOTS)
python3 -c "import sys; sys.exit(0 if float('$SECS') <= 5.0 else 1)" || WIN=no

{
    echo
    echo "================================ RESULT ================================"
    echo "search: ${SECS}s per searched decision (limit 5 s)"
    if [ "$WIN" = yes ]; then
        echo "search passes every spot — playing 10k hands with it."
    else
        echo "search does not pass every spot (or was too slow / unfaithful) — no match played."
    fi
} | tee -a "$OUT/summary.txt"
[ "$WIN" = yes ] || exit 0

printf '=== %s  START  slumbot 10000 --search\n' "$(date +%H:%M:%S)"
nice -n 19 "$BIN" slumbot 10000 --cap=2 --search --search-iters="$ITERS" \
    --log-hands="$OUT/hands_10k_search.jsonl" > "$OUT/slumbot_10k_search.log" 2>&1
printf '=== %s  DONE   slumbot  exit %d\n' "$(date +%H:%M:%S)" "$?"
{
    echo
    echo "Slumbot, 10k hands with --search (baseline: luck-adjusted -19.6 ± 19.9 bb/100):"
    grep -E "^Done:|^Abstraction tracking" "$OUT/slumbot_10k_search.log"
    python3 scripts/spot_report.py "$OUT/hands_10k_search.jsonl"
} | tee -a "$OUT/summary.txt"
