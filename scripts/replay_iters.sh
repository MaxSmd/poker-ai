#!/usr/bin/env bash
#
# One round: does the search stop over-betting with more iterations — and if
# so, does it change the result against Slumbot?
#
#   tmux new-session -d -s iters '~/poker-ai/scripts/replay_iters.sh; exec bash'
#
# The 10k match with --search at 2000 iterations fixed the four spots it was
# replayed on but turned aggressive elsewhere: the big blind leads the flop
# 53% (Slumbot 16%) and check-raises it 51% (Slumbot 7%).  The likely cause
# is under-training: at 2000 iterations most opponent hands in the search
# still play like the blueprint, which folds too much, so betting into them
# looks better than it is against Slumbot.  More iterations fix that, and the
# search ran at 0.12 s a decision against a 5 s budget.
#
# 1. Replays five spots from that match's log, 600 each, at the logged
#    setting (default = 2000 iterations) and at 20000 and 50000:
#      flop_lead    BB first to act on the flop     we bet 53%, Slumbot 16%
#      flop_cbet    SB, BB checked to us            we bet 56%, Slumbot 55%
#      flop_defend  BB facing a 40-80% bet          fold 29% / raise 51%, Slumbot 50% / 7%
#      turn_lead    BB first to act on the turn     we bet 39%, Slumbot 28%
#      turn_defend  SB facing a 40-80% lead         we fold 51%, Slumbot 37%
# 2. Winner, by scripts/replay_spots.py `best`: an arm passing ALL FIVE —
#    within 10 points of Slumbot; facing a bet also within 10 points of its
#    raise rate and top pair or better folded at most 10% — with every
#    default replay faithful and at most 5 s a decision; the smallest total
#    gap to Slumbot wins.
# 3. If there is one: a logged 10k-hand Slumbot match with it.
#
# About 7.5 h of replays, plus roughly a day for the match.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=${1:-$(ls -t data/replay_search_*/hands_10k_search.jsonl 2>/dev/null | head -1)}
BIN=./target/release/play
OUT=${OUT:-"data/replay_iters_$(date +%Y%m%d_%H%M)"}

[ -n "$LOG" ] && [ -f "$LOG" ] || { echo "no match log (pass one as the first argument)"; exit 1; }
[ -x "$BIN" ] || { echo "missing $BIN — run: cargo build --release"; exit 1; }
mkdir -p "$OUT" || exit 1
echo "Spots from $LOG"
sel() { python3 scripts/replay_spots.py select "$LOG" --max=600 --out="$OUT/$1.jsonl" "${@:2}" || exit 1; }
sel flop_lead --street=flop --seat=BB --spot=unopened
sel flop_cbet --street=flop --seat=SB --spot=unopened
sel flop_defend --street=flop --seat=BB --spot="vs bet" --size=40-80%
sel turn_lead --street=turn --seat=BB --spot=unopened
sel turn_defend --street=turn --seat=SB --spot="vs bet" --size=40-80%
SPOTS="flop_lead flop_cbet flop_defend turn_lead turn_defend"
ARMS="default it20k it50k"
flags() {
    case $1 in
        default) echo --search --search-iters=2000 ;;
        it20k) echo --search --search-iters=20000 ;;
        it50k) echo --search --search-iters=50000 ;;
    esac
}

for name in $ARMS; do
    for spot in $SPOTS; do
        printf '=== %s  START  %-8s %-12s %s\n' "$(date +%H:%M:%S)" "$name" "$spot" "$(flags "$name")"
        # shellcheck disable=SC2046  # flags expands to a word list
        nice -n 19 "$BIN" replay --spots="$OUT/$spot.jsonl" --out="$OUT/${spot}_$name.jsonl" --cap=2 \
            $(flags "$name") > "$OUT/${spot}_$name.log" 2>&1
        printf '=== %s  DONE   %-8s %-12s exit %d\n' "$(date +%H:%M:%S)" "$name" "$spot" "$?"
        grep -E "^Done:" "$OUT/${spot}_$name.log" | sed 's/^/      /'
    done
done

: > "$OUT/summary.txt"
for spot in $SPOTS; do
    args=""
    for name in $ARMS; do args="$args $name=$OUT/${spot}_$name.jsonl"; done
    # shellcheck disable=SC2086
    python3 scripts/replay_spots.py summarize "$OUT/$spot.jsonl" $args >> "$OUT/summary.txt"
done

# Too slow is out regardless of the frequencies.
FAST=""
for name in $ARMS; do
    secs=$(python3 -c "
import json, sys
s = [json.loads(l)['secs'] for spot in sys.argv[1:] for l in open(f'$OUT/{spot}_$name.jsonl') if l.strip()]
print(f'{sum(s) / len(s):.2f}' if s else 'nan')" $SPOTS)
    echo "$name: ${secs}s per decision" >> "$OUT/summary.txt"
    python3 -c "import sys; sys.exit(0 if float('$secs') <= 5.0 else 1)" && FAST="$FAST,$name"
done
WINNER=$(python3 scripts/replay_spots.py best "$OUT" --spots="$(echo $SPOTS | tr ' ' ,)" --arms="${FAST#,}")

{
    echo
    echo "================================ RESULT ================================"
    if [ -z "$WINNER" ]; then
        echo "No arm passed all five spots within 5 s a decision (or a replay was not faithful) — no match played."
    else
        echo "Winner: $WINNER  ($(flags "$WINNER"))"
    fi
} | tee -a "$OUT/summary.txt"
[ -n "$WINNER" ] || exit 0

printf '=== %s  START  slumbot 10000 with %s\n' "$(date +%H:%M:%S)" "$WINNER"
# shellcheck disable=SC2046
nice -n 19 "$BIN" slumbot 10000 --cap=2 $(flags "$WINNER") \
    --log-hands="$OUT/hands_10k_$WINNER.jsonl" > "$OUT/slumbot_10k_$WINNER.log" 2>&1
printf '=== %s  DONE   slumbot  exit %d\n' "$(date +%H:%M:%S)" "$?"
{
    echo
    echo "Slumbot, 10k hands with $WINNER (previous: luck-adjusted -13.0 ± 25.3 bb/100 at 2000 iterations):"
    grep -E "^Done:|^Abstraction tracking" "$OUT/slumbot_10k_$WINNER.log"
    python3 scripts/spot_report.py "$OUT/hands_10k_$WINNER.jsonl"
} | tee -a "$OUT/summary.txt"
