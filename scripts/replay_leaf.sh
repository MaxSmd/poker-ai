#!/usr/bin/env bash
#
# One round: is the leaf model what makes the search over-aggressive — and
# if a variant fixes it, does it change the result against Slumbot?
#
#   tmux new-session -d -s leaf '~/poker-ai/scripts/replay_leaf.sh; exec bash'
#
# replay_iters.sh showed more iterations do not fix it.  From 2k to 50k the
# search bluffs less and less, yet two leaks stay put: it check-raises the
# flop 41-50% (Slumbot 7%) and folds to a turn lead 53% (Slumbot 37%; the
# blueprint 39-41%).  A leak that survives convergence is in the game the
# search solves, and the suspect is the leaf: both players choose among four
# continuations, so we can "plan" to keep raising (aggression now looks
# better) and the opponent can pick per hand (our calls look worse).
#
# 1. Replays the same five spots, 600 each, at the logged setting (default)
#    and at 20000 iterations with:
#      opp_only    only the opponent chooses; we continue with the blueprint
#                  (Modicum's original formulation)
#      k1          nobody chooses: both continue with the blueprint
#      opp_bias2   opponent only, biased continuations leaning 2x, not 5x
# 2. Winner by scripts/replay_spots.py `best`, same rule as replay_iters.sh
#    (all five spots within 10 points of Slumbot, raise rate too when facing
#    a bet, top pair+ folded at most 10%, every default replay faithful, at
#    most 5 s a decision; smallest total gap wins).
# 3. If there is one: a logged 10k-hand Slumbot match with it.
#
# About 6.5 h of replays, plus roughly a day for the match.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=${1:-$(ls -t data/replay_search_*/hands_10k_search.jsonl 2>/dev/null | head -1)}
BIN=./target/release/play
OUT=${OUT:-"data/replay_leaf_$(date +%Y%m%d_%H%M)"}

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
ARMS="default opp_only k1 opp_bias2"
flags() {
    case $1 in
        default) echo --search --search-iters=2000 ;;
        opp_only) echo --search --search-iters=20000 --search-opp-only ;;
        k1) echo --search --search-iters=20000 --search-k=1 ;;
        opp_bias2) echo --search --search-iters=20000 --search-opp-only --search-bias=2 ;;
    esac
}

for name in $ARMS; do
    for spot in $SPOTS; do
        printf '=== %s  START  %-9s %-12s %s\n' "$(date +%H:%M:%S)" "$name" "$spot" "$(flags "$name")"
        # shellcheck disable=SC2046  # flags expands to a word list
        nice -n 19 "$BIN" replay --spots="$OUT/$spot.jsonl" --out="$OUT/${spot}_$name.jsonl" --cap=2 \
            $(flags "$name") > "$OUT/${spot}_$name.log" 2>&1
        printf '=== %s  DONE   %-9s %-12s exit %d\n' "$(date +%H:%M:%S)" "$name" "$spot" "$?"
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
