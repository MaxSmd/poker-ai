#!/usr/bin/env bash
#
# One round: does any setting fix our flop play — and if one does, does it
# change the result against Slumbot?
#
#   tmux new-session -d -s replay '~/poker-ai/scripts/replay_flop.sh; exec bash'
#
# 1. Replays (~3 h): the arms below on two flop spots.
# 2. Picks the first arm that passes BOTH spots by the fixed rule, if any.
# 3. Plays a logged 10k-hand Slumbot match with it (~17 h) and adds the match
#    result and the spot report of its log to summary.txt.  With no winner,
#    the match is skipped and summary.txt says so.
#
# The flop is still re-solved with the K-continuation leaf that made the turn
# over-fold (scripts/replay_turn_folds.sh).  Two flop spots from the log,
# 600 each (a fixed random sample), replayed under each setting:
#
#   cbet     we are the SB, the BB checked to us: we bet 28%, Slumbot 58%
#   defend   we are the BB, the SB bet 40-80% pot: we fold 57%, Slumbot 49%
#
# Arms:
#   default         the configuration the log was played with; must reproduce
#                   the logged rates, or the replay is unfaithful
#   blueprint_flop  no flop re-solve (what fixed the turn)
#   leaf_checkdown  a plain check-down leaf
#   leaf_mild       continuations 0 and 0.75 only, dropping the large pot scales
#
# Verdicts by the rule in scripts/replay_spots.py: within 10 points of
# Slumbot on either side, and, when facing a bet, top pair or better folded
# at most 10%.  A setting ships only if it passes BOTH spots — fixing the
# c-bet by breaking the defence is not a fix.  One process at a time
# (~16 GB each).  About 20 hours with the match, 3 without.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG=${1:-data/hands_10k_turnbp.jsonl}
BIN=./target/release/play
OUT=${OUT:-"data/replay_flop_$(date +%Y%m%d_%H%M)"}

[ -x "$BIN" ] || { echo "missing $BIN — run: cargo build --release"; exit 1; }
mkdir -p "$OUT" || exit 1
python3 scripts/replay_spots.py select "$LOG" --street=flop --seat=SB --spot=unopened --max=600 \
    --out="$OUT/cbet.jsonl" || exit 1
python3 scripts/replay_spots.py select "$LOG" --street=flop --seat=BB --spot="vs bet" --size=40-80% --max=600 \
    --out="$OUT/defend.jsonl" || exit 1

NAMES="default blueprint_flop leaf_checkdown leaf_mild"
flags() {
    case $1 in
        default) echo --no-search ;;
        blueprint_flop) echo --no-search --no-resolve-flop ;;
        leaf_checkdown) echo --no-search --continuations=0.0 ;;
        leaf_mild) echo --no-search --continuations=0.0,0.75 ;;
    esac
}

for name in $NAMES; do
    for spot in cbet defend; do
        printf '=== %s  START  %-15s %-7s %s\n' "$(date +%H:%M:%S)" "$name" "$spot" "$(flags "$name")"
        # shellcheck disable=SC2046  # flags expands to a word list
        nice -n 19 "$BIN" replay --spots="$OUT/$spot.jsonl" --out="$OUT/${spot}_$name.jsonl" --cap=2 \
            $(flags "$name") > "$OUT/${spot}_$name.log" 2>&1
        printf '=== %s  DONE   %-15s %-7s exit %d\n' "$(date +%H:%M:%S)" "$name" "$spot" "$?"
        grep -E "^Done:" "$OUT/${spot}_$name.log" | sed 's/^/      /'
    done
done

for spot in cbet defend; do
    args=""
    for name in $NAMES; do
        args="$args $name=$OUT/${spot}_$name.jsonl"
    done
    # shellcheck disable=SC2086  # a word list by construction
    python3 scripts/replay_spots.py summarize "$OUT/$spot.jsonl" $args > "$OUT/summary_$spot.txt"
done
cat "$OUT/summary_cbet.txt" "$OUT/summary_defend.txt" > "$OUT/summary.txt"

# The winner: the first non-default arm passing both spots.  Both summaries
# must also have found the replay faithful, or no verdict counts.
passes() { grep -qE "^  $1 +FIXES the leak" "$OUT/summary_$2.txt"; }
WINNER=""
if grep -q "replay faithful" "$OUT/summary_cbet.txt" && grep -q "replay faithful" "$OUT/summary_defend.txt"; then
    for name in $NAMES; do
        [ "$name" = default ] && continue
        if passes "$name" cbet && passes "$name" defend; then WINNER=$name; break; fi
    done
fi

{
    echo
    echo "================================ RESULT ================================"
    if [ -z "$WINNER" ]; then
        echo "No arm passed both spots (or the replay was not faithful) — no match played."
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
    echo "Slumbot, 10k hands with $WINNER (baseline: luck-adjusted -19.6 ± 19.9 bb/100):"
    grep -E "^Done:|^Abstraction tracking" "$OUT/slumbot_10k_$WINNER.log"
    python3 scripts/spot_report.py "$OUT/hands_10k_$WINNER.jsonl"
} | tee -a "$OUT/summary.txt"
