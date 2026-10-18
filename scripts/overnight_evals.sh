#!/usr/bin/env bash
#
# Overnight evaluation sweep — every measurement that produces a *result*
# number, run back to back, unattended.
#
#   tmux new-session -d -s evals '~/poker-ai/scripts/overnight_evals.sh; exec bash'
#
# ## Memory
#
# Strictly ONE process at a time.  The blueprint is ~16.3 GB resident
# (593.8 M entries x 27.4 B) and this box is shared, so arms run sequentially
# rather than in parallel: peak is ~17 GB, well inside a 32 GB budget.  Do not
# "speed this up" by backgrounding the arms — four of them is 65 GB.
#
# ## Ordering
#
# Cheapest and most important first, so a night that runs short still leaves
# the headline result on disk.  Every arm writes its own log; nothing depends
# on a previous arm succeeding.
#
# ## Pairing
#
# All LBR arms use the same --lbr-seed, and `run_lbr` keeps dealing on its own
# RNG stream, so every arm sees the IDENTICAL 3000 deals regardless of how the
# agent played.  The ablations are therefore paired comparisons, which is worth
# far more than extra hands.
#
# ## Selecting arms
#
#   ARMS="lbr_river_only lbr_no_continual" OUT=data/evals_20260822_0030 \
#       scripts/overnight_evals.sh
#
# ARMS limits the run to the named arms (default: all).  Pointing OUT at an
# earlier sweep adds the new arms beside it, which keeps them paired with the
# old ones as long as HANDS and SEED match.
set -u

cd "$(dirname "$0")/.." || exit 1
BIN=./target/release/play
HANDS=${HANDS:-3000}
SEED=${SEED:-1}
OUT=${OUT:-"data/evals_$(date +%Y%m%d_%H%M)"}
ARMS=${ARMS:-}

[ -x "$BIN" ] || { echo "missing $BIN — run: cargo build --release"; exit 1; }
mkdir -p "$OUT" || exit 1
echo "Writing logs to $OUT ; $HANDS hands per LBR arm, seed $SEED"
echo

run() {
    name=$1
    shift
    if [ -n "$ARMS" ] && [[ " $ARMS " != *" $name "* ]]; then
        return
    fi
    printf '=== %s  START  %-22s  %s\n' "$(date +%H:%M:%S)" "$name" "$*"
    nice -n 19 "$BIN" "$@" > "$OUT/$name.log" 2>&1
    rc=$?
    printf '=== %s  DONE   %-22s  exit %d\n' "$(date +%H:%M:%S)" "$name" "$rc"
    grep -E "lower bound|LBR LOST|luck-adjusted|Abstract-game exploitability|Blueprint lookups|Abstraction tracking" "$OUT/$name.log" \
        | sed 's/^/      /'
    echo
}

# 1. Baseline: how exploitable is the blueprint when it plays alone?  No
#    re-solving, so this is minutes, and it is the reference every other LBR
#    arm is compared against.
run lbr_blueprint_only lbr "$HANDS" --cap=2 --no-resolve --lbr-seed="$SEED"

# 2. The headline: the shipped agent (river and flop re-solved, turn from the
#    blueprint since 2026-10).
#    (1) vs (2) is the ablation that justifies the resolver.
run lbr_resolve_all   lbr "$HANDS" --cap=2 --lbr-seed="$SEED"

# 3. A tighter blueprint exploitability figure than the 16-flop estimate:
#    4x the flops, and board-samples=3 to match the historical runs.
run expl_flops64      expl --cap=2 --flops=64 --board-samples=3 --seed="$SEED"

# 4. Purification was tuned when only the river was re-solved.  With three
#    streets re-solved, small probabilities are more likely to be real mixing
#    than abstraction noise, so 0.1 may now be discarding strategy.
run lbr_purify0       lbr "$HANDS" --cap=2 --purify=0.0 --lbr-seed="$SEED"

# 5. Isolate the streets: river+turn only.  Against (2) this prices the flop
#    resolve specifically — the most expensive one at 2.1 s/decision.
run lbr_no_flop       lbr "$HANDS" --cap=2 --no-resolve-flop --lbr-seed="$SEED"

# 6-8. Why does re-solving lose ground to LBR?  Each of (1)->(5) re-solved
#    another street and the agent won less; these isolate the suspects.
#    River only: between (1) and (5), so it prices the river resolve alone.
run lbr_river_only    lbr "$HANDS" --cap=2 --no-resolve-turn --no-resolve-flop --lbr-seed="$SEED"

#    Continual off: every resolve starts unconstrained.  If the gadget or the
#    values it carries are at fault, this closes the gap to (1).
run lbr_no_continual  lbr "$HANDS" --cap=2 --no-continual --lbr-seed="$SEED"

#    Resolve cap 2: the resolves default to 3 raises a street against a cap-2
#    blueprint, so they solve a different betting game than the one the
#    ranges entering them came from.
run lbr_river_cap2    lbr "$HANDS" --cap=2 --river-cap=2 --lbr-seed="$SEED"

echo "================================ SUMMARY ================================"
for f in "$OUT"/*.log; do
    printf '%-24s ' "$(basename "$f" .log)"
    grep -hE "LBR luck-adjusted|lower bound|Abstract-game exploitability" "$f" | tail -1 \
        || echo "(no result — check the log)"
done
echo "========================================================================="
echo "Logs: $OUT"
