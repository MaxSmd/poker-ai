#!/usr/bin/env python3
"""Post-mortem for a Slumbot match logged with `play slumbot --log-hands`.

Reads data/slumbot_hands.jsonl (one JSON object per hand) and breaks the
result down along the axes that tell you *where* the money goes, so a raw
bb/100 becomes a diagnosis:

  * by position (are we bleeding as SB, as BB, or both?)
  * by the street the hand reached (preflop folds vs. deep pots)
  * by pot size (is variance/loss concentrated in big all-in pots?)
  * showdown vs. non-showdown (are we losing at showdown = bad hand
    selection, or before = getting bluffed / folding too much?)
  * aggression: how often the hand went to each street, our fold-to-bet feel

All figures are bb/100 (net bb per 100 hands) with a 95% CI, plus the share of
total loss each bucket carries.  Usage:

    python scripts/analyze_slumbot.py [data/slumbot_hands.jsonl]
"""
import json
import math
import sys
from collections import defaultdict

BIG_BLIND = 100


def load(path):
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"(skipped {bad} malformed line(s))")
    return rows


def bb100(vals):
    """(mean*100, ci95*100) over a list of per-hand bb winnings."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mean = sum(vals) / n
    var = max(sum((v - mean) ** 2 for v in vals) / n, 0.0)
    ci = 1.96 * math.sqrt(var / n) * 100 if n > 1 else 0.0
    return mean * 100, ci


def line(label, vals, total_net):
    n = len(vals)
    if n == 0:
        print(f"  {label:<22} {'—':>8}  (0 hands)")
        return
    net = sum(vals)
    m, ci = bb100(vals)
    share = 100 * net / total_net if total_net != 0 else 0.0
    print(
        f"  {label:<22} {m:>8.1f} ± {ci:>5.1f} bb/100   "
        f"n={n:>5} ({100*n/TOTAL:>4.1f}%)   net {net:>8.1f} bb ({share:>5.1f}% of total)"
    )


def bucketize(rows, keyfn):
    b = defaultdict(list)
    for r in rows:
        b[keyfn(r)].append(r["winnings"] / BIG_BLIND)
    return b


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/slumbot_hands.jsonl"
    rows = load(path)
    if not rows:
        print(f"no hands in {path}")
        return

    global TOTAL
    TOTAL = len(rows)
    allv = [r["winnings"] / BIG_BLIND for r in rows]
    total_net = sum(allv)
    m, ci = bb100(allv)
    print(f"\n=== {path}: {TOTAL} hands ===")
    print(f"Overall: {m:.1f} ± {ci:.1f} bb/100   net {total_net:.1f} bb\n")

    # client_pos: 0 = big blind, 1 = small blind (Slumbot convention).
    print("By position:")
    pos = bucketize(rows, lambda r: "SB (button)" if r["pos"] == 1 else "BB")
    for k in ("SB (button)", "BB"):
        line(k, pos.get(k, []), total_net)

    print("\nBy street reached (0=preflop … 3=river):")
    st = bucketize(rows, lambda r: r.get("reached_street", 0))
    names = {0: "preflop-only", 1: "saw flop", 2: "saw turn", 3: "saw river"}
    for k in sorted(st):
        line(names.get(k, str(k)), st[k], total_net)

    # Showdown = the hand ended with no fold (all checks/calls to a decision).
    # Slumbot returns its hole cards every hand, so bot_hole is NOT the signal.
    print("\nShowdown vs not (showdown = no fold in the action):")
    sd = bucketize(rows, lambda r: "no showdown" if "f" in r.get("action", "") else "showdown")
    for k in ("showdown", "no showdown"):
        line(k, sd.get(k, []), total_net)

    # Won/lost split isolates whether losses are a few big pots or a steady drip.
    print("\nBy outcome size (|winnings| in bb):")
    def size_bucket(r):
        w = abs(r["winnings"] / BIG_BLIND)
        if w == 0:
            return "0 (chop/checkfold)"
        for hi, lab in [(5, "1  small (<5)"), (25, "2  medium (5-25)"),
                        (75, "3  large (25-75)"), (1e9, "4  stack+ (>75)")]:
            if w < hi:
                return lab
        return "?"
    for k, v in sorted(bucketize(rows, size_bucket).items()):
        line(k, v, total_net)

    # The single most useful cross-tab: position x street, where leaks localize.
    print("\nPosition x street (net bb, share of total):")
    for p, plab in ((1, "SB"), (0, "BB")):
        for s in range(4):
            v = [r["winnings"] / BIG_BLIND for r in rows
                 if r["pos"] == p and r.get("reached_street", 0) == s]
            if v:
                line(f"{plab} {names[s]}", v, total_net)

    biggest = sorted(rows, key=lambda r: r["winnings"])[:5]
    print("\n5 biggest losses:")
    for r in biggest:
        print(f"  {r['winnings']/BIG_BLIND:>7.1f} bb  pos={r['pos']}  "
              f"hole={r.get('hole')}  bot={r.get('bot_hole')}  "
              f"board={r.get('board')}  {r.get('action','')}")


if __name__ == "__main__":
    main()
