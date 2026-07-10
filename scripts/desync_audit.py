#!/usr/bin/env python3
"""Split a Slumbot match log by whether the abstraction desynced.

The blueprint's betting tree allows at most `--cap` bet-increases per street
(`BlueprintHoldem::next_raises` counts *any* action that raises `current_bet`,
so the opening postflop bet is raise 1).  When the opponent makes one more, the
abstraction has no node for it: the event cannot be translated, the tracker
stops advancing, and every later decision in that hand is taken by the fallback
rather than by the blueprint.

That makes desync hands the *only* population any change to the fallback can
touch.  Hands without a desync are a control: two runs of the same blueprint
must agree there up to noise.  If a run-to-run difference lives in the control,
it is variance, not the fallback.

    python scripts/desync_audit.py hands_a.jsonl [hands_b.jsonl ...]
"""
import json
import math
import sys

BIG_BLIND = 100
STREETS = ["preflop", "flop", "turn", "river"]


def replay(action, cap=3):
    """Events as (street, pos, token, to_amount, raises_after, is_over_cap).

    `pos` is Slumbot's: 0 = big blind, 1 = small blind.  The small blind acts
    first preflop, the big blind first on every later street.
    """
    events = []
    for street, s in enumerate(action.split("/")):
        cur = BIG_BLIND if street == 0 else 0
        pos = 1 if street == 0 else 0
        raises = 0
        i = 0
        while i < len(s):
            if s[i] == "b":
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                to = int(s[i + 1 : j])
                over = False
                if to > cur:
                    raises += 1
                    over = raises > cap
                    cur = to
                events.append((street, pos, "b", to, raises, over))
                i = j
            else:
                events.append((street, pos, s[i], None, raises, False))
                i += 1
            pos ^= 1
    return events


def desync(row, cap=3):
    """(street, our_response) of the first opponent action past the cap, or None.

    `our_response` is the token we played facing it ('f', 'c', 'b', or None if
    the hand ended before we acted again).
    """
    us = row["pos"]
    events = replay(row.get("action", ""), cap)
    for k, (street, pos, _tok, _to, _r, over) in enumerate(events):
        if not over or pos == us:
            continue
        response = next((e[2] for e in events[k + 1 :] if e[1] == us), None)
        return street, response
    return None


def stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(vals) / n
    var = max(sum((v - mean) ** 2 for v in vals) / n, 0.0)
    ci = 1.96 * math.sqrt(var / n) * 100 if n > 1 else 0.0
    return mean * 100, ci, sum(vals)


def line(label, vals, total):
    m, ci, net = stats(vals)
    if not vals:
        print(f"  {label:<34} {'—':>8}   (0 hands)")
        return
    print(
        f"  {label:<34} {m:>8.1f} ± {ci:>6.1f} bb/100   "
        f"n={len(vals):>5} ({100*len(vals)/total:>4.1f}%)   net {net:>8.1f} bb"
    )


def audit(path, cap):
    rows, bad = [], 0
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"(skipped {bad} malformed line(s))")

    total = len(rows)
    clean, hit = [], []
    by_street = {s: [] for s in range(4)}
    by_response = {}
    for r in rows:
        bb = r["winnings"] / BIG_BLIND
        d = desync(r, cap)
        if d is None:
            clean.append(bb)
            continue
        hit.append(bb)
        street, response = d
        by_street[street].append(bb)
        by_response.setdefault(response or "hand ended", []).append(bb)

    print(f"\n=== {path}: {total} hands (cap {cap}) ===")
    line("OVERALL", clean + hit, total)
    print()
    line("no desync (control)", clean, total)
    line("desync (fallback decided)", hit, total)

    if hit:
        print("\n  desync hands, by street of the over-cap raise:")
        for s in range(4):
            if by_street[s]:
                line(f"    {STREETS[s]}", by_street[s], total)
        print("\n  desync hands, by our response to it:")
        for k in sorted(by_response):
            line(f"    we played '{k}'", by_response[k], total)


def self_check():
    # Preflop 5-bet: b250(r1) b750(r2) b1657(r3) b6750(r4 -> over cap).
    ev = replay("b250b750b1657b6750c", cap=3)
    over = [e for e in ev if e[5]]
    assert len(over) == 1 and over[0][:2] == (0, 0) and over[0][3] == 6750, over
    # Flop raise war from the AA loss: b947(r1) b2900(r2) b5581(r3) b19300(r4).
    ev = replay("b200b700c/b947b2900b5581b19300c//", cap=3)
    over = [e for e in ev if e[5]]
    assert len(over) == 1 and over[0][0] == 1 and over[0][3] == 19300, over
    # A hand that never exceeds the cap has no trigger.
    assert not any(e[5] for e in replay("b200c/kk/kk/kk", cap=3))
    # The opening postflop bet is raise 1, so three postflop bets sit at the cap.
    assert not any(e[5] for e in replay("b200c/b100b300b900c", cap=3))


if __name__ == "__main__":
    self_check()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--cap=")), 3)
    if not args:
        print(__doc__)
        sys.exit(2)
    for p in args:
        audit(p, cap)
