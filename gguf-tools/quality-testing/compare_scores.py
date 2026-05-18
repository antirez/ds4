#!/usr/bin/env python3
"""Compare two local model scores on official continuations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = {}
        for row in csv.DictReader(fp, delimiter="\t"):
            rows[row["id"]] = {
                "target_tokens": int(row["target_tokens"]),
                "nll": float(row["nll"]),
                "avg_nll": float(row["avg_nll"]),
                "first_match": int(row["first_match"]),
                "greedy_lcp": int(row["greedy_lcp"]),
            }
        return rows


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} OLD.tsv NEW.tsv", file=sys.stderr)
        return 2

    old = load(Path(sys.argv[1]))
    new = load(Path(sys.argv[2]))
    ids = sorted(set(old) & set(new))
    if not ids:
        raise SystemExit("no common cases")

    old_nll = new_nll = 0.0
    old_first = new_first = 0
    old_lcp = new_lcp = 0
    tokens = 0
    new_case_wins = old_case_wins = ties = 0
    deltas = []

    for case_id in ids:
        o = old[case_id]
        n = new[case_id]
        if o["target_tokens"] != n["target_tokens"]:
            raise SystemExit(f"token-count mismatch for {case_id}")
        t = int(o["target_tokens"])
        tokens += t
        old_nll += o["nll"]
        new_nll += n["nll"]
        old_first += int(o["first_match"])
        new_first += int(n["first_match"])
        old_lcp += int(o["greedy_lcp"])
        new_lcp += int(n["greedy_lcp"])
        delta = n["nll"] - o["nll"]
        deltas.append((delta, case_id, t, o["avg_nll"], n["avg_nll"]))
        if delta < -1e-9:
            new_case_wins += 1
        elif delta > 1e-9:
            old_case_wins += 1
        else:
            ties += 1

    avg_old = old_nll / tokens
    avg_new = new_nll / tokens
    print(f"cases\t{len(ids)}")
    print(f"tokens\t{tokens}")
    print(f"old_avg_nll\t{avg_old:.9f}")
    print(f"new_avg_nll\t{avg_new:.9f}")
    print(f"delta_new_minus_old\t{avg_new - avg_old:.9f}")
    print(f"relative_nll_change\t{(avg_new / avg_old - 1.0) * 100.0:.3f}%")
    print(f"case_wins_new_old_ties\t{new_case_wins}\t{old_case_wins}\t{ties}")
    print(f"first_token_matches_old_new\t{old_first}\t{new_first}")
    print(f"avg_greedy_lcp_old_new\t{old_lcp / len(ids):.3f}\t{new_lcp / len(ids):.3f}")

    print("\nnew best cases:")
    for delta, case_id, t, old_avg, new_avg in sorted(deltas)[:8]:
        print(f"{case_id}\tdelta_nll={delta:.6f}\ttokens={t}\told={old_avg:.6f}\tnew={new_avg:.6f}")

    print("\nold best cases:")
    for delta, case_id, t, old_avg, new_avg in sorted(deltas, reverse=True)[:8]:
        print(f"{case_id}\tdelta_nll={delta:.6f}\ttokens={t}\told={old_avg:.6f}\tnew={new_avg:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
