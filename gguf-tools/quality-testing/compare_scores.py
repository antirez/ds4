#!/usr/bin/env python3
"""Compare two local model scores on official continuations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def parse_cell(value: str) -> float | int:
    v = float(value)
    if v.is_integer():
        return int(v)
    return v


def load(path: Path) -> dict[str, dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as fp:
        rows = {}
        for row in csv.DictReader(fp, delimiter="\t"):
            parsed = {
                "target_tokens": int(row["target_tokens"]),
                "nll": float(row["nll"]),
                "avg_nll": float(row["avg_nll"]),
                "first_match": int(row["first_match"]),
                "greedy_lcp": int(row["greedy_lcp"]),
            }
            for key, value in row.items():
                if key is None or key == "id" or key in parsed or value is None or value == "":
                    continue
                parsed[key] = parse_cell(value)
            rows[row["id"]] = parsed
        return rows


def has_api(rows: dict[str, dict[str, float | int]], ids: list[str]) -> bool:
    return all(
        "api_target_tokens" in rows[case_id] and "api_top_items" in rows[case_id]
        for case_id in ids
    )


def sum_key(rows: dict[str, dict[str, float | int]], ids: list[str], key: str) -> float:
    return sum(float(rows[case_id].get(key, 0.0)) for case_id in ids)


def weighted_avg(
    rows: dict[str, dict[str, float | int]], ids: list[str], value_key: str, count_key: str
) -> float:
    count = sum_key(rows, ids, count_key)
    if count == 0:
        return 0.0
    total = sum(
        float(rows[case_id].get(value_key, 0.0)) * float(rows[case_id].get(count_key, 0.0))
        for case_id in ids
    )
    return total / count


def ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def print_api_summary(label: str, rows: dict[str, dict[str, float | int]], ids: list[str]) -> None:
    target_tokens = sum_key(rows, ids, "api_target_tokens")
    top_items = sum_key(rows, ids, "api_top_items")
    top_mapped = sum_key(rows, ids, "api_top_mapped")
    top1_count = sum_key(rows, ids, "api_top1_count")
    top1_match = sum_key(rows, ids, "api_top1_match")
    topn_ref = sum_key(rows, ids, "api_topn_ref")
    topn_hit = sum_key(rows, ids, "api_topn_hit")
    top_logprob_count = sum_key(rows, ids, "api_top_logprob_count")
    pair_total = sum_key(rows, ids, "api_pair_total")
    pair_agree = sum_key(rows, ids, "api_pair_agree")

    print(f"{label}_api_target_tokens\t{int(target_tokens)}")
    print(f"{label}_api_target_mae\t{weighted_avg(rows, ids, 'api_target_mae', 'api_target_tokens'):.9f}")
    print(
        f"{label}_api_target_mean_delta\t"
        f"{weighted_avg(rows, ids, 'api_target_mean_delta', 'api_target_tokens'):.9f}"
    )
    print(f"{label}_api_top_coverage\t{ratio(top_mapped, top_items):.9f}")
    print(f"{label}_api_top1_rate\t{ratio(top1_match, top1_count):.9f}")
    print(f"{label}_api_topn_recall\t{ratio(topn_hit, topn_ref):.9f}")
    print(f"{label}_api_top_mae\t{weighted_avg(rows, ids, 'api_top_mae', 'api_top_logprob_count'):.9f}")
    print(
        f"{label}_api_top_mean_delta\t"
        f"{weighted_avg(rows, ids, 'api_top_mean_delta', 'api_top_logprob_count'):.9f}"
    )
    print(f"{label}_api_pair_rate\t{ratio(pair_agree, pair_total):.9f}")


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
    if has_api(old, ids) and has_api(new, ids):
        print_api_summary("old", old, ids)
        print_api_summary("new", new, ids)

    print("\nnew best cases:")
    for delta, case_id, t, old_avg, new_avg in sorted(deltas)[:8]:
        print(f"{case_id}\tdelta_nll={delta:.6f}\ttokens={t}\told={old_avg:.6f}\tnew={new_avg:.6f}")

    print("\nold best cases:")
    for delta, case_id, t, old_avg, new_avg in sorted(deltas, reverse=True)[:8]:
        print(f"{case_id}\tdelta_nll={delta:.6f}\ttokens={t}\told={old_avg:.6f}\tnew={new_avg:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
