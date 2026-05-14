#!/usr/bin/env python3
"""Compare full-logit dumps produced by ./ds4 --dump-logits.

Example:
  ./ds4 -m q2.gguf --metal -mt off --dump-logits /tmp/q2-off.json \
      --nothink --prompt-file prompt.txt
  ./ds4 -m q2.gguf --metal -mt auto --dump-logits /tmp/q2-mt.json \
      --nothink --prompt-file prompt.txt
  ./ds4 -m q4.gguf --metal -mt off --dump-logits /tmp/q4-off.json \
      --nothink --prompt-file prompt.txt
  python3 speed-bench/compare_logit_drift.py /tmp/q2-off.json \
      /tmp/q2-mt.json /tmp/q4-off.json --labels q2_mt q4_off
"""

from __future__ import annotations

import argparse
import json
import math
from heapq import nlargest
from pathlib import Path
from typing import Any


def load_dump(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    logits_raw = data.get("logits")
    if not isinstance(logits_raw, list) or not logits_raw:
        raise SystemExit(f"{path}: missing non-empty logits array")
    logits = [float("nan") if v is None else float(v) for v in logits_raw]
    vocab = int(data.get("vocab", len(logits)))
    if vocab != len(logits):
        raise SystemExit(f"{path}: vocab={vocab} does not match logits={len(logits)}")
    data["logits"] = logits
    data["_path"] = str(path)
    return data


def dump_label(data: dict[str, Any]) -> str:
    model = Path(str(data.get("model", data.get("_path", "dump")))).name
    quant = data.get("quant_bits", "?")
    mt = data.get("mt", "?")
    quality = data.get("quality")
    suffix = f":quality={quality}" if isinstance(quality, bool) else ""
    return f"{model}:q{quant}:mt={mt}{suffix}"


def finite_indices(logits: list[float]) -> list[int]:
    return [i for i, v in enumerate(logits) if math.isfinite(v)]


def topk(logits: list[float], k: int) -> list[int]:
    # Match the C test's tie behavior: higher logit first, lower token id first.
    return nlargest(k, finite_indices(logits), key=lambda i: (logits[i], -i))


def overlap(a: list[int], b: list[int], k: int) -> int:
    return len(set(a[:k]) & set(b[:k]))


def rank_delta(ref_top: list[int], cand_top: list[int]) -> int:
    cand_rank = {token: i for i, token in enumerate(cand_top)}
    worst = 0
    for i, token in enumerate(ref_top):
        if token in cand_rank:
            worst = max(worst, abs(cand_rank[token] - i))
    return worst


def top_union_max_abs(
    ref: list[float],
    cand: list[float],
    ref_top: list[int],
    cand_top: list[int],
    k: int,
) -> float:
    ids = set(ref_top[:k]) | set(cand_top[:k])
    worst = 0.0
    for token in ids:
        if math.isfinite(ref[token]) and math.isfinite(cand[token]):
            worst = max(worst, abs(cand[token] - ref[token]))
    return worst


def compare(ref_dump: dict[str, Any], cand_dump: dict[str, Any], top_k: int) -> dict[str, Any]:
    ref = ref_dump["logits"]
    cand = cand_dump["logits"]
    if len(ref) != len(cand):
        raise SystemExit(
            f"vocab mismatch: {ref_dump['_path']} has {len(ref)}, "
            f"{cand_dump['_path']} has {len(cand)}"
        )

    ref_top = topk(ref, top_k)
    cand_top = topk(cand, top_k)
    sumsq = 0.0
    max_abs = 0.0
    nonfinite = 0
    largest: list[tuple[float, int, float, float]] = []
    for token, (rv, cv) in enumerate(zip(ref, cand)):
        if not math.isfinite(rv) or not math.isfinite(cv):
            nonfinite += 1
            continue
        delta = cv - rv
        abs_delta = abs(delta)
        sumsq += delta * delta
        max_abs = max(max_abs, abs_delta)
        if len(largest) < 5:
            largest.append((abs_delta, token, rv, cv))
            largest.sort(reverse=True)
        elif abs_delta > largest[-1][0]:
            largest[-1] = (abs_delta, token, rv, cv)
            largest.sort(reverse=True)

    return {
        "same_top1": bool(ref_top and cand_top and ref_top[0] == cand_top[0]),
        "ref_top1": ref_top[0] if ref_top else None,
        "cand_top1": cand_top[0] if cand_top else None,
        "top5_overlap": overlap(ref_top, cand_top, min(5, top_k)),
        "top20_overlap": overlap(ref_top, cand_top, min(20, top_k)),
        "top_k": top_k,
        "max_rank_delta": rank_delta(ref_top, cand_top),
        "rms": math.sqrt(sumsq / len(ref)),
        "max_abs": max_abs,
        "top20_max_abs": top_union_max_abs(ref, cand, ref_top, cand_top, min(20, top_k)),
        "nonfinite": nonfinite,
        "largest_deltas": [
            {"token": token, "ref": rv, "cand": cv, "abs": abs_delta}
            for abs_delta, token, rv, cv in largest
        ],
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "candidate",
        "same_top1",
        "top5",
        "top20",
        "rank",
        "rms",
        "max_abs",
        "top20_abs",
        "nonfinite",
    ]
    print(" | ".join(headers))
    print(" | ".join("-" * len(h) for h in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    row["label"],
                    "yes" if row["same_top1"] else "no",
                    f"{row['top5_overlap']}/5",
                    f"{row['top20_overlap']}/20",
                    str(row["max_rank_delta"]),
                    f"{row['rms']:.6g}",
                    f"{row['max_abs']:.6g}",
                    f"{row['top20_max_abs']:.6g}",
                    str(row["nonfinite"]),
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ds4 full-logit JSON dumps from --dump-logits."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", help="Labels for candidate dumps.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if args.top_k < 20:
        raise SystemExit("--top-k must be at least 20")
    if args.labels and len(args.labels) != len(args.candidates):
        raise SystemExit("--labels count must match candidate count")

    ref = load_dump(args.reference)
    candidates = [load_dump(path) for path in args.candidates]
    labels = args.labels or [dump_label(data) for data in candidates]

    print(f"reference: {dump_label(ref)}")
    print(
        "prompt_tokens: "
        f"{ref.get('prompt_tokens', '?')}  ctx: {ref.get('ctx', '?')}  "
        f"vocab: {ref.get('vocab', len(ref['logits']))}"
    )
    rows = []
    for label, candidate in zip(labels, candidates):
        if candidate.get("prompt_tokens") != ref.get("prompt_tokens"):
            print(
                f"warning: prompt token mismatch for {label}: "
                f"ref={ref.get('prompt_tokens')} cand={candidate.get('prompt_tokens')}"
            )
        metrics = compare(ref, candidate, args.top_k)
        metrics["label"] = label
        metrics["path"] = candidate["_path"]
        rows.append(metrics)

    print_table(rows)
    for row in rows:
        print(f"\n{row['label']} largest deltas:")
        for delta in row["largest_deltas"]:
            print(
                "  token={token} ref={ref:.9g} cand={cand:.9g} abs={abs:.9g}".format(
                    **delta
                )
            )

    if args.json_output:
        payload = {
            "reference": {"path": ref["_path"], "label": dump_label(ref)},
            "rows": rows,
        }
        with args.json_output.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
            fp.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
