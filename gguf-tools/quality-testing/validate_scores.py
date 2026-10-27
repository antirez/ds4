#!/usr/bin/env python3
"""Validate complete score_official TSV pairs before reporting differences."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import sys


COLUMNS = (
    "id prompt_tokens target_tokens nll avg_nll first_match greedy_lcp "
    "api_ref_tokens api_target_tokens api_target_mae api_target_mean_delta "
    "api_top_items api_top_mapped api_top_coverage api_top1_count api_top1_match "
    "api_top1_rate api_topn_ref api_topn_hit api_topn_recall api_top_logprob_count "
    "api_top_mae api_top_mean_delta api_pair_total api_pair_agree api_pair_rate"
).split()
FLOAT_COLUMNS = {
    "nll", "avg_nll", "api_target_mae", "api_target_mean_delta",
    "api_top_coverage", "api_top1_rate", "api_topn_recall", "api_top_mae",
    "api_top_mean_delta", "api_pair_rate",
}
COUNT_COLUMNS = set(COLUMNS[1:]) - FLOAT_COLUMNS
DENOMINATORS = (
    "prompt_tokens", "target_tokens", "api_ref_tokens", "api_target_tokens",
    "api_top_items", "api_top_mapped", "api_top1_count", "api_topn_ref",
    "api_top_logprob_count", "api_pair_total",
)
RATIOS = (
    ("api_top_coverage", "api_top_mapped", "api_top_items"),
    ("api_top1_rate", "api_top1_match", "api_top1_count"),
    ("api_topn_recall", "api_topn_hit", "api_topn_ref"),
    ("api_pair_rate", "api_pair_agree", "api_pair_total"),
)
MANIFEST_HEADER = "# id\tprompt_file\tcontinuation_file\tresponse_file"
HALF_PRINT_UNIT = Decimal("0.0000000005")
MAX_COUNT = 2**63 - 1


class ValidationError(ValueError):
    """An input or comparison cannot be trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_lines(path: Path) -> tuple[list[str], str]:
    data = path.read_bytes()
    require(bool(data), f"{path}: empty file")
    require(data.endswith(b"\n"), f"{path}: missing final newline (possibly truncated)")
    require(b"\x00" not in data, f"{path}: NUL byte")
    text = data.decode("utf-8")
    # score_official emits LF. Accept CRLF transport without accepting bare CR.
    text = text.replace("\r\n", "\n")
    require("\r" not in text, f"{path}: bare carriage return")
    return text[:-1].split("\n"), hashlib.sha256(data).hexdigest()


def manifest_ids(path: Path) -> tuple[tuple[str, ...], str]:
    lines, digest = read_lines(path)
    require(lines[0] == MANIFEST_HEADER, f"{path}: unexpected manifest header")
    ids = []
    seen = set()
    for line_number, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        require(len(cells) == 4, f"{path}:{line_number}: expected four manifest fields")
        require(all(v and v == v.strip() and v.lower() not in {"null", "none"}
                    for v in cells), f"{path}:{line_number}: empty/null/whitespace manifest field")
        case_id = cells[0]
        require(not any(c.isspace() for c in case_id), f"{path}:{line_number}: whitespace in case ID")
        require(case_id not in seen, f"{path}:{line_number}: duplicate manifest ID {case_id}")
        seen.add(case_id)
        ids.append(case_id)
    require(bool(ids), f"{path}: manifest has no cases")
    return tuple(ids), digest


def parse_number(value: str, field: str, where: str) -> int | Decimal:
    if field in COUNT_COLUMNS:
        require(bool(re.fullmatch(r"0|[1-9][0-9]*", value)), f"{where}: {field} is not an unsigned integer")
        require(len(value) <= 19, f"{where}: {field} exceeds scorer count range")
        number = int(value)
        require(number <= MAX_COUNT, f"{where}: {field} exceeds scorer count range")
        return number
    # %.9f is the source format. NaN/Inf, blanks, null, whitespace, scientific
    # notation and truncated decimal fields are deliberately not accepted.
    require(bool(re.fullmatch(r"-?(?:0|[1-9][0-9]{0,308})\.[0-9]{9}", value)),
            f"{where}: {field} must be a finite number with nine decimal places")
    require(math.isfinite(float(value)), f"{where}: {field} is not finite")
    return Decimal(value)


def validate_row(row: dict[str, int | Decimal], where: str) -> None:
    for field in DENOMINATORS:
        require(row[field] > 0, f"{where}: {field} must be positive; missing API coverage is not a pass")
    target = row["target_tokens"]
    require(row["api_ref_tokens"] == row["api_target_tokens"] == target,
            f"{where}: API reference/target counts must equal target_tokens")
    require(row["first_match"] in (0, 1), f"{where}: first_match must be 0 or 1")
    require(0 <= row["greedy_lcp"] <= target, f"{where}: greedy_lcp outside target length")
    require(bool(row["first_match"]) == (row["greedy_lcp"] > 0),
            f"{where}: first_match inconsistent with greedy_lcp")
    require(row["nll"] >= 0 and row["avg_nll"] >= 0, f"{where}: negative NLL")
    require(abs(row["nll"] - row["avg_nll"] * target) <= HALF_PRINT_UNIT * (target + 1),
            f"{where}: nll/avg_nll inconsistent with target_tokens and source print precision")
    require(row["api_top_mapped"] == row["api_topn_ref"] == row["api_top_logprob_count"],
            f"{where}: mapped/top-N/logprob counts differ")
    require(row["api_top1_count"] <= min(target, row["api_top_mapped"]),
            f"{where}: api_top1_count exceeds target/mapped count")
    require(row["api_pair_total"] <= row["api_top_mapped"] * 63 // 2,
            f"{where}: pair count exceeds at-most-64-mapped-alternatives bound")
    for field, numerator, denominator in RATIOS:
        require(row[numerator] <= row[denominator], f"{where}: {numerator} exceeds {denominator}")
        require(0 <= row[field] <= 1, f"{where}: {field} outside [0,1]")
        expected = Decimal(row[numerator]) / row[denominator]
        require(abs(row[field] - expected) <= HALF_PRINT_UNIT,
                f"{where}: {field} inconsistent with counts and source print precision")
    for prefix in ("api_target", "api_top"):
        mae = row[f"{prefix}_mae"]
        signed = row[f"{prefix}_mean_delta"]
        require(mae >= 0, f"{where}: {prefix}_mae is negative")
        require(abs(signed) <= mae + 2 * HALF_PRINT_UNIT,
                f"{where}: absolute signed delta exceeds MAE")


def load_scores(path: Path, ids: tuple[str, ...]) -> tuple[dict, str]:
    lines, digest = read_lines(path)
    require(lines[0].split("\t") == COLUMNS, f"{path}: unexpected score header/order/duplicate or missing column")
    rows = {}
    expected = set(ids)
    with localcontext() as context:
        context.prec = 400
        for line_number, line in enumerate(lines[1:], 2):
            where = f"{path}:{line_number}"
            values = line.split("\t")
            require(len(values) == len(COLUMNS), f"{where}: expected {len(COLUMNS)} fields, got {len(values)}")
            case_id = values[0]
            require(case_id in expected, f"{where}: unexpected case ID {case_id!r}")
            require(case_id not in rows, f"{where}: duplicate case ID {case_id}")
            row = {field: parse_number(value, field, where)
                   for field, value in zip(COLUMNS[1:], values[1:])}
            validate_row(row, where)
            rows[case_id] = row
    missing = expected - rows.keys()
    require(not missing, f"{path}: missing manifest cases: {', '.join(sorted(missing))}")
    return rows, digest


def aggregate(rows: dict, ids: tuple[str, ...]) -> dict:
    counts = {field: sum(rows[case_id][field] for case_id in ids) for field in COUNT_COLUMNS}
    nll = sum(rows[case_id]["nll"] for case_id in ids)
    result = {"counts": counts, "nll": nll, "avg_nll": nll / counts["target_tokens"]}
    for prefix, denominator in (("api_target", "api_target_tokens"), ("api_top", "api_top_logprob_count")):
        for suffix in ("mae", "mean_delta"):
            field = f"{prefix}_{suffix}"
            result[field] = sum(rows[i][field] * rows[i][denominator] for i in ids) / counts[denominator]
    for field, numerator, denominator in RATIOS:
        result[field] = Decimal(counts[numerator]) / counts[denominator]
    return result


def compare(manifest: Path, baseline: Path, candidate: Path) -> dict:
    ids, manifest_hash = manifest_ids(manifest)
    old, old_hash = load_scores(baseline, ids)
    new, new_hash = load_scores(candidate, ids)
    changed = []
    per_case = []
    with localcontext() as context:
        context.prec = 400
        for case_id in ids:
            for field in DENOMINATORS:
                require(old[case_id][field] == new[case_id][field],
                        f"{case_id}: baseline/candidate {field} mismatch")
            fields = [field for field in COLUMNS[1:] if old[case_id][field] != new[case_id][field]]
            if fields:
                changed.append({"id": case_id, "fields": fields})
            per_case.append({"id": case_id, "target_tokens": old[case_id]["target_tokens"],
                             "delta_nll": new[case_id]["nll"] - old[case_id]["nll"],
                             "delta_first_match": new[case_id]["first_match"] - old[case_id]["first_match"],
                             "delta_greedy_lcp": new[case_id]["greedy_lcp"] - old[case_id]["greedy_lcp"]})
        old_summary = aggregate(old, ids)
        new_summary = aggregate(new, ids)
        deltas = {field: new_summary[field] - old_summary[field]
                  for field in old_summary if field != "counts"}
    return {
        "validation": "complete_manifest_finite_consistent_paired_scores",
        "quality_verdict": "not_decided_by_this_tool",
        "cases": len(ids), "identical_reported_values": not changed,
        "sha256": {"manifest": manifest_hash, "baseline": old_hash, "candidate": new_hash},
        "baseline": old_summary, "candidate": new_summary,
        "delta_candidate_minus_baseline": deltas,
        "changed_cases": changed, "per_case": per_case,
    }


def json_number(value):
    if isinstance(value, Decimal):
        # Preserve decimal arithmetic; JSON numeric strings are deliberate.
        with localcontext() as context:
            context.prec = 18
            return str(+value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--strict-identical", action="store_true",
                        help="fail if any parsed per-case numeric value differs")
    args = parser.parse_args(argv)
    try:
        result = compare(args.manifest, args.baseline, args.candidate)
        failed = args.strict_identical and not result["identical_reported_values"]
        result["strict_identical_requested"] = args.strict_identical
        result["requested_check_passed"] = not failed
        print(json.dumps(result, indent=2, default=json_number, allow_nan=False, sort_keys=True))
        if failed:
            print("FAIL: reported score values differ in strict-identical mode", file=sys.stderr)
            return 1
        return 0
    except (ValidationError, OSError, UnicodeError, ArithmeticError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
