#!/usr/bin/env python3
"""Summarize one DS4 Metal encoder timeline without guessing kernel shares.

B/E records measure traced command buffers and encoder passes. C records tag
those buffers with their phase and absolute pre-eval KV position. The optional
step CSV is CPU wall time; it is never used as a GPU coverage denominator.
Legacy traces require --phase all --legacy-all-phases and cannot yield token
attribution. Invalid counter results retain their counts and raw records but
are excluded from timing aggregates.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

U64_MAX = (1 << 64) - 1
PHASES = ("setup", "prefill", "warmup", "decode", "selection", "cleanup")
DENOMINATOR = (
    "Sum of valid GPU durations of selected traced B command buffers; not "
    "full GPU coverage, an interval union, CPU wall time, or end-to-end latency."
)
CATEGORIES = {
    "kernel_mul_mv_f16_f32_pair_compressor_store_4": "f16_compressor_pair",
    "kernel_mul_mv_f16_f32_quad_compressor_store_4": "f16_compressor_quad",
    "kernel_dsv4_qkv_pair_quad_compressor_store_q8_0": "compound_q8_qkv_plus_f16",
    "kernel_mul_mv_f16_f32_pair_4": "ordinary_f16_pair_candidate",
    "kernel_dsv4_shared_down_hc_expand4_q8_0": "q8_hc",
    "kernel_dsv4_q8_hc_expand4_q8_0": "q8_hc",
    "kernel_dsv4_q8_hc_expand4_q8_0_vec_hc": "q8_hc",
}
STEP_COLUMNS = ("phase", "step", "position", "token", "seconds", "select_seconds", "eval_seconds")


class ProfileError(ValueError):
    """The input cannot safely be attributed to a complete single run."""


def _integer(text, label, minimum=0, maximum=U64_MAX):
    try:
        value = int(text, 0) if text.lower().startswith("0x") else int(text)
    except ValueError as exc:
        raise ProfileError(f"invalid {label}: {text!r}") from exc
    if not minimum <= value <= maximum:
        raise ProfileError(f"out-of-range {label}: {text!r}")
    return value


def _finite(text, label, minimum=None):
    try:
        value = float(text)
    except ValueError as exc:
        raise ProfileError(f"invalid {label}: {text!r}") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ProfileError(f"invalid {label}: {text!r}")
    return value


def _dimensions(text):
    values = text.split("x")
    if len(values) != 3:
        raise ProfileError(f"invalid launch dimensions: {text!r}")
    return [_integer(value, "launch dimension", maximum=(1 << 32) - 1) for value in values]


def _warning(warnings, code, message, **details):
    warnings.append({"code": code, "message": message, **details})


def parse_timeline(text, *, legacy_all_phases=False):
    """Parse and validate records; completion order need not match B sequence."""
    batches, contexts, encoders, headers, comments = {}, {}, {}, [], []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append({"line": line_number, "text": raw})
            if line.startswith("# ds4 encoder timeline;"):
                match = re.search(r"\bpid=(\d+)\b", line)
                if not match:
                    raise ProfileError(f"line {line_number}: run header has no process ID")
                headers.append({"line": line_number, "pid": int(match[1]), "text": raw})
                if len(headers) > 1:
                    raise ProfileError("multiple concatenated timeline runs are not supported (repeated run header)")
            continue
        fields = line.split()
        tag = fields[0]
        try:
            if tag == "B":
                if len(fields) != 5:
                    raise ProfileError("B requires seq n_encoders gpu_start_ns gpu_end_ns")
                seq = _integer(fields[1], "batch sequence", minimum=1)
                if seq in batches:
                    raise ProfileError(f"duplicate B record for batch {seq}")
                batches[seq] = {"seq": seq, "n_encoders": _integer(fields[2], "encoder count"),
                    "gpu_start_ns": _integer(fields[3], "batch start timestamp"),
                    "gpu_end_ns": _integer(fields[4], "batch end timestamp"),
                    "line": line_number, "raw": raw}
            elif tag == "C":
                if len(fields) != 5:
                    raise ProfileError("C requires seq phase token_index dropped_encoders")
                seq = _integer(fields[1], "context sequence", minimum=1)
                if seq in contexts:
                    raise ProfileError(f"duplicate C record for batch {seq}")
                phase = fields[2]
                if phase not in PHASES:
                    raise ProfileError(f"unknown context phase {phase!r}")
                token = _integer(fields[3], "token index", minimum=-1)
                if phase in ("decode", "warmup", "selection") and token < 0:
                    raise ProfileError(f"{phase} context requires a nonnegative absolute token index")
                contexts[seq] = {"phase": phase, "token_index": token,
                    "dropped_encoders": _integer(fields[4], "dropped encoder count"),
                    "line": line_number, "raw": raw}
            elif tag == "E":
                if len(fields) != 12:
                    raise ProfileError("E requires seq idx start end dur gap caller n_dispatch tg tpt kernel")
                seq = _integer(fields[1], "encoder sequence", minimum=1)
                idx = _integer(fields[2], "encoder index")
                key = (seq, idx)
                if key in encoders:
                    raise ProfileError(f"duplicate E record for batch {seq}, index {idx}")
                encoders[key] = {"seq": seq, "idx": idx,
                    "start_ns": _integer(fields[3], "encoder start timestamp"),
                    "end_ns": _integer(fields[4], "encoder end timestamp"),
                    "reported_dur_us": _finite(fields[5], "reported duration"),
                    "reported_gap_us": _finite(fields[6], "reported gap"),
                    "caller": fields[7], "caller_address": _integer(fields[7], "caller address"),
                    "n_dispatch": _integer(fields[8], "dispatch count"),
                    "threadgroups": _dimensions(fields[9]), "threads_per_threadgroup": _dimensions(fields[10]),
                    "kernel": fields[11], "line": line_number, "raw": raw}
            else:
                raise ProfileError(f"unknown record type {tag!r}")
        except ProfileError as exc:
            raise ProfileError(f"line {line_number}: {exc}") from exc
    if not batches:
        raise ProfileError("timeline contains no B records")
    warnings = []
    if not headers:
        if not legacy_all_phases:
            raise ProfileError("timeline has no single-run process header")
        _warning(warnings, "missing_run_header", "Legacy input has no process header; its run identity is unverified.")
    orphan_contexts = set(contexts) - set(batches)
    orphan_encoders = {seq for seq, _ in encoders} - set(batches)
    if orphan_contexts or orphan_encoders:
        raise ProfileError(f"missing B records for context/encoder batches {sorted(orphan_contexts | orphan_encoders)}")
    ordered = sorted(batches)
    if any(seq != expected for expected, seq in enumerate(ordered, 1)):
        if not legacy_all_phases:
            raise ProfileError("missing B sequence(s): a complete run must contain contiguous batches beginning at 1")
        _warning(warnings, "missing_batch_sequences", "Legacy input has missing batch sequences; it is incomplete.")
    grouped = {seq: [] for seq in batches}
    for (seq, _), encoder in encoders.items():
        grouped[seq].append(encoder)
    result = []
    for seq in ordered:
        batch = batches[seq]
        context = contexts.get(seq)
        if context is None:
            if not legacy_all_phases:
                raise ProfileError(f"missing C context for batch {seq}; token/phase attribution is unsafe")
            context = {"phase": "unknown", "token_index": None, "dropped_encoders": None, "raw": None}
            _warning(warnings, "missing_context", "Legacy batch has no phase/token/drop context.", seq=seq)
        batch["context"] = context
        items = sorted(grouped[seq], key=lambda item: item["idx"])
        complete = len(items) == batch["n_encoders"] and all(item["idx"] == idx for idx, item in enumerate(items))
        if not complete:
            if not legacy_all_phases:
                raise ProfileError(f"incomplete encoder records for batch {seq}: expected {batch['n_encoders']}, found {len(items)} or noncontiguous indices")
            _warning(warnings, "incomplete_encoder_records", "Legacy encoder records do not match the batch count/indices.", seq=seq)
        batch["records_complete"] = complete
        start, end = batch["gpu_start_ns"], batch["gpu_end_ns"]
        batch["timing_valid"] = 0 < start < end < U64_MAX
        batch["gpu_ns"] = end - start if batch["timing_valid"] else None
        if not batch["timing_valid"]:
            _warning(warnings, "invalid_batch_timestamps", "Zero, sentinel or nonmonotonic B timestamps excluded from the GPU denominator.", seq=seq)
        if context["dropped_encoders"]:
            _warning(warnings, "dropped_encoders", "Some encoder passes were not traced; kernel counts and timings are partial.", seq=seq, count=context["dropped_encoders"])
        previous_end = None
        for item in items:
            start, end = item["start_ns"], item["end_ns"]
            reasons = []
            if not 0 < start < end < U64_MAX:
                reasons.append("zero_sentinel_or_nonmonotonic_counter")
            if previous_end is not None and start < previous_end:
                reasons.append("counter_order_not_monotonic")
            if not reasons:
                previous_end = end
                duration = (end - start) / 1000
                if abs(item["reported_dur_us"] - duration) > 0.005001:
                    reasons.append("reported_duration_mismatch")
            item["timing_valid"] = not reasons
            item["invalid_timing_reasons"] = reasons
            item["gpu_ns"] = end - start if not reasons else None
            if reasons:
                _warning(warnings, "invalid_encoder_timestamps", "Encoder counter result excluded from timing; its observed dispatch count is retained.", seq=seq, idx=item["idx"], reasons=reasons)
            name = item["kernel"]
            ambiguous = "+" in name or name in ("?", "unknown", "(null)") or len(name) >= 79
            if item["n_dispatch"] == 0 and name != "blit":
                ambiguous = True
            item["attribution_ambiguous"] = ambiguous
            if ambiguous:
                item["category"] = "ambiguous_encoder"
                _warning(warnings, "ambiguous_kernel", "Mixed, unknown, possibly truncated or empty-dispatch pass cannot be assigned to an exact kernel.", seq=seq, idx=item["idx"], kernel=name)
            else:
                item["category"] = CATEGORIES.get(name, "blit" if name == "blit" else "other_named_kernel")
            if item["n_dispatch"] > 1:
                _warning(warnings, "multiple_dispatches", "Encoder duration covers multiple dispatches; no individual dispatch latency is inferred.", seq=seq, idx=item["idx"], count=item["n_dispatch"])
        batch["encoders"] = items
        result.append(batch)
    if legacy_all_phases:
        _warning(warnings, "legacy_all_phases", "Explicit legacy mode combines all phases and disables per-token attribution; missing records remain visible.")
    return {"header": headers[0] if headers else None, "comments": comments,
            "batches": result, "warnings": warnings, "legacy_all_phases": legacy_all_phases}


def parse_steps(text):
    """Read the driver's explicit CPU wall time, without reconstructing it."""
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)) or not set(STEP_COLUMNS).issubset(reader.fieldnames):
        raise ProfileError("steps CSV requires unique phase,step,position,token,seconds,select_seconds,eval_seconds columns")
    rows, keys, identities = [], set(), set()
    for line, raw in enumerate(reader, 2):
        if None in raw or any(value is None for value in raw.values()):
            raise ProfileError(f"steps CSV line {line}: malformed row")
        phase = raw["phase"]
        if phase not in PHASES:
            raise ProfileError(f"steps CSV line {line}: unknown phase {phase!r}")
        row = {"phase": phase, "step": _integer(raw["step"], "CSV step", minimum=-1),
            "position": _integer(raw["position"], "CSV position"),
            "token": _integer(raw["token"], "CSV token", minimum=-1),
            **{name: _finite(raw[name], name, 0) for name in ("seconds", "select_seconds", "eval_seconds")},
            "line": line, "raw": raw}
        key, identity = (phase, row["position"]), (phase, row["step"])
        if key in keys or identity in identities:
            raise ProfileError(f"steps CSV line {line}: duplicate phase/position or phase/step")
        if row["select_seconds"] + row["eval_seconds"] > row["seconds"] + 1e-9:
            raise ProfileError(f"steps CSV line {line}: components exceed whole-step wall seconds")
        keys.add(key); identities.add(identity); rows.append(row)
    if not rows:
        raise ProfileError("steps CSV contains no rows")
    return rows


def summarize(timeline, *, phase="decode", steps=None):
    if phase not in PHASES + ("all",):
        raise ProfileError(f"unknown requested phase {phase!r}")
    if timeline["legacy_all_phases"] and phase != "all":
        raise ProfileError("legacy traces require --phase all; phase/token estimates are unavailable")
    batches = [batch for batch in timeline["batches"] if phase == "all" or batch["context"]["phase"] == phase]
    if not batches:
        raise ProfileError(f"no traced batches for phase {phase!r}")
    warnings = list(timeline["warnings"])
    denominator = sum(batch["gpu_ns"] for batch in batches if batch["timing_valid"])
    kernels, categories = {}, {}
    def add(target, key, item, batch):
        row = target.setdefault(key, {"name": key, "encoder_passes": 0, "observed_dispatches": 0,
            "valid_counter_passes": 0, "invalid_counter_passes": 0,
            "timed_passes_in_valid_batches": 0,
            "valid_counter_gpu_ns": 0, "gpu_ns_in_valid_traced_batches": 0})
        row["encoder_passes"] += 1
        row["observed_dispatches"] += item["n_dispatch"]
        row["valid_counter_passes" if item["timing_valid"] else "invalid_counter_passes"] += 1
        if item["timing_valid"]:
            row["valid_counter_gpu_ns"] += item["gpu_ns"]
            if batch["timing_valid"]:
                row["timed_passes_in_valid_batches"] += 1
                row["gpu_ns_in_valid_traced_batches"] += item["gpu_ns"]
    for batch in batches:
        for item in batch["encoders"]:
            add(categories, item["category"], item, batch)
            if not item["attribution_ambiguous"]:
                add(kernels, item["kernel"], item, batch)
    for target in (kernels, categories):
        for row in target.values():
            row["percent_of_summed_valid_traced_batch_gpu_time"] = (
                100 * row["gpu_ns_in_valid_traced_batches"] / denominator
                if denominator and row["timed_passes_in_valid_batches"] else None)
    other_names = sorted(name for name in kernels if name not in CATEGORIES and name != "blit")
    if other_names:
        _warning(warnings, "other_kernel_names", "Names outside the explicit target-kernel registry are retained as other_named_kernel, not inferred to be compressors.", names=other_names)
    timed_encoders = sum(row["gpu_ns_in_valid_traced_batches"] for row in categories.values())
    if timed_encoders > denominator:
        _warning(warnings, "encoder_time_exceeds_batch_time", "Summed encoder counter time exceeds the selected B duration denominator; inspect timestamp domains/results before interpreting percentages.")
    wall_rows = [] if steps is None else [row for row in steps if phase == "all" or row["phase"] == phase]
    wall = None
    per_token = []
    if steps is not None:
        durations = [row["seconds"] for row in wall_rows]
        wall = {"source": "driver CPU wall-clock CSV", "scope": "Whole measured steps including selection and eval; separate from the phase-selected GPU counters.",
            "row_count": len(wall_rows), "summed_step_seconds": sum(durations),
            "median_step_seconds": statistics.median(durations) if durations else None,
            "summed_select_seconds": sum(row["select_seconds"] for row in wall_rows),
            "summed_eval_seconds": sum(row["eval_seconds"] for row in wall_rows), "rows": wall_rows}
        by_key = {(row["phase"], row["position"]): row for row in steps}
        token_batches = [batch for batch in batches if batch["context"]["phase"] in ("decode", "warmup")]
        context_keys = {(batch["context"]["phase"], batch["context"]["token_index"]) for batch in token_batches}
        missing = context_keys - set(by_key)
        untraced = {(row["phase"], row["position"]) for row in wall_rows
                    if row["phase"] in ("decode", "warmup")} - context_keys
        if untraced:
            _warning(warnings, "untraced_step_rows", "Some CPU steps have no selected traced batches; no GPU duration is inferred for those steps.", contexts=sorted(untraced))
        if missing:
            _warning(warnings, "missing_step_rows", "Some token contexts have no matching CPU CSV row; per-token report disabled.", contexts=sorted(missing))
        elif not timeline["legacy_all_phases"]:
            for key in sorted(context_keys):
                matching = [batch for batch in token_batches if (batch["context"]["phase"], batch["context"]["token_index"]) == key]
                per_token.append({"phase": key[0], "position": key[1], "cpu_wall": by_key[key],
                    "traced_batch_sequences": [batch["seq"] for batch in matching],
                    "summed_valid_traced_batch_gpu_ns": sum(batch["gpu_ns"] for batch in matching if batch["timing_valid"]),
                    "dropped_encoders": sum(batch["context"]["dropped_encoders"] for batch in matching)})
    selected_sequences = {batch["seq"] for batch in batches}
    selected_codes = {"other_kernel_names", "encoder_time_exceeds_batch_time",
                      "missing_step_rows", "untraced_step_rows"}
    warnings = [{**warning, "scope": (
        "selected_phase" if warning.get("seq") in selected_sequences or warning["code"] in selected_codes
        else "outside_selected_phase" if "seq" in warning else "whole_run")} for warning in warnings]
    return {"schema_version": 1, "phase": phase, "run": timeline["header"],
        "gpu_denominator_description": DENOMINATOR,
        "summary": {"selected_batches": len(batches), "selected_encoder_passes": sum(len(batch["encoders"]) for batch in batches),
            "valid_batch_timestamps": sum(batch["timing_valid"] for batch in batches),
            "invalid_batch_timestamps": sum(not batch["timing_valid"] for batch in batches),
            "summed_valid_traced_batch_gpu_ns": denominator,
            "summed_valid_encoder_gpu_ns_in_valid_traced_batches": timed_encoders,
            "dropped_encoders": (sum(batch["context"]["dropped_encoders"] for batch in batches)
                if all(batch["context"]["dropped_encoders"] is not None for batch in batches) else None),
            "ambiguous_encoder_passes": sum(item["attribution_ambiguous"] for batch in batches for item in batch["encoders"])},
        "kernels": sorted(kernels.values(), key=lambda row: (-row["gpu_ns_in_valid_traced_batches"], row["name"])),
        "categories": sorted(categories.values(), key=lambda row: row["name"]),
        "cpu_wall_steps": wall, "per_token": per_token, "warnings": warnings,
        "raw_timeline": timeline, "raw_step_rows": steps}


def render_report(report):
    summary = report["summary"]
    dropped = summary["dropped_encoders"]
    denominator_text = (f"{summary['summed_valid_traced_batch_gpu_ns'] / 1e6:.6f} ms"
                        if summary["valid_batch_timestamps"] else "n/a")
    lines = [f"Metal encoder timeline — phase: {report['phase']}",
        f"{summary['selected_batches']} traced batches; {summary['selected_encoder_passes']} encoder passes; {'unknown' if dropped is None else dropped} dropped encoders.",
        f"GPU denominator: {denominator_text} ({summary['valid_batch_timestamps']} valid / {summary['invalid_batch_timestamps']} invalid batch timestamps).",
        report["gpu_denominator_description"], "", "Category                                    passes  dispatches  counters valid/invalid   GPU ms      % of denominator"]
    for row in report["categories"]:
        pct = row["percent_of_summed_valid_traced_batch_gpu_time"]
        gpu = f"{row['gpu_ns_in_valid_traced_batches']/1e6:.6f}" if row["timed_passes_in_valid_batches"] else "n/a"
        pct_text = f"{pct:.3f}" if pct is not None and row["timed_passes_in_valid_batches"] else "n/a"
        counters = f"{row['valid_counter_passes']}/{row['invalid_counter_passes']}"
        lines.append(f"{row['name']:<43} {row['encoder_passes']:>6} {row['observed_dispatches']:>11} {counters:>23} {gpu:>10} {pct_text:>12}")
    lines += ["", "Exact named-kernel counts (mixed/unknown passes excluded):"]
    for row in report["kernels"]:
        gpu = f"{row['gpu_ns_in_valid_traced_batches']/1e6:.6f} ms" if row["timed_passes_in_valid_batches"] else "n/a"
        lines.append(f"  {row['name']}: {row['observed_dispatches']} dispatches in {row['encoder_passes']} passes; GPU {gpu} ({row['timed_passes_in_valid_batches']} included; counters valid={row['valid_counter_passes']} invalid={row['invalid_counter_passes']})")
    lines += ["", "Compound Q8 QKV + F16 time stays compound; none is assigned to an estimated F16 share.",
              "Ordinary F16 pair is only a candidate category; timing alone does not prove a compressor call."]
    if report["cpu_wall_steps"] is not None:
        wall = report["cpu_wall_steps"]
        lines += ["", f"CPU wall steps from CSV: {wall['row_count']}; total {wall['summed_step_seconds']:.6f} s; selection {wall['summed_select_seconds']:.6f} s; eval {wall['summed_eval_seconds']:.6f} s."]
        for row in wall["rows"]:
            lines.append(f"  {row['phase']} step={row['step']} position={row['position']} token={row['token']}: {row['seconds']:.6f} s")
    if report["warnings"]:
        lines += ["", "Warnings (grouped; every original warning remains in JSON):"]
        for scope, label in (("selected_phase", "Selected phase"),
                             ("outside_selected_phase", "Outside selected phase"),
                             ("whole_run", "Whole run")):
            scoped = [warning for warning in report["warnings"] if warning["scope"] == scope]
            if not scoped:
                continue
            lines.append(f"  {label}: {len(scoped)} warning(s)")
            by_code = {}
            for warning in scoped:
                by_code.setdefault(warning["code"], []).append(warning)
            for code, grouped in sorted(by_code.items()):
                examples = []
                for warning in grouped[:3]:
                    if "seq" in warning:
                        example = f"B{warning['seq']}"
                        if "idx" in warning:
                            example += f"/E{warning['idx']}"
                        examples.append(example)
                where = f" Examples: {', '.join(examples)}." if examples else ""
                lines.append(f"    [{code}] {len(grouped)}: {grouped[0]['message']}{where}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--phase", choices=PHASES + ("all",), default="decode")
    parser.add_argument("--legacy-all-phases", action="store_true")
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--json", metavar="PATH", help="also write complete report/raw details as JSON; '-' prints only JSON")
    args = parser.parse_args(argv)
    if args.legacy_all_phases and args.phase != "all":
        parser.error("--legacy-all-phases requires --phase all")
    try:
        timeline = parse_timeline(args.timeline.read_text(), legacy_all_phases=args.legacy_all_phases)
        steps = parse_steps(args.steps.read_text()) if args.steps else None
        report = summarize(timeline, phase=args.phase, steps=steps)
        if args.json:
            payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
            if args.json == "-":
                print(payload, end="")
                return 0
            Path(args.json).write_text(payload)
        print(render_report(report), end="")
    except (OSError, ProfileError) as exc:
        parser.exit(2, f"metal_decode_profile: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
