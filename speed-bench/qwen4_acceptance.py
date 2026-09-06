#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Validate a native DS4 Qwen acceptance result file.

The runner writes raw samples; this checker owns the acceptance rules so results
cannot quietly use warmups, cached prefixes, or different aggregation methods.
See QWEN38_FLASH_NEXT.md for the model setup.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


REQUIRED_PREFILL = {"prefill-10k+1", "prefill-50k+1", "prefill-100k+1"}
REQUIRED_DECODE = {"decode-code-500", "decode-prose-500", "decode-8.5k-500"}
REQUIRED_REPORTS = {
    "end-to-end-10k+500",
    "cold-ple-10k+1",
    "mtp-10k+500",
}
COLD_PLE_NAME = "cold-ple-10k+1"
COLD_PLE_SCHEMA = "ds4.qwen4.ple-cold-evidence"
COLD_PHASE_MARKER = "cold_ple_ds4_only_first_request"
COLD_PROMPT_TOKENS = 10_000
COLD_COMPLETION_TOKENS = 1
COLD_SELECTED_CHUNK = 8192
COLD_LOOKUP_ROWS = 160_000
COLD_ROW_BYTES = 100
COLD_LOGICAL_BYTES = 16_000_000
MTP_CASE_NAME = "mtp-10k+500"
MTP_DRAFT_TOKENS = 4
MTP_TIMING_SCHEMA = "ds4.qwen4.mtp-timing"
PLE_BINDING_SCHEMA = "ds4.qwen4.ple-binding"
SSD_BACKING_SCHEMA = "ds4.qwen4.ssd-backing"


def fail(message: str) -> None:
    raise ValueError(message)


def finite_positive(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric")
    if not (result > 0.0 and result < float("inf")):
        fail(f"{label} must be finite and positive")
    return result


def validate_metadata(document: dict) -> None:
    meta = document.get("metadata")
    if not isinstance(meta, dict):
        fail("metadata object is required")
    if meta.get("hardware") != "Apple M3 Ultra":
        fail("metadata.hardware must be 'Apple M3 Ultra'")
    if int(meta.get("memory_gib", 0)) != 512:
        fail("metadata.memory_gib must be 512")
    if int(meta.get("warmups_discarded", 0)) < 1:
        fail("at least one warmup must be discarded")
    if int(meta.get("batch_size", 0)) != 1:
        fail("batch_size must be one")
    if meta.get("mtp_enabled") is not False:
        fail("core acceptance requires MTP disabled")
    sequence = meta.get("execution_sequence")
    if not isinstance(sequence, list) or not sequence or (
            sequence[0] != COLD_PHASE_MARKER):
        fail("DS4-only cold PLE evidence must be the first execution phase")
    required_phases = {
        COLD_PHASE_MARKER,
        "core_ds4",
        "chunk_uncached_prefix",
        "mtp_ds4",
    }
    if len(sequence) != len(set(sequence)) or not required_phases.issubset(sequence):
        fail("metadata.execution_sequence is missing a required benchmark phase")
    validate_ple_bindings(meta)
    core = meta["core_configuration"]
    mtp = meta["mtp_configuration"]
    if core.get("mtp_enabled") is not False or (
            core.get("ds4_mtp_draft") is not None) or (
            core.get("ds4_mtp_timing") is not False):
        fail("core configuration must keep DS4 MTP disabled")
    if mtp.get("mtp_enabled") is not True or (
            exact_nonnegative_int(
                mtp.get("ds4_mtp_draft"), "MTP draft depth"
            ) != MTP_DRAFT_TOKENS) or mtp.get("ds4_mtp_timing") is not True:
        fail("MTP configuration must enable --mtp-draft 4 --mtp-timing")


def sample_median(case: dict, implementation: str,
                  metric: str) -> tuple[float, set[str]]:
    samples = case.get(implementation)
    if not isinstance(samples, list) or len(samples) < 3 or len(samples) % 2 == 0:
        fail(f"{case.get('name')}: {implementation} needs an odd sample count >= 3")
    values = []
    prompts = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(f"{case.get('name')}: invalid {implementation} sample {index}")
        if int(sample.get("cached_tokens", -1)) != 0:
            fail(f"{case.get('name')}: {implementation} sample {index} used a cache")
        prompt_id = sample.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            fail(f"{case.get('name')}: sample {index} has no unique prompt_id")
        if prompt_id in prompts:
            fail(f"{case.get('name')}: repeated prompt_id {prompt_id!r}")
        prompts.add(prompt_id)
        values.append(finite_positive(sample.get(metric),
                                      f"{case.get('name')} {implementation} {metric}"))
    return statistics.median(values), prompts


def optional_median(case: dict, implementation: str, metric: str):
    samples = case.get(implementation, [])
    values = [float(sample[metric]) for sample in samples
              if isinstance(sample, dict) and sample.get(metric) is not None]
    return statistics.median(values) if values else None


def validate_mtp_timing(sample: dict, index: int) -> dict:
    label = f"{MTP_CASE_NAME}: DS4 sample {index}"
    evidence = sample.get("mtp_timing")
    if not isinstance(evidence, dict) or (
            evidence.get("schema") != MTP_TIMING_SCHEMA) or (
            evidence.get("version") != 1):
        fail(f"{label} has no supported MTP timing evidence")
    cycles = exact_nonnegative_int(evidence.get("cycles"), f"{label} cycles")
    drafted = exact_nonnegative_int(
        evidence.get("drafted"), f"{label} drafted tokens"
    )
    max_drafted = exact_nonnegative_int(
        evidence.get("max_drafted"), f"{label} maximum drafts per cycle"
    )
    accepted = exact_nonnegative_int(
        evidence.get("accepted"), f"{label} accepted tokens"
    )
    target_tokens = exact_nonnegative_int(
        evidence.get("target_tokens"), f"{label} target tokens"
    )
    completion_tokens = exact_nonnegative_int(
        sample.get("completion_tokens"), f"{label} completion tokens"
    )
    try:
        cycle_ms = float(evidence.get("cycle_ms"))
    except (TypeError, ValueError):
        fail(f"{label} cycle_ms is not numeric")
    if not math.isfinite(cycle_ms) or cycle_ms < 0.0:
        fail(f"{label} cycle_ms must be finite and nonnegative")
    if cycles <= 0 or drafted <= 0 or accepted <= 0:
        fail(f"{label} must show cycles, drafts, and accepted draft tokens")
    if accepted > drafted:
        fail(f"{label} accepted more tokens than it drafted")
    if max_drafted <= 0 or max_drafted > 4 or max_drafted > drafted:
        fail(f"{label} exceeds the configured draft depth")
    if target_tokens != cycles + accepted:
        fail(f"{label} has inconsistent cycle/token accounting")
    verifier = evidence.get("verifier")
    if not isinstance(verifier, dict) or not verifier:
        fail(f"{label} has no verifier evidence")
    verifier_cycles = 0
    for mode, count in verifier.items():
        if not isinstance(mode, str) or not mode:
            fail(f"{label} has an invalid verifier mode")
        verifier_cycles += exact_nonnegative_int(
            count, f"{label} {mode} verifier cycles"
        )
    if verifier_cycles != cycles:
        fail(f"{label} verifier cycle count differs from MTP cycles")
    if exact_nonnegative_int(
            verifier.get("block", 0),
            f"{label} block verifier cycles") <= 0:
        fail(f"{label} did not observe the block verifier")
    if target_tokens != completion_tokens:
        fail(f"{label} target-token sum differs from successful completion tokens")
    return {
        "cycles": cycles,
        "drafted": drafted,
        "accepted": accepted,
        "target_tokens": target_tokens,
    }


def exact_nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool):
        fail(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        fail(f"{label} must be an integer")
    if result < 0 or result != value:
        fail(f"{label} must be a nonnegative integer")
    return result


def valid_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def validate_ple_binding(binding: dict) -> None:
    if binding.get("schema") != PLE_BINDING_SCHEMA or binding.get("version") != 1:
        fail("unsupported PLE binding schema/version")
    if not isinstance(binding.get("path"), str) or not binding["path"]:
        fail("PLE binding path is required")
    identity = binding.get("descriptor_identity")
    if not isinstance(identity, dict):
        fail("PLE descriptor identity is required")
    for field in ("device", "inode", "size", "mtime_ns", "ctime_ns"):
        exact_nonnegative_int(identity.get(field), f"PLE identity {field}")
    if identity["size"] == 0:
        fail("PLE descriptor size must be positive")
    pack = binding.get("pack")
    if not isinstance(pack, dict) or not all((
        isinstance(pack.get("pack_id"), str) and bool(pack["pack_id"]),
        valid_sha256(pack.get("manifest_sha256")),
        valid_sha256(pack.get("tensor_manifest_sha256")),
        valid_sha256(pack.get("ple_sha256")),
        exact_nonnegative_int(pack.get("ple_bytes"), "pack PLE bytes")
            == identity["size"],
    )):
        fail("PLE binding has incomplete SHA/pack identity")
    ssd = binding.get("ssd_backing")
    if not isinstance(ssd, dict) or ssd.get("schema") != SSD_BACKING_SCHEMA or (
            ssd.get("version") != 1) or ssd.get("local") is not True or (
            ssd.get("solid_state") is not True):
        fail("PLE must have local solid-state backing evidence")
    for field in ("filesystem", "device_node"):
        if not isinstance(ssd.get(field), str) or not ssd[field].startswith("/dev/"):
            fail(f"PLE SSD evidence has invalid {field}")
    if not isinstance(ssd.get("mount_point"), str) or not ssd["mount_point"]:
        fail("PLE SSD evidence has no mount point")


def validate_ple_bindings(metadata: dict) -> None:
    keys = (
        "cold_ple_configuration", "core_configuration",
        "chunk_configuration", "mtp_configuration",
    )
    configs = []
    for key in keys:
        config = metadata.get(key)
        if not isinstance(config, dict):
            fail(f"metadata.{key} is required")
        configs.append((key, config))
    canonical = configs[0][1].get("ple_binding")
    if not isinstance(canonical, dict):
        fail("cold PLE binding is required")
    validate_ple_binding(canonical)
    descriptor = canonical["descriptor_identity"]
    for key, config in configs:
        if config.get("ple_binding") != canonical:
            fail(f"{key} does not use the cold PLE descriptor/pack/SSD binding")
        if config.get("ple_identity_before") != descriptor or (
                config.get("ple_identity_after") != descriptor):
            fail(f"{key} did not preserve the PLE descriptor identity")


def validate_cold_ple(case: dict, metadata: dict) -> dict:
    if case.get("kind") != "diagnostic":
        fail(f"{COLD_PLE_NAME}: kind must be diagnostic")
    if case.get("ple_cache") != "darwin-mincore-target-pages":
        fail(f"{COLD_PLE_NAME}: exact target-page mincore evidence is required")
    samples = case.get("ds4")
    if not isinstance(samples, list) or len(samples) != 1:
        fail(f"{COLD_PLE_NAME}: exactly one first-request DS4 sample is required")
    sample = samples[0]
    if not isinstance(sample, dict):
        fail(f"{COLD_PLE_NAME}: invalid DS4 sample")
    if exact_nonnegative_int(
            sample.get("cached_tokens", -1), "cold cached_tokens") != 0:
        fail(f"{COLD_PLE_NAME}: prefix cache was reused")
    prompt_id = sample.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        fail(f"{COLD_PLE_NAME}: prompt_id is required")
    prompt_tokens = exact_nonnegative_int(
        sample.get("prompt_tokens"), "cold prompt_tokens"
    )
    completion_tokens = exact_nonnegative_int(
        sample.get("completion_tokens"), "cold completion_tokens"
    )
    selected_chunk = exact_nonnegative_int(
        sample.get("selected_chunk"), "cold selected_chunk"
    )
    if prompt_tokens != COLD_PROMPT_TOKENS or (
            completion_tokens != COLD_COMPLETION_TOKENS) or (
            selected_chunk != COLD_SELECTED_CHUNK):
        fail(
            f"{COLD_PLE_NAME}: requires exactly 10K prompt tokens, one "
            "completion token, and the 8192 prefill path"
        )
    raw_ds4_rate = finite_positive(
        sample.get("prefill_tps"), f"{COLD_PLE_NAME} prefill_tps"
    )
    raw_first_token = finite_positive(
        sample.get("first_token_ms"), f"{COLD_PLE_NAME} first_token_ms"
    )
    peak_memory = finite_positive(
        sample.get("peak_memory_gib"), f"{COLD_PLE_NAME} peak_memory_gib"
    )

    evidence = sample.get("ple_cold_evidence")
    if not isinstance(evidence, dict):
        fail(f"{COLD_PLE_NAME}: PLE cold evidence object is required")
    if evidence.get("schema") != COLD_PLE_SCHEMA or evidence.get("version") != 1:
        fail(f"{COLD_PLE_NAME}: unsupported PLE evidence schema/version")
    if exact_nonnegative_int(evidence.get("sequence"), "cold sequence") != 1:
        fail(f"{COLD_PLE_NAME}: evidence sequence must be one")
    if evidence.get("measurement_order") != (
            "after-full-ngram-hash-before-first-submit"):
        fail(f"{COLD_PLE_NAME}: evidence was sampled at the wrong boundary")
    probe_ms = finite_positive(
        evidence.get("probe_ms"), f"{COLD_PLE_NAME} probe_ms"
    )

    flags = evidence.get("startup_flags")
    required_flags = (
        "validation_nocache_requested",
        "validation_nocache_enabled",
        "validation_nocache_cleared",
        "runtime_readahead_requested",
        "runtime_readahead_disabled",
    )
    if not isinstance(flags, dict) or any(
            flags.get(name) is not True for name in required_flags):
        fail(
            f"{COLD_PLE_NAME}: F_NOCACHE enable/clear and F_RDAHEAD "
            "disable must all be confirmed"
        )

    counts = evidence.get("request_counts")
    if not isinstance(counts, dict) or exact_nonnegative_int(
            counts.get("prompt_requests"), "cold prompt request count") != 1:
        fail(f"{COLD_PLE_NAME}: evidence must come from the first prompt")
    if exact_nonnegative_int(
            counts.get("ple_submits_before"), "cold submit count") != 0:
        fail(f"{COLD_PLE_NAME}: PLE was submitted before residency sampling")

    target = evidence.get("target")
    residency = evidence.get("residency")
    if not isinstance(target, dict) or not isinstance(residency, dict):
        fail(f"{COLD_PLE_NAME}: target/residency evidence is required")
    if exact_nonnegative_int(target.get("tokens"), "cold target tokens") != prompt_tokens:
        fail(f"{COLD_PLE_NAME}: evidence does not cover the complete prompt")
    heads = exact_nonnegative_int(target.get("ngram_heads"), "cold ngram heads")
    rows = exact_nonnegative_int(target.get("ngram_rows"), "cold ngram rows")
    lookup_rows = exact_nonnegative_int(
        target.get("lookup_rows"), "cold lookup rows"
    )
    row_bytes = exact_nonnegative_int(target.get("row_bytes"), "cold row bytes")
    logical_bytes = exact_nonnegative_int(
        target.get("logical_bytes"), "cold logical bytes"
    )
    if heads != 16 or rows != COLD_LOOKUP_ROWS or (
            lookup_rows != COLD_LOOKUP_ROWS) or row_bytes != COLD_ROW_BYTES or (
            logical_bytes != COLD_LOGICAL_BYTES) or (
            rows != prompt_tokens * heads) or logical_bytes != rows * row_bytes:
        fail(f"{COLD_PLE_NAME}: target must be 160K rows and 16MB logical PLE")
    page_size = exact_nonnegative_int(target.get("page_size"), "cold page size")
    target_pages = exact_nonnegative_int(target.get("pages"), "cold target pages")
    resident_pages = exact_nonnegative_int(
        residency.get("resident_pages"), "cold resident pages"
    )
    cold_pages = exact_nonnegative_int(
        residency.get("cold_pages"), "cold pages"
    )
    if page_size == 0 or target_pages == 0 or (
            resident_pages + cold_pages != target_pages):
        fail(f"{COLD_PLE_NAME}: inconsistent exact target-page counts")
    cold_fraction = cold_pages / target_pages
    reported_fraction = float(residency.get("cold_fraction", -1.0))
    if abs(reported_fraction - cold_fraction) > 1.0e-8:
        fail(f"{COLD_PLE_NAME}: reported cold fraction is not derived from pages")
    if cold_fraction < 0.95:
        fail(
            f"{COLD_PLE_NAME}: only {cold_fraction:.1%} of exact target "
            "pages were cold; at least 95% is required"
        )

    before = sample.get("ple_identity_before")
    after = sample.get("ple_identity_after")
    identity = evidence.get("ple_identity")
    cold_config = metadata.get("cold_ple_configuration", {})
    configured = cold_config.get("ple_identity_before")
    binding = cold_config.get("ple_binding", {})
    if not all(isinstance(value, dict) for value in (
            before, after, identity, configured)):
        fail(f"{COLD_PLE_NAME}: stable PLE stat identity is required")
    if before != after or before != configured:
        fail(f"{COLD_PLE_NAME}: PLE artifact changed during measurement")
    identity_mtime_ns = exact_nonnegative_int(
        identity.get("mtime_sec"), "cold identity mtime_sec"
    ) * 1_000_000_000 + exact_nonnegative_int(
        identity.get("mtime_nsec"), "cold identity mtime_nsec"
    )
    identity_ctime_ns = exact_nonnegative_int(
        identity.get("ctime_sec"), "cold identity ctime_sec"
    ) * 1_000_000_000 + exact_nonnegative_int(
        identity.get("ctime_nsec"), "cold identity ctime_nsec"
    )
    if identity.get("stable") is not True or any((
        exact_nonnegative_int(identity.get("device"), "cold identity device")
            != before.get("device"),
        exact_nonnegative_int(identity.get("inode"), "cold identity inode")
            != before.get("inode"),
        exact_nonnegative_int(identity.get("size"), "cold identity size")
            != before.get("size"),
        identity_mtime_ns != before.get("mtime_ns"),
        identity_ctime_ns != before.get("ctime_ns"),
    )):
        fail(f"{COLD_PLE_NAME}: runtime PLE identity does not match the harness")
    digest = identity.get("sha256")
    if not valid_sha256(digest) or digest != binding.get("pack", {}).get(
            "ple_sha256"):
        fail(f"{COLD_PLE_NAME}: validated PLE SHA-256 identity is required")

    rusage = sample.get("rusage_v2")
    if not isinstance(rusage, dict) or rusage.get("flavor") != 2:
        fail(f"{COLD_PLE_NAME}: proc_pid_rusage V2 corroboration is required")
    r_before, r_after, delta = (
        rusage.get("before"), rusage.get("after"), rusage.get("delta")
    )
    if not all(isinstance(value, dict) for value in (
            r_before, r_after, delta)):
        fail(f"{COLD_PLE_NAME}: incomplete proc_pid_rusage V2 evidence")
    before_start = exact_nonnegative_int(
        r_before.get("proc_start_abstime"), "rusage before process start"
    )
    after_start = exact_nonnegative_int(
        r_after.get("proc_start_abstime"), "rusage after process start"
    )
    if before_start == 0 or after_start != before_start:
        fail(f"{COLD_PLE_NAME}: rusage snapshots changed process lifetime")
    for field in ("pageins", "diskio_bytesread", "diskio_byteswritten"):
        left = exact_nonnegative_int(r_before.get(field), f"rusage before {field}")
        right = exact_nonnegative_int(r_after.get(field), f"rusage after {field}")
        change = exact_nonnegative_int(delta.get(field), f"rusage delta {field}")
        if right < left or change != right - left:
            fail(f"{COLD_PLE_NAME}: inconsistent rusage delta for {field}")
    if int(delta["diskio_bytesread"]) < COLD_LOGICAL_BYTES:
        fail(f"{COLD_PLE_NAME}: fewer than 16MB of cold disk reads were observed")

    timing = sample.get("timing")
    raw_timing = timing.get("raw_observed") if isinstance(timing, dict) else None
    adjusted = timing.get("probe_adjusted") if isinstance(timing, dict) else None
    if not isinstance(raw_timing, dict) or not isinstance(adjusted, dict):
        fail(f"{COLD_PLE_NAME}: raw/probe-adjusted timing evidence is required")
    timing_probe_ms = finite_positive(
        timing.get("probe_ms"), "cold timing probe_ms"
    )
    if not math.isclose(timing_probe_ms, probe_ms,
                        rel_tol=0.0, abs_tol=1.0e-6):
        fail(f"{COLD_PLE_NAME}: timing probe does not match residency evidence")
    raw_prefill_seconds = finite_positive(
        raw_timing.get("prefill_seconds"), "cold raw prefill seconds"
    )
    raw_request_ms = finite_positive(
        raw_timing.get("request_elapsed_ms"), "cold raw request milliseconds"
    )
    if not math.isclose(
            raw_prefill_seconds,
            finite_positive(sample.get("prefill_seconds"),
                            "cold prefill seconds"),
            rel_tol=0.0, abs_tol=1.0e-9) or not math.isclose(
            finite_positive(raw_timing.get("prefill_tps"), "cold raw prefill tps"),
            raw_ds4_rate, rel_tol=0.0, abs_tol=1.0e-9) or not math.isclose(
            finite_positive(raw_timing.get("first_token_ms"),
                            "cold raw first token"),
            raw_first_token, rel_tol=0.0, abs_tol=1.0e-9) or not math.isclose(
            raw_request_ms,
            finite_positive(sample.get("request_elapsed_ms"),
                            "cold request elapsed ms"),
            rel_tol=0.0, abs_tol=1.0e-9):
        fail(f"{COLD_PLE_NAME}: raw observed timing was not preserved")
    expected_prefill_seconds = raw_prefill_seconds - probe_ms / 1000.0
    expected_first_token = raw_first_token - probe_ms
    expected_request_ms = raw_request_ms - probe_ms
    if min(expected_prefill_seconds, expected_first_token,
           expected_request_ms) <= 0.0:
        fail(f"{COLD_PLE_NAME}: probe exceeds the raw observed timing")
    adjusted_prefill_seconds = finite_positive(
        adjusted.get("prefill_seconds"), "cold adjusted prefill seconds"
    )
    adjusted_rate = finite_positive(
        adjusted.get("prefill_tps"), "cold adjusted prefill tps"
    )
    adjusted_first_token = finite_positive(
        adjusted.get("first_token_ms"), "cold adjusted first token"
    )
    adjusted_request_ms = finite_positive(
        adjusted.get("request_elapsed_ms"), "cold adjusted request ms"
    )
    expected_rate = prompt_tokens / expected_prefill_seconds
    checks = (
        (adjusted_prefill_seconds, expected_prefill_seconds),
        (adjusted_rate, expected_rate),
        (adjusted_first_token, expected_first_token),
        (adjusted_request_ms, expected_request_ms),
    )
    if any(not math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-6)
           for left, right in checks):
        fail(f"{COLD_PLE_NAME}: probe-adjusted timing is not derived from raw timing")

    return {
        "name": COLD_PLE_NAME,
        "kind": "diagnostic",
        "metric": "prefill_tps",
        "ds4_median": adjusted_rate,
        "ds4_first_token_ms": adjusted_first_token,
        "ds4_peak_memory_gib": peak_memory,
        "cold_target_fraction": cold_fraction,
        "raw_ds4_prefill_tps": raw_ds4_rate,
        "raw_ds4_first_token_ms": raw_first_token,
        "probe_ms": probe_ms,
    }


def validate_chunk_policy(document: dict) -> list[str]:
    rows = document.get("chunk_comparisons")
    if not isinstance(rows, list) or len(rows) != 1:
        fail("chunk_comparisons must contain exactly one uncached-prefix-10k comparison")
    row = rows[0]
    if not isinstance(row, dict) or row.get("name") != "uncached-prefix-10k":
        fail("chunk_comparisons must contain the uncached-prefix-10k comparison")
    name = row["name"]
    t2 = finite_positive(row.get("chunk_2048_tps"), f"{name} chunk_2048_tps")
    t8 = finite_positive(row.get("chunk_8192_tps"), f"{name} chunk_8192_tps")
    try:
        selected = int(row.get("auto_selected", 0))
    except (TypeError, ValueError):
        fail(f"{name}: auto_selected must be 2048 or 8192")
    expected = 8192 if t8 >= 0.97 * t2 else 2048
    if selected != expected:
        if expected == 2048:
            fail(f"{name}: 8K regresses more than 3%; auto must retain 2048")
        fail(f"{name}: qualifying 8K prefill must be selected automatically")
    return [f"{name}: 8K/2K={t8 / t2:.3f}, auto={selected}"]


def validate(document: dict) -> tuple[list[dict], list[str]]:
    validate_metadata(document)
    metadata = document["metadata"]
    cases = document.get("cases")
    if not isinstance(cases, list):
        fail("cases array is required")
    case_names = [case.get("name") for case in cases if isinstance(case, dict)]
    if len(case_names) != len(cases) or len(set(case_names)) != len(case_names):
        fail("cases must be objects with unique names")
    names = set(case_names)
    missing = (REQUIRED_PREFILL | REQUIRED_DECODE | REQUIRED_REPORTS) - names
    if missing:
        fail("missing required cases: " + ", ".join(sorted(missing)))

    results = []
    for case in cases:
        name = case.get("name")
        kind = case.get("kind")
        if kind not in ("prefill", "decode", "end_to_end", "diagnostic"):
            fail(f"{name}: invalid kind {kind!r}")
        if name == COLD_PLE_NAME:
            results.append(validate_cold_ple(case, metadata))
            continue
        metric = "prefill_tps" if kind == "prefill" else (
            "decode_tps" if kind == "decode" else "total_tps")
        ds4, _ = sample_median(case, "ds4", metric)
        mtp_evidence = None
        if name == MTP_CASE_NAME:
            if case.get("mtp_enabled") is not True:
                fail(f"{MTP_CASE_NAME}: mtp_enabled must be true")
            mtp_evidence = [
                validate_mtp_timing(sample, index)
                for index, sample in enumerate(case["ds4"])
            ]
        first_ds4 = optional_median(case, "ds4", "first_token_ms")
        peak_ds4 = optional_median(case, "ds4", "peak_memory_gib")
        if name in (REQUIRED_REPORTS - {COLD_PLE_NAME}) and None in (
                first_ds4, peak_ds4):
            fail(
                f"{name}: first-token latency and peak memory are required "
                "for DS4"
            )
        result = {
            "name": name,
            "kind": kind,
            "metric": metric,
            "ds4_median": ds4,
            "ds4_first_token_ms": first_ds4,
            "ds4_peak_memory_gib": peak_ds4,
        }
        if mtp_evidence is not None:
            result.update({
                "ds4_mtp_cycles_median": statistics.median(
                    row["cycles"] for row in mtp_evidence
                ),
                "ds4_mtp_drafted_median": statistics.median(
                    row["drafted"] for row in mtp_evidence
                ),
                "ds4_mtp_accepted_median": statistics.median(
                    row["accepted"] for row in mtp_evidence
                ),
            })
        results.append(result)
    return results, validate_chunk_policy(document)


def render_markdown(results: list[dict], notes: list[str]) -> str:
    lines = [
        "| Case | Metric | DS4 median |",
        "|---|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['metric']} | "
            f"{row['ds4_median']:.2f} |"
        )
    reported = [
        row for row in results
        if row["ds4_first_token_ms"] is not None
        and row["ds4_peak_memory_gib"] is not None
    ]
    if reported:
        lines.extend([
            "",
            "| Case | DS4 first token ms | DS4 peak GiB |",
            "|---|---:|---:|",
        ])
        for row in reported:
            lines.append(
                f"| {row['name']} | {row['ds4_first_token_ms']:.2f} | "
                f"{row['ds4_peak_memory_gib']:.2f} |"
            )
    if notes:
        lines.extend(["", "Chunk policy:"] + [f"- {note}" for note in notes])
    cold = next((row for row in results if row.get("cold_target_fraction")
                 is not None), None)
    if cold:
        lines.extend([
            "",
            f"Cold PLE target pages: {cold['cold_target_fraction']:.1%} cold "
            "before the first gather (Darwin mincore; rusage V2 corroborated).",
            f"Cold PLE timing: observed {cold['raw_ds4_prefill_tps']:.2f} tok/s, "
            f"probe-adjusted {cold['ds4_median']:.2f} tok/s "
            f"({cold['probe_ms']:.2f} ms residency probe reported separately).",
        ])
    mtp = next((row for row in results if row["name"] == MTP_CASE_NAME), None)
    if mtp:
        lines.extend([
            "",
            "DS4 MTP evidence: median "
            f"{mtp['ds4_mtp_cycles_median']:.0f} cycles, "
            f"{mtp['ds4_mtp_drafted_median']:.0f} drafted tokens, and "
            f"{mtp['ds4_mtp_accepted_median']:.0f} accepted draft tokens; "
            "the block verifier covered the successful completion.",
        ])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.result.read_text())
    results, notes = validate(document)
    report = render_markdown(results, notes)
    if args.markdown:
        args.markdown.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen4-acceptance: {error}", file=sys.stderr)
        raise SystemExit(2)
