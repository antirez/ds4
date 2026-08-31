#!/usr/bin/env python3

import copy
import unittest

from qwen4_acceptance import validate


class Qwen4AcceptanceTests(unittest.TestCase):
    def document(self):
        identity = {
            "device": 9,
            "inode": 42,
            "size": 32_000_000_000,
            "mtime_ns": 2_000_000_003,
            "ctime_ns": 4_000_000_005,
        }
        binding = {
            "schema": "ds4.qwen4.ple-binding",
            "version": 1,
            "path": "/pack/qwen-ple.gguf",
            "descriptor_identity": identity,
            "pack": {
                "manifest_path": "/pack/qwen3.8-flash-next-q4.manifest.json",
                "manifest_sha256": "b" * 64,
                "pack_id": "pack-1",
                "tensor_manifest_sha256": "c" * 64,
                "ple_sha256": "a" * 64,
                "ple_bytes": 32_000_000_000,
            },
            "ssd_backing": {
                "schema": "ds4.qwen4.ssd-backing",
                "version": 1,
                "probe": "df-P+diskutil-info-plist",
                "filesystem": "/dev/disk3s5",
                "mount_point": "/System/Volumes/Data",
                "device_node": "/dev/disk3s5",
                "device_identifier": "disk3s5",
                "parent_whole_disk": "disk3",
                "bus_protocol": "Apple Fabric",
                "internal": True,
                "local": True,
                "solid_state": True,
            },
        }
        cases = []
        definitions = {
            "prefill-10k+1": "prefill",
            "prefill-50k+1": "prefill",
            "prefill-100k+1": "prefill",
            "decode-code-500": "decode",
            "decode-prose-500": "decode",
            "decode-8.5k-500": "decode",
            "end-to-end-10k+500": "end_to_end",
            "cold-ple-10k+1": "diagnostic",
            "mtp-10k+500": "end_to_end",
        }
        for name, kind in definitions.items():
            if name == "cold-ple-10k+1":
                evidence = {
                    "schema": "ds4.qwen4.ple-cold-evidence",
                    "version": 1,
                    "sequence": 1,
                    "measurement_order": (
                        "after-full-ngram-hash-before-first-submit"
                    ),
                    "ple_identity": {
                        "device": 9,
                        "inode": 42,
                        "size": 32_000_000_000,
                        "mtime_sec": 2,
                        "mtime_nsec": 3,
                        "ctime_sec": 4,
                        "ctime_nsec": 5,
                        "sha256": "a" * 64,
                        "stable": True,
                    },
                    "startup_flags": {
                        "validation_nocache_requested": True,
                        "validation_nocache_enabled": True,
                        "validation_nocache_cleared": True,
                        "runtime_readahead_requested": True,
                        "runtime_readahead_disabled": True,
                    },
                    "request_counts": {
                        "prompt_requests": 1,
                        "ple_submits_before": 0,
                    },
                    "target": {
                        "tokens": 10_000,
                        "ngram_heads": 16,
                        "ngram_rows": 160_000,
                        "lookup_rows": 160_000,
                        "row_bytes": 100,
                        "logical_bytes": 16_000_000,
                        "page_size": 16_384,
                        "pages": 100,
                    },
                    "residency": {
                        "resident_pages": 4,
                        "cold_pages": 96,
                        "cold_fraction": 0.96,
                    },
                    "probe_ms": 100.0,
                }
                cases.append({
                    "name": name,
                    "kind": kind,
                    "ple_cache": "darwin-mincore-target-pages",
                    "ds4": [{
                        "prompt_id": "cold-first-prompt",
                        "prompt_tokens": 10_000,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                        "selected_chunk": 8192,
                        "prefill_seconds": 20.0,
                        "prefill_tps": 500.0,
                        "first_token_ms": 20_100.0,
                        "request_elapsed_ms": 20_200.0,
                        "peak_memory_gib": 80.0,
                        "ple_cold_evidence": evidence,
                        "ple_identity_before": identity,
                        "ple_identity_after": identity,
                        "rusage_v2": {
                            "flavor": 2,
                            "before": {
                                "proc_start_abstime": 1234,
                                "pageins": 10,
                                "diskio_bytesread": 100,
                                "diskio_byteswritten": 0,
                            },
                            "after": {
                                "proc_start_abstime": 1234,
                                "pageins": 12,
                                "diskio_bytesread": 20_000_100,
                                "diskio_byteswritten": 0,
                            },
                            "delta": {
                                "pageins": 2,
                                "diskio_bytesread": 20_000_000,
                                "diskio_byteswritten": 0,
                            },
                        },
                        "timing": {
                            "probe_ms": 100.0,
                            "raw_observed": {
                                "prefill_seconds": 20.0,
                                "prefill_tps": 500.0,
                                "first_token_ms": 20_100.0,
                                "request_elapsed_ms": 20_200.0,
                            },
                            "probe_adjusted": {
                                "prefill_seconds": 19.9,
                                "prefill_tps": 10_000 / 19.9,
                                "first_token_ms": 20_000.0,
                                "request_elapsed_ms": 20_100.0,
                            },
                        },
                    }],
                })
                continue
            metric = "prefill_tps" if kind == "prefill" else (
                "decode_tps" if kind == "decode" else "total_tps"
            )
            case = {
                "name": name,
                "kind": kind,
                "ds4": [],
            }
            for sample in range(3):
                row = {
                    "prompt_id": f"{name}-{sample}",
                    "cached_tokens": 0,
                    metric: 95.0 + sample,
                }
                if name in {
                    "end-to-end-10k+500",
                    "mtp-10k+500",
                }:
                    row["first_token_ms"] = 100.0 + sample
                    row["peak_memory_gib"] = 80.0 + sample
                if name == "mtp-10k+500":
                    row["completion_tokens"] = 500
                    row["mtp_timing"] = {
                        "schema": "ds4.qwen4.mtp-timing",
                        "version": 1,
                        "cycles": 300 - sample,
                        "drafted": 300 + sample,
                        "max_drafted": 4,
                        "accepted": 200 + sample,
                        "target_tokens": 500,
                        "cycle_ms": 900.0 + sample,
                        "verifier": {
                            "block": 299 - sample,
                            "unlabeled": 1,
                        },
                    }
                case["ds4"].append(row)
            if name == "mtp-10k+500":
                case["mtp_enabled"] = True
            cases.append(case)
        return {
            "metadata": {
                "hardware": "Apple M3 Ultra",
                "memory_gib": 512,
                "warmups_discarded": 1,
                "batch_size": 1,
                "mtp_enabled": False,
                "execution_sequence": [
                    "cold_ple_ds4_only_first_request",
                    "core_ds4",
                    "chunk_uncached_prefix",
                    "mtp_ds4",
                ],
                "cold_ple_configuration": {
                    "ds4_ple": "/pack/qwen-ple.gguf",
                    "ple_binding": binding,
                    "ple_identity_before": identity,
                    "ple_identity_after": identity,
                },
                "core_configuration": {
                    "ds4_ple": "/pack/qwen-ple.gguf",
                    "mtp_enabled": False,
                    "ds4_mtp_draft": None,
                    "ds4_mtp_timing": False,
                    "ple_binding": binding,
                    "ple_identity_before": identity,
                    "ple_identity_after": identity,
                },
                "chunk_configuration": {
                    "ds4_ple": "/pack/qwen-ple.gguf",
                    "ple_binding": binding,
                    "ple_identity_before": identity,
                    "ple_identity_after": identity,
                },
                "mtp_configuration": {
                    "ds4_ple": "/pack/qwen-ple.gguf",
                    "mtp_enabled": True,
                    "ds4_mtp_draft": 4,
                    "ds4_mtp_timing": True,
                    "ple_binding": binding,
                    "ple_identity_before": identity,
                    "ple_identity_after": identity,
                },
            },
            "cases": cases,
            "chunk_comparisons": [{
                "name": "uncached-prefix-10k",
                "chunk_2048_tps": 100.0,
                "chunk_8192_tps": 98.0,
                "auto_selected": 8192,
            }],
        }

    def test_complete_report_passes(self):
        results, notes = validate(self.document())
        self.assertEqual(len(results), 9)
        self.assertEqual(len(notes), 1)

    def test_report_cases_require_latency_and_memory(self):
        document = self.document()
        case = next(
            row for row in document["cases"]
            if row["name"] == "end-to-end-10k+500"
        )
        for sample in case["ds4"]:
            sample.pop("peak_memory_gib")
        with self.assertRaisesRegex(ValueError, "peak memory"):
            validate(document)

    def test_mtp_samples_require_real_speculative_timing(self):
        mutations = (
            ("cycles", 0),
            ("drafted", 0),
            ("accepted", 0),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                document = copy.deepcopy(self.document())
                case = next(row for row in document["cases"]
                            if row["name"] == "mtp-10k+500")
                case["ds4"][0]["mtp_timing"][field] = value
                with self.assertRaisesRegex(
                        ValueError, "cycles, drafts, and accepted"):
                    validate(document)

    def test_mtp_acceptance_cannot_exceed_drafts(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "mtp-10k+500")["ds4"][0]
        sample["mtp_timing"]["accepted"] = 301
        with self.assertRaisesRegex(ValueError, "more tokens than it drafted"):
            validate(document)

    def test_mtp_maximum_draft_depth_is_enforced(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "mtp-10k+500")["ds4"][0]
        sample["mtp_timing"]["max_drafted"] = 5
        with self.assertRaisesRegex(ValueError, "configured draft depth"):
            validate(document)

    def test_mtp_requires_block_verifier_evidence(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "mtp-10k+500")["ds4"][0]
        sample["mtp_timing"]["verifier"] = {
            "unlabeled": sample["mtp_timing"]["cycles"]
        }
        with self.assertRaisesRegex(ValueError, "block verifier"):
            validate(document)

    def test_mtp_target_sum_must_cover_successful_completion(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "mtp-10k+500")["ds4"][0]
        sample["completion_tokens"] = 499
        with self.assertRaisesRegex(ValueError, "successful completion tokens"):
            validate(document)

    def test_mtp_cycle_token_accounting_must_be_consistent(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "mtp-10k+500")["ds4"][0]
        sample["mtp_timing"]["cycles"] -= 1
        sample["mtp_timing"]["verifier"]["block"] -= 1
        with self.assertRaisesRegex(ValueError, "cycle/token accounting"):
            validate(document)

    def test_mtp_configuration_requires_draft_four_and_timing(self):
        document = copy.deepcopy(self.document())
        document["metadata"]["mtp_configuration"]["ds4_mtp_draft"] = 1
        with self.assertRaisesRegex(ValueError, "--mtp-draft 4 --mtp-timing"):
            validate(document)

    def test_slow_8k_requires_2k_auto_selection(self):
        document = copy.deepcopy(self.document())
        document["chunk_comparisons"][0].update({
            "chunk_2048_tps": 100.0,
            "chunk_8192_tps": 96.9,
            "auto_selected": 8192,
        })
        with self.assertRaisesRegex(ValueError, "auto must retain 2048"):
            validate(document)

    def test_qualifying_8k_requires_8k_auto_selection(self):
        document = copy.deepcopy(self.document())
        document["chunk_comparisons"][0]["auto_selected"] = 2048
        with self.assertRaisesRegex(ValueError, "selected automatically"):
            validate(document)

    def test_chunk_comparison_is_required(self):
        document = copy.deepcopy(self.document())
        document["chunk_comparisons"] = []
        with self.assertRaisesRegex(ValueError, "exactly one uncached-prefix-10k"):
            validate(document)

    def test_only_uncached_prefix_chunk_comparison_is_accepted(self):
        document = copy.deepcopy(self.document())
        document["chunk_comparisons"][0]["name"] = "warm-10k"
        with self.assertRaisesRegex(ValueError, "uncached-prefix-10k comparison"):
            validate(document)

    def test_cold_label_without_page_evidence_is_rejected(self):
        document = copy.deepcopy(self.document())
        case = next(row for row in document["cases"]
                    if row["name"] == "cold-ple-10k+1")
        case["ds4"][0].pop("ple_cold_evidence")
        with self.assertRaisesRegex(ValueError, "evidence object"):
            validate(document)

    def test_cold_fraction_is_computed_from_exact_pages(self):
        document = copy.deepcopy(self.document())
        case = next(row for row in document["cases"]
                    if row["name"] == "cold-ple-10k+1")
        evidence = case["ds4"][0]["ple_cold_evidence"]
        evidence["residency"].update({
            "resident_pages": 6,
            "cold_pages": 94,
            "cold_fraction": 0.94,
        })
        with self.assertRaisesRegex(ValueError, "at least 95%"):
            validate(document)

    def test_cold_startup_flags_are_actual_outcomes(self):
        document = copy.deepcopy(self.document())
        case = next(row for row in document["cases"]
                    if row["name"] == "cold-ple-10k+1")
        case["ds4"][0]["ple_cold_evidence"]["startup_flags"][
            "runtime_readahead_disabled"
        ] = False
        with self.assertRaisesRegex(ValueError, "F_NOCACHE"):
            validate(document)

    def test_cold_phase_must_be_first(self):
        document = copy.deepcopy(self.document())
        document["metadata"]["execution_sequence"].reverse()
        with self.assertRaisesRegex(ValueError, "first execution phase"):
            validate(document)

    def test_cold_request_geometry_is_exact(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "cold-ple-10k+1")["ds4"][0]
        sample["selected_chunk"] = 2048
        with self.assertRaisesRegex(ValueError, "8192 prefill path"):
            validate(document)

    def test_cold_logical_payload_is_exact(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "cold-ple-10k+1")["ds4"][0]
        sample["ple_cold_evidence"]["target"]["logical_bytes"] -= 100
        with self.assertRaisesRegex(ValueError, "16MB logical PLE"):
            validate(document)

    def test_cold_disk_read_delta_covers_logical_payload(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "cold-ple-10k+1")["ds4"][0]
        sample["rusage_v2"]["after"]["diskio_bytesread"] = 15_000_100
        sample["rusage_v2"]["delta"]["diskio_bytesread"] = 15_000_000
        with self.assertRaisesRegex(ValueError, "fewer than 16MB"):
            validate(document)

    def test_rusage_process_lifetime_must_match(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "cold-ple-10k+1")["ds4"][0]
        sample["rusage_v2"]["after"]["proc_start_abstime"] += 1
        with self.assertRaisesRegex(ValueError, "process lifetime"):
            validate(document)

    def test_probe_adjustment_must_be_derived(self):
        document = copy.deepcopy(self.document())
        sample = next(row for row in document["cases"]
                      if row["name"] == "cold-ple-10k+1")["ds4"][0]
        sample["timing"]["probe_adjusted"]["prefill_tps"] += 1.0
        with self.assertRaisesRegex(ValueError, "not derived"):
            validate(document)

    def test_all_phases_share_the_cold_ple_binding(self):
        document = copy.deepcopy(self.document())
        document["metadata"]["chunk_configuration"]["ple_binding"] = copy.deepcopy(
            document["metadata"]["chunk_configuration"]["ple_binding"]
        )
        document["metadata"]["chunk_configuration"]["ple_binding"]["pack"][
            "pack_id"
        ] = "different"
        with self.assertRaisesRegex(ValueError, "descriptor/pack/SSD binding"):
            validate(document)

    def test_local_ssd_backing_is_required(self):
        document = copy.deepcopy(self.document())
        document["metadata"]["cold_ple_configuration"]["ple_binding"][
            "ssd_backing"
        ]["solid_state"] = False
        with self.assertRaisesRegex(ValueError, "solid-state backing"):
            validate(document)


if __name__ == "__main__":
    unittest.main()
