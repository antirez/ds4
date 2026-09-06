#!/usr/bin/env python3

import json
import plistlib
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from qwen4_benchmark import (
    COLD_PHASE_MARKER,
    add_probe_adjusted_timing,
    command_lines,
    new_document,
    parse_ds4_log,
    parse_qwen_mtp_timing,
    prepare_cold_phase_order,
    require_cold_ple_binding,
    require_ds4_mtp_timing,
    rusage_v2_evidence,
    ssd_backing_evidence,
)


class Qwen4BenchmarkTests(unittest.TestCase):
    def test_ds4_log_parser_reports_uncached_chunk_and_rates(self):
        evidence = {
            "schema": "ds4.qwen4.ple-cold-evidence",
            "version": 1,
        }
        result = parse_ds4_log(
            "\n".join((
                "ds4: Qwen PLE cold evidence " + json.dumps(evidence),
                "ds4: Qwen prefill selected chunk=8192 microtile=512 "
                "PLE=SSD/BF16-double-buffer reason=cold-suffix",
                "ds4-server: completion ctx=0..10000:10000 prompt done 20.000s",
                "ds4-server: completion ctx=10000..10500:500 gen=500 "
                "decoding chunk=25.00 t/s avg=20.00 t/s 25.000s",
            )),
            500,
        )
        self.assertEqual(result["cached_tokens"], 0)
        self.assertEqual(result["selected_chunk"], 8192)
        self.assertEqual(result["prefill_tps"], 500.0)
        self.assertEqual(result["decode_tps"], 20.0)
        self.assertEqual(result["ple_cold_evidence"], evidence)

    def test_ds4_log_parser_accepts_one_token_prefill_case(self):
        result = parse_ds4_log(
            "ds4-server: completion ctx=0..50000:50000 prompt done 50.000s",
            1,
        )
        self.assertEqual(result["prefill_tps"], 1000.0)
        self.assertNotIn("decode_tps", result)

    def test_qwen_mtp_parser_aggregates_request_local_cycles(self):
        evidence = parse_qwen_mtp_timing("\n".join((
            "ds4: Qwen MTP timing drafted=4 accepted=3 "
            "target_tokens=4 cycle=12.500 ms verifier=block",
            "ds4: Qwen MTP timing drafted=4 accepted=1 "
            "target_tokens=2 cycle=9.250 ms verifier=block",
            "ds4: Qwen MTP stages snapshot=1.000 ms draft=2.000 ms "
            "verify=3.000 ms commit=4.000 ms path=restore-replay",
            "ds4: Qwen MTP timing drafted=0 accepted=0 "
            "target_tokens=1 cycle=1.000 ms",
        )))
        self.assertEqual(evidence, {
            "schema": "ds4.qwen4.mtp-timing",
            "version": 1,
            "cycles": 3,
            "drafted": 8,
            "max_drafted": 4,
            "accepted": 4,
            "target_tokens": 7,
            "cycle_ms": 22.75,
            "verifier": {"block": 2, "unlabeled": 1},
        })

    def test_qwen_mtp_requirement_accepts_block_verifier(self):
        evidence = parse_qwen_mtp_timing(
            "ds4: Qwen MTP timing drafted=4 accepted=3 "
            "target_tokens=4 cycle=12.500 ms verifier=block"
        )
        require_ds4_mtp_timing(evidence, 4)

    def test_qwen_mtp_requirement_rejects_sequential_verifier(self):
        evidence = parse_qwen_mtp_timing(
            "ds4: Qwen MTP timing drafted=4 accepted=3 "
            "target_tokens=4 cycle=12.500 ms verifier=sequential"
        )
        with self.assertRaisesRegex(RuntimeError, "block verifier"):
            require_ds4_mtp_timing(evidence, 4)

    def test_qwen_mtp_parser_rejects_malformed_evidence(self):
        with self.assertRaisesRegex(ValueError, "malformed"):
            parse_qwen_mtp_timing(
                "ds4: Qwen MTP timing drafted=4 accepted=? target_tokens=5"
            )
        with self.assertRaisesRegex(ValueError, "accepted exceeds drafted"):
            parse_qwen_mtp_timing(
                "ds4: Qwen MTP timing drafted=1 accepted=2 "
                "target_tokens=3 cycle=1.000 ms verifier=block"
            )
        with self.assertRaisesRegex(ValueError, "1 \+ accepted"):
            parse_qwen_mtp_timing(
                "ds4: Qwen MTP timing drafted=4 accepted=2 "
                "target_tokens=4 cycle=1.000 ms verifier=block"
            )

    def test_mtp_command_enables_draft_four_and_timing_only_for_mtp(self):
        args = SimpleNamespace(
            ds4_server=Path("ds4-server"),
            ds4_model=Path("model.gguf"),
            ple=Path("ple.gguf"),
            ds4_mtp=Path("mtp.gguf"),
            ctx=101_024,
            ds4_port=18123,
            prefill_chunk="auto",
            startup_timeout=30.0,
            ds4_extra=[],
        )
        mtp_ds4 = command_lines(args, True, Path("logs"))
        draft_index = mtp_ds4.command.index("--mtp-draft")
        self.assertEqual(mtp_ds4.command[draft_index + 1], "4")
        self.assertIn("--mtp-model", mtp_ds4.command)
        self.assertIn("--mtp-timing", mtp_ds4.command)

        core_ds4 = command_lines(args, False, Path("logs"))
        for flag in ("--mtp-model", "--mtp", "--mtp-draft", "--mtp-timing"):
            self.assertNotIn(flag, core_ds4.command)

    def test_managed_mtp_flags_cannot_be_overridden_by_extras(self):
        args = SimpleNamespace(
            ds4_server=Path("ds4-server"), ds4_model=Path("model.gguf"),
            ple=Path("ple.gguf"), ds4_mtp=Path("mtp.gguf"),
            ctx=101_024, ds4_port=18123,
            prefill_chunk="auto", startup_timeout=30.0,
            ds4_extra=["--mtp-draft"],
        )
        with self.assertRaisesRegex(ValueError, "harness-managed MTP flag"):
            command_lines(args, False, Path("logs"))

    def test_rusage_v2_delta_is_derived_from_snapshots(self):
        evidence = rusage_v2_evidence(
            {
                "proc_start_abstime": 123,
                "pageins": 4,
                "diskio_bytesread": 10,
                "diskio_byteswritten": 2,
                "resident_size": 100,
                "phys_footprint": 120,
            },
            {
                "proc_start_abstime": 123,
                "pageins": 7,
                "diskio_bytesread": 90,
                "diskio_byteswritten": 2,
                "resident_size": 110,
                "phys_footprint": 130,
            },
        )
        self.assertEqual(evidence["flavor"], 2)
        self.assertEqual(evidence["delta"]["pageins"], 3)
        self.assertEqual(evidence["delta"]["diskio_bytesread"], 80)
        self.assertEqual(evidence["before"]["proc_start_abstime"], 123)

    def test_rusage_v2_rejects_a_different_process_lifetime(self):
        before = {
            "proc_start_abstime": 123,
            "pageins": 4,
            "diskio_bytesread": 10,
            "diskio_byteswritten": 0,
        }
        after = {**before, "proc_start_abstime": 124}
        with self.assertRaisesRegex(ValueError, "process lifetimes"):
            rusage_v2_evidence(before, after)

    def test_probe_adjusted_timing_preserves_raw_values(self):
        metrics = {
            "prompt_tokens": 10_000,
            "cached_tokens": 0,
            "prefill_seconds": 20.0,
            "prefill_tps": 500.0,
            "first_token_ms": 20_100.0,
            "request_elapsed_ms": 20_200.0,
        }
        add_probe_adjusted_timing(metrics, {"probe_ms": 100.0})
        self.assertEqual(metrics["timing"]["raw_observed"]["prefill_tps"], 500.0)
        self.assertAlmostEqual(
            metrics["timing"]["probe_adjusted"]["prefill_tps"],
            10_000 / 19.9,
        )

    @mock.patch("qwen4_benchmark.subprocess.run")
    def test_external_local_ssd_is_accepted(self, run):
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0,
                stdout=(
                    "Filesystem 512-blocks Used Available Capacity Mounted on\n"
                    "/dev/disk9s1 100 1 99 1% /Volumes/Qwen PLE\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [], 0,
                stdout=plistlib.dumps({
                    "DeviceNode": "/dev/disk9s1",
                    "DeviceIdentifier": "disk9s1",
                    "ParentWholeDisk": "disk9",
                    "BusProtocol": "USB",
                    "Internal": False,
                    "SolidState": True,
                }),
                stderr=b"",
            ),
        ]
        evidence = ssd_backing_evidence(Path("/Volumes/Qwen PLE/table"))
        self.assertTrue(evidence["solid_state"])
        self.assertFalse(evidence["internal"])
        self.assertEqual(evidence["mount_point"], "/Volumes/Qwen PLE")

    @mock.patch("qwen4_benchmark.subprocess.run")
    def test_rotational_backing_is_rejected(self, run):
        run.side_effect = [
            subprocess.CompletedProcess(
                [], 0,
                stdout=(
                    "Filesystem 512-blocks Used Available Capacity Mounted on\n"
                    "/dev/disk9s1 100 1 99 1% /Volumes/PLE\n"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                [], 0,
                stdout=plistlib.dumps({
                    "DeviceNode": "/dev/disk9s1",
                    "SolidState": False,
                }),
                stderr=b"",
            ),
        ]
        with self.assertRaisesRegex(ValueError, "not reported as solid state"):
            ssd_backing_evidence(Path("/Volumes/PLE/table"))

    def test_cold_phase_is_reserved_before_any_measurement(self):
        document = new_document()
        self.assertTrue(prepare_cold_phase_order(document))
        self.assertEqual(
            document["metadata"]["execution_sequence"],
            [COLD_PHASE_MARKER],
        )
        # A crash after reserving the phase may safely restart the fresh
        # DS4-only process because no sample has run yet.
        self.assertTrue(prepare_cold_phase_order(document))

    def test_cold_phase_cannot_be_backfilled_after_a_warmup(self):
        document = new_document()
        document["cases"].append({
            "name": "prefill-10k+1",
            "kind": "prefill",
            "ds4": [{"prompt_id": "already-ran"}],
        })
        with self.assertRaisesRegex(ValueError, "before every warmup"):
            prepare_cold_phase_order(document)

    def test_every_phase_must_match_the_cold_binding(self):
        binding = {"schema": "binding", "descriptor_identity": {"inode": 7}}
        document = new_document()
        document["metadata"]["cold_ple_configuration"] = {
            "ple_binding": binding,
        }
        require_cold_ple_binding(document, binding)
        with self.assertRaisesRegex(ValueError, "differs from cold evidence"):
            require_cold_ple_binding(document, {**binding, "version": 2})


if __name__ == "__main__":
    unittest.main()
