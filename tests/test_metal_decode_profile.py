#!/usr/bin/env python3
"""Fixture tests for strict Metal timeline attribution (no GPU required)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench/metal_decode_profile.py"
spec = importlib.util.spec_from_file_location("metal_decode_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
HEADER = "# ds4 encoder timeline; slide=0x1000 pid=701\n"
PAIR = "kernel_mul_mv_f16_f32_pair_compressor_store_4"
QUAD = "kernel_mul_mv_f16_f32_quad_compressor_store_4"
COMPOUND = "kernel_dsv4_qkv_pair_quad_compressor_store_q8_0"
ORDINARY = "kernel_mul_mv_f16_f32_pair_4"
HC = "kernel_dsv4_shared_down_hc_expand4_q8_0"


def encoder(seq, idx, kernel=PAIR, start=None, end=None, dispatches=1):
    start = seq * 100000 + idx * 10000 + 1000 if start is None else start
    end = start + 5000 if end is None else end
    duration = (end-start)/1000 if end >= start else -1
    return f"E {seq} {idx} {start} {end} {duration:.2f} 0.00 0x1234 {dispatches} 32x1x1 32x8x1 {kernel}\n"


def batch(seq=1, phase="decode", position=128, names=(PAIR,), dropped=0):
    return (f"B {seq} {len(names)} {seq*100000} {seq*100000+90000}\n"
            f"C {seq} {phase} {position} {dropped}\n" +
            "".join(encoder(seq,idx,name) for idx,name in enumerate(names)))


def csv_text(rows):
    return "phase,step,position,token,seconds,select_seconds,eval_seconds\n" + "\n".join(rows) + "\n"


class ProfileTests(unittest.TestCase):
    def parse(self, text):
        return profile.parse_timeline(HEADER + text)

    def report(self, text, **kwargs):
        return profile.summarize(self.parse(text), **kwargs)

    def codes(self, report):
        return {warning["code"] for warning in report["warnings"]}

    def test_completion_order_and_phase_selection(self):
        timeline = self.parse(batch(3,"decode",129,(QUAD,HC)) +
                              batch(1,"prefill",128,(ORDINARY,)) + batch(2))
        report = profile.summarize(timeline)
        self.assertEqual([item["seq"] for item in timeline["batches"]], [1,2,3])
        self.assertEqual(report["summary"]["selected_batches"],2)
        self.assertEqual(report["summary"]["selected_encoder_passes"],3)
        self.assertEqual(report["summary"]["summed_valid_traced_batch_gpu_ns"],180000)
        self.assertEqual(report["summary"]["summed_valid_encoder_gpu_ns_in_valid_traced_batches"],15000)
        self.assertNotIn(ORDINARY,{row["name"] for row in report["kernels"]})
        self.assertIn("not full GPU coverage", report["gpu_denominator_description"])
        self.assertEqual(len(profile.summarize(timeline,phase="all")["kernels"]),4)

    def test_compound_is_not_assigned_to_f16(self):
        report = self.report(batch(names=(PAIR,QUAD,COMPOUND,ORDINARY,HC)))
        categories = {row["name"]:row for row in report["categories"]}
        self.assertEqual(categories["compound_q8_qkv_plus_f16"]["gpu_ns_in_valid_traced_batches"],5000)
        self.assertEqual(categories["f16_compressor_pair"]["gpu_ns_in_valid_traced_batches"],5000)
        self.assertEqual(categories["f16_compressor_quad"]["gpu_ns_in_valid_traced_batches"],5000)
        self.assertIn("ordinary_f16_pair_candidate",categories)
        self.assertIn("q8_hc",categories)
        self.assertNotIn("estimated_f16",json.dumps(report))
        self.assertIn("none is assigned",profile.render_report(report))

    def test_mixed_names_never_produce_per_kernel_counts(self):
        report = self.report(batch(names=(PAIR+"+"+HC,)))
        self.assertEqual(report["kernels"],[])
        self.assertEqual(report["summary"]["ambiguous_encoder_passes"],1)
        self.assertIn("ambiguous_kernel",self.codes(report))
        self.assertEqual(report["raw_timeline"]["batches"][0]["encoders"][0]["kernel"],PAIR+"+"+HC)

    def test_multi_dispatch_single_kernel_counts_exact_but_no_latency_guess(self):
        text = batch().replace("0x1234 1 ","0x1234 7 ")
        report = self.report(text)
        self.assertEqual(report["kernels"][0]["observed_dispatches"],7)
        self.assertEqual(report["kernels"][0]["encoder_passes"],1)
        self.assertEqual(report["kernels"][0]["gpu_ns_in_valid_traced_batches"],5000)
        self.assertIn("multiple_dispatches",self.codes(report))
        self.assertNotIn("per_dispatch",json.dumps(report))

    def test_unknown_truncated_and_no_dispatch(self):
        for name in ("?", "unknown", "kernel_"+"x"*72):
            with self.subTest(name=name):
                self.assertEqual(self.report(batch(names=(name,)))["kernels"],[])
        self.assertIn("other_kernel_names",self.codes(self.report(batch(names=("kernel_other",)))))
        self.assertEqual(self.report(batch().replace("0x1234 1 ","0x1234 0 "))["kernels"],[])
        blit = self.report(batch(names=("blit",)).replace("0x1234 1 ","0x1234 0 "))
        self.assertEqual(blit["categories"][0]["name"],"blit")

    def test_invalid_counters_preserve_counts_not_timing(self):
        for start,end in ((0,200),(100,0),(100,100),(200,100),
                          (100,profile.U64_MAX),(profile.U64_MAX,profile.U64_MAX)):
            with self.subTest(start=start,end=end):
                text = "B 1 1 100 90000\nC 1 decode 128 0\n"+encoder(1,0,start=start,end=end)
                report = self.report(text)
                self.assertEqual(report["kernels"][0]["observed_dispatches"],1)
                self.assertEqual(report["kernels"][0]["gpu_ns_in_valid_traced_batches"],0)
                self.assertEqual(report["kernels"][0]["invalid_counter_passes"],1)
                self.assertIn("invalid_encoder_timestamps",self.codes(report))
                json.dumps(report,allow_nan=False)

    def test_nonmonotonic_encoder_order_is_invalid(self):
        text = "B 1 2 100 90000\nC 1 decode 128 0\n"
        text += encoder(1,0,start=1000,end=6000)+encoder(1,1,start=5000,end=8000)
        report = self.report(text)
        self.assertEqual(report["kernels"][0]["valid_counter_passes"],1)
        self.assertEqual(report["kernels"][0]["invalid_counter_passes"],1)
        self.assertEqual(report["kernels"][0]["gpu_ns_in_valid_traced_batches"],5000)

    def test_reported_counter_duration_mismatch(self):
        report = self.report(batch().replace("5.00 0.00","9.00 0.00"))
        self.assertEqual(report["kernels"][0]["invalid_counter_passes"],1)
        self.assertIn("reported_duration_mismatch",report["raw_timeline"]["batches"][0]["encoders"][0]["invalid_timing_reasons"])

    def test_missing_timing_prints_na_and_counter_counts(self):
        report=self.report(batch().replace("5.00 0.00","9.00 0.00"))
        line=next(line for line in profile.render_report(report).splitlines() if line.startswith("  "+PAIR+":"))
        self.assertIn("GPU n/a",line)
        self.assertIn("valid=0 invalid=1",line)
        self.assertNotIn("0.000000 ms",line)
        self.assertIsNone(report["kernels"][0]["percent_of_summed_valid_traced_batch_gpu_time"])

    def test_warning_report_is_compact_and_phase_scoped(self):
        timeline=self.parse(batch(1,"prefill",128,names=("?",)*30)+batch(2,names=("?",)*4))
        report=profile.summarize(timeline)
        rendered=profile.render_report(report)
        self.assertIn("Selected phase: 4 warning(s)",rendered)
        self.assertIn("Outside selected phase: 30 warning(s)",rendered)
        self.assertEqual(rendered.count("[ambiguous_kernel]"),2)
        self.assertEqual(len(report["warnings"]),34)
        self.assertEqual(sum(warning["scope"]=="selected_phase" for warning in report["warnings"]),4)

    def test_invalid_batch_excludes_denominator_and_share_numerator(self):
        for start,end in ((0,10),(10,0),(10,10),(20,10),(10,profile.U64_MAX)):
            with self.subTest(start=start,end=end):
                report = self.report(batch().replace("B 1 1 100000 190000",f"B 1 1 {start} {end}"))
                self.assertEqual(report["summary"]["summed_valid_traced_batch_gpu_ns"],0)
                self.assertEqual(report["kernels"][0]["gpu_ns_in_valid_traced_batches"],0)
                self.assertEqual(report["kernels"][0]["valid_counter_gpu_ns"],5000)
                self.assertIsNone(report["kernels"][0]["percent_of_summed_valid_traced_batch_gpu_time"])

    def test_dropped_passes_are_explicit(self):
        report = self.report(batch(dropped=3))
        self.assertEqual(report["summary"]["dropped_encoders"],3)
        self.assertIn("dropped_encoders",self.codes(report))
        self.assertEqual(report["kernels"][0]["observed_dispatches"],1)

    def test_duplicate_records_and_concatenated_runs_rejected(self):
        good=batch()
        for text in (good+good,good+"C 1 decode 128 0\n",good+encoder(1,0),good+HEADER+batch(2)):
            with self.subTest(text=text):
                with self.assertRaises(profile.ProfileError):self.parse(text)
        with self.assertRaises(profile.ProfileError):
            profile.parse_timeline(HEADER+good+HEADER+batch(2),legacy_all_phases=True)

    def test_missing_context_batch_sequence_and_encoder_records_rejected(self):
        for text in (batch().replace("C 1 decode 128 0\n",""),batch()+"C 2 decode 129 0\n",
                     batch()+encoder(2,0),batch(2),batch()+batch(3),
                     batch().replace("B 1 1","B 1 2"),batch().replace("E 1 0","E 1 1")):
            with self.subTest(text=text):
                with self.assertRaises(profile.ProfileError):self.parse(text)
        with self.assertRaises(profile.ProfileError):profile.parse_timeline(batch())

    def test_legacy_requires_explicit_all_phase_mode(self):
        text = batch(2).replace("C 2 decode 128 0\n","").replace("B 2 1","B 2 2")
        timeline = profile.parse_timeline(text,legacy_all_phases=True)
        with self.assertRaises(profile.ProfileError):profile.summarize(timeline)
        report=profile.summarize(timeline,phase="all")
        self.assertEqual(report["per_token"],[])
        self.assertIsNone(report["summary"]["dropped_encoders"])
        self.assertTrue({"legacy_all_phases","missing_context","incomplete_encoder_records","missing_batch_sequences","missing_run_header"}.issubset(self.codes(report)))

    def test_context_and_structural_metadata_invalid(self):
        for old,new in (("decode 128","decode -1"),("decode 128","no_phase 128"),
                        ("32x1x1","32x1"),("0x1234","notaddress"),
                        ("5.00 0.00","nan 0.00"),("decode 128 0","decode 128 -1")):
            with self.subTest(new=new):
                with self.assertRaises(profile.ProfileError):self.parse(batch().replace(old,new))
        with self.assertRaises(profile.ProfileError):self.parse("bad record\n")

    def test_cpu_wall_steps_are_explicit_and_joined_by_absolute_position(self):
        steps=profile.parse_steps(csv_text(["prefill,-1,128,-1,1,0,1",
            "decode,0,128,42,0.2,0.03,0.16","decode,1,129,84,0.3,0.04,0.25"]))
        report=self.report(batch()+batch(2,"selection",128,("kernel_argmax",))+batch(3,"decode",129),steps=steps)
        wall=report["cpu_wall_steps"]
        self.assertEqual(wall["row_count"],2)
        self.assertAlmostEqual(wall["summed_step_seconds"],0.5)
        self.assertAlmostEqual(wall["summed_select_seconds"],0.07)
        self.assertEqual([row["position"] for row in report["per_token"]],[128,129])
        self.assertEqual(report["per_token"][1]["traced_batch_sequences"],[3])
        self.assertNotIn("wall_percent",json.dumps(report))
        self.assertIn("position=128",profile.render_report(report))

    def test_missing_step_row_disables_token_estimates(self):
        steps=profile.parse_steps(csv_text(["decode,0,127,42,0.2,0.03,0.16"]))
        report=self.report(batch(),steps=steps)
        self.assertEqual(report["per_token"],[])
        self.assertIn("missing_step_rows",self.codes(report))

    def test_csv_step_without_traced_batches_is_not_zero_gpu_time(self):
        steps=profile.parse_steps(csv_text(["decode,0,128,42,0.2,0.03,0.16",
            "decode,1,129,43,0.2,0.03,0.16"]))
        report=self.report(batch(),steps=steps)
        self.assertEqual(len(report["per_token"]),1)
        self.assertEqual(report["per_token"][0]["position"],128)
        self.assertIn("untraced_step_rows",self.codes(report))

    def test_csv_bad_numbers_duplicates_and_shape_rejected(self):
        good="decode,0,128,42,0.2,0.03,0.16"
        for text in (csv_text([good,good]),csv_text([good.replace("0.2","nan")]),
                     csv_text([good.replace("0.2","-1")]),csv_text([good.replace("0.2","0.1")]),
                     csv_text([good+",extra"]),"phase,seconds\ndecode,1\n",
                     csv_text([good.replace("decode","unknown")])):
            with self.subTest(text=text):
                with self.assertRaises(profile.ProfileError):profile.parse_steps(text)

    def test_cli_default_json_and_errors(self):
        with tempfile.TemporaryDirectory(prefix="ds4-profile-test-") as directory:
            path=Path(directory)/"timeline.log"; output=Path(directory)/"report.json"
            path.write_text(HEADER+batch())
            result=subprocess.run([sys.executable,str(SCRIPT),str(path),"--json",str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("phase: decode",result.stdout)
            self.assertEqual(json.loads(output.read_text())["phase"],"decode")
            result=subprocess.run([sys.executable,str(SCRIPT),str(path),"--legacy-all-phases"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("requires --phase all",result.stderr)
            path.write_text(HEADER+batch().replace("C 1 decode 128 0\n",""))
            result=subprocess.run([sys.executable,str(SCRIPT),str(path)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("missing C",result.stderr)


if __name__ == "__main__":unittest.main()
