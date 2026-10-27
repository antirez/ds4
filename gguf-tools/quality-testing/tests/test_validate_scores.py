"""Fault injection and independently calculated comparator checks; no GPU."""

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "validate_scores.py"
SPEC = importlib.util.spec_from_file_location("validate_scores", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

# Literal source-format fixtures keep parser/comparator bugs independent of a
# fixture generator implementing the same arithmetic as the validator.
HEADER = "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\tapi_ref_tokens\tapi_target_tokens\tapi_target_mae\tapi_target_mean_delta\tapi_top_items\tapi_top_mapped\tapi_top_coverage\tapi_top1_count\tapi_top1_match\tapi_top1_rate\tapi_topn_ref\tapi_topn_hit\tapi_topn_recall\tapi_top_logprob_count\tapi_top_mae\tapi_top_mean_delta\tapi_pair_total\tapi_pair_agree\tapi_pair_rate"
ROWS = [
    "case_000\t10\t4\t4.000000000\t1.000000000\t1\t2\t4\t4\t0.100000000\t-0.050000000\t8\t8\t1.000000000\t4\t3\t0.750000000\t8\t6\t0.750000000\t8\t0.300000000\t-0.100000000\t4\t3\t0.750000000",
    "case_001\t12\t2\t4.000000000\t2.000000000\t0\t0\t2\t2\t0.400000000\t0.200000000\t6\t6\t1.000000000\t2\t1\t0.500000000\t6\t3\t0.500000000\t6\t0.600000000\t0.200000000\t6\t3\t0.500000000",
]
MANIFEST = "# id\tprompt_file\tcontinuation_file\tresponse_file\ncase_000\tp0.txt\tc0.txt\tr0.json\ncase_001\tp1.txt\tc1.txt\tr1.json\n"


class ScoresTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.tsv"
        self.baseline = self.root / "baseline.tsv"
        self.candidate = self.root / "candidate.tsv"
        self.manifest.write_text(MANIFEST)
        self.valid = HEADER + "\n" + "\n".join(ROWS) + "\n"
        self.baseline.write_text(self.valid)
        self.candidate.write_text(self.valid)

    def compare(self):
        return validator.compare(self.manifest, self.baseline, self.candidate)

    def changed(self, **fields):
        rows = [line.split("\t") for line in ROWS]
        for field, value in fields.items():
            rows[0][HEADER.split("\t").index(field)] = value
        return HEADER + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"

    def reject(self, text, message=None):
        self.candidate.write_text(text)
        with self.assertRaises(validator.ValidationError) as caught:
            self.compare()
        if message:
            self.assertIn(message, str(caught.exception))

    def cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(self.manifest),
                               "--baseline", str(self.baseline), "--candidate", str(self.candidate),
                               *args], capture_output=True, text=True)

    def test_complete_identity_and_independent_weighted_metrics(self):
        result = self.compare()
        self.assertTrue(result["identical_reported_values"])
        self.assertEqual(result["cases"], 2)
        old = result["baseline"]
        self.assertEqual(old["counts"]["target_tokens"], 6)
        self.assertEqual(old["nll"], Decimal(8))
        self.assertAlmostEqual(float(old["avg_nll"]), 8 / 6)
        self.assertEqual(old["api_target_mae"], Decimal("0.2"))
        self.assertAlmostEqual(float(old["api_target_mean_delta"]), 0.2 / 6)
        self.assertAlmostEqual(float(old["api_top_mae"]), 6 / 14)
        self.assertAlmostEqual(float(old["api_top1_rate"]), 4 / 6)
        self.assertAlmostEqual(float(old["api_pair_rate"]), 6 / 10)

    def test_paired_difference_sign_denominator_and_case_identity(self):
        self.candidate.write_text(self.changed(nll="4.400000000", avg_nll="1.100000000"))
        result = self.compare()
        self.assertFalse(result["identical_reported_values"])
        self.assertEqual(result["delta_candidate_minus_baseline"]["nll"], Decimal("0.4"))
        self.assertAlmostEqual(float(result["delta_candidate_minus_baseline"]["avg_nll"]), .4 / 6)
        self.assertEqual(result["changed_cases"], [{"id": "case_000", "fields": ["nll", "avg_nll"]}])
        self.assertEqual(result["per_case"][0]["delta_nll"], Decimal("0.4"))
        self.assertEqual(result["per_case"][1]["delta_nll"], 0)

    def test_pairing_uses_ids_not_row_positions(self):
        self.candidate.write_text(HEADER + "\n" + "\n".join(reversed(ROWS)) + "\n")
        self.assertTrue(self.compare()["identical_reported_values"])

    def test_exact_complete_manifest_coverage(self):
        for rows in (ROWS[:1], ROWS[1:], [], ROWS + [ROWS[0]], [ROWS[0], ROWS[0]],
                     ROWS + [ROWS[0].replace("case_000", "case_999")]):
            with self.subTest(rows=rows):
                self.reject(HEADER + "\n" + "\n".join(rows) + "\n")

    def test_rejects_all_empty_null_nonfinite_numeric_fields(self):
        for field in HEADER.split("\t")[1:]:
            for value in ("", "null", "None", "NaN", "nan", "inf", "-Inf", "Infinity", "1e999", " 1 "):
                with self.subTest(field=field, value=value):
                    self.reject(self.changed(**{field: value}))

    def test_rejects_malformed_and_negative_counts(self):
        for field in validator.COUNT_COLUMNS:
            for value in ("1.0", "-1", "+1", "01", "1e0", "1_0", "١", str(2**63)):
                with self.subTest(field=field, value=value):
                    self.reject(self.changed(**{field: value}))

    def test_rejects_zero_denominators(self):
        for field in validator.DENOMINATORS:
            with self.subTest(field=field):
                self.reject(self.changed(**{field: "0"}))

    def test_finite_looking_decimal_double_overflow_fails(self):
        self.reject(self.changed(nll="2" + "0" * 308 + ".000000000"), "not finite")

    def test_schema_truncation_and_trailing_junk(self):
        for text in ("", HEADER + "\n", self.valid[:-1], self.valid[:-8],
                     self.valid.replace("\t0.750000000\n", "\n", 1),
                     self.valid.replace("\t0.750000000\n", "\t0.750000000\textra\n", 1),
                     self.valid + "\n", self.valid + "# done\n", self.valid + "\x00\n",
                     self.valid.replace("\tnll\t", "\tavg_nll\t", 1),
                     self.valid.replace("\tnll\tavg_nll", "\tavg_nll\tnll", 1),
                     self.valid.replace("4.000000000", "4.00000000", 1)):
            with self.subTest(text=text[:80]):
                self.reject(text)

    def test_manifest_corruption(self):
        for text in ("", MANIFEST.splitlines()[0] + "\n", MANIFEST[:-1],
                     MANIFEST + MANIFEST.splitlines()[1] + "\n",
                     MANIFEST.replace("\tr0.json", ""),
                     MANIFEST.replace("\tr0.json", "\t"),
                     MANIFEST.replace("\tr0.json", "\tnull"),
                     MANIFEST.replace("case_001", "case_999"),
                     MANIFEST.replace("# id", "id")):
            with self.subTest(text=text):
                self.manifest.write_text(text)
                with self.assertRaises(validator.ValidationError):
                    self.compare()

    def test_internal_consistency_faults(self):
        mutations = [dict(nll="-1.000000000"), dict(avg_nll="1.010000000"),
                     dict(first_match="2"), dict(first_match="0"), dict(greedy_lcp="0"),
                     dict(greedy_lcp="5"), dict(api_ref_tokens="3"), dict(api_target_tokens="3"),
                     dict(api_top_items="7"), dict(api_topn_ref="7"), dict(api_top_logprob_count="7"),
                     dict(api_top1_count="5"), dict(api_pair_total="253"),
                     dict(api_top1_match="5"), dict(api_topn_hit="9"), dict(api_pair_agree="5"),
                     dict(api_top_coverage="0.900000000"), dict(api_top1_rate="1.100000000"),
                     dict(api_topn_recall="0.700000000"), dict(api_pair_rate="-0.100000000"),
                     dict(api_target_mae="-0.100000000"), dict(api_target_mean_delta="0.200000000"),
                     dict(api_top_mean_delta="0.400000000")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reject(self.changed(**mutation))

    def test_valid_individual_rows_with_mismatched_pair_denominators_fail(self):
        mutations = [dict(prompt_tokens="11"),
                     dict(target_tokens="8", api_ref_tokens="8", api_target_tokens="8",
                          nll="8.000000000"),
                     dict(api_top_items="16", api_top_coverage="0.500000000"),
                     dict(api_top_mapped="7", api_topn_ref="7", api_top_logprob_count="7",
                          api_top_coverage="0.875000000", api_topn_recall="0.857142857"),
                     dict(api_top1_count="3", api_top1_rate="1.000000000"),
                     dict(api_pair_total="5", api_pair_rate="0.600000000")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.candidate.write_text(self.changed(**mutation))
                ids, _ = validator.manifest_ids(self.manifest)
                validator.load_scores(self.candidate, ids)
                with self.assertRaisesRegex(validator.ValidationError, "baseline/candidate"):
                    self.compare()

    def test_ratio_rounding_bound_is_serialization_only(self):
        self.candidate.write_text(self.changed(api_top1_count="3", api_top1_match="1", api_top1_rate="0.333333333"))
        ids, _ = validator.manifest_ids(self.manifest)
        validator.load_scores(self.candidate, ids)
        self.reject(self.changed(api_top1_count="3", api_top1_match="1", api_top1_rate="0.333333334"))

    def test_cli_strict_difference_fails_but_metrics_remain_reviewable(self):
        success = self.cli("--strict-identical")
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(success.stdout)["requested_check_passed"])
        self.candidate.write_text(self.changed(nll="4.400000000", avg_nll="1.100000000"))
        failure = self.cli("--strict-identical")
        self.assertNotEqual(failure.returncode, 0)
        self.assertFalse(json.loads(failure.stdout)["requested_check_passed"])
        self.assertIn("FAIL", failure.stderr)
        diagnostic = self.cli()
        self.assertEqual(diagnostic.returncode, 0)
        self.assertEqual(json.loads(diagnostic.stdout)["quality_verdict"], "not_decided_by_this_tool")

    def test_cli_malformed_inputs_do_not_emit_success_report(self):
        self.candidate.write_text(self.changed(nll="NaN"))
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("FAIL", result.stderr)
        self.candidate.unlink()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_baseline_is_validated_too(self):
        self.baseline.write_text(self.changed(api_top_mae="NaN"))
        with self.assertRaises(validator.ValidationError):
            self.compare()

    def test_same_aggregate_can_hide_per_case_drift_and_strict_catches_it(self):
        lines = self.valid.splitlines()
        lines[1] = lines[1].replace("4.000000000\t1.000000000", "4.400000000\t1.100000000")
        lines[2] = lines[2].replace("4.000000000\t2.000000000", "3.600000000\t1.800000000")
        self.candidate.write_text("\n".join(lines) + "\n")
        result = self.compare()
        self.assertEqual(result["delta_candidate_minus_baseline"]["nll"], 0)
        self.assertEqual(len(result["changed_cases"]), 2)
        self.assertNotEqual(self.cli("--strict-identical").returncode, 0)


if __name__ == "__main__":
    unittest.main()
