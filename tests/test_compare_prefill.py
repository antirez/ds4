# SPDX-License-Identifier: MIT
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "speed-bench/compare_prefill.py"
SPEC = importlib.util.spec_from_file_location("compare_prefill", MODULE)
compare_prefill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_prefill)


class ComparePrefillTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("ctx_tokens,prefill_tokens,prefill_tps,gen_tokens\n" + rows)
        return path

    def test_total_uses_elapsed_time_and_keeps_each_run(self):
        a = [self.write("a1.csv", "100,100,100,0\n1000,900,10,0\n"),
             self.write("a2.csv", "100,100,200,0\n1000,900,20,0\n")]
        b = [self.write("b1.csv", "100,100,200,0\n1000,900,20,0\n"),
             self.write("b2.csv", "100,100,400,0\n1000,900,40,0\n")]
        report = compare_prefill.compare(a, b)
        total = report["aggregate"]
        self.assertAlmostEqual(total["baseline"]["runs_tps"][0], 1000 / 91)
        self.assertAlmostEqual(total["baseline"]["median_tps"], 1500 / 91)
        self.assertAlmostEqual(total["observed_speedup_percent"], 100)
        self.assertTrue(total["target_met_by_medians"])
        self.assertEqual(report["frontiers"][1]["prefill_tokens"], 900)
        report = compare_prefill.compare(b, a)
        self.assertAlmostEqual(report["aggregate"]["observed_speedup_percent"], -50)
        self.assertFalse(report["aggregate"]["target_met_by_medians"])

    def test_rejects_invalid_or_incomparable_data(self):
        good = [self.write("a1.csv", "100,100,100,0\n"),
                self.write("a2.csv", "100,100,101,0\n")]
        for i, rows in enumerate(("", "100,100,0,0\n", "100,100,nan,0\n",
                                  "100,100,inf,0\n", "100,100,100,1\n",
                                  "100,99,100,0\n", "101,101,100,0\n",
                                  "100,100,100,0\n100,100,100,0\n")):
            b = [self.write(f"b{i}1.csv", rows), self.write(f"b{i}2.csv", rows)]
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                compare_prefill.compare(good, b)
        with self.assertRaises(ValueError):
            compare_prefill.compare(good, good)
        with self.assertRaises(ValueError):
            compare_prefill.compare(good, good[:1])


if __name__ == "__main__":
    unittest.main()
