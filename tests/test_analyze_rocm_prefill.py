"""Regression checks for the stage profiler's attribution boundaries."""

import contextlib
import importlib.util
import io
import math
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/analyze_rocm_prefill.py"
SPEC = importlib.util.spec_from_file_location("analyze_rocm_prefill", SCRIPT)
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_nested_q_is_separate_and_first_frontier_decode_are_excluded(self):
        lines = [
            "ds4-bench: context buffers 123 MiB (ctx=32768, backend=rocm, prefill_chunk=4096, raw_kv_rows=8192)",
            "ds4: metal layer stage part=attn layer=0 pos=0 tokens=4096 q_path=999.000 ms",
            "ds4: metal layer stage part=attn layer=0 pos=8192 tokens=4096 q_path=20.000 ms",
            "ds4: metal Q path stage layer=0 pos=8192 tokens=4096 pre_q=1.000 ms",
            "ds4: metal Q path stage layer=0 pos=8192 tokens=4096 q_b=19.000 ms",
            "ds4: metal layer stage part=attn layer=0 pos=8192 tokens=4096 output_proj=10.000 ms",
            "ds4: metal layer stage part=ffn layer=0 pos=8192 tokens=4096 routed=70.000 ms",
            "ds4: metal layer stage part=decode layer=0 pos=16384 tokens=1 q_path=555.000 ms",
            "ds4: metal layer stage part=output layer=43 pos=16384 tokens=1 logits=555.000 ms",
            "ds4: metal Q path stage layer=0 pos=16384 tokens=1 q_b=555.000 ms",
        ]
        result = profile.read_profile(lines, 8192)
        timings, layers, batches, caps = result
        self.assertEqual(sum(sum(v) for (kind, _), v in timings.items() if kind == "layer"), 100)
        self.assertEqual(timings[("Q", "q_b")], [19])
        self.assertEqual(timings[("Q", "pre_q")], [1])
        self.assertEqual((layers, batches, caps), ({0}, {(8192, 4096)}, {4096}))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            profile.render_profile("fixture", result, 8)
        self.assertIn("layer stages: 100.000 ms", output.getvalue())
        self.assertIn("Q stages: 20.000 ms", output.getvalue())
        self.assertIn("includes pre_q setup", output.getvalue())
        self.assertIn("74.1%", output.getvalue())
        self.assertIn("does not isolate attn_output_b", output.getvalue())

    def test_negative_and_nonfinite_times_fail(self):
        for value in ("nan", "inf", "-1", "bad"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.read_profile([
                    f"ds4: metal layer stage part=attn layer=0 pos=8192 tokens=4096 q_path={value} ms"
                ], 8192)

    def test_empty_profile_is_not_a_zero_cost_success(self):
        with self.assertRaises(ValueError):
            profile.render_profile("empty", profile.read_profile([], 8192), 8)

    def test_tps_and_time_targets_are_reciprocal(self):
        self.assertAlmostEqual(profile.required_saving(8), 0.074074074074074)
        for value in (0, -1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile.required_saving(value)


if __name__ == "__main__":
    unittest.main()
