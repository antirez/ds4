#!/usr/bin/env python3
"""Source-contract checks for nested exact-span shared-row batching.

These checks deliberately require no GPU/model. Runtime arithmetic and state
identity remain ROCm/Metal acceptance work documented in the self-review.
"""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")
REVIEW = (ROOT / "DSPARK-DIST-SELF-REVIEW.md").read_text(encoding="utf-8")


def c_function(name: str) -> str:
    start = SOURCE.index(name + "(")
    brace = SOURCE.index("{", start)
    depth = 0
    for pos in range(brace, len(SOURCE)):
        if SOURCE[pos] == "{":
            depth += 1
        elif SOURCE[pos] == "}":
            depth -= 1
            if depth == 0:
                return SOURCE[start : pos + 1]
    raise AssertionError(f"unterminated function {name}")


class ExactSharedRowsStaticTest(unittest.TestCase):
    def test_opt_in_is_nested_and_preflight_precedes_mutation(self) -> None:
        enabled = c_function(
            "ds4_session_eval_layer_slice_exact_shared_rows_enabled"
        )
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        self.assertIn('getenv("DS4_DIST_SPEC_EXACT_SHARED_ROWS")', enabled)
        self.assertIn("EXPERIMENTAL EXACT-SPAN SHARED-ROW BATCHING", enabled)
        self.assertEqual(
            SOURCE.count(
                "ds4_session_eval_layer_slice_exact_shared_rows_enabled"
            ),
            2,
        )
        required = c_function(
            "ds4_session_eval_layer_slice_exact_shared_rows_required"
        )
        self.assertIn(
            'getenv("DS4_DIST_SPEC_EXACT_SHARED_ROWS_REQUIRED")', required
        )
        admission = span.index(
            "ds4_session_eval_layer_slice_exact_shared_rows_supported"
        )
        required_gate = span.index(
            "ds4_session_eval_layer_slice_exact_shared_rows_required"
        )
        active = span.index("exact-span shared rows ACTIVE backend=ROCm")
        self.assertLess(admission, required_gate)
        self.assertLess(required_gate, active)
        self.assertLess(active, span.index("ds4_gpu_tensor_write"))
        self.assertLess(admission, span.index("ds4_gpu_embed_token_hc_tensor"))
        self.assertLess(admission, span.index("metal_graph_dspark_capture_begin"))

    def test_preflight_is_fail_closed(self) -> None:
        supported = c_function(
            "ds4_session_eval_layer_slice_exact_shared_rows_supported"
        )
        self.assertIn("#if !defined(DS4_ROCM_BUILD)", supported)
        self.assertIn("return false;\n#else", supported)
        required_guards = (
            "g->placement",
            "g->quality",
            "e->tp.active",
            "g->tp_world",
            "g->materialize_ffn_out",
            "DS4_ROCM_DSV4_PREQUANT_DECODE",
            "g->debug_deferred_active",
            "g_expert_profile.active",
            "metal_graph_hc_norm_fusion_check_enabled",
            "metal_graph_use_reference_shared_down_hc",
            "metal_graph_directional_steering_attn_enabled",
            "metal_graph_directional_steering_ffn_enabled",
            "metal_graph_batch_cur_hc",
            "metal_graph_shared_out",
            "metal_graph_batch_ffn_norm",
            "metal_graph_batch_routed_out",
            "metal_graph_batch_after_attn_hc",
            "metal_graph_batch_hc_split",
            "metal_graph_batch_shared_gate",
            "metal_graph_batch_shared_up",
            "metal_graph_batch_shared_mid",
            "DS4_TENSOR_Q8_0",
            "metal_graph_needs_ffn_out",
        )
        for guard in required_guards:
            with self.subTest(guard=guard):
                self.assertIn(guard, supported)

    def test_only_shared_gate_up_swiglu_is_batched(self) -> None:
        layer = c_function(
            "ds4_session_eval_layer_slice_exact_shared_rows_layer"
        )
        phase = layer.index("METAL_DECODE_LAYER_TO_SHARED_MID")
        batch = layer.index(
            "ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor"
        )
        down = layer.index("ds4_gpu_shared_down_hc_expand_q8_0_tensor")
        capture = layer.index("metal_graph_dspark_capture_decode_layer")
        self.assertLess(phase, batch)
        self.assertLess(batch, down)
        self.assertLess(down, capture)
        self.assertEqual(
            layer.count("ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor"),
            1,
        )
        self.assertNotIn("metal_graph_encode_native_session_batch_shared", layer)
        self.assertNotIn("metal_graph_session_batch_shared", layer)
        self.assertNotIn("metal_graph_encode_decode_layer(g", layer)

    def test_admitted_failure_invalidates_without_fallback(self) -> None:
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        failure = span[span.rindex("if (!ok)") :]
        self.assertIn("if (batch_shared_rows)", failure)
        self.assertIn("ds4_session_invalidate(s)", failure)
        required_gate = span[
            span.index("ds4_session_eval_layer_slice_exact_shared_rows_required") :
            span.index("ds4_gpu_tensor_write")
        ]
        self.assertIn("!batch_shared_rows", required_gate)
        self.assertIn("ds4_session_invalidate(s)", required_gate)
        self.assertIn("no mid-span", REVIEW.lower())
        self.assertIn("fallback", REVIEW.lower())
        self.assertIn("K=2,3,4,5", REVIEW)


if __name__ == "__main__":
    unittest.main()
