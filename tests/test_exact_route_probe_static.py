#!/usr/bin/env python3
"""Source-contract checks for the default-off exact-span route probe."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")


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


class ExactRouteProbeStaticTest(unittest.TestCase):
    def test_probe_is_default_off_and_span_scoped(self) -> None:
        enabled = c_function(
            "ds4_session_eval_layer_slice_exact_route_probe_enabled"
        )
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        prepare = c_function("ds4_exact_route_probe_prepare")
        self.assertIn('getenv("DS4_DIST_SPEC_EXACT_ROUTE_PROBE")', enabled)
        self.assertIn("static int cached = -1", enabled)
        self.assertEqual(
            SOURCE.count(
                "ds4_session_eval_layer_slice_exact_route_probe_enabled"
            ),
            2,
        )
        self.assertIn("ds4_gpu_tensor_alloc_ptr_on(0", prepare)
        self.assertLess(
            span.index("route_probe_requested"),
            span.index("ds4_gpu_tensor_write"),
        )
        self.assertIn(
            "route_probe_requested &&\n        s->exact_route_probe_spans < route_probe_limit",
            span,
        )

    def test_copy_is_at_router_boundary_and_covers_both_loops(self) -> None:
        decode = c_function("metal_graph_encode_decode_layer_phase")
        shared = c_function(
            "ds4_session_eval_layer_slice_exact_shared_rows_layer"
        )
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        capture = decode.index("g->exact_route_probe_arena")
        # Use the last occurrence: the first is the cached default-off check.
        capture = decode.index("ds4_gpu_tensor_copy(", capture)
        self.assertLess(
            decode.index("metal_graph_decode_set_hash_selected_override"),
            capture,
        )
        self.assertLess(capture, decode.index('DS4_METAL_PROFILE_DECODE_STAGE("router")'))
        self.assertIn("metal_graph_router_selected(g)", decode[capture:])
        self.assertIn("g->exact_route_probe_offset += selected_bytes", decode)
        self.assertIn("METAL_DECODE_LAYER_TO_SHARED_MID", shared)
        self.assertIn("metal_graph_encode_decode_layer(g", span)
        self.assertIn("active_route_probe", span)

    def test_one_read_follows_existing_sync_and_emits_auditable_fields(self) -> None:
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        emit = c_function("ds4_exact_route_probe_emit")
        end = span.index("ds4_gpu_end_commands")
        read = span.index("ds4_gpu_tensor_read(active_route_probe->arena")
        self.assertLess(end, read)
        self.assertEqual(span.count("ds4_gpu_tensor_read(active_route_probe->arena"), 1)
        for field in (
            "pos0=%u",
            "rows=%u",
            "layers=%u:%u",
            "layer=%u",
            "raw_sets=",
            "intersection=[",
            "jaccard=%.6f",
            "union=[",
        ):
            with self.subTest(field=field):
                self.assertIn(field, emit)
        self.assertIn('fprintf(stderr, "%s\\n", line)', emit)

    def test_requested_failures_are_closed_and_limit_is_per_session(self) -> None:
        limit = c_function(
            "ds4_session_eval_layer_slice_exact_route_probe_limit"
        )
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        session_start = SOURCE.index("struct ds4_session {")
        session_end = SOURCE.index("};", session_start)
        session = SOURCE[session_start:session_end]
        self.assertIn(
            'getenv("DS4_DIST_SPEC_EXACT_ROUTE_PROBE_LIMIT")', limit
        )
        self.assertIn("value == 0u", limit)
        self.assertIn("value > UINT32_MAX", limit)
        self.assertIn("exact_route_probe_spans", session)
        self.assertIn("s->exact_route_probe_spans++", span)
        failure = span[span.index("if (!ok)") :]
        self.assertIn("active_route_probe", failure)
        self.assertIn("ds4_session_invalidate(s)", failure)
        self.assertIn("exact-span route probe allocation failed", span)

    def test_failure_sync_precedes_cleanup_and_leaks_only_probe_arena(self) -> None:
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        failure_sync = span.index("if (!ok && !failure_sync_attempted)")
        synchronize = span.index("ds4_gpu_synchronize()", failure_sync)
        leak = span.index(
            "if (!ok && !failure_sync_ok && active_route_probe)", synchronize
        )
        release = span.index("ds4_gpu_tensor_free(cur[i])", leak)
        discard = span.index("ds4_exact_route_probe_discard(&route_probe)", release)
        self.assertLess(synchronize, leak)
        self.assertLess(synchronize, release)
        self.assertLess(synchronize, discard)

        failed_sync = span[leak:release]
        self.assertIn("g->exact_route_probe_arena = NULL", failed_sync)
        self.assertIn("g->exact_route_probe_offset = 0", failed_sync)
        self.assertIn("route_probe.arena = NULL", failed_sync)
        self.assertIn("free(route_probe.host_ids)", failed_sync)
        self.assertIn("leaking %llu-byte exact-span route probe GPU arena", failed_sync)
        self.assertNotIn("ds4_gpu_tensor_free", failed_sync)

    def test_exact_probe_has_no_graph_wide_quarantine_guards(self) -> None:
        span = c_function("ds4_session_eval_layer_slice_exact_span")
        decode = c_function("metal_graph_encode_decode_layer_phase")
        graph_free = c_function("metal_graph_free")
        self.assertNotIn("exact_span_lifetime_quarantined", SOURCE)
        self.assertNotIn("quarantin", span)
        self.assertNotIn("exact_span", graph_free)
        self.assertIn(
            "const bool exact_route_probe =\n"
            "        g->exact_route_probe_arena != NULL",
            decode,
        )


if __name__ == "__main__":
    unittest.main()
