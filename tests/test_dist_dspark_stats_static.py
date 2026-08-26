#!/usr/bin/env python3
"""Source-contract checks for distributed DSpark worker proposal telemetry."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ds4.c").read_text(encoding="utf-8")
DIST_SOURCE = (ROOT / "ds4_distributed.c").read_text(encoding="utf-8")
TP_SOURCE = (ROOT / "ds4_tp.c").read_text(encoding="utf-8")
TP_NHI_SOURCE = (ROOT / "ds4_tp_nhi.c").read_text(encoding="utf-8")
Q8_TEST_SOURCE = (ROOT / "tests" / "test_q8_krow_rocm.c").read_text(
    encoding="utf-8"
)


def c_function_from(source: str, name: str) -> str:
    start = source.index(name + "(")
    brace = source.index("{", start)
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"unterminated function {name}")


def c_function(name: str) -> str:
    return c_function_from(SOURCE, name)


class DistributedDSparkStatsStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fn = c_function("ds4_session_dist_dspark_draft")

    def test_every_valid_call_counts_one_cycle_and_first_token(self) -> None:
        prepare = self.fn.index("ds4_session_prepare_dspark_draft")
        cycle = self.fn.index("s->dspark_stats.cycles++")
        no_draft = self.fn.index("if (!s->dspark_draft_valid")
        self.assertLess(prepare, cycle)
        self.assertLess(cycle, no_draft)
        self.assertIn("s->dspark_stats.first_tokens++", self.fn)
        marker = self.fn.index("DSpark distributed worker stats are proposal-only")
        self.assertLess(marker, cycle)
        self.assertIn("if (s->dspark_stats.cycles == 0)", self.fn)

    def test_empty_and_nonempty_block_lengths_are_recorded(self) -> None:
        no_draft = self.fn.index("if (!s->dspark_draft_valid")
        success = self.fn.index("int n = (int)s->dspark_draft_len", no_draft)
        empty = self.fn[no_draft:success]
        nonempty = self.fn[success:]
        self.assertIn("s->dspark_stats.no_draft++", empty)
        self.assertIn("draft_len_hist, 0", empty)
        self.assertIn("s->dspark_stats.proposed_tokens += (uint64_t)n", nonempty)
        self.assertIn("draft_len_hist,\n                                  (uint32_t)n", nonempty)

    def test_worker_does_not_invent_acceptance_feedback(self) -> None:
        self.assertNotIn("accepted_draft", self.fn)
        self.assertNotIn("accepted_len_hist", self.fn)
        self.assertIn("Distributed acceptance is", self.fn)
        self.assertIn("not fed back to this worker", self.fn)
        self.assertIn("acceptance feedback unavailable", self.fn)


class DistributedSafetyContractsStaticTest(unittest.TestCase):
    def test_result_bounds_precede_peer_payload_allocation(self) -> None:
        fn = c_function_from(DIST_SOURCE, "dist_recv_result_alloc_leased")
        bounds = fn.index("ds4_dist_v3_result_payload_validate")
        allocation = fn.index("malloc(result.payload_bytes)")
        self.assertLess(bounds, allocation)
        self.assertIn("result.telemetry_count > state->n_layers + 1u", fn)

    def test_tp_frame_bounds_precede_command_allocation(self) -> None:
        fn = c_function_from(TP_SOURCE, "ds4_tp_recv_command")
        header = fn.index("tp_read_frame_header")
        bounds = fn.index("tp_command_frame_size_allowed")
        allocation = fn.index("malloc(bytes)")
        self.assertLess(header, bounds)
        self.assertLess(bounds, allocation)
        self.assertIn("shutdown(tp->control_fd, SHUT_RDWR)", fn)
        self.assertNotIn("atoi(tmo)", TP_SOURCE)

    def test_gpu_cleanup_precedes_model_unmap(self) -> None:
        definition = SOURCE[SOURCE.index("void ds4_engine_close(") :]
        fn = c_function_from(definition, "ds4_engine_close")
        cleanup = fn.index("ds4_gpu_cleanup()")
        support_close = fn.index("model_close(&e->mtp_model)")
        target_close = fn.index("model_close(&e->model)")
        self.assertLess(cleanup, support_close)
        self.assertLess(cleanup, target_close)

    def test_exact_verify_is_explicitly_required_on_worker(self) -> None:
        self.assertGreaterEqual(
            DIST_SOURCE.count("DS4_DIST_WORK_F_SPEC_EXACT_REQUIRED"), 5
        )
        worker = c_function_from(DIST_SOURCE, "dist_worker_process_work_message")
        coordinator = c_function_from(DIST_SOURCE, "dist_coordinator_eval_span")
        self.assertIn("worker exact speculative verify gate is unavailable", worker)
        self.assertIn("spec_exact_required &&", worker)
        self.assertIn("coordinator exact speculative verify gate is unavailable", coordinator)

    def test_partial_tp_pool_failure_closes_both_export_fds(self) -> None:
        open_fn = c_function_from(TP_NHI_SOURCE, "ds4_tp_nhi_open")
        failed_alloc = open_fn.index("uncached TP pool allocation/export failed")
        prefix = open_fn[:failed_alloc]
        self.assertIn("if (tx_fd >= 0) close(tx_fd)", prefix)
        self.assertIn("if (rx_fd >= 0) close(rx_fd)", prefix)

    def test_primary_cache_test_forces_cached_selector(self) -> None:
        self.assertIn('setenv("DS4_ROCM_SHARED_DOWN_CUBLAS"', Q8_TEST_SOURCE)

    def test_trust_all_advances_target_state_before_emitting(self) -> None:
        cycle = c_function_from(DIST_SOURCE, "ds4_dist_session_mtp_spec_cycle")
        start = cycle.index("if (ds4_dist_spec_trust_all_enabled())")
        end = cycle.index("/* 0. Fused span:", start)
        trust = cycle[start:end]
        advance = trust.index("dist_coordinator_eval_span")
        final_logits = trust.index("ds4_session_set_logits", advance)
        account = trust.index("d->spec_accepted", final_logits)
        self.assertLess(advance, final_logits)
        self.assertLess(final_logits, account)
        self.assertIn("DS4_DIST_WORK_F_OUTPUT_ALL_LOGITS", trust)
        self.assertIn("trusted_span=", trust)


if __name__ == "__main__":
    unittest.main()
