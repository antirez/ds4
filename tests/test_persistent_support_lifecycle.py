#!/usr/bin/env python3
"""No-model source-order and admission-allocation correspondence guard."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def main() -> None:
    engine = (ROOT / "ds4.c").read_text()
    metal = (ROOT / "ds4_metal.m").read_text()
    header = (ROOT / "ds4_gpu.h").read_text()
    server = (ROOT / "ds4_server.c").read_text()
    ssd_header = (ROOT / "ds4_ssd.h").read_text()

    close = between(engine, "void ds4_engine_close(ds4_engine *e)", "static bool ds4_dspark_stats_enabled")
    assert "ds4_gpu_release_persistent_support_map();" in close
    assert close.index("ds4_gpu_release_persistent_support_map();") < close.index(
        "if (e->mtp_model.map) model_close(&e->mtp_model);"
    )

    release = between(
        metal,
        "void ds4_gpu_release_persistent_support_map(void)",
        "static id<MTLBuffer> ds4_gpu_wrap_model_range",
    )
    assert "ds4_gpu_synchronize()" in release
    assert release.index("ds4_gpu_synchronize()") < release.rindex("g_support_model_buffer = nil;")
    assert "newBufferWithBytesNoCopy" in metal
    assert "ds4_gpu_register_persistent_support_map" in header
    assert "ds4_gpu_release_persistent_support_map" in header

    bind = engine.index("dspark_weights_bind_optional(&e->dspark_weights")
    detect = engine.index(
        "support_model_detect(&e->mtp_model, &e->support_stages, &dspark);",
        engine.index("if (opt->mtp_path && opt->mtp_path[0] &&", bind),
    )
    startup_policy = engine.index(
        "!support_kind_allowed_for_startup(\n"
        "                    e->dspark, e->ssd_streaming, e->support_kind)",
        detect,
    )
    legacy_bind = engine.index(
        "if (e->support_kind == DS4_SUPPORT_MTP_LEGACY)", startup_policy
    )
    readiness = engine.index(
        "!dspark_weights_ready_for_enabled_runtime(\n"
        "                        &e->dspark_weights)",
        bind,
    )
    persistent_register = engine.index(
        "ds4_gpu_register_persistent_support_map(", readiness
    )
    assert detect < startup_policy < legacy_bind
    assert bind < readiness < persistent_register

    admission_guard = between(
        engine,
        "static bool ds4_engine_dspark_ssd_admission_guard(",
        "static bool cpu_directional_steering_enabled",
    )
    assert ".prelocked_bytes = e->simulated_memory.bytes" in admission_guard
    assert ".session_count = session_count" in admission_guard
    assert ".session_kv_bytes = session_kv_bytes" in admission_guard
    assert "ds4_engine_dspark_session_context_scratch_bytes(" in admission_guard
    assert ".session_context_scratch_bytes = session_context_scratch_bytes" in admission_guard
    assert ".session_graph_bytes = session_graph_bytes" in admission_guard
    assert ".session_speculative_bytes = session_speculative_bytes" in admission_guard
    assert ".session_host_bytes = session_host_bytes" in admission_guard
    assert ".session_prefill_workspace_bytes =" in admission_guard
    assert ".shared_prefill_workspace_bytes = shared_prefill_workspace_bytes" in admission_guard
    assert "share_session_prefill_workspace" not in ssd_header

    speculative_estimate = between(
        engine,
        "static bool ds4_engine_dspark_session_speculative_bytes(",
        "static bool ds4_engine_dspark_session_host_bytes(",
    )
    assert "2u + 2u * DS4_SPEC_PREFIX_SLOTS" in speculative_estimate
    assert "DS4_DSPARK_MAX_BLOCK_SIZE" in speculative_estimate
    assert "dw->target_layer_count" in speculative_estimate
    assert "dw->n_stages" in speculative_estimate
    assert "context->raw_cap" in speculative_estimate
    assert "context->prefill_cap" in speculative_estimate

    ordinary_graph_estimate = between(
        engine,
        "static bool ds4_engine_dspark_session_ordinary_graph_bytes(",
        "/* The verifier and DSpark target-capture graph",
    )
    for term in (
        "coff * DS4_N_HEAD_DIM",
        "coff * DS4_N_INDEXER_HEAD_DIM",
        "DS4_N_VOCAB",
        "ffn_out",
        "workspace_prefill_cap",
    ):
        assert term in ordinary_graph_estimate

    ordinary_graph_allocated = between(
        engine,
        "static bool metal_graph_ordinary_session_bytes(",
        "static void metal_graph_free_prefill_workspace(",
    )
    for field in (
        "layer_attn_state_kv", "layer_attn_state_score",
        "layer_index_state_kv", "layer_index_state_score",
        "cur_hc_by_tier", "routed_out_by_tier", "ffn_out_by_tier",
        "output_pre_by_tier", "logits_by_tier",
    ):
        assert f"g->{field}" in ordinary_graph_allocated

    speculative_allocated = between(
        engine,
        "static bool metal_graph_dspark_speculative_bytes(",
        "static void metal_graph_free_prefill_workspace(",
    )
    speculative_fields = {
        "mtp_embed", "mtp_enorm", "mtp_eproj", "mtp_eproj_hc",
        "mtp_hnorm_hc", "mtp_hproj_hc", "mtp_input_hc",
        "mtp_state_hc", "mtp_next_hc", "mtp_raw_cache", "spec_logits",
        "spec_attn_state_kv", "spec_attn_state_score",
        "spec_index_state_kv", "spec_index_state_score",
        "spec_prefix1_attn_state_kv", "spec_prefix1_attn_state_score",
        "spec_prefix1_index_state_kv", "spec_prefix1_index_state_score",
        "dspark_hc_mean_weights", "dspark_hc_mean_rows",
        "dspark_target_hidden", "dspark_target_hidden_batch",
        "dspark_stage0_packed", "dspark_stage0_proj", "dspark_main_x",
        "dspark_draft_tokens", "dspark_draft_hc", "dspark_target_hc",
        "dspark_stage_input_hc", "dspark_stage_output_hc",
        "dspark_position_ids", "dspark_raw_cache",
    }
    for field in speculative_fields:
        assert f"g->{field}" in speculative_allocated

    host_estimate = between(
        engine,
        "static bool ds4_engine_dspark_session_host_bytes(\n"
        "        const ds4_engine *e,\n"
        "        uint64_t         *out) {",
        "#ifndef DS4_NO_GPU",
    )
    assert "sizeof(ds4_session)" in host_estimate
    assert "4, DS4_N_VOCAB" in host_estimate
    assert "DS4_N_EMBD + e->dspark_weights.markov_rank" in host_estimate

    workspace_macro = between(
        engine,
        "#define DS4_GPU_PREFILL_WORKSPACE_FIELDS(X)",
        "/* Class H accessors.",
    )
    workspace_fields = set(re.findall(r"X\((\w+)\)", workspace_macro))
    expected_workspace_fields = {
        "prefill_tokens", "batch_ffn_out", "batch_routed_out",
        "batch_routed_down", "batch_routed_mid", "batch_routed_up",
        "batch_routed_gate", "batch_router_weights",
        "batch_router_selected", "batch_router_probs",
        "batch_router_logits", "batch_shared_out", "batch_shared_mid",
        "batch_shared_up", "batch_shared_gate", "batch_ffn_norm",
        "batch_ffn_cur", "batch_after_attn_hc", "batch_low_tmp",
        "batch_group_tmp", "batch_attn_out", "batch_attn_low",
        "batch_heads", "batch_indexer_weights", "batch_indexer_q",
        "batch_comp_sc", "batch_comp_kv", "batch_kv", "batch_kv_raw",
        "batch_q", "batch_qr_norm", "batch_qr", "batch_attn_norm",
        "batch_attn_cur", "batch_hc_split", "batch_hc_mix",
        "batch_flat_hc", "batch_next_hc", "batch_cur_hc",
    }
    assert workspace_fields == expected_workspace_fields

    allocation = between(
        engine,
        "static bool metal_graph_alloc_raw_cap(",
        "bool layer_cache_ok = true;",
    )
    for field in workspace_fields - {"batch_ffn_out"}:
        assert f"g->{field}_by_tier" in allocation
    ensure_ffn = between(
        engine,
        "static bool metal_graph_ensure_batch_ffn_out_on(",
        "static bool metal_graph_ensure_batch_ffn_out(",
    )
    assert "g->batch_ffn_out_by_tier[t]" in ensure_ffn

    copy_workspace = between(
        engine,
        "static void metal_graph_copy_prefill_workspace_pointers(",
        "static void metal_graph_transfer_prefill_workspace(",
    )
    count_workspace = between(
        engine,
        "static uint64_t metal_graph_prefill_workspace_bytes(",
        "static void metal_graph_free_prefill_workspace(",
    )
    free_workspace = between(
        engine,
        "static void metal_graph_free_prefill_workspace(",
        "/* Release every Metal tensor",
    )
    for section in (copy_workspace, count_workspace, free_workspace):
        assert "DS4_GPU_PREFILL_WORKSPACE_FIELDS" in section

    workspace_estimate = between(
        engine,
        "static bool ds4_engine_prefill_workspace_bytes(",
        "static bool ds4_engine_dspark_session_speculative_bytes(",
    )
    assert "4u * hc_dim" in workspace_estimate
    assert "8u * DS4_N_EMBD" in workspace_estimate
    assert "3ull * DS4_N_EXPERT_USED * DS4_N_FF_EXP" in workspace_estimate
    assert "DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS" in workspace_estimate

    session_create_impl = between(
        engine,
        "static int ds4_session_create_impl(ds4_session **out,",
        "int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size)",
    )
    assert "metal_graph_transfer_prefill_workspace(" in session_create_impl
    assert "metal_graph_prefill_workspace_bytes(&s->graph)" in session_create_impl
    assert session_create_impl.index("metal_graph_prefill_workspace_bytes(&s->graph)") < session_create_impl.index(
        "metal_graph_transfer_prefill_workspace("
    )
    for buffer_name in (
        "spec_row_logits", "dspark_markov_bias", "dspark_conf_features"
    ):
        assert f"s->{buffer_name}" in session_create_impl
    assert "workspace_bytes != s->admission_expected_workspace_bytes" in session_create_impl
    assert "allocated_graph_bytes > s->admission_expected_graph_bytes" in session_create_impl
    assert (
        "allocated_speculative_bytes !=\n"
        "                    s->admission_expected_speculative_bytes"
    ) in session_create_impl

    session_create = between(
        engine,
        "int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size)",
        "void ds4_session_free(ds4_session *s)",
    )
    assert session_create.index("ds4_session_creation_begin(") < session_create.index(
        "ds4_session_create_impl("
    ) < session_create.index("ds4_session_creation_end(")

    memory_lock = engine.index(
        "!ds4_ssd_memory_lock_acquire(&e->simulated_memory"
    )
    admission = engine.index(
        "!ds4_engine_dspark_ssd_admission_guard(e, opt->context_size)",
        memory_lock,
    )
    assert memory_lock < admission < persistent_register

    session_hint = server.index(
        "cfg.engine.placement_session_count_hint =\n"
        "        cfg.batched_sessions > 0 ? cfg.batched_sessions : 1;"
    )
    shared_workspace = server.index(
        "cfg.engine.share_session_prefill_workspace = cfg.batched_sessions > 0;",
        session_hint,
    )
    engine_open = server.index("ds4_engine_open(&engine, &cfg.engine)", shared_workspace)
    assert session_hint < shared_workspace < engine_open
    print("test_persistent_support_lifecycle: PASS")


if __name__ == "__main__":
    main()
