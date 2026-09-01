#!/usr/bin/env python3
"""No-model final wiring guard for the DSpark/SSD remediation.

This uses explicit runtime checks so it still enforces every condition when
run with PYTHONOPTIMIZE=1.
"""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = (ROOT / "ds4.c").read_text()
METAL = (ROOT / "ds4_metal.m").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()
GITIGNORE = (ROOT / ".gitignore").read_text()


class WiringFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WiringFailure(message)


def between(text: str, start: str, end: str) -> str:
    begin = text.find(start)
    require(begin >= 0, f"missing start marker: {start}")
    finish = text.find(end, begin + len(start))
    require(finish >= 0, f"missing end marker after {start}: {end}")
    return text[begin:finish]


def require_in(text: str, needle: str, where: str) -> None:
    require(needle in text, f"missing {needle!r} in {where}")


def require_ordered(text, needles, where):
    position = -1
    for needle in needles:
        next_position = text.find(needle, position + 1)
        require(next_position >= 0, f"missing {needle!r} in {where}")
        require(next_position > position, f"out-of-order {needle!r} in {where}")
        position = next_position


def make_target(target, makefile):
    """Return the real prerequisites and tab-prefixed recipe of one target."""
    lines = makefile.splitlines()
    rule = re.compile(rf"^{re.escape(target)}\s*:\s*(.*)$")
    start = next((i for i, line in enumerate(lines) if rule.match(line)), None)
    require(start is not None, f"missing Makefile target {target}")

    header_parts = []
    index = start
    while True:
        match = rule.match(lines[index]) if index == start else None
        part = match.group(1) if match else lines[index].strip()
        continued = part.rstrip().endswith("\\")
        if continued:
            part = part.rstrip()[:-1]
        header_parts.append(part.strip())
        if not continued:
            break
        index += 1
        require(index < len(lines), f"unterminated prerequisites for {target}")

    prerequisites = {
        token for token in " ".join(header_parts).split()
        if token and not token.startswith("#")
    }
    recipe = []
    next_rule = re.compile(r"^[A-Za-z0-9_./%+-]+(?:\s+[^:]*)?:")
    for line in lines[index + 1:]:
        if next_rule.match(line):
            break
        if line.startswith("\t"):
            recipe.append(line.strip())
    return prerequisites, recipe


def require_make_test_wiring(makefile: str) -> None:
    prerequisites, recipe = make_target("test", makefile)
    required_binaries = {
        "tests/test_ssd_admission",
        "tests/test_dspark_readiness",
        "tests/test_dspark_session_admission",
        "tests/test_dspark_startup_guards",
    }
    missing_binaries = required_binaries - prerequisites
    require(not missing_binaries,
            "ordinary make test is missing prerequisites: "
            + ", ".join(sorted(missing_binaries)))

    required_commands = {
        "./tests/test_ssd_admission",
        "./tests/test_dspark_readiness",
        "./tests/test_dspark_session_admission",
        "./tests/test_dspark_startup_guards",
        "PYTHONDONTWRITEBYTECODE=1 python3 tests/test_persistent_support_lifecycle.py",
        "PYTHONDONTWRITEBYTECODE=1 python3 tests/test_dspark_final_wiring.py",
    }
    recipe_commands = set(recipe)
    missing_commands = required_commands - recipe_commands
    require(not missing_commands,
            "ordinary make test is missing recipe commands: "
            + ", ".join(sorted(missing_commands)))


def test_support_teardown_purges_every_no_copy_owner_before_unmap() -> None:
    release = between(
        METAL,
        "void ds4_gpu_release_persistent_support_map(void)",
        "static id<MTLBuffer> ds4_gpu_wrap_model_range(",
    )
    require("if (!g_support_model_buffer) return" not in release,
            "support-map release may skip cache teardown")
    require_ordered(
        release,
        (
            "ds4_gpu_synchronize()",
            "[g_q4_expert_layer_residency_cache removeAllObjects]",
            "[g_q4_expert_table_cache removeAllObjects]",
            'ds4_gpu_model_buffer_cache_clear("support-unmap")',
            "[g_transient_buffers removeAllObjects]",
            "g_support_model_buffer = nil",
            "g_support_model_map_ptr = NULL",
        ),
        "persistent-support release",
    )

    close = between(
        ENGINE,
        "void ds4_engine_close(ds4_engine *e)",
        "static bool ds4_dspark_stats_enabled",
    )
    require_ordered(
        close,
        (
            "ds4_gpu_release_persistent_support_map();",
            "model_close(&e->mtp_model)",
            "model_close(&e->model)",
            "ds4_gpu_cleanup();",
        ),
        "engine close",
    )


def test_shared_workspace_lifetime_and_admission_precede_allocation() -> None:
    acquire = between(
        ENGINE,
        "static bool ds4_shared_prefill_workspace_acquire(",
        "static void ds4_shared_prefill_workspace_abort_create(",
    )
    require_in(acquire, "state->creating", "shared-workspace acquire")
    require_in(acquire, "state->borrowers++", "shared-workspace acquire")
    require_in(acquire, "pthread_cond_wait", "shared-workspace acquire")

    publish = between(
        ENGINE,
        "static bool ds4_shared_prefill_workspace_begin_publish(",
        "static void ds4_shared_prefill_workspace_release_borrow(",
    )
    require_in(publish, "state->closing", "shared-workspace publish")
    require_in(publish, "state->borrowers++", "shared-workspace publish")

    close = between(
        ENGINE,
        "static bool ds4_shared_prefill_workspace_begin_close(",
        "static void ds4_shared_prefill_workspace_close_unlock(",
    )
    require_in(close, "state->closing = true", "shared-workspace close")
    require_in(close, "state->creating || state->borrowers != 0",
               "shared-workspace close")

    claim = between(
        ENGINE,
        "static bool ds4_engine_dspark_session_admission_claim(",
        "static void ds4_engine_dspark_session_admission_release(",
    )
    for term in (
        "ctx_size > e->admitted_context_size",
        "e->admitted_session_limit",
        "e->admitted_session_context_scratch_bytes",
        "e->admitted_session_graph_bytes",
        "e->admitted_session_speculative_bytes",
        "e->admitted_prefill_workspace_bytes",
        "*expected_context_scratch_bytes = context_scratch_bytes",
        "*expected_speculative_bytes = speculative_bytes",
        "*expected_workspace_bytes = workspace_bytes",
        "workspace_context.prefill_cap",
        "__atomic_compare_exchange_n",
    ):
        require_in(claim, term, "session admission claim")

    create_impl = between(
        ENGINE,
        "static int ds4_session_create_impl(",
        "int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size)",
    )
    create = between(
        ENGINE,
        "int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size)",
        "void ds4_session_free(ds4_session *s)",
    )
    require_ordered(
        create,
        (
            "ds4_session_creation_begin(&e->session_creation_state)",
            "ds4_session_create_impl(out, e, ctx_size)",
            "ds4_session_creation_end(&e->session_creation_state)",
        ),
        "public session creation lifecycle",
    )
    require(create_impl.find("ds4_engine_dspark_session_admission_claim(") <
            create_impl.find("metal_graph_alloc_raw_cap("),
            "session allocation precedes its admission claim")
    for term in (
        "ds4_shared_prefill_workspace_acquire(",
        "ds4_shared_prefill_workspace_abort_create(",
        "ds4_shared_prefill_workspace_begin_publish(",
        "ds4_shared_prefill_workspace_publish_locked(",
        "s->shared_prefill_workspace_borrowed = true",
        "ds4_session_release_engine_lifetime_borrows(s);",
        "workspace_bytes != s->admission_expected_workspace_bytes",
        "admission_expected_context_scratch_bytes",
        "metal_graph_context_scratch_bytes(",
        "metal_graph_ordinary_session_bytes(",
        "allocated_graph_bytes > s->admission_expected_graph_bytes",
    ):
        require_in(create_impl, term, "session creation")

    engine_close = between(
        ENGINE,
        "void ds4_engine_close(ds4_engine *e)",
        "static bool ds4_dspark_stats_enabled",
    )
    require_ordered(
        engine_close,
        (
            "ds4_session_creation_begin_close(",
            "ds4_shared_prefill_workspace_begin_close(",
            "weights_free(&e->weights)",
        ),
        "engine teardown lifecycle",
    )
    lifecycle = between(
        ENGINE,
        "static bool ds4_session_creation_begin_close(",
        "/* Admission helpers are defined",
    )
    require_in(lifecycle, "state->closing = true", "session-creation close gate")
    require_in(lifecycle, "state->in_flight != 0", "session-creation close gate")

    release_lifetimes = between(
        ENGINE,
        "static void ds4_session_release_engine_lifetime_borrows(",
        "static int ds4_session_create_impl(",
    )
    require_ordered(
        release_lifetimes,
        (
            "ds4_session_release_admission_slot(s);",
            "ds4_session_release_shared_prefill_workspace(s);",
        ),
        "session teardown lifetime order",
    )
    session_free = between(
        ENGINE,
        "void ds4_session_free(ds4_session *s)",
        "#ifdef DS4_TEST_HOOKS",
    )
    require_ordered(
        session_free,
        (
            "token_vec_free(&s->checkpoint);",
            "token_vec_free(&s->greedy_splitkv_segment);",
            "free(s->logits);",
            "free(s->sample_probs);",
            "free(s->mtp_logits);",
            "ds4_session_release_engine_lifetime_borrows(s);",
            "free(s);",
        ),
        "ordinary session teardown admission accounting",
    )
    distributed_begin = create_impl.rfind(
        "if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR)")
    distributed_end = create_impl.find(
        "if (!ds4_session_tp_register(s))", distributed_begin)
    require(distributed_begin >= 0 and distributed_end >= 0,
            "missing distributed-create failure cleanup")
    distributed_create_failure = create_impl[distributed_begin:distributed_end]
    require_ordered(
        distributed_create_failure,
        (
            "free(s->logits);",
            "free(s->sample_probs);",
            "free(s->mtp_logits);",
            "free(s->spec_row_logits);",
            "free(s->dspark_markov_bias);",
            "free(s->dspark_conf_features);",
            "ds4_session_release_engine_lifetime_borrows(s);",
            "free(s);",
        ),
        "distributed-create failure admission accounting",
    )

    scratch = between(
        ENGINE,
        "static bool ds4_context_scratch_bytes_for_workspace(",
        "static bool ds4_engine_dspark_session_context_scratch_bytes(",
    )
    require_in(scratch, "workspace_prefill_cap", "small-context scratch accounting")
    require_in(scratch, "context->scratch_bytes - one_tier_scores",
               "small-context scratch accounting")


def test_metadata_logging_and_actual_make_test_wiring() -> None:
    array_reader = between(
        ENGINE,
        "static bool model_get_u32_array_any(",
        "static bool ds4_tensor_mtp_stage(",
    )
    require_in(array_reader, "arr.len > cap", "DSpark metadata reader")
    require_in(array_reader, "*overflow_out = true", "DSpark metadata reader")
    require_in(ENGINE, "dw->target_layers_overflow = summary->target_layers_overflow",
               "DSpark metadata binding")

    memory_log = between(
        ENGINE,
        "static void ds4_engine_print_startup_memory(",
        "static uint64_t ds4_engine_host_memory_bytes(",
    )
    for category in (
        "complete admitted SSD+DSpark ledger",
        "ordinary graph",
        "speculative state",
        "shared prefill workspace",
    ):
        require_in(memory_log, category, "startup memory ledger")

    require_make_test_wiring(MAKEFILE)

    # The check must fail if a command is removed from the actual ordinary
    # test recipe while its standalone target still remains elsewhere.
    mutated = MAKEFILE.replace("\t./tests/test_ssd_admission\n", "", 1)
    require(mutated != MAKEFILE, "mutation fixture did not remove recipe command")
    try:
        require_make_test_wiring(mutated)
    except WiringFailure:
        pass
    else:
        raise WiringFailure("wiring guard accepted a mutated ordinary test recipe")


def test_new_test_artifacts_are_cleaned_and_ignored() -> None:
    generated_binaries = (
        "tests/test_ssd_admission",
        "tests/test_dspark_readiness",
        "tests/test_dspark_session_admission",
        "tests/test_dspark_startup_guards",
    )
    for binary in generated_binaries:
        require_in(GITIGNORE, f"/{binary}", ".gitignore")
    for pattern in ("*.o", "__pycache__/", "*.pyc"):
        require_in(GITIGNORE, pattern, ".gitignore")
    _, clean_recipe = make_target("clean", MAKEFILE)
    clean_text = "\n".join(clean_recipe)
    for binary in generated_binaries:
        require_in(clean_text, binary, "make clean")


def main() -> int:
    try:
        test_support_teardown_purges_every_no_copy_owner_before_unmap()
        test_shared_workspace_lifetime_and_admission_precede_allocation()
        test_metadata_logging_and_actual_make_test_wiring()
        test_new_test_artifacts_are_cleaned_and_ignored()
    except WiringFailure as error:
        print(f"test_dspark_final_wiring: FAIL: {error}", file=sys.stderr)
        return 1
    print("test_dspark_final_wiring: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
