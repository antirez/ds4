"""Run the actual Q4 frontend preflight policy with an instrumented host backend.

Extract production functions from ds4.c and compile them with ASan/UBSan. The
backend records descriptors, requested rows and memory reserves; an independent
chunk walker checks the width predictor. No production function bodies are
copied here. This checks frontend policy and error propagation, not GPU memory
availability, allocator behavior, asynchronous work or concurrent sessions.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from kernel_source import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production_bodies():
    source = (ROOT / "ds4.c").read_text()
    signatures = (
        "static uint64_t ds4_add_sat_u64(",
        "static uint32_t metal_graph_prefill_chunk_rows_at(",
        "static uint32_t metal_graph_prefill_max_chunk_rows(",
        "static int ds4_session_prepare_q4_attn_q_b_sidecars(",
        "int ds4_session_prepare_sync(",
        "static void ds4_session_mark_engine_counted(",
    )
    # The descriptor collector also has an earlier forward declaration.
    marker = "#ifndef DS4_NO_GPU\nstatic int ds4_prepare_q4_attn_q_b_sidecars("
    if source.count(marker) != 1:
        raise AssertionError("expected one guarded Q4 descriptor collector")
    collector = extract_function(
        source[source.index(marker):], "static int ds4_prepare_q4_attn_q_b_sidecars(")
    bodies = [extract_function(source, signature) for signature in signatures]
    bodies.insert(3, collector)
    return "\n".join(bodies)


SHIM = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include "ds4_gpu.h"

/* Small model dimensions exercise descriptor filtering and stack capacity. */
#define DS4_MAX_LAYER 4
#define DS4_N_LAYER 4
#define DS4_N_HEAD 8
#define DS4_N_HEAD_DIM 16
#define DS4_N_LORA_Q 32
#define DS4_TENSOR_Q4_K 12
#define DS4_TENSOR_Q8_0 8

typedef struct {
    unsigned type, ndim;
    uint64_t dim[2], abs_offset, bytes;
} ds4_tensor;
typedef struct { const ds4_tensor *attn_q_b; } test_layer;
typedef struct { test_layer layer[DS4_MAX_LAYER]; } ds4_weights;
typedef struct { void *map; uint64_t size; } ds4_model;
typedef struct {
    uint32_t prefill_cap, raw_cap;
    void *deepseek4_vision_weights;
} ds4_gpu_graph;
typedef struct { uint64_t total_bytes; } ds4_context_memory;
typedef struct { int *v, len, cap; } ds4_tokens;
typedef struct {
    int backend;
    uint32_t sessions, live_session_count, prefill_chunk;
    bool ssd_streaming;
    ds4_model model;
    ds4_weights weights;
} ds4_engine;
typedef struct {
    ds4_engine *engine;
    bool glm, cpu, ds41, engine_session_counted, checkpoint_valid, vision_matches;
    uint64_t q4_attn_q_b_f16_sidecars_generation;
    uint32_t q4_attn_q_b_f16_prepared_rows;
    int ctx_size;
    ds4_gpu_graph graph;
    ds4_tokens checkpoint;
    void *sync_images;
    size_t sync_image_count;
} ds4_session;

static unsigned calls, desc_count;
static unsigned generation_queries, prefix_queries;
static uint64_t reserve, generation = 1, memory_bytes = 100, streaming_bytes = 50;
static int prepare_result = 1;
static bool grow_during_prepare;
static uint32_t rows_seen;
static ds4_gpu_q4_attn_q_b_f16_sidecar_desc saved[DS4_MAX_LAYER];
static const void *map_seen;
static uint64_t size_seen;

static bool ds4_backend_uses_graph(int backend) { return backend == 1; }
static bool ds4_session_is_cpu(ds4_session *s) { return s->cpu; }
static bool ds4_session_is_glm(ds4_session *s) { return s->glm; }
static bool ds4_session_is_ds41(ds4_session *s) { return s->ds41; }
static bool ds4_session_vision_prefix_matches(ds4_session *s,
                                             void *images, size_t count) {
    ++prefix_queries;
    (void)images;
    (void)count;
    return s->vision_matches;
}
static bool ds4_tokens_starts_with(const ds4_tokens *prompt,
                                   const ds4_tokens *checkpoint) {
    return !memcmp(prompt->v, checkpoint->v,
                   (size_t)checkpoint->len * sizeof(int));
}
static uint32_t metal_graph_resume_prefill_min_tokens(void) { return 16; }
static const char *ds4_backend_name(int backend) { (void)backend; return "test"; }
static uint32_t engine_placement_session_count(ds4_engine *e) { return e->sessions; }
uint64_t ds4_gpu_q4_attn_q_b_f16_cache_generation(void) {
    ++generation_queries;
    return generation;
}
static uint64_t ds4_engine_streaming_transient_guard_bytes(ds4_engine *e) {
    (void)e;
    return streaming_bytes;
}
static ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        int backend, int context, uint32_t chunk, bool ssd) {
    (void)backend;
    (void)context;
    (void)chunk;
    (void)ssd;
    return (ds4_context_memory){memory_bytes};
}
int ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
        const void *map, uint64_t size,
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *descs,
        uint32_t count, uint32_t rows, uint64_t requested_reserve,
        uint64_t *prepared_bytes) {
    assert(count <= DS4_MAX_LAYER);
    ++calls;
    desc_count = count;
    rows_seen = rows;
    reserve = requested_reserve;
    map_seen = map;
    size_seen = size;
    memcpy(saved, descs, count * sizeof(*descs));
    *prepared_bytes = 64;
    if (prepare_result > 0 && grow_during_prepare) ++generation;
    return prepare_result;
}
'''

DRIVER = r'''
static void check_descriptors(ds4_engine *e, ds4_tensor *q4) {
    assert(desc_count == 2 && saved[0].layer == 0 && saved[1].layer == 3);
    assert(map_seen == e->model.map && size_seen == e->model.size);
    for (unsigned i = 0; i < desc_count; ++i) {
        assert(saved[i].weight_offset == q4->abs_offset);
        assert(saved[i].weight_bytes == q4->bytes);
        assert(saved[i].in_dim == q4->dim[0]);
        assert(saved[i].out_dim == q4->dim[1]);
        assert(saved[i].weight_type == DS4_TENSOR_Q4_K);
    }
}

/* Independent oracle: walk every emitted text chunk, keeping the largest.
 * It neither calls the production per-chunk helper nor uses the predictor's
 * first-chunk/remainder shortcut. */
static unsigned check_chunk_schedules(void) {
    unsigned cases = 0;
    for (unsigned cap = 1; cap <= 65; cap += 4)
    for (unsigned raw = 1; raw <= 69; raw += 4)
    for (unsigned start = 0; start < 130; start += 7)
    for (unsigned len = 1; len <= 260; len += 7) {
        ds4_gpu_graph graph = {.prefill_cap = cap, .raw_cap = raw};
        unsigned pos = start, left = len, actual = 0;
        while (left) {
            unsigned width = cap;
            if (start && width > raw) width = raw;
            if (start && pos % cap && width > cap - pos % cap)
                width = cap - pos % cap;
            const unsigned chunk = left < width ? left : width;
            if (chunk > actual) actual = chunk;
            left -= chunk;
            pos += chunk;
        }
        assert(metal_graph_prefill_max_chunk_rows(&graph, start, len) == actual);
        ++cases;
    }
    assert(cases == 220932);
    ds4_gpu_graph empty = {0};
    assert(metal_graph_prefill_max_chunk_rows(NULL, 0, 10) == 0);
    assert(metal_graph_prefill_max_chunk_rows(&empty, 0, 10) == 0);
    empty.prefill_cap = 128;
    assert(metal_graph_prefill_max_chunk_rows(&empty, 1, 10) == 0);
    assert(metal_graph_prefill_max_chunk_rows(&empty, 0, 0) == 0);
    return cases;
}

int main(void) {
    ds4_tensor q4 = {
        .type = DS4_TENSOR_Q4_K, .ndim = 2, .dim = {32, 128},
        .abs_offset = 72, .bytes = 99,
    };
    ds4_tensor q8 = q4, bad = q4;
    q8.type = DS4_TENSOR_Q8_0;
    bad.dim[1]++;
    int model_storage = 0;
    ds4_engine engine = {
        .backend = 1, .sessions = 4, .live_session_count = 1,
        .model = {&model_storage, sizeof(model_storage)},
        .weights = {.layer = {{&q4}, {&q8}, {&bad}, {&q4}}},
    };
    ds4_session session = {
        .engine = &engine, .engine_session_counted = true, .ctx_size = 1024,
        .vision_matches = true, .graph = {.prefill_cap = 128, .raw_cap = 64},
    };
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(calls == 1 && reserve == 350);  /* Three future sessions + streaming. */
    check_descriptors(&engine, &q4);
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32) && calls == 1);
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 64) && calls == 2);
    ++generation;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32) && calls == 3);

    session.engine_session_counted = false;
    ++generation;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(reserve == 250);  /* Eager current session is not registered yet. */
    memory_bytes = UINT64_MAX;
    ++generation;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(reserve == UINT64_MAX);  /* Saturating multiplication. */
    memory_bytes = UINT64_MAX / 2;
    ++generation;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(reserve == UINT64_MAX);  /* Product fits; adding streaming does not. */
    memory_bytes = 100;
    engine.live_session_count = UINT32_MAX;
    ++generation;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(reserve == streaming_bytes);  /* No counter increment wraparound. */
    engine.live_session_count = 1;
    session.engine_session_counted = true;

    prepare_result = -1;
    ++generation;
    assert(!ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(session.q4_attn_q_b_f16_sidecars_generation != generation);
    prepare_result = 0;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(session.q4_attn_q_b_f16_sidecars_generation != generation);
    prepare_result = 1;
    grow_during_prepare = true;
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 32));
    assert(session.q4_attn_q_b_f16_sidecars_generation == generation);
    grow_during_prepare = false;

    /* Collectors ignore non-Q4/malformed/absent tensors and admit a full array. */
    unsigned before = calls;
    engine.weights.layer[0].attn_q_b = &q8;
    engine.weights.layer[3].attn_q_b = NULL;
    assert(ds4_prepare_q4_attn_q_b_sidecars(&engine.model, &engine.weights, 32, 0));
    assert(calls == before);
    engine.weights.layer[0].attn_q_b = &bad;
    bad.dim[1] = 128;
    bad.dim[0] = 33;
    assert(ds4_prepare_q4_attn_q_b_sidecars(&engine.model, &engine.weights, 32, 0));
    bad.dim[0] = 32;
    bad.ndim = 1;
    assert(ds4_prepare_q4_attn_q_b_sidecars(&engine.model, &engine.weights, 32, 0));
    assert(calls == before);
    assert(!ds4_prepare_q4_attn_q_b_sidecars(NULL, &engine.weights, 32, 0));
    assert(!ds4_prepare_q4_attn_q_b_sidecars(&engine.model, NULL, 32, 0));
    for (unsigned i = 0; i < DS4_MAX_LAYER; ++i)
        engine.weights.layer[i].attn_q_b = &q4;
    assert(ds4_prepare_q4_attn_q_b_sidecars(&engine.model, &engine.weights, 32, 0));
    assert(desc_count == DS4_MAX_LAYER && saved[3].layer == 3);

    const unsigned schedules = check_chunk_schedules();
    int tokens[300] = {0};
    ds4_tokens prompt = {tokens, 100, 300};
    char err[128] = "";
    session.checkpoint = (ds4_tokens){tokens, 0, 300};
    session.checkpoint_valid = false;
    ++generation;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(rows_seen == 100);
    session.checkpoint_valid = true;
    session.checkpoint.len = 70;
    prompt.len = 170;
    ++generation;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(rows_seen == 58);  /* Boundary at 128; remaining chunk has 42 rows. */
    prompt.len = 75;
    before = calls;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(calls == before);  /* Short suffix uses decode, not batch prefill. */
    prompt.len = session.checkpoint.len;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(calls == before);  /* Identical prompt has no work. */
    prompt.len = 75;
    session.vision_matches = false;
    ++generation;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(rows_seen == 75);  /* Changed images require cold-prefill capacity. */
    session.graph.deepseek4_vision_weights = &session;
    session.vision_matches = true;
    prompt.len = 170;
    ++generation;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(rows_seen == 64);  /* Vision may cross the first text boundary. */

    prepare_result = -1;
    ++generation;
    assert(ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)) == 1);
    assert(strstr(err, "preflight failed"));
    prepare_result = 0;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(session.q4_attn_q_b_f16_sidecars_generation != generation);
    prepare_result = 1;
    before = calls;
    session.cpu = true;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    session.cpu = false;
    session.glm = true;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    session.glm = false;

    /* V4.1 owns a separate graph. Give its unused V4 graph plausible batch
     * dimensions so a missing family guard cannot hide behind zero capacity.
     * A required V4 sidecar failure must not block the V4.1 session. */
    session.ds41 = true;
    prepare_result = -1;
    ++generation;
    const unsigned generation_before = generation_queries, prefix_before = prefix_queries;
    unsigned char ds41_before[sizeof(session)];
    memcpy(ds41_before, &session, sizeof(session));
    assert(ds4_session_prepare_q4_attn_q_b_sidecars(&session, 128));
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(calls == before && generation_queries == generation_before);
    assert(prefix_queries == prefix_before);
    assert(!memcmp(&session, ds41_before, sizeof(session)));

    /* The common registration helper must count each V4.1 session once.
     * The public session fixture also checks its new creation-path call. */
    ds4_engine ds41_engine = {.backend = 1};
    ds4_session first = {.engine = &ds41_engine, .ds41 = true};
    ds4_session second = {.engine = &ds41_engine, .ds41 = true};
    ds4_session no_engine = {0};
    ds4_session_mark_engine_counted(NULL);
    ds4_session_mark_engine_counted(&no_engine);
    assert(!no_engine.engine_session_counted);
    ds4_session_mark_engine_counted(&first);
    assert(first.engine_session_counted && ds41_engine.live_session_count == 1);
    ds4_session_mark_engine_counted(&first);
    assert(ds41_engine.live_session_count == 1);
    ds4_session_mark_engine_counted(&second);
    assert(second.engine_session_counted && ds41_engine.live_session_count == 2);
    session.ds41 = false;
    prepare_result = 1;
    engine.backend = 0;
    assert(!ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)));
    assert(calls == before);
    assert(ds4_session_prepare_sync(NULL, &prompt, err, sizeof(err)) == 1);
    assert(ds4_session_prepare_sync(&session, NULL, NULL, 0) == 1);
    prompt.len = 0;
    assert(ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)) == 1);
    prompt.len = session.ctx_size;
    assert(ds4_session_prepare_sync(&session, &prompt, err, sizeof(err)) == 1);
    assert(calls == before);

    printf("PASS Q4 preflight: %u chunk schedules, descriptors, future-session "
           "reserves, saturation, generation changes, rc -1/0/1 and prompt "
           "resume/vision/CPU/GLM/V4.1 paths and idempotent session counts. "
           "GPU/concurrency unverified.\n", schedules);
    return 0;
}
'''


def main():
    program = SHIM + production_bodies() + DRIVER
    with tempfile.TemporaryDirectory(prefix="ds4-q4-preflight-") as tmp:
        source = Path(tmp) / "preflight.c"
        binary = Path(tmp) / "preflight"
        source.write_text(program)
        subprocess.run(
            shlex.split(os.environ.get("CC", "cc")) +
            ["-std=c11", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
             "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
             "-fno-omit-frame-pointer", "-I", str(ROOT), str(source),
             "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
