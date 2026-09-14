/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ds4_indexer_plan.h"

static uint64_t checks, policy_cases;
#define CHECK(expr) do { \
    ++checks; \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
        exit(1); \
    } \
} while (0)

static ds4_indexer_plan_request request_default(void) {
    const ds4_indexer_plan_request r = {
        .backend = DS4_INDEXER_BACKEND_METAL,
        .phase = DS4_GPU_PHASE_AUTO,
        .n_comp = 4096, .n_tokens = 128, .pos0 = 0,
        .n_head = 64, .head_dim = 128, .ratio = 4, .quality = false,
    };
    return r;
}

static ds4_indexer_plan_caps caps_default(void) {
    const ds4_indexer_plan_caps c = {
        .max_threads = 1024, .max_shared_bytes = 32768,
        .max_grid_x = UINT32_MAX, .max_grid_y = UINT32_MAX,
        .max_buffer_bytes = UINT64_MAX, .nax_available = false,
        .allow_grouped = false, .preferred_head_group = 0,
    };
    return c;
}

static ds4_indexer_plan build(const ds4_indexer_plan_request *r,
                              const ds4_indexer_plan_caps *c) {
    struct guarded {
        uint64_t before;
        ds4_indexer_plan plan;
        uint64_t after;
    } g;
    memset(&g, 0x69, sizeof(g));
    const uint64_t guard = g.before;
    CHECK(ds4_indexer_plan_build(r, c, &g.plan));
    CHECK(g.before == guard && g.after == guard);
    CHECK(g.plan.backend == r->backend && g.plan.phase == r->phase);
    return g.plan;
}

static void decline(const ds4_indexer_plan_request *r,
                    const ds4_indexer_plan_caps *c) {
    ds4_indexer_plan out, original;
    memset(&out, 0xa5, sizeof(out));
    memcpy(&original, &out, sizeof(out));
    CHECK(!ds4_indexer_plan_build(r, c, &out));
    CHECK(memcmp(&original, &out, sizeof(out)) == 0);
}

static void policy_matrix(void) {
    const uint32_t rows[] = {1, 7, 8, 15, 16, 17, 31, 128, 257};
    const uint32_t columns[] = {1, 31, 32, 33, 4096};
    const uint32_t heads[] = {1, 2, 63, 64, 65};
    const uint32_t groups[] = {0, 1, 2, 4};
    for (unsigned backend = 0; backend < 3; ++backend)
    for (unsigned phase = 0; phase < 6; ++phase)
    for (unsigned n = 0; n < sizeof(rows)/sizeof(rows[0]); ++n)
    for (unsigned k = 0; k < sizeof(columns)/sizeof(columns[0]); ++k)
    for (unsigned h = 0; h < sizeof(heads)/sizeof(heads[0]); ++h)
    for (unsigned mode = 0; mode < 8; ++mode)
    for (unsigned group = 0; group < sizeof(groups)/sizeof(groups[0]); ++group) {
        ++policy_cases;
        ds4_indexer_plan_request r = request_default();
        ds4_indexer_plan_caps c = caps_default();
        r.backend = (ds4_indexer_backend)backend;
        r.phase = (ds4_gpu_execution_phase)phase;
        r.n_tokens = rows[n]; r.n_comp = columns[k]; r.n_head = heads[h];
        r.pos0 = 4096; r.quality = (mode & 1) != 0;
        c.nax_available = (mode & 2) != 0;
        c.allow_grouped = (mode & 4) != 0;
        c.preferred_head_group = groups[group];
        const ds4_indexer_plan p = build(&r, &c);
        CHECK(p.q_bytes == (uint64_t)r.n_tokens * r.n_head * 128 * 4);
        CHECK(p.weight_bytes == (uint64_t)r.n_tokens * r.n_head * 4);
        CHECK(p.index_bytes == (uint64_t)r.n_comp * 128 * 4);
        CHECK(p.score_bytes == (uint64_t)r.n_comp * r.n_tokens * 4);
        if (backend) {
            CHECK(p.kernel == DS4_INDEXER_KERNEL_BACKEND_NATIVE);
            CHECK(p.precision == DS4_INDEXER_PRECISION_BACKEND_NATIVE);
            CHECK(p.tile_m == 0 && p.tile_n == 0 && p.head_group == 0 &&
                  p.threads == 0 && p.shared_bytes == 0 &&
                  p.grid_x == 0 && p.grid_y == 0);
            continue;
        }
        CHECK(p.tile_n == 32 && p.threads == 128);
        CHECK(p.grid_x == r.n_comp / 32 + (r.n_comp % 32 != 0));
        CHECK(p.grid_y == r.n_tokens / p.tile_m + (r.n_tokens % p.tile_m != 0));
        if (c.nax_available && r.n_tokens >= 16) {
            CHECK(p.kernel == DS4_INDEXER_KERNEL_METAL_NAX);
            CHECK(p.precision == DS4_INDEXER_PRECISION_F16_F32_ACC);
            CHECK(p.tile_m == 16 && p.head_group == 2 && p.shared_bytes == 16384);
        } else if (r.quality) {
            CHECK(p.kernel == DS4_INDEXER_KERNEL_METAL_TILED_F32);
            CHECK(p.precision == DS4_INDEXER_PRECISION_F32);
            CHECK(p.tile_m == 8 && p.head_group == 1 && p.shared_bytes == 21504);
        } else if (c.allow_grouped && heads[h] == 64 && phase < 2) {
            const uint32_t expected_group = groups[group] ? groups[group] : 2;
            CHECK(p.kernel == (expected_group == 1 ? DS4_INDEXER_KERNEL_METAL_HEAD1 :
                  expected_group == 4 ? DS4_INDEXER_KERNEL_METAL_HEAD4 :
                                        DS4_INDEXER_KERNEL_METAL_HEAD2));
            CHECK(p.precision == DS4_INDEXER_PRECISION_F16_F32_ACC);
            CHECK(p.tile_m == 8 && p.head_group == expected_group);
            CHECK(p.shared_bytes == (expected_group == 1 ? 11264 :
                                    expected_group == 4 ? 20480 : 14336));
        } else {
            CHECK(p.kernel == DS4_INDEXER_KERNEL_METAL_TILED_HALF);
            CHECK(p.precision == DS4_INDEXER_PRECISION_F16_F32_ACC);
            CHECK(p.tile_m == 8 && p.head_group == 1 && p.shared_bytes == 11264);
        }
    }
}

static void invalid_and_bounds(void) {
    ds4_indexer_plan_request r = request_default(), bad;
    ds4_indexer_plan_caps c = caps_default();
    decline(NULL, &c); decline(&r, NULL);
    CHECK(!ds4_indexer_plan_build(&r, &c, NULL));
    bad = r; bad.backend = (ds4_indexer_backend)-1; decline(&bad, &c);
    bad.backend = (ds4_indexer_backend)3; decline(&bad, &c);
    bad = r; bad.phase = (ds4_gpu_execution_phase)-1; decline(&bad, &c);
    bad.phase = (ds4_gpu_execution_phase)6; decline(&bad, &c);
    bad = r; bad.n_comp = 0; decline(&bad, &c);
    bad = r; bad.n_tokens = 0; decline(&bad, &c);
    bad = r; bad.n_head = 0; decline(&bad, &c);
    bad = r; bad.head_dim = 0; decline(&bad, &c);
    bad = r; bad.ratio = 0; decline(&bad, &c);
    bad = r; bad.n_comp = (uint32_t)INT32_MAX + 1; decline(&bad, &c);
    bad = r; bad.n_tokens = (uint32_t)INT32_MAX + 1; decline(&bad, &c);
    bad = r; bad.n_head = (uint32_t)INT32_MAX + 1; decline(&bad, &c);
    bad = r; bad.head_dim = (uint32_t)INT32_MAX + 1; decline(&bad, &c);
    bad = r; bad.pos0 = INT32_MAX - bad.n_tokens + 1; decline(&bad, &c);
    --bad.pos0; (void)build(&bad, &c);
    bad.pos0 = UINT32_MAX; decline(&bad, &c);
    bad = r; bad.head_dim = 64; decline(&bad, &c);
    bad.backend = DS4_INDEXER_BACKEND_CUDA;
    CHECK(build(&bad, &c).q_bytes == (uint64_t)bad.n_tokens * bad.n_head * 64 * 4);
    bad.backend = DS4_INDEXER_BACKEND_HIP; (void)build(&bad, &c);

    c.max_buffer_bytes = 0; decline(&r, &c);
    c = caps_default(); c.allow_grouped = true; c.preferred_head_group = 3;
    decline(&r, &c);
    c.nax_available = true; decline(&r, &c);
    c.allow_grouped = false;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_NAX);

    /* Four bytes are sufficient for the smallest native F32 prefixes. */
    c = caps_default(); c.max_buffer_bytes = 4;
    bad = r; bad.backend = DS4_INDEXER_BACKEND_HIP;
    bad.n_comp = bad.n_tokens = bad.n_head = bad.head_dim = 1;
    const ds4_indexer_plan tiny = build(&bad, &c);
    CHECK(tiny.q_bytes == 4 && tiny.weight_bytes == 4 &&
          tiny.index_bytes == 4 && tiny.score_bytes == 4);
    c.max_buffer_bytes = 3; decline(&bad, &c);

    c = caps_default();
    bad.n_tokens = bad.n_head = INT32_MAX;
    const uint64_t exact_square_bytes = UINT64_C(18446744056529682436);
    CHECK(build(&bad, &c).q_bytes == exact_square_bytes);
    c.max_buffer_bytes = exact_square_bytes - 1; decline(&bad, &c);
    c = caps_default();
    bad.head_dim = 2; decline(&bad, &c); /* F32 byte count overflows. */
    bad.head_dim = INT32_MAX; decline(&bad, &c); /* Element count overflows. */
    bad.n_tokens = bad.n_head = 1; bad.n_comp = INT32_MAX;
    CHECK(build(&bad, &c).index_bytes == exact_square_bytes);
    bad.head_dim = 1; bad.n_tokens = INT32_MAX;
    CHECK(build(&bad, &c).score_bytes == exact_square_bytes);

    /* Padded final grid remains representable at the signed-index limit. */
    bad = r; bad.n_comp = INT32_MAX; bad.n_tokens = INT32_MAX;
    bad.n_head = 1;
    const ds4_indexer_plan huge = build(&bad, &c);
    CHECK(huge.grid_x == 67108864 && huge.grid_y == 268435456);
    CHECK(huge.score_bytes == exact_square_bytes);
    c.max_grid_x = huge.grid_x - 1; decline(&bad, &c);
    c.max_grid_x = huge.grid_x; c.max_grid_y = huge.grid_y - 1; decline(&bad, &c);
    c.max_grid_y = huge.grid_y; (void)build(&bad, &c);
}

static void hardware_limits(void) {
    ds4_indexer_plan_request r = request_default();
    ds4_indexer_plan_caps c = caps_default();
    c.max_threads = 127; decline(&r, &c);
    c.max_threads = 128; (void)build(&r, &c);
    c.max_shared_bytes = 11263; decline(&r, &c);
    c.max_shared_bytes = 11264;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_TILED_HALF);
    c.preferred_head_group = 1;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_TILED_HALF);
    c.allow_grouped = true;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_HEAD1);
    --c.max_shared_bytes; decline(&r, &c);
    c.preferred_head_group = 0;
    c.allow_grouped = true; c.max_shared_bytes = 14335;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_TILED_HALF);
    ++c.max_shared_bytes;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_HEAD2);
    c.preferred_head_group = 4; c.max_shared_bytes = 20479;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_TILED_HALF);
    ++c.max_shared_bytes;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_HEAD4);
    r.quality = true; c.max_shared_bytes = 21503; decline(&r, &c);
    ++c.max_shared_bytes;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_TILED_F32);
    c.nax_available = true; c.max_shared_bytes = 16383; decline(&r, &c);
    ++c.max_shared_bytes;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_METAL_NAX);

    r = request_default(); c = caps_default();
    r.n_comp = 33; r.n_tokens = 17;
    c.max_grid_x = 1; decline(&r, &c);
    c.max_grid_x = 2; c.max_grid_y = 2; decline(&r, &c);
    c.max_grid_y = 3; (void)build(&r, &c);
    c.nax_available = true; c.max_grid_y = 1; decline(&r, &c);
    c.max_grid_y = 2; (void)build(&r, &c);
    c.max_grid_x = 0; decline(&r, &c);
    c.max_grid_x = 2; c.max_grid_y = 0; decline(&r, &c);

    /* Native is only a validated-shape delegation, never a made-up launch. */
    r.backend = DS4_INDEXER_BACKEND_CUDA;
    c.max_threads = c.max_shared_bytes = c.max_grid_x = c.max_grid_y = 0;
    CHECK(build(&r, &c).kernel == DS4_INDEXER_KERNEL_BACKEND_NATIVE);
    r.backend = DS4_INDEXER_BACKEND_HIP; (void)build(&r, &c);
}

int main(void) {
    policy_matrix();
    invalid_and_bounds();
    hardware_limits();
    printf("PASS indexer plan: %" PRIu64 " checks, %" PRIu64 " policy cases; "
           "invalid output unchanged, phase/precision/resource/overflow boundaries\n",
           checks, policy_cases);
    return 0;
}
