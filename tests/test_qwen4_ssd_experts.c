#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>

/* Real SSD reads with a sparse 512-expert file. Only 24 experts contain data;
 * indices above the older 384-expert cache bound participate in every route. */
enum { D = 256, F = 640, E = 512, S = 10, BUDGET = 12, GUARD = 4, HC = 4 };
static const int32_t active[] = {511, 384, 383, 0, 1, 2, 3, 4, 5, 6,
                                7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const float poison = -1234.5f;
static uint32_t rng = 123;
typedef struct {
    uint32_t gate_type, down_type, shared_down_type;
    ds4_gpu_stream_expert_table table;
    uint64_t shared_gate, shared_up, shared_down;
} weights;
typedef struct { ds4_gpu_tensor *base, *view; uint64_t n; } guarded;

/* This binary links a test-only backend object whose pread calls reach this
 * wrapper. All accepted requests still perform real file I/O. Sparse holes
 * alone cannot prove selected-only reads: reading a hole succeeds with zeros. */
static pthread_mutex_t probe_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct {
    int enabled, fd, empty_fd, violation, fail_projection, require_gate_up_first;
    uint32_t inflight;
    uint64_t offset[3], stride[3], bytes, calls, coverage[3][E];
    unsigned char selected[E];
} probe;
typedef struct { uint64_t bytes, calls; } read_stats;

static void need(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "Qwen SSD: %s failed\n", what); exit(1); }
}
ssize_t ds4_test_pread(int fd, void *dst, size_t bytes, off_t offset) {
    pthread_mutex_lock(&probe_mutex);
    if (!probe.enabled || (fd != probe.fd && fd != probe.empty_fd)) {
        pthread_mutex_unlock(&probe_mutex);
        return pread(fd, dst, bytes, offset);
    }
    int projection = -1;
    const uint64_t start = (uint64_t)offset;
    if (offset >= 0 && bytes) for (unsigned p = 0; p < 3; p++) {
        if (start < probe.offset[p]) continue;
        const uint64_t relative = start - probe.offset[p], total = E * probe.stride[p];
        if (relative >= total || bytes > total - relative) continue;
        projection = (int)p;
        const uint32_t first = (uint32_t)(relative / probe.stride[p]);
        const uint32_t last = (uint32_t)((relative + bytes - 1) / probe.stride[p]);
        for (uint32_t e = first; e <= last; e++) if (!probe.selected[e]) projection = -1;
        break;
    }
    probe.calls++;
    if (projection < 0) {
        probe.violation = 1;
        pthread_mutex_unlock(&probe_mutex);
        errno = EIO; return -1;
    }
    if (projection == 2 && probe.require_gate_up_first) {
        for (unsigned e = 0; e < E; e++) if (probe.selected[e]) {
            if (probe.coverage[0][e] != probe.stride[0] ||
                probe.coverage[1][e] != probe.stride[1]) probe.violation = 1;
        }
    }
    if (projection == probe.fail_projection) {
        pthread_mutex_unlock(&probe_mutex);
        errno = EIO; return -1;
    }
    probe.inflight++;
    pthread_mutex_unlock(&probe_mutex);
    const ssize_t got = pread(fd, dst, bytes, offset);
    const int saved_errno = errno;
    pthread_mutex_lock(&probe_mutex);
    probe.inflight--;
    if (got > 0) {
        probe.bytes += (uint64_t)got;
        uint64_t relative = start - probe.offset[projection], remaining = (uint64_t)got;
        const uint64_t stride = probe.stride[projection];
        while (remaining) {
            const uint32_t e = (uint32_t)(relative / stride);
            const uint64_t room = stride - relative % stride, n = remaining < room ? remaining : room;
            probe.coverage[projection][e] += n; remaining -= n; relative += n;
        }
    }
    pthread_mutex_unlock(&probe_mutex);
    errno = saved_errno; return got;
}
static void probe_begin(const weights *w, const int32_t *ids, uint64_t n, int fd, int empty_fd) {
    pthread_mutex_lock(&probe_mutex);
    need(!probe.enabled && !probe.inflight, "read probe idle");
    memset(&probe, 0, sizeof(probe));
    probe.fd = fd; probe.empty_fd = empty_fd; probe.enabled = 1;
    probe.fail_projection = -1;
    probe.offset[0] = w->table.gate_offset; probe.offset[1] = w->table.up_offset;
    probe.offset[2] = w->table.down_offset;
    probe.stride[0] = probe.stride[1] = w->table.gate_expert_bytes;
    probe.stride[2] = w->table.down_expert_bytes;
    for (uint64_t i = 0; i < n; i++) {
        need(ids[i] >= 0 && ids[i] < E, "probe selected ID"); probe.selected[ids[i]] = 1;
    }
    pthread_mutex_unlock(&probe_mutex);
}
static read_stats probe_end(int all_selected, int no_reads) {
    pthread_mutex_lock(&probe_mutex);
    need(probe.enabled && !probe.inflight, "read workers retired");
    need(!probe.violation, "pread never accesses an unselected expert");
    if (no_reads) need(!probe.calls && !probe.bytes, "resident/cache-hit path performs no pread");
    for (unsigned p = 0; p < 3; p++) for (unsigned e = 0; e < E; e++) {
        const uint64_t expected = probe.selected[e] ? probe.stride[p] : 0;
        need(probe.coverage[p][e] <= expected, "expert bytes read at most once");
        if (all_selected) need(probe.coverage[p][e] == expected, "each selected projection fully read");
    }
    const read_stats result = {probe.bytes, probe.calls}; probe.enabled = 0;
    pthread_mutex_unlock(&probe_mutex); return result;
}
static uint32_t random_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }
static uint32_t block_bytes(uint32_t type) {
    switch (type) { case 8: return 34; case 10: return 84; case 12: return 144;
                    case 16: return 66; case 39: return 17; default: abort(); }
}
static uint64_t row_bytes(uint32_t type, uint32_t n) {
    const uint32_t block = type == 8 || type == 39 ? 32 : 256;
    return ((uint64_t)n + block - 1) / block * block_bytes(type);
}
static void half_at(uint8_t *p, uint16_t h) { memcpy(p, &h, sizeof(h)); }
static void fill_weights(uint8_t *p, uint64_t bytes, uint32_t type) {
    const uint32_t block = block_bytes(type);
    for (uint64_t off = 0; off < bytes; off += block) {
        uint8_t *b = p + off;
        for (uint32_t j = 0; j < block; j++) b[j] = random_u32();
        const uint16_t scale = 0x1800u + (random_u32() % 5u) * 0x100u;
        if (type == 39) b[0] = 118u + random_u32() % 5u;
        else if (type == 10) { half_at(b + 80, scale); half_at(b + 82, 0x1400); }
        else { half_at(b, scale); if (type == 12) half_at(b + 2, 0x1400); }
    }
}
static ds4_gpu_tensor *upload(const void *data, uint64_t bytes) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    need(t != NULL, "tensor allocation");
    if (data) need(ds4_gpu_tensor_write(t, 0, data, bytes), "tensor upload");
    return t;
}
static guarded output(uint64_t n) {
    guarded t = {upload(NULL, (n + 2 * GUARD) * sizeof(float)), NULL, n};
    t.view = ds4_gpu_tensor_view(t.base, GUARD * sizeof(float), n * sizeof(float));
    need(t.view != NULL, "guarded view"); return t;
}
static void reset(guarded *t) {
    need(ds4_gpu_tensor_fill_f32(t->base, poison, t->n + 2 * GUARD), "output poison");
}
static float *read_output(const guarded *t) {
    const uint64_t n = t->n + 2 * GUARD;
    float *v = malloc(n * sizeof(float)); need(v != NULL, "host output");
    need(ds4_gpu_tensor_read(t->base, 0, v, n * sizeof(float)), "read output");
    for (uint64_t i = 0; i < n; i++) {
        if (i < GUARD || i >= GUARD + t->n)
            need(memcmp(v + i, &poison, sizeof(float)) == 0, "output guards");
        else need(isfinite(v[i]), "finite output");
    }
    return v;
}
static void exact(const char *stage, const float *ref, const float *got, uint64_t n) {
    if (!memcmp(ref, got, n * sizeof(float))) return;
    for (uint64_t i = 0; i < n; i++) if (memcmp(ref + i, got + i, sizeof(float))) {
        uint32_t a, b; memcpy(&a, ref + i, 4); memcpy(&b, got + i, 4);
        fprintf(stderr, "Qwen SSD %s mismatch at %llu: %08x != %08x\n",
                stage, (unsigned long long)i, a, b); exit(1);
    }
}
static void free_output(guarded *t) {
    ds4_gpu_tensor_free(t->view); ds4_gpu_tensor_free(t->base);
}

static const char *mm_env[] = {"DS4_QWEN4_MOE_MID_TILES", "DS4_QWEN4_MOE_DOWN_TILES",
                               "DS4_QWEN4_MOE_MID_NT", "DS4_QWEN4_MOE_DOWN_NT",
                               "DS4_QWEN4_MOE_MM_SPECIALIZE"};
static void mm_variant(unsigned nt) {
    for (unsigned i = 0; i < 5; i++) {
        const char *value = i == 4 ? "0" : i < 2 || nt == 4 ? "4" : "1";
        need((nt ? setenv(mm_env[i], value, 1) : unsetenv(mm_env[i])) == 0,
             "select generic full-grid/default MM dispatch");
    }
}

/* Check the real GPU-built lists against the frozen router IDs, without
 * assuming the order in which GPU atomics append a token's slot. */
static void check_mm_lists(const int32_t *ids, uint32_t T, unsigned route,
                           const float *lists, const float *counts) {
    uint32_t freq[E] = {0}, hot = 0, tail = 0, inactive = 0;
    unsigned char *seen = calloc((size_t)T * S, 1);
    need(seen != NULL, "list membership scratch");
    for (uint32_t p = 0; p < T * S; p++) freq[ids[p]]++;
    for (uint32_t e = 0; e < E; e++) {
        int32_t count; memcpy(&count, counts + GUARD + e, 4);
        need(count >= 0 && (uint32_t)count == freq[e], "GPU list frequency matches router IDs");
        hot += count > 8; tail += count > 0 && count < 8; inactive += count == 0;
        for (int32_t j = 0; j < count; j++) {
            int32_t p; memcpy(&p, lists + GUARD + (uint64_t)e * T + (uint32_t)j, 4);
            need(p >= 0 && (uint32_t)p < T * S && ids[p] == (int32_t)e && !seen[p],
                 "GPU lists contain every routed slot exactly once");
            seen[p] = 1;
        }
    }
    for (uint32_t p = 0; p < T * S; p++) need(seen[p], "complete GPU list membership");
    if (route == 4) need(hot && tail && inactive > 400, "skewed lists span hot, tail and inactive experts");
    free(seen);
}

static int project_rows(const weights *w, uint32_t T, uint32_t slots, int has_shared,
                        int streamed, ds4_gpu_tensor *mid, ds4_gpu_tensor *part,
                        ds4_gpu_tensor *x, ds4_gpu_tensor *ids) {
    const uint32_t shared = has_shared ? 8u : UINT32_MAX;
    const uint32_t shared_down = has_shared ? w->shared_down_type : UINT32_MAX;
    const ds4_gpu_stream_expert_table *a = &w->table;
    if (streamed) return ds4_gpu_qwen4_moe_stream_tensor(mid, part, x, ids, NULL, NULL,
        a, w->gate_type, w->down_type, T, slots, D, F, T,
        w->shared_gate, w->shared_up, w->shared_down, shared, shared_down);
    return ds4_gpu_qwen4_moe_mid_tensor(mid, x, ids, a->model_map, a->model_size,
        a->gate_offset, a->up_offset, w->gate_type, E, T, slots, D, F,
        w->shared_gate, w->shared_up, shared) &&
        ds4_gpu_qwen4_moe_down_tensor(part, mid, ids, a->model_map, a->model_size,
        a->down_offset, w->down_type, E, T, slots, F, D, w->shared_down, shared_down);
}

static int project(const weights *w, uint32_t T, int streamed,
                   ds4_gpu_tensor *mid, ds4_gpu_tensor *part, ds4_gpu_tensor *x,
                   ds4_gpu_tensor *ids, ds4_gpu_tensor *lists, ds4_gpu_tensor *counts) {
    const int mm = lists != NULL;
    if (!mm) return project_rows(w, T, S, 1, streamed, mid, part, x, ids);
    const ds4_gpu_stream_expert_table *a = &w->table;
    if (streamed) return ds4_gpu_qwen4_moe_stream_tensor(mid, part, x, ids, lists, counts,
        a, w->gate_type, w->down_type, T, S, D, F, T,
        w->shared_gate, w->shared_up, w->shared_down, UINT32_MAX, UINT32_MAX);
    return ds4_gpu_qwen4_moe_mm_mid_tensor(mid, x, lists, counts,
        a->model_map, a->model_size, a->gate_offset, a->up_offset, w->gate_type,
        E, T, S, S, D, F, T) &&
        ds4_gpu_qwen4_moe_mm_down_tensor(part, mid, lists, counts,
        a->model_map, a->model_size, a->down_offset, w->down_type, E, T, S, S, F, D, T);
}

static int project_split(const weights *w, uint32_t T, uint32_t slots, int shared,
                         int streamed, ds4_gpu_tensor *mid, ds4_gpu_tensor *part,
                         ds4_gpu_tensor *x, ds4_gpu_tensor *ids,
                         ds4_gpu_tensor *lists, ds4_gpu_tensor *counts) {
    if (!lists) return project_rows(w, T, slots, shared, streamed, mid, part, x, ids);
    need(slots == S && !shared, "routed-only MM split shape");
    return project(w, T, streamed, mid, part, x, ids, lists, counts);
}

/* The automatic prefill overlap currently targets M1 Max. Other devices
 * still run these numerical and I/O cases, but have no prefix on read failure. */
static int mm_overlap_expected(void) {
    char brand[128] = {0}; size_t bytes = sizeof(brand);
    need(sysctlbyname("machdep.cpu.brand_string", brand, &bytes, NULL, 0) == 0,
         "read MM overlap device policy");
    return strstr(brand, "M1 Max") != NULL;
}

/* Both arms use the same row/MM arithmetic and the same final reduction/HC.
 * Prefill's shared dense expert is outside this routed-only API. Decode also
 * checks the shared slot, including mixed Q8/MXFP4 weights and its reduction. */
static void check_case_impl(const weights *w, uint32_t T, unsigned route, int good_fd,
                            int empty_fd, int fail_first, int warm_probe,
                            uint32_t reuse_seed_count) {
    const int mm = T > 8;
    const int compare_mm = mm && T <= 128 && w->gate_type == 16 && w->down_type == 10;
    char *saved_mm_env[5] = {NULL, NULL, NULL, NULL, NULL};
    if (compare_mm) {
        for (unsigned i = 0; i < 5; i++) {
            const char *v = getenv(mm_env[i]);
            saved_mm_env[i] = v ? strdup(v) : NULL;
            need(!v || saved_mm_env[i], "save MM dispatch environment");
        }
        mm_variant(0);
    }
    const uint32_t stride = S + !mm;
    const uint64_t xn = (uint64_t)T * D, sn = (uint64_t)T * S;
    const uint64_t rn = xn * HC, in = (uint64_t)T * HC * DS4_QWEN4_HC_CHUNKS * HC;
    rng = 123u + T * 17u + route;
    float *x = malloc(xn * 4), *mix = malloc(sn * 4), *r = malloc(rn * 4);
    float *inj = malloc(in * 4), *sg = malloc(T * 4);
    int32_t *ids = malloc(sn * 4);
    need(x && mix && r && inj && sg && ids, "host inputs");
    for (uint64_t i = 0; i < xn; i++) x[i] = ((int)(random_u32() % 257) - 128) / 1024.f;
    for (uint64_t i = 0; i < rn; i++) r[i] = ((int)(random_u32() % 257) - 128) / 256.f;
    for (uint64_t i = 0; i < in; i++) inj[i] = ((int)(random_u32() % 257) - 128) / 1024.f;
    for (uint32_t t = 0; t < T; t++) {
        sg[t] = ((int)(t % 7) - 3) * 0.25f;
        for (uint32_t s = 0; s < S; s++) {
            ids[t * S + s] = route == 3 ? (int32_t)((t * S + s) % E) :
                route == 4 ? active[s < 8 ? s : 8 + (2 * t + s - 8) % 16] :
                active[route == 2 ? (t + s) % 24 : route * 10 + (t + s) % 10];
            mix[t * S + s] = (s + 1) / 55.f;
        }
    }
    ds4_gpu_tensor *gx = upload(x, xn * 4), *gi = upload(ids, sn * 4);
    ds4_gpu_tensor *gw = upload(mix, sn * 4), *ginj = upload(inj, in * 4), *gsg = upload(sg, T * 4);
    guarded routing[2] = {{0}, {0}};
    float *routing_reference[2] = {NULL, NULL};
    if (mm) {
        routing[0] = output((uint64_t)E * T); routing[1] = output(E);
        reset(routing); reset(routing + 1);
    }
    ds4_gpu_tensor *lists = routing[0].view, *counts = routing[1].view;
    guarded outputs[] = {output((uint64_t)T * stride * F), output((uint64_t)T * stride * D),
                         output(xn), output(rn)};
    const char *stages[] = {"mid", "down", "reduce", "HC residual"};
    float *reference[4] = {NULL, NULL, NULL, NULL};
    if (mm) {
        need(ds4_gpu_qwen4_moe_build_lists_tensor(lists, counts, gi, T, S, E, T), "expert lists");
        for (unsigned i = 0; i < 2; i++) routing_reference[i] = read_output(routing + i);
        check_mm_lists(ids, T, route, routing_reference[0], routing_reference[1]);
    }
    ds4_gpu_qwen4_set_verify_rows_exact(T == 3);
    if (w->gate_type == 16 && T == 1 && route == 0) {
        weights invalid = *w; invalid.table.down_expert_bytes--;
        for (unsigned i = 0; i < 4; i++) reset(outputs + i);
        need(ds4_gpu_begin_commands(), "invalid-size batch begin");
        need(!project(&invalid, T, 1, outputs[0].view, outputs[1].view, gx, gi, NULL, NULL),
             "reject wrong physical Q2 row size");
        need(ds4_gpu_commands_active() && ds4_gpu_end_commands(), "invalid-size batch ownership");
        const int32_t invalid_id = E;
        need(ds4_gpu_tensor_write(gi, 0, &invalid_id, 4) && ds4_gpu_begin_commands(), "invalid-ID batch begin");
        need(!project(w, T, 1, outputs[0].view, outputs[1].view, gx, gi, NULL, NULL), "reject expert ID512");
        need(ds4_gpu_commands_active() && ds4_gpu_end_commands(), "invalid-ID batch ownership");
        need(ds4_gpu_tensor_write(gi, 0, ids, sn * 4), "restore valid IDs");
        for (unsigned i = 0; i < 4; i++) {
            float *got = read_output(outputs + i);
            for (uint64_t j = 0; j < outputs[i].n + 2 * GUARD; j++)
                need(!memcmp(got + j, &poison, 4), "invalid request leaves output untouched");
            free(got);
        }
    }
    if (fail_first) {
        need(ds4_gpu_set_model_fd(empty_fd), "empty input fd");
        need(ds4_gpu_begin_commands(), "failure batch begin");
        probe_begin(w, ids, sn, good_fd, empty_fd);
        const int accepted = project(w, T, 1, outputs[0].view, outputs[1].view, gx, gi, lists, counts);
        need(!accepted, "uncached read must fail on empty file");
        need(ds4_gpu_commands_active(), "failure preserves active batch ownership");
        need(ds4_gpu_end_commands(), "failure batch drain");
        (void)probe_end(0, 0);
        need(ds4_gpu_set_model_fd(good_fd), "restore input fd after failure");
    }
    const unsigned variants[] = {4, 0, 4, 1, 0, 1};
    const unsigned runs = compare_mm ? 6 : warm_probe ? 3 : 2;
    read_stats reads[6] = {{0}};
    for (unsigned run = 0; run < runs; run++) {
        if (run == 1 && reuse_seed_count) {
            /* Seed only after the independent resident oracle. The next SSD
             * call must evict 24 same-size entries for its disjoint route2
             * union, exercising the plan beyond the older 16-entry bound. */
            need(reuse_seed_count == 24 && route == 2 && warm_probe && !fail_first,
                 "full-cache reuse fixture shape");
            int32_t seeds[24];
            for (uint32_t i = 0; i < reuse_seed_count; i++) {
                seeds[i] = (int32_t)(32u + i);
                for (uint64_t j = 0; j < sn; j++) need(seeds[i] != ids[j], "reuse seed is unselected");
            }
            ds4_gpu_set_streaming_expert_cache_budget(reuse_seed_count);
            ds4_gpu_set_streaming_expert_cache_expert_bytes(
                2 * w->table.gate_expert_bytes + w->table.down_expert_bytes);
            need(ds4_gpu_stream_expert_cache_current_count() == 0, "reuse seed starts empty");
            probe_begin(w, seeds, reuse_seed_count, good_fd, empty_fd);
            need(ds4_gpu_stream_expert_cache_seed_experts(&w->table, seeds, NULL,
                 reuse_seed_count), "prime disjoint full cache");
            (void)probe_end(1, 0);
            need(ds4_gpu_stream_expert_cache_current_count() == reuse_seed_count,
                 "reuse cache full before selected batch");
        }
        /* Preserve the independent generic resident reference. Low-bit MM runs
         * default SSD cold -> old NT4 SSD on these same buffers, followed by
         * NT1/grid4 -> default -> NT1/grid4. This covers both automatic tile
         * widths and specialization against forced generic kernels. Warm
         * cases remove the readable fd; overflow cases must reread only the
         * selected union on every arm. TILES alone does not force old NT4. */
        if (compare_mm) mm_variant(variants[run]);
        const int warm = warm_probe && run >= 2;
        if (warm) need(ds4_gpu_set_model_fd(empty_fd), "warm-cache empty fd");
        for (unsigned i = 0; i < 4; i++) reset(outputs + i);
        need(ds4_gpu_tensor_write(outputs[3].view, 0, r, rn * 4), "reset HC input");
        need(ds4_gpu_begin_commands(), "projection batch begin");
        probe_begin(w, ids, sn, good_fd, empty_fd);
        need(project(w, T, run != 0, outputs[0].view, outputs[1].view, gx, gi, lists, counts), "resident/SSD projection");
        need(ds4_gpu_commands_active(), "projection preserves active batch ownership");
        need(ds4_gpu_qwen4_moe_reduce_tensor(outputs[2].view, outputs[1].view, gw,
            mm ? NULL : gsg, NULL, outputs[3].view, ginj, T, S, stride, D, HC), "reduce and HC");
        need(ds4_gpu_end_commands(), "projection batch drain");
        reads[run] = probe_end(run && (!warm_probe || (reuse_seed_count && run == 1)),
                               run == 0 || warm);
        for (unsigned i = 0; i < 4; i++) {
            float *got = read_output(outputs + i);
            if (!run) reference[i] = got;
            else { exact(stages[i], reference[i], got, outputs[i].n + 2 * GUARD); free(got); }
        }
        if (mm) for (unsigned i = 0; i < 2; i++) {
            float *got = read_output(routing + i);
            exact("frozen lists/counts and guards", routing_reference[i], got, routing[i].n + 2 * GUARD);
            free(got);
        }
        const uint32_t cached = ds4_gpu_stream_expert_cache_current_count();
        need(cached <= (reuse_seed_count ? reuse_seed_count : BUDGET), "bounded cache count");
        if (run && warm_probe) need(cached >= S, "selected experts cached");
        if (run && reuse_seed_count) need(cached == reuse_seed_count, "full selected union remains cached");
    }
    need(ds4_gpu_set_model_fd(good_fd), "restore input fd");
    float *read_x = malloc(xn * 4); int32_t *read_ids = malloc(sn * 4);
    need(read_x && read_ids && ds4_gpu_tensor_read(gx, 0, read_x, xn * 4) &&
         ds4_gpu_tensor_read(gi, 0, read_ids, sn * 4), "input integrity read");
    need(!memcmp(x, read_x, xn * 4) && !memcmp(ids, read_ids, sn * 4), "input and selected IDs unchanged");
    printf("PASS Qwen SSD %u/%u T=%u %s route=%u: exact mid/down/reduce/HC, guards; pread=%llu bytes/%llu calls%s%s%s\n",
           w->gate_type, w->down_type, T, mm ? "MM" : "rows", route,
           (unsigned long long)reads[1].bytes, (unsigned long long)reads[1].calls,
           warm_probe ? ", warm hits without readable fd" : "",
           fail_first ? ", failed read and exact retry" : "",
           compare_mm ? ", generic NT4/grid4 reference exact; generic NT1/grid4/default/generic NT1/grid4 on same buffers" : "");
    for (unsigned i = 0; i < 4; i++) { free(reference[i]); free_output(outputs + i); }
    ds4_gpu_qwen4_set_verify_rows_exact(false);
    ds4_gpu_tensor_free(gx); ds4_gpu_tensor_free(gi); ds4_gpu_tensor_free(gw);
    ds4_gpu_tensor_free(ginj); ds4_gpu_tensor_free(gsg);
    if (mm) for (unsigned i = 0; i < 2; i++) { free(routing_reference[i]); free_output(routing + i); }
    if (compare_mm) for (unsigned i = 0; i < 5; i++) {
        need((saved_mm_env[i] ? setenv(mm_env[i], saved_mm_env[i], 1) : unsetenv(mm_env[i])) == 0,
             "restore MM dispatch environment");
        free(saved_mm_env[i]);
    }
    free(x); free(mix); free(r); free(inj); free(sg); free(ids); free(read_x); free(read_ids);
}

static void check_case(const weights *w, uint32_t T, unsigned route, int good_fd,
                       int empty_fd, int fail_first, int warm_probe) {
    check_case_impl(w, T, route, good_fd, empty_fd, fail_first, warm_probe, 0);
}

/* Exercise row masks and MM count partitions with real cache hits. The read
 * allowlist contains only missing experts, so slots cannot duplicate I/O.
 * On read failure, completed resident/shared mid rows prove that the early
 * GPU dispatch actually ran; missing rows and final reduction must remain untouched. */
static void check_split_one(const weights *w, uint32_t T, uint32_t slots, int shared,
                            uint32_t seed_count, int duplicates, int fail_first,
                            int good_fd, int empty_fd) {
    enum { MAX_SLOTS = 16, INJ = HC * DS4_QWEN4_HC_CHUNKS * HC };
    const int mm = T > 8;
    need(slots <= MAX_SLOTS && slots >= 4 &&
         (mm ? T <= 128 && slots == S && !shared && !duplicates && (seed_count == 0 || seed_count == 12) :
               T >= 1 && T <= 3 && seed_count <= 4), "split fixture shape");
    const uint32_t unique = mm ? 24u : slots - (duplicates ? 3u : 0u), stride = slots + shared;
    const uint32_t budget = unique > BUDGET ? unique : BUDGET;
    int32_t *ids = malloc((size_t)T * slots * sizeof(*ids)), seeds[24], missing[24];
    unsigned char cached[E] = {0};
    float *x = malloc((size_t)T * D * sizeof(*x)), *mix = malloc((size_t)T * slots * sizeof(*mix));
    float *r = malloc((size_t)T * D * HC * sizeof(*r)), *inj = malloc((size_t)T * INJ * sizeof(*inj));
    float *shared_gate = malloc((size_t)T * sizeof(*shared_gate));
    need(ids && x && mix && r && inj && shared_gate, "split host inputs");
    for (uint32_t t = 0; t < T; t++) {
        const float values[] = {0.375f, -0.125f, 0.5f};
        shared_gate[t] = values[t % 3u];
    }
    rng = 0x5618u + slots * 3u + w->gate_type;
    for (uint32_t i = 0; i < T * D; i++) x[i] = ((int)(random_u32() % 257) - 128) / 1024.f;
    for (uint32_t i = 0; i < T * D * HC; i++) r[i] = ((int)(random_u32() % 257) - 128) / 256.f;
    for (uint32_t i = 0; i < T * INJ; i++) inj[i] = ((int)(random_u32() % 257) - 128) / 1024.f;
    for (uint32_t i = 0; i < seed_count; i++) { seeds[i] = active[2 * i]; cached[seeds[i]] = 1; }
    for (uint32_t t = 0; t < T; t++) for (uint32_t s = 0; s < slots; s++) {
        uint32_t e = s % unique;
        if (mm) {
            /* Eight hot experts coexist with small tails. Past row 32,
             * concentrate the remaining slots so T128 still has 1..8 tails.
             * Distinct IDs within each token keep all counts <= list_cap. */
            e = s < 8u ? s : t < 33u ? 8u + (2u * t + s - 8u) % 16u : s;
            /* T33 keeps eight counts at 33 to exercise automatic NT1.
             * Other batches also contain all-missing/all-resident rows. */
            if (T != 33u && t == 1u && seed_count) e = 2u * s + 1u;
            if (T != 33u && t == 2u && seed_count) e = 2u * s;
        } else if (t == 1 && seed_count) {
            /* This row has no routed residents; its empty prefix mask must
             * skip all slots when there is no shared expert. */
            while (cached[active[e]]) e = (e + 1u) % unique;
        } else if (t == 2 && seed_count) {
            /* All resident: its empty suffix mask must leave the completed
             * mid/down outputs untouched while the other rows finish. */
            e = 2u * (s % seed_count);
        }
        ids[t * slots + s] = active[e];
        mix[t * slots + s] = (s + 1.f) / (slots * (slots + 1.f) / 2.f);
    }
    guarded inputs[] = {output(T * D), output(T * slots), output(T * slots), output(T), output(T * INJ)};
    const void *host[] = {x, ids, mix, shared_gate, inj};
    float *input_copy[5];
    for (unsigned i = 0; i < 5; i++) {
        reset(inputs + i);
        need(ds4_gpu_tensor_write(inputs[i].view, 0, host[i], inputs[i].n * 4), "split guarded input");
        input_copy[i] = read_output(inputs + i);
    }
    guarded routing[2] = {{0}, {0}};
    float *routing_copy[2] = {NULL, NULL};
    if (mm) {
        routing[0] = output((uint64_t)E * T); routing[1] = output(E);
        reset(routing); reset(routing + 1);
        need(ds4_gpu_qwen4_moe_build_lists_tensor(routing[0].view, routing[1].view,
             inputs[1].view, T, slots, E, T), "split GPU lists");
        for (unsigned i = 0; i < 2; i++) routing_copy[i] = read_output(routing + i);
        check_mm_lists(ids, T, 4, routing_copy[0], routing_copy[1]);
    }
    ds4_gpu_tensor *lists = routing[0].view, *counts = routing[1].view;
    guarded outputs[] = {output((uint64_t)T * stride * F), output((uint64_t)T * stride * D), output(T * D), output(T * D * HC)};
    const char *stages[] = {"split mid", "split down", "split reduce", "split HC"};
    float *reference[4];
    for (unsigned i = 0; i < 4; i++) reset(outputs + i);
    need(ds4_gpu_tensor_write(outputs[3].view, 0, r, T * D * HC * sizeof(float)), "split reference residual");
    need(ds4_gpu_begin_commands() && project_split(w, T, slots, shared, 0,
         outputs[0].view, outputs[1].view, inputs[0].view, inputs[1].view, lists, counts), "split resident reference");
    need(ds4_gpu_qwen4_moe_reduce_tensor(outputs[2].view, outputs[1].view, inputs[2].view,
         shared ? inputs[3].view : NULL, NULL, outputs[3].view, inputs[4].view,
         T, slots, stride, D, HC) && ds4_gpu_end_commands(), "split reference reduce");
    for (unsigned i = 0; i < 4; i++) reference[i] = read_output(outputs + i);

    ds4_gpu_set_streaming_expert_cache_budget(budget);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * w->table.gate_expert_bytes + w->table.down_expert_bytes);
    need(ds4_gpu_set_model_fd(good_fd), "split source fd");
    if (seed_count) need(ds4_gpu_stream_expert_cache_seed_experts(&w->table, seeds, NULL, seed_count), "split cache seed");
    need(ds4_gpu_stream_expert_cache_current_count() == seed_count, "split exact initial cache");
    for (unsigned failure = 0; failure < (fail_first == 2 ? 4u : !!fail_first); failure++) {
        uint32_t n_missing = 0;
        for (uint32_t i = 0; i < unique; i++) if (!cached[active[i]]) missing[n_missing++] = active[i];
        for (unsigned i = 0; i < 4; i++) reset(outputs + i);
        need(ds4_gpu_tensor_write(outputs[3].view, 0, r, T * D * HC * sizeof(float)), "split failure residual");
        need(ds4_gpu_set_model_fd(fail_first == 2 ? good_fd : empty_fd) &&
             ds4_gpu_begin_commands(), "split failed-read begin");
        probe_begin(w, missing, n_missing, good_fd, empty_fd);
        if (fail_first == 2) {
            pthread_mutex_lock(&probe_mutex);
            probe.fail_projection = 2;
            probe.require_gate_up_first = mm_overlap_expected();
            pthread_mutex_unlock(&probe_mutex);
        }
        need(!project_split(w, T, slots, shared, 1, outputs[0].view, outputs[1].view,
                            inputs[0].view, inputs[1].view, lists, counts), "split read failure");
        need(ds4_gpu_commands_active() && ds4_gpu_end_commands(), "split failure drains and preserves batch");
        const read_stats reads = probe_end(0, 0);
        need(reads.calls && reads.bytes == (fail_first == 2 ?
             (uint64_t)n_missing * 2u * w->table.gate_expert_bytes : 0u),
             "split failure consumes only complete gate/up or no payload");
        const int prefix = !mm || mm_overlap_expected();
        for (unsigned stage = 0; stage < 2; stage++) {
            const uint32_t dim = stage ? D : F;
            float *got = read_output(outputs + stage);
            for (uint32_t t = 0; t < T; t++) for (uint32_t s = 0; s < stride; s++) {
                const uint64_t off = GUARD + ((uint64_t)t * stride + s) * dim;
                if (prefix && (s == slots || cached[ids[t * slots + s]] ||
                               (stage == 0 && mm && fail_first == 2)))
                    exact("completed early mid/down slot", reference[stage] + off, got + off, dim);
                else for (uint32_t i = 0; i < dim; i++)
                    need(!memcmp(got + off + i, &poison, 4), "unavailable projection untouched on failure");
            }
            exact("failed prefix leading guard", reference[stage], got, GUARD);
            exact("failed prefix trailing guard", reference[stage] + GUARD + outputs[stage].n,
                  got + GUARD + outputs[stage].n, GUARD);
            free(got);
        }
        float *reduced = read_output(outputs + 2);
        for (uint64_t i = 0; i < outputs[2].n + 2 * GUARD; i++)
            need(!memcmp(reduced + i, &poison, 4), "reduce untouched on read failure");
        free(reduced);
        float *residual = read_output(outputs + 3);
        exact("residual unchanged on read failure", r, residual + GUARD, T * D * HC); free(residual);
        /* Existing cache cleanup may retire the whole layer after I/O failure.
         * The retry must account for those formerly hot experts becoming misses. */
        const uint32_t left = ds4_gpu_stream_expert_cache_current_count();
        need(left == 0 || left == seed_count, "failed-read cache cleanup");
        if (!left) memset(cached, 0, sizeof(cached));
    }
    read_stats first = {0, 0};
    for (unsigned run = 0; run < 2; run++) {
        uint32_t n_missing = 0;
        for (uint32_t i = 0; i < unique; i++) if (!cached[active[i]]) missing[n_missing++] = active[i];
        need(ds4_gpu_set_model_fd(run ? empty_fd : good_fd), "split retry/warm fd");
        for (unsigned i = 0; i < 4; i++) reset(outputs + i);
        need(ds4_gpu_tensor_write(outputs[3].view, 0, r, T * D * HC * sizeof(float)), "split reset residual");
        need(ds4_gpu_begin_commands(), "split projection begin");
        probe_begin(w, missing, n_missing, good_fd, empty_fd);
        need(project_split(w, T, slots, shared, 1, outputs[0].view, outputs[1].view,
                           inputs[0].view, inputs[1].view, lists, counts), "split retry/warm projection");
        need(ds4_gpu_commands_active(), "split projection batch ownership");
        need(ds4_gpu_qwen4_moe_reduce_tensor(outputs[2].view, outputs[1].view, inputs[2].view,
             shared ? inputs[3].view : NULL, NULL, outputs[3].view, inputs[4].view,
             T, slots, stride, D, HC) && ds4_gpu_end_commands(), "split retry/warm reduce");
        const read_stats reads = probe_end(1, !n_missing);
        if (!run) first = reads;
        for (unsigned i = 0; i < 4; i++) {
            float *got = read_output(outputs + i);
            exact(stages[i], reference[i], got, outputs[i].n + 2 * GUARD); free(got);
        }
        need(ds4_gpu_stream_expert_cache_current_count() == unique, "split final cache contains unique experts");
        for (uint32_t i = 0; i < unique; i++) cached[active[i]] = 1;
    }
    need(ds4_gpu_set_model_fd(good_fd), "split restore fd");
    for (unsigned i = 0; i < 5; i++) {
        float *got = read_output(inputs + i);
        exact("split frozen input and guards", input_copy[i], got, inputs[i].n + 2 * GUARD);
        free(got); free(input_copy[i]); free_output(inputs + i);
    }
    if (mm) for (unsigned i = 0; i < 2; i++) {
        float *got = read_output(routing + i);
        exact("split frozen lists/counts and guards", routing_copy[i], got, routing[i].n + 2 * GUARD);
        free(got); free(routing_copy[i]); free_output(routing + i);
    }
    for (unsigned i = 0; i < 4; i++) { free(reference[i]); free_output(outputs + i); }
    free(ids); free(x); free(mix); free(r); free(inj); free(shared_gate);
    printf("PASS Qwen SSD split T%u %s %u/%u slots=%u shared=%d seeded=%u unique=%u retry=%d: exact all stages, missing-only %llu bytes, warm0\n",
           T, mm ? "MM" : "rows", w->gate_type, w->down_type, slots, shared, seed_count, unique, fail_first, (unsigned long long)first.bytes);
}

int main(void) {
    weights formats[] = {{.gate_type=16, .down_type=10, .shared_down_type=8},
                         {.gate_type=12, .down_type=39, .shared_down_type=39},
                         {.gate_type=8, .down_type=8, .shared_down_type=8}};
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t bytes = page;
    for (unsigned i = 0; i < 3; i++) {
        weights *w = formats + i; ds4_gpu_stream_expert_table *t = &w->table;
        t->layer = 3 + i; t->n_total_expert = E;
        t->gate_expert_bytes = row_bytes(w->gate_type, D) * F;
        t->down_expert_bytes = row_bytes(w->down_type, F) * D;
        t->gate_offset = bytes; bytes = align_up(bytes + E * t->gate_expert_bytes, page);
        t->up_offset = bytes; bytes = align_up(bytes + E * t->gate_expert_bytes, page);
        t->down_offset = bytes; bytes = align_up(bytes + E * t->down_expert_bytes, page);
        w->shared_gate = bytes; bytes = align_up(bytes + row_bytes(8, D) * F, page);
        w->shared_up = bytes; bytes = align_up(bytes + row_bytes(8, D) * F, page);
        w->shared_down = bytes; bytes = align_up(bytes + row_bytes(w->shared_down_type, F) * D, page);
    }
    FILE *file = tmpfile(), *empty = tmpfile();
    need(file && empty && ftruncate(fileno(file), (off_t)bytes) == 0, "temporary sparse expert file");
    uint8_t *map = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file), 0);
    need(map != MAP_FAILED, "expert file mapping");
    for (unsigned i = 0; i < 3; i++) {
        weights *w = formats + i; ds4_gpu_stream_expert_table *t = &w->table;
        t->model_map = map; t->model_size = bytes;
        for (unsigned j = 0; j < sizeof(active) / sizeof(*active); j++) {
            const uint32_t e = (uint32_t)active[j];
            fill_weights(map + t->gate_offset + e * t->gate_expert_bytes, t->gate_expert_bytes, w->gate_type);
            fill_weights(map + t->up_offset + e * t->gate_expert_bytes, t->gate_expert_bytes, w->gate_type);
            /* Q2's physical tail has nonzero data and must never enter the dot product. */
            fill_weights(map + t->down_offset + e * t->down_expert_bytes, t->down_expert_bytes, w->down_type);
        }
        fill_weights(map + w->shared_gate, row_bytes(8, D) * F, 8);
        fill_weights(map + w->shared_up, row_bytes(8, D) * F, 8);
        fill_weights(map + w->shared_down, row_bytes(w->shared_down_type, F) * D, w->shared_down_type);
    }
    need(msync(map, bytes, MS_SYNC) == 0 && ds4_gpu_init(), "fixture initialization");
    ds4_gpu_set_quality(false); ds4_gpu_set_ssd_streaming(true);
    need(ds4_gpu_set_model_map(map, bytes) && ds4_gpu_set_model_fd(fileno(file)), "model registration");
    const uint32_t shapes[][2] = {{0,1}, {0,2}, {0,3}, {0,9}, {0,128},
                                  {1,1}, {1,17}, {2,2}, {2,9}};
    for (unsigned c = 0; c < sizeof(shapes) / sizeof(*shapes); c++) {
        const weights *w = formats + shapes[c][0]; const uint32_t T = shapes[c][1];
        ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * w->table.gate_expert_bytes + w->table.down_expert_bytes);
        need(ds4_gpu_stream_expert_cache_current_count() == 0, "cold cache reset");
        check_case(w, T, 0, fileno(file), fileno(empty), 0, 1);
        check_case(w, T, 1, fileno(file), fileno(empty), c == 0, 1);
        check_case(w, T, 0, fileno(file), fileno(empty), 0, 1);
        if (T == 128) {
            /* Eight experts appear in all 29 rows, sixteen in only 3/4 rows:
             * both >8-token counts and partial tiles, with 488 inactive. */
            check_case(w, 29, 4, fileno(file), fileno(empty), 0, 0);
            if (w->gate_type == 16) {
                /* Both sides of the measured M1 SSD batch-size boundary. */
                check_case(w, 32, 3, fileno(file), fileno(empty), 0, 0);
                check_case(w, 33, 3, fileno(file), fileno(empty), 0, 0);
            }
            check_case(w, T, 2, fileno(file), fileno(empty), 1, 0);
            check_case(w, T, 3, fileno(file), fileno(empty), 0, 0);
        }
    }
    check_case_impl(formats, 128, 2, fileno(file), fileno(empty), 0, 1, 24);
    puts("PASS Qwen SSD full-cache reuse: 24 disjoint victims, missing-only reads, warm0");
    /* A different-size MTP layer must use owned reads without replacing the
     * trunk's slab class. The old trunk selection must still hit with no fd. */
    ds4_gpu_set_streaming_expert_cache_budget(BUDGET);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(2 * formats[0].table.gate_expert_bytes + formats[0].table.down_expert_bytes);
    check_case(formats, 2, 0, fileno(file), fileno(empty), 0, 1);
    check_case(formats + 1, 2, 0, fileno(file), fileno(empty), 1, 0);
    check_case(formats + 1, 3, 0, fileno(file), fileno(empty), 0, 0);
    need(ds4_gpu_set_model_fd(fileno(empty)), "trunk retained after off-size MTP");
    check_case(formats, 2, 0, fileno(file), fileno(empty), 0, 1);
    check_split_one(formats, 1, 10, 1, 0, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 1, 10, 1, 4, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 1, 10, 1, 4, 1, 0, fileno(file), fileno(empty));
    check_split_one(formats, 1, 10, 0, 4, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 1, 10, 0, 4, 0, 0, fileno(file), fileno(empty));
    check_split_one(formats, 1, 10, 0, 0, 0, 0, fileno(file), fileno(empty));
    check_split_one(formats + 1, 1, 10, 1, 4, 0, 0, fileno(file), fileno(empty));
    check_split_one(formats + 1, 1, 16, 1, 4, 0, 0, fileno(file), fileno(empty));
    for (uint32_t T = 2; T <= 3; T++) {
        check_split_one(formats, T, 10, 1, 4, 0, 1, fileno(file), fileno(empty));
        check_split_one(formats, T, 10, 0, 4, 1, 1, fileno(file), fileno(empty));
        check_split_one(formats, T, 10, 1, 0, 0, 1, fileno(file), fileno(empty));
        check_split_one(formats + 1, T, 16, 1, 4, 0, 1, fileno(file), fileno(empty));
        check_split_one(formats + 2, T, 10, 1, 4, 0, 1, fileno(file), fileno(empty));
    }
    check_split_one(formats, 29, S, 0, 12, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 33, S, 0, 12, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 128, S, 0, 12, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats, 128, S, 0, 12, 0, 0, fileno(file), fileno(empty));
    check_split_one(formats, 33, S, 0, 0, 0, 1, fileno(file), fileno(empty));
    check_split_one(formats + 1, 33, S, 0, 12, 0, 0, fileno(file), fileno(empty));
    check_split_one(formats + 2, 29, S, 0, 12, 0, 0, fileno(file), fileno(empty));
    /* Gate/up completion must precede down reads on the enabled device.
     * Other GPUs keep the single read batch and may stop before all gate/up
     * tasks finish. Repeated failure also checks reservation cleanup. */
    for (uint32_t f = 0; mm_overlap_expected() && f < 3; f++) {
        const uint32_t shapes[] = {29, 33, 128};
        for (uint32_t i = 0; i < 3; i++) {
            check_split_one(formats + f, shapes[i], S, 0, 0, 0, 2, fileno(file), fileno(empty));
            check_split_one(formats + f, shapes[i], S, 0, 12, 0, 2, fileno(file), fileno(empty));
        }
    }
    ds4_gpu_cleanup(); munmap(map, bytes); fclose(empty); fclose(file);
    puts("PASS Qwen SSD: cache, exact selected-only reads, full-selection fallback and retry");
    return 0;
}
