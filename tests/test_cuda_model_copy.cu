/* The startup walk over a registered model mapping copies the tensor spans
 * into device memory on a discrete GPU when they fit next to the planned
 * context, and leaves them on the mapping otherwise and on an integrated GPU.
 * Checks the decision against free device memory, the opt-out and a map that
 * is not the registered one; that the copies hold the source bytes (read back
 * through a weight kernel at a span start and at interior offsets); that a
 * span the walk did not copy still resolves to the mapping; and that device
 * memory moved by the span total. */
#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static const uint64_t MiB = 1048576ull;
static const uint64_t MODEL_BYTES = 64 * MiB;
static const uint64_t HEADER_BYTES = 4 * MiB;
static const uint32_t N_VOCAB = 1024, N_EMBD = 256, N_HC = 1;

/* Three spans with a gap the walk never covers, like merged GGUF tensors. */
struct tspan { uint64_t off, bytes; };
static const tspan SPANS[3] = {
    {4 * MiB, 16 * MiB},
    {20 * MiB + 65536, 24 * MiB - 65536},
    {48 * MiB, 16 * MiB},
};
static const uint64_t GAP_OFF = 44 * MiB;                    /* [44 MiB, 48 MiB) stays on the mapping */
static const uint64_t TABLE_IN_SPAN = SPANS[1].off + 12346u;  /* interior, half-aligned */
static const uint64_t TABLE_AT_START = SPANS[2].off;
static const uint64_t TABLE_IN_GAP = GAP_OFF + 2222u;

static void fill_table(unsigned char *base, uint64_t off, uint32_t seed) {
    __half *h = (__half *)(base + off);
    for (uint64_t i = 0; i < (uint64_t)N_VOCAB * N_EMBD; i++) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = __float2half((float)(int)(seed >> 20) / 64.0f);
    }
}

/* Gather rows through the weight path; returns the output bits. */
static std::vector<uint32_t> gather(const unsigned char *map, uint64_t table_off, const int32_t *tokens, uint32_t n_tokens) {
    ds4_gpu_tensor *tok = ds4_gpu_tensor_alloc((uint64_t)n_tokens * sizeof(int32_t));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * N_HC * N_EMBD * sizeof(float));
    CHECK(tok && out);
    CHECK(ds4_gpu_tensor_write(tok, 0, tokens, (uint64_t)n_tokens * sizeof(int32_t)));
    CHECK(ds4_gpu_embed_tokens_hc_tensor(out, tok, map, MODEL_BYTES, table_off, N_VOCAB, n_tokens, N_EMBD, N_HC));
    std::vector<uint32_t> bits((size_t)n_tokens * N_HC * N_EMBD);
    CHECK(ds4_gpu_tensor_read(out, 0, bits.data(), bits.size() * sizeof(uint32_t)));
    ds4_gpu_tensor_free(tok);
    ds4_gpu_tensor_free(out);
    return bits;
}

static std::vector<uint32_t> reference(const unsigned char *map, uint64_t table_off, const int32_t *tokens, uint32_t n_tokens) {
    const __half *h = (const __half *)(map + table_off);
    std::vector<uint32_t> bits;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t hc = 0; hc < N_HC; hc++) {
            for (uint32_t d = 0; d < N_EMBD; d++) {
                const float f = __half2float(h[(uint64_t)tokens[t] * N_EMBD + d]);
                uint32_t b;
                memcpy(&b, &f, sizeof(b));
                bits.push_back(b);
            }
        }
    }
    return bits;
}

static uint64_t device_free(void) {
    size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b) == cudaSuccess);
    return (uint64_t)free_b;
}

static void walk(const unsigned char *map) {
    for (const tspan &s : SPANS) CHECK(ds4_gpu_cache_model_range(map, MODEL_BYTES, s.off, s.bytes, "span"));
}

int main(void) {
    int dev_count = 0;
    (void)cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) { fprintf(stderr, "no CUDA devices\n"); return 0; }
    CHECK(ds4_gpu_init());
    int device = 0, integrated = 0;
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    CHECK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) == cudaSuccess);

    /* An anonymous mapping stands in for the GGUF mmap. */
    unsigned char *map = (unsigned char *)mmap(NULL, (size_t)MODEL_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(map != MAP_FAILED);
    uint32_t rng = 7;
    for (uint64_t i = 0; i < MODEL_BYTES; i++) { rng = rng * 1664525u + 1013904223u; map[i] = (unsigned char)(rng >> 24); }
    fill_table(map, TABLE_IN_SPAN, 11);
    fill_table(map, TABLE_AT_START, 22);
    fill_table(map, TABLE_IN_GAP, 33);
    const int32_t tokens[5] = {0, 1, 517, 1022, 1023};
    const uint32_t n_tokens = 5;
    uint64_t span_bytes = 0;
    for (const tspan &s : SPANS) span_bytes += s.bytes;

    CHECK(ds4_gpu_set_model_map_range(map, MODEL_BYTES, HEADER_BYTES, MODEL_BYTES - HEADER_BYTES, 0));

    if (integrated) {
        /* Unified memory: the mapping is device memory, nothing to copy. */
        CHECK(ds4_gpu_plan_model_cache(map, span_bytes, 0) == 0);
        walk(map);
        CHECK(gather(map, TABLE_IN_SPAN, tokens, n_tokens) == reference(map, TABLE_IN_SPAN, tokens, n_tokens));
        fprintf(stderr, "test_cuda_model_copy: integrated GPU keeps the mapping: PASS\n");
        ds4_gpu_cleanup();
        return 0;
    }

    /* Warm the allocator so the free-memory deltas below only see the copies. */
    (void)gather(map, TABLE_IN_SPAN, tokens, n_tokens);

    /* 1. The planned context leaves no room: the walk resolves the mapping. */
    const uint64_t free0 = device_free();
    CHECK(ds4_gpu_plan_model_cache(map, span_bytes, free0) == 0);
    walk(map);
    const uint64_t free1 = device_free();
    CHECK(free1 + 2 * MiB >= free0);
    const std::vector<uint32_t> mapped_in_span = gather(map, TABLE_IN_SPAN, tokens, n_tokens);
    const std::vector<uint32_t> mapped_at_start = gather(map, TABLE_AT_START, tokens, n_tokens);
    CHECK(mapped_in_span == reference(map, TABLE_IN_SPAN, tokens, n_tokens));
    CHECK(mapped_at_start == reference(map, TABLE_AT_START, tokens, n_tokens));

    /* 2. Opt-out keeps the mapping even when everything fits. */
    CHECK(setenv("DS4_CUDA_NO_MODEL_COPY", "1", 1) == 0);
    CHECK(ds4_gpu_plan_model_cache(map, span_bytes, 0) == 0);
    CHECK(unsetenv("DS4_CUDA_NO_MODEL_COPY") == 0);
    /* Only the registered mapping is planned; another map is left to the walk as before. */
    CHECK(ds4_gpu_plan_model_cache(map + HEADER_BYTES, span_bytes, 0) == 0);

    /* 3. It fits: the walk copies every span, and the copies serve the reads. */
    CHECK(ds4_gpu_plan_model_cache(map, span_bytes, 0) == 1);
    walk(map);
    const uint64_t free2 = device_free();
    CHECK(free1 >= free2);
    const uint64_t moved = free1 - free2;
    CHECK(moved >= span_bytes);
    CHECK(moved <= span_bytes + 3 * 2 * MiB);
    CHECK(gather(map, TABLE_IN_SPAN, tokens, n_tokens) == mapped_in_span);
    CHECK(gather(map, TABLE_AT_START, tokens, n_tokens) == mapped_at_start);
    /* The gap was never walked: still the mapping, still the source bytes. */
    CHECK(gather(map, TABLE_IN_GAP, tokens, n_tokens) == reference(map, TABLE_IN_GAP, tokens, n_tokens));
    /* The copies are device memory: changing the source no longer changes a read. */
    fill_table(map, TABLE_AT_START, 44);
    CHECK(gather(map, TABLE_AT_START, tokens, n_tokens) == mapped_at_start);
    CHECK(gather(map, TABLE_AT_START, tokens, n_tokens) != reference(map, TABLE_AT_START, tokens, n_tokens));

    ds4_gpu_cleanup();
    CHECK(munmap(map, (size_t)MODEL_BYTES) == 0);
    fprintf(stderr, "test_cuda_model_copy: %.0f MiB copied, reads identical to the mapping: PASS\n",
            (double)span_bytes / (double)MiB);
    return 0;
}
