/* Staged MXFP4 gate/up rows must match the byte-load kernel bit for bit.
 * The staged path needs 16-byte aligned weights, so the same weights copied
 * to base + 1 reach the byte-load kernel through the same entry point.
 * Fused down rows also compare with independent per-slot MMVQ results.
 * An optional output path dumps down results for cross-build comparisons. */
#include "ds4_mmq.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int ds4_cuda_q8_fold_take_q81(
        const void *src, uint64_t in_dim, const void **q81) {
    (void)src;
    (void)in_dim;
    if (q81) *q81 = nullptr;
    return 0;
}

namespace {

constexpr int QK = 32;
constexpr int TOPK = 6;
FILE *down_dump = nullptr;

struct block_mxfp4_test {
    uint8_t e;
    uint8_t qs[QK / 2];
};
static_assert(sizeof(block_mxfp4_test) == 17, "unexpected MXFP4 layout");

struct pattern {
    std::string name;
    int n_tokens;
    std::vector<int32_t> ids;   /* n_tokens * 6, token-major */
};

bool ok_cuda(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

/* Device allocation freed on every exit path. */
struct dev_mem {
    void *p = nullptr;
    dev_mem() = default;
    dev_mem(const dev_mem &) = delete;
    dev_mem &operator=(const dev_mem &) = delete;
    ~dev_mem() { if (p) cudaFree(p); }
    bool alloc(size_t bytes, const char *what) { return ok_cuda(cudaMalloc(&p, bytes), what); }
    template <class T> T *as() const { return (T *)p; }
};

bool upload(dev_mem &m, const void *src, size_t bytes, const char *what) {
    return m.alloc(bytes, what) &&
           ok_cuda(cudaMemcpy(m.p, src, bytes, cudaMemcpyHostToDevice), what);
}

/* The same bytes at base + 1; the allocation is exactly bytes + 1 long so
 * memcheck still sees overruns. */
bool misaligned_copy(dev_mem &m, const void *d_src, size_t bytes) {
    return m.alloc(bytes + 1, "alloc misaligned") &&
           ok_cuda(cudaMemcpy(m.as<char>() + 1, d_src, bytes, cudaMemcpyDeviceToDevice),
                   "copy misaligned");
}

/* Six distinct experts per token from a skewed popularity; with owned_mask
 * the upper half of the experts becomes -1 as for the other TP rank. */
std::vector<int32_t> realistic_ids(int n_tokens, int n_experts, bool owned_mask,
                                   std::mt19937 &rng) {
    std::vector<double> pop((size_t)n_experts);
    for (int e = 0; e < n_experts; e++) pop[(size_t)e] = 1.0 / (1.0 + 0.35 * e);
    std::vector<int32_t> ids((size_t)n_tokens * TOPK);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<double> p = pop;
        for (int s = 0; s < TOPK; s++) {
            double total = 0.0;
            for (double v : p) total += v;
            double r = u(rng) * total;
            int e = 0;
            for (; e < n_experts - 1; e++) {
                r -= p[(size_t)e];
                if (r <= 0.0) break;
            }
            while (p[(size_t)e] == 0.0) e = (e + 1) % n_experts;
            p[(size_t)e] = 0.0;
            ids[(size_t)t * TOPK + s] = e;
        }
    }
    if (owned_mask) {
        for (int32_t &id : ids) if (id >= n_experts / 2) id = -1;
    }
    return ids;
}

bool run_pattern(const pattern &p, int M, int K, int n_experts, float clamp,
                 const void *d_gate, const void *d_up, const void *d_gate_base,
                 const void *d_up_base, std::mt19937 &rng) {
    const int n = p.n_tokens;
    std::uniform_real_distribution<float> act(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(0.02f, 0.4f);
    std::vector<float> x((size_t)n * K), w((size_t)n * TOPK);
    for (float &v : x) v = act(rng);
    for (float &v : w) v = wdist(rng);

    const size_t mid_floats = (size_t)n * TOPK * M;
    dev_mem d_x, d_w, d_ids, d_mid;
    if (!upload(d_x, x.data(), x.size() * sizeof(float), "x") ||
        !upload(d_w, w.data(), w.size() * sizeof(float), "weights") ||
        !upload(d_ids, p.ids.data(), p.ids.size() * sizeof(int32_t), "ids") ||
        !d_mid.alloc(mid_floats * sizeof(float), "alloc mid")) {
        std::printf("%s: setup failed  FAIL\n", p.name.c_str());
        return false;
    }

    /* Pass 0: misaligned copy (byte-load kernel); pass 1: staged rows. */
    std::vector<uint32_t> out[2];
    for (int pass = 0; pass < 2; pass++) {
        out[pass].assign(mid_floats, 0u);
        if (!ok_cuda(cudaMemset(d_mid.p, 0xFF, mid_floats * sizeof(float)), "poison mid")) {
            std::printf("%s: setup failed  FAIL\n", p.name.c_str());
            return false;
        }
        const int rc = ds4_mmq_mxfp4_moe_gate_up_mid_vec(
                pass ? d_gate : d_gate_base, pass ? d_up : d_up_base,
                d_x.as<float>(), d_ids.as<int32_t>(), d_w.as<float>(), d_mid.as<float>(),
                M, K, n, n_experts, TOPK, clamp, (cudaStream_t)0);
        if (rc != 0 || !ok_cuda(cudaDeviceSynchronize(), "gate/up run") ||
            !ok_cuda(cudaMemcpy(out[pass].data(), d_mid.p, mid_floats * sizeof(float),
                                cudaMemcpyDeviceToHost), "read mid")) {
            std::printf("M=%-5d K=%-5d %s: pass %d failed (rc %d)  FAIL\n",
                        M, K, p.name.c_str(), pass, rc);
            return false;
        }
    }
    size_t diff = 0, unwritten = 0, first = (size_t)-1;
    for (size_t i = 0; i < mid_floats; i++) {
        if (out[0][i] != out[1][i]) {
            if (first == (size_t)-1) first = i;
            diff++;
        }
        if (out[1][i] == 0xFFFFFFFFu) unwritten++;
    }
    /* Invalid assignments must still produce exact zeros. */
    size_t bad_invalid = 0;
    for (int a = 0; a < n * TOPK; a++) {
        if (p.ids[a] >= 0 && p.ids[a] < n_experts) continue;
        for (int r = 0; r < M; r++) {
            float v;
            std::memcpy(&v, &out[1][(size_t)a * M + r], sizeof(v));
            if (v != 0.0f) bad_invalid++;
        }
    }
    const bool pass_ok = diff == 0 && unwritten == 0 && bad_invalid == 0;
    std::printf("M=%-5d K=%-5d clamp=%-3g %-30s n=%d  bit-diff=%zu unwritten=%zu invalid-nonzero=%zu  %s\n",
                M, K, (double)clamp, p.name.c_str(), n, diff, unwritten, bad_invalid,
                pass_ok ? "PASS" : "FAIL");
    if (first != (size_t)-1) {
        std::printf("    first diff: assignment %zu row %zu\n", first / M, first % M);
    }
    return pass_ok;
}

bool run_down_pattern(const pattern &p, int M, int K, int n_experts,
                      const void *W, const void *W_base, std::mt19937 &rng) {
    const int n = p.n_tokens;
    std::uniform_real_distribution<float> act(-1.0f, 1.0f);
    std::vector<float> x((size_t)n * TOPK * K);
    for (float &v : x) v = act(rng);
    const size_t count = (size_t)n * M;
    dev_mem d_x, d_ids, d_out;
    if (!upload(d_x, x.data(), x.size() * sizeof(float), "down x") ||
        !upload(d_ids, p.ids.data(), p.ids.size() * sizeof(int32_t), "down ids") ||
        !d_out.alloc(count * sizeof(float), "down out")) return false;
    std::vector<uint32_t> out[2];
    for (int pass = 0; pass < 2; ++pass) {
        out[pass].resize(count);
        if (!ok_cuda(cudaMemset(d_out.p, 0xFF, count * sizeof(float)), "poison down")) return false;
        const int rc = ds4_mmq_mxfp4_moe_down_sum6_vec(pass ? W : W_base,
                d_x.as<float>(), d_ids.as<int32_t>(), d_out.as<float>(),
                M, K, n, n_experts, TOPK, (cudaStream_t)0);
        if (rc || !ok_cuda(cudaDeviceSynchronize(), "down run") ||
            !ok_cuda(cudaMemcpy(out[pass].data(), d_out.p, count * sizeof(float),
                               cudaMemcpyDeviceToHost), "read down")) return false;
    }
    size_t diff = 0, unwritten = 0, invalid = 0;
    for (size_t i = 0; i < count; ++i) {
        diff += out[0][i] != out[1][i];
        unwritten += out[1][i] == 0xFFFFFFFFu;
    }
    for (int t = 0; t < n; ++t) {
        bool all_invalid = true;
        for (int s = 0; s < TOPK; ++s) {
            const int id = p.ids[(size_t)t * TOPK + s];
            all_invalid &= id < 0 || id >= n_experts;
        }
        if (all_invalid) {
            for (int r = 0; r < M; ++r) invalid += out[1][(size_t)t * M + r] != 0u;
        }
    }
    /* The ordinary MMVQ path returns each slot separately, with a different
     * warp reduction. Normalize out-of-range IDs to its -1 sentinel. */
    std::vector<int32_t> ref_ids = p.ids;
    for (int32_t &id : ref_ids) if (id < 0 || id >= n_experts) id = -1;
    dev_mem d_ref_ids, d_slots;
    const size_t slot_count = (size_t)n * TOPK * M;
    if (!upload(d_ref_ids, ref_ids.data(), ref_ids.size() * sizeof(int32_t), "reference ids") ||
        !d_slots.alloc(slot_count * sizeof(float), "reference slots")) return false;
    if (ds4_mmq_mxfp4_moe_vec(W, d_x.as<float>(), d_ref_ids.as<int32_t>(),
                             d_slots.as<float>(), M, K, n * TOPK, n_experts, 1,
                             (cudaStream_t)0) != 0 ||
        !ok_cuda(cudaDeviceSynchronize(), "reference down")) return false;
    std::vector<float> slots(slot_count);
    if (!ok_cuda(cudaMemcpy(slots.data(), d_slots.p, slot_count * sizeof(float),
                            cudaMemcpyDeviceToHost), "read reference slots")) return false;
    size_t reference_diff = 0;
    for (int t = 0; t < n; ++t) {
        for (int r = 0; r < M; ++r) {
            float ref = 0.0f;
            for (int s = 0; s < TOPK; ++s) {
                const float v = slots[((size_t)t * TOPK + s) * M + r];
                ref += std::isfinite(v) ? v : 0.0f;
            }
            float got;
            std::memcpy(&got, &out[1][(size_t)t * M + r], sizeof(got));
            if (!std::isfinite(got) || std::fabs(got - ref) > 1e-4f + 3e-5f * std::fabs(ref))
                ++reference_diff;
        }
    }
    const bool pass = diff == 0 && unwritten == 0 && invalid == 0 && reference_diff == 0;
    std::printf("DOWN M=%d K=%d n=%d %s bit-diff=%zu unwritten=%zu invalid=%zu ref-diff=%zu %s\n",
                M, K, n, p.name.c_str(), diff, unwritten, invalid, reference_diff, pass ? "PASS" : "FAIL");
    if (down_dump && std::fwrite(out[1].data(), sizeof(uint32_t), count, down_dump) != count) {
        std::fprintf(stderr, "failed writing down output dump\n");
        return false;
    }
    return pass;
}

std::vector<pattern> make_patterns(int n_experts, std::mt19937 &rng) {
    const int32_t bad = n_experts;   /* first out-of-range expert id */
    const int32_t last = n_experts - 1;
    std::vector<pattern> pats;
    pats.push_back({"three pairs on slot 0", 6, {
        3, 0, 1, 2, 4, 5,
        3, 6, 7, 0, 1, 2,
        3, 4, 5, 6, 7, 0,
        3, 1, 2, 4, 5, 6,
        3, 7, 0, 1, 2, 4,
        3, 5, 6, 7, 0, 1}});
    pats.push_back({"odd count: pair + single", 3, {
        5, 0, 1, 2, 3, 4,
        6, 5, 7, 0, 1, 2,
        3, 4, 5, 6, 7, 0}});
    pats.push_back({"invalid ids mixed", 4, {
        -1, 2, 7, bad, 1, 0,
         2, -1, 7, 3, 4, 5,
         7, 2, -1, 99, 6, 1,
        -1, -1, 2, 7, 0, 1}});
    std::vector<int32_t> dense8((size_t)8 * TOPK);
    for (int t = 0; t < 8; t++) {
        for (int s = 0; s < TOPK; s++) dense8[(size_t)t * TOPK + s] = (t * 3 + s) % 8;
    }
    pats.push_back({"n=8 heavy reuse", 8, dense8});
    pats.push_back({"no duplicates", 2, {
        0, 1, 2, 3, 4, 5,
        6, 7, -1, -1, -1, -1}});
    pats.push_back({"single token", 1, {7, 0, 1, 2, 3, 4}});
    pats.push_back({"all invalid token", 2, {-1, bad, -1, bad, -1, bad,
                                           0, 1, 2, 3, 4, 5}});
    std::vector<int32_t> mixed7((size_t)7 * TOPK);
    for (size_t i = 0; i < mixed7.size(); ++i) mixed7[i] = i % 9 == 0 ? -1 : (int32_t)(i % n_experts);
    pats.push_back({"n=7 mixed reuse", 7, mixed7});
    pats.push_back({"last expert pairs", 5, {
        last, 0, 1, 2, 3, 4,
        last, 1, 2, 3, 4, 5,
        last, 2, 3, 4, 5, 6,
        last, 3, 4, 5, 6, 0,
        6, last, 4, 5, 0, 1}});
    for (int n : {6, 8}) {
        for (int owned = 0; owned < 2; owned++) {
            pats.push_back({std::string("realistic n=") + std::to_string(n) +
                                (owned ? " owned" : " all"),
                            n, realistic_ids(n, n_experts, owned != 0, rng)});
        }
    }
    return pats;
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [down-output-dump]\n", argv[0]);
        return 2;
    }
    if (argc == 2) {
        down_dump = std::fopen(argv[1], "wb");
        if (!down_dump) {
            std::perror(argv[1]);
            return 2;
        }
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "no CUDA device\n");
        return 1;
    }
    if (prop.major != 12 || prop.minor != 1) {
        std::printf("SKIP: staged MXFP4 rows are enabled on DGX Spark (sm_121) only\n");
        return 0;
    }
    if (ds4_mmq_init(0) != 0) {
        std::fprintf(stderr, "ds4_mmq_init failed\n");
        return 1;
    }
    struct shape { int M, K, n_experts; };
    /* 200/201 rows: block tails with even and odd counts.  K = 256 is
     * declined (K/32 not a multiple of 16).  2048x4096 and 1024x4096 are the
     * Flash routed gate/up shapes without and with the TP intermediate split.
     * K = 8192 and 11264 fit the 48 KiB shared-memory budget;
     * K = 12288 needs 51 KiB and must take the byte-load kernel. */
    const shape shapes[] = {{200, 512, 12}, {201, 512, 12}, {200, 256, 12},
                            {4096, 1024, 32}, {4096, 2048, 32}, {97, 1536, 8},
                            {2048, 4096, 32}, {1024, 4096, 64},
                            {72, 8192, 8}, {72, 11264, 8}, {72, 12288, 8}};
    const float clamps[] = {0.0f, 7.0f};
    std::mt19937 rng(0x50414952u);
    int fails = 0, runs = 0;
    for (const shape &sh : shapes) {
        const int M = sh.M, K = sh.K, n_experts = sh.n_experts;
        /* Exact-size weight stacks so memcheck sees any read past the last expert. */
        const size_t blocks = (size_t)n_experts * M * (K / QK);
        const size_t bytes = blocks * sizeof(block_mxfp4_test);
        std::vector<block_mxfp4_test> gate(blocks), up(blocks);
        std::uniform_int_distribution<int> ex(118, 127), byte(0, 255);
        for (auto *v : {&gate, &up}) {
            for (auto &b : *v) {
                b.e = (uint8_t)ex(rng);
                for (uint8_t &q : b.qs) q = (uint8_t)byte(rng);
            }
        }
        dev_mem d_gate, d_up, gate_base, up_base;
        if (!upload(d_gate, gate.data(), bytes, "gate") ||
            !upload(d_up, up.data(), bytes, "up") ||
            !misaligned_copy(gate_base, d_gate.p, bytes) ||
            !misaligned_copy(up_base, d_up.p, bytes)) {
            std::printf("M=%d K=%d: weight setup failed  FAIL\n", M, K);
            return 1;
        }

        const std::vector<pattern> pats = make_patterns(n_experts, rng);
        for (const pattern &p : pats) {
            fails += !run_down_pattern(p, M, K, n_experts,
                                       d_gate.p, gate_base.as<char>() + 1, rng);
            ++runs;
        }
        for (float clamp : clamps) {
            for (const pattern &p : pats) {
                fails += !run_pattern(p, M, K, n_experts, clamp, d_gate.p, d_up.p,
                                      gate_base.as<char>() + 1, up_base.as<char>() + 1, rng);
                runs++;
            }
        }
    }
    std::printf("MXFP4 staged gate/up and fused down: %s (%d of %d runs failing)\n",
                fails ? "FAIL" : "PASS", fails, runs);
    if (down_dump && std::fclose(down_dump) != 0) {
        std::perror("closing down output dump");
        return 2;
    }
    return fails ? 1 : 0;
}
