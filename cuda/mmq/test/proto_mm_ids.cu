/* proto_mm_ids.cu — mm_ids_helper parity harness.
 *
 * P4 Inc1 (landed 86c5d4d): mm_ids_helper<1> fast path for the routed-MoE
 * down matmul (ds4_mmq.cu ds4_mmq_moe_impl: n_expert_used=1, si1=1, sis1=1,
 * nchannels_y=1, n_experts=256).  Without a case-1 dispatch the down matmul
 * fell to the generic <0> template: one warp per expert scanning all
 * assignment rows with a single active lane — 22.5 ms/launch at W4096.
 *
 * P5 (this extension): mm_ids_helper_global — the large-n variant with NO
 * shared-memory staging.  The smem kernel stages the compact (it, iex_used)
 * list in n_tokens*4 B of dynamic shared memory, capping n_tokens at smpbo/4
 * (~25k on GB10); the down matmul passes n_tokens = assignment rows = 6x the
 * forward width, so 8192-row prefill chunks hit 48384 "tokens", the mmq
 * entries refused, and the WHOLE MoE block fell back to the pre-mmq
 * expert-tile kernels (the W8192 prefill cliff, trace cmtp5w8.sqlite).  The
 * global variant runs the same per-expert scan twice: pass 0 counts
 * (nex_prev, bucket size), pass 1 re-scans and writes ids_src1/ids_dst
 * directly at final offsets.
 *
 * Gate: ids_src1 / ids_dst / expert_bounds BIT-IDENTICAL across
 *   (a) smem <0> vs smem <1>/<6>          (the landed Inc1 claim),
 *   (b) smem vs global at every sub-cap shape,
 *   (c) global vs a host reference at EVERY shape, including past-cap legs
 *       (48384/49152 assignments = the W8192 down shape; a >cap
 *       n_expert_used=6 leg; the cap boundary +/-1)
 * with the production pre-zero of both id maps (dropped -1 router ids leave
 * slots unwritten; expert_bounds[0] counts them, so the zeros sit below
 * expert 0's bucket and are never consumed).
 *
 * Bit-exactness argument for the global variant: identical (it, iex_used)
 * tuples in identical token order and identical output expressions; the
 * (lossless) 22/10-bit store round-trip is simply removed.  Both variants
 * are undefined if one token lists the same expert twice (router top-k is
 * without replacement); the generators here sample distinct experts.
 *
 * Build (GB10):
 *   nvcc -O3 --use_fast_math -std=c++17 -arch=sm_121 proto_mm_ids.cu -o proto_mm_ids
 */
#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static constexpr int WARP_SIZE = 32;

/* ------------------------------------------------------------------ */
/* Vendored from cuda/mmq/mmid.cu + common.cuh (verbatim semantics)     */

template <int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == WARP_SIZE) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

struct mm_ids_helper_store {
    uint32_t data;
    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }
    __device__ uint32_t it() const { return data & 0x003FFFFF; }
    __device__ uint32_t iex_used() const { return data >> 22; }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

template <int n_expert_used_template>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = WARP_SIZE;
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0;
    int it_compact = 0;

    if constexpr (n_expert_used_template == 0) {
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1;
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }
            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }
            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        static_assert(n_expert_used_template == 6 || warp_size % n_expert_used_template == 0, "bad n_expert_used");
        constexpr int neu_padded = n_expert_used_template == 6 ? 8 : n_expert_used_template;
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded;
            const int expert_used = (neu_padded == n_expert_used_template || iex < n_expert_used_template) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    __syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }
    expert_bounds[expert] = nex_prev;
    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }
    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

/* Vendored mm_ids_helper_global (P5 large-n variant, mmid.cu) */
template <int n_expert_used_template>
__launch_bounds__(WARP_SIZE, 1)
static __global__ void mm_ids_helper_global(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = WARP_SIZE;
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    int nex_prev         = 0;
    int it_compact_count = 0;

#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        int it_compact = 0;

        if constexpr (n_expert_used_template == 0) {
            for (int it = 0; it < n_tokens; ++it) {
                int iex_used = -1;
                for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                    const int expert_used = ids[it*si1 + iex];
                    if (pass == 0) {
                        nex_prev += expert_used < expert;
                    }
                    if (expert_used == expert) {
                        iex_used = iex;
                    }
                }

                if (pass == 1 && iex_used != -1) {
                    ids_src1[nex_prev + it_compact] = it*sis1          + iex_used % nchannels_y;
                    ids_dst [nex_prev + it_compact] = it*n_expert_used + iex_used;
                }

                if (warp_reduce_any<warp_size>(iex_used != -1)) {
                    it_compact++;
                }
            }
        } else {
            static_assert(n_expert_used_template == 6 || warp_size % n_expert_used_template == 0, "bad n_expert_used");
            constexpr int neu_padded = n_expert_used_template == 6 ? 8 : n_expert_used_template;
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it = it0 + threadIdx.x / neu_padded;

                const int iex = threadIdx.x % neu_padded;
                const int expert_used = (neu_padded == n_expert_used_template || iex < n_expert_used_template) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;
                const int iex_used = expert_used == expert ? iex : -1;
                if (pass == 0) {
                    nex_prev += expert_used < expert;
                }

                const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

                int it_compact_add_lower = 0;
#pragma unroll
                for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                    const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                    if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                        it_compact_add_lower += tmp;
                    }
                }

                if (pass == 1 && iex_used != -1) {
                    const int itc = it_compact + it_compact_add_lower;
                    ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
                    ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
                }

                it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
            }
        }

        if (pass == 0) {
            it_compact_count = it_compact;
            nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact_count;
}

/* NOTE: the proto instantiations replace the vendored kernels' runtime-const
 * neu_padded with a constexpr derived from the template arg (identical value
 * per instantiation; the vendored code relies on the same constant folding). */

/* ------------------------------------------------------------------ */

static constexpr int N_EXPERTS = 256;

/* Host reference implementing the mm_ids_helper output contract:
 *   expert_bounds[e] = #{assignment values < e}  (signed: -1 counts for all
 *   experts, so invalid rows occupy never-consumed slots below expert 0);
 *   bucket e = token-ascending (it, iex) pairs with ids[it*si1+iex] == e;
 *   ids_src1[pos] = it*sis1 + iex % nchannels_y; ids_dst[pos] = it*neu + iex.
 * src1/dst must be pre-zeroed by the caller (production memset semantics). */
static void cpu_ref(const int32_t *ids, int nt, int neu, int nchannels_y, int si1, int sis1,
                    int32_t *src1, int32_t *dst, int32_t *eb) {
    std::vector<int32_t> h(N_EXPERTS + 2, 0); // h[b+1] counts value b, b in [-1, N_EXPERTS]
    for (int it = 0; it < nt; ++it) {
        for (int iex = 0; iex < neu; ++iex) {
            int v = ids[it*si1 + iex];
            if (v < -1) v = -1;
            if (v > N_EXPERTS) v = N_EXPERTS;
            h[v + 1]++;
        }
    }
    std::vector<int32_t> below(N_EXPERTS + 1, 0); // below[e] = #{v < e}
    int32_t acc = h[0];
    for (int e = 0; e <= N_EXPERTS; ++e) {
        below[e] = acc;
        if (e < N_EXPERTS) acc += h[e + 1];
    }
    for (int e = 0; e <= N_EXPERTS; ++e) {
        eb[e] = below[e];
    }
    std::vector<int32_t> fill(N_EXPERTS, 0);
    for (int it = 0; it < nt; ++it) {
        for (int iex = 0; iex < neu; ++iex) {
            const int e = ids[it*si1 + iex];
            if (e < 0 || e >= N_EXPERTS) continue;
            const int64_t pos = (int64_t)below[e] + fill[e]++;
            src1[pos] = it*sis1 + iex % nchannels_y;
            dst [pos] = it*neu  + iex;
        }
    }
}

struct leg {
    const char *name;
    int n_tokens;      // assignment rows (neu=1) or tokens (neu=6)
    int neu;           // n_expert_used: 1 = down shape, 6 = gate/up shape
    int invalid_pct;   // % of slots with -1 (router NaN path)
    bool skew;         // hot-expert distribution
};

int main() {
    int smpbo = 0;
    CHECK(cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CHECK(cudaFuncSetAttribute((const void*)mm_ids_helper<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
    CHECK(cudaFuncSetAttribute((const void*)mm_ids_helper<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
    CHECK(cudaFuncSetAttribute((const void*)mm_ids_helper<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, smpbo));
    const int cap_tokens = smpbo / 4;
    printf("smpbo=%d (cap %d tokens)\n", smpbo, cap_tokens);

    std::vector<leg> legs = {
        // down shape (neu=1): n_tokens = rows*6 assignments
        {"W4096x6dn uniform",   4096*6, 1, 0, false},
        {"W4096x6dn 2%invalid", 4096*6, 1, 2, false},
        {"W4096x6dn skewed",    4096*6, 1, 1, true},
        {"W2048x6dn uniform",   2048*6, 1, 2, false},
        {"decode w1x6",         6,      1, 0, false},
        {"decode w8x6",         48,     1, 0, false},
        // past-cap down shapes (the P5 cliff: W8192 -> 8064*6 = 48384 traced, 8192*6 = 49152 max)
        {"W8192x6dn uniform",   48384,  1, 0, false},
        {"W8192x6dn 2%invalid", 48384,  1, 2, false},
        {"W8192x6dn skewed",    48384,  1, 1, true},
        {"W8192x6dn max",       49152,  1, 2, false},
        // gate/up shape (neu=6): n_tokens = rows, si1 = 6, distinct experts/token
        {"W8192gu uniform",     8064,   6, 0, false},
        {"W8192gu 2%invalid",   8064,   6, 2, false},
        {"gu past-cap",         30000,  6, 2, false},
    };
    // cap-boundary seam legs (runtime-sized)
    leg seam_at   = {"cap-boundary",  cap_tokens,     1, 2, false};
    leg seam_over = {"cap-boundary+1", cap_tokens + 1, 1, 2, false};
    legs.push_back(seam_at);
    legs.push_back(seam_over);

    int failures = 0;

    for (const leg &L : legs) {
        const int nt  = L.n_tokens;
        const int neu = L.neu;
        const int si1 = neu, sis1 = 1, nchannels_y = 1;
        const int64_t n_ids  = (int64_t)nt * neu;
        const size_t  smem   = (size_t)nt * 4;
        const bool smem_fits = smem <= (size_t)smpbo;

        int32_t *h_ids = (int32_t*)malloc(n_ids * sizeof(int32_t));
        srand(20260708);
        if (neu == 1) {
            for (int i = 0; i < nt; i++) {
                if (L.invalid_pct && rand() % 100 < L.invalid_pct) { h_ids[i] = -1; continue; }
                h_ids[i] = L.skew ? (rand() % 100 < 60 ? rand() % 8 : rand() % N_EXPERTS)
                                  : rand() % N_EXPERTS;
            }
        } else {
            // router top-k semantics: distinct experts per token
            for (int it = 0; it < nt; it++) {
                int chosen[8]; int c = 0;
                while (c < neu) {
                    const int e = L.skew ? (rand() % 100 < 60 ? rand() % 8 : rand() % N_EXPERTS)
                                         : rand() % N_EXPERTS;
                    bool dup = false;
                    for (int j = 0; j < c; j++) dup = dup || chosen[j] == e;
                    if (!dup) chosen[c++] = e;
                }
                for (int iex = 0; iex < neu; iex++) {
                    h_ids[(int64_t)it*neu + iex] =
                        (L.invalid_pct && rand() % 100 < L.invalid_pct) ? -1 : chosen[iex];
                }
            }
        }

        // host reference
        int32_t *r_src1 = (int32_t*)calloc(n_ids, sizeof(int32_t));
        int32_t *r_dst  = (int32_t*)calloc(n_ids, sizeof(int32_t));
        int32_t r_eb[N_EXPERTS+1];
        cpu_ref(h_ids, nt, neu, nchannels_y, si1, sis1, r_src1, r_dst, r_eb);

        int32_t *d_ids, *d_src1, *d_dst, *d_eb;
        CHECK(cudaMalloc(&d_ids,  n_ids * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_src1, n_ids * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_dst,  n_ids * sizeof(int32_t)));
        CHECK(cudaMalloc(&d_eb,   (N_EXPERTS+1) * sizeof(int32_t)));
        CHECK(cudaMemcpy(d_ids, h_ids, n_ids * sizeof(int32_t), cudaMemcpyHostToDevice));

        const dim3 grid(N_EXPERTS, 1, 1), block(WARP_SIZE, 1, 1);
        int32_t *h_src1 = (int32_t*)malloc(n_ids * sizeof(int32_t));
        int32_t *h_dst  = (int32_t*)malloc(n_ids * sizeof(int32_t));
        int32_t h_eb[N_EXPERTS+1];

        // candidate kernels for this leg: smem generic, smem fast, global fast
        struct cand { const char *name; int variant; }; // 0=<0> smem, 1=<neu> smem, 2=<neu> global
        std::vector<cand> cands;
        if (smem_fits) { cands.push_back({"smem<0>", 0}); cands.push_back({"smem<t>", 1}); }
        cands.push_back({"global<t>", 2});

        bool ok = true;
        float ms[3] = {0, 0, 0};
        for (const cand &C : cands) {
            // production pre-zero of both id maps
            CHECK(cudaMemset(d_src1, 0, n_ids * sizeof(int32_t)));
            CHECK(cudaMemset(d_dst,  0, n_ids * sizeof(int32_t)));
            CHECK(cudaMemset(d_eb, 0xFF, (N_EXPERTS+1) * sizeof(int32_t)));

            auto launch = [&](void) {
                switch (C.variant) {
                    case 0: mm_ids_helper<0><<<grid, block, smem>>>(d_ids, d_src1, d_dst, d_eb, nt, neu, nchannels_y, si1, sis1); break;
                    case 1:
                        if (neu == 1) mm_ids_helper<1><<<grid, block, smem>>>(d_ids, d_src1, d_dst, d_eb, nt, neu, nchannels_y, si1, sis1);
                        else          mm_ids_helper<6><<<grid, block, smem>>>(d_ids, d_src1, d_dst, d_eb, nt, neu, nchannels_y, si1, sis1);
                        break;
                    case 2:
                        if (neu == 1) mm_ids_helper_global<1><<<grid, block, 0>>>(d_ids, d_src1, d_dst, d_eb, nt, neu, nchannels_y, si1, sis1);
                        else          mm_ids_helper_global<6><<<grid, block, 0>>>(d_ids, d_src1, d_dst, d_eb, nt, neu, nchannels_y, si1, sis1);
                        break;
                }
            };
            launch();
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            CHECK(cudaMemcpy(h_src1, d_src1, n_ids*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_dst,  d_dst,  n_ids*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_eb,   d_eb,   (N_EXPERTS+1)*4, cudaMemcpyDeviceToHost));
            const bool m1 = memcmp(h_src1, r_src1, n_ids*4) == 0;
            const bool m2 = memcmp(h_dst,  r_dst,  n_ids*4) == 0;
            const bool m3 = memcmp(h_eb,   r_eb,   (N_EXPERTS+1)*4) == 0;
            if (!(m1 && m2 && m3)) {
                ok = false;
                printf("  %-18s %-9s MISMATCH src1=%d dst=%d eb=%d\n", L.name, C.name, m1, m2, m3);
            }

            // timing (50 reps after 5 warmups)
            cudaEvent_t t0, t1;
            CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
            for (int w = 0; w < 5; w++) launch();
            CHECK(cudaEventRecord(t0));
            for (int r = 0; r < 50; r++) launch();
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
            float m = 0; CHECK(cudaEventElapsedTime(&m, t0, t1));
            ms[C.variant] = m / 50;
            CHECK(cudaEventDestroy(t0)); CHECK(cudaEventDestroy(t1));
        }

        if (smem_fits) {
            printf("%-20s nt=%-6d neu=%d %s  smem<0> %8.3f ms  smem<t> %8.3f ms  global<t> %8.3f ms\n",
                   L.name, nt, neu, ok ? "PARITY-OK " : "MISMATCH!!", ms[0], ms[1], ms[2]);
        } else {
            printf("%-20s nt=%-6d neu=%d %s  (past cap)                              global<t> %8.3f ms\n",
                   L.name, nt, neu, ok ? "PARITY-OK " : "MISMATCH!!", ms[2]);
        }
        if (!ok) failures++;

        free(h_ids); free(h_src1); free(h_dst); free(r_src1); free(r_dst);
        cudaFree(d_ids); cudaFree(d_src1); cudaFree(d_dst); cudaFree(d_eb);
    }

    printf(failures ? "PROTO FAIL (%d legs)\n" : "PROTO PASS\n", failures);
    return failures ? 1 : 0;
}
