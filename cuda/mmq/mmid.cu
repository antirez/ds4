#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
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
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
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

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Post-Volta independent thread scheduling: the store[] writes above are done by
    // some warp lanes and read below by other lanes. Without an explicit warp barrier
    // those shared-memory writes are not guaranteed visible to the cross-lane reads,
    // so a lane could read a stale/uninitialized store[] slot -> wrong compacted
    // expert ids -> nondeterministic MoE routing. On GB10 (sm_121) this realized as
    // the cont-multiseq non-determinism + BOS-spam (compute-sanitizer racecheck:
    // "RAW hazard at __shared__" between the store writes and these reads).
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

// ds4 local (P5): large-n variant of mm_ids_helper with NO shared-memory
// staging.  The smem kernel above stages the compact (it, iex_used) list in
// n_tokens*4 B of dynamic shared memory, capping n_tokens at smpbo/4 (~25k on
// GB10); the routed-MoE down matmul passes n_tokens = assignment rows
// (6x the forward width), so 8192-row prefill chunks hit 48384 "tokens" and
// the whole MoE block used to fall back to the pre-mmq expert-tile kernels.
// This variant runs the SAME per-expert scan twice: pass 0 only counts
// (nex_prev, bucket size), pass 1 re-scans and writes ids_src1/ids_dst
// directly at their final offsets.  Bit-identical to the smem kernel by
// construction: identical (it, iex_used) tuples in identical token order and
// identical output expressions — the (lossless) 22/10-bit store round-trip is
// simply removed.  The rescan is cheap: the ids array at the target shape is
// ~190 KB and L2-resident across all expert blocks.  Like the smem kernel,
// behavior is undefined if one token lists the same expert in multiple slots
// (the router's top-k is without replacement, so this cannot occur).
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper_global(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    int nex_prev         = 0; // Number of columns for experts with a lower index.
    int it_compact_count = 0; // Bucket size for this expert (pass-0 result).

#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
        int it_compact = 0; // Running index for the compact slice of this expert.

        if constexpr (n_expert_used_template == 0) {
            // Generic implementation:
            for (int it = 0; it < n_tokens; ++it) {
                int iex_used = -1; // The index at which the expert is used, if any.
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
            // Implementation optimized for specific numbers of experts used:
            static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
            const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it = it0 + threadIdx.x / neu_padded;

                const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
                const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;
                const int iex_used = expert_used == expert ? iex : -1;
                if (pass == 0) {
                    nex_prev += expert_used < expert;
                }

                // Whether the threads at this token position have used the expert:
                const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

                // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
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

                // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
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

// ds4 local: kill switch for the large-n global path (DS4_MMID_LARGE=0).
// With the switch off the ds4_mmq.cu callers refuse past-cap shapes exactly
// as before (whole-MoE fallback to the expert-tile kernels).
bool ds4_mmid_large_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("DS4_MMID_LARGE");
        cached = !(env && env[0] == '0');
    }
    return cached != 0;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);

    // ds4 local (P5): past the smem cap, take the two-pass global variant
    // (bit-identical outputs, see mm_ids_helper_global).  One-shot stderr
    // line = path proof for the gate harnesses.
    if (nbytes_shared > smpbo) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "ds4: mm_ids_helper large-n global path engaged (P5, n_tokens=%d > cap %zu)\n",
                    n_tokens, smpbo / sizeof(mm_ids_helper_store));
        }
        mm_ids_helper_global<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
        return;
    }

    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

// ds4 local: kill switch for the case-1 fast path below (DS4_MMID_CASE1=0).
static bool ds4_mmid_case1_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("DS4_MMID_CASE1");
        cached = !(env && env[0] == '0');
    }
    return cached != 0;
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    switch (n_expert_used) {
        case  1:
            // ds4 local: the routed-MoE down matmul reinterprets (token, slot)
            // assignment rows as single-expert "tokens" (ds4_mmq.cu
            // ds4_mmq_moe_impl, n_expert_used=1).  Without this case it fell to
            // the generic <0> template: one warp per expert scanning all
            // assignment rows with a single active lane -> 22.5 ms/launch at
            // W4096 prefill (2.90 s of a 12k admission).  The optimized template
            // at neu_padded=1 covers 32 rows/iteration and emits bit-identical
            // id maps (proto_mm_ids.cu: parity on uniform/skewed/2%-invalid/
            // decode shapes, 20.4x at the W4096 shape).  DS4_MMID_CASE1=0
            // reverts to the generic path.
            if (ds4_mmid_case1_enabled()) {
                launch_mm_ids_helper< 1>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            } else {
                launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            }
            break;
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
            break;
    }
}
