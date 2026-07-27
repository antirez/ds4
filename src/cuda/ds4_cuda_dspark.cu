#include "ds4_cuda_internal.h"


/*
 * Each block reduces the TOP-2 over its slice of the vocabulary and writes the
 * partial winner + runner-up to block_best_id/val[blockIdx.x] and
 * block_second_id/val[blockIdx.x].  The kernel does NOT reduce across blocks —
 * the host caller reads the grid_dim partials back and merges to the global
 * top-2.  (An earlier version wrote only element 0 into a 4-byte buffer, which
 * both overran the allocation and returned block 0's argmax over just the first
 * blockDim entries.)
 *
 * DTree Phase 0: the top-1 (best) track is BYTE-IDENTICAL to the prior
 * argmax-only kernel — every best_* update below is unchanged and the added
 * second_* track never feeds back into it — so refined_id_dst is bit-exact with
 * or without the runner-up requested.  The runner-up (drafter #2) is used only
 * when the host passes a non-NULL refined_id2_dst (measurement path).
 */
__global__ static void dspark_markov_step_kernel(
        float *refined_logits,
        int32_t *block_best_id,
        float *block_best_val,
        int32_t *block_second_id,
        float *block_second_val,
        const float *base_logits,
        const float *markov_w1,
        const float *markov_w2,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    const float *embed = markov_w1 + (uint64_t)prev_token * embed_dim;
    float best_val = -INFINITY;
    int32_t best_id = 0;
    float second_val = -INFINITY;
    int32_t second_id = 0;

    for (uint32_t v = threadIdx.x + blockIdx.x * blockDim.x; v < vocab_size;
         v += blockDim.x * gridDim.x) {
        float dot = 0.0f;
        const float *w2_row = markov_w2 + (uint64_t)v * embed_dim;
        for (uint32_t i = 0; i < embed_dim; i++) dot += w2_row[i] * embed[i];
        float val = base_logits[v] + dot;
        refined_logits[v] = val;
        if (val > best_val) { second_val = best_val; second_id = best_id;
                              best_val = val; best_id = (int32_t)v; }
        else if (val > second_val) { second_val = val; second_id = (int32_t)v; }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    __shared__ float sec_vals[256];
    __shared__ int32_t sec_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    sec_vals[tid] = second_val;
    sec_ids[tid] = second_id;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            /* Merge two sorted (best>=sec) pairs into the top-2 at tid. The
             * best branch is exactly the original strict-'>' update, so best_*
             * evolves identically to the argmax-only kernel. */
            const float ob = best_vals[tid];        const int32_t oi = best_ids[tid];
            const float os = sec_vals[tid];         const int32_t oj = sec_ids[tid];
            const float nb = best_vals[tid + stride]; const int32_t ni = best_ids[tid + stride];
            const float ns = sec_vals[tid + stride]; const int32_t nj = sec_ids[tid + stride];
            if (nb > ob) {
                best_vals[tid] = nb; best_ids[tid] = ni;
                if (ob > ns) { sec_vals[tid] = ob; sec_ids[tid] = oi; }
                else         { sec_vals[tid] = ns; sec_ids[tid] = nj; }
            } else if (nb > os) {
                sec_vals[tid] = nb; sec_ids[tid] = ni;
            }
            (void)oj;
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_best_id[blockIdx.x] = best_ids[0];
        block_best_val[blockIdx.x] = best_vals[0];
        block_second_id[blockIdx.x] = sec_ids[0];
        block_second_val[blockIdx.x] = sec_vals[0];
    }
}

/*
 * Merge the per-block top-2 partials to the global top-2 ON DEVICE, then read
 * back only the two winning ids (8 bytes, one copy).  The old path read all
 * four partial arrays to the host (4 device-serializing D2H copies per draft
 * position, 5 positions per spec step at production depth) and merged there.
 *
 * Tie-break: blocks map to ascending contiguous vocab ranges and the
 * per-block reduction breaks ties toward the lowest id.  Each thread here
 * scans its strided subset of blocks in ascending order with the same
 * strict-'>' merge, and the shared-memory tree keeps the lower-index side as
 * the incumbent, so the lowest-id global argmax is preserved exactly —
 * matching a sequential argmax over the vocab (verified logic mirror of the
 * removed host merge).  The best track is exactly the original argmax merge;
 * out_ids[1] (runner-up) feeds the optional refined_id2 path.
 */
__global__ static void dspark_markov_reduce_kernel(
        int32_t *out_ids,           /* [2]: best id, runner-up id */
        const int32_t *ids,
        const float *vals,
        const int32_t *ids2,
        const float *vals2,
        uint32_t n_blocks) {
    float best_val = -INFINITY;
    int32_t best_id = 0;
    float sec_val = -INFINITY;
    int32_t sec_id = 0;
    for (uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        const float nb = vals[b];  const int32_t ni = ids[b];
        const float ns = vals2[b]; const int32_t nj = ids2[b];
        if (nb > best_val) {
            if (best_val > ns) { sec_val = best_val; sec_id = best_id; }
            else               { sec_val = ns;       sec_id = nj; }
            best_val = nb; best_id = ni;
        } else if (nb > sec_val) {
            sec_val = nb; sec_id = ni;
        }
    }

    __shared__ float best_vals[256];
    __shared__ int32_t best_ids[256];
    __shared__ float sec_vals[256];
    __shared__ int32_t sec_ids[256];
    const uint32_t tid = threadIdx.x;
    best_vals[tid] = best_val;
    best_ids[tid] = best_id;
    sec_vals[tid] = sec_val;
    sec_ids[tid] = sec_id;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float ob = best_vals[tid];          const int32_t oi = best_ids[tid];
            const float os = sec_vals[tid];
            const float nb = best_vals[tid + stride]; const int32_t ni = best_ids[tid + stride];
            const float ns = sec_vals[tid + stride];  const int32_t nj = sec_ids[tid + stride];
            if (nb > ob) {
                best_vals[tid] = nb; best_ids[tid] = ni;
                if (ob > ns) { sec_vals[tid] = ob; sec_ids[tid] = oi; }
                else         { sec_vals[tid] = ns; sec_ids[tid] = nj; }
            } else if (nb > os) {
                sec_vals[tid] = nb; sec_ids[tid] = ni;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_ids[0] = best_ids[0];
        out_ids[1] = sec_ids[0];
    }
}

static int dspark_markov_reduce_blocks(const ds4_gpu_tensor *id_dev,
                                        const ds4_gpu_tensor *val_dev,
                                        const ds4_gpu_tensor *id2_dev,
                                        const ds4_gpu_tensor *val2_dev,
                                        ds4_gpu_tensor *out_dev, /* 2 x int32 */
                                        uint32_t grid_dim,
                                        int32_t *refined_id_dst,
                                        int32_t *refined_id2_dst) {
    dspark_markov_reduce_kernel<<<1, 256>>>(
        (int32_t *)out_dev->ptr,
        (const int32_t *)id_dev->ptr,
        (const float *)val_dev->ptr,
        (const int32_t *)id2_dev->ptr,
        (const float *)val2_dev->ptr,
        grid_dim);
    if (cudaGetLastError() != cudaSuccess) return 0;
    int32_t out[2];
    if (!ds4_gpu_tensor_read(out_dev, 0, out, sizeof(out))) return 0;
    *refined_id_dst = out[0];
    if (refined_id2_dst) *refined_id2_dst = out[1];
    return 1;
}


extern "C" int ds4_gpu_dspark_markov_step_model(
        ds4_gpu_tensor *refined_logits,
        int32_t *refined_id_dst,
        int32_t *refined_id2_dst,
        const ds4_gpu_tensor *base_logits,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t markov_w2_offset,
        int32_t prev_token,
        uint32_t vocab_size,
        uint32_t embed_dim) {
    if (!refined_logits || !refined_id_dst || !base_logits || !dspark_model_map)
        return 0;
    if (vocab_size == 0 || embed_dim == 0 || embed_dim > 1024) return 0;
    if (refined_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if (base_logits->bytes < (uint64_t)vocab_size * sizeof(float)) return 0;
    if ((uint64_t)prev_token >= vocab_size) return 0;

    const uint64_t w_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    if (markov_w1_offset > dspark_model_size ||
        w_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (markov_w2_offset > dspark_model_size ||
        w_bytes > dspark_model_size - markov_w2_offset) return 0;

    const float *w1 = (const float *)cuda_model_range_ptr(
        dspark_model_map, markov_w1_offset, w_bytes, "dspark_markov_w1");
    const float *w2 = (const float *)cuda_model_range_ptr(
        dspark_model_map, markov_w2_offset, w_bytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (vocab_size + block_dim - 1) / block_dim;
    if (grid_dim > 65535) return 0;

    /* Process-persistent reduce buffers, grown on demand: grid_dim is fixed for a
     * given vocab, and this runs once per draft position per spec step -- per-call
     * cudaMalloc/cudaFree pairs were 2 device-serializing allocs each. Single
     * submission thread. Grouped into one struct (Pulsar C++ port, plan-70 TU
     * dspark) -- same alloc sizes, free order, and grow-on-demand logic as the
     * prior loose statics, so the launches are byte-identical. */
    struct DsparkReduceBufs {
        ds4_gpu_tensor *id, *val, *id2, *val2, *out;
        uint32_t cap;
    };
    static DsparkReduceBufs rb = {};
    if (grid_dim > rb.cap) {
        ds4_gpu_tensor_free(rb.id);
        ds4_gpu_tensor_free(rb.val);
        ds4_gpu_tensor_free(rb.id2);
        ds4_gpu_tensor_free(rb.val2);
        ds4_gpu_tensor_free(rb.out);
        rb.id   = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        rb.val  = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        rb.id2  = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(int32_t));
        rb.val2 = ds4_gpu_tensor_alloc((uint64_t)grid_dim * sizeof(float));
        rb.out  = ds4_gpu_tensor_alloc(2u * sizeof(int32_t));
        rb.cap  = (rb.id && rb.val && rb.id2 && rb.val2 && rb.out) ? grid_dim : 0;
    }
    if (!rb.id || !rb.val || !rb.id2 || !rb.val2 || !rb.out) return 0;

    dspark_markov_step_kernel<<<grid_dim, block_dim>>>(
        (float *)refined_logits->ptr,
        (int32_t *)rb.id->ptr,
        (float *)rb.val->ptr,
        (int32_t *)rb.id2->ptr,
        (float *)rb.val2->ptr,
        (const float *)base_logits->ptr,
        w1, w2, prev_token, vocab_size, embed_dim);

    int rc = 0;
    if (cudaGetLastError() == cudaSuccess)
        rc = dspark_markov_reduce_blocks(rb.id, rb.val, rb.id2, rb.val2,
                                         rb.out, grid_dim,
                                         refined_id_dst, refined_id2_dst);
    return rc;
}


/* DSpark confidence head (model.py DSparkConfidenceHead): per block position,
 * scores[p] = sigmoid( proj . cat(hidden[p], markov_embed[token_ids[p]]) ), where
 * hidden is the post-hc_head drafter hidden and markov_embed is a row of markov_w1
 * gathered by token id. Used to size the verify budget (confidence-scheduled
 * verification) -- higher confidence => the draft is more likely accepted. */
__global__ static void dspark_confidence_score_kernel(
        float *scores,
        const float *hidden,        /* [n_positions, hidden_dim] */
        const int32_t *token_ids,   /* [n_positions] */
        const float *markov_w1,     /* [vocab, embed_dim] */
        const float *proj,          /* [hidden_dim + embed_dim] */
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size) {
    uint32_t p = blockIdx.x;
    if (p >= n_positions) return;
    int32_t t = token_ids[p];
    if (t < 0 || (uint32_t)t >= vocab_size) t = 0;
    const float *hp = hidden + (uint64_t)p * hidden_dim;
    const float *emb = markov_w1 + (uint64_t)t * embed_dim;
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) dot += hp[i] * proj[i];
    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x) dot += emb[i] * proj[hidden_dim + i];
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[p] = 1.0f / (1.0f + expf(-partial[0]));
}


extern "C" int ds4_gpu_dspark_confidence_score_model(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *token_ids,
        const void *dspark_model_map,
        uint64_t dspark_model_size,
        uint64_t markov_w1_offset,
        uint64_t proj_offset,
        uint32_t n_positions, uint32_t hidden_dim, uint32_t embed_dim, uint32_t vocab_size) {
    if (!scores || !hidden || !token_ids || !dspark_model_map) return 0;
    if (n_positions == 0 || hidden_dim == 0 || embed_dim == 0 || vocab_size == 0) return 0;
    if (scores->bytes < (uint64_t)n_positions * sizeof(float)) return 0;
    if (hidden->bytes < (uint64_t)n_positions * hidden_dim * sizeof(float)) return 0;
    if (token_ids->bytes < (uint64_t)n_positions * sizeof(int32_t)) return 0;
    const uint64_t w1_bytes = (uint64_t)vocab_size * embed_dim * sizeof(float);
    const uint64_t proj_bytes = (uint64_t)(hidden_dim + embed_dim) * sizeof(float);
    if (markov_w1_offset > dspark_model_size || w1_bytes > dspark_model_size - markov_w1_offset) return 0;
    if (proj_offset > dspark_model_size || proj_bytes > dspark_model_size - proj_offset) return 0;
    const float *w1 = (const float *)cuda_model_range_ptr(dspark_model_map, markov_w1_offset, w1_bytes, "dspark_conf_w1");
    const float *proj = (const float *)cuda_model_range_ptr(dspark_model_map, proj_offset, proj_bytes, "dspark_conf_proj");
    if (!w1 || !proj) return 0;
    dspark_confidence_score_kernel<<<n_positions, 256>>>(
        (float *)scores->ptr, (const float *)hidden->ptr, (const int32_t *)token_ids->ptr,
        w1, proj, n_positions, hidden_dim, embed_dim, vocab_size);
    return cuda_ok(cudaGetLastError(), "dspark confidence score");
}


/* after_ffn_hc is an HC residual CARRIER (BF16 storage; task #62) — load via
 * ds4_hc_load, accumulate in f32. Output is a plain f32 embedding row. */
__global__ static void dspark_hc_mean_reduce_kernel(
        float *out,
        const ds4_hc_t *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += ds4_hc_load(after_ffn_hc, (uint64_t)hc * n_embd + d);
        out[d] = sum / (float)n_hc;
    }
}

extern "C" int ds4_gpu_dspark_hc_mean_reduce(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *after_ffn_hc,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!out || !after_ffn_hc || n_embd == 0 || n_hc == 0) return 0;
    if (out->bytes < (uint64_t)n_embd * sizeof(float)) return 0;
    if (after_ffn_hc->bytes < (uint64_t)n_hc * n_embd * DS4_HC_ELT_SIZE) return 0;   /* carrier */

    const uint32_t block_dim = 256;
    const uint32_t grid_dim = (n_embd + block_dim - 1) / block_dim;

    dspark_hc_mean_reduce_kernel<<<grid_dim, block_dim>>>(
        (float *)out->ptr,
        (const ds4_hc_t *)after_ffn_hc->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce");
}


/* Batched variant over a [n_tokens, n_hc, n_embd] HC tensor: out[p] = mean over
 * hc of in[p]. Captures the drafter's anchor-layer hidden for EVERY position of
 * a spec verify batch, so the fused loop can pick the last-accepted position's
 * hidden without replaying. */
__global__ static void dspark_hc_mean_reduce_batch_kernel(
        float *out,
        const ds4_hc_t *hc_batch,   /* HC residual carrier (BF16); task #62 */
        uint32_t n_embd,
        uint32_t n_hc) {
    const uint32_t p = blockIdx.y;
    const ds4_hc_t *in = hc_batch + (uint64_t)p * n_hc * n_embd;
    float *op = out + (uint64_t)p * n_embd;
    for (uint32_t d = threadIdx.x + blockIdx.x * blockDim.x; d < n_embd;
         d += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (uint32_t hc = 0; hc < n_hc; hc++)
            sum += ds4_hc_load(in, (uint64_t)hc * n_embd + d);
        op[d] = sum / (float)n_hc;
    }
}


extern "C" int ds4_gpu_dspark_hc_mean_reduce_batch(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *hc_batch,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens) {
    if (!out || !hc_batch || n_embd == 0 || n_hc == 0 || n_tokens == 0) return 0;
    if (out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float)) return 0;
    if (hc_batch->bytes < (uint64_t)n_tokens * n_hc * n_embd * DS4_HC_ELT_SIZE) return 0;   /* carrier */

    const uint32_t block_dim = 256;
    dim3 grid((n_embd + block_dim - 1) / block_dim, n_tokens, 1);
    dspark_hc_mean_reduce_batch_kernel<<<grid, block_dim>>>(
        (float *)out->ptr,
        (const ds4_hc_t *)hc_batch->ptr,
        n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "dspark hc mean reduce batch");
}
