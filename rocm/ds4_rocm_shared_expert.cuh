extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(gate, n) || !cuda_tensor_has_f32(up, n)) return 0;
    if (n == 0u) return 1;
    swiglu_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes) ||
        !cuda_u64_mul3_checked(in_dim, 1u, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(out_dim, 1u, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(x, x_bytes) || !cuda_tensor_has_bytes(gate, out_bytes) ||
        !cuda_tensor_has_bytes(up, out_bytes) || !cuda_tensor_has_bytes(mid, out_bytes)) {
        return 0;
    }
    if (in_dim == 4096u && (in_dim & 31u) == 0u &&
        cuda_model_range_fits(model_size, gate_offset, weight_bytes) &&
        cuda_model_range_fits(model_size, up_offset, weight_bytes) &&
        !cuda_runtime_config()->disable_shared_gate_up_fused_w32) {
        const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8");
        const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8");
        if (!wg || !wu) return 0;
        const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
        const unsigned rows_per_block = 32u;
        shared_gate_up_swiglu_q8_0_rows_w32_kernel<<<
                (unsigned)((out_dim + rows_per_block - 1u) / rows_per_block),
                rows_per_block * 32u>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                reinterpret_cast<const unsigned char *>(wg),
                reinterpret_cast<const unsigned char *>(wu),
                (const float *)x->ptr,
                (uint32_t)blocks,
                out_dim,
                row_bytes,
                store_gate_up,
                clamp);
        return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 launch");
    }
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

static cudaStream_t g_shared_gate_up_stream = NULL;
static cudaEvent_t g_shared_gate_up_ready_event = NULL;
static void *g_shared_gate_up_tmp = NULL;
static uint64_t g_shared_gate_up_tmp_bytes = 0;
static int g_shared_gate_up_pending = 0;

static int cuda_shared_gate_up_async_wait_internal(void) {
    if (!g_shared_gate_up_pending) return 1;
    cudaError_t err = cudaStreamSynchronize(g_shared_gate_up_stream);
    g_shared_gate_up_pending = 0;
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async wait failed: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static void *cuda_shared_gate_up_async_tmp_alloc(uint64_t bytes) {
    if (bytes == 0) return NULL;
    if (g_shared_gate_up_tmp_bytes >= bytes) return g_shared_gate_up_tmp;
    if (g_shared_gate_up_tmp) {
        (void)cuda_shared_gate_up_async_wait_internal();
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async temp alloc failed (%.2f MiB): %s\n",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_shared_gate_up_tmp = ptr;
    g_shared_gate_up_tmp_bytes = bytes;
    return g_shared_gate_up_tmp;
}

static void cuda_shared_gate_up_async_cleanup(void) {
    if (g_shared_gate_up_stream) {
        (void)cuda_shared_gate_up_async_wait_internal();
    }
    if (g_shared_gate_up_tmp) {
        (void)cudaFree(g_shared_gate_up_tmp);
        g_shared_gate_up_tmp = NULL;
        g_shared_gate_up_tmp_bytes = 0;
    }
    if (g_shared_gate_up_ready_event) {
        (void)cudaEventDestroy(g_shared_gate_up_ready_event);
        g_shared_gate_up_ready_event = NULL;
    }
    if (g_shared_gate_up_stream) {
        (void)cudaStreamDestroy(g_shared_gate_up_stream);
        g_shared_gate_up_stream = NULL;
    }
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_async_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (g_quality_mode || cuda_runtime_config()->graph_dump) return 0;
    if (g_shared_gate_up_pending && !cuda_shared_gate_up_async_wait_internal()) return 0;
    if (!gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) {
        return 0;
    }
    if (g_quality_mode ||
        !gate || !up || !mid || !model_map || !x ||
        in_dim == 0u || out_dim == 0u || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset ||
        x->bytes < in_dim * sizeof(float) ||
        gate->bytes < out_dim * sizeof(float) ||
        up->bytes < out_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_pair_async");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_pair_async");
    if (!wg || !wu) return 0;
    if (!g_shared_gate_up_stream) {
        int least_priority = 0;
        int greatest_priority = 0;
#ifdef __HIP_PLATFORM_AMD__
        hipError_t err = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == hipSuccess) {
            err = hipStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = hipStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != hipSuccess) return 0;
#else
        cudaError_t err = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (err == cudaSuccess) {
            err = cudaStreamCreateWithPriority(&g_shared_gate_up_stream, cudaStreamNonBlocking, least_priority);
        } else {
            (void)cudaGetLastError();
            err = cudaStreamCreateWithFlags(&g_shared_gate_up_stream, cudaStreamNonBlocking);
        }
        if (err != cudaSuccess) return 0;
#endif
    }
    if (!g_shared_gate_up_ready_event) {
        cudaError_t err = cudaEventCreateWithFlags(&g_shared_gate_up_ready_event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async event create failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    /*
     * This stream is intentionally non-blocking so it can overlap routed MoE.
     * Non-blocking streams do not inherit default-stream ordering, so explicitly
     * wait until the default-stream producer of x (ffn_norm) has completed before
     * quantizing it here.
     */
    cudaError_t dep_err = cudaEventRecord(g_shared_gate_up_ready_event, 0);
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency record failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
#ifdef __HIP_PLATFORM_AMD__
    dep_err = hipStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#else
    dep_err = cudaStreamWaitEvent(g_shared_gate_up_stream, g_shared_gate_up_ready_event, 0);
#endif
    if (dep_err != cudaSuccess) {
        fprintf(stderr, DS4_GPU_LOG_PREFIX "shared gate/up async dependency wait failed: %s\n", cudaGetErrorString(dep_err));
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_shared_gate_up_async_tmp_alloc(tmp_bytes);
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = 1;
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, g_shared_gate_up_stream>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async quantize launch")) return 0;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            reinterpret_cast<const unsigned char *>(wg),
            reinterpret_cast<const unsigned char *>(wu),
            xq,
            xscale,
            in_dim,
            out_dim,
            out_dim,
            blocks,
            use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async pair launch")) return 0;
    swiglu_kernel<<<((unsigned)out_dim + 255u) / 256u, 256, 0, g_shared_gate_up_stream>>>(
            (float *)mid->ptr,
            (const float *)gate->ptr,
            (const float *)up->ptr,
            (uint32_t)out_dim,
            clamp,
            1.0f);
    if (!cuda_ok(cudaGetLastError(), "shared gate/up async swiglu launch")) return 0;
    g_shared_gate_up_pending = 1;
    return 1;
}

extern "C" int ds4_gpu_shared_gate_up_async_wait(void) {
    return cuda_shared_gate_up_async_wait_internal();
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_batch_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    uint64_t x_bytes = 0, out_bytes = 0;
    if (!gate || !up || !mid || !model_map || !x || n_tok == 0 ||
        (in_dim & 31u) != 0u || in_dim == 0u || out_dim == 0u ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX ||
        !cuda_u64_mul3_checked(n_tok, in_dim, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked(n_tok, out_dim, sizeof(float), &out_bytes) ||
        x->bytes < x_bytes || gate->bytes < out_bytes || up->bytes < out_bytes || mid->bytes < out_bytes) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(out_dim, row_bytes, &weight_bytes)) return 0;
    if (gate_offset > model_size || up_offset > model_size ||
        weight_bytes > model_size - gate_offset || weight_bytes > model_size - up_offset) {
        return 0;
    }
    const char *wg = cuda_model_range_ptr(model_map, gate_offset, weight_bytes, "shared_gate_q8_batch");
    const char *wu = cuda_model_range_ptr(model_map, up_offset, weight_bytes, "shared_up_q8_batch");
    if (!wg || !wu) return 0;

    const uint32_t rows_per_block = 32u;
    const uint32_t tile = 16u;
    const uint32_t block_tile = 16u;
    const dim3 grid((uint32_t)((out_dim + rows_per_block - 1u) / rows_per_block),
                    (uint32_t)((n_tok + tile - 1u) / tile),
                    1u);
    const size_t shmem = (size_t)tile * block_tile * 32u * sizeof(float);
    const int store_gate_up = (g_quality_mode || cuda_runtime_config()->graph_dump) ? 1 : 0;
#define DS4_LAUNCH_SHARED_GU_BATCH(TT, BT) \
    shared_gate_up_swiglu_q8_0_batch_sharedx_w32_kernel<TT, BT><<<grid, rows_per_block * 32u, shmem>>>( \
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr, \
            reinterpret_cast<const unsigned char *>(wg), reinterpret_cast<const unsigned char *>(wu), \
            (const float *)x->ptr, (uint32_t)blocks, (uint32_t)out_dim, (uint32_t)n_tok, row_bytes, store_gate_up)
    DS4_LAUNCH_SHARED_GU_BATCH(16u, 16u);
#undef DS4_LAUNCH_SHARED_GU_BATCH
    return cuda_ok(cudaGetLastError(), "shared gate/up fused q8 batch launch");
}
