// TurboQuant Metal shaders for DS4
// KV cache compression via PolarQuant + Walsh-Hadamard rotation
// Based on: arXiv 2504.19874 (ICLR 2026)
// Reference: github.com/TheTom/llama-cpp-turboquant
//
// 3-bit (turbo3_0): 8 centroids, 14 bytes per 32-element block
// 4-bit (turbo4_0): 16 centroids, 18 bytes per 32-element block
// Per-block FWHT rotation (32-element butterfly, 160 ops vs 1024 for dense)
//
// NOTE: This file is concatenated into the single Metal library by ds4_metal.m.
// All content must be self-contained (no #include of other metal headers).

// --- Block type definitions ---

#define QK_TURBO 32

struct block_turbo3_0 {
    // 2 low bits per element: 32 * 2 = 64 bits = 8 bytes
    uchar qs[QK_TURBO / 4];
    // 1 high bit per element: 32 * 1 = 32 bits = 4 bytes
    uchar signs[QK_TURBO / 8];
    // per-block norm (fp16)
    half norm;
};

struct block_turbo4_0 {
    // 4 bits (nibble) per element: 32 * 4 = 128 bits = 16 bytes
    uchar qs[QK_TURBO / 2];
    // per-block norm (fp16)
    half norm;
};

// --- FWHT-32 sign array ---
constant float turbo_signs_32[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

// --- 3-bit centroid tables ---
constant float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

constant float turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// --- 4-bit centroid tables ---
constant float turbo_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

constant float turbo_mid_4bit[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f,  0.000000f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};

// --- Argument structs ---

struct ds4_metal_args_turbo_quantize {
    int32_t ne00;   // head_dim
    int32_t ne01;   // n_tok
    int32_t ne02;   // n_head
    int32_t ne03;
    int32_t n_rot;  // RoPE dimension
    uint64_t nb00;  // src element stride
    uint64_t nb01;  // src row stride
    uint64_t nb02;  // src head stride
    uint64_t nb03;
    uint64_t nb0;   // dst element stride (turbo block stride)
    uint64_t nb1;   // dst row stride
    uint64_t nb2;   // dst head stride
    uint64_t nb3;
};

struct ds4_metal_args_turbo_dequant {
    int32_t n_blocks; // number of turbo blocks
    int32_t n_rot;    // RoPE tail elements
    int32_t n_rows;   // total rows
    uint64_t src_stride; // bytes per row in source (turbo blocks + rope tail)
    uint64_t dst_stride; // bytes per row in destination (half values)
};

struct ds4_metal_args_turbo_select_dequant {
    int32_t n_blocks;
    int32_t n_rot;
    int32_t n_comp;
    int32_t top_k;
    int32_t n_tokens;
    uint64_t src_stride;
    uint64_t dst_stride;
    uint64_t topk_token_stride;
};

// --- FWHT-32: in-place butterfly + normalize (thread-safe, parallelized) ---
// Each thread handles its element for sign application, then cooperates on butterfly
// stages with barriers between each stage to avoid race conditions on threadgroup memory.
static void turbo_fwht_32(threadgroup float * x, uint tid) {
    const float inv_sqrt_32 = 0.17677669529663688f;

    // signs1 (per-element, no race)
    x[tid] *= turbo_signs_32[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // butterfly stages with barriers
    for (int h = 1; h < 32; h *= 2) {
        int i = (int)tid;
        int block = i / (h * 2);
        int offset = i % (h * 2);
        if (offset < h) {
            int idx_a = block * h * 2 + offset;
            int idx_b = idx_a + h;
            float a = x[idx_a];
            float b = x[idx_b];
            x[idx_a] = a + b;
            x[idx_b] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // normalize + signs2 (sign array is self-inverse since +/-1)
    x[tid] *= inv_sqrt_32 * turbo_signs_32[tid];
}

// --- Nearest centroid helpers ---
static int nearest_centroid_3bit(float val) {
    if (val < turbo_mid_3bit[0]) return 0;
    if (val < turbo_mid_3bit[1]) return 1;
    if (val < turbo_mid_3bit[2]) return 2;
    if (val < turbo_mid_3bit[3]) return 3;
    if (val < turbo_mid_3bit[4]) return 4;
    if (val < turbo_mid_3bit[5]) return 5;
    if (val < turbo_mid_3bit[6]) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    if (val < turbo_mid_4bit[ 0]) return 0;
    if (val < turbo_mid_4bit[ 1]) return 1;
    if (val < turbo_mid_4bit[ 2]) return 2;
    if (val < turbo_mid_4bit[ 3]) return 3;
    if (val < turbo_mid_4bit[ 4]) return 4;
    if (val < turbo_mid_4bit[ 5]) return 5;
    if (val < turbo_mid_4bit[ 6]) return 6;
    if (val < turbo_mid_4bit[ 7]) return 7;
    if (val < turbo_mid_4bit[ 8]) return 8;
    if (val < turbo_mid_4bit[ 9]) return 9;
    if (val < turbo_mid_4bit[10]) return 10;
    if (val < turbo_mid_4bit[11]) return 11;
    if (val < turbo_mid_4bit[12]) return 12;
    if (val < turbo_mid_4bit[13]) return 13;
    if (val < turbo_mid_4bit[14]) return 14;
    return 15;
}

// ============================================================================
// 3-bit quantize kernel
// ============================================================================
// Grid: (n_tok * n_head, 1, 1), Threadgroup: (32, 1, 1)
// Each threadgroup processes one 32-element block.
// One row (head_dim=512, n_rot=64) = 14 blocks. RoPE tail stays in source.
kernel void kernel_turbo3_quantize_f32(
        constant ds4_metal_args_turbo_quantize & args,
        device  const float * src,
        device  block_turbo3_0 * dst,
        threadgroup float * scratch [[threadgroup(0)]],
        uint block_row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {

    const int32_t head_dim = args.ne00;
    const int32_t n_rot    = args.n_rot;
    const int32_t n_nope   = head_dim - n_rot;
    const int32_t n_blocks = n_nope / QK_TURBO;

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    const int64_t row   = block_row / n_blocks;
    const int64_t blk   = block_row % n_blocks;

    if (row >= n_rows || blk >= n_blocks) return;

    const int64_t i1 = (row / args.ne02) % args.ne01;
    const int64_t i2 = row % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const float  * src_row = (device const float  *)((device const char *)src + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device block_turbo3_0 * dst_blk = (device block_turbo3_0 *)((device char *)dst + row*args.nb1 + blk*sizeof(block_turbo3_0));

    // Step 1: Load 32 elements into scratch, compute L2 norm
    float v = src_row[blk * QK_TURBO + tid];
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) {
        norm_sq += scratch[i] * scratch[i];
    }
    float blk_norm = sqrt(norm_sq);
    float inv_norm = (blk_norm > 1e-10f) ? 1.0f / blk_norm : 0.0f;

    // Step 2: Normalize
    scratch[tid] *= inv_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: FWHT-32 rotation
    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Quantize, then pack with a single writer per output byte.
    float v_rot = scratch[tid];
    int idx = nearest_centroid_3bit(v_rot);
    scratch[tid] = (float)idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < QK_TURBO / 4) {
        uchar packed = 0;
        for (uint j = 0; j < 4; j++) {
            uint elem = tid * 4 + j;
            uint qidx = (uint)scratch[elem];
            packed |= (uchar)((qidx & 0x3u) << (j * 2));
        }
        dst_blk->qs[tid] = packed;
    }
    if (tid < QK_TURBO / 8) {
        uchar packed = 0;
        for (uint j = 0; j < 8; j++) {
            uint elem = tid * 8 + j;
            uint qidx = (uint)scratch[elem];
            packed |= (uchar)(((qidx >> 2) & 0x1u) << j);
        }
        dst_blk->signs[tid] = packed;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate recon norm squared
    scratch[tid] = turbo_centroids_3bit[idx] * turbo_centroids_3bit[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction for block recon norm
    for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 5: Store corrected norm
    if (tid == 0) {
        float recon_norm = sqrt(scratch[0]);
        float corrected = (recon_norm > 1e-10f) ? blk_norm / recon_norm : blk_norm;
        dst_blk->norm = (half)corrected;
    }

    if (blk == 0) {
        device float * rope_dst = (device float *)((device char *)dst + row * args.nb1 +
                                                   n_blocks * sizeof(block_turbo3_0));
        for (int i = tid; i < n_rot; i += QK_TURBO) {
            rope_dst[i] = src_row[n_nope + i];
        }
    }
}

// ============================================================================
// 4-bit quantize kernel
// ============================================================================
kernel void kernel_turbo4_quantize_f32(
        constant ds4_metal_args_turbo_quantize & args,
        device  const float * src,
        device  block_turbo4_0 * dst,
        threadgroup float * scratch [[threadgroup(0)]],
        uint block_row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {

    const int32_t head_dim = args.ne00;
    const int32_t n_rot    = args.n_rot;
    const int32_t n_nope   = head_dim - n_rot;
    const int32_t n_blocks = n_nope / QK_TURBO;

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    const int64_t row   = block_row / n_blocks;
    const int64_t blk   = block_row % n_blocks;

    if (row >= n_rows || blk >= n_blocks) return;

    const int64_t i1 = (row / args.ne02) % args.ne01;
    const int64_t i2 = row % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    device const float  * src_row = (device const float  *)((device const char *)src + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device block_turbo4_0 * dst_blk = (device block_turbo4_0 *)((device char *)dst + row*args.nb1 + blk*sizeof(block_turbo4_0));

    // Step 1: Load, compute norm
    float v = src_row[blk * QK_TURBO + tid];
    scratch[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO; i++) norm_sq += scratch[i] * scratch[i];
    float blk_norm = sqrt(norm_sq);
    float inv_norm = (blk_norm > 1e-10f) ? 1.0f / blk_norm : 0.0f;

    // Step 2: Normalize
    scratch[tid] *= inv_norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: FWHT-32 rotation
    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: 4-bit quantize and nibble pack
    float v_rot = scratch[tid];
    int idx = nearest_centroid_4bit(v_rot);

    scratch[tid] = (float)idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < QK_TURBO / 2) {
        uint lo = (uint)scratch[tid * 2 + 0];
        uint hi = (uint)scratch[tid * 2 + 1];
        dst_blk->qs[tid] = (uchar)((lo & 0xFu) | ((hi & 0xFu) << 4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    scratch[tid] = turbo_centroids_4bit[idx] * turbo_centroids_4bit[idx];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 16; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 5: Store corrected norm
    if (tid == 0) {
        float recon_norm = sqrt(scratch[0]);
        float corrected = (recon_norm > 1e-10f) ? blk_norm / recon_norm : blk_norm;
        dst_blk->norm = (half)corrected;
    }

    if (blk == 0) {
        device float * rope_dst = (device float *)((device char *)dst + row * args.nb1 +
                                                   n_blocks * sizeof(block_turbo4_0));
        for (int i = tid; i < n_rot; i += QK_TURBO) {
            rope_dst[i] = src_row[n_nope + i];
        }
    }
}

// ============================================================================
// Dequantize-to-f32 kernels (for feeding existing flash attention)
// ============================================================================
// Output format is float32 in the original KV basis to match the existing
// attention dispatch.  Grid: (n_rows * n_blocks, 1, 1), Threadgroup: (32, 1, 1).

kernel void kernel_turbo3_dequant_f32(
        constant ds4_metal_args_turbo_dequant & args,
        device  const block_turbo3_0 * src,
        device        float * dst,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {

    const uint row = gid / (uint)args.n_blocks;
    const uint blk = gid % (uint)args.n_blocks;
    if ((int)row >= args.n_rows) return;

    device const block_turbo3_0 * src_row = (device const block_turbo3_0 *)((device const char *)src + row * args.src_stride);
    device float * dst_row = (device float *)((device char *)dst + row * args.dst_stride);

    device const block_turbo3_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar q_byte   = blk_ptr->qs[tid / 4];
    uchar s_byte   = blk_ptr->signs[tid / 8];
    uchar low2     = (q_byte >> ((tid % 4) * 2)) & 0x3;
    uchar hi1      = (s_byte >> (tid % 8)) & 0x1;
    uchar idx      = low2 | (hi1 << 2);
    scratch[tid] = turbo_centroids_3bit[idx] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = rope_src[i];
        }
    }
}

kernel void kernel_turbo4_dequant_f32(
        constant ds4_metal_args_turbo_dequant & args,
        device  const block_turbo4_0 * src,
        device        float * dst,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {

    const uint row = gid / (uint)args.n_blocks;
    const uint blk = gid % (uint)args.n_blocks;
    if ((int)row >= args.n_rows) return;

    device const block_turbo4_0 * src_row = (device const block_turbo4_0 *)((device const char *)src + row * args.src_stride);
    device float * dst_row = (device float *)((device char *)dst + row * args.dst_stride);

    device const block_turbo4_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar nibble = (blk_ptr->qs[tid / 2] >> ((tid % 2) * 4)) & 0xF;
    scratch[tid] = turbo_centroids_4bit[nibble] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = rope_src[i];
        }
    }
}

kernel void kernel_turbo3_dequant_selected_f32(
        constant ds4_metal_args_turbo_select_dequant & args,
        device  const block_turbo3_0 * src,
        device  const int32_t * topk,
        device        float * dst,
        device        int32_t * identity_topk,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint block_count = (uint)args.n_blocks;
    const uint sel = (gid / block_count) % (uint)args.top_k;
    const uint token = gid / (block_count * (uint)args.top_k);
    const uint blk = gid % block_count;
    if ((int)token >= args.n_tokens) return;

    device const int32_t *row_topk = (device const int32_t *)((device const char *)topk +
        (uint64_t)token * args.topk_token_stride);
    const int32_t src_idx = row_topk[sel];
    if (src_idx < 0 || src_idx >= args.n_comp) return;
    if (blk == 0 && tid == 0) {
        identity_topk[(uint64_t)token * (uint)args.top_k + sel] = (int32_t)sel;
    }

    device const block_turbo3_0 * src_row =
        (device const block_turbo3_0 *)((device const char *)src + (uint64_t)(uint)src_idx * args.src_stride);
    device float * dst_row = (device float *)((device char *)dst +
        ((uint64_t)token * (uint)args.top_k + sel) * args.dst_stride);

    device const block_turbo3_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar q_byte = blk_ptr->qs[tid / 4];
    uchar s_byte = blk_ptr->signs[tid / 8];
    uchar low2 = (q_byte >> ((tid % 4) * 2)) & 0x3;
    uchar hi1 = (s_byte >> (tid % 8)) & 0x1;
    uchar idx = low2 | (hi1 << 2);
    scratch[tid] = turbo_centroids_3bit[idx] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = rope_src[i];
        }
    }
}

kernel void kernel_turbo4_dequant_selected_f32(
        constant ds4_metal_args_turbo_select_dequant & args,
        device  const block_turbo4_0 * src,
        device  const int32_t * topk,
        device        float * dst,
        device        int32_t * identity_topk,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint block_count = (uint)args.n_blocks;
    const uint sel = (gid / block_count) % (uint)args.top_k;
    const uint token = gid / (block_count * (uint)args.top_k);
    const uint blk = gid % block_count;
    if ((int)token >= args.n_tokens) return;

    device const int32_t *row_topk = (device const int32_t *)((device const char *)topk +
        (uint64_t)token * args.topk_token_stride);
    const int32_t src_idx = row_topk[sel];
    if (src_idx < 0 || src_idx >= args.n_comp) return;
    if (blk == 0 && tid == 0) {
        identity_topk[(uint64_t)token * (uint)args.top_k + sel] = (int32_t)sel;
    }

    device const block_turbo4_0 * src_row =
        (device const block_turbo4_0 *)((device const char *)src + (uint64_t)(uint)src_idx * args.src_stride);
    device float * dst_row = (device float *)((device char *)dst +
        ((uint64_t)token * (uint)args.top_k + sel) * args.dst_stride);

    device const block_turbo4_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar nibble = (blk_ptr->qs[tid / 2] >> ((tid % 2) * 4)) & 0xF;
    scratch[tid] = turbo_centroids_4bit[nibble] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = rope_src[i];
        }
    }
}

kernel void kernel_turbo3_dequant_selected_f16(
        constant ds4_metal_args_turbo_select_dequant & args,
        device  const block_turbo3_0 * src,
        device  const int32_t * topk,
        device        half * dst,
        device        int32_t * identity_topk,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint block_count = (uint)args.n_blocks;
    const uint sel = (gid / block_count) % (uint)args.top_k;
    const uint token = gid / (block_count * (uint)args.top_k);
    const uint blk = gid % block_count;
    if ((int)token >= args.n_tokens) return;

    device const int32_t *row_topk = (device const int32_t *)((device const char *)topk +
        (uint64_t)token * args.topk_token_stride);
    const int32_t src_idx = row_topk[sel];
    if (src_idx < 0 || src_idx >= args.n_comp) return;
    if (blk == 0 && tid == 0) {
        identity_topk[(uint64_t)token * (uint)args.top_k + sel] = (int32_t)sel;
    }

    device const block_turbo3_0 * src_row =
        (device const block_turbo3_0 *)((device const char *)src + (uint64_t)(uint)src_idx * args.src_stride);
    device half * dst_row = (device half *)((device char *)dst +
        ((uint64_t)token * (uint)args.top_k + sel) * args.dst_stride);

    device const block_turbo3_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar q_byte = blk_ptr->qs[tid / 4];
    uchar s_byte = blk_ptr->signs[tid / 8];
    uchar low2 = (q_byte >> ((tid % 4) * 2)) & 0x3;
    uchar hi1 = (s_byte >> (tid % 8)) & 0x1;
    uchar idx = low2 | (hi1 << 2);
    scratch[tid] = turbo_centroids_3bit[idx] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = (half)scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = (half)rope_src[i];
        }
    }
}

kernel void kernel_turbo4_dequant_selected_f16(
        constant ds4_metal_args_turbo_select_dequant & args,
        device  const block_turbo4_0 * src,
        device  const int32_t * topk,
        device        half * dst,
        device        int32_t * identity_topk,
        threadgroup float * scratch [[threadgroup(0)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint block_count = (uint)args.n_blocks;
    const uint sel = (gid / block_count) % (uint)args.top_k;
    const uint token = gid / (block_count * (uint)args.top_k);
    const uint blk = gid % block_count;
    if ((int)token >= args.n_tokens) return;

    device const int32_t *row_topk = (device const int32_t *)((device const char *)topk +
        (uint64_t)token * args.topk_token_stride);
    const int32_t src_idx = row_topk[sel];
    if (src_idx < 0 || src_idx >= args.n_comp) return;
    if (blk == 0 && tid == 0) {
        identity_topk[(uint64_t)token * (uint)args.top_k + sel] = (int32_t)sel;
    }

    device const block_turbo4_0 * src_row =
        (device const block_turbo4_0 *)((device const char *)src + (uint64_t)(uint)src_idx * args.src_stride);
    device half * dst_row = (device half *)((device char *)dst +
        ((uint64_t)token * (uint)args.top_k + sel) * args.dst_stride);

    device const block_turbo4_0 * blk_ptr = &src_row[blk];
    float norm = (float)blk_ptr->norm;

    uchar nibble = (blk_ptr->qs[tid / 2] >> ((tid % 2) * 4)) & 0xF;
    scratch[tid] = turbo_centroids_4bit[nibble] * norm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    turbo_fwht_32(scratch, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst_row[blk * QK_TURBO + tid] = (half)scratch[tid];

    if (blk == 0) {
        device const float * rope_src = (device const float *)(src_row + args.n_blocks);
        for (int i = tid; i < args.n_rot; i += QK_TURBO) {
            dst_row[args.n_blocks * QK_TURBO + i] = (half)rope_src[i];
        }
    }
}

// ============================================================================
// Fused indexed attention row staging
// ============================================================================
// These kernels keep the existing PolarQuant/WHT cache format but avoid the
// decode-time full-cache dequantization pass. Only the selected top-k rows are
// expanded into the indexed attention threadgroup tile.

static inline void dsv4_indexed_mixed_attention_comp_f16_impl(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared,
        uint2  tgpig,
        ushort tid,
        ushort lane,
        ushort sg,
        bool rb4) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        if (rb4) {
            for (uint pos0 = first; pos0 <= last; pos0 += 4u) {
                const uint n_rows = min(4u, last - pos0 + 1u);
                for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                    const uint r = off >> 7;
                    const uint c = off & 127u;
                    const uint logical = pos0 + r - first_raw_pos;
                    const uint row = (args.raw_start + logical) % args.raw_cap;
                    device const float4 *src = (device const float4 *)(raw_kv +
                        (uint64_t)row * args.raw_row_stride);
                    kv_shared[off] = src[c];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint r = 0; r < n_rows; r++) {
                    dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                         r,
                                                         q0, q1, q2, q3,
                                                         args.scale,
                                                         lane,
                                                         M, S,
                                                         o0, o1, o2, o3);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (uint pos = first; pos <= last; pos++) {
                const uint logical = pos - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                if (tid < 128) kv_shared[tid] = src[tid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                                  q0, q1, q2, q3,
                                                  args.scale,
                                                  lane,
                                                  M, S,
                                                  o0, o1, o2, o3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    if (rb4) {
        bool stop = false;
        for (uint i = 0; i < args.top_k && !stop; i += 4u) {
            uint rows[4];
            uint n_rows = 0;
            for (uint j = 0; j < 4u && i + j < args.top_k; j++) {
                const int32_t idx = row_topk[i + j];
                if (idx < 0) {
                    continue;
                }
                if ((uint)idx >= visible) {
                    stop = true;
                    break;
                }
                rows[n_rows++] = (uint)idx;
            }
            if (n_rows == 0) {
                continue;
            }
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                device const half4 *src = (device const half4 *)(comp_kv +
                    (uint64_t)rows[r] * args.comp_row_stride);
                kv_shared[off] = (float4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                     r,
                                                     q0, q1, q2, q3,
                                                     args.scale,
                                                     lane,
                                                     M, S,
                                                     o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint i = 0; i < args.top_k; i++) {
            const int32_t idx = row_topk[i];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                break;
            }
            device const half4 *src = (device const half4 *)(comp_kv +
                (uint64_t)(uint)idx * args.comp_row_stride);
            if (tid < 128) kv_shared[tid] = (float4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                              q0, q1, q2, q3,
                                              args.scale,
                                              lane,
                                              M, S,
                                              o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_comp_f16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_comp_f16_impl(args, q, raw_kv, comp_kv,
                                               topk, sinks, dst, kv_shared,
                                               tgpig, tid, lane, sg, false);
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb4_comp_f16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_comp_f16_impl(args, q, raw_kv, comp_kv,
                                               topk, sinks, dst, kv_shared,
                                               tgpig, tid, lane, sg, true);
}

static void turbo_fwht_32_simd(threadgroup float * x, uint tid) {
    const float inv_sqrt_32 = 0.17677669529663688f;

    x[tid] *= turbo_signs_32[tid];
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = 1; h < 32; h *= 2) {
        int i = (int)tid;
        int block = i / (h * 2);
        int offset = i % (h * 2);
        if (offset < h) {
            int idx_a = block * h * 2 + offset;
            int idx_b = idx_a + h;
            float a = x[idx_a];
            float b = x[idx_b];
            x[idx_a] = a + b;
            x[idx_b] = a - b;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    x[tid] *= inv_sqrt_32 * turbo_signs_32[tid];
}

static inline void dsv4_turbo3_dequant_shared_row(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *comp_kv,
        uint row,
        threadgroup float4 *dst4,
        threadgroup float *scratch,
        ushort lane,
        ushort sg) {
    const uint n_blocks = args.turbo_n_blocks;
    const uint n_rot = args.turbo_n_rot;
    const uint n_nope = n_blocks * QK_TURBO;
    device const block_turbo3_0 *src_row =
        (device const block_turbo3_0 *)(comp_kv + (uint64_t)row * args.comp_row_stride);
    threadgroup float *dst = (threadgroup float *)dst4;
    threadgroup float *s = scratch + (uint)sg * QK_TURBO;

    for (uint blk = (uint)sg; blk < n_blocks; blk += 8u) {
        device const block_turbo3_0 *blk_ptr = &src_row[blk];
        const uint t = (uint)lane;
        uchar q_byte = blk_ptr->qs[t / 4u];
        uchar s_byte = blk_ptr->signs[t / 8u];
        uchar low2 = (q_byte >> ((t % 4u) * 2u)) & 0x3;
        uchar hi1 = (s_byte >> (t % 8u)) & 0x1;
        uchar idx = low2 | (hi1 << 2);
        s[t] = turbo_centroids_3bit[idx] * (float)blk_ptr->norm;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        turbo_fwht_32_simd(s, t);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        dst[blk * QK_TURBO + t] = s[t];
    }

    device const float *rope_src = (device const float *)(src_row + n_blocks);
    for (uint i = (uint)sg * QK_TURBO + (uint)lane; i < n_rot; i += 8u * QK_TURBO) {
        dst[n_nope + i] = rope_src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void dsv4_turbo4_dequant_shared_row(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *comp_kv,
        uint row,
        threadgroup float4 *dst4,
        threadgroup float *scratch,
        ushort lane,
        ushort sg) {
    const uint n_blocks = args.turbo_n_blocks;
    const uint n_rot = args.turbo_n_rot;
    const uint n_nope = n_blocks * QK_TURBO;
    device const block_turbo4_0 *src_row =
        (device const block_turbo4_0 *)(comp_kv + (uint64_t)row * args.comp_row_stride);
    threadgroup float *dst = (threadgroup float *)dst4;
    threadgroup float *s = scratch + (uint)sg * QK_TURBO;

    for (uint blk = (uint)sg; blk < n_blocks; blk += 8u) {
        device const block_turbo4_0 *blk_ptr = &src_row[blk];
        const uint t = (uint)lane;
        uchar nibble = (blk_ptr->qs[t / 2u] >> ((t % 2u) * 4u)) & 0xF;
        s[t] = turbo_centroids_4bit[nibble] * (float)blk_ptr->norm;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        turbo_fwht_32_simd(s, t);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        dst[blk * QK_TURBO + t] = s[t];
    }

    device const float *rope_src = (device const float *)(src_row + n_blocks);
    for (uint i = (uint)sg * QK_TURBO + (uint)lane; i < n_rot; i += 8u * QK_TURBO) {
        dst[n_nope + i] = rope_src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void dsv4_turbo_dequant_shared_row(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *comp_kv,
        uint row,
        threadgroup float4 *dst4,
        threadgroup float *scratch,
        ushort lane,
        ushort sg,
        bool turbo4) {
    if (turbo4) {
        dsv4_turbo4_dequant_shared_row(args, comp_kv, row, dst4, scratch, lane, sg);
    } else {
        dsv4_turbo3_dequant_shared_row(args, comp_kv, row, dst4, scratch, lane, sg);
    }
}

static inline void dsv4_indexed_mixed_attention_turbo_impl(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared,
        uint2  tgpig,
        ushort tid,
        ushort lane,
        ushort sg,
        bool turbo4,
        bool rb4) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    const uint shared_rows = rb4 ? 4u : 1u;
    threadgroup float *scratch = (threadgroup float *)(kv_shared + shared_rows * 128u);

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        if (rb4) {
            for (uint pos0 = first; pos0 <= last; pos0 += 4u) {
                const uint n_rows = min(4u, last - pos0 + 1u);
                for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                    const uint r = off >> 7;
                    const uint c = off & 127u;
                    const uint logical = pos0 + r - first_raw_pos;
                    const uint row = (args.raw_start + logical) % args.raw_cap;
                    device const float4 *src = (device const float4 *)(raw_kv +
                        (uint64_t)row * args.raw_row_stride);
                    kv_shared[off] = src[c];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint r = 0; r < n_rows; r++) {
                    dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                         r,
                                                         q0, q1, q2, q3,
                                                         args.scale,
                                                         lane,
                                                         M, S,
                                                         o0, o1, o2, o3);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else {
            for (uint pos = first; pos <= last; pos++) {
                const uint logical = pos - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                if (tid < 128) kv_shared[tid] = src[tid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                                  q0, q1, q2, q3,
                                                  args.scale,
                                                  lane,
                                                  M, S,
                                                  o0, o1, o2, o3);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    if (rb4) {
        bool stop = false;
        for (uint i = 0; i < args.top_k && !stop; i += 4u) {
            uint rows[4];
            uint n_rows = 0;
            for (uint j = 0; j < 4u && i + j < args.top_k; j++) {
                const int32_t idx = row_topk[i + j];
                if (idx < 0) {
                    continue;
                }
                if ((uint)idx >= visible) {
                    stop = true;
                    break;
                }
                rows[n_rows++] = (uint)idx;
            }
            if (n_rows == 0) {
                continue;
            }
            for (uint r = 0; r < n_rows; r++) {
                dsv4_turbo_dequant_shared_row(args,
                                              comp_kv,
                                              rows[r],
                                              kv_shared + r * 128u,
                                              scratch,
                                              lane,
                                              sg,
                                              turbo4);
            }
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_f32_row_as_f16_at(kv_shared,
                                                     r,
                                                     q0, q1, q2, q3,
                                                     args.scale,
                                                     lane,
                                                     M, S,
                                                     o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint i = 0; i < args.top_k; i++) {
            const int32_t idx = row_topk[i];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                break;
            }
            dsv4_turbo_dequant_shared_row(args,
                                          comp_kv,
                                          (uint)idx,
                                          kv_shared,
                                          scratch,
                                          lane,
                                          sg,
                                          turbo4);
            dsv4_attend_shared_f32_row_as_f16(kv_shared,
                                              q0, q1, q2, q3,
                                              args.scale,
                                              lane,
                                              M, S,
                                              o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_turbo3(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_turbo_impl(args, q, raw_kv, comp_kv, topk,
                                            sinks, dst, kv_shared, tgpig,
                                            tid, lane, sg, false, false);
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_turbo4(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_turbo_impl(args, q, raw_kv, comp_kv, topk,
                                            sinks, dst, kv_shared, tgpig,
                                            tid, lane, sg, true, false);
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb4_turbo3(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_turbo_impl(args, q, raw_kv, comp_kv, topk,
                                            sinks, dst, kv_shared, tgpig,
                                            tid, lane, sg, false, true);
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb4_turbo4(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup float4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    dsv4_indexed_mixed_attention_turbo_impl(args, q, raw_kv, comp_kv, topk,
                                            sinks, dst, kv_shared, tgpig,
                                            tid, lane, sg, true, true);
}
