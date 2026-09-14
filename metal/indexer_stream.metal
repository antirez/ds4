// SPDX-License-Identifier: MIT
// Score/index records retain the exact score bits after each leaf. Later
// merges need no full score matrix and keep the legacy left-biased ties.
struct ds4_metal_indexer_stream_pair {
    float score;
    int32_t index;
};

struct ds4_metal_args_indexer_stream_leaf {
    uint32_t columns;
    uint32_t n_rows;
    uint32_t output_width;
    uint32_t leaf_length;
    uint32_t index_base;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

// ABI is ds4_indexer_topk_merge_args; score_stride is unused because every
// retained candidate carries its score. Widths are row strides in records.
struct ds4_metal_args_indexer_stream_merge {
    uint64_t score_stride;
    uint32_t n_rows;
    uint32_t input_width;
    uint32_t run_length;
    uint32_t output_width;
    uint32_t output_run_length;
    uint32_t pad;
};

kernel void kernel_indexer_stream_leaf_f32_i32(
        constant ds4_metal_args_indexer_stream_leaf & args,
        device const float *scores,
        device ds4_metal_indexer_stream_pair *dst,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int col = tpitg.x;
    const int ib = tgpig.x / args.n_rows;
    const int i00 = ib * ntg.x;
    const uint row = tgpig.x % args.n_rows;
    device const float *src0_row = scores + (uint64_t)row * args.columns;
    shmem_i32[col] = i00 + col;
    threadgroup float *shmem_f32 = (threadgroup float *)(shmem_i32 + ntg.x);
    if ((uint)(i00 + col) < args.columns) shmem_f32[col] = src0_row[i00 + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This is the unchanged descending argsort leaf network. The host fixes
    // ntg.x to the baseline's global leaf width, including short last chunks.
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if ((uint)shmem_i32[col] >= args.columns ||
                        ((uint)shmem_i32[ixj] < args.columns &&
                         shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00])) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if ((uint)shmem_i32[ixj] >= args.columns ||
                        ((uint)shmem_i32[col] < args.columns &&
                         shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00])) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const uint kept = min(args.leaf_length, args.columns - (uint)i00);
    if ((uint)col < kept) {
        const int32_t index = shmem_i32[col];
        const uint64_t offset = (uint64_t)row * args.output_width +
                                (uint64_t)ib * args.leaf_length + col;
        dst[offset].score = shmem_f32[index - i00];
        dst[offset].index = (int32_t)(args.index_base + (uint)index);
    }
}

template<bool final_output>
kernel void kernel_indexer_stream_merge(
        constant ds4_metal_args_indexer_stream_merge & args,
        device const ds4_metal_indexer_stream_pair *src,
        device char *dst_bytes,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint im = tgpig.x / args.n_rows;
    const uint row = tgpig.x % args.n_rows;
    const uint64_t start = (uint64_t)im * (2ull * args.run_length);
    if (start >= args.input_width) return;
    const uint len0 = min((uint64_t)args.run_length, (uint64_t)args.input_width - start);
    const uint len1 = start + len0 < args.input_width
        ? min((uint64_t)args.run_length, (uint64_t)args.input_width - start - len0) : 0u;
    const uint kept = min(args.output_run_length, len0 + len1);
    device const ds4_metal_indexer_stream_pair *left = src + (uint64_t)row * args.input_width + start;
    device const ds4_metal_indexer_stream_pair *right = left + len0;
    const uint64_t output_base = (uint64_t)row * args.output_width +
                                 (uint64_t)im * args.output_run_length;
    device ds4_metal_indexer_stream_pair *dst = (device ds4_metal_indexer_stream_pair *)dst_bytes;
    device int32_t *selected = (device int32_t *)dst_bytes;
    const uint chunk = (kept + ntg.x - 1u) / ntg.x;
    const uint k0 = tpitg.x * chunk;
    const uint k1 = min(k0 + chunk, kept);
    if (k0 >= kept) return;
    uint low = k0 > len1 ? k0 - len1 : 0u;
    uint high = min(k0, len0);
    while (low < high) {
        const uint mid = (low + high) >> 1u;
        if (left[mid].score >= right[k0 - mid - 1u].score) low = mid + 1u;
        else high = mid;
    }
    uint i = low, j = k0 - i;
    ds4_metal_indexer_stream_pair a = {0.0f, 0}, b = {0.0f, 0};
    if (i < len0) a = left[i];
    if (j < len1) b = right[j];
    for (uint k = k0; k < k1; ++k) {
        ds4_metal_indexer_stream_pair value;
        if (j >= len1 || (i < len0 && a.score >= b.score)) {
            value = a;
            if (++i < len0) a = left[i];
        } else {
            value = b;
            if (++j < len1) b = right[j];
        }
        if (final_output) selected[output_base + k] = value.index;
        else dst[output_base + k] = value;
    }
}

typedef void (indexer_stream_merge_t)(
        constant ds4_metal_args_indexer_stream_merge & args,
        device const ds4_metal_indexer_stream_pair *src, device char *dst_bytes,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]);
template [[host_name("kernel_indexer_stream_merge_f32_i32")]]
kernel indexer_stream_merge_t kernel_indexer_stream_merge<false>;
template [[host_name("kernel_indexer_stream_merge_final_f32_i32")]]
kernel indexer_stream_merge_t kernel_indexer_stream_merge<true>;
