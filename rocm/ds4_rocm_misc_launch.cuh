extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(a, n) || !cuda_tensor_has_f32(b, n)) return 0;
    if (n == 0u) return 1;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0) return 0;
    uint64_t x_bytes = 0, dir_bytes = 0;
    if (!cuda_u64_mul3_checked(width, rows, sizeof(float), &x_bytes) ||
        !cuda_u64_mul3_checked((uint64_t)layer + 1u, width, sizeof(float), &dir_bytes) ||
        x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;
    if (scale == 0.0f) return 1;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
