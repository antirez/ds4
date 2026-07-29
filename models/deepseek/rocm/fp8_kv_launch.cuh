extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (n_rot > head_dim || !cuda_tensor_has_elems2(x, n_tok, head_dim, sizeof(float))) return 0;
    if (n_tok == 0u || head_dim == 0u) return 1;
    const uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    const uint32_t groups = (n_nope + 63u) / 64u;
    fp8_kv_quantize_kernel<<<dim3(n_tok, groups), 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
