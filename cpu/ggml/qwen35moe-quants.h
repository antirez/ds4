#ifndef DS4_QWEN35MOE_QUANTS_H
#define DS4_QWEN35MOE_QUANTS_H
#include <stddef.h>
#include <stdint.h>
void ds4_qwen35moe_init(void);
float ds4_qwen35moe_half_lookup(uint16_t half);
typedef void (*ds4_qwen35moe_dot_kernel)(int, float *, size_t, const void *, size_t,
                                    const void *, size_t, int);
ds4_qwen35moe_dot_kernel ds4_qwen35moe_select_dot(int type);
size_t ds4_qwen35moe_activation_bytes(int n);
void ds4_qwen35moe_quantize(int type, const float *x, void *out, int n);
void ds4_qwen35moe_dequant(int type, const void *x, float *out, int n);
float ds4_qwen35moe_dot(int type, int n, const void *x, const void *activation);
float ds4_qwen35moe_float_dot(int n, const float *x, const float *y);
void ds4_qwen35moe_silu(int n, float *y, const float *x);
double ds4_qwen35moe_softmax(int n, float *y, const float *x, float max);
void ds4_qwen35moe_gemm_f32(float *c, const float *a, const float *b, int m, int k, int n);
void ds4_qwen35moe_f16_to_f32(int n, const unsigned short *src, float *dst);
void ds4_qwen35moe_iqp_matbatch(int type, int n, const void *weights, size_t weight_stride,
                           const void *activations, size_t activation_stride,
                           float *out, size_t output_stride, int rows, int count);
void ds4_qwen35moe_q4k_matbatch(int n, const void *weights, size_t weight_stride,
                           const void *activations, size_t activation_stride,
                           float *out, size_t output_stride, int rows, int count);
void ds4_qwen35moe_q4k_gemv_dot(int n, float *out, size_t out_stride,
                           const void *weight, size_t weight_stride,
                           const void *activation, size_t activation_stride, int count);
float ds4_qwen35moe_sgemm_float_dot(int n, const float *weights, const float *activation);
void ds4_qwen35moe_sgemm_float_tile4x2(int n, const float *weights, size_t weight_stride,
                                   const float *activation, size_t activation_stride,
                                   float *out, size_t output_stride, int rows, int columns);
void ds4_qwen35moe_f16_scale(int n, unsigned short *accumulator, float scale);
void ds4_qwen35moe_f16_mad(int n, unsigned short *accumulator,
                      const unsigned short *value, float scale);
void ds4_qwen35moe_router_top8(const float probabilities[256], int chosen[8]);
#endif
