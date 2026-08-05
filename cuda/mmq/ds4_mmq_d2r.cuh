// SPDX-License-Identifier: MIT
// Internal launcher for the gated D2R Q2_K MoE down-GEMM path.

#pragma once

#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

bool ds4_mmq_q2_K_moe_d2r_available(int cc);
bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc);

size_t ds4_mmq_q2_K_moe_d2r_scratch_bytes(int64_t ncols_max, int n_experts);
size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts);
size_t ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
    int64_t ncols_max, int n_experts);

int ds4_mmq_q2_K_moe_d2r_launch(
    const void    * W_soa,
    int64_t         soa_blocks,
    const void    * q8,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out,
    int             M,
    int             K,
    int64_t         ne_get_rows,
    int             n_experts,
    void          * worklist_scratch,
    size_t          worklist_scratch_bytes,
    cudaStream_t    stream);

bool ds4_mmq_q8_0_dense_d2r_available(int cc);

// Dense Q8_0 D2R on the kind-5 aligned artifact (--repack-q8-aligned).
// q8 = block_q8_1_mmq D4 activation buffer ([k128][col], stride N cols,
// over-allocated >= 128 blocks past N*K/128 for the guarded last col tile).
int ds4_mmq_q8_0_dense_d2r_launch(
    const void   * W_aligned,
    const void   * q8,
    float        * out,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream);

int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
    const void    * gate_soa,
    const void    * up_soa,
    int64_t         soa_blocks,
    const void    * q8,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    float         * out_gate,
    float         * out_up,
    int             M,
    int             K,
    int64_t         ne_get_rows,
    int             n_experts,
    void          * worklist_scratch,
    size_t          worklist_scratch_bytes,
    cudaStream_t    stream);

// Complete target-prefill gate/up path: both IQ2_XXS projections share one
// activation tile, then sanitize + clamp + SwiGLU + routing weight are folded
// directly into the expert-major Q8_1 D2S6 input consumed by Q2_K down.
//
// flat-pool p5b: with ids_src != NULL, input_q8 is TOKEN-COMPACT (n_tokens
// rows quantized once, no gather) and each assignment column stages its
// token's row through ids_src[col]; with ids_src == NULL, input_q8 is the
// classic slot-gathered buffer (ne_get_rows rows) and n_tokens is ignored.
int ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
    const void    * gate_soa,
    const void    * up_soa,
    int64_t         soa_blocks,
    const void    * input_q8,
    const int32_t * ids_src,
    int             n_tokens,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const float   * router_weights,
    void          * down_q8,
    int             M,
    int             K,
    int64_t         ne_get_rows,
    int             n_experts,
    float           clamp,
    void          * worklist_scratch,
    size_t          worklist_scratch_bytes,
    cudaStream_t    stream);
