#version 460 core
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_shader_atomic_float : enable

/* =========================================================================
 * Common types and utilities shared across DS4 Vulkan compute shaders.
 * ========================================================================= */

/* Block types matching ggml/DS4 quantization formats */

/* Q8_0 block: 32 weights, each 1 byte (int8), one f16 scale per block */
struct block_q8_0 {
    float  d;        /* scale (f32 for simplicity, was f16) */
    int8_t qs[32];   /* quantized weights */
};

/* Q4_K block: 256 weights in groups of 32, 6-bit scales */
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

/* Q2_K block */
struct block_q2_K {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

/* IQ2_XXS block for DS4 routed experts */
struct block_iq2_xxs {
    uint16_t d;
    uint16_t qs[32];  /* 32 * 16 = 512 bits = 64 bytes for 256 quants */
};

/* D = super-block scale */
#define QK_K 256
#define QK8_0 32
#define QK4_K 256
#define QK2_K 256
#define QK_IQ2_XXS 256

/* Shared memory barrier */
#define WORKGROUP_BARRIER barrier()
#define SUBGROUP_BARRIER subgroupBarrier()

/* Half-precision helpers — use float for simplicity in shaders */

float f16_to_f32(uint16_t v) {
    uint sign  = (v >> 15) & 0x1;
    uint exp   = (v >> 10) & 0x1f;
    uint mant  = v & 0x3ff;
    float result;
    if (exp == 0) {
        result = (sign ? -1.0 : 1.0) * ldexp(float(mant), -24);
    } else if (exp == 31) {
        result = (mant == 0) ? (sign ? -1.0/0.0 : 1.0/0.0) : 0.0/0.0;
    } else {
        result = (sign ? -1.0 : 1.0) * ldexp(float(mant) + 1024.0, exp - 25);
    }
    return result;
}

/* Dequantize a single Q8_0 block to floats in shared memory.
 * x is the shared memory output (QK8_0 = 32 floats).
 * qs points to the Q8_0 block data (8 bytes d + 32 bytes qs = 40 bytes). */
void dequant_q8_0(float x[32], float d, int8_t qs[32]) {
    for (int i = 0; i < 32; i++) {
        x[i] = d * float(qs[i]);
    }
}

/* Dequantize IQ2_XXS block — simplified version using iq2_xxs tables.
 * For the full version we use the pre-computed iq2_xxs tables from
 * ds4_iq2_tables_cuda.inc embedded in the C++ backend. */
void dequant_iq2_xxs(float x[QK_IQ2_XXS], uint16_t d, uint16_t qs[32], uint16_t iq2_table[512]) {
    /* To be implemented via the full table lookup in C++; stub here. */
    for (int i = 0; i < QK_IQ2_XXS; i++) {
        x[i] = float(d) * 0.5; /* placeholder */
    }
}
