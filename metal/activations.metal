// Shared activation functions. This source is loaded before every kernel file.

// Keep finite activation behavior aligned with the stable CPU implementation.
// exp(-abs(x)) avoids overflow without a divergent branch on the sign.
static inline float ds4_sigmoid_stable(float x) {
    const float e = exp(-fabs(x));
    const float numer = x >= 0.0f ? 1.0f : e;
    return numer / (1.0f + e);
}

static inline float4 ds4_sigmoid_stable(float4 x) {
    const float4 e = exp(-fabs(x));
    const float4 numer = select(e, float4(1.0f), x >= 0.0f);
    return numer / (1.0f + e);
}

static inline float ds4_silu(float x) {
    // Metal fast-math may flush exp(x) before x can lift a subnormal tail
    // back into the normal range. Here 1 + exp(x) rounds to 1, so fold |x|
    // into the exponent and produce the final magnitude without a subnormal
    // intermediate. The precise calls also prevent unsafe fast-math rewrites.
    if (x < -87.0f) {
        return -precise::exp(x + precise::log(-x));
    }
    return x * ds4_sigmoid_stable(x);
}

static inline float4 ds4_silu(float4 x) {
    float4 result = x * ds4_sigmoid_stable(x);
    if (x.x < -87.0f) result.x = ds4_silu(x.x);
    if (x.y < -87.0f) result.y = ds4_silu(x.y);
    if (x.z < -87.0f) result.z = ds4_silu(x.z);
    if (x.w < -87.0f) result.w = ds4_silu(x.w);
    return result;
}
