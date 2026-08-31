#include "ds4_qwen4.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

bool ds4_qwen4_mtp_plan_greedy(
        const int32_t *drafts,
        uint32_t proposed,
        int32_t target_after_first,
        const int32_t *verifier_tops,
        uint32_t verifier_rows,
        ds4_qwen4_mtp_accept_plan *plan) {
    if (!plan) return false;
    memset(plan, 0, sizeof(*plan));
    if (!drafts || proposed == 0u ||
        proposed > DS4_QWEN4_MTP_MAX_DRAFTS ||
        verifier_rows != proposed ||
        (proposed > 1u && !verifier_tops)) return false;

    plan->proposed = proposed;
    plan->rejected_at = 0u;
    if (drafts[0] != target_after_first) return true;

    plan->first_draft_matches = true;
    plan->accepted = 1u;
    while (plan->accepted < proposed) {
        const uint32_t next = plan->accepted;
        if (verifier_tops[next - 1u] != drafts[next]) {
            plan->rejected_at = next;
            return true;
        }
        plan->accepted++;
    }
    plan->rejected_at = proposed;
    plan->full_accept = true;
    return true;
}

static void qwen4_error(char *out, size_t cap, const char *fmt, ...) {
    if (!out || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(out, cap, fmt, ap);
    va_end(ap);
}

bool ds4_qwen4_pack_profile_from_version(uint32_t version,
                                         ds4_qwen4_pack_profile *out) {
    ds4_qwen4_pack_profile profile = DS4_QWEN4_PACK_PROFILE_INVALID;
    if (version == DS4_QWEN4_PACK_VERSION_Q4) {
        profile = DS4_QWEN4_PACK_PROFILE_Q4_K;
    } else if (version == DS4_QWEN4_PACK_VERSION_Q2) {
        profile = DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K;
    }
    if (out) *out = profile;
    return profile != DS4_QWEN4_PACK_PROFILE_INVALID;
}

const char *ds4_qwen4_pack_profile_name(ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K: return "q4_k";
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K:
            return "iq2_xxs_gate_up_q2_k_down";
        default: return "invalid";
    }
}

const char *ds4_qwen4_pack_manifest_file(ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K:
            return DS4_QWEN4_PACK_MANIFEST_FILE;
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K:
            return DS4_QWEN4_PACK_Q2_MANIFEST_FILE;
        default: return NULL;
    }
}

const char *ds4_qwen4_pack_routed_name(ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K: return "Q4_K";
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K: return "mixed-q2";
        default: return NULL;
    }
}

const char *ds4_qwen4_pack_routed_gate_up_name(
        ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K: return "Q4_K";
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K: return "IQ2_XXS";
        default: return NULL;
    }
}

const char *ds4_qwen4_pack_routed_down_name(
        ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K: return "Q4_K";
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K: return "Q2_K";
        default: return NULL;
    }
}

bool ds4_qwen4_pack_routed_metadata_valid(
        ds4_qwen4_pack_profile profile,
        const char *routed,
        const char *routed_gate_up,
        const char *routed_down,
        char *error,
        size_t error_cap) {
    const char *expected_routed = ds4_qwen4_pack_routed_name(profile);
    const char *expected_gate_up =
        ds4_qwen4_pack_routed_gate_up_name(profile);
    const char *expected_down = ds4_qwen4_pack_routed_down_name(profile);
    if (!expected_routed || !expected_gate_up || !expected_down) {
        qwen4_error(error, error_cap, "unsupported Qwen quantization profile");
        return false;
    }
    if (!routed || strcmp(routed, expected_routed) != 0) {
        qwen4_error(error, error_cap,
                    "Qwen %s metadata routed=%s; expected %s",
                    ds4_qwen4_pack_profile_name(profile),
                    routed ? routed : "(missing)", expected_routed);
        return false;
    }
    if (profile == DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K) {
        if (!routed_gate_up || strcmp(routed_gate_up, expected_gate_up) != 0 ||
            !routed_down || strcmp(routed_down, expected_down) != 0) {
            qwen4_error(error, error_cap,
                        "Qwen %s metadata gate_up/down=%s/%s; expected %s/%s",
                        ds4_qwen4_pack_profile_name(profile),
                        routed_gate_up ? routed_gate_up : "(missing)",
                        routed_down ? routed_down : "(missing)",
                        expected_gate_up, expected_down);
            return false;
        }
    } else {
        if ((routed_gate_up && strcmp(routed_gate_up, expected_gate_up) != 0) ||
            (routed_down && strcmp(routed_down, expected_down) != 0)) {
            qwen4_error(error, error_cap,
                        "Qwen q4_k metadata contains a conflicting routed "
                        "gate_up/down override");
            return false;
        }
    }
    return true;
}

bool ds4_qwen4_pack_routed_qtypes_valid(
        ds4_qwen4_pack_profile profile,
        uint32_t gate_qtype,
        uint32_t up_qtype,
        uint32_t down_qtype,
        char *error,
        size_t error_cap) {
    uint32_t expected_gate = UINT32_MAX;
    uint32_t expected_up = UINT32_MAX;
    uint32_t expected_down = UINT32_MAX;
    if (profile == DS4_QWEN4_PACK_PROFILE_Q4_K) {
        expected_gate = DS4_QWEN4_QTYPE_Q4_K;
        expected_up = DS4_QWEN4_QTYPE_Q4_K;
        expected_down = DS4_QWEN4_QTYPE_Q4_K;
    } else if (profile == DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K) {
        expected_gate = DS4_QWEN4_QTYPE_IQ2_XXS;
        expected_up = DS4_QWEN4_QTYPE_IQ2_XXS;
        expected_down = DS4_QWEN4_QTYPE_Q2_K;
    } else {
        qwen4_error(error, error_cap, "unsupported Qwen quantization profile");
        return false;
    }
    if (gate_qtype != expected_gate || up_qtype != expected_up ||
        down_qtype != expected_down) {
        qwen4_error(error, error_cap,
                    "Qwen %s routed tensors have qtypes %u/%u/%u; "
                    "expected %u/%u/%u",
                    ds4_qwen4_pack_profile_name(profile),
                    gate_qtype, up_qtype, down_qtype,
                    expected_gate, expected_up, expected_down);
        return false;
    }
    return true;
}

uint64_t ds4_qwen4_pack_fast_required_mask(
        ds4_qwen4_pack_profile profile) {
    switch (profile) {
        case DS4_QWEN4_PACK_PROFILE_Q4_K:
            return DS4_QWEN4_FAST_Q4_METAL_REQUIRED_MASK |
                   DS4_QWEN4_FAST_PLE_STAGER;
        case DS4_QWEN4_PACK_PROFILE_IQ2_XXS_Q2_K:
            return DS4_QWEN4_FAST_Q2_METAL_REQUIRED_MASK |
                   DS4_QWEN4_FAST_PLE_STAGER;
        default:
            return 0u;
    }
}

typedef struct {
    uint32_t state[8];
    uint64_t bits;
    uint8_t block[64];
    size_t used;
} qwen4_sha256;

typedef struct {
    bool nocache_requested;
    bool nocache_enabled;
    bool nocache_cleared;
} qwen4_sha256_io_status;

static uint32_t qwen4_rotr32(uint32_t value, uint32_t shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void qwen4_sha256_transform(qwen4_sha256 *ctx, const uint8_t block[64]) {
    static const uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t words[64];
    for (uint32_t i = 0; i < 16; i++) {
        words[i] = ((uint32_t)block[i * 4u] << 24) |
                   ((uint32_t)block[i * 4u + 1u] << 16) |
                   ((uint32_t)block[i * 4u + 2u] << 8) |
                   (uint32_t)block[i * 4u + 3u];
    }
    for (uint32_t i = 16; i < 64; i++) {
        const uint32_t s0 = qwen4_rotr32(words[i - 15u], 7) ^
                            qwen4_rotr32(words[i - 15u], 18) ^
                            (words[i - 15u] >> 3);
        const uint32_t s1 = qwen4_rotr32(words[i - 2u], 17) ^
                            qwen4_rotr32(words[i - 2u], 19) ^
                            (words[i - 2u] >> 10);
        words[i] = words[i - 16u] + s0 + words[i - 7u] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (uint32_t i = 0; i < 64; i++) {
        const uint32_t s1 = qwen4_rotr32(e, 6) ^ qwen4_rotr32(e, 11) ^ qwen4_rotr32(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + choose + constants[i] + words[i];
        const uint32_t s0 = qwen4_rotr32(a, 2) ^ qwen4_rotr32(a, 13) ^ qwen4_rotr32(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void qwen4_sha256_init(qwen4_sha256 *ctx) {
    static const uint32_t initial[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u,
    };
    memset(ctx, 0, sizeof(*ctx));
    memcpy(ctx->state, initial, sizeof(initial));
}

static void qwen4_sha256_update(qwen4_sha256 *ctx, const void *data, size_t size) {
    const uint8_t *input = data;
    ctx->bits += (uint64_t)size * 8u;
    while (size) {
        const size_t amount = size < 64u - ctx->used ? size : 64u - ctx->used;
        memcpy(ctx->block + ctx->used, input, amount);
        ctx->used += amount;
        input += amount;
        size -= amount;
        if (ctx->used == 64u) {
            qwen4_sha256_transform(ctx, ctx->block);
            ctx->used = 0;
        }
    }
}

static void qwen4_sha256_final(qwen4_sha256 *ctx, uint8_t digest[32]) {
    const uint64_t message_bits = ctx->bits;
    ctx->block[ctx->used++] = 0x80u;
    if (ctx->used > 56u) {
        memset(ctx->block + ctx->used, 0, 64u - ctx->used);
        qwen4_sha256_transform(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56u - ctx->used);
    for (uint32_t i = 0; i < 8; i++)
        ctx->block[63u - i] = (uint8_t)(message_bits >> (i * 8u));
    qwen4_sha256_transform(ctx, ctx->block);
    for (uint32_t i = 0; i < 8; i++) {
        digest[i * 4u] = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4u + 1u] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4u + 2u] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4u + 3u] = (uint8_t)ctx->state[i];
    }
}

static void qwen4_sha256_digest_hex(const uint8_t digest[32],
                                    char hex_out[65]) {
    static const char digits[] = "0123456789abcdef";
    for (uint32_t i = 0; i < 32; i++) {
        hex_out[i * 2u] = digits[digest[i] >> 4];
        hex_out[i * 2u + 1u] = digits[digest[i] & 15u];
    }
    hex_out[64] = '\0';
}

static void qwen4_sha256_buffer_hex(const void *data,
                                    size_t size,
                                    char hex_out[65]) {
    qwen4_sha256 sha;
    uint8_t digest[32];
    qwen4_sha256_init(&sha);
    qwen4_sha256_update(&sha, data, size);
    qwen4_sha256_final(&sha, digest);
    qwen4_sha256_digest_hex(digest, hex_out);
}

static int qwen4_sha256_fd(int fd,
                           const char *label,
                           bool uncached,
                           char hex_out[65],
                           uint64_t *bytes_out,
                           qwen4_sha256_io_status *io_status,
                           char *error,
                           size_t error_cap) {
    if (io_status) {
        memset(io_status, 0, sizeof(*io_status));
        io_status->nocache_requested = uncached;
    }
    if (fd < 0 || !hex_out) {
        qwen4_error(error, error_cap, "missing SHA-256 descriptor/output");
        return 1;
    }
#if defined(__APPLE__) && defined(F_NOCACHE)
    const bool cache_disabled = uncached && fcntl(fd, F_NOCACHE, 1) == 0;
    if (io_status) io_status->nocache_enabled = cache_disabled;
#else
    (void)uncached;
#endif
    qwen4_sha256 sha;
    qwen4_sha256_init(&sha);
    uint8_t *buffer = malloc(8u * 1024u * 1024u);
    uint64_t total = 0;
    if (!buffer) {
        qwen4_error(error, error_cap, "out of memory hashing %s", label);
#if defined(__APPLE__) && defined(F_NOCACHE)
        if (cache_disabled) {
            const bool cleared = fcntl(fd, F_NOCACHE, 0) == 0;
            if (io_status) io_status->nocache_cleared = cleared;
        }
#endif
        return 1;
    }
    int failed = 0;
    while (1) {
        ssize_t got = pread(fd, buffer, 8u * 1024u * 1024u, (off_t)total);
        if (got == 0) break;
        if (got < 0) {
            if (errno == EINTR) continue;
            qwen4_error(error, error_cap, "cannot read %s: %s",
                        label, strerror(errno));
            failed = 1;
            break;
        }
        qwen4_sha256_update(&sha, buffer, (size_t)got);
        total += (uint64_t)got;
    }
    free(buffer);
#if defined(__APPLE__) && defined(F_NOCACHE)
    if (cache_disabled) {
        const bool cleared = fcntl(fd, F_NOCACHE, 0) == 0;
        if (io_status) io_status->nocache_cleared = cleared;
    }
#endif
    if (failed) return 1;
    uint8_t digest[32];
    qwen4_sha256_final(&sha, digest);
    qwen4_sha256_digest_hex(digest, hex_out);
    if (bytes_out) *bytes_out = total;
    return 0;
}

int ds4_qwen4_sha256_file(const char *path,
                          char hex_out[65],
                          uint64_t *bytes_out,
                          char *error,
                          size_t error_cap) {
    if (!path || !hex_out) {
        qwen4_error(error, error_cap, "missing SHA-256 path/output");
        return 1;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        qwen4_error(error, error_cap, "cannot open %s: %s", path, strerror(errno));
        return 1;
    }
    const int result = qwen4_sha256_fd(
        fd, path, false, hex_out, bytes_out, NULL, error, error_cap);
    close(fd);
    return result;
}

bool ds4_qwen4_parse_prefill_mode(const char *value,
                                  ds4_qwen4_prefill_mode *out,
                                  char *error,
                                  size_t error_cap) {
    if (!value || !out) {
        qwen4_error(error, error_cap, "missing prefill mode");
        return false;
    }
    if (!strcmp(value, "auto")) {
        *out = DS4_QWEN4_PREFILL_AUTO;
        return true;
    }
    if (!strcmp(value, "2048")) {
        *out = DS4_QWEN4_PREFILL_2048;
        return true;
    }
    if (!strcmp(value, "4096")) {
        *out = DS4_QWEN4_PREFILL_4096;
        return true;
    }
    if (!strcmp(value, "8192")) {
        *out = DS4_QWEN4_PREFILL_8192;
        return true;
    }
    qwen4_error(error, error_cap,
                "expected auto, 2048, 4096, or 8192; got '%s'", value);
    return false;
}

uint32_t ds4_qwen4_select_prefill_chunk(ds4_qwen4_prefill_mode mode,
                                        bool fast_path_complete,
                                        uint32_t admitted_cap,
                                        uint32_t uncached_tokens,
                                        uint32_t cached_tokens,
                                        const char **reason) {
    uint32_t selected = (uint32_t)mode;
    if (reason) *reason = "explicit";
    if (mode == DS4_QWEN4_PREFILL_AUTO) {
        selected = 8192u;
        if (!fast_path_complete) {
            if (reason) *reason = "mandatory Qwen fast path incomplete";
            selected = 2048u;
        } else if (cached_tokens != 0u) {
            if (reason) *reason = "resuming prefix cache";
            selected = 2048u;
        } else if (uncached_tokens < 8192u) {
            if (reason) *reason = "uncached suffix shorter than 8192";
            selected = 2048u;
        } else if (reason) {
            *reason = "cold Qwen native path";
        }
    }
    if (selected > admitted_cap) {
        if (reason) *reason = "pre-admitted scratch cap";
        selected = admitted_cap;
    }
    return selected;
}

static uint64_t qwen4_sat_add(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static uint64_t qwen4_sat_mul(uint64_t a, uint64_t b) {
    return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

uint64_t ds4_qwen4_prefill_scratch_bytes(uint32_t chunk) {
    if (chunk != 2048u && chunk != 4096u && chunk != 8192u) return UINT64_MAX;

    /* These are live-value upper bounds, not total work.  Storage is reused at
     * layer boundaries, so the estimate takes the largest attention/GDN
     * island and the largest MoE island rather than summing 48 layers. */
    const uint64_t hc_f32 =
        3ull * DS4_QWEN4_HC_COUNT * DS4_QWEN4_HIDDEN;
    const uint64_t projection_f32 =
        10240ull + 6144ull + 6144ull + 2ull * DS4_QWEN4_HIDDEN;
    const uint64_t moe_f32 =
        DS4_QWEN4_EXPERTS +
        3ull * DS4_QWEN4_EXPERTS_USED * DS4_QWEN4_EXPERT_FF +
        2ull * DS4_QWEN4_HIDDEN;
    const uint64_t attention_f32 =
        2ull * DS4_QWEN4_ATTN_HEADS * DS4_QWEN4_HEAD_DIM +
        2ull * DS4_QWEN4_KV_HEADS * DS4_QWEN4_HEAD_DIM +
        (DS4_QWEN4_INDEXER_HEADS + DS4_QWEN4_INDEXER_KV_HEADS) *
            DS4_QWEN4_INDEXER_HEAD_DIM;
    uint64_t per_token = hc_f32;
    per_token = qwen4_sat_add(per_token, projection_f32);
    per_token = qwen4_sat_add(per_token, moe_f32);
    per_token = qwen4_sat_add(per_token, attention_f32);
    uint64_t bytes = qwen4_sat_mul(
        qwen4_sat_mul((uint64_t)chunk, per_token), sizeof(float));

    const uint64_t recurrence =
        36ull * DS4_QWEN4_LINEAR_VALUE_HEADS *
        DS4_QWEN4_LINEAR_HEAD_DIM * DS4_QWEN4_LINEAR_HEAD_DIM *
        sizeof(float);
    const uint64_t qsa_tile =
        (uint64_t)DS4_QWEN4_QSA_MICROTILE * 1024ull * sizeof(float);
    const uint64_t qsa_heaps =
        (uint64_t)DS4_QWEN4_QSA_MICROTILE * 512ull *
        (sizeof(float) + sizeof(uint32_t));
    const uint64_t ple_cpu_staging =
        2ull * DS4_QWEN4_QSA_MICROTILE * DS4_QWEN4_NGRAM_HEADS *
        DS4_QWEN4_PLE_ROW_DIM * sizeof(uint16_t);
    const uint64_t ple_gpu_staging =
        (uint64_t)chunk * DS4_QWEN4_PLE_EMBED_DIM * sizeof(uint16_t);
    bytes = qwen4_sat_add(bytes, recurrence);
    bytes = qwen4_sat_add(bytes, qsa_tile);
    bytes = qwen4_sat_add(bytes, qsa_heaps);
    bytes = qwen4_sat_add(bytes, ple_cpu_staging);
    bytes = qwen4_sat_add(bytes, ple_gpu_staging);
    bytes = qwen4_sat_add(bytes, 256ull * 1024ull * 1024ull);
    return bytes;
}

static uint64_t qwen4_prefill_scratch_with_extra(
        uint32_t chunk,
        uint64_t extra_bytes_per_token) {
    return qwen4_sat_add(
        ds4_qwen4_prefill_scratch_bytes(chunk),
        qwen4_sat_mul(chunk, extra_bytes_per_token));
}

bool ds4_qwen4_admit_prefill_with_extra(
                             ds4_qwen4_prefill_mode mode,
                             bool fast_path_complete,
                             const char *missing_fast_path,
                             uint64_t scratch_available,
                             uint64_t extra_bytes_per_token,
                             ds4_qwen4_prefill_admission *out,
                             char *error,
                             size_t error_cap) {
    if (out) memset(out, 0, sizeof(*out));
    if (!out || (mode != DS4_QWEN4_PREFILL_AUTO &&
                 mode != DS4_QWEN4_PREFILL_2048 &&
                 mode != DS4_QWEN4_PREFILL_4096 &&
                 mode != DS4_QWEN4_PREFILL_8192)) {
        qwen4_error(error, error_cap, "invalid Qwen prefill admission request");
        return false;
    }

    const uint64_t need_2k = qwen4_prefill_scratch_with_extra(
        2048u, extra_bytes_per_token);
    const uint64_t need_8k = qwen4_prefill_scratch_with_extra(
        8192u, extra_bytes_per_token);
    if (scratch_available < need_2k) {
        qwen4_error(error, error_cap,
                    "Qwen prefill requires at least 2048-token scratch "
                    "(%.2f GiB), but only %.2f GiB is admitted",
                    (double)need_2k / 1073741824.0,
                    (double)scratch_available / 1073741824.0);
        return false;
    }

    if (mode == DS4_QWEN4_PREFILL_AUTO) {
        if (fast_path_complete && scratch_available >= need_8k) {
            out->admitted_cap = 8192u;
            out->scratch_required = need_8k;
            out->reason = "complete fast path and 8K scratch admitted";
        } else {
            out->admitted_cap = 2048u;
            out->scratch_required = need_2k;
            out->reason = fast_path_complete
                ? "8K scratch unavailable; admitted 2K"
                : "mandatory fast path incomplete; admitted 2K";
        }
        return true;
    }

    const uint32_t requested = (uint32_t)mode;
    if (requested == 8192u && !fast_path_complete) {
        qwen4_error(error, error_cap,
                    "requested Qwen --prefill-chunk 8192 requires the "
                    "complete native fast path: %s",
                    missing_fast_path && missing_fast_path[0]
                        ? missing_fast_path : "mandatory capability missing");
        return false;
    }
    const uint64_t required = qwen4_prefill_scratch_with_extra(
        requested, extra_bytes_per_token);
    if (scratch_available < required) {
        qwen4_error(error, error_cap,
                    "requested Qwen --prefill-chunk %u requires %.2f GiB "
                    "scratch, but only %.2f GiB is admitted",
                    requested,
                    (double)required / 1073741824.0,
                    (double)scratch_available / 1073741824.0);
        return false;
    }
    out->admitted_cap = requested;
    out->scratch_required = required;
    out->reason = "explicit chunk admitted";
    return true;
}

bool ds4_qwen4_admit_prefill(ds4_qwen4_prefill_mode mode,
                             bool fast_path_complete,
                             const char *missing_fast_path,
                             uint64_t scratch_available,
                             ds4_qwen4_prefill_admission *out,
                             char *error,
                             size_t error_cap) {
    return ds4_qwen4_admit_prefill_with_extra(
        mode, fast_path_complete, missing_fast_path, scratch_available, 0u,
        out, error, error_cap);
}

bool ds4_qwen4_build_mrope_positions(
        int32_t                    *positions,
        uint32_t                    capacity,
        uint32_t                    prompt_len,
        const ds4_qwen4_mrope_span *images,
        size_t                      image_count,
        int32_t                    *delta_out) {
    if (!positions || capacity == 0u || prompt_len > capacity ||
        (image_count != 0u && !images)) return false;

#define QWEN4_MROPE_SET(token_, t_, h_, w_) do { \
        positions[(token_)] = (int32_t)(t_); \
        positions[(uint64_t)capacity + (token_)] = (int32_t)(h_); \
        positions[2ull * capacity + (token_)] = (int32_t)(w_); \
    } while (0)

    uint32_t source = 0u;
    uint64_t next_position = 0u;
    for (size_t image = 0; image < image_count; image++) {
        const ds4_qwen4_mrope_span *span = &images[image];
        const uint32_t grid_h = span->grid_height;
        const uint32_t grid_w = span->grid_width;
        if (grid_h == 0u || grid_w == 0u ||
            grid_h % DS4_QWEN4_VISION_MERGE != 0u ||
            grid_w % DS4_QWEN4_VISION_MERGE != 0u) return false;
        const uint32_t llm_h = grid_h / DS4_QWEN4_VISION_MERGE;
        const uint32_t llm_w = grid_w / DS4_QWEN4_VISION_MERGE;
        const uint64_t expected = (uint64_t)llm_h * llm_w;
        const uint64_t image_end =
            (uint64_t)span->token_start + span->token_count;
        if (span->token_start < source || image_end > prompt_len ||
            expected != span->token_count) return false;

        for (uint32_t token = source; token < span->token_start; token++) {
            if (next_position > INT32_MAX) return false;
            QWEN4_MROPE_SET(token, next_position, next_position,
                            next_position);
            next_position++;
        }
        if (next_position > INT32_MAX ||
            next_position + llm_h - 1u > INT32_MAX ||
            next_position + llm_w - 1u > INT32_MAX) return false;
        for (uint32_t token = 0u; token < span->token_count; token++) {
            const uint32_t h = token / llm_w;
            const uint32_t w = token % llm_w;
            QWEN4_MROPE_SET(span->token_start + token,
                            next_position, next_position + h,
                            next_position + w);
        }
        const uint32_t contracted = llm_h > llm_w ? llm_h : llm_w;
        next_position += contracted > 0u ? contracted : 1u;
        source = (uint32_t)image_end;
    }
    for (uint32_t token = source; token < prompt_len; token++) {
        if (next_position > INT32_MAX) return false;
        QWEN4_MROPE_SET(token, next_position, next_position, next_position);
        next_position++;
    }

    const int64_t delta64 = (int64_t)next_position - (int64_t)prompt_len;
    if (delta64 < INT32_MIN || delta64 > INT32_MAX) return false;
    for (uint32_t token = prompt_len; token < capacity; token++) {
        const int64_t value = (int64_t)token + delta64;
        if (value < 0 || value > INT32_MAX) return false;
        QWEN4_MROPE_SET(token, value, value, value);
    }
#undef QWEN4_MROPE_SET
    if (delta_out) *delta_out = (int32_t)delta64;
    return true;
}

static uint64_t splitmix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static bool is_prime(uint64_t value) {
    if (value < 2) return false;
    if ((value & 1u) == 0) return value == 2;
    for (uint64_t d = 3; d <= value / d; d += 2) {
        if (value % d == 0) return false;
    }
    return true;
}

static uint64_t nth_prime_after(uint64_t start, uint32_t count) {
    uint64_t value = start;
    for (uint32_t i = 0; i < count; i++) {
        do value++; while (!is_prime(value));
    }
    return value;
}

bool ds4_qwen4_ngram_hash_init(ds4_qwen4_ngram_hash *hash,
                               uint32_t unigram_vocab,
                               uint32_t ngram_size,
                               uint32_t heads_per_ngram,
                               uint64_t vocab_base,
                               uint64_t divisor,
                               uint64_t seed,
                               uint32_t ple_layer_index,
                               uint32_t eos) {
    if (!hash || unigram_vocab == 0 || ngram_size < 2 ||
        ngram_size > DS4_QWEN4_NGRAM_MAX_SIZE ||
        heads_per_ngram == 0 || divisor == 0 || vocab_base < 2) {
        return false;
    }
    const uint32_t n_heads = (ngram_size - 1u) * heads_per_ngram;
    if (n_heads > DS4_QWEN4_NGRAM_MAX_HEADS) return false;

    memset(hash, 0, sizeof(*hash));
    hash->ngram_size = ngram_size;
    hash->heads_per_ngram = heads_per_ngram;
    hash->n_heads = n_heads;
    hash->eos = eos;

    const uint64_t max_long = INT64_MAX;
    uint64_t half_bound = (max_long / unigram_vocab) / 2u;
    if (half_bound == 0) half_bound = 1;
    const uint64_t base_seed = seed + UINT64_C(10007) * ple_layer_index;
    for (uint32_t i = 0; i < ngram_size; i++) {
        const uint64_t value =
            base_seed + UINT64_C(0x9e3779b97f4a7c15) * (i + 1u);
        hash->multipliers[i] =
            (int64_t)(2u * (splitmix64(value) % half_bound) + 1u);
    }

    uint64_t total = 0;
    for (uint32_t i = 0; i < n_heads; i++) {
        const uint32_t global = ple_layer_index * n_heads + i;
        const uint64_t size = nth_prime_after(vocab_base - 1u, global + 1u);
        if (size > INT64_MAX || total > INT64_MAX - size) return false;
        hash->vocab[i] = (int64_t)size;
        hash->offsets[i] = (int64_t)total;
        total += size;
    }
    if (total > UINT64_MAX - (divisor - 1u)) return false;
    hash->total_rows = ((total + divisor - 1u) / divisor) * divisor;
    return true;
}

static uint32_t token_at(const uint32_t *previous,
                         size_t previous_count,
                         const uint32_t *tokens,
                         size_t index) {
    return index < previous_count ? previous[index] : tokens[index - previous_count];
}

static int64_t positive_mod_i64(int64_t value, int64_t divisor) {
    int64_t result = value % divisor;
    return result < 0 ? result + divisor : result;
}

bool ds4_qwen4_ngram_row_ids(const ds4_qwen4_ngram_hash *hash,
                             const uint32_t *previous,
                             size_t previous_count,
                             const uint32_t *tokens,
                             size_t token_count,
                             int64_t *rows,
                             size_t row_count) {
    if (!hash || !previous || !tokens || !rows ||
        previous_count != hash->ngram_size - 1u ||
        token_count > SIZE_MAX / hash->n_heads ||
        row_count < token_count * hash->n_heads) {
        return false;
    }

    int64_t last_eos = -1;
    const size_t total = previous_count + token_count;
    for (size_t t = 0; t < total; t++) {
        const uint32_t token = token_at(previous, previous_count, tokens, t);
        if (t >= previous_count) {
            const int64_t segment_pos = (int64_t)t - (last_eos + 1);
            uint64_t mixed = (uint64_t)(int64_t)token *
                             (uint64_t)hash->multipliers[0];
            uint32_t position = 1;
            for (uint32_t n = 2; n <= hash->ngram_size; n++) {
                while (position < n) {
                    const uint32_t shifted =
                        segment_pos >= (int64_t)position && t >= position ?
                        token_at(previous, previous_count, tokens, t - position) :
                        hash->eos;
                    mixed ^= (uint64_t)(int64_t)shifted *
                             (uint64_t)hash->multipliers[position];
                    position++;
                }
                const uint32_t first = (n - 2u) * hash->heads_per_ngram;
                for (uint32_t h = first;
                     h < first + hash->heads_per_ngram; h++) {
                    const int64_t id =
                        positive_mod_i64((int64_t)mixed, hash->vocab[h]) +
                        hash->offsets[h];
                    if (id < 0 || (uint64_t)id >= hash->total_rows) return false;
                    rows[(t - previous_count) * hash->n_heads + h] = id;
                }
            }
        }
        if (token == hash->eos) last_eos = (int64_t)t;
    }
    return true;
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static uint64_t load_u64_le(const uint8_t *p) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)p[i] << (8u * i);
    return value;
}

static uint32_t load_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static float f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t)(value & UINT16_C(0x8000)) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t fraction = value & 0x3ffu;
    uint32_t bits = sign;
    if (exponent == 0u) {
        if (fraction != 0u) {
            uint32_t shift = 0u;
            while ((fraction & 0x400u) == 0u) {
                fraction <<= 1;
                shift++;
            }
            fraction &= 0x3ffu;
            bits |= (UINT32_C(127) - 14u - shift) << 23;
            bits |= fraction << 13;
        }
    } else if (exponent == 0x1fu) {
        bits |= UINT32_C(0x7f800000) | (fraction << 13);
    } else {
        bits |= (exponent + UINT32_C(112)) << 23;
        bits |= fraction << 13;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounded = bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static const char *json_key(const char *json, const char *key) {
    char pattern[96];
    const int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(pattern)) return NULL;
    const char *search = json;
    while (search && *search) {
        const char *match = strstr(search, pattern);
        if (!match) return NULL;
        const char *p = match + (size_t)n;
        while (isspace((unsigned char)*p)) p++;
        if (*p == ':') {
            p++;
            while (isspace((unsigned char)*p)) p++;
            return p;
        }
        search = match + 1u;
    }
    return NULL;
}

static bool json_u64(const char *json, const char *key, uint64_t *out) {
    const char *p = json_key(json, key);
    if (!p) return false;
    if (*p == '"') p++;
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(p, &end, 10);
    if (end == p || errno != 0) return false;
    *out = (uint64_t)value;
    return true;
}

static const char *json_object(const char *json, const char *key) {
    const char *p = json_key(json, key);
    return p && *p == '{' ? p : NULL;
}

static bool json_string_value(const char *json, const char *key,
                              char *out, size_t out_cap) {
    const char *p = json_key(json, key);
    if (!p || *p++ != '"' || !out || out_cap == 0) return false;
    size_t used = 0;
    while (*p && *p != '"') {
        unsigned char value = (unsigned char)*p++;
        if (value == '\\') {
            const unsigned char escaped = (unsigned char)*p++;
            if (escaped == '"' || escaped == '\\' || escaped == '/') {
                value = escaped;
            } else {
                return false;
            }
        }
        if (value < 0x20u || used + 1u >= out_cap) return false;
        out[used++] = (char)value;
    }
    if (*p != '"') return false;
    out[used] = '\0';
    return true;
}

static const char *qwen4_json_skip_space(const char *p) {
    while (p && isspace((unsigned char)*p)) p++;
    return p;
}

static bool qwen4_json_hex_digit(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

static const char *qwen4_json_string_end(const char *start) {
    if (!start || *start != '"') return NULL;
    for (const char *p = start + 1u; *p; p++) {
        const unsigned char c = (unsigned char)*p;
        if (c == '"') return p + 1u;
        if (c < 0x20u) return NULL;
        if (c != '\\') continue;
        const unsigned char escaped = (unsigned char)*++p;
        if (!escaped) return NULL;
        if (escaped == '"' || escaped == '\\' || escaped == '/' ||
            escaped == 'b' || escaped == 'f' || escaped == 'n' ||
            escaped == 'r' || escaped == 't') continue;
        if (escaped != 'u') return NULL;
        for (uint32_t i = 0; i < 4u; i++) {
            if (!p[1] || !qwen4_json_hex_digit((unsigned char)p[1]))
                return NULL;
            p++;
        }
    }
    return NULL;
}

static const char *qwen4_json_number_end(const char *start) {
    const char *p = start;
    if (*p == '-') p++;
    if (*p == '0') {
        p++;
    } else {
        if (*p < '1' || *p > '9') return NULL;
        do p++; while (*p >= '0' && *p <= '9');
    }
    if (*p == '.') {
        p++;
        if (*p < '0' || *p > '9') return NULL;
        do p++; while (*p >= '0' && *p <= '9');
    }
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (*p < '0' || *p > '9') return NULL;
        do p++; while (*p >= '0' && *p <= '9');
    }
    return p;
}

static const char *qwen4_json_value_end(const char *start, uint32_t depth);

static const char *qwen4_json_compound_end(const char *start,
                                           uint32_t depth) {
    if (!start || depth >= 64u || (*start != '{' && *start != '['))
        return NULL;
    const bool object = *start == '{';
    const char close = object ? '}' : ']';
    const char *p = qwen4_json_skip_space(start + 1u);
    if (*p == close) return p + 1u;
    while (*p) {
        if (object) {
            p = qwen4_json_string_end(p);
            if (!p) return NULL;
            p = qwen4_json_skip_space(p);
            if (*p != ':') return NULL;
            p = qwen4_json_skip_space(p + 1u);
        }
        p = qwen4_json_value_end(p, depth + 1u);
        if (!p) return NULL;
        p = qwen4_json_skip_space(p);
        if (*p == close) return p + 1u;
        if (*p != ',') return NULL;
        p = qwen4_json_skip_space(p + 1u);
        if (*p == close || !*p) return NULL;
    }
    return NULL;
}

static const char *qwen4_json_value_end(const char *start, uint32_t depth) {
    const char *p = qwen4_json_skip_space(start);
    if (!p || !*p) return NULL;
    if (*p == '"') return qwen4_json_string_end(p);
    if (*p == '{' || *p == '[') return qwen4_json_compound_end(p, depth);
    if (!strncmp(p, "true", 4u)) return p + 4u;
    if (!strncmp(p, "false", 5u)) return p + 5u;
    if (!strncmp(p, "null", 4u)) return p + 4u;
    return qwen4_json_number_end(p);
}

/* Return one exact top-level JSON member value.  Walking the root object's
 * grammar, rather than using substring search, prevents a nested record or a
 * quoted string from being mistaken for the manifest-level contract. */
static bool qwen4_json_top_level_value(const char *json,
                                       const char *wanted,
                                       const char **value_start,
                                       const char **value_end) {
    if (value_start) *value_start = NULL;
    if (value_end) *value_end = NULL;
    if (!json || !wanted || !value_start || !value_end) return false;
    const size_t wanted_size = strlen(wanted);
    const char *p = qwen4_json_skip_space(json);
    if (*p != '{') return false;
    p = qwen4_json_skip_space(p + 1u);
    bool found = false;
    while (*p && *p != '}') {
        const char *key_start = p;
        const char *key_end = qwen4_json_string_end(key_start);
        if (!key_end) return false;
        const bool matches =
            (size_t)(key_end - key_start) == wanted_size + 2u &&
            !memcmp(key_start + 1u, wanted, wanted_size);
        p = qwen4_json_skip_space(key_end);
        if (*p != ':') return false;
        const char *member_start = qwen4_json_skip_space(p + 1u);
        const char *member_end = qwen4_json_value_end(member_start, 1u);
        if (!member_end) return false;
        if (matches) {
            if (found) return false;
            *value_start = member_start;
            *value_end = member_end;
            found = true;
        }
        p = qwen4_json_skip_space(member_end);
        if (*p == '}') break;
        if (*p != ',') return false;
        p = qwen4_json_skip_space(p + 1u);
        if (*p == '}' || !*p) return false;
    }
    if (*p != '}') return false;
    p = qwen4_json_skip_space(p + 1u);
    return found && *p == '\0';
}

static bool qwen4_sha256_text_valid(const char *text) {
    if (!text || strlen(text) != 64u) return false;
    for (size_t i = 0; i < 64u; i++) {
        const unsigned char c = (unsigned char)text[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

static bool qwen4_artifact_path_valid(const char *path) {
    if (!path || !path[0] || !strcmp(path, ".") || !strcmp(path, "..")) return false;
    return !strchr(path, '/') && !strchr(path, '\\');
}

bool ds4_qwen4_artifact_tensor_count_valid(
        ds4_qwen4_artifact_kind kind,
        uint64_t count) {
    switch (kind) {
        case DS4_QWEN4_ARTIFACT_BASE:
            return count == DS4_QWEN4_BASE_TENSOR_COUNT;
        case DS4_QWEN4_ARTIFACT_PLE:
            return count == DS4_QWEN4_PLE_TENSOR_COUNT;
        case DS4_QWEN4_ARTIFACT_VISION:
            return count == DS4_QWEN4_VISION_TENSOR_COUNT;
        case DS4_QWEN4_ARTIFACT_MTP:
            return count == DS4_QWEN4_MTP_TENSOR_COUNT;
        default:
            return false;
    }
}

static bool qwen4_artifact_kind_parse(const char *text,
                                      ds4_qwen4_artifact_kind *kind) {
    if (!strcmp(text, "base")) *kind = DS4_QWEN4_ARTIFACT_BASE;
    else if (!strcmp(text, "ple")) *kind = DS4_QWEN4_ARTIFACT_PLE;
    else if (!strcmp(text, "vision")) *kind = DS4_QWEN4_ARTIFACT_VISION;
    else if (!strcmp(text, "mtp")) *kind = DS4_QWEN4_ARTIFACT_MTP;
    else return false;
    return true;
}

static const char *qwen4_json_object_end(const char *start) {
    if (!start || *start != '{') return NULL;
    uint32_t depth = 0;
    bool quoted = false;
    bool escaped = false;
    for (const char *p = start; *p; p++) {
        const char c = *p;
        if (quoted) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') quoted = false;
            continue;
        }
        if (c == '"') quoted = true;
        else if (c == '{') depth++;
        else if (c == '}' && --depth == 0) return p + 1;
    }
    return NULL;
}

int ds4_qwen4_pack_manifest_load(ds4_qwen4_pack_manifest *manifest,
                                 const char *path,
                                 char *error,
                                 size_t error_cap) {
    if (!manifest || !path) {
        qwen4_error(error, error_cap, "missing Qwen pack manifest path");
        return 1;
    }
    memset(manifest, 0, sizeof(*manifest));
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        qwen4_error(error, error_cap, "cannot open Qwen pack manifest %s: %s",
                    path, strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || st.st_size > (128ll << 20)) {
        qwen4_error(error, error_cap, "invalid Qwen pack manifest size: %s", path);
        close(fd);
        return 1;
    }
    char *json = malloc((size_t)st.st_size + 1u);
    if (!json) {
        qwen4_error(error, error_cap, "out of memory reading Qwen pack manifest");
        close(fd);
        return 1;
    }
    size_t used = 0;
    while (used < (size_t)st.st_size) {
        ssize_t got = read(fd, json + used, (size_t)st.st_size - used);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        used += (size_t)got;
    }
    close(fd);
    json[used] = '\0';
    if (used != (size_t)st.st_size) {
        qwen4_error(error, error_cap, "short read of Qwen pack manifest %s", path);
        free(json);
        return 1;
    }

    const char *tensors_start = NULL, *tensors_end = NULL;
    const char *digest_start = NULL, *digest_end = NULL;
    if (!qwen4_json_top_level_value(
            json, "tensors", &tensors_start, &tensors_end) ||
        !tensors_start || *tensors_start != '{') {
        qwen4_error(error, error_cap,
                    "%s has a missing or malformed top-level tensors object",
                    path);
        free(json);
        return 1;
    }
    if (!qwen4_json_top_level_value(
            json, "tensor_manifest_sha256", &digest_start, &digest_end) ||
        !digest_start || !digest_end || digest_end - digest_start != 66 ||
        *digest_start != '"' || digest_end[-1] != '"') {
        qwen4_error(error, error_cap,
                    "%s has a missing or malformed tensor manifest digest",
                    path);
        free(json);
        return 1;
    }
    memcpy(manifest->tensor_manifest_sha256, digest_start + 1u, 64u);
    manifest->tensor_manifest_sha256[64] = '\0';
    char computed_tensor_digest[65];
    qwen4_sha256_buffer_hex(
        tensors_start, (size_t)(tensors_end - tensors_start),
        computed_tensor_digest);
    if (!qwen4_sha256_text_valid(manifest->tensor_manifest_sha256) ||
        strcmp(computed_tensor_digest, manifest->tensor_manifest_sha256)) {
        qwen4_error(error, error_cap,
                    "%s tensor manifest SHA-256 mismatch", path);
        free(json);
        return 1;
    }

    char schema[64], architecture[64];
    uint64_t version = 0;
    ds4_qwen4_pack_profile profile = DS4_QWEN4_PACK_PROFILE_INVALID;
    const char *source = json_object(json, "source");
    bool ok = json_string_value(json, "schema", schema, sizeof(schema)) &&
              !strcmp(schema, DS4_QWEN4_PACK_SCHEMA) &&
              json_u64(json, "version", &version) &&
              version <= UINT32_MAX &&
              ds4_qwen4_pack_profile_from_version((uint32_t)version,
                                                   &profile) &&
              json_string_value(json, "architecture", architecture,
                                sizeof(architecture)) &&
              !strcmp(architecture, DS4_QWEN4_ARCHITECTURE) &&
              json_string_value(json, "pack_id", manifest->pack_id,
                                sizeof(manifest->pack_id)) &&
              source &&
              json_string_value(source, "revision", manifest->source_revision,
                                sizeof(manifest->source_revision)) &&
              qwen4_sha256_text_valid(manifest->tensor_manifest_sha256);
    if (!ok || !manifest->pack_id[0] || !manifest->source_revision[0]) {
        qwen4_error(error, error_cap,
                    "%s is not a compatible DS4 Qwen fast-pack manifest", path);
        free(json);
        return 1;
    }
    manifest->version = (uint32_t)version;
    manifest->profile = profile;
    manifest->base_shard_count = DS4_QWEN4_PACK_BASE_SHARDS;

    const char *array = json_key(json, "artifacts");
    if (!array || *array++ != '[') ok = false;
    bool base_seen[DS4_QWEN4_PACK_MAX_BASE_SHARDS] = {false};
    bool ple_seen = false, vision_seen = false, mtp_seen = false;
    while (ok) {
        while (isspace((unsigned char)*array) || *array == ',') array++;
        if (*array == ']') break;
        if (*array != '{' ||
            manifest->artifact_count >= DS4_QWEN4_PACK_MAX_ARTIFACTS) {
            ok = false;
            break;
        }
        const char *end = qwen4_json_object_end(array);
        if (!end || (size_t)(end - array) > 4096u) {
            ok = false;
            break;
        }
        const size_t object_size = (size_t)(end - array);
        char object[4097];
        memcpy(object, array, object_size);
        object[object_size] = '\0';
        ds4_qwen4_pack_artifact *artifact =
            &manifest->artifacts[manifest->artifact_count];
        char kind[32];
        uint64_t bytes = 0, shard_index = 0;
        ok = json_string_value(object, "kind", kind, sizeof(kind)) &&
             qwen4_artifact_kind_parse(kind, &artifact->kind) &&
             json_string_value(object, "path", artifact->path,
                               sizeof(artifact->path)) &&
             qwen4_artifact_path_valid(artifact->path) &&
             json_string_value(object, "sha256", artifact->sha256,
                               sizeof(artifact->sha256)) &&
             qwen4_sha256_text_valid(artifact->sha256) &&
             json_u64(object, "bytes", &bytes) && bytes != 0;
        if (!ok) break;
        artifact->bytes = bytes;
        if (artifact->kind == DS4_QWEN4_ARTIFACT_BASE) {
            const bool has_shard_index =
                json_u64(object, "shard_index", &shard_index);
            if (!has_shard_index && manifest->base_shard_count == 1u)
                shard_index = 0u;
            ok = (has_shard_index || manifest->base_shard_count == 1u) &&
                 shard_index < manifest->base_shard_count &&
                 !base_seen[shard_index];
            if (!ok) break;
            artifact->shard_index = (uint32_t)shard_index;
            base_seen[shard_index] = true;
        } else if (artifact->kind == DS4_QWEN4_ARTIFACT_PLE) {
            ok = !ple_seen;
            ple_seen = true;
        } else if (artifact->kind == DS4_QWEN4_ARTIFACT_VISION) {
            ok = !vision_seen;
            vision_seen = true;
        } else {
            ok = !mtp_seen;
            mtp_seen = true;
        }
        manifest->artifact_count++;
        array = end;
    }
    for (uint32_t i = 0; i < manifest->base_shard_count; i++)
        ok = ok && base_seen[i];
    ok = ok && ple_seen &&
         manifest->artifact_count >= manifest->base_shard_count + 1u;
    free(json);
    if (!ok) {
        qwen4_error(error, error_cap,
                    "%s has incomplete, duplicate, or invalid artifacts", path);
        memset(manifest, 0, sizeof(*manifest));
        return 1;
    }
    return 0;
}

const ds4_qwen4_pack_artifact *ds4_qwen4_pack_find_artifact(
        const ds4_qwen4_pack_manifest *manifest,
        ds4_qwen4_artifact_kind kind,
        uint32_t shard_index) {
    if (!manifest) return NULL;
    for (uint32_t i = 0; i < manifest->artifact_count; i++) {
        const ds4_qwen4_pack_artifact *artifact = &manifest->artifacts[i];
        if (artifact->kind == kind &&
            (kind != DS4_QWEN4_ARTIFACT_BASE ||
             artifact->shard_index == shard_index)) return artifact;
    }
    return NULL;
}

int ds4_qwen4_pack_validation_mask(
        const ds4_qwen4_pack_manifest *manifest,
        bool vision_requested,
        bool mtp_requested,
        uint32_t *mask_out,
        char *error,
        size_t error_cap) {
    if (mask_out) *mask_out = 0u;
    if (!manifest || !mask_out ||
        manifest->artifact_count > DS4_QWEN4_PACK_MAX_ARTIFACTS) {
        qwen4_error(error, error_cap,
                    "invalid Qwen pack artifact selection request");
        return 1;
    }

    uint32_t mask = 0u;
    for (uint32_t i = 0; i < manifest->artifact_count; i++) {
        const ds4_qwen4_pack_artifact *artifact = &manifest->artifacts[i];
        bool selected = false;
        switch (artifact->kind) {
            case DS4_QWEN4_ARTIFACT_BASE:
            case DS4_QWEN4_ARTIFACT_PLE:
                selected = true;
                break;
            case DS4_QWEN4_ARTIFACT_VISION:
                selected = vision_requested;
                break;
            case DS4_QWEN4_ARTIFACT_MTP:
                selected = mtp_requested;
                break;
            default:
                qwen4_error(error, error_cap,
                            "unknown Qwen pack artifact kind");
                return 1;
        }
        if (selected) mask |= UINT32_C(1) << i;
    }

    for (uint32_t i = 0; i < manifest->base_shard_count; i++) {
        if (!ds4_qwen4_pack_find_artifact(
                manifest, DS4_QWEN4_ARTIFACT_BASE, i)) {
            qwen4_error(error, error_cap,
                        "Qwen pack is missing required base shard %u", i + 1u);
            return 1;
        }
    }
    if (!ds4_qwen4_pack_find_artifact(
            manifest, DS4_QWEN4_ARTIFACT_PLE, 0u)) {
        qwen4_error(error, error_cap,
                    "Qwen pack is missing the required PLE artifact");
        return 1;
    }
    if (vision_requested &&
        !ds4_qwen4_pack_find_artifact(
            manifest, DS4_QWEN4_ARTIFACT_VISION, 0u)) {
        qwen4_error(error, error_cap,
                    "--vision does not belong to this Qwen pack");
        return 1;
    }
    if (mtp_requested &&
        !ds4_qwen4_pack_find_artifact(
            manifest, DS4_QWEN4_ARTIFACT_MTP, 0u)) {
        qwen4_error(error, error_cap,
                    "--mtp-model does not belong to this Qwen pack");
        return 1;
    }
    *mask_out = mask;
    return 0;
}

int ds4_qwen4_pack_validate_artifact(const ds4_qwen4_pack_artifact *artifact,
                                     const char *path,
                                     char *error,
                                     size_t error_cap) {
    if (!artifact || !path) {
        qwen4_error(error, error_cap, "missing Qwen pack artifact");
        return 1;
    }
    char digest[65];
    uint64_t bytes = 0;
    if (ds4_qwen4_sha256_file(path, digest, &bytes, error, error_cap) != 0)
        return 1;
    if (bytes != artifact->bytes) {
        qwen4_error(error, error_cap,
                    "Qwen pack artifact %s has size %" PRIu64
                    ", expected %" PRIu64,
                    path, bytes, artifact->bytes);
        return 1;
    }
    if (strcmp(digest, artifact->sha256)) {
        qwen4_error(error, error_cap,
                    "Qwen pack artifact %s checksum mismatch: got %.16s..., expected %.16s...",
                    path, digest, artifact->sha256);
        return 1;
    }
    return 0;
}

static void qwen4_ple_table_init_empty(ds4_qwen4_ple_table *table) {
    if (!table) return;
    memset(table, 0, sizeof(*table));
    table->fd = -1;
}

static void qwen4_stat_mtime(const struct stat *st,
                             int64_t *seconds,
                             int64_t *nanoseconds) {
#if defined(__APPLE__)
    *seconds = (int64_t)st->st_mtimespec.tv_sec;
    *nanoseconds = (int64_t)st->st_mtimespec.tv_nsec;
#else
    *seconds = (int64_t)st->st_mtime;
    *nanoseconds = 0;
#endif
}

static void qwen4_stat_ctime(const struct stat *st,
                             int64_t *seconds,
                             int64_t *nanoseconds) {
#if defined(__APPLE__)
    *seconds = (int64_t)st->st_ctimespec.tv_sec;
    *nanoseconds = (int64_t)st->st_ctimespec.tv_nsec;
#else
    *seconds = (int64_t)st->st_ctime;
    *nanoseconds = 0;
#endif
}

static bool qwen4_stat_same_identity(const struct stat *left,
                                     const struct stat *right) {
    int64_t left_sec = 0, left_nsec = 0;
    int64_t right_sec = 0, right_nsec = 0;
    int64_t left_csec = 0, left_cnsec = 0;
    int64_t right_csec = 0, right_cnsec = 0;
    qwen4_stat_mtime(left, &left_sec, &left_nsec);
    qwen4_stat_mtime(right, &right_sec, &right_nsec);
    qwen4_stat_ctime(left, &left_csec, &left_cnsec);
    qwen4_stat_ctime(right, &right_csec, &right_cnsec);
    return left->st_dev == right->st_dev && left->st_ino == right->st_ino &&
           left->st_size == right->st_size && left_sec == right_sec &&
           left_nsec == right_nsec && left_csec == right_csec &&
           left_cnsec == right_cnsec;
}

static void qwen4_ple_table_record_identity(ds4_qwen4_ple_table *table,
                                            const struct stat *st) {
    if (!table || !st) return;
    table->identity_device = (uint64_t)st->st_dev;
    table->identity_inode = (uint64_t)st->st_ino;
    qwen4_stat_mtime(st, &table->identity_mtime_sec,
                     &table->identity_mtime_nsec);
    qwen4_stat_ctime(st, &table->identity_ctime_sec,
                     &table->identity_ctime_nsec);
}

static bool qwen4_ple_identity_is_stable(
        const ds4_qwen4_ple_table *table,
        const struct stat *current);

enum {
    QWEN4_GGUF_TYPE_UINT8 = 0,
    QWEN4_GGUF_TYPE_INT8 = 1,
    QWEN4_GGUF_TYPE_UINT16 = 2,
    QWEN4_GGUF_TYPE_INT16 = 3,
    QWEN4_GGUF_TYPE_UINT32 = 4,
    QWEN4_GGUF_TYPE_INT32 = 5,
    QWEN4_GGUF_TYPE_FLOAT32 = 6,
    QWEN4_GGUF_TYPE_BOOL = 7,
    QWEN4_GGUF_TYPE_STRING = 8,
    QWEN4_GGUF_TYPE_ARRAY = 9,
    QWEN4_GGUF_TYPE_UINT64 = 10,
    QWEN4_GGUF_TYPE_INT64 = 11,
    QWEN4_GGUF_TYPE_FLOAT64 = 12,
    QWEN4_GGML_TYPE_Q4_1 = 3,
    QWEN4_GGML_TYPE_I64 = 27,
    QWEN4_Q4_1_BLOCK_SIZE = 32,
    QWEN4_Q4_1_BLOCK_BYTES = 20,
};

typedef struct {
    const uint8_t *data;
    uint64_t size;
    uint64_t position;
} qwen4_gguf_cursor;

typedef struct {
    const uint8_t *data;
    uint64_t size;
} qwen4_gguf_string;

typedef struct {
    bool found;
    uint32_t dimensions;
    uint32_t qtype;
    uint64_t shape[4];
    uint64_t offset;
    uint64_t bytes;
} qwen4_gguf_tensor;

static bool qwen4_gguf_take(qwen4_gguf_cursor *cursor,
                            uint64_t size,
                            const uint8_t **data) {
    if (!cursor || cursor->position > cursor->size ||
        size > cursor->size - cursor->position) return false;
    if (data) *data = cursor->data + cursor->position;
    cursor->position += size;
    return true;
}

static bool qwen4_gguf_u32(qwen4_gguf_cursor *cursor, uint32_t *value) {
    const uint8_t *data = NULL;
    if (!qwen4_gguf_take(cursor, 4u, &data)) return false;
    if (value) *value = load_u32_le(data);
    return true;
}

static bool qwen4_gguf_u64(qwen4_gguf_cursor *cursor, uint64_t *value) {
    const uint8_t *data = NULL;
    if (!qwen4_gguf_take(cursor, 8u, &data)) return false;
    if (value) *value = load_u64_le(data);
    return true;
}

static bool qwen4_gguf_string_read(qwen4_gguf_cursor *cursor,
                                   qwen4_gguf_string *string) {
    uint64_t size = 0;
    const uint8_t *data = NULL;
    if (!qwen4_gguf_u64(cursor, &size) ||
        !qwen4_gguf_take(cursor, size, &data)) return false;
    if (string) {
        string->data = data;
        string->size = size;
    }
    return true;
}

static bool qwen4_gguf_string_is(const qwen4_gguf_string *string,
                                 const char *expected) {
    const size_t size = strlen(expected);
    return string && string->size == size &&
           !memcmp(string->data, expected, size);
}

static bool qwen4_gguf_string_copy(const qwen4_gguf_string *string,
                                   char *out,
                                   size_t out_cap) {
    if (!string || !out || string->size == 0u ||
        string->size >= out_cap) return false;
    memcpy(out, string->data, (size_t)string->size);
    out[string->size] = '\0';
    return true;
}

static bool qwen4_gguf_skip_value(qwen4_gguf_cursor *cursor,
                                  uint32_t type,
                                  uint32_t depth) {
    uint64_t width = 0;
    switch (type) {
        case QWEN4_GGUF_TYPE_UINT8:
        case QWEN4_GGUF_TYPE_INT8:
        case QWEN4_GGUF_TYPE_BOOL: width = 1u; break;
        case QWEN4_GGUF_TYPE_UINT16:
        case QWEN4_GGUF_TYPE_INT16: width = 2u; break;
        case QWEN4_GGUF_TYPE_UINT32:
        case QWEN4_GGUF_TYPE_INT32:
        case QWEN4_GGUF_TYPE_FLOAT32: width = 4u; break;
        case QWEN4_GGUF_TYPE_UINT64:
        case QWEN4_GGUF_TYPE_INT64:
        case QWEN4_GGUF_TYPE_FLOAT64: width = 8u; break;
        case QWEN4_GGUF_TYPE_STRING:
            return qwen4_gguf_string_read(cursor, NULL);
        case QWEN4_GGUF_TYPE_ARRAY: {
            uint32_t element_type = 0;
            uint64_t count = 0;
            if (depth >= 4u || !qwen4_gguf_u32(cursor, &element_type) ||
                !qwen4_gguf_u64(cursor, &count) || count > UINT64_C(1048576))
                return false;
            for (uint64_t i = 0; i < count; i++) {
                if (!qwen4_gguf_skip_value(cursor, element_type, depth + 1u))
                    return false;
            }
            return true;
        }
        default: return false;
    }
    return qwen4_gguf_take(cursor, width, NULL);
}

static bool qwen4_gguf_tensor_bounds(const qwen4_gguf_tensor *tensor,
                                     uint64_t data_start,
                                     uint64_t alignment,
                                     uint64_t file_size,
                                     uint64_t *absolute) {
    uint64_t start = 0;
    return tensor && tensor->found && tensor->bytes != 0u &&
           tensor->offset % alignment == 0u &&
           checked_add_u64(data_start, tensor->offset, &start) &&
           start <= file_size && tensor->bytes <= file_size - start &&
           (!absolute || (*absolute = start, true));
}

static bool qwen4_gguf_i64_vector_is(const uint8_t *map,
                                     uint64_t map_size,
                                     uint64_t absolute,
                                     const int64_t *expected,
                                     uint32_t count) {
    uint64_t bytes = 0;
    if (!map || !expected ||
        !checked_mul_u64(count, sizeof(int64_t), &bytes) ||
        absolute > map_size || bytes > map_size - absolute) return false;
    for (uint32_t i = 0; i < count; i++) {
        if (load_u64_le(map + absolute + (uint64_t)i * sizeof(int64_t)) !=
            (uint64_t)expected[i]) return false;
    }
    return true;
}

static int qwen4_ple_table_adopt_fd(ds4_qwen4_ple_table *table,
                                    int fd,
                                    const char *path,
                                    char *error,
                                    size_t error_cap) {
    if (!table || fd < 0 || !path) {
        qwen4_error(error, error_cap, "missing PLE table descriptor/path");
        return 1;
    }
    table->fd = fd;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        qwen4_error(error, error_cap, "cannot stat or table is too small: %s", path);
        ds4_qwen4_ple_table_close(table);
        return 1;
    }
    if (table->sha256[0] != '\0' &&
        !qwen4_ple_identity_is_stable(table, &st)) {
        qwen4_error(error, error_cap,
                    "Qwen PLE artifact changed before it was mapped: %s",
                    path);
        ds4_qwen4_ple_table_close(table);
        return 1;
    }
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        qwen4_error(error, error_cap, "cannot mmap %s: %s", path, strerror(errno));
        ds4_qwen4_ple_table_close(table);
        return 1;
    }
    table->map = mapped;
    table->size = (uint64_t)st.st_size;
    if (table->identity_device == 0 && table->identity_inode == 0)
        qwen4_ple_table_record_identity(table, &st);

    qwen4_gguf_cursor cursor = {
        .data = table->map,
        .size = table->size,
        .position = 0u,
    };
    const uint8_t *magic = NULL;
    uint32_t version = 0;
    uint64_t tensor_count = 0, metadata_count = 0;
    bool ok = qwen4_gguf_take(&cursor, 4u, &magic) &&
              !memcmp(magic, "GGUF", 4u) &&
              qwen4_gguf_u32(&cursor, &version) && version == 3u &&
              qwen4_gguf_u64(&cursor, &tensor_count) &&
              ds4_qwen4_artifact_tensor_count_valid(
                  DS4_QWEN4_ARTIFACT_PLE, tensor_count) &&
              qwen4_gguf_u64(&cursor, &metadata_count) && metadata_count <= 4096u;
    uint64_t alignment = 32u;
    bool architecture_seen = false;
    bool pack_version_seen = false;
    bool pack_id_seen = false;
    bool source_revision_seen = false;
    bool artifact_seen = false;
    bool quant_seen = false;
    for (uint64_t i = 0; ok && i < metadata_count; i++) {
        qwen4_gguf_string key = {0};
        uint32_t type = 0;
        ok = qwen4_gguf_string_read(&cursor, &key) &&
             qwen4_gguf_u32(&cursor, &type);
        if (!ok) break;
        if (qwen4_gguf_string_is(&key, "general.alignment")) {
            uint32_t value = 0;
            ok = type == QWEN4_GGUF_TYPE_UINT32 &&
                 qwen4_gguf_u32(&cursor, &value) && value != 0u &&
                 value <= UINT32_C(1048576) && (value & (value - 1u)) == 0u;
            alignment = value;
        } else if (qwen4_gguf_string_is(&key, "general.architecture")) {
            qwen4_gguf_string value = {0};
            ok = type == QWEN4_GGUF_TYPE_STRING &&
                 qwen4_gguf_string_read(&cursor, &value) &&
                 qwen4_gguf_string_is(&value, DS4_QWEN4_PLE_ARCHITECTURE);
            architecture_seen = ok;
        } else if (qwen4_gguf_string_is(&key, "ds4.pack.version")) {
            uint32_t value = 0;
            ds4_qwen4_pack_profile profile =
                DS4_QWEN4_PACK_PROFILE_INVALID;
            ok = type == QWEN4_GGUF_TYPE_UINT32 &&
                 qwen4_gguf_u32(&cursor, &value) &&
                 ds4_qwen4_pack_profile_from_version(value, &profile);
            if (ok) table->pack_version = value;
            pack_version_seen = ok;
        } else if (qwen4_gguf_string_is(&key, "ds4.pack.id")) {
            qwen4_gguf_string value = {0};
            ok = !pack_id_seen && type == QWEN4_GGUF_TYPE_STRING &&
                 qwen4_gguf_string_read(&cursor, &value) &&
                 qwen4_gguf_string_copy(
                     &value, table->pack_id, sizeof(table->pack_id));
            pack_id_seen = ok;
        } else if (qwen4_gguf_string_is(
                       &key, "general.source.revision")) {
            qwen4_gguf_string value = {0};
            ok = !source_revision_seen && type == QWEN4_GGUF_TYPE_STRING &&
                 qwen4_gguf_string_read(&cursor, &value) &&
                 qwen4_gguf_string_copy(
                     &value, table->source_revision,
                     sizeof(table->source_revision));
            source_revision_seen = ok;
        } else if (qwen4_gguf_string_is(&key, "ds4.pack.artifact")) {
            qwen4_gguf_string value = {0};
            ok = type == QWEN4_GGUF_TYPE_STRING &&
                 qwen4_gguf_string_read(&cursor, &value) &&
                 qwen4_gguf_string_is(&value, "ple");
            artifact_seen = ok;
        } else if (qwen4_gguf_string_is(&key, "ds4.pack.quant.ple")) {
            qwen4_gguf_string value = {0};
            ok = type == QWEN4_GGUF_TYPE_STRING &&
                 qwen4_gguf_string_read(&cursor, &value) &&
                 qwen4_gguf_string_is(&value, "Q4_1");
            quant_seen = ok;
        } else {
            ok = qwen4_gguf_skip_value(&cursor, type, 0u);
        }
    }

    qwen4_gguf_tensor weight = {0};
    qwen4_gguf_tensor multipliers = {0};
    qwen4_gguf_tensor offsets = {0};
    qwen4_gguf_tensor vocab_sizes = {0};
    for (uint64_t i = 0; ok && i < tensor_count; i++) {
        qwen4_gguf_string name = {0};
        uint32_t dimensions = 0;
        ok = qwen4_gguf_string_read(&cursor, &name) &&
             qwen4_gguf_u32(&cursor, &dimensions) &&
             dimensions > 0u && dimensions <= 4u;
        qwen4_gguf_tensor *tensor = NULL;
        if (ok && qwen4_gguf_string_is(&name, "ple.weight")) tensor = &weight;
        else if (ok && qwen4_gguf_string_is(
                     &name, "ple.layer_multipliers")) tensor = &multipliers;
        else if (ok && qwen4_gguf_string_is(
                     &name, "ple.ngram_heads_offsets")) tensor = &offsets;
        else if (ok && qwen4_gguf_string_is(
                     &name, "ple.ngram_heads_vocab_sizes")) tensor = &vocab_sizes;
        else ok = false;
        if (!ok || tensor->found) {
            ok = false;
            break;
        }
        tensor->found = true;
        tensor->dimensions = dimensions;
        for (uint32_t d = 0; ok && d < dimensions; d++)
            ok = qwen4_gguf_u64(&cursor, &tensor->shape[d]);
        ok = ok && qwen4_gguf_u32(&cursor, &tensor->qtype) &&
             qwen4_gguf_u64(&cursor, &tensor->offset);
    }

    uint64_t row_bytes = 0, weight_bytes = 0;
    ok = ok && architecture_seen && pack_version_seen && pack_id_seen &&
         source_revision_seen && artifact_seen && quant_seen &&
         weight.found && multipliers.found && offsets.found &&
         vocab_sizes.found &&
         weight.dimensions == 2u &&
         weight.shape[0] == DS4_QWEN4_PLE_ROW_DIM && weight.shape[1] != 0u &&
         weight.shape[1] <= INT64_MAX &&
         weight.qtype == QWEN4_GGML_TYPE_Q4_1 &&
         multipliers.dimensions == 1u && multipliers.shape[0] == 3u &&
         multipliers.qtype == QWEN4_GGML_TYPE_I64 &&
         offsets.dimensions == 1u &&
         offsets.shape[0] == DS4_QWEN4_NGRAM_HEADS &&
         offsets.qtype == QWEN4_GGML_TYPE_I64 &&
         vocab_sizes.dimensions == 1u &&
         vocab_sizes.shape[0] == DS4_QWEN4_NGRAM_HEADS &&
         vocab_sizes.qtype == QWEN4_GGML_TYPE_I64 &&
         checked_mul_u64(
             DS4_QWEN4_PLE_ROW_DIM / QWEN4_Q4_1_BLOCK_SIZE,
             QWEN4_Q4_1_BLOCK_BYTES, &row_bytes) &&
         checked_mul_u64(weight.shape[1], row_bytes, &weight_bytes);
    weight.bytes = weight_bytes;
    multipliers.bytes = 3u * sizeof(int64_t);
    offsets.bytes = DS4_QWEN4_NGRAM_HEADS * sizeof(int64_t);
    vocab_sizes.bytes = DS4_QWEN4_NGRAM_HEADS * sizeof(int64_t);
    uint64_t aligned_position = 0;
    if (ok) {
        const uint64_t mask = alignment - 1u;
        ok = checked_add_u64(cursor.position, mask, &aligned_position);
        aligned_position &= ~mask;
    }
    uint64_t weight_absolute = 0;
    uint64_t multipliers_absolute = 0;
    uint64_t offsets_absolute = 0;
    uint64_t vocab_sizes_absolute = 0;
    ok = ok && aligned_position <= table->size &&
         qwen4_gguf_tensor_bounds(
             &weight, aligned_position, alignment, table->size,
             &weight_absolute) &&
         qwen4_gguf_tensor_bounds(
             &multipliers, aligned_position, alignment, table->size,
             &multipliers_absolute) &&
         qwen4_gguf_tensor_bounds(
             &offsets, aligned_position, alignment, table->size,
             &offsets_absolute) &&
         qwen4_gguf_tensor_bounds(
             &vocab_sizes, aligned_position, alignment, table->size,
             &vocab_sizes_absolute);
    ds4_qwen4_ngram_hash expected_hash;
    ok = ok && ds4_qwen4_ngram_hash_init(
                    &expected_hash, DS4_QWEN4_VOCAB,
                    DS4_QWEN4_NGRAM_SIZE, DS4_QWEN4_HEADS_PER_NGRAM,
                    DS4_QWEN4_NGRAM_VOCAB_BASE,
                    DS4_QWEN4_NGRAM_VOCAB_DIVISOR,
                    DS4_QWEN4_NGRAM_SEED, 0u, DS4_QWEN4_NGRAM_EOS) &&
         qwen4_gguf_i64_vector_is(
             table->map, table->size, multipliers_absolute,
             expected_hash.multipliers, DS4_QWEN4_NGRAM_SIZE) &&
         qwen4_gguf_i64_vector_is(
             table->map, table->size, offsets_absolute,
             expected_hash.offsets, DS4_QWEN4_NGRAM_HEADS) &&
         qwen4_gguf_i64_vector_is(
             table->map, table->size, vocab_sizes_absolute,
             expected_hash.vocab, DS4_QWEN4_NGRAM_HEADS);
    if (!ok) {
        qwen4_error(error, error_cap,
                    "%s is not a valid DS4 Qwen v3/v4 Q4_1 PLE GGUF", path);
        ds4_qwen4_ple_table_close(table);
        return 1;
    }
    table->rows = weight.shape[1];
    table->dim = DS4_QWEN4_PLE_ROW_DIM;
    table->qtype = QWEN4_GGML_TYPE_Q4_1;
    table->block_size = QWEN4_Q4_1_BLOCK_SIZE;
    table->blocks_per_row =
        DS4_QWEN4_PLE_ROW_DIM / QWEN4_Q4_1_BLOCK_SIZE;
    table->row_bytes = (uint32_t)row_bytes;
    table->data_offset = weight_absolute;
    struct stat stable;
    if (table->sha256[0] != '\0' &&
        (fstat(fd, &stable) != 0 ||
         !qwen4_ple_identity_is_stable(table, &stable))) {
        qwen4_error(error, error_cap,
                    "Qwen PLE artifact changed while it was mapped: %s",
                    path);
        ds4_qwen4_ple_table_close(table);
        return 1;
    }
    return 0;
}

int ds4_qwen4_ple_table_open(ds4_qwen4_ple_table *table,
                             const char *path,
                             char *error,
                             size_t error_cap) {
    qwen4_ple_table_init_empty(table);
    if (!table || !path) {
        qwen4_error(error, error_cap, "missing PLE table path");
        return 1;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        qwen4_error(error, error_cap, "cannot open %s: %s", path, strerror(errno));
        return 1;
    }
    return qwen4_ple_table_adopt_fd(table, fd, path, error, error_cap);
}

int ds4_qwen4_pack_validate_ple_open(
        const ds4_qwen4_pack_artifact *artifact,
        const char *path,
        ds4_qwen4_ple_table *table,
        char *error,
        size_t error_cap) {
    qwen4_ple_table_init_empty(table);
    if (!artifact || artifact->kind != DS4_QWEN4_ARTIFACT_PLE ||
        !path || !table) {
        qwen4_error(error, error_cap, "missing Qwen PLE artifact/table");
        return 1;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        qwen4_error(error, error_cap, "cannot open %s: %s", path, strerror(errno));
        return 1;
    }
    struct stat before;
    if (fstat(fd, &before) != 0 || before.st_size < 0) {
        qwen4_error(error, error_cap, "cannot stat %s: %s", path, strerror(errno));
        close(fd);
        return 1;
    }
    if ((uint64_t)before.st_size != artifact->bytes) {
        qwen4_error(error, error_cap,
                    "Qwen pack artifact %s has size %" PRIu64
                    ", expected %" PRIu64,
                    path, (uint64_t)before.st_size, artifact->bytes);
        close(fd);
        return 1;
    }
    char digest[65];
    uint64_t bytes = 0;
    qwen4_sha256_io_status io_status;
    if (qwen4_sha256_fd(fd, path, true, digest, &bytes,
                        &io_status, error, error_cap) != 0) {
        close(fd);
        return 1;
    }
    if (bytes != artifact->bytes || strcmp(digest, artifact->sha256)) {
        qwen4_error(error, error_cap,
                    "Qwen pack artifact %s checksum mismatch: got %.16s..., expected %.16s...",
                    path, digest, artifact->sha256);
        close(fd);
        return 1;
    }
    struct stat after;
    if (fstat(fd, &after) != 0 || !qwen4_stat_same_identity(&before, &after)) {
        qwen4_error(error, error_cap,
                    "Qwen PLE artifact changed while it was validated: %s",
                    path);
        close(fd);
        return 1;
    }
    table->validation_nocache_requested = io_status.nocache_requested;
    table->validation_nocache_enabled = io_status.nocache_enabled;
    table->validation_nocache_cleared = io_status.nocache_cleared;
    memcpy(table->sha256, digest, sizeof(table->sha256));
    qwen4_ple_table_record_identity(table, &after);
    table->size = (uint64_t)after.st_size;
#if defined(__APPLE__) && defined(F_RDAHEAD)
    /* Runtime PLE reads are sparse and explicitly tiled.  Sequential read-ahead
     * would populate unrelated 16 KiB pages from the 29.8 GiB sidecar. */
    table->runtime_readahead_requested = true;
    table->runtime_readahead_disabled = fcntl(fd, F_RDAHEAD, 0) == 0;
#endif
    return qwen4_ple_table_adopt_fd(table, fd, path, error, error_cap);
}

void ds4_qwen4_ple_table_close(ds4_qwen4_ple_table *table) {
    if (!table) return;
    if (table->map && table->size) munmap((void *)table->map, (size_t)table->size);
    if (table->fd >= 0) close(table->fd);
    memset(table, 0, sizeof(*table));
    table->fd = -1;
}

static int qwen4_compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return a < b ? -1 : a > b;
}

static bool qwen4_ple_plan_push(ds4_qwen4_ple_page_plan *plan,
                                size_t *capacity,
                                uint64_t page) {
    if (plan->count == *capacity) {
        size_t next = *capacity ? *capacity * 2u : 256u;
        if (next < *capacity || next > SIZE_MAX / sizeof(plan->pages[0]))
            return false;
        uint64_t *grown = realloc(plan->pages,
                                  next * sizeof(plan->pages[0]));
        if (!grown) return false;
        plan->pages = grown;
        *capacity = next;
    }
    plan->pages[plan->count++] = page;
    return true;
}

static bool qwen4_ple_plan_range(ds4_qwen4_ple_page_plan *plan,
                                 size_t *capacity,
                                 uint64_t offset,
                                 uint64_t length,
                                 uint64_t file_size) {
    if (length == 0 || offset > file_size || length > file_size - offset)
        return false;
    const uint64_t first = offset / plan->page_size;
    const uint64_t last = (offset + length - 1u) / plan->page_size;
    for (uint64_t page = first;; page++) {
        if (!qwen4_ple_plan_push(plan, capacity, page)) return false;
        if (page == last) break;
    }
    return true;
}

static bool qwen4_ple_row_offset(const ds4_qwen4_ple_table *table,
                                 uint64_t row,
                                 uint64_t *offset) {
    uint64_t delta = 0, absolute = 0;
    if (!table || row >= table->rows ||
        table->qtype != QWEN4_GGML_TYPE_Q4_1 ||
        table->block_size != QWEN4_Q4_1_BLOCK_SIZE ||
        table->blocks_per_row == 0u ||
        table->dim != table->blocks_per_row * QWEN4_Q4_1_BLOCK_SIZE ||
        table->row_bytes != table->blocks_per_row * QWEN4_Q4_1_BLOCK_BYTES ||
        !checked_mul_u64(row, table->row_bytes, &delta) ||
        !checked_add_u64(table->data_offset, delta, &absolute) ||
        absolute > table->size || table->row_bytes > table->size - absolute)
        return false;
    if (offset) *offset = absolute;
    return true;
}

bool ds4_qwen4_ple_plan_pages(const ds4_qwen4_ple_table *table,
                              const int64_t *rows,
                              size_t row_count,
                              uint64_t page_size,
                              ds4_qwen4_ple_page_plan *plan,
                              char *error,
                              size_t error_cap) {
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!table || !rows || !plan || row_count == 0 || page_size == 0 ||
        table->size == 0 || table->rows == 0) {
        qwen4_error(error, error_cap, "invalid Qwen PLE page-plan input");
        return false;
    }
    plan->page_size = page_size;
    size_t capacity = 0;
    for (size_t i = 0; i < row_count; i++) {
        if (rows[i] < 0 || (uint64_t)rows[i] >= table->rows) {
            qwen4_error(error, error_cap,
                        "Qwen PLE row %" PRId64 " is out of range", rows[i]);
            ds4_qwen4_ple_page_plan_free(plan);
            return false;
        }
        uint64_t row_offset = 0;
        if (!qwen4_ple_row_offset(table, (uint64_t)rows[i], &row_offset) ||
            !qwen4_ple_plan_range(plan, &capacity, row_offset,
                                  table->row_bytes, table->size)) {
            qwen4_error(error, error_cap,
                        "Qwen PLE row page range is invalid or out of memory");
            ds4_qwen4_ple_page_plan_free(plan);
            return false;
        }
    }
    qsort(plan->pages, plan->count, sizeof(plan->pages[0]),
          qwen4_compare_u64);
    size_t unique = 0;
    for (size_t i = 0; i < plan->count; i++) {
        if (unique == 0 || plan->pages[i] != plan->pages[unique - 1u])
            plan->pages[unique++] = plan->pages[i];
    }
    plan->count = unique;
    return true;
}

void ds4_qwen4_ple_page_plan_free(ds4_qwen4_ple_page_plan *plan) {
    if (!plan) return;
    free(plan->pages);
    memset(plan, 0, sizeof(*plan));
}

static bool qwen4_ple_identity_is_stable(
        const ds4_qwen4_ple_table *table,
        const struct stat *current) {
    int64_t seconds = 0, nanoseconds = 0;
    int64_t ctime_seconds = 0, ctime_nanoseconds = 0;
    qwen4_stat_mtime(current, &seconds, &nanoseconds);
    qwen4_stat_ctime(current, &ctime_seconds, &ctime_nanoseconds);
    return (uint64_t)current->st_dev == table->identity_device &&
           (uint64_t)current->st_ino == table->identity_inode &&
           current->st_size >= 0 &&
           (uint64_t)current->st_size == table->size &&
           seconds == table->identity_mtime_sec &&
           nanoseconds == table->identity_mtime_nsec &&
           ctime_seconds == table->identity_ctime_sec &&
           ctime_nanoseconds == table->identity_ctime_nsec;
}

bool ds4_qwen4_ple_measure_residency(
        const ds4_qwen4_ple_table *table,
        const int64_t *rows,
        size_t row_count,
        ds4_qwen4_ple_residency *residency,
        char *error,
        size_t error_cap) {
    if (residency) memset(residency, 0, sizeof(*residency));
    if (!table || table->fd < 0 || !table->map || !rows || !residency) {
        qwen4_error(error, error_cap,
                    "invalid Qwen PLE residency measurement input");
        return false;
    }
#if !defined(__APPLE__)
    (void)row_count;
    qwen4_error(error, error_cap,
                "Qwen PLE cold evidence requires Darwin mincore");
    return false;
#else
    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        qwen4_error(error, error_cap, "cannot determine Darwin page size");
        return false;
    }
    const uint64_t page_size = (uint64_t)page_size_long;
    ds4_qwen4_ple_page_plan plan;
    if (!ds4_qwen4_ple_plan_pages(table, rows, row_count, page_size,
                                  &plan, error, error_cap)) return false;
    const uint64_t file_pages =
        table->size / page_size + (table->size % page_size != 0u);
    if (file_pages == 0 || file_pages > SIZE_MAX) {
        qwen4_error(error, error_cap, "Qwen PLE mapping is too large for mincore");
        ds4_qwen4_ple_page_plan_free(&plan);
        return false;
    }
    unsigned char *state = calloc((size_t)file_pages, 1u);
    if (!state) {
        qwen4_error(error, error_cap,
                    "out of memory allocating Qwen PLE mincore vector");
        ds4_qwen4_ple_page_plan_free(&plan);
        return false;
    }
    if (mincore((void *)table->map, (size_t)table->size, (char *)state) != 0) {
        qwen4_error(error, error_cap, "mincore failed for Qwen PLE: %s",
                    strerror(errno));
        free(state);
        ds4_qwen4_ple_page_plan_free(&plan);
        return false;
    }
    uint64_t resident = 0;
    for (size_t i = 0; i < plan.count; i++) {
        if (plan.pages[i] >= file_pages) {
            qwen4_error(error, error_cap,
                        "Qwen PLE target page is outside the mapping");
            free(state);
            ds4_qwen4_ple_page_plan_free(&plan);
            return false;
        }
        if ((state[plan.pages[i]] & 1u) != 0u) resident++;
    }
    struct stat current;
    const bool stat_ok = fstat(table->fd, &current) == 0;
    residency->page_size = page_size;
    residency->target_pages = plan.count;
    residency->resident_target_pages = resident;
    residency->cold_target_pages = plan.count - resident;
    residency->identity_device = stat_ok ? (uint64_t)current.st_dev : 0;
    residency->identity_inode = stat_ok ? (uint64_t)current.st_ino : 0;
    residency->identity_size =
        stat_ok && current.st_size >= 0 ? (uint64_t)current.st_size : 0;
    if (stat_ok) {
        qwen4_stat_mtime(&current, &residency->identity_mtime_sec,
                         &residency->identity_mtime_nsec);
        qwen4_stat_ctime(&current, &residency->identity_ctime_sec,
                         &residency->identity_ctime_nsec);
    }
    residency->identity_stable =
        stat_ok && qwen4_ple_identity_is_stable(table, &current);
    free(state);
    ds4_qwen4_ple_page_plan_free(&plan);
    if (!residency->identity_stable) {
        qwen4_error(error, error_cap,
                    "Qwen PLE artifact changed after startup validation");
        return false;
    }
    return true;
#endif
}

static void qwen4_q4_1_row_f32(const uint8_t *row,
                               uint32_t blocks,
                               float *out) {
    for (uint32_t block_index = 0; block_index < blocks; block_index++) {
        const uint8_t *block =
            row + (uint64_t)block_index * QWEN4_Q4_1_BLOCK_BYTES;
        const float scale = f16_to_f32(load_u16_le(block));
        const float minimum = f16_to_f32(load_u16_le(block + 2u));
        float *dst = out + (uint64_t)block_index * QWEN4_Q4_1_BLOCK_SIZE;
        for (uint32_t i = 0; i < 16u; i++) {
            const uint8_t packed = block[4u + i];
            dst[i] = minimum + scale * (float)(packed & 0x0fu);
            dst[i + 16u] = minimum + scale * (float)(packed >> 4);
        }
    }
}

static void qwen4_q4_1_row_bf16(const uint8_t *row,
                                uint32_t blocks,
                                uint16_t *out) {
    for (uint32_t block_index = 0; block_index < blocks; block_index++) {
        const uint8_t *block =
            row + (uint64_t)block_index * QWEN4_Q4_1_BLOCK_BYTES;
        const float scale = f16_to_f32(load_u16_le(block));
        const float minimum = f16_to_f32(load_u16_le(block + 2u));
        uint16_t *dst = out +
            (uint64_t)block_index * QWEN4_Q4_1_BLOCK_SIZE;
        for (uint32_t i = 0; i < 16u; i++) {
            const uint8_t packed = block[4u + i];
            dst[i] = f32_to_bf16(
                minimum + scale * (float)(packed & 0x0fu));
            dst[i + 16u] = f32_to_bf16(
                minimum + scale * (float)(packed >> 4));
        }
    }
}

bool ds4_qwen4_ple_row_f32(const ds4_qwen4_ple_table *table,
                           uint64_t row,
                           float *out,
                           size_t out_count) {
    uint64_t offset = 0;
    if (!table || !table->map || !out || out_count < table->dim ||
        !qwen4_ple_row_offset(table, row, &offset)) return false;
    qwen4_q4_1_row_f32(
        table->map + offset, table->blocks_per_row, out);
    return true;
}

bool ds4_qwen4_ple_gather_bf16(const ds4_qwen4_ple_table *table,
                               const int64_t *rows,
                               size_t row_count,
                               uint16_t *out,
                               size_t out_count) {
    if (!table || !rows || !out ||
        row_count > SIZE_MAX / table->dim ||
        out_count < row_count * table->dim) return false;
    for (size_t r = 0; r < row_count; r++) {
        if (rows[r] < 0 || (uint64_t)rows[r] >= table->rows) return false;
        uint64_t offset = 0;
        if (!qwen4_ple_row_offset(table, (uint64_t)rows[r], &offset))
            return false;
        qwen4_q4_1_row_bf16(table->map + offset, table->blocks_per_row,
                            out + r * table->dim);
    }
    return true;
}

typedef enum {
    QWEN4_STAGE_IDLE = 0,
    QWEN4_STAGE_QUEUED,
    QWEN4_STAGE_RUNNING,
    QWEN4_STAGE_DONE,
    QWEN4_STAGE_FAILED,
} qwen4_stage_state;

typedef struct qwen4_ple_stager_impl qwen4_ple_stager_impl;

typedef struct {
    qwen4_ple_stager_impl *owner;
    uint32_t index;
    pthread_t thread;
    int64_t *rows;
    uint16_t *values;
    size_t row_count;
    qwen4_stage_state state;
    bool started;
} qwen4_ple_stage_slot;

struct qwen4_ple_stager_impl {
    const ds4_qwen4_ple_table *table;
    size_t max_rows;
    ds4_qwen4_ple_gather_mode mode;
    bool random_advice_requested;
    bool random_advice_succeeded;
    uint64_t gather_count;
    uint64_t gather_rows;
    double gather_seconds;
    double wait_seconds;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool stopping;
    qwen4_ple_stage_slot slots[2];
};

const char *ds4_qwen4_ple_gather_mode_name(
        ds4_qwen4_ple_gather_mode mode) {
    return mode == DS4_QWEN4_PLE_GATHER_MMAP ? "mmap" : "pread";
}

static double qwen4_monotonic_seconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static bool qwen4_pread_exact(int fd, void *dst, size_t size, uint64_t offset) {
    uint8_t *p = dst;
    while (size) {
        const ssize_t got = pread(fd, p, size, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        p += (size_t)got;
        size -= (size_t)got;
        offset += (uint64_t)got;
    }
    return true;
}

/* Prefill gathers thousands of scattered 100-byte rows per 512-token tile;
 * one serial pread per row runs the SSD at queue depth one and a cold
 * region costs ~60us/row (7.7 s per fresh 8K chunk, measured on the M3
 * Ultra).  Splitting one gather across threads keeps the staged bytes
 * identical because every row writes a disjoint output slice.  Decode and
 * verifier gathers stay serial through the row-count threshold. */
static uint32_t qwen4_ple_pread_thread_count(size_t row_count) {
    static int32_t configured = -1;
    if (configured < 0) {
        const char *raw = getenv("DS4_QWEN4_PLE_GATHER_THREADS");
        configured = raw && *raw ? atoi(raw) : 32;
        if (configured < 1) configured = 1;
        if (configured > 64) configured = 64;
    }
    if ((uint32_t)configured < 2u || row_count < 512u) return 1u;
    if ((size_t)configured > row_count) return (uint32_t)row_count;
    return (uint32_t)configured;
}

typedef struct {
    const ds4_qwen4_ple_table *table;
    const int64_t *rows;
    uint16_t *out;
    size_t begin;
    size_t end;
    bool ok;
} qwen4_ple_pread_slice;

static void qwen4_ple_gather_pread_slice(qwen4_ple_pread_slice *slice) {
    const ds4_qwen4_ple_table *table = slice->table;
    const int64_t *rows = slice->rows;
    const size_t row_bytes = table->row_bytes;
    uint8_t *row_buffer = malloc(row_bytes ? row_bytes : 1u);
    if (!row_buffer) {
        slice->ok = false;
        return;
    }
    bool ok = true;
    for (size_t r = slice->begin; ok && r < slice->end; r++) {
        if (rows[r] < 0 || (uint64_t)rows[r] >= table->rows) {
            ok = false;
            break;
        }
        uint64_t offset = 0;
        ok = qwen4_ple_row_offset(table, (uint64_t)rows[r], &offset) &&
             qwen4_pread_exact(table->fd, row_buffer, row_bytes, offset);
        if (ok)
            qwen4_q4_1_row_bf16(row_buffer, table->blocks_per_row,
                                slice->out + r * table->dim);
    }
    free(row_buffer);
    slice->ok = ok;
}

static void *qwen4_ple_gather_pread_thread(void *opaque) {
    qwen4_ple_pread_slice *slice = opaque;
    qwen4_ple_gather_pread_slice(slice);
    return NULL;
}

static bool qwen4_ple_gather_pread_bf16(
        const ds4_qwen4_ple_table *table,
        const int64_t *rows,
        size_t row_count,
        uint16_t *out) {
    if (!table || table->fd < 0 || !rows || !out) return false;
    const uint32_t threads = qwen4_ple_pread_thread_count(row_count);
    if (threads < 2u) {
        qwen4_ple_pread_slice serial = {
            .table = table, .rows = rows, .out = out,
            .begin = 0, .end = row_count, .ok = false,
        };
        qwen4_ple_gather_pread_slice(&serial);
        return serial.ok;
    }
    qwen4_ple_pread_slice *slices = calloc(threads, sizeof(*slices));
    if (!slices) return false;
    pthread_t helpers[64];
    bool helper_created[64];
    memset(helper_created, 0, sizeof(helper_created));
    const size_t rows_per_slice = (row_count + threads - 1u) / threads;
    for (uint32_t t = 0; t < threads; t++) {
        const size_t begin = (size_t)t * rows_per_slice;
        slices[t].table = table;
        slices[t].rows = rows;
        slices[t].out = out;
        slices[t].begin = begin;
        slices[t].end = begin + rows_per_slice < row_count
            ? begin + rows_per_slice : row_count;
    }
    for (uint32_t t = 0; t + 1u < threads; t++) {
        /* A failed creation degrades to fewer workers, never to wrong
         * output: the calling slot thread runs that slice itself. */
        if (pthread_create(&helpers[t], NULL,
                           qwen4_ple_gather_pread_thread,
                           &slices[t]) == 0)
            helper_created[t] = true;
        else
            qwen4_ple_gather_pread_slice(&slices[t]);
    }
    qwen4_ple_gather_pread_slice(&slices[threads - 1u]);
    for (uint32_t t = 0; t + 1u < threads; t++)
        if (helper_created[t]) pthread_join(helpers[t], NULL);
    bool ok = true;
    for (uint32_t t = 0; t < threads; t++) ok = ok && slices[t].ok;
    free(slices);
    return ok;
}

static void *qwen4_ple_stage_worker(void *opaque) {
    qwen4_ple_stage_slot *slot = opaque;
    qwen4_ple_stager_impl *impl = slot->owner;
    pthread_mutex_lock(&impl->mutex);
    while (!impl->stopping) {
        while (!impl->stopping && slot->state != QWEN4_STAGE_QUEUED)
            pthread_cond_wait(&impl->condition, &impl->mutex);
        if (impl->stopping) break;
        slot->state = QWEN4_STAGE_RUNNING;
        const size_t row_count = slot->row_count;
        pthread_mutex_unlock(&impl->mutex);
        const double gather_t0 = qwen4_monotonic_seconds();
        const bool ok = impl->mode == DS4_QWEN4_PLE_GATHER_MMAP
            ? ds4_qwen4_ple_gather_bf16(
                impl->table, slot->rows, row_count, slot->values,
                row_count * impl->table->dim)
            : qwen4_ple_gather_pread_bf16(
                impl->table, slot->rows, row_count, slot->values);
        const double gather_t1 = qwen4_monotonic_seconds();
        pthread_mutex_lock(&impl->mutex);
        impl->gather_count++;
        impl->gather_rows += row_count;
        if (gather_t1 >= gather_t0)
            impl->gather_seconds += gather_t1 - gather_t0;
        slot->state = ok ? QWEN4_STAGE_DONE : QWEN4_STAGE_FAILED;
        pthread_cond_broadcast(&impl->condition);
    }
    pthread_mutex_unlock(&impl->mutex);
    return NULL;
}

int ds4_qwen4_ple_stager_init(ds4_qwen4_ple_stager *stager,
                              const ds4_qwen4_ple_table *table,
                              size_t max_rows,
                              char *error,
                              size_t error_cap) {
    if (stager) memset(stager, 0, sizeof(*stager));
    if (!stager || !table || table->fd < 0 || table->dim == 0u ||
        max_rows == 0 ||
        max_rows > SIZE_MAX / table->dim ||
        max_rows * table->dim > SIZE_MAX / sizeof(uint16_t)) {
        qwen4_error(error, error_cap, "invalid Qwen PLE stager geometry");
        return 1;
    }
    ds4_qwen4_ple_gather_mode mode = DS4_QWEN4_PLE_GATHER_PREAD;
    const char *mode_env = getenv("DS4_QWEN4_PLE_GATHER");
    if (mode_env && mode_env[0] && strcmp(mode_env, "pread")) {
        if (!strcmp(mode_env, "mmap")) {
            mode = DS4_QWEN4_PLE_GATHER_MMAP;
        } else {
            qwen4_error(error, error_cap,
                        "invalid DS4_QWEN4_PLE_GATHER=%s; expected pread or mmap",
                        mode_env);
            return 1;
        }
    }
    if (mode == DS4_QWEN4_PLE_GATHER_MMAP &&
        (!table->map || table->size == 0u || table->size > SIZE_MAX)) {
        qwen4_error(error, error_cap,
                    "Qwen PLE mmap gather requires a complete mapped table");
        return 1;
    }
    qwen4_ple_stager_impl *impl = calloc(1, sizeof(*impl));
    if (!impl) {
        qwen4_error(error, error_cap, "out of memory creating Qwen PLE stager");
        return 1;
    }
    impl->table = table;
    impl->max_rows = max_rows;
    impl->mode = mode;
    if (mode == DS4_QWEN4_PLE_GATHER_MMAP) {
        impl->random_advice_requested = true;
#if defined(POSIX_MADV_RANDOM)
        const int advice_error = posix_madvise(
            (void *)table->map, (size_t)table->size, POSIX_MADV_RANDOM);
        impl->random_advice_succeeded = advice_error == 0;
        if (impl->random_advice_succeeded) {
            fprintf(stderr,
                    "ds4: Qwen PLE %s gather enabled; "
                    "POSIX_MADV_RANDOM applied to the file mapping\n",
                    ds4_qwen4_ple_gather_mode_name(mode));
        } else {
            fprintf(stderr,
                    "ds4: Qwen PLE %s gather enabled; "
                    "POSIX_MADV_RANDOM failed: %s\n",
                    ds4_qwen4_ple_gather_mode_name(mode),
                    strerror(advice_error));
        }
#elif defined(MADV_RANDOM)
        const int advice_result = madvise(
            (void *)table->map, (size_t)table->size, MADV_RANDOM);
        impl->random_advice_succeeded = advice_result == 0;
        if (impl->random_advice_succeeded) {
            fprintf(stderr,
                    "ds4: Qwen PLE %s gather enabled; "
                    "MADV_RANDOM applied to the file mapping\n",
                    ds4_qwen4_ple_gather_mode_name(mode));
        } else {
            fprintf(stderr,
                    "ds4: Qwen PLE %s gather enabled; MADV_RANDOM failed: %s\n",
                    ds4_qwen4_ple_gather_mode_name(mode),
                    strerror(errno));
        }
#else
        fprintf(stderr,
                "ds4: Qwen PLE %s gather enabled; "
                "random mapping advice is unavailable on this platform\n",
                ds4_qwen4_ple_gather_mode_name(mode));
#endif
    }
    if (pthread_mutex_init(&impl->mutex, NULL) != 0) {
        qwen4_error(error, error_cap, "cannot initialize Qwen PLE stager synchronization");
        free(impl);
        return 1;
    }
    if (pthread_cond_init(&impl->condition, NULL) != 0) {
        qwen4_error(error, error_cap, "cannot initialize Qwen PLE stager synchronization");
        pthread_mutex_destroy(&impl->mutex);
        free(impl);
        return 1;
    }
    for (uint32_t i = 0; i < 2u; i++) {
        qwen4_ple_stage_slot *slot = &impl->slots[i];
        slot->owner = impl;
        slot->index = i;
        slot->rows = malloc(max_rows * sizeof(slot->rows[0]));
        slot->values = malloc(max_rows * table->dim * sizeof(slot->values[0]));
        if (!slot->rows || !slot->values ||
            pthread_create(&slot->thread, NULL, qwen4_ple_stage_worker, slot) != 0) {
            qwen4_error(error, error_cap, "cannot allocate Qwen PLE staging slot %u", i);
            pthread_mutex_lock(&impl->mutex);
            impl->stopping = true;
            pthread_cond_broadcast(&impl->condition);
            pthread_mutex_unlock(&impl->mutex);
            for (uint32_t j = 0; j <= i; j++) {
                if (impl->slots[j].started)
                    pthread_join(impl->slots[j].thread, NULL);
                free(impl->slots[j].rows);
                free(impl->slots[j].values);
            }
            pthread_cond_destroy(&impl->condition);
            pthread_mutex_destroy(&impl->mutex);
            free(impl);
            return 1;
        }
        slot->started = true;
    }
    stager->impl = impl;
    return 0;
}

void ds4_qwen4_ple_stager_destroy(ds4_qwen4_ple_stager *stager) {
    if (!stager || !stager->impl) return;
    qwen4_ple_stager_impl *impl = stager->impl;
    pthread_mutex_lock(&impl->mutex);
    impl->stopping = true;
    pthread_cond_broadcast(&impl->condition);
    pthread_mutex_unlock(&impl->mutex);
    for (uint32_t i = 0; i < 2u; i++) {
        if (impl->slots[i].started) pthread_join(impl->slots[i].thread, NULL);
        free(impl->slots[i].rows);
        free(impl->slots[i].values);
    }
    pthread_cond_destroy(&impl->condition);
    pthread_mutex_destroy(&impl->mutex);
    free(impl);
    stager->impl = NULL;
}

bool ds4_qwen4_ple_stager_submit(ds4_qwen4_ple_stager *stager,
                                 uint32_t slot_index,
                                 const int64_t *rows,
                                 size_t row_count) {
    if (!stager || !stager->impl || !rows || slot_index >= 2u) return false;
    qwen4_ple_stager_impl *impl = stager->impl;
    if (row_count > impl->max_rows) return false;
    pthread_mutex_lock(&impl->mutex);
    qwen4_ple_stage_slot *slot = &impl->slots[slot_index];
    const bool available = slot->state == QWEN4_STAGE_IDLE;
    if (available) {
        memcpy(slot->rows, rows, row_count * sizeof(rows[0]));
        slot->row_count = row_count;
        slot->state = QWEN4_STAGE_QUEUED;
        pthread_cond_broadcast(&impl->condition);
    }
    pthread_mutex_unlock(&impl->mutex);
    return available;
}

bool ds4_qwen4_ple_stager_wait(ds4_qwen4_ple_stager *stager,
                               uint32_t slot_index,
                               const uint16_t **data,
                               size_t *value_count) {
    if (!stager || !stager->impl || slot_index >= 2u || !data || !value_count)
        return false;
    qwen4_ple_stager_impl *impl = stager->impl;
    const double wait_t0 = qwen4_monotonic_seconds();
    bool waited = false;
    pthread_mutex_lock(&impl->mutex);
    qwen4_ple_stage_slot *slot = &impl->slots[slot_index];
    while (!impl->stopping &&
           (slot->state == QWEN4_STAGE_QUEUED ||
            slot->state == QWEN4_STAGE_RUNNING)) {
        waited = true;
        pthread_cond_wait(&impl->condition, &impl->mutex);
    }
    if (waited) {
        const double wait_t1 = qwen4_monotonic_seconds();
        if (wait_t1 >= wait_t0) impl->wait_seconds += wait_t1 - wait_t0;
    }
    const bool ok = slot->state == QWEN4_STAGE_DONE;
    if (ok) {
        *data = slot->values;
        *value_count = slot->row_count * impl->table->dim;
    }
    slot->state = QWEN4_STAGE_IDLE;
    pthread_mutex_unlock(&impl->mutex);
    return ok;
}

bool ds4_qwen4_ple_stager_snapshot(
        ds4_qwen4_ple_stager *stager,
        ds4_qwen4_ple_stager_stats *stats) {
    if (!stager || !stager->impl || !stats) return false;
    qwen4_ple_stager_impl *impl = stager->impl;
    pthread_mutex_lock(&impl->mutex);
    stats->mode = impl->mode;
    stats->random_advice_requested = impl->random_advice_requested;
    stats->random_advice_succeeded = impl->random_advice_succeeded;
    stats->gather_count = impl->gather_count;
    stats->gather_rows = impl->gather_rows;
    stats->gather_seconds = impl->gather_seconds;
    stats->wait_seconds = impl->wait_seconds;
    pthread_mutex_unlock(&impl->mutex);
    return true;
}
