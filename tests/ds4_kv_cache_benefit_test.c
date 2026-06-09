#include "../ds4.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} test_buf;

typedef struct {
    int top1_a;
    int top1_b;
    int nonfinite;
    double rms;
    float max_abs;
    bool same_top1;
} logit_cmp;

static void fail(const char *msg) {
    fprintf(stderr, "ds4_kv_cache_benefit_test: %s\n", msg);
    exit(1);
}

#define CHECK(cond, msg) do { if (!(cond)) fail(msg); } while (0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    CHECK(p != NULL, "malloc failed");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static void buf_reserve(test_buf *b, size_t add) {
    if (b->len + add + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 4096;
    while (cap < b->len + add + 1) cap *= 2;
    char *p = realloc(b->ptr, cap);
    CHECK(p != NULL, "realloc failed");
    b->ptr = p;
    b->cap = cap;
}

static void buf_append(test_buf *b, const char *s, size_t n) {
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_puts(test_buf *b, const char *s) {
    buf_append(b, s, strlen(s));
}

static int env_int(const char *name, int def, int min, int max) {
    const char *s = getenv(name);
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!end || *end || v < min || v > max) {
        fprintf(stderr, "ds4_kv_cache_benefit_test: ignoring invalid %s=%s\n",
                name, s);
        return def;
    }
    return (int)v;
}

static const char *model_path(void) {
    const char *path = getenv("DS4_TEST_MODEL");
    return path && path[0] ? path : "ds4flash.gguf";
}

static char *make_prompt_text(int lines) {
    test_buf b = {0};
    buf_puts(&b,
        "KV cache benchmark corpus. Every line below is deterministic and "
        "contains canary facts that should survive exact model-state restore.\n");
    for (int i = 0; i < lines; i++) {
        char line[256];
        snprintf(line, sizeof(line),
                 "Fact %04d: project=DS4 cache_test=enabled "
                 "canary=CANARY-BENCH-%04d checksum=%08x "
                 "instruction=preserve-prefix-state-without-refill.\n",
                 i, i, (unsigned)(i * 2654435761u));
        buf_puts(&b, line);
    }
    return b.ptr ? b.ptr : xstrdup("");
}

static void build_prompt(ds4_engine *engine, int target_tokens, int ctx,
                         ds4_tokens *prompt, int *lines_out) {
    int lines = 32;
    for (int attempt = 0; attempt < 18; attempt++) {
        char *text = make_prompt_text(lines);
        ds4_tokens_free(prompt);
        memset(prompt, 0, sizeof(*prompt));
        ds4_encode_chat_prompt(engine, "", text, DS4_THINK_NONE, prompt);
        free(text);

        if (prompt->len >= target_tokens && prompt->len + 256 < ctx) break;
        if (prompt->len + 256 >= ctx && lines > 8) {
            lines = (lines * 3) / 4;
            if (lines < 8) lines = 8;
        } else if (prompt->len < target_tokens) {
            lines *= 2;
        } else {
            break;
        }
    }
    CHECK(prompt->len > 256, "benchmark prompt too small");
    CHECK(prompt->len + 256 < ctx, "benchmark prompt does not fit context");
    if (lines_out) *lines_out = lines;
}

static char *temp_payload_path(void) {
    const char *base = getenv("TMPDIR");
    if (!base || !base[0]) base = "/tmp";
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "%s/ds4-kv-cache-benefit-%ld-XXXXXX",
             base, (long)getpid());
    int fd = mkstemp(tmpl);
    CHECK(fd >= 0, strerror(errno));
    close(fd);
    return xstrdup(tmpl);
}

static void progress_cb(void *ud, const char *event, int current, int total) {
    const char *label = ud ? (const char *)ud : "sync";
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 512 == 0) {
        fprintf(stderr, "ds4-kv-benefit: %s prefill %d/%d\n",
                label, current, total);
    }
}

static int logit_argmax(const float *x, int n) {
    int best = -1;
    float best_v = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) continue;
        if (best < 0 || x[i] > best_v) {
            best = i;
            best_v = x[i];
        }
    }
    return best;
}

static logit_cmp compare_logits(const float *a, const float *b, int n) {
    logit_cmp c = {0};
    c.top1_a = logit_argmax(a, n);
    c.top1_b = logit_argmax(b, n);
    c.same_top1 = c.top1_a >= 0 && c.top1_a == c.top1_b;
    double sumsq = 0.0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            c.nonfinite++;
            continue;
        }
        float d = b[i] - a[i];
        float ad = fabsf(d);
        if (ad > c.max_abs) c.max_abs = ad;
        sumsq += (double)d * (double)d;
    }
    c.rms = sqrt(sumsq / (double)n);
    return c;
}

static uint64_t file_size_or_die(const char *path) {
    struct stat st;
    CHECK(stat(path, &st) == 0, strerror(errno));
    CHECK(st.st_size >= 0, "negative file size");
    return (uint64_t)st.st_size;
}

int main(void) {
    const int ctx = env_int("DS4_KV_BENCH_CTX", 4096, 1024, 262144);
    const int target_tokens = env_int("DS4_KV_BENCH_TARGET_TOKENS",
                                      ctx / 2, 256, ctx - 512);

    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
    };
    CHECK(ds4_engine_open(&engine, &opt) == 0, "failed to open DS4 engine");
    const int vocab = ds4_engine_vocab_size(engine);
    CHECK(vocab > 0, "invalid vocab size");

    ds4_tokens prompt = {0};
    int prompt_lines = 0;
    build_prompt(engine, target_tokens, ctx, &prompt, &prompt_lines);

    ds4_session *base = NULL;
    CHECK(ds4_session_create(&base, engine, ctx) == 0, "failed to create base session");
    char err[256] = {0};
    ds4_session_set_progress(base, progress_cb, "base");
    double t0 = now_sec();
    CHECK(ds4_session_sync(base, &prompt, err, sizeof(err)) == 0,
          err[0] ? err : "base prefill failed");
    double base_sync_sec = now_sec() - t0;
    ds4_session_set_progress(base, NULL, NULL);
    CHECK(ds4_session_pos(base) == prompt.len, "base session token count mismatch");

    float *base_logits = xmalloc((size_t)vocab * sizeof(*base_logits));
    CHECK(ds4_session_copy_logits(base, base_logits, vocab) == vocab,
          "failed to copy base logits");

    uint64_t payload_bytes = ds4_session_payload_bytes(base);
    CHECK(payload_bytes > 0, "base session has no KV payload");
    char *payload_path = temp_payload_path();
    FILE *fp = fopen(payload_path, "wb");
    CHECK(fp != NULL, strerror(errno));
    t0 = now_sec();
    CHECK(ds4_session_save_payload(base, fp, err, sizeof(err)) == 0,
          err[0] ? err : "failed to save KV payload");
    CHECK(fclose(fp) == 0, "failed to close KV payload");
    double save_sec = now_sec() - t0;
    CHECK(file_size_or_die(payload_path) == payload_bytes,
          "payload byte count mismatch");

    ds4_session *restored = NULL;
    CHECK(ds4_session_create(&restored, engine, ctx) == 0,
          "failed to create restored session");
    fp = fopen(payload_path, "rb");
    CHECK(fp != NULL, strerror(errno));
    t0 = now_sec();
    CHECK(ds4_session_load_payload(restored, fp, payload_bytes,
                                   err, sizeof(err)) == 0,
          err[0] ? err : "failed to load KV payload");
    double load_sec = now_sec() - t0;
    fclose(fp);
    CHECK(ds4_session_pos(restored) == prompt.len,
          "restored session token count mismatch");

    float *loaded_logits = xmalloc((size_t)vocab * sizeof(*loaded_logits));
    CHECK(ds4_session_copy_logits(restored, loaded_logits, vocab) == vocab,
          "failed to copy loaded logits");
    logit_cmp base_cmp = compare_logits(base_logits, loaded_logits, vocab);
    CHECK(base_cmp.nonfinite == 0, "non-finite logits after KV load");
    CHECK(base_cmp.same_top1, "KV load changed top-1 token");
    CHECK(base_cmp.max_abs <= 1.0e-4f, "KV load changed base logits");

    ds4_tokens suffix = {0};
    ds4_tokenize_text(engine,
        "\n\nKV cache continuation probe: report CANARY-BENCH-0042 exactly once.",
        &suffix);
    CHECK(suffix.len > 0 && suffix.len < 128, "unexpected suffix token count");
    ds4_tokens extended = {0};
    ds4_tokens_copy(&extended, &prompt);
    for (int i = 0; i < suffix.len; i++) ds4_tokens_push(&extended, suffix.v[i]);
    CHECK(extended.len + 64 < ctx, "extended prompt does not fit context");

    int common = ds4_session_common_prefix(restored, &extended);
    int cached = common == ds4_session_pos(restored) &&
                 extended.len >= ds4_session_pos(restored) ? common : 0;
    int restored_prefill_tokens = extended.len - cached;
    CHECK(cached == prompt.len, "restored session did not retain prompt prefix");
    CHECK(restored_prefill_tokens == suffix.len,
          "restored session would prefill more than the suffix");

    ds4_session_set_progress(restored, progress_cb, "restored-suffix");
    t0 = now_sec();
    CHECK(ds4_session_sync(restored, &extended, err, sizeof(err)) == 0,
          err[0] ? err : "suffix sync failed");
    double suffix_sync_sec = now_sec() - t0;
    ds4_session_set_progress(restored, NULL, NULL);
    CHECK(ds4_session_pos(restored) == extended.len,
          "restored suffix token count mismatch");

    float *restored_suffix_logits = xmalloc((size_t)vocab * sizeof(*restored_suffix_logits));
    CHECK(ds4_session_copy_logits(restored, restored_suffix_logits, vocab) == vocab,
          "failed to copy restored suffix logits");

    ds4_session *full = NULL;
    CHECK(ds4_session_create(&full, engine, ctx) == 0,
          "failed to create full-prefill session");
    ds4_session_set_progress(full, progress_cb, "full");
    t0 = now_sec();
    CHECK(ds4_session_sync(full, &extended, err, sizeof(err)) == 0,
          err[0] ? err : "full prefill failed");
    double full_sync_sec = now_sec() - t0;
    ds4_session_set_progress(full, NULL, NULL);

    float *full_logits = xmalloc((size_t)vocab * sizeof(*full_logits));
    CHECK(ds4_session_copy_logits(full, full_logits, vocab) == vocab,
          "failed to copy full logits");
    logit_cmp extended_cmp = compare_logits(full_logits, restored_suffix_logits, vocab);
    CHECK(extended_cmp.nonfinite == 0, "non-finite logits after suffix sync");
    CHECK(extended_cmp.same_top1,
          "KV restore plus suffix changed top-1 versus full prefill");

    printf("kv-cache-benefit: prompt_lines=%d base_tokens=%d suffix_tokens=%d "
           "full_prefill_tokens=%d restored_prefill_tokens=%d saved_prefill_tokens=%d "
           "payload_bytes=%" PRIu64 " base_sync_sec=%.3f save_sec=%.3f "
           "load_sec=%.3f suffix_sync_sec=%.3f full_extended_sync_sec=%.3f "
           "base_top1_equal=%s base_max_abs=%g extended_top1_equal=%s "
           "extended_max_abs=%g extended_rms=%g quality_guard=logits_equivalence\n",
           prompt_lines, prompt.len, suffix.len,
           extended.len, restored_prefill_tokens,
           extended.len - restored_prefill_tokens,
           payload_bytes, base_sync_sec, save_sec,
           load_sec, suffix_sync_sec, full_sync_sec,
           base_cmp.same_top1 ? "true" : "false", base_cmp.max_abs,
           extended_cmp.same_top1 ? "true" : "false",
           extended_cmp.max_abs, extended_cmp.rms);

    unlink(payload_path);
    free(payload_path);
    free(full_logits);
    free(restored_suffix_logits);
    free(loaded_logits);
    free(base_logits);
    ds4_tokens_free(&extended);
    ds4_tokens_free(&suffix);
    ds4_tokens_free(&prompt);
    ds4_session_free(full);
    ds4_session_free(restored);
    ds4_session_free(base);
    ds4_engine_close(engine);
    return 0;
}
