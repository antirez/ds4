/* Optional real Qwen Metal checkpoint oracle; not part of default make test.
 *
 * Build: make tests/test_qwen4_prompt_checkpoint
 * Run: ./tests/test_qwen4_prompt_checkpoint MODEL PROMPT PREFIX APPEND NEW_DIR
 * Use tests/long_context_story_prompt.txt as PROMPT, PREFIX=16384 or 131072,
 * and APPEND=1, 8 or 8192. Run the six combinations serially with no server or
 * other model process. Native ctx262144/chunk8192, one active session at a time.
 * The native Q4 matrix requires ample memory (128 GiB host used locally).
 * NEW_DIR must not exist; retain five payload files plus the isolated cache.
 * Allow room for several full states and temporary payload staging on disk.
 *
 * This compares valid serialized state and full logits at equal computational
 * boundaries, including a nonzero-temperature speculative-API history frozen in-process.
 * MTP capability is required; draft acceptance or cycle counts are not asserted.
 * The opaque visible key/trailer tests storage transport; it does not test the
 * HTTP renderer/tool-ID map or equivalence to ordinary sampled generation.
 */
#include "ds4.h"
#include "ds4_kvstore.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void check(int ok, const char *why) {
    if (!ok) {
        fprintf(stderr, "Qwen prompt checkpoint failed: %s\n", why);
        exit(1);
    }
}

static int number(const char *text) {
    char *end;
    long n = strtol(text, &end, 10);
    check(text[0] && !*end && n >= 0 && n <= 131072, "integer argument");
    return (int)n;
}

static void path_join(char *out, size_t cap, const char *dir, const char *name) {
    int n = snprintf(out, cap, "%s/%s", dir, name);
    check(n >= 0 && (size_t)n < cap, "output path length");
}

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    check(fp != NULL, "prompt open");
    check(!fseek(fp, 0, SEEK_END), "prompt seek");
    long n = ftell(fp);
    check(n > 0 && !fseek(fp, 0, SEEK_SET), "prompt size");
    char *text = malloc((size_t)n + 1);
    check(text != NULL, "prompt allocation");
    check(fread(text, 1, (size_t)n, fp) == (size_t)n, "prompt read");
    text[n] = 0;
    check(!fclose(fp), "prompt close");
    return text;
}

static void logits(ds4_session *s, float *out, int vocab) {
    check(ds4_session_copy_logits(s, out, vocab) == vocab, "logits copy");
    for (int i = 0; i < vocab; i++) check(isfinite(out[i]), "finite logits");
}

static void save_payload(ds4_session *s, const char *path) {
    FILE *fp = fopen(path, "wbx");
    check(fp != NULL, "fresh payload file");
    char err[256] = {0};
    check(!ds4_session_save_payload(s, fp, err, sizeof(err)), err);
    check(!fclose(fp), "payload close");
}

static void equal_payloads(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    check(fa && fb, "payload compare open");
    unsigned char x[65536], y[65536];
    for (;;) {
        size_t nx = fread(x, 1, sizeof(x), fa), ny = fread(y, 1, sizeof(y), fb);
        check(nx == ny && !memcmp(x, y, nx), "valid serialized payload changed");
        if (nx < sizeof(x)) {
            check(!ferror(fa) && !ferror(fb), "payload compare read");
            break;
        }
    }
    check(!fclose(fa) && !fclose(fb), "payload compare close");
}

static bool trailer_size(void *ud, const char *text, uint64_t *bytes) {
    (void)ud; (void)text;
    *bytes = 8;
    return true;
}

static bool trailer_write(void *ud, FILE *fp, const char *text, uint64_t *bytes) {
    (void)ud; (void)text;
    *bytes = 8;
    return fwrite("QWENMAP1", 1, 8, fp) == 8;
}

static int trailer_load(void *ud, FILE *fp, const void *wanted) {
    (void)wanted;
    char bytes[8];
    check(fread(bytes, 1, 8, fp) == 8 && !memcmp(bytes, "QWENMAP1", 8), "trailer roundtrip");
    (*(int *)ud)++;
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL PROMPT PREFIX APPEND(1|8|8192) NEW_DIR\n", argv[0]);
        return 2;
    }
    int prefix = number(argv[3]), append = number(argv[4]);
    check(prefix >= 16384 && prefix <= 131072 &&
          (append == 1 || append == 8 || append == 8192), "arguments");
    check(!mkdir(argv[5], 0700), "new output directory only");
    char cache[4096], before[4096], after[4096], expected[4096], restored[4096], loaded_path[4096];
    path_join(cache, sizeof(cache), argv[5], "kv");
    path_join(before, sizeof(before), argv[5], "before.payload");
    path_join(after, sizeof(after), argv[5], "after.payload");
    path_join(expected, sizeof(expected), argv[5], "expected.payload");
    path_join(restored, sizeof(restored), argv[5], "restored.payload");
    path_join(loaded_path, sizeof(loaded_path), argv[5], "loaded.payload");
    ds4_engine_options options = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL, .n_threads = 1,
        .context_size = 262144, .prefill_chunk = 8192, .warm_weights = true,
        .glm_mtp = true, .dspark_exact_sampling = false,
    };
    ds4_engine *engine = NULL;
    check(!ds4_engine_open(&engine, &options), "engine open");
    check(ds4_engine_is_qwen4(engine), "Qwen model");
    check(ds4_engine_mtp_draft_tokens(engine) > 1, "Qwen model with MTP support");
    char *text = read_text(argv[2]);
    ds4_tokens fixture = {0}, prompt = {0};
    ds4_tokenize_text(engine, text, &fixture);
    free(text);
    check(fixture.len > 0, "nonempty fixture");
    for (int i = 0; i < prefix; i++) ds4_tokens_push(&prompt, fixture.v[i % fixture.len]);
    ds4_session *session = NULL;
    char err[256] = {0};
    check(!ds4_session_create(&session, engine, 262144), "session create");
    check(!ds4_session_sync(session, &prompt, err, sizeof(err)), err);
    uint64_t rng = 12345;
    int generated = 0;
    while (generated < 32) {
        int token = ds4_session_sample(session, 0.6f, 0, 1.0f, 0.05f, &rng);
        check(token >= 0, "sample");
        int ids[4];
        int n = ds4_session_eval_speculative(session, token, 32 - generated, -1,
                  0.6f, 0, 1.0f, 0.05f, &rng, ids, 4, err, sizeof(err));
        check(n > 0 && n <= 4 && n <= 32 - generated, "speculative history count");
        generated += n;
    }
    ds4_tokens frontier = {0};
    ds4_tokens_copy(&frontier, ds4_session_tokens(session));
    check(frontier.len == prefix + 32, "history frontier length");
    int vocab = ds4_engine_vocab_size(engine);
    size_t row_bytes = (size_t)vocab * sizeof(float);
    float *start = malloc(row_bytes), *a = malloc(row_bytes);
    float *b = malloc(row_bytes), *next_row = malloc(row_bytes);
    check(start && a && b && next_row, "row allocation");
    logits(session, start, vocab);
    save_payload(session, before);
    ds4_kvstore store = {0};
    check(ds4_kvstore_open(&store, cache, 256000, false,
          ds4_kvstore_default_options(), "qwen-checkpoint-test", NULL, NULL), "cache open");
    int trailer_seen = 0;
    ds4_kvstore_trailer_hooks hooks = {
        .ud = &trailer_seen, .ext_flag = DS4_KVSTORE_EXT_TOOL_MAP,
        .serialized_size = trailer_size, .write = trailer_write, .load = trailer_load,
    };
    const char *key = "Qwen synthetic visible prefix";
    check(ds4_kvstore_store_live_prefix_text(&store, engine, session,
          &frontier, frontier.len, "near-tail", key,
          DS4_KVSTORE_EXT_RESPONSES_VISIBLE, "responses-prompt", &hooks, err, sizeof(err)), err);
    logits(session, b, vocab);
    check(!memcmp(start, b, row_bytes), "save changed logits");
    check(ds4_tokens_starts_with(ds4_session_tokens(session), &frontier) &&
          ds4_session_tokens(session)->len == frontier.len, "save changed tokens");
    save_payload(session, after);
    equal_payloads(before, after);
    ds4_tokens extended = {0};
    ds4_tokens_copy(&extended, &frontier);
    for (int i = 0; i < append; i++) ds4_tokens_push(&extended, fixture.v[i % fixture.len]);
    check(!ds4_session_sync(session, &extended, err, sizeof(err)), err);
    logits(session, a, vocab);
    int next = ds4_session_argmax(session);
    check(next >= 0 && !ds4_session_eval(session, next, err, sizeof(err)), err);
    logits(session, next_row, vocab);
    save_payload(session, expected);
    ds4_session_free(session);
    session = NULL;
    check(!ds4_session_create(&session, engine, 262144), "restored session");
    ds4_tokens effective = {0};
    ds4_kvstore_load_result loaded = {0};
    char incoming[128];
    snprintf(incoming, sizeof(incoming), "%s suffix", key);
    check(ds4_kvstore_try_load_text(&store, engine, session, incoming,
          &effective, &loaded, &hooks, true) == frontier.len, "visible key load");
    check((loaded.ext_flags & DS4_KVSTORE_EXT_RESPONSES_VISIBLE) &&
          trailer_seen == 1, "visible flag/trailer");
    check(ds4_tokens_starts_with(&effective, &frontier) &&
          effective.len > frontier.len, "exact tokens plus suffix");
    logits(session, b, vocab);
    check(!memcmp(start, b, row_bytes), "restored frontier logits");
    check(ds4_tokens_starts_with(ds4_session_tokens(session), &frontier) &&
          ds4_session_tokens(session)->len == frontier.len, "restored frontier tokens");
    save_payload(session, loaded_path);
    equal_payloads(before, loaded_path);
    check(!ds4_session_sync(session, &extended, err, sizeof(err)), err);
    logits(session, b, vocab);
    check(!memcmp(a, b, row_bytes), "same-boundary append logits");
    check(!ds4_session_eval(session, next, err, sizeof(err)), err);
    logits(session, b, vocab);
    check(!memcmp(next_row, b, row_bytes), "next decode logits");
    save_payload(session, restored);
    equal_payloads(expected, restored);
    printf("Qwen checkpoint PASS model=%s prompt=%s prefix=%d history=%d frontier=%d append=%d next=%d vocab=%d\n",
           argv[1], argv[2], prefix, generated, frontier.len, append, next, vocab);
    ds4_kvstore_load_result_free(&loaded);
    ds4_tokens_free(&effective); ds4_tokens_free(&extended); ds4_tokens_free(&frontier);
    ds4_tokens_free(&fixture); ds4_tokens_free(&prompt);
    free(start); free(a); free(b); free(next_row);
    ds4_kvstore_close(&store);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
