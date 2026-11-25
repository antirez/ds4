#define _POSIX_C_SOURCE 200809L
#include "../ds4.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Run save and restore as separate processes with the same model and paths. */
static void check(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "test_k3_session_resume: %s\n", message);
        exit(1);
    }
}

static double seconds(void) {
    struct timespec ts;
    check(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "clock_gettime");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t file_bytes(FILE *file) {
    struct stat st;
    check(fstat(fileno(file), &st) == 0 && st.st_size > 0, "payload size");
    return (uint64_t)st.st_size;
}

static void trace(FILE *file, ds4_session *s, int vocab, int save) {
    size_t bytes = (size_t)vocab * sizeof(float);
    float *row = malloc(bytes);
    check(row && ds4_session_copy_logits(s, row, vocab) == vocab, "read logits");
    for (int i = 0; i < vocab; i++) check(isfinite(row[i]), "non-finite logit");
    if (save) {
        check(fwrite(row, 1, bytes, file) == bytes, "write reference logits");
    } else {
        float *reference = malloc(bytes);
        check(reference && fread(reference, 1, bytes, file) == bytes, "read reference logits");
        check(memcmp(reference, row, bytes) == 0, "restored logits differ from uninterrupted run");
        free(reference);
    }
    free(row);
}

static void check_reuse(ds4_session *s, FILE *payload, uint64_t bytes, int tokens) {
    unsigned char header[24];
    rewind(payload);
    check(fread(header, 1, sizeof(header), payload) == sizeof(header), "read version");
    int native = header[4] == 2;
    rewind(payload);
    check(ds4_session_cached_payload_current(s, payload, bytes, false) == native,
          "cache version migration");
    if (!native) {
        rewind(payload);
        check(ds4_session_cached_payload_current(s, payload, bytes, true),
              "compact disk cache did not retain the legacy checkpoint");
    }
    if (native) {
        /* This check only reads the header, tokens and identity. */
        size_t prefix = sizeof(header) + (size_t)tokens * 4;
        size_t length = prefix + 7 * sizeof(uint64_t);
        unsigned char *data = malloc(length);
        FILE *wrong_model = tmpfile();
        rewind(payload);
        check(wrong_model && data && fread(data, 1, length, payload) == length,
              "read model identity");
        data[prefix] ^= 1;
        check(fwrite(data, 1, length, wrong_model) == length, "write other identity");
        rewind(wrong_model);
        check(!ds4_session_cached_payload_current(s, wrong_model, bytes, false),
              "cache from a different model accepted");
        free(data);
        check(fclose(wrong_model) == 0, "close identity fixture");
    }
}

int main(int argc, char **argv) {
    check(argc == 5 || argc == 6, "usage: save|restore MODEL.gguf PAYLOAD LOGITS [DAMAGED_PAYLOAD]");
    const int save = strcmp(argv[1], "save") == 0;
    check((save && argc == 5) || (!save && strcmp(argv[1], "restore") == 0), "invalid mode");
    const double begin = seconds();
    ds4_engine_options opt = {
        .model_path = argv[2], .backend = DS4_BACKEND_METAL,
        .context_size = 1024, .prefill_chunk = 64, .power_percent = 100,
        .ssd_streaming = true, .ssd_streaming_cache_bytes = UINT64_C(10) << 30,
    };
    ds4_engine *engine = NULL;
    ds4_session *s = NULL;
    check(ds4_engine_open(&engine, &opt) == 0 && ds4_engine_is_k3(engine), "open K3 model");
    check(ds4_session_create(&s, engine, opt.context_size) == 0, "create session");
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, "You are a helpful assistant", "Ciao, come stai?",
                           DS4_THINK_NONE, &prompt);
    const int vocab = ds4_engine_vocab_size(engine);
    char err[512] = {0};
    FILE *payload = fopen(argv[3], save ? "wb" : "rb");
    FILE *reference = fopen(argv[4], save ? "wb" : "rb");
    check(payload && reference, "open test files");
    double start = seconds();
    uint64_t bytes;
    if (save) {
        check(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, err);
        printf("prefill_s=%.6f ready_s=%.6f\n", seconds() - start, seconds() - begin);
        start = seconds();
        bytes = ds4_session_payload_bytes(s);
        check(ds4_session_save_payload(s, payload, err, sizeof(err)) == 0, err);
        check(fflush(payload) == 0 && fsync(fileno(payload)) == 0, "flush native payload");
        check(file_bytes(payload) == bytes && bytes > 24 + (uint64_t)prompt.len * 4,
              "expected native session state");
        printf("save_fsync_s=%.6f\n", seconds() - start);
    } else {
        bytes = file_bytes(payload);
        check(ds4_session_load_payload(s, payload, bytes, err, sizeof(err)) == 0, err);
        printf("restore_s=%.6f ready_s=%.6f\n", seconds() - start, seconds() - begin);
        check_reuse(s, payload, bytes, prompt.len);
        if (argc == 6) {
            FILE *bad = fopen(argv[5], "rb");
            check(bad != NULL && file_bytes(bad) == bytes, "damaged payload size");
            check(ds4_session_load_payload(s, bad, bytes, err, sizeof(err)) != 0,
                  "damaged payload accepted");
            check(fclose(bad) == 0, "close damaged payload");
            rewind(payload);
            check(ds4_session_load_payload(s, payload, bytes - 1, err, sizeof(err)) != 0,
                  "truncated payload accepted");
        }
    }
    check(fclose(payload) == 0, "close payload");
    check(ds4_session_pos(s) == prompt.len && ds4_session_common_prefix(s, &prompt) == prompt.len,
          "restored prompt differs");
    trace(reference, s, vocab, save);
    int token = ds4_session_argmax_ignoring_eos(s, DS4_THINK_NONE);
    check(token >= 0 && token < vocab, "invalid greedy token");
    printf("prompt_tokens=%d vocab=%d bytes=%" PRIu64 " greedy=%d\n", prompt.len, vocab, bytes, token);
    fflush(stdout);
    start = seconds();
    check(ds4_session_eval(s, token, err, sizeof(err)) == 0, err);
    printf("decode_s=%.6f\n", seconds() - start);
    check(ds4_session_pos(s) == prompt.len + 1, "position after decode");
    trace(reference, s, vocab, save);
    if (!save) check(fgetc(reference) == EOF && !ferror(reference), "trailing reference logits");
    check(fclose(reference) == 0, "close logits");
    ds4_tokens_free(&prompt);
    ds4_session_free(s);
    ds4_engine_close(engine);
    puts("ok");
    return 0;
}
