#include "../ds4_agent_context.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void test_fail(const char *msg) {
    fprintf(stderr, "ds4_agent_context_test: %s\n", msg);
    exit(1);
}

#define CHECK(cond, msg) do { if (!(cond)) test_fail(msg); } while (0)

static char *test_strdup(const char *s) {
    size_t n = strlen(s);
    char *out = malloc(n + 1);
    CHECK(out != NULL, "malloc failed");
    memcpy(out, s, n + 1);
    return out;
}

static char *make_temp_dir(void) {
    const char *base = getenv("TMPDIR");
    if (!base || !base[0]) base = "/tmp";
    for (int i = 0; i < 100; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/ds4-agent-context-test-%ld-%ld-%d",
                 base, (long)getpid(), (long)time(NULL), i);
        if (mkdir(path, 0700) == 0) return test_strdup(path);
        if (errno != EEXIST) break;
    }
    test_fail("failed to create temp dir");
    return NULL;
}

static void write_raw_file(const char *path, const char *id, const char *kv_path) {
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL, "failed to open raw metadata");
    fprintf(fp,
            "{\n"
            "  \"id\": \"%s\",\n"
            "  \"label\": \"unsafe\",\n"
            "  \"created_at\": 1,\n"
            "  \"world_epoch\": 1,\n"
            "  \"transcript_tokens\": 10,\n"
            "  \"kv_path\": \"%s\",\n"
            "  \"memory_path\": \"%s.memory.md\"\n"
            "}\n",
            id, kv_path, id);
    CHECK(fclose(fp) == 0, "failed to write raw metadata");
}

static void write_key_collision_file(const char *path, const char *id) {
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL, "failed to open key collision metadata");
    fprintf(fp,
            "{\n"
            "  \"id\": \"%s\",\n"
            "  \"label\": \"label mentions \\\"world_epoch\\\": 99 before the real key\",\n"
            "  \"created_at\": 7,\n"
            "  \"world_epoch\": 42,\n"
            "  \"transcript_tokens\": 77,\n"
            "  \"kv_path\": \"%s.kv\",\n"
            "  \"memory_path\": \"%s.memory.md\"\n"
            "}\n",
            id, id, id);
    CHECK(fclose(fp) == 0, "failed to write key collision metadata");
}

static void fill_meta(ds4_agent_context_meta *m, const char *id,
                      const char *label, uint64_t epoch, int tokens) {
    snprintf(m->id, sizeof(m->id), "%s", id);
    m->label = test_strdup(label);
    m->kv_file = ds4_agent_context_file_name(m->id, ".kv");
    m->memory_file = ds4_agent_context_file_name(m->id, ".memory.md");
    m->created_at = 1234;
    m->world_epoch = epoch;
    m->transcript_tokens = tokens;
}

int main(void) {
    static const char id1[] = "1111111111111111111111111111111111111111";
    static const char id2[] = "2222222222222222222222222222222222222222";
    static const char id3[] = "3333333333333333333333333333333333333333";
    static const char id4[] = "4444444444444444444444444444444444444444";
    static const char id5[] = "5555555555555555555555555555555555555555";
    char err[256] = {0};
    char *dir = make_temp_dir();

    char *meta1_name = ds4_agent_context_file_name(id1, ".meta.json");
    char *meta2_name = ds4_agent_context_file_name(id2, ".meta.json");
    char *meta1_path = ds4_agent_context_path_for_file(dir, meta1_name);
    char *meta2_path = ds4_agent_context_path_for_file(dir, meta2_name);

    ds4_agent_context_meta m1 = {0};
    fill_meta(&m1, id1, "first \"checkpoint\"\\line\nnext", 4, 101);
    CHECK(ds4_agent_context_write_meta(&m1, meta1_path, err, sizeof(err)),
          "failed to write first metadata");
    ds4_agent_context_meta_free(&m1);

    ds4_agent_context_meta m2 = {0};
    fill_meta(&m2, id2, "second", 12, 202);
    CHECK(ds4_agent_context_write_meta(&m2, meta2_path, err, sizeof(err)),
          "failed to write second metadata");
    ds4_agent_context_meta_free(&m2);

    ds4_agent_context_meta read_back = {0};
    CHECK(ds4_agent_context_read_meta_file(meta1_path, &read_back, err, sizeof(err)),
          "failed to read metadata roundtrip");
    CHECK(strcmp(read_back.id, id1) == 0, "roundtrip id mismatch");
    CHECK(strcmp(read_back.label, "first \"checkpoint\"\\line\nnext") == 0,
          "roundtrip label mismatch");
    CHECK(read_back.world_epoch == 4, "roundtrip epoch mismatch");
    CHECK(read_back.transcript_tokens == 101, "roundtrip tokens mismatch");
    ds4_agent_context_meta_free(&read_back);

    CHECK(ds4_agent_context_count_checkpoints(dir) == 2, "checkpoint count mismatch");
    CHECK(ds4_agent_context_max_world_epoch(dir) == 12, "max world epoch mismatch");
    CHECK(ds4_agent_context_file_component_safe("safe.kv"),
          "safe component rejected");
    CHECK(!ds4_agent_context_file_component_safe("."),
          "dot component should be rejected");
    CHECK(!ds4_agent_context_file_component_safe(".."),
          "dotdot component should be rejected");
    CHECK(ds4_agent_context_meta_filename(meta1_name),
          "valid metadata filename rejected");
    CHECK(!ds4_agent_context_meta_filename("junk.meta.json"),
          "invalid metadata filename accepted");

    ds4_agent_context_meta found = {0};
    char *found_meta_path = NULL;
    char *found_kv_path = NULL;
    CHECK(ds4_agent_context_find_checkpoint(dir, "2222", &found,
                                            &found_meta_path, &found_kv_path,
                                            err, sizeof(err)),
          "failed to find checkpoint by prefix");
    CHECK(strcmp(found.id, id2) == 0, "found checkpoint id mismatch");
    CHECK(strstr(found_kv_path, "2222222222222222222222222222222222222222.kv") != NULL,
          "found kv path mismatch");
    ds4_agent_context_meta_free(&found);
    free(found_meta_path);
    free(found_kv_path);

    char *unsafe_name = ds4_agent_context_file_name(id3, ".meta.json");
    char *unsafe_path = ds4_agent_context_path_for_file(dir, unsafe_name);
    write_raw_file(unsafe_path, id3, "../escape.kv");
    ds4_agent_context_meta unsafe = {0};
    CHECK(!ds4_agent_context_read_meta_file(unsafe_path, &unsafe, err, sizeof(err)),
          "unsafe metadata path should be rejected");
    ds4_agent_context_meta_free(&unsafe);

    char *dotdot_name = ds4_agent_context_file_name(id5, ".meta.json");
    char *dotdot_path = ds4_agent_context_path_for_file(dir, dotdot_name);
    write_raw_file(dotdot_path, id5, "..");
    ds4_agent_context_meta dotdot = {0};
    CHECK(!ds4_agent_context_read_meta_file(dotdot_path, &dotdot, err, sizeof(err)),
          "dotdot metadata path should be rejected");
    ds4_agent_context_meta_free(&dotdot);

    char *collision_name = ds4_agent_context_file_name(id4, ".meta.json");
    char *collision_path = ds4_agent_context_path_for_file(dir, collision_name);
    write_key_collision_file(collision_path, id4);
    ds4_agent_context_meta collision = {0};
    CHECK(ds4_agent_context_read_meta_file(collision_path, &collision,
                                           err, sizeof(err)),
          "key collision metadata should parse");
    CHECK(collision.world_epoch == 42,
          "parser matched key text inside a string value");
    CHECK(collision.transcript_tokens == 77,
          "key collision transcript tokens mismatch");
    ds4_agent_context_meta_free(&collision);

    CHECK(ds4_agent_context_restore_epoch_guard(12, 12, false, err, sizeof(err)),
          "equal epoch restore should be allowed");
    CHECK(ds4_agent_context_restore_epoch_guard(13, 12, true, err, sizeof(err)),
          "explicit side-effect override should be allowed");
    CHECK(!ds4_agent_context_restore_epoch_guard(13, 12, false, err, sizeof(err)),
          "epoch mismatch restore should be rejected");
    CHECK(strstr(err, "world_epoch=13 to 12") != NULL,
          "epoch guard error missing epoch details");
    CHECK(ds4_agent_context_no_running_bash_guard("restore", 0, err, sizeof(err)),
          "restore should allow no running bash jobs");
    CHECK(!ds4_agent_context_no_running_bash_guard("restore", 2, err, sizeof(err)),
          "restore should reject running bash jobs");
    CHECK(strstr(err, "2 bash job(s)") != NULL,
          "bash guard error missing job count");

    ds4_agent_context_restore_metrics metrics = {
        .checkpoint_tokens = 101,
        .restore_notice_tokens = 13,
        .restored_tokens = 114,
    };
    char *metrics_line = ds4_agent_context_restore_expected_metrics_line(&metrics);
    CHECK(strstr(metrics_line, "KV restore expected metrics:") != NULL,
          "restore metrics line missing expected label");
    CHECK(strstr(metrics_line, "checkpoint_tokens=101") != NULL,
          "restore metrics line missing checkpoint tokens");
    CHECK(strstr(metrics_line, "expected_restore_notice_tokens=13") != NULL,
          "restore metrics line missing notice tokens");
    CHECK(strstr(metrics_line, "expected_prefill_suffix_tokens=13") != NULL,
          "restore metrics line missing expected prefill suffix");
    CHECK(strstr(metrics_line, "expected_saved_prefill_tokens=101") != NULL,
          "restore metrics line missing expected saved prefill");
    CHECK(strstr(metrics_line, " saved_prefill_tokens=") == NULL,
          "restore metrics line should not present expected values as actual");
    free(metrics_line);

    ds4_agent_side_effects effects = {0};
    uint64_t epoch = 3;
    epoch = ds4_agent_side_effects_note(&effects, epoch,
                                        "write", "experiment.md\nsecond line");
    CHECK(epoch == 4, "side effect epoch mismatch");
    char *summary = ds4_agent_side_effects_summary_since(&effects, 3);
    CHECK(strstr(summary, "Known side effects after checkpoint:") != NULL,
          "side effect summary header missing");
    CHECK(strstr(summary, "epoch=4 write experiment.md second line") != NULL,
          "side effect summary content missing");
    free(summary);
    summary = ds4_agent_side_effects_summary_since(&effects, 4);
    CHECK(strcmp(summary, "") == 0, "empty side effect summary mismatch");
    free(summary);
    ds4_agent_side_effects_free(&effects);

    for (int i = 0; i < 70; i++) {
        char detail[32];
        snprintf(detail, sizeof(detail), "effect-%d", i + 1);
        epoch = ds4_agent_side_effects_note(&effects, epoch, "bash", detail);
    }
    CHECK(effects.count == 64, "side effect retained count mismatch");
    CHECK(effects.evicted_count == 6, "side effect evicted count mismatch");
    summary = ds4_agent_side_effects_summary_since(&effects, 4);
    CHECK(strstr(summary, "may be incomplete") != NULL,
          "truncated side effect warning missing");
    CHECK(strstr(summary, "6 older side effect") != NULL,
          "truncated side effect count missing");
    CHECK(strstr(summary, "... more side effects omitted ...") != NULL,
          "retained side effect omission marker missing");
    free(summary);
    ds4_agent_side_effects_free(&effects);

    unlink(meta1_path);
    unlink(meta2_path);
    unlink(unsafe_path);
    unlink(dotdot_path);
    unlink(collision_path);
    rmdir(dir);
    free(meta1_name);
    free(meta2_name);
    free(unsafe_name);
    free(dotdot_name);
    free(collision_name);
    free(meta1_path);
    free(meta2_path);
    free(unsafe_path);
    free(dotdot_path);
    free(collision_path);
    free(dir);
    return 0;
}
