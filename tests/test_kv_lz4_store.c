/* Exercise the real store/eviction/codec paths without loading a model. Only
 * the engine boundary is replaced: snapshots contain a deterministic payload. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
static int fail_rename, fail_alloc;
static int test_rename(const char *from, const char *to) {
    if (fail_rename) { errno = EACCES; return -1; }
    return rename(from, to);
}
static void *test_malloc(size_t n) {
    if (fail_alloc && n >= (8u << 20)) {
        fail_alloc = 0;
        errno = ENOMEM;
        return NULL;
    }
    return malloc(n);
}
#define rename test_rename
#define malloc test_malloc
#define ds4_engine_model_id test_model_id
#define ds4_engine_routed_quant_bits test_quant_bits
#define ds4_session_ctx test_ctx
#define ds4_session_tokens test_tokens
#define ds4_session_stage_payload test_stage
#define ds4_session_load_payload test_load
#define ds4_session_invalidate test_invalidate
#include "../ds4_kvstore.c"
#undef rename
#undef malloc
#include <assert.h>

static int token_ids[128];
static ds4_tokens tokens = {.v = token_ids, .len = 128, .cap = 128};
static int load_errno, load_calls, invalidated;
static bool random_payload;
static const uint64_t payload_size = 8u << 20;

int test_model_id(ds4_engine *e) { (void)e; return 1; }
int test_quant_bits(ds4_engine *e) { (void)e; return 2; }
int test_ctx(ds4_session *s) { (void)s; return 65536; }
const ds4_tokens *test_tokens(ds4_session *s) { (void)s; return &tokens; }
void test_invalidate(ds4_session *s) { (void)s; invalidated++; }
int test_stage(ds4_session *s, ds4_session_payload_file *out, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    char path[] = "/tmp/ds4-lz4-stage-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp);
    uint8_t buf[65536] = {0};
    uint64_t rng = 186;
    for (uint64_t off = 0; off < payload_size; off += sizeof(buf)) {
        if (random_payload) {
            for (size_t i = 0; i < sizeof(buf); i++) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                buf[i] = (uint8_t)(rng >> 33);
            }
        }
        assert(fwrite(buf, 1, sizeof(buf), fp) == sizeof(buf));
    }
    assert(fclose(fp) == 0);
    out->path = strdup(path);
    out->bytes = payload_size;
    return 0;
}
int test_load(ds4_session *s, FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    (void)s;
    load_calls++;
    if (load_errno) { errno = load_errno; snprintf(err, errlen, "injected resource failure"); return 1; }
    uint8_t buf[65536];
    while (bytes) {
        size_t n = bytes < sizeof(buf) ? (size_t)bytes : sizeof(buf);
        if (fread(buf, 1, n, fp) != n) return 1;
        bytes -= n;
    }
    return 0;
}

static char *entry_path(ds4_kvstore *kc, const char *text) {
    char sha[41];
    ds4_kvstore_sha1_bytes_hex(text, strlen(text), sha);
    return ds4_kvstore_path_for_sha(kc, sha);
}

/* A structurally valid existing file is enough to exercise eviction; loading
 * its model payload is deliberately outside this fixture's purpose. */
static char *seed_entry(ds4_kvstore *kc, const char *text, uint64_t size) {
    char *path = entry_path(kc, text);
    FILE *fp = fopen(path, "wb");
    assert(fp);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER], tb[4];
    ds4_kvstore_fill_header(h, 1, 2, DS4_KVSTORE_REASON_COLD, 0, 0, 0,
                            128, 100, 65536, 1, 1, size - sizeof(h) - 4 - strlen(text));
    ds4_kvstore_le_put32(tb, (uint32_t)strlen(text));
    assert(fwrite(h, 1, sizeof(h), fp) == sizeof(h));
    assert(fwrite(tb, 1, 4, fp) == 4);
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    assert(fseeko(fp, (off_t)size - 1, SEEK_SET) == 0 && fputc(0, fp) != EOF);
    assert(fclose(fp) == 0);
    return path;
}

static bool store(ds4_kvstore *kc, const ds4_kvstore_trailer_hooks *hooks) {
    char err[160] = {0};
    return ds4_kvstore_store_live_prefix_text(kc, NULL, NULL, &tokens, 128,
            "cold", "incoming checkpoint", 0, NULL, hooks, err, sizeof(err));
}

static bool fail_trailer(void *ud, FILE *fp, const char *text, uint64_t *bytes) {
    (void)ud; (void)fp; (void)text; (void)bytes;
    errno = ENOSPC;
    return false;
}

static void run_case(const char *name) {
    char dir[] = "/tmp/ds4-lz4-store-XXXXXX";
    assert(mkdtemp(dir));
    ds4_kvstore kc;
    ds4_kvstore_options opt = ds4_kvstore_default_options();
    opt.min_tokens = 1;
    opt.compression_threads = 1;
    assert(ds4_kvstore_open(&kc, dir, 10, false, opt, "test", NULL, NULL));
    char *old = seed_entry(&kc, "older checkpoint", 3u << 20);
    char *incoming = entry_path(&kc, "incoming checkpoint");
    if (!strcmp(name, "admission")) {
        kc.budget_bytes = 4u << 20;
        assert(store(&kc, NULL));
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) == 0);
    } else if (!strcmp(name, "eviction")) {
        assert(store(&kc, NULL));
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) == 0);
    } else if (!strcmp(name, "publish-failure")) {
        fail_rename = 1;
        assert(!store(&kc, NULL));
        fail_rename = 0;
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) != 0);
    } else if (!strcmp(name, "write-failure")) {
        ds4_kvstore_trailer_hooks hooks = {.write = fail_trailer};
        assert(!store(&kc, &hooks));
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) != 0);
    } else if (!strcmp(name, "raw-fallback-budget")) {
        kc.budget_bytes = 4u << 20;
        fail_alloc = 1;
        assert(!store(&kc, NULL));
        assert(fail_alloc == 0);
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) != 0);
    } else if (!strcmp(name, "expansion-budget")) {
        random_payload = true;
        kc.budget_bytes = payload_size + payload_size / 100 + 4096;
        assert(!store(&kc, NULL));
        random_payload = false;
        assert(access(old, F_OK) == 0 && access(incoming, F_OK) != 0);
    } else if (!strcmp(name, "protect-published")) {
        kc.opt.compression_threads = 0;
        assert(store(&kc, NULL));
        assert(access(old, F_OK) != 0 && access(incoming, F_OK) == 0);
    } else if (!strcmp(name, "replace-failure") || !strcmp(name, "replace-success")) {
        free(seed_entry(&kc, "incoming checkpoint", 3u << 20));
        FILE *fp = fopen(incoming, "r+b");
        assert(fp && fseek(fp, 7, SEEK_SET) == 0);
        assert(fputc(2, fp) == 2 && fclose(fp) == 0); /* Different model. */
        fail_rename = !strcmp(name, "replace-failure");
        assert(store(&kc, NULL) == !fail_rename);
        fp = fopen(incoming, "rb");
        assert(fp && fseek(fp, 7, SEEK_SET) == 0);
        assert(fgetc(fp) == (fail_rename ? 2 : 1));
        assert(fclose(fp) == 0 && access(old, F_OK) == 0);
        fail_rename = 0;
    } else {
        assert(store(&kc, NULL));
        if (!strcmp(name, "corrupt-recovery")) {
            FILE *fp = fopen(incoming, "r+b");
            assert(fp);
            uint8_t total[8];
            kv_le_put64(total, payload_size + DS4_KVSTORE_DEFAULT_CHUNK_BYTES);
            assert(fseeko(fp, 52 + strlen("incoming checkpoint"), SEEK_SET) == 0);
            assert(fwrite(total, 1, 8, fp) == 8 && fclose(fp) == 0);
        } else if (!strcmp(name, "load-oom")) {
            fail_alloc = 1;
        } else if (!strcmp(name, "engine-oom")) {
            load_errno = ENOMEM;
        } else if (!strcmp(name, "engine-io")) {
            load_errno = EIO;
        } else { assert(!"unknown case"); }
        kv_cache_refresh(&kc);
        load_calls = invalidated = 0;
        assert(ds4_kvstore_try_load_text(&kc, NULL, NULL, "incoming checkpoint", NULL, NULL, NULL, false) == 0);
        assert(invalidated == 1);
        if (!strcmp(name, "corrupt-recovery")) {
            assert(load_calls == 0 && access(incoming, F_OK) != 0);
            assert(store(&kc, NULL));
            kv_cache_refresh(&kc);
            assert(ds4_kvstore_try_load_text(&kc, NULL, NULL, "incoming checkpoint", NULL, NULL, NULL, false) == 128);
        } else {
            assert(access(incoming, F_OK) == 0);
        }
        fail_alloc = load_errno = 0;
        assert(ds4_kvstore_try_load_text(&kc, NULL, NULL, "incoming checkpoint",
                                        NULL, NULL, NULL, false) == 128);
    }
    DIR *dp = opendir(dir);
    assert(dp);
    struct dirent *de;
    while ((de = readdir(dp))) {
        if (de->d_name[0] == '.') continue;
        assert(strstr(de->d_name, ".tmp.") == NULL);
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);
        assert(unlink(path) == 0);
    }
    closedir(dp);
    ds4_kvstore_close(&kc);
    assert(rmdir(dir) == 0);
    free(old); free(incoming);
    printf("  %s: PASS\n", name);
}

int main(int argc, char **argv) {
    if (!KV_LZ4_HAVE_FWRAP) {
        puts("Store compression tests require cookie streams: SKIP");
        return 0;
    }
    if (argc == 2) { run_case(argv[1]); return 0; }
    const char *cases[] = {"admission", "eviction", "publish-failure", "write-failure",
        "raw-fallback-budget", "expansion-budget", "protect-published", "corrupt-recovery",
        "load-oom", "engine-oom", "engine-io", "replace-failure", "replace-success"};
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); i++) run_case(cases[i]);
    return 0;
}
