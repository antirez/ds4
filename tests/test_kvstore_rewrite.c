/* Exercise the real private trailer writer without widening the KV API.
 * No engine, weights or GPU initialization is used by this test. */
#include "../ds4_kvstore.c"
#include <assert.h>

static int trailer_writes;

static bool trailer_size(void *ud, const char *text, uint64_t *bytes) {
    (void)ud; (void)text;
    *bytes = 8;
    return true;
}

static bool trailer_write(void *ud, FILE *fp, const char *text, uint64_t *bytes) {
    (void)ud; (void)text;
    trailer_writes++;
    *bytes = 8;
    return fwrite("KTM-test", 1, 8, fp) == 8;
}

static void write_fixture(const char *path, uint8_t version, uint8_t flags,
                          uint8_t abi) {
    uint8_t bytes[68] = {0};
    ds4_kvstore_fill_header(bytes, 5, 2,
        flags ? DS4_KVSTORE_REASON_AGENT_SESSION : DS4_KVSTORE_REASON_COLD,
        flags, 512, 0, 32768, 100, 100, 4);
    bytes[3] = version;
    bytes[20] = abi;
    /* Payload and Agent trailer are opaque here; only v1 may be rewritten. */
    memcpy(bytes + 52, "KV!!private-tail", 16);
    FILE *fp = fopen(path, "wb");
    assert(fp);
    assert(fwrite(bytes, 1, sizeof(bytes), fp) == sizeof(bytes));
    assert(fclose(fp) == 0);
}

static void assert_rewrite_ignored(const char *path,
                                    const ds4_kvstore_trailer_hooks *hooks) {
    uint8_t before[68], after[68];
    FILE *fp = fopen(path, "rb");
    assert(fp);
    assert(fread(before, 1, sizeof(before), fp) == sizeof(before));
    assert(fclose(fp) == 0);
    ds4_kvstore cache = {0};
    kv_cache_rewrite_trailer(&cache, path, "", hooks);
    assert(trailer_writes == 0);
    fp = fopen(path, "rb");
    assert(fp);
    assert(fread(after, 1, sizeof(after), fp) == sizeof(after));
    assert(fgetc(fp) == EOF);
    assert(fclose(fp) == 0);
    assert(memcmp(before, after, sizeof(before)) == 0);
}

int main(void) {
    char path[] = "/tmp/ds4-kv-rewrite-test.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    assert(close(fd) == 0);
    ds4_kvstore_trailer_hooks hooks = {
        .ext_flag = DS4_KVSTORE_EXT_TOOL_MAP,
        .serialized_size = trailer_size, .write = trailer_write,
    };
    const uint8_t input = DS4_KVSTORE_EXT_SESSION_TITLE | DS4_KVSTORE_EXT_AGENT_INPUT;
    write_fixture(path, DS4_KVSTORE_AGENT_VERSION, input, 2);
    assert_rewrite_ignored(path, &hooks);
    write_fixture(path, 1, input, 2); /* A visual file mislabeled as v1. */
    assert_rewrite_ignored(path, &hooks);
    write_fixture(path, 1, 0, 0); /* Obsolete graph ABI. */
    assert_rewrite_ignored(path, &hooks);

    write_fixture(path, 1, 0, 2);
    ds4_kvstore cache = {0};
    kv_cache_rewrite_trailer(&cache, path, "", &hooks);
    assert(trailer_writes == 1);
    FILE *fp = fopen(path, "rb");
    assert(fp);
    ds4_kvstore_entry entry = {0};
    uint32_t text_bytes = 0;
    assert(ds4_kvstore_read_header(fp, &entry, &text_bytes));
    assert(entry.ext_flags == DS4_KVSTORE_EXT_TOOL_MAP && entry.payload_bytes == 4);
    assert(fseeko(fp, entry.payload_bytes, SEEK_CUR) == 0);
    char trailer[8];
    assert(fread(trailer, 1, sizeof(trailer), fp) == sizeof(trailer));
    assert(memcmp(trailer, "KTM-test", sizeof(trailer)) == 0);
    assert(fgetc(fp) == EOF);
    assert(fclose(fp) == 0);
    assert(unlink(path) == 0);
    puts("KV trailer rewrite: PASS (v2/forged v1/bad ABI untouched, v1 rewritten; no model)");
    return 0;
}
