/* Internal loader fixture: links Metal support but never initializes/runs it. */
#include "../k3_infer.c"
#include <sys/resource.h>
#include <sys/wait.h>
#include <assert.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static void write_all(int fd, const void *p_, size_t bytes, uint64_t offset) {
    const unsigned char *p = p_;
    while (bytes) {
        ssize_t n = pwrite(fd, p, bytes, (off_t)offset);
        assert(n > 0);
        p += n; bytes -= (size_t)n; offset += (uint64_t)n;
    }
}


static unsigned char large_pattern(uint64_t offset) {
    return (unsigned char)((offset * 131u + offset / 251u +
                             (offset >> 20u) * 17u + 29u) & 255u);
}

static void test_large_multiworker(void) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t bytes = UINT64_C(67) * 1024u * 1024u + 111u;
    const uint64_t payload = UINT64_C(4294967296) + 17u + bytes;
    char path[] = "/tmp/k3-stage-large-fixture-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    char header[256];
    int n = snprintf(header, sizeof(header),
        "{\"data\":{\"dtype\":\"U8\",\"shape\":[%llu],\"data_offsets\":[0,%llu]}}",
        (unsigned long long)payload, (unsigned long long)payload);
    assert(n > 0 && n < (int)sizeof(header));
    const uint64_t header_bytes = (uint64_t)n;
    write_all(fd, &header_bytes, 8, 0);
    write_all(fd, header, (size_t)n, 8);
    const uint64_t offset = 8u + header_bytes + UINT64_C(4294967296) + 17u;
    assert(ftruncate(fd, (off_t)(offset + bytes)) == 0);
    unsigned char block[65536];
    for (uint64_t begin=0; begin<bytes; begin+=sizeof(block)) {
        const uint64_t count = bytes-begin < sizeof(block)
            ? bytes-begin : sizeof(block);
        for (uint64_t i=0; i<count; i++) block[i] = large_pattern(begin+i);
        write_all(fd, block, (size_t)count, offset+begin);
    }
    k3_st_shard *shard = NULL;
    char err[512] = {0};
    assert(k3_st_open(&shard, path, err, sizeof(err)));
    k3_layer layer = {.index=0, .shard=shard,
        .static_offset=offset, .static_size=bytes};
    k3_model model = {.gguf=true};
    k3_context ctx = {.model=&model, .layer_start=0, .layer_end=0};
    ctx.layers[0] = &layer;
    assert(alloc_static_slots(&ctx, err, sizeof(err)));
    ctx.gguf_staging_bytes = static_slots_capacity(&ctx);
    ctx.gguf_transient_bytes = ctx.gguf_staging_bytes + page;
    assert(lock_gguf_static_slots(&ctx, err, sizeof(err)));
    k3_static_slot *slot = &ctx.static_slots[0];
    assert(static_io_threads() == 16u);
    /* ceil(67MiB / 64MiB) limits the requested 16 workers to exactly two.
     * Their boundary rounds down to 16KiB; the final tail stays unaligned. */
    assert((bytes + K3_PREAD_CHUNK_BYTES - 1u) / K3_PREAD_CHUNK_BYTES == 2u);
    memset(slot->data, 0xa5, (size_t)slot->capacity);
    k3_static_load load = {0};
    assert(static_load_start(&load, &layer, slot, err, sizeof(err)));
    assert(static_load_join(&load, err, sizeof(err)));
    assert(slot->bytes == bytes && !load.started);
    const unsigned char *actual = slot->data;
    for (uint64_t i=0; i<bytes; i++) assert(actual[i] == large_pattern(i));
    for (uint64_t i=bytes; i<slot->capacity; i++) assert(actual[i] == 0xa5);
    /* Exercise another two-worker partition with nonzero source bias and a
     * changed split point, so stale/single-partition results cannot pass. */
    const uint64_t shift = page + 37u;
    const uint64_t second_bytes = bytes - shift - 23u;
    assert(second_bytes > K3_PREAD_CHUNK_BYTES);
    assert(static_span_load_start(&load, &layer, slot,
                                 offset+shift, second_bytes, 0, 0,
                                 err, sizeof(err)));
    assert(static_load_join(&load, err, sizeof(err)));
    assert(slot->bytes == second_bytes);
    for (uint64_t i=0; i<second_bytes; i++)
        assert(actual[i] == large_pattern(i+shift));
    /* One worker reaches EOF: joining still drains both workers and a
     * partial buffer must not be marked complete. */
    assert(static_span_load_start(&load, &layer, slot,
                                 offset+1u, bytes, 0, 0, err, sizeof(err)));
    assert(!static_load_join(&load, err, sizeof(err)));
    assert(!load.started && slot->bytes == 0);
    assert(strstr(err, "unexpected end of file"));
    free_static_slots(&ctx);
    assert(!ctx.gguf_staging_locked_bytes && !slot->data);
    k3_st_close(shard); close(fd); unlink(path);
    puts("static staging: optional 67MiB two-worker partition, bias, tail and EOF passed");
}

int main(void) {
    const char *large_env = getenv("K3_TEST_STATIC_STAGING_LARGE");
    const int large = large_env && large_env[0] && strcmp(large_env, "0") != 0;
    if (large) assert(setenv("K3_STATIC_IO_THREADS", "16", 1) == 0);
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t payload = UINT64_C(4294967296) + 13u * page;
    char path[] = "/tmp/k3-stage-fixture-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    char header[256];
    int n = snprintf(header, sizeof(header),
        "{\"data\":{\"dtype\":\"U8\",\"shape\":[%llu],\"data_offsets\":[0,%llu]}}",
        (unsigned long long)payload, (unsigned long long)payload);
    assert(n > 0 && n < (int)sizeof(header));
    uint64_t header_bytes = (uint64_t)n;
    write_all(fd, &header_bytes, 8, 0);
    write_all(fd, header, (size_t)n, 8);
    uint64_t data_start = 8u + header_bytes;
    assert(ftruncate(fd, (off_t)(data_start + payload)) == 0);
    const uint64_t high = data_start + UINT64_C(4294967296) + 17u;
    const size_t pattern_bytes = (size_t)(12u * page);
    unsigned char *pattern = malloc(pattern_bytes); assert(pattern);
    for (size_t i = 0; i < pattern_bytes; i++)
        pattern[i] = (unsigned char)((i * 131u + i / 251u + 17u) & 255u);
    write_all(fd, pattern, pattern_bytes, high);
    char err[512] = {0};
    k3_st_shard *shard = NULL;
    assert(k3_st_open(&shard, path, err, sizeof(err)));
    k3_layer layers[2] = {
        {.index=0, .shard=shard, .static_offset=high, .static_size=5u*page+37u},
        {.index=1, .shard=shard, .static_offset=high+6u*page+3u, .static_size=3u*page+11u},
    };
    k3_model model = {.gguf=true};
    k3_context ctx = {.model=&model, .layer_start=0, .layer_end=1};
    ctx.layers[0] = &layers[0]; ctx.layers[1] = &layers[1];
    assert(alloc_static_slots(&ctx, err, sizeof(err)));
    ctx.gguf_staging_bytes = static_slots_capacity(&ctx);
    const uint64_t original_transient = UINT64_C(256)*1024u*1024u;
    ctx.gguf_transient_bytes = original_transient + ctx.gguf_staging_bytes;
    assert(lock_gguf_static_slots(&ctx, err, sizeof(err)));
    assert(ctx.gguf_staging_locked_bytes == ctx.gguf_staging_bytes);
    assert(ctx.gguf_transient_bytes - ctx.gguf_staging_locked_bytes == original_transient);
    assert(ctx.static_slots[0].locked && ctx.static_slots[1].locked);
    /* An invalid 'locked' claim must fail instead of reducing other reserves. */
    ctx.gguf_staging_locked_bytes++;
    assert(!check_gguf_memory(&ctx, "fixture", 0, err, sizeof(err)));
    ctx.gguf_staging_locked_bytes--; err[0] = 0;
    assert(check_gguf_memory(&ctx, "fixture", 0, err, sizeof(err)));
    k3_context clone = {.model=&model, .shared_owner=&ctx,
        .gguf_staging_locked_bytes=UINT64_MAX};
    /* Clones must consult the shared owner's locks, not stale local copies. */
    assert(check_gguf_memory(&clone, "fixture clone", 0, err, sizeof(err)));
    ctx.gguf_staging_locked_bytes++;
    assert(!check_gguf_memory(&clone, "fixture clone", 0, err, sizeof(err)));
    ctx.gguf_staging_locked_bytes--; err[0] = 0;

    k3_static_load a = {0}, b = {0};
    assert(static_load_start(&a, &layers[0], &ctx.static_slots[0], err, sizeof(err)));
    assert(static_load_start(&b, &layers[1], &ctx.static_slots[1], err, sizeof(err)));
    assert(static_load_join(&a, err, sizeof(err)));
    assert(static_load_join(&b, err, sizeof(err)));
    assert(!a.started && !b.started);
    assert(ctx.static_slots[0].bytes == layers[0].static_size);
    assert(ctx.static_slots[1].bytes == layers[1].static_size);
    assert(memcmp(ctx.static_slots[0].data, pattern, layers[0].static_size)==0);
    assert(memcmp(ctx.static_slots[1].data, pattern+6u*page+3u, layers[1].static_size)==0);
    /* Nonzero >4GiB source bias resolves tensor bytes within the staged slot. */
    k3_weight_source source = {.base=ctx.static_slots[0].data,
        .size=layers[0].static_size, .bias=layers[0].static_offset};
    const uint64_t tensor_offset = high + page + 23u;
    assert(memcmp((const unsigned char *)source.base + tensor_offset - source.bias,
                  pattern + page + 23u, 97u)==0);
    /* Reuse a slot with a smaller odd size and new offset. Locked trailing
     * pages must remain intact rather than being discarded/reclaimed. */
    const uint64_t smaller = page + 19u;
    memset((unsigned char *)ctx.static_slots[0].data + 3u*page, 0x5a, (size_t)page);
    assert(static_span_load_start(&a, &layers[0], &ctx.static_slots[0],
                                 high+2u*page+9u, smaller, 0, 0, err, sizeof(err)));
    assert(static_load_join(&a, err, sizeof(err)));
    assert(ctx.static_slots[0].bytes == smaller);
    assert(memcmp(ctx.static_slots[0].data, pattern+2u*page+9u, (size_t)smaller)==0);
    for (uint64_t i=3u*page; i<4u*page; i++)
        assert(((unsigned char *)ctx.static_slots[0].data)[i]==0x5a);
    /* A joined cancellation drains the reader before reuse/destruction. */
    assert(static_load_start(&b, &layers[1], &ctx.static_slots[1], err, sizeof(err)));
    static_load_cancel(&b); assert(!b.started);
    assert(memcmp(ctx.static_slots[1].data, pattern+6u*page+3u, layers[1].static_size)==0);
    /* EOF is an error even after a prior successful read; never expose old
     * slot.bytes as a completed new payload. */
    assert(static_span_load_start(&a, &layers[0], &ctx.static_slots[0],
                                 data_start+payload-5u, 17u, 0, 0, err, sizeof(err)));
    assert(!static_load_join(&a, err, sizeof(err)));
    assert(!a.started && ctx.static_slots[0].bytes==0);
    assert(strstr(err, "unexpected end of file"));
    assert(!static_span_load_start(&a, &layers[0], &ctx.static_slots[0],
                                  high, ctx.static_slots[0].capacity+1u, 0, 0, err, sizeof(err)));
    free_static_slots(&ctx);
    assert(!ctx.static_staging && !ctx.gguf_staging_locked_bytes);
    assert(!ctx.static_slots[0].data && !ctx.static_slots[1].data);
    /* Failure path: deny mlock only inside our fixture child, then verify all
     * partial allocations can be unmapped without a false locked-byte claim. */
    pid_t child = fork(); assert(child >= 0);
    if (child == 0) {
        struct rlimit limit = {0, 0};
        assert(setrlimit(RLIMIT_MEMLOCK, &limit)==0);
        assert(alloc_static_slots(&ctx, err, sizeof(err)));
        assert(!lock_gguf_static_slots(&ctx, err, sizeof(err)));
        assert(!ctx.gguf_staging_locked_bytes && !ctx.static_slots[0].locked);
        free_static_slots(&ctx);
        assert(!ctx.static_slots[0].data && !ctx.static_slots[1].data);
        _exit(0);
    }
    int status=0; assert(waitpid(child, &status, 0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==0);
    /* Also force the second mlock to fail after the first succeeded. */
    child = fork(); assert(child >= 0);
    if (child == 0) {
        const rlim_t first_capacity = (rlim_t)(6u * page);
        struct rlimit limit = {first_capacity, first_capacity};
        assert(setrlimit(RLIMIT_MEMLOCK, &limit)==0);
        assert(alloc_static_slots(&ctx, err, sizeof(err)));
        assert(!lock_gguf_static_slots(&ctx, err, sizeof(err)));
        assert(ctx.gguf_staging_locked_bytes == ctx.static_slots[0].capacity);
        assert(ctx.static_slots[0].locked && !ctx.static_slots[1].locked);
        free_static_slots(&ctx);
        assert(!ctx.gguf_staging_locked_bytes);
        assert(!ctx.static_slots[0].data && !ctx.static_slots[1].data);
        _exit(0);
    }
    status=0; assert(waitpid(child, &status, 0)==child);
    assert(WIFEXITED(status) && WEXITSTATUS(status)==0);
    k3_st_close(shard); close(fd); unlink(path); free(pattern);
    if (large) test_large_multiworker();
    puts("static staging: exact offset/tail/reuse bytes, EOF, lock accounting and cleanup passed (no GPU)");
    return 0;
}
