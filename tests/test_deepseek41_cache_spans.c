/* Linux CPU-only regression for accelerator startup over a disk-only GGUF.
 * Include the real span builders, replace only GPU cache entry points, and
 * discard unrelated engine sections at link time. No GPU library is linked. */
#include "../ds4.c"
#include <assert.h>

static uint64_t cache_calls, cache_bytes;
static const ds4_model *expected_model;

int ds4_gpu_cache_model_range(const void *map, uint64_t size, uint64_t off,
                              uint64_t bytes, const char *label) {
    (void)label;
    assert(expected_model && map == expected_model->map && size == expected_model->size);
    assert(off <= size && bytes <= size - off);
    assert(off == expected_model->tensor_data_pos && bytes == 64);
    cache_calls++;
    cache_bytes += bytes;
    return 1;
}

#ifndef DS4_ROCM_BUILD
int ds4_gpu_model_range_replaced(const void *map, uint64_t off, uint64_t bytes) {
    (void)map; (void)off; (void)bytes;
    return 0;
}

int ds4_gpu_cache_q8_f16_range(const void *map, uint64_t size, uint64_t off,
                              uint64_t bytes, uint64_t in, uint64_t out, const char *label) {
    (void)map; (void)size; (void)off; (void)bytes; (void)in; (void)out; (void)label;
    assert(!"fixture has no Q8 tensors");
    return 0;
}
#endif

/* Same sparse GGUF layout as test_deepseek41_gguf, with one row per table. */
static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }
static void putstr(FILE *fp, const char *s) {
    put64(fp, strlen(s));
    assert(fwrite(s, 1, strlen(s), fp) == strlen(s));
}
static void string_kv(FILE *fp, const char *key, const char *value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_STRING); putstr(fp, value);
}
static void tensor(FILE *fp, const char *name, uint32_t type,
                   uint64_t width, uint64_t rows, uint64_t offset) {
    putstr(fp, name); put32(fp, 2); put64(fp, width); put64(fp, rows);
    put32(fp, type); put64(fp, offset);
}

static void check_unmapped(const ds4_model *m) {
    unsigned char resident;
    errno = 0;
    assert(mincore((void *)(m->map + m->size), (size_t)sysconf(_SC_PAGESIZE),
                   &resident) == -1 && errno == ENOMEM);
}

static void check_mapping(const char *path, bool shared, uint64_t first, uint64_t file_size) {
    ds4_model m;
    model_open(&m, path, shared, false);
    expected_model = &m;
    cache_calls = cache_bytes = 0;
    assert(m.size == first && m.file_size == file_size);
    check_unmapped(&m);
    uint64_t prepared = UINT64_MAX;
#ifndef DS4_ROCM_BUILD
    /* CUDA startup spans still reject disk-only descriptors, while its Q8
     * cache skips V4.1 Engram tables as of upstream a04f46f. */
    assert(!accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    assert(accelerator_cache_q8_tensors(&m, NULL, NULL, 0));
    assert(cache_calls == 0);
    const uint64_t tensor_count = m.n_tensors;
    m.n_tensors = 1;
    assert(accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    assert(cache_calls == 1 && cache_bytes == 64 && prepared == 64);
    assert(accelerator_cache_q8_tensors(&m, NULL, NULL, 0));
    m.n_tensors = tensor_count;
    model_close(&m);
    expected_model = NULL;
    return;
#endif
    /* Before the fix this returned false with zero cache calls: the Engram
     * descriptors were incorrectly checked against the shorter weight map. */
    assert(accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    assert(cache_calls == 1 && cache_bytes == 64 && prepared == 64);
    const uint64_t off = m.tensor_data_pos, bytes = 64;
    assert(accelerator_prepare_model_tensor_spans(&m, &off, &bytes, 1, &prepared));
    assert(cache_calls == 2 && cache_bytes == 128 && prepared == 64);
    /* Explicit GPU map requests into the disk-only tail remain invalid. */
    assert(!accelerator_prepare_model_tensor_spans(&m, &first, &bytes, 1, &prepared));
    assert(cache_calls == 2);
#ifndef DS4_ROCM_BUILD
    assert(accelerator_cache_q8_tensors(&m, NULL, NULL, 0));
#endif
    /* A filter cannot conceal an invalid ordinary tensor descriptor. */
    const uint64_t saved_offset = m.tensors[0].abs_offset;
    m.tensors[0].abs_offset = m.size + 64;
    assert(!accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    assert(!accelerator_prepare_model_tensor_spans(&m, &off, &bytes, 1, &prepared));
    assert(cache_calls == 2);
#ifndef DS4_ROCM_BUILD
    assert(!accelerator_cache_q8_tensors(&m, NULL, NULL, 0));
#endif
    m.tensors[0].abs_offset = saved_offset;
    /* Neither an arbitrary tail tensor nor a malformed recognized table can
     * acquire the disk-only exemption. Normal mapped-range rejection stays. */
    const ds4_tensor saved_table = m.tensors[1];
    m.tensors[1].name = (ds4_str){.ptr = "other.weight", .len = 12};
    assert(!accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    m.tensors[1] = saved_table;
    m.tensors[1].type = DS4_TENSOR_F32;
    assert(!accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    m.tensors[1] = saved_table;
    m.tensors[1].bytes = m.file_size;
    assert(!accelerator_prepare_model_tensor_spans(&m, NULL, NULL, 0, &prepared));
    assert(cache_calls == 2);
#ifndef DS4_ROCM_BUILD
    assert(!accelerator_cache_q8_tensors(&m, NULL, NULL, 0));
#endif
    m.tensors[1] = saved_table;
    check_unmapped(&m);
    model_close(&m);
    expected_model = NULL;
}

int main(void) {
    enum { ALIGN = 16384 };
    const uint64_t first = 2 * ALIGN, second = 3 * ALIGN, file_size = second + 264;
    char path[] = "/tmp/ds41-cache-spans.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 3); put64(fp, 3);
    string_kv(fp, "general.architecture", "deepseek41");
    string_kv(fp, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    putstr(fp, "general.alignment"); put32(fp, GGUF_VALUE_UINT32); put32(fp, ALIGN);
    tensor(fp, "test.weight", DS4_TENSOR_F32, 16, 1, 0);
    tensor(fp, "blk.1.engram_embd.weight", DS4_TENSOR_I8, 264, 1, first - ALIGN);
    tensor(fp, "blk.14.engram_embd.weight", DS4_TENSOR_I8, 264, 1, second - ALIGN);
    assert(ftell(fp) < ALIGN);
    assert(fflush(fp) == 0 && ftruncate(fd, (off_t)file_size) == 0);
    check_mapping(path, false, first, file_size);
    check_mapping(path, true, first, file_size);
    assert(fclose(fp) == 0 && unlink(path) == 0);
#ifdef DS4_ROCM_BUILD
    puts("V4.1 ROCm startup spans: disk-only exclusion and resident/file/filter bounds PASS (CPU-only)");
#else
    puts("V4.1 CUDA startup spans and optional Q8 scan: disk-only exclusion and bounds PASS (CPU-only)");
#endif
    return 0;
}
