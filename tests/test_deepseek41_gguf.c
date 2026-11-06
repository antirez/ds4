/* Header-only sparse fixtures: no real model or GPU allocation. */
#include "../ds4.c"
#include "../ds4_engram.h"
#include <assert.h>
#include <sys/wait.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

static void check_unmapped(const void *ptr, size_t page) {
#ifdef __APPLE__
    (void)page;
    mach_vm_address_t address = (uintptr_t)ptr;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    kern_return_t result = mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    assert(result == KERN_INVALID_ADDRESS ||
           (result == KERN_SUCCESS && address > (uintptr_t)ptr));
#else
    unsigned char resident;
    errno = 0;
    assert(mincore((void *)ptr, page, &resident) == -1 && errno == ENOMEM);
#endif
}

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

static int run_fixture(int bad_layout) {
    enum { ALIGN = 16384 };
    const uint32_t rows = (1u << 24) + 3;
    const uint64_t table_bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    const uint64_t first = 2 * ALIGN + (bad_layout ? 32 : 0);
    const uint64_t second = align_up(first + table_bytes, ALIGN);
    const uint64_t file_size = second + table_bytes;
    char path[] = "/tmp/ds41-gguf.XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 3); put64(fp, 3);
    string_kv(fp, "general.architecture", "deepseek41");
    string_kv(fp, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    putstr(fp, "general.alignment"); put32(fp, GGUF_VALUE_UINT32); put32(fp, ALIGN);
    tensor(fp, "test.weight", DS4_TENSOR_F32, 16, 1, 0);
    tensor(fp, "blk.1.engram_embd.weight", 24, 264, rows, first - ALIGN);
    tensor(fp, "blk.14.engram_embd.weight", 24, 264, rows, second - ALIGN);
    assert(ftell(fp) < ALIGN);
    assert(fflush(fp) == 0 && ftruncate(fd, (off_t)file_size) == 0);
    uint8_t row[DS4_ENGRAM_ROW_BYTES];
    memset(row, 56, DS4_ENGRAM_DIM);
    memset(row + DS4_ENGRAM_DIM, 127, DS4_ENGRAM_ROW_BYTES - DS4_ENGRAM_DIM);
    assert(pwrite(fd, row, sizeof(row), (off_t)(second + table_bytes - sizeof(row))) == sizeof(row));
    if (bad_layout) {
        pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            ds4_model m;
            model_open(&m, path, false, false);
            model_close(&m);
            _exit(0);
        }
        int status;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
    } else for (int shared = 0; shared < 2; shared++) {
        ds4_model m;
        model_open(&m, path, shared != 0, false);
        assert(m.file_size == file_size && m.size == first);
        assert(m.max_tensor_bytes == 64);
        char resident;
        assert(mincore((void *)m.map, ALIGN, &resident) == 0);
        check_unmapped(m.map + first, ALIGN);
        const ds4_tensor *t = model_find_tensor(&m, "blk.14.engram_embd.weight");
        assert(t && t->abs_offset == second && t->bytes == table_bytes);
        assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "test.weight")) == 0);
        model_warm_weights(&m);
        ds4_engram_table table;
        assert(ds4_engram_table_open(&table, path, t->abs_offset, rows));
        const uint32_t id = rows - 1;
        float values[DS4_ENGRAM_DIM];
        assert(ds4_engram_read(&table, &id, 1, values));
        for (int i = 0; i < DS4_ENGRAM_DIM; i++) assert(values[i] == 1);
        ds4_engram_table_close(&table);
        model_close(&m);
    }
    assert(fclose(fp) == 0 && unlink(path) == 0);
    return 0;
}

static void check_model_layout(const char *path) {
    ds4_model m;
    model_open(&m, path, false, false);
    config_validate_model(&m);
    assert(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    assert(weights_layers_bound(&w, 0, UINT32_MAX));
    assert(weights_have_output_head(&w) && !w.output_hc_fn);
    check_unmapped(m.map + m.size, 16384);
    ds4_model_map_span_vec all = {0}, dense = {0};
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        assert((w.layer[il].attn_compressor_kv != NULL) == ds41_kv_source(il));
        assert((w.layer[il].indexer_attn_q_b != NULL) == ds41_index_source(il));
        assert((w.layer[il].engram_kv != NULL) == ds41_engram_layer(il));
        model_map_span_vec_include_layer(&all, &w.layer[il]);
        model_map_span_vec_include_layer_decode_static(&dense, &w.layer[il]);
    }
    for (uint32_t i = 0; i < all.len; i++) assert(all.v[i].end <= m.size);
    for (uint32_t i = 0; i < dense.len; i++) assert(dense.v[i].end <= m.size);
    free(all.v);
    free(dense.v);
    model_summary(&m);
    model_close(&m);
    puts("V4.1 complete model layout: PASS");
}

int main(int argc, char **argv) {
    if (argc == 2) {
        check_model_layout(argv[1]);
        return 0;
    }
    assert(argc == 1);
    run_fixture(0);
    run_fixture(1);
    puts("V4.1 disk-only GGUF extent: PASS");
    return 0;
}
