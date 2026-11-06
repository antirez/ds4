/* Exercise the C entry points without an Objective-C caller's autorelease pool. */
#include "ds4_gpu.h"
#include <mach/mach.h>
#include <malloc/malloc.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static int exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    (void)layer; (void)gate;
    return seq == __atomic_add_fetch((uint64_t *)ud, 1, __ATOMIC_RELAXED);
}

static int batch_exchange(void *ud, uint32_t layer, uint32_t rows, uint64_t seq) {
    return rows == 1 && exchange(ud, layer, 1, seq);
}

static int big_exchange(void *ud, uint32_t layer, uint64_t seq,
                         const void *out, void *in, uint64_t bytes) {
    if (bytes != 1024 || !exchange(ud, layer, 1, seq)) return 0;
    memcpy(in, out, (size_t)bytes);
    return 1;
}

/* C callers have no outer autorelease pool. Releasing a CPU-touched tensor
 * must release its Metal allocation too, rather than retaining it until the
 * thread exits. Heap statistics alone miss these large shared buffers. */
static int check_tensor_memory(void) {
    ds4_gpu_tensor *tensor = NULL;
    const uint64_t bytes = 16u * 1024u * 1024u;
    uint64_t baseline = 0, footprint = 0;
    int rc = 1;
    CHECK(ds4_gpu_init());
    for (int i = 0; i < 40; i++) {
        tensor = ds4_gpu_tensor_alloc(bytes);
        CHECK(tensor);
        unsigned char *data = ds4_gpu_tensor_contents(tensor);
        CHECK(data);
        memset(data, i, (size_t)bytes);
        const uint32_t value = (uint32_t)i + 1234u;
        uint32_t readback = 0;
        CHECK(ds4_gpu_tensor_write(tensor, bytes - sizeof(value), &value, sizeof(value)));
        CHECK(ds4_gpu_tensor_read(tensor, bytes - sizeof(value), &readback, sizeof(readback)));
        CHECK(readback == value && data[0] == (unsigned char)i);
        ds4_gpu_tensor_free(tensor);
        tensor = NULL;
        if (i == 7 || i == 39) {
            task_vm_info_data_t info = {0};
            mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
            CHECK(task_info(mach_task_self(), TASK_VM_INFO,
                            (task_info_t)&info, &count) == KERN_SUCCESS);
            footprint = info.phys_footprint;
            if (i == 7) baseline = footprint;
        }
    }
    fprintf(stderr, "Metal CPU-touched buffers: footprint %.3f -> %.3f MiB\n",
            baseline / 1048576.0, footprint / 1048576.0);
    /* A leaked buffer per iteration adds 512 MiB; allow bounded driver
     * bookkeeping without requiring exact allocator reuse. */
    CHECK(footprint <= baseline + 32u * 1024u * 1024u);
    puts("Metal CPU-touched buffer lifetime: PASS");
    rc = 0;
done:
    ds4_gpu_tensor_free(tensor);
    ds4_gpu_cleanup();
    return rc;
}

static int check_tp_memory(int mode) {
    ds4_gpu_tensor *a = NULL, *slab = NULL, *out = NULL, *in = NULL;
    uint64_t exchanges = 0;
    int rc = 1;
    CHECK(ds4_gpu_init());
    a = ds4_gpu_tensor_alloc(1024);
    in = ds4_gpu_tensor_alloc(1024);
    slab = ds4_gpu_tensor_alloc(8192);
    CHECK(a && in && slab);
    CHECK(ds4_gpu_tensor_fill_f32(slab, 0, 2048));
    out = ds4_gpu_tensor_view(slab, 4096, 1024);
    CHECK(out);
    CHECK(ds4_gpu_tp_init(0, slab, 0, 4096, 1024, exchange, &exchanges));
    if (mode == 1) ds4_gpu_tp_set_session_batch_mode(1);
    ds4_gpu_tp_set_batch_exchange(batch_exchange);
    ds4_gpu_tp_set_big_exchange(big_exchange);
    float input[256], output[256];
    for (int j = 0; j < 256; j++) input[j] = j / 4.0f;
    CHECK(ds4_gpu_tensor_write(a, 0, input, sizeof(input)));
    size_t baseline = 0;
    for (int i = 0; i < 8192; i++) {
        CHECK(ds4_gpu_begin_commands());
        CHECK(ds4_gpu_add_tensor(out, a, a, 256));
        if (mode == 3) CHECK(ds4_gpu_tp_big_gate_encode(0, 1, out, in, 1024));
        else if (mode == 2) CHECK(ds4_gpu_tp_batch_gate_encode(0, 1));
        else CHECK(ds4_gpu_tp_gate_encode(0, 0));
        CHECK(ds4_gpu_end_commands());
        CHECK(!ds4_gpu_tp_failed());
        CHECK(ds4_gpu_tensor_read(mode == 3 ? in : out, 0, output, sizeof(output)));
        for (int j = 0; j < 256; j++) CHECK(output[j] == 2 * input[j]);
        if (i % 1024 == 1023) {
            malloc_statistics_t stats;
            malloc_zone_statistics(NULL, &stats);
            if (!baseline) baseline = stats.size_in_use;
            fprintf(stderr, "Metal TP mode %d: %d gates, heap %.3f MiB\n",
                    mode, i + 1, stats.size_in_use / 1048576.0);
            CHECK(stats.size_in_use <= baseline + 8u * 1024u * 1024u);
        }
    }
    CHECK(__atomic_load_n(&exchanges, __ATOMIC_RELAXED) == 8192);
    puts("Metal TP payloads, ordered loopback exchanges and bounded command heap: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tp_shutdown();
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(slab);
    ds4_gpu_tensor_free(in);
    ds4_gpu_tensor_free(a);
    ds4_gpu_cleanup();
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        if (!strcmp(argv[1], "buffers")) return check_tensor_memory();
        const char *modes[] = {"row", "session", "batch", "big"};
        for (int i = 0; i < 4; i++)
            if (!strcmp(argv[1], modes[i])) return check_tp_memory(i);
    }
    if (argc != 1) {
        fprintf(stderr, "usage: %s [buffers|row|session|batch|big]\n", argv[0]);
        return 2;
    }
    ds4_gpu_tensor *a = NULL, *b = NULL;
    int rc = 1;
    CHECK(ds4_gpu_init());
    a = ds4_gpu_tensor_alloc(1024);
    b = ds4_gpu_tensor_alloc(1024);
    CHECK(a && b);
    float input[256], output[256];
    for (int i = 0; i < 256; i++) input[i] = (float)i / 4.0f;
    CHECK(ds4_gpu_tensor_write(a, 0, input, sizeof(input)));
    size_t baseline = 0;
    for (int i = 0; i < 8192; i++) {
        CHECK(ds4_gpu_begin_commands());
        for (int j = 0; j < 8; j++) {
            CHECK(ds4_gpu_add_tensor(b, a, a, 256));
            CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, sizeof(input)));
            if (j == 3 && i % 2) CHECK(ds4_gpu_flush_commands());
        }
        CHECK(ds4_gpu_add_tensor(b, a, a, 256));
        CHECK(ds4_gpu_end_commands());
        CHECK(ds4_gpu_tensor_read(b, 0, output, sizeof(output)));
        for (int j = 0; j < 256; j++) CHECK(output[j] == 2.0f * input[j]);
        if (i % 1024 == 1023) {
            malloc_statistics_t stats;
            malloc_zone_statistics(NULL, &stats);
            if (!baseline) baseline = stats.size_in_use;
            fprintf(stderr, "Metal copies: %d batches, heap %.3f MiB\n",
                    i + 1, stats.size_in_use / 1048576.0);
            /* Leave room for driver caches, but not per-copy retained objects.
             * Before the fix this grows by about 35 MiB after warmup. */
            CHECK(stats.size_in_use <= baseline + 8u * 1024u * 1024u);
        }
    }
    puts("Metal copy contents, asynchronous flush and bounded command heap: PASS");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(b);
    ds4_gpu_cleanup();
    return rc;
}
