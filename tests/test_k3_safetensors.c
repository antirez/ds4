#define _POSIX_C_SOURCE 200809L

#include "k3_safetensors.h"

#include <assert.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_exact(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len != 0) {
        const ssize_t n = write(fd, p, len);
        assert(n > 0);
        p += n;
        len -= (size_t)n;
    }
}

static void write_u64_le(int fd, uint64_t value) {
    uint8_t bytes[8];
    for (uint32_t i = 0; i < 8; i++) bytes[i] = (uint8_t)(value >> (8u * i));
    write_exact(fd, bytes, sizeof(bytes));
}

static void test_gguf(const char *path) {
    char error[256] = {0};
    k3_st_shard *gguf = NULL;
    assert(k3_st_open(&gguf, path, error, sizeof(error)));
    assert(k3_st_is_gguf(gguf));
    assert(k3_st_tensor_count(gguf) == 2736);
    assert(k3_st_data_offset(gguf) == 3059360);

    const k3_st_tensor *gate =
        k3_st_find(gguf, "blk.1.ffn_gate_exps.weight");
    assert(gate != NULL);
    assert(gate->dtype == K3_ST_DTYPE_IQ2_XXS);
    assert(gate->n_dims == 3);
    assert(gate->shape[0] == 896);
    assert(gate->shape[1] == 3072);
    assert(gate->shape[2] == 3584);
    assert(gate->size == 2543321088ull);

    const k3_st_tensor *shared = k3_st_find(
        gguf,
        "language_model.model.layers.1.block_sparse_moe."
        "shared_experts.gate_proj.weight");
    assert(shared != NULL);
    assert(shared->dtype == K3_ST_DTYPE_MXFP4);
    assert(shared->shape[0] == 6144);
    assert(shared->shape[1] == 7168);
    k3_st_close(gguf);
}

int main(int argc, char **argv) {
    char path[] = "/tmp/ds4-k3-st-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);

    const char header[] =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"a\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]},"
        "\"expert.weight_packed\":{\"dtype\":\"U8\",\"shape\":[2,2],"
        "\"data_offsets\":[8,12]}}";
    const uint8_t data[] = {
        0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40,
        0x10, 0x32, 0x54, 0x76,
    };
    write_u64_le(fd, sizeof(header) - 1u);
    write_exact(fd, header, sizeof(header) - 1u);
    write_exact(fd, data, sizeof(data));
    close(fd);

    char error[256];
    k3_st_shard *shard = NULL;
    assert(k3_st_open(&shard, path, error, sizeof(error)));
    assert(k3_st_tensor_count(shard) == 2);
    assert(k3_st_data_offset(shard) == 8u + sizeof(header) - 1u);
    assert(k3_st_file_size(shard) == k3_st_data_offset(shard) + sizeof(data));

    const k3_st_tensor *a = k3_st_find(shard, "a");
    assert(a != NULL);
    assert(a->dtype == K3_ST_DTYPE_BF16);
    assert(a->n_dims == 2 && a->shape[0] == 2 && a->shape[1] == 2);
    assert(a->size == 8 && k3_st_nelements(a) == 4);

    const k3_st_tensor *packed = k3_st_find(shard, "expert.weight_packed");
    assert(packed != NULL);
    assert(packed->dtype == K3_ST_DTYPE_U8);
    assert(packed->offset == a->offset + a->size);
    assert(k3_st_find(shard, "missing") == NULL);

    assert(k3_st_map(shard, error, sizeof(error)));
    const uint8_t *map = k3_st_map_base(shard);
    assert(memcmp(map + packed->offset, data + 8, 4) == 0);
    assert(k3_st_prefetch(shard,
                          k3_st_data_offset(shard),
                          sizeof(data),
                          true,
                          error,
                          sizeof(error)));

    k3_st_close(shard);
    unlink(path);
    if (argc > 1) test_gguf(argv[1]);
    puts("test_k3_safetensors: ok");
    return 0;
}
