#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

/*
 * End-to-end regression for direct F16 -> Q4_K requantization of
 * blk.*.indexer.attn_q_b.weight.
 *
 * The fixture is deliberately GGUF-library-free.  It writes three tiny
 * tensors, invokes the production CLI, then parses the output and compares
 * the Q4_K payload with the public quantization facade.  Seventeen rows cross
 * the writer's 16-row conversion batch boundary.
 */

#include "quants.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    GGUF_VERSION = 3,
    GGUF_ALIGNMENT = 32,
    FIXTURE_TENSORS = 3,
    INDEXER_COLS = 256,
    INDEXER_ROWS = 17,
};

static const char *const k_before_name = "test.sentinel_before";
static const char *const k_indexer_name =
    "blk.2.indexer.attn_q_b.weight";
static const char *const k_after_name = "test.sentinel_after";

static uint8_t k_before_payload[28];
static uint8_t k_after_payload[26];

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[DS4Q_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    size_t size;
} tensor_info;

typedef struct {
    FILE *fp;
    uint64_t data_offset;
    tensor_info tensors[FIXTURE_TENSORS];
} parsed_gguf;

static int g_failures;

static void fail(const char *what) {
    fprintf(stderr, "test_quantizer_indexer_q4 FAIL: %s\n", what);
    g_failures++;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

static bool write_bytes(FILE *fp, const void *data, size_t size) {
    return size == 0 || fwrite(data, 1, size, fp) == size;
}

static bool write_u16_le(FILE *fp, uint16_t value) {
    const uint8_t bytes[2] = {
        (uint8_t)value,
        (uint8_t)(value >> 8u),
    };
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_u32_le(FILE *fp, uint32_t value) {
    uint8_t bytes[4];
    for (uint32_t i = 0; i < 4u; i++) {
        bytes[i] = (uint8_t)(value >> (8u * i));
    }
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_u64_le(FILE *fp, uint64_t value) {
    uint8_t bytes[8];
    for (uint32_t i = 0; i < 8u; i++) {
        bytes[i] = (uint8_t)(value >> (8u * i));
    }
    return write_bytes(fp, bytes, sizeof(bytes));
}

static bool write_string(FILE *fp, const char *value) {
    const size_t len = strlen(value);
    return write_u64_le(fp, len) && write_bytes(fp, value, len);
}

static bool write_zeros(FILE *fp, uint64_t count) {
    static const uint8_t zeros[256] = {0};
    while (count != 0) {
        const size_t chunk = count < sizeof(zeros) ? (size_t)count : sizeof(zeros);
        if (!write_bytes(fp, zeros, chunk)) return false;
        count -= chunk;
    }
    return true;
}

static uint16_t *make_indexer_f16(uint32_t cols, uint32_t rows) {
    const size_t count = (size_t)cols * rows;
    float *values = malloc(count * sizeof(*values));
    uint16_t *half = malloc(count * sizeof(*half));
    if (!values || !half) {
        free(half);
        free(values);
        return NULL;
    }
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t col = 0; col < cols; col++) {
            const int32_t raw =
                (int32_t)((row * 67u + col * 29u + (row ^ col) * 3u) % 257u) -
                128;
            values[(size_t)row * cols + col] = (float)raw / 32.0f;
        }
    }
    ds4q_f32_to_f16_row(values, half, (int64_t)count);
    free(values);
    return half;
}

static bool write_tensor_info(FILE *fp, const char *name, uint32_t n_dims,
                              uint64_t d0, uint64_t d1, uint32_t type,
                              uint64_t offset) {
    return write_string(fp, name) &&
           write_u32_le(fp, n_dims) &&
           write_u64_le(fp, d0) &&
           (n_dims == 1u || write_u64_le(fp, d1)) &&
           write_u32_le(fp, type) &&
           write_u64_le(fp, offset);
}

static bool write_fixture(const char *path, uint32_t cols, uint32_t rows,
                          const uint16_t *indexer_f16) {
    const uint64_t before_offset = 0;
    const uint64_t indexer_offset =
        align_up(sizeof(k_before_payload), GGUF_ALIGNMENT);
    const uint64_t indexer_bytes = (uint64_t)cols * rows * sizeof(uint16_t);
    const uint64_t after_offset =
        align_up(indexer_offset + indexer_bytes, GGUF_ALIGNMENT);

    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool ok = write_bytes(fp, "GGUF", 4) &&
              write_u32_le(fp, GGUF_VERSION) &&
              write_u64_le(fp, FIXTURE_TENSORS) &&
              write_u64_le(fp, 0) &&
              write_tensor_info(fp, k_before_name, 1,
                                sizeof(k_before_payload) / sizeof(uint32_t),
                                0, DS4Q_TYPE_F32, before_offset) &&
              write_tensor_info(fp, k_indexer_name, 2, cols, rows,
                                DS4Q_TYPE_F16, indexer_offset) &&
              write_tensor_info(fp, k_after_name, 1,
                                sizeof(k_after_payload) / sizeof(uint16_t),
                                0, DS4Q_TYPE_F16, after_offset);

    const off_t metadata_end = ftello(fp);
    if (metadata_end < 0) ok = false;
    const uint64_t data_offset = metadata_end < 0 ? 0 :
        align_up((uint64_t)metadata_end, GGUF_ALIGNMENT);
    if (ok) ok = write_zeros(fp, data_offset - (uint64_t)metadata_end);
    if (ok) ok = write_bytes(fp, k_before_payload, sizeof(k_before_payload));
    if (ok) ok = write_zeros(fp, indexer_offset - sizeof(k_before_payload));
    for (uint64_t i = 0; ok && i < (uint64_t)cols * rows; i++) {
        ok = write_u16_le(fp, indexer_f16[i]);
    }
    if (ok) {
        ok = write_zeros(fp, after_offset - indexer_offset - indexer_bytes) &&
             write_bytes(fp, k_after_payload, sizeof(k_after_payload));
    }
    if (fclose(fp) != 0) ok = false;
    return ok;
}

static bool read_exact(FILE *fp, void *data, size_t size) {
    return size == 0 || fread(data, 1, size, fp) == size;
}

static bool read_u32_le(FILE *fp, uint32_t *value) {
    uint8_t bytes[4];
    if (!read_exact(fp, bytes, sizeof(bytes))) return false;
    *value = 0;
    for (uint32_t i = 0; i < 4u; i++) {
        *value |= (uint32_t)bytes[i] << (8u * i);
    }
    return true;
}

static bool read_u64_le(FILE *fp, uint64_t *value) {
    uint8_t bytes[8];
    if (!read_exact(fp, bytes, sizeof(bytes))) return false;
    *value = 0;
    for (uint32_t i = 0; i < 8u; i++) {
        *value |= (uint64_t)bytes[i] << (8u * i);
    }
    return true;
}

static char *read_string(FILE *fp) {
    uint64_t len = 0;
    if (!read_u64_le(fp, &len) || len > 4096u || len > SIZE_MAX - 1u) return NULL;
    char *value = malloc((size_t)len + 1u);
    if (!value) return NULL;
    if (!read_exact(fp, value, (size_t)len)) {
        free(value);
        return NULL;
    }
    value[len] = '\0';
    return value;
}

static size_t tensor_size(const tensor_info *tensor) {
    if (tensor->n_dims == 0 || tensor->n_dims > DS4Q_MAX_DIMS ||
        tensor->dims[0] > INT64_MAX) {
        return 0;
    }
    const size_t row = ds4q_row_size((ds4q_type)tensor->type,
                                     (int64_t)tensor->dims[0]);
    if (row == 0) return 0;
    size_t size = row;
    for (uint32_t d = 1; d < tensor->n_dims; d++) {
        if (tensor->dims[d] > SIZE_MAX ||
            (tensor->dims[d] != 0 && size > SIZE_MAX / tensor->dims[d])) {
            return 0;
        }
        size *= (size_t)tensor->dims[d];
    }
    return size;
}

static void close_parsed(parsed_gguf *gguf) {
    if (gguf->fp) fclose(gguf->fp);
    gguf->fp = NULL;
    for (uint32_t i = 0; i < FIXTURE_TENSORS; i++) {
        free(gguf->tensors[i].name);
        gguf->tensors[i].name = NULL;
    }
}

static bool parse_output(const char *path, parsed_gguf *gguf) {
    memset(gguf, 0, sizeof(*gguf));
    gguf->fp = fopen(path, "rb");
    if (!gguf->fp) return false;

    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    bool ok = read_exact(gguf->fp, magic, sizeof(magic)) &&
              memcmp(magic, "GGUF", sizeof(magic)) == 0 &&
              read_u32_le(gguf->fp, &version) &&
              read_u64_le(gguf->fp, &n_tensors) &&
              read_u64_le(gguf->fp, &n_kv) &&
              version == GGUF_VERSION &&
              n_tensors == FIXTURE_TENSORS && n_kv == 0;

    for (uint32_t i = 0; ok && i < FIXTURE_TENSORS; i++) {
        tensor_info *tensor = &gguf->tensors[i];
        tensor->name = read_string(gguf->fp);
        ok = tensor->name && read_u32_le(gguf->fp, &tensor->n_dims) &&
             tensor->n_dims >= 1u && tensor->n_dims <= DS4Q_MAX_DIMS;
        for (uint32_t d = 0; ok && d < tensor->n_dims; d++) {
            ok = read_u64_le(gguf->fp, &tensor->dims[d]);
        }
        ok = ok && read_u32_le(gguf->fp, &tensor->type) &&
             read_u64_le(gguf->fp, &tensor->offset);
        if (ok) {
            tensor->size = tensor_size(tensor);
            ok = tensor->size != 0;
        }
    }
    const off_t metadata_end = ok ? ftello(gguf->fp) : -1;
    if (metadata_end < 0) ok = false;
    if (ok) {
        gguf->data_offset = align_up((uint64_t)metadata_end, GGUF_ALIGNMENT);
    } else {
        close_parsed(gguf);
    }
    return ok;
}

static bool read_payload(const parsed_gguf *gguf, uint32_t index,
                         void *data, size_t size) {
    if (!gguf->fp || index >= FIXTURE_TENSORS ||
        size != gguf->tensors[index].size ||
        gguf->data_offset > UINT64_MAX - gguf->tensors[index].offset) {
        return false;
    }
    const uint64_t absolute =
        gguf->data_offset + gguf->tensors[index].offset;
    return absolute <= (uint64_t)INT64_MAX &&
           fseeko(gguf->fp, (off_t)absolute, SEEK_SET) == 0 &&
           read_exact(gguf->fp, data, size);
}

static int run_quantizer(const char *tool, const char *source,
                         const char *output) {
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execl(tool, tool,
              "--source-gguf", source,
              "--out", output,
              "--indexer-q", "q4_k",
              (char *)NULL);
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void verify_positive(const char *output, const uint16_t *half) {
    parsed_gguf gguf;
    if (!parse_output(output, &gguf)) {
        fail("parse positive output");
        return;
    }

    const tensor_info *before = &gguf.tensors[0];
    const tensor_info *indexer = &gguf.tensors[1];
    const tensor_info *after = &gguf.tensors[2];
    if (strcmp(before->name, k_before_name) != 0 ||
        before->type != DS4Q_TYPE_F32 ||
        before->size != sizeof(k_before_payload)) {
        fail("before sentinel metadata");
    }
    if (strcmp(indexer->name, k_indexer_name) != 0 ||
        indexer->type != DS4Q_TYPE_Q4_K || indexer->n_dims != 2u ||
        indexer->dims[0] != INDEXER_COLS ||
        indexer->dims[1] != INDEXER_ROWS ||
        indexer->size != INDEXER_ROWS * 144u) {
        fail("indexer Q4_K metadata");
    }
    if (strcmp(after->name, k_after_name) != 0 ||
        after->type != DS4Q_TYPE_F16 ||
        after->size != sizeof(k_after_payload)) {
        fail("after sentinel metadata");
    }

    uint8_t before_actual[sizeof(k_before_payload)];
    uint8_t after_actual[sizeof(k_after_payload)];
    if (!read_payload(&gguf, 0, before_actual, sizeof(before_actual)) ||
        memcmp(before_actual, k_before_payload, sizeof(before_actual)) != 0) {
        fail("before sentinel payload changed");
    }
    if (!read_payload(&gguf, 2, after_actual, sizeof(after_actual)) ||
        memcmp(after_actual, k_after_payload, sizeof(after_actual)) != 0) {
        fail("after sentinel payload changed");
    }

    const size_t count = (size_t)INDEXER_COLS * INDEXER_ROWS;
    const size_t q4_size = INDEXER_ROWS * ds4q_row_size(
        DS4Q_TYPE_Q4_K, INDEXER_COLS);
    float *rounded = malloc(count * sizeof(*rounded));
    uint8_t *expected = malloc(q4_size);
    uint8_t *actual = malloc(q4_size);
    if (!rounded || !expected || !actual) {
        fail("reference allocation");
    } else {
        for (size_t i = 0; i < count; i++) {
            rounded[i] = ds4q_f16_to_f32(half[i]);
        }
        ds4q_quantize_init(DS4Q_TYPE_Q4_K);
        const size_t written = ds4q_quantize_chunk(
            DS4Q_TYPE_Q4_K, rounded, expected, 0,
            INDEXER_ROWS, INDEXER_COLS, NULL);
        if (written != q4_size) {
            fail("reference Q4_K size");
        } else if (!read_payload(&gguf, 1, actual, q4_size)) {
            fail("read indexer payload");
        } else if (memcmp(actual, expected, q4_size) != 0) {
            size_t first = 0;
            while (first < q4_size && actual[first] == expected[first]) first++;
            fprintf(stderr,
                    "test_quantizer_indexer_q4: first Q4 mismatch at %zu/%zu\n",
                    first, q4_size);
            fail("indexer Q4_K payload mismatch");
        }
    }
    free(actual);
    free(expected);
    free(rounded);
    close_parsed(&gguf);
}

int main(int argc, char **argv) {
    const char *tool = argc > 1 ? argv[1] : "./gguf-tools/deepseek4-quantize";
    if (argc > 2) {
        fprintf(stderr, "usage: %s [deepseek4-quantize]\n", argv[0]);
        return 2;
    }
    if (access(tool, X_OK) != 0) {
        fprintf(stderr, "test_quantizer_indexer_q4: executable not found: %s\n",
                tool);
        return 2;
    }
    for (size_t i = 0; i < sizeof(k_before_payload); i++) {
        k_before_payload[i] = (uint8_t)(0x31u + i * 7u);
    }
    for (size_t i = 0; i < sizeof(k_after_payload); i++) {
        k_after_payload[i] = (uint8_t)(0xd3u - i * 5u);
    }

    char tmpdir[] = "/tmp/ds4-indexer-q4.XXXXXX";
    if (!mkdtemp(tmpdir)) {
        perror("mkdtemp");
        return 1;
    }
    char source[512];
    char output[512];
    char bad_source[512];
    char bad_output[512];
    if (snprintf(source, sizeof(source), "%s/source.gguf", tmpdir) >=
            (int)sizeof(source) ||
        snprintf(output, sizeof(output), "%s/output.gguf", tmpdir) >=
            (int)sizeof(output) ||
        snprintf(bad_source, sizeof(bad_source), "%s/bad-source.gguf", tmpdir) >=
            (int)sizeof(bad_source) ||
        snprintf(bad_output, sizeof(bad_output), "%s/bad-output.gguf", tmpdir) >=
            (int)sizeof(bad_output)) {
        fail("temporary path too long");
        goto cleanup_dir;
    }

    uint16_t *half = make_indexer_f16(INDEXER_COLS, INDEXER_ROWS);
    if (!half || !write_fixture(source, INDEXER_COLS, INDEXER_ROWS, half)) {
        fail("write positive fixture");
    } else {
        const int status = run_quantizer(tool, source, output);
        if (status != 0) {
            fprintf(stderr,
                    "test_quantizer_indexer_q4: positive quantizer exit=%d\n",
                    status);
            fail("positive quantizer invocation");
        } else {
            verify_positive(output, half);
        }
    }
    free(half);

    uint16_t *bad_half = make_indexer_f16(INDEXER_COLS - 1u, 2u);
    if (!bad_half ||
        !write_fixture(bad_source, INDEXER_COLS - 1u, 2u, bad_half)) {
        fail("write unaligned fixture");
    } else {
        const int status = run_quantizer(tool, bad_source, bad_output);
        if (status == 0) fail("unaligned width unexpectedly accepted");
        if (access(bad_output, F_OK) == 0) {
            fail("unaligned conversion created output");
        }
    }
    free(bad_half);

    unlink(bad_output);
    unlink(bad_source);
    unlink(output);
    unlink(source);
cleanup_dir:
    if (rmdir(tmpdir) != 0 && errno != ENOENT) {
        fail("remove temporary directory");
    }

    if (g_failures != 0) {
        fprintf(stderr, "test_quantizer_indexer_q4: %d failure(s)\n", g_failures);
        return 1;
    }
    fprintf(stderr,
            "test_quantizer_indexer_q4 PASS f16_rows=17 q4_bytes=%zu "
            "sentinels=2 unaligned_rejected=1\n",
            (size_t)INDEXER_ROWS * ds4q_row_size(DS4Q_TYPE_Q4_K, INDEXER_COLS));
    return 0;
}
