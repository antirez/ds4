#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *name;
    const char *dtype;
    int64_t shape[2];
    int n_dims;
    const uint8_t *data;
    size_t size;
} fixture_tensor;

typedef enum {
    FIXTURE_GOOD,
    FIXTURE_BAD_DTYPE,
    FIXTURE_BAD_SCALE_SHAPE,
    FIXTURE_BAD_PAYLOAD,
    FIXTURE_MISSING_SCALE,
    FIXTURE_UNALIGNED,
} fixture_kind;

typedef struct {
    int status;
    char *out;
    char *err;
} command_result;

static void fail_errno(const char *what) {
    fprintf(stderr, "FAIL: %s: %s\n", what, strerror(errno));
    exit(1);
}

static void appendf(char *buf, size_t cap, size_t *len, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + *len, cap - *len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap - *len) {
        fprintf(stderr, "FAIL: fixture JSON overflow\n");
        exit(1);
    }
    *len += (size_t)n;
}

static void write_all(FILE *f, const void *data, size_t size) {
    if (size != 0 && fwrite(data, 1, size, f) != size) fail_errno("write fixture");
}

static void fill_bytes(uint8_t *data, size_t size, uint8_t salt) {
    for (size_t i = 0; i < size; i++) data[i] = (uint8_t)(salt + i);
}

static void fill_scales(uint8_t *data, size_t size, uint8_t salt) {
    for (size_t i = 0; i < size; i++) data[i] = (uint8_t)(124u + (salt + i) % 5u);
}

static void write_fixture(const char *root, fixture_kind kind) {
    uint8_t w1[256], s1[16], w2[64], s2[2], w3[256], s3[16];
    const float norm[2] = {1.0f, 2.0f};
    fill_bytes(w1, sizeof(w1), 3);
    fill_scales(s1, sizeof(s1), 3);
    fill_bytes(w2, sizeof(w2), 11);
    fill_scales(s2, sizeof(s2), 11);
    fill_bytes(w3, sizeof(w3), 19);
    fill_scales(s3, sizeof(s3), 19);

    fixture_tensor tensors[] = {
        {"mtp.0.ffn.experts.0.w1.weight", "I8", {2, 128}, 2, w1, 256},
        {"mtp.0.ffn.experts.0.w1.scale", "F8_E8M0", {2, 8}, 2, s1, 16},
        {"mtp.0.ffn.experts.0.w2.weight", "I8", {2, 16}, 2, w2, 32},
        {"mtp.0.ffn.experts.0.w2.scale", "F8_E8M0", {2, 1}, 2, s2, 2},
        {"mtp.0.ffn.experts.0.w3.weight", "I8", {2, 128}, 2, w3, 256},
        {"mtp.0.ffn.experts.0.w3.scale", "F8_E8M0", {2, 8}, 2, s3, 16},
        {"mtp.0.attn_norm.weight", "F32", {2, 0}, 1,
         (const uint8_t *)norm, sizeof(norm)},
    };

    if (kind == FIXTURE_BAD_DTYPE) {
        tensors[2].dtype = "F16";
        tensors[2].size = 64;
    } else if (kind == FIXTURE_BAD_SCALE_SHAPE) {
        tensors[3].shape[0] = 1;
        tensors[3].shape[1] = 2;
    } else if (kind == FIXTURE_BAD_PAYLOAD) {
        tensors[2].size = 31;
    } else if (kind == FIXTURE_UNALIGNED) {
        tensors[2].shape[1] = 15;
        tensors[2].size = 30;
        tensors[3].shape[1] = 0;
        tensors[3].size = 0;
    }

    char header[8192];
    size_t header_len = 0;
    size_t offset = 0;
    appendf(header, sizeof(header), &header_len, "{");
    bool first = true;
    for (size_t i = 0; i < ARRAY_LEN(tensors); i++) {
        if (kind == FIXTURE_MISSING_SCALE && i == 3) continue;
        appendf(header, sizeof(header), &header_len,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                first ? "" : ",", tensors[i].name, tensors[i].dtype);
        for (int d = 0; d < tensors[i].n_dims; d++) {
            appendf(header, sizeof(header), &header_len, "%s%" PRId64,
                    d == 0 ? "" : ",", tensors[i].shape[d]);
        }
        appendf(header, sizeof(header), &header_len,
                "],\"data_offsets\":[%zu,%zu]}", offset,
                offset + tensors[i].size);
        offset += tensors[i].size;
        first = false;
    }
    appendf(header, sizeof(header), &header_len, "}");

    char path[4096];
    snprintf(path, sizeof(path), "%s/model-00001-of-00001.safetensors", root);
    FILE *f = fopen(path, "wb");
    if (!f) fail_errno("open safetensors fixture");
    uint8_t length_le[8];
    for (unsigned i = 0; i < 8; i++) {
        length_le[i] = (uint8_t)((uint64_t)header_len >> (8u * i));
    }
    write_all(f, length_le, sizeof(length_le));
    write_all(f, header, header_len);
    for (size_t i = 0; i < ARRAY_LEN(tensors); i++) {
        if (kind == FIXTURE_MISSING_SCALE && i == 3) continue;
        write_all(f, tensors[i].data, tensors[i].size);
    }
    if (fclose(f) != 0) fail_errno("close safetensors fixture");

    char index[4096];
    size_t index_len = 0;
    appendf(index, sizeof(index), &index_len, "{\"weight_map\":{");
    first = true;
    for (size_t i = 0; i < ARRAY_LEN(tensors); i++) {
        if (kind == FIXTURE_MISSING_SCALE && i == 3) continue;
        appendf(index, sizeof(index), &index_len,
                "%s\"%s\":\"model-00001-of-00001.safetensors\"",
                first ? "" : ",", tensors[i].name);
        first = false;
    }
    appendf(index, sizeof(index), &index_len, "}}");
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", root);
    f = fopen(path, "wb");
    if (!f) fail_errno("open safetensors index");
    write_all(f, index, index_len);
    if (fclose(f) != 0) fail_errno("close safetensors index");
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) fail_errno("open command output");
    if (fseek(f, 0, SEEK_END) != 0) fail_errno("seek command output");
    long length = ftell(f);
    if (length < 0) fail_errno("size command output");
    rewind(f);
    char *buf = malloc((size_t)length + 1);
    if (!buf) fail_errno("allocate command output");
    if (fread(buf, 1, (size_t)length, f) != (size_t)length) {
        fail_errno("read command output");
    }
    buf[length] = '\0';
    fclose(f);
    return buf;
}

static command_result run_quantizer(const char *binary, const char *root,
                                    const char *const args[]) {
    char out_path[4096], err_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/stdout", root);
    snprintf(err_path, sizeof(err_path), "%s/stderr", root);
    pid_t pid = fork();
    if (pid < 0) fail_errno("fork quantizer");
    if (pid == 0) {
        int out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0) {
            _exit(126);
        }
        close(out);
        close(err);
        char *argv[32];
        size_t n = 0;
        argv[n++] = (char *)binary;
        argv[n++] = "--hf";
        argv[n++] = (char *)root;
        argv[n++] = "--dspark-support";
        for (size_t i = 0; args[i]; i++) argv[n++] = (char *)args[i];
        argv[n] = NULL;
        execvp(binary, argv);
        _exit(127);
    }
    int status;
    if (waitpid(pid, &status, 0) < 0) fail_errno("wait for quantizer");
    command_result result = {
        .status = WIFEXITED(status) ? WEXITSTATUS(status) : 128,
        .out = read_file(out_path),
        .err = read_file(err_path),
    };
    return result;
}

static int check_case(const char *name, const char *binary, const char *root,
                      fixture_kind fixture, const char *const args[],
                      bool success, const char *out_text, const char *err_text) {
    write_fixture(root, fixture);
    command_result result = run_quantizer(binary, root, args);
    bool ok = (result.status == 0) == success;
    if (out_text) ok = ok && strstr(result.out, out_text) != NULL;
    if (err_text) ok = ok && strstr(result.err, err_text) != NULL;
    if (!ok) {
        fprintf(stderr, "FAIL: %s (exit %d)\nstdout:\n%s\nstderr:\n%s",
                name, result.status, result.out, result.err);
    }
    free(result.out);
    free(result.err);
    return ok ? 0 : 1;
}

static uint64_t expected_w2_hash(void) {
    uint8_t weight[32], scales[2], packed[34];
    fill_bytes(weight, sizeof(weight), 11);
    fill_scales(scales, sizeof(scales), 11);
    for (size_t row = 0; row < 2; row++) {
        const uint8_t *src = weight + row * 16;
        uint8_t *dst = packed + row * 17;
        dst[0] = scales[row];
        for (size_t i = 0; i < 16; i++) {
            uint8_t lo = (src[i / 2] >> ((i & 1u) * 4u)) & 0x0f;
            uint8_t hi = (src[(16 + i) / 2] >> ((i & 1u) * 4u)) & 0x0f;
            dst[1 + i] = (uint8_t)(lo | (hi << 4u));
        }
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < sizeof(packed); i++) {
        hash ^= packed[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    const char *binary = argc > 1 ? argv[1] : "gguf-tools/deepseek4-quantize";
    char root[] = "/tmp/ds4-dspark-mxfp4.XXXXXX";
    if (!mkdtemp(root)) fail_errno("create fixture directory");

    const char *accepted[] = {
        "--experts", "iq2_xxs", "--routed-w2", "mxfp4",
        "--n-experts", "1", "--dry-run", NULL,
    };
    const char *converted[] = {
        "--experts", "iq2_xxs", "--routed-w2", "mxfp4",
        "--n-experts", "1", "--compare-tensor",
        "mtp.0.ffn_down_exps.weight", NULL,
    };
    const char *non_expert[] = {
        "--tensor-type", "mtp.0.attn_norm.weight=mxfp4", "--dry-run", NULL,
    };
    const char *preserve[] = {"--routed-w2", "mxfp4", "--dry-run", NULL};
    int failed = 0;
    failed += check_case("accept routed MXFP4", binary, root, FIXTURE_GOOD,
                         accepted, true,
                         "tensor_types: f32=1 iq2_xxs=2 mxfp4=1", NULL);

    char hash_text[96];
    snprintf(hash_text, sizeof(hash_text),
             "generated_bytes: 34\ngenerated_fnv1a64: %016" PRIx64,
             expected_w2_hash());
    failed += check_case("repack routed MXFP4", binary, root, FIXTURE_GOOD,
                         converted, true, hash_text, NULL);
    failed += check_case("reject non-expert MXFP4", binary, root, FIXTURE_GOOD,
                         non_expert, false, NULL,
                         "unsupported DSpark planned tensor type");
    failed += check_case("reject non-I8 source", binary, root, FIXTURE_BAD_DTYPE,
                         preserve, false, NULL,
                         "requires a packed 2D I8 expert weight");
    failed += check_case("reject scale shape", binary, root,
                         FIXTURE_BAD_SCALE_SHAPE, preserve, false, NULL,
                         "expert scale shape mismatch");
    failed += check_case("reject weight payload", binary, root,
                         FIXTURE_BAD_PAYLOAD, preserve, false, NULL,
                         "expert weight payload size mismatch");
    failed += check_case("reject missing scale", binary, root,
                         FIXTURE_MISSING_SCALE, preserve, false, NULL,
                         "HF tensor not found: mtp.0.ffn.experts.0.w2.scale");
    failed += check_case("reject unaligned width", binary, root,
                         FIXTURE_UNALIGNED, preserve, false, NULL,
                         "expert dimensions are invalid");

    const char *files[] = {
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json", "stdout", "stderr",
    };
    for (size_t i = 0; i < ARRAY_LEN(files); i++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", root, files[i]);
        unlink(path);
    }
    rmdir(root);
    if (failed) return 1;
    printf("DSpark preserved MXFP4 planner/repacker: PASS\n");
    return 0;
}
