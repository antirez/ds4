/* Paired Q8 projections must write every requested row, even when the
 * model has single-row aligned artifacts. */
#include "ds4_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
template<class T> static void append(std::vector<unsigned char> &v, T x) {
    const auto *p = reinterpret_cast<const unsigned char *>(&x);
    v.insert(v.end(), p, p + sizeof(x));
}
static void string(std::vector<unsigned char> &v, const char *s) {
    append<uint64_t>(v, strlen(s));
    v.insert(v.end(), s, s + strlen(s));
}
static unsigned rng = 398;
static unsigned random_bits() { rng = rng * 1664525u + 1013904223u; return rng; }

int main() {
    constexpr unsigned width = 4096, rows = 8, blocks = width / 32;
    const unsigned outputs[2] = {512, 1024};
    const char *names[2] = {"blk.0.attn_kv.weight", "blk.0.attn_q_a.weight"};
    std::vector<unsigned char> model;
    append<uint32_t>(model, 0x46554747);
    append<uint32_t>(model, 3);
    append<uint64_t>(model, 2);
    append<uint64_t>(model, 0);
    uint64_t offsets[2], sizes[2], payload = 0;
    for (unsigned i = 0; i < 2; i++) {
        string(model, names[i]);
        append<uint32_t>(model, 2);
        append<uint64_t>(model, width);
        append<uint64_t>(model, outputs[i]);
        append<uint32_t>(model, 8);  /* GGML_TYPE_Q8_0 */
        append<uint64_t>(model, payload);
        offsets[i] = payload;
        sizes[i] = (uint64_t)outputs[i] * blocks * 34;
        payload += sizes[i];
    }
    const size_t header = (model.size() + 31u) & ~size_t(31);
    model.resize(header + payload);
    for (unsigned i = 0; i < 2; i++) {
        offsets[i] += header;
        for (size_t b = 0; b < sizes[i] / 34; b++) {
            const uint16_t d = 0x2c00;  /* 1/16 */
            memcpy(model.data() + offsets[i] + b * 34, &d, 2);
            for (unsigned k = 0; k < 32; k++)
                model[offsets[i] + b * 34 + 2 + k] = random_bits() >> 24;
        }
    }
    char path[] = "/tmp/ds4-q8-pair-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    CHECK(file && fwrite(model.data(), 1, model.size(), file) == model.size());
    CHECK(fclose(file) == 0);
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark()) {
        ds4_gpu_cleanup();
        CHECK(close(fd) == 0 && unlink(path) == 0);
        puts("Q8 aligned pair: Spark required, SKIP");
        return 0;
    }
    CHECK(ds4_gpu_build_derived_artifacts(model.data(), model.size(), path) == 2);
    CHECK(ds4_gpu_set_model_map_spans(model.data(), model.size(), offsets, sizes, 2, 0));
    CHECK(ds4_gpu_set_model_fd_for_map(fd, model.data()));
    CHECK(unlink(path) == 0);
    std::vector<float> input(rows * width), expected[2], actual[2];
    for (unsigned r = 0; r < rows; r++) for (unsigned k = 0; k < width; k++)
        input[r * width + k] = k % 32 == 0 ? 127 : (int)(random_bits() % 255) - 127;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * 4), *y[2];
    CHECK(x && ds4_gpu_tensor_write(x, 0, input.data(), input.size() * 4));
    for (unsigned i = 0; i < 2; i++) {
        expected[i].resize((size_t)rows * outputs[i]);
        actual[i].resize(expected[i].size() + 8);
        y[i] = ds4_gpu_tensor_alloc(actual[i].size() * 4);
        CHECK(y[i]);
        for (unsigned r = 0; r < rows; r++) for (unsigned o = 0; o < outputs[i]; o++) {
            double sum = 0;
            for (unsigned k = 0; k < width; k++)
                sum += (double)(int8_t)model[offsets[i] + ((size_t)o * blocks + k / 32) * 34 + 2 + k % 32] *
                       input[r * width + k] / 16.0;
            expected[i][r * outputs[i] + o] = (float)sum;
        }
    }
    for (unsigned aligned = 0; aligned < 2; aligned++) {
        if (aligned) CHECK(unsetenv("DS4_CUDA_NO_FUSED_ALIGNED_Q8_PAIR") == 0);
        else CHECK(setenv("DS4_CUDA_NO_FUSED_ALIGNED_Q8_PAIR", "1", 1) == 0);
        for (unsigned n = 1; n <= rows; n++) {
            for (unsigned i = 0; i < 2; i++) CHECK(ds4_gpu_tensor_fill_f32(y[i], NAN, actual[i].size()));
            CHECK(ds4_gpu_matmul_q8_0_pair_tensor(y[0], y[1], model.data(), model.size(),
                offsets[0], offsets[1], width, outputs[0], outputs[1], x, n));
            for (unsigned i = 0; i < 2; i++) {
                CHECK(ds4_gpu_tensor_read(y[i], 0, actual[i].data(), actual[i].size() * 4));
                for (unsigned j = 0; j < n * outputs[i]; j++) {
                    if (!std::isfinite(actual[i][j]) || fabs(actual[i][j] - expected[i][j]) > 0.002) {
                        fprintf(stderr, "aligned=%u rows=%u tensor=%u row=%u col=%u got=%g expected=%g\n",
                            aligned, n, i, j / outputs[i], j % outputs[i], actual[i][j], expected[i][j]);
                        exit(1);
                    }
                }
                for (size_t j = n * outputs[i]; j < actual[i].size(); j++) CHECK(std::isnan(actual[i][j]));
            }
            printf("Q8 pair aligned=%u rows=%u: PASS\n", aligned, n);
        }
    }
    /* Reject incomplete batched buffers before writing either output. */
    for (unsigned which = 0; which < 3; which++) {
        ds4_gpu_tensor *args[] = {x, y[0], y[1]};
        const uint64_t widths[] = {width, outputs[0], outputs[1]};
        auto *short_view = ds4_gpu_tensor_view(args[which], 0, 6 * widths[which] * 4 - 4);
        CHECK(short_view);
        args[which] = short_view;
        for (unsigned i = 0; i < 2; i++) CHECK(ds4_gpu_tensor_fill_f32(y[i], NAN, actual[i].size()));
        CHECK(!ds4_gpu_matmul_q8_0_pair_tensor(args[1], args[2], model.data(), model.size(),
            offsets[0], offsets[1], width, outputs[0], outputs[1], args[0], 6));
        ds4_gpu_tensor_free(short_view);
        for (unsigned i = 0; i < 2; i++) {
            CHECK(ds4_gpu_tensor_read(y[i], 0, actual[i].data(), actual[i].size() * 4));
            for (float value : actual[i]) CHECK(std::isnan(value));
        }
    }
    CHECK(!ds4_gpu_matmul_q8_0_pair_tensor(y[0], y[1], model.data(), model.size(),
        offsets[0], offsets[1], width, outputs[0], outputs[1], x, UINT64_MAX));
    CHECK(!ds4_gpu_matmul_q8_0_pair_tensor(y[0], y[1], model.data(), model.size(),
        offsets[0], offsets[1], UINT64_MAX, outputs[0], outputs[1], x, 1));
    CHECK(!ds4_gpu_matmul_q8_0_pair_tensor(y[0], y[1], model.data(), model.size(),
        offsets[0], offsets[1], width, UINT64_MAX, outputs[1], x, 1));
    puts("Q8 pair undersized buffers and overflow rejection: PASS");
    for (auto *t : {x, y[0], y[1]}) ds4_gpu_tensor_free(t);
    ds4_gpu_cleanup();
    CHECK(close(fd) == 0);
}
