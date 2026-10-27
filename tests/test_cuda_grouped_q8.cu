/* Grouped attention projection and Q8 K-slice rows: batching must preserve
 * each scalar row. */
#include "ds4_gpu.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

static uint32_t random_state = 1739;
static uint32_t random_bits(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return random_state;
}

/* Fills the model with random Q8_0 blocks, initializes the GPU and maps it.
 * Returns the backing file descriptor. */
static int map_random_q8(std::vector<unsigned char> &model) {
    const uint64_t size = model.size(), offset = 0;
    for (uint64_t b = 0; b < size / 34; b++) {
        const uint16_t d = 0x2000 + random_bits() % 1024;
        memcpy(model.data() + b * 34, &d, 2);
        for (unsigned k = 0; k < 32; k++) model[b * 34 + 2 + k] = random_bits() >> 24;
    }
    char path[] = "/tmp/ds4-grouped-q8-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    FILE *file = fdopen(dup(fd), "wb");
    assert(file && fwrite(model.data(), 1, size, file) == size);
    assert(fclose(file) == 0 && unlink(path) == 0);
    assert(ds4_gpu_init());
    assert(ds4_gpu_set_model_map_spans(model.data(), size, &offset, &size, 1, 0));
    assert(ds4_gpu_set_model_fd_for_map(fd, model.data()));
    return fd;
}

static void run(unsigned width, unsigned rank, unsigned groups, bool bench) {
    const unsigned rows = 33, outputs = 32, low_dim = rank * groups;
    const uint64_t a_bytes = (uint64_t)groups * rank * ((width + 31) / 32) * 34;
    const uint64_t b_bytes = (uint64_t)outputs * ((low_dim + 31) / 32) * 34;
    const uint64_t size = a_bytes + b_bytes;
    std::vector<unsigned char> model(size);
    const int fd = map_random_q8(model);
    std::vector<float> input((size_t)rows * width * groups);
    for (float &v : input) v = ((int)(random_bits() >> 16) - 32768) / 8192.0f;
    auto *x = ds4_gpu_tensor_alloc(input.size() * 4);
    auto *y = ds4_gpu_tensor_alloc((uint64_t)rows * outputs * 4);
    auto *low = ds4_gpu_tensor_alloc((uint64_t)rows * low_dim * 4);
    assert(x && y && low && ds4_gpu_tensor_write(x, 0, input.data(), input.size() * 4));
    auto invoke = [&](ds4_gpu_tensor *dest, const ds4_gpu_tensor *src, unsigned n) {
        assert(ds4_gpu_attention_output_q8_batch_tensor(y, dest, nullptr, nullptr,
            model.data(), size, 0, a_bytes, width, rank, groups, outputs, src, n));
    };
    for (unsigned r = 0; r < rows; r++) {
        auto *xr = ds4_gpu_tensor_view(x, (uint64_t)r * width * groups * 4,
                                      (uint64_t)width * groups * 4);
        auto *yr = ds4_gpu_tensor_view(low, (uint64_t)r * low_dim * 4, low_dim * 4);
        assert(xr && yr);
        invoke(yr, xr, 1);
        ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
    }
    std::vector<float> expected((size_t)rows * low_dim), actual(expected.size());
    assert(ds4_gpu_tensor_read(low, 0, expected.data(), expected.size() * 4));
    for (unsigned n : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 15u, 16u, 17u, 32u, 33u}) {
        assert(ds4_gpu_tensor_fill_f32(low, NAN, expected.size()));
        invoke(low, x, n);
        assert(ds4_gpu_tensor_read(low, 0, actual.data(), actual.size() * 4));
        for (size_t i = 0; i < (size_t)n * low_dim; i++) {
            if (memcmp(&actual[i], &expected[i], 4)) {
                fprintf(stderr, "grouped Q8 width=%u rank=%u groups=%u rows=%u i=%zu: %.9g != %.9g\n",
                    width, rank, groups, n, i, actual[i], expected[i]);
                abort();
            }
        }
        for (size_t i = (size_t)n * low_dim; i < actual.size(); i++) assert(std::isnan(actual[i]));
        for (unsigned first = 0; first < groups; first++) {
            const unsigned count = groups - first;
            assert(ds4_gpu_tensor_fill_f32(low, NAN, expected.size()));
            assert(ds4_gpu_attention_output_low_q8_rows_exact_tensor(low,
                model.data(), size, 0, width, rank, groups, first, count, x, n));
            assert(ds4_gpu_tensor_read(low, 0, actual.data(), actual.size() * 4));
            for (unsigned r = 0; r < n; r++)
                assert(!memcmp(actual.data() + (size_t)r * count * rank,
                    expected.data() + (size_t)r * low_dim + first * rank,
                    count * rank * 4));
            for (size_t i = (size_t)n * count * rank; i < actual.size(); i++)
                assert(std::isnan(actual[i]));
        }
        if (bench) {
            for (unsigned sliced = 0; sliced < 2; sliced++) {
                std::vector<double> times;
                for (unsigned trial = 0; trial < 13; trial++) {
                    const auto begin = std::chrono::steady_clock::now();
                    for (unsigned repeat = 0; repeat < 20; repeat++) {
                        if (sliced) {
                            assert(ds4_gpu_attention_output_low_q8_rows_exact_tensor(low,
                                model.data(), size, 0, width, rank, groups, 1, groups - 1, x, n));
                        } else {
                            invoke(low, x, n);
                        }
                    }
                    assert(ds4_gpu_synchronize());
                    if (trial >= 2) times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin).count() / 20);
                }
                std::sort(times.begin(), times.end());
                printf("grouped Q8 width=%u rank=%u groups=%u rows=%u sliced=%u: %.3f ms\n",
                    width, rank, groups, n, sliced, times[times.size() / 2]);
            }
        }
    }
    ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    assert(close(fd) == 0);
    printf("grouped Q8 width=%u rank=%u groups=%u: scalar rows exact PASS\n", width, rank, groups);
}

/* Q8 K-slice rows (the TP o_b and shared-expert down halves): small and
 * prefill-sized batches must reproduce each scalar row, for both K halves,
 * ragged output counts and slices shorter than or not a multiple of a warp's
 * 32 blocks. Rows past the batch must stay untouched. */
static void run_kslice(unsigned in_dim, unsigned out_dim, unsigned split) {
    const unsigned rows = 129;
    const uint64_t size = (uint64_t)out_dim * ((in_dim + 31) / 32) * 34;
    std::vector<unsigned char> model(size);
    const int fd = map_random_q8(model);
    const unsigned starts[2] = {0, split}, counts[2] = {split, (in_dim & ~31u) - split};
    for (unsigned half = 0; half < 2; half++) {
        const unsigned start = starts[half], count = counts[half];
        std::vector<float> input((size_t)rows * count);
        for (float &v : input) v = ((int)(random_bits() >> 16) - 32768) / 8192.0f;
        auto *x = ds4_gpu_tensor_alloc(input.size() * 4);
        auto *out = ds4_gpu_tensor_alloc((uint64_t)rows * out_dim * 4);
        assert(x && out && ds4_gpu_tensor_write(x, 0, input.data(), input.size() * 4));
        assert(ds4_gpu_tensor_fill_f32(out, NAN, (uint64_t)rows * out_dim));
        for (unsigned r = 0; r < rows; r++) {
            auto *xr = ds4_gpu_tensor_view(x, (uint64_t)r * count * 4, (uint64_t)count * 4);
            auto *yr = ds4_gpu_tensor_view(out, (uint64_t)r * out_dim * 4, (uint64_t)out_dim * 4);
            assert(xr && yr);
            assert(ds4_gpu_matmul_q8_0_kslice_rows_tensor(yr, model.data(), size, 0,
                in_dim, out_dim, start, count, xr, 1));
            ds4_gpu_tensor_free(yr); ds4_gpu_tensor_free(xr);
        }
        std::vector<float> expected((size_t)rows * out_dim), actual(expected.size());
        assert(ds4_gpu_tensor_read(out, 0, expected.data(), expected.size() * 4));
        for (float v : expected) assert(!std::isnan(v));
        for (unsigned n : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 15u, 16u,
                           17u, 32u, 33u, 127u, 128u, 129u}) {
            assert(ds4_gpu_tensor_fill_f32(out, NAN, expected.size()));
            assert(ds4_gpu_matmul_q8_0_kslice_rows_tensor(out, model.data(), size, 0,
                in_dim, out_dim, start, count, x, n));
            assert(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * 4));
            for (size_t i = 0; i < (size_t)n * out_dim; i++) {
                if (memcmp(&actual[i], &expected[i], 4)) {
                    fprintf(stderr, "Q8 K-slice in=%u out=%u start=%u count=%u rows=%u i=%zu: %.9g != %.9g\n",
                        in_dim, out_dim, start, count, n, i, actual[i], expected[i]);
                    abort();
                }
            }
            for (size_t i = (size_t)n * out_dim; i < actual.size(); i++) assert(std::isnan(actual[i]));
        }
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(x);
    }
    ds4_gpu_cleanup();
    assert(close(fd) == 0);
    printf("Q8 K-slice in=%u out=%u split=%u: scalar rows exact PASS\n", in_dim, out_dim, split);
}

int main(int argc, char **argv) {
    const bool bench = argc == 2 && !strcmp(argv[1], "--bench");
    if (argc != 1 && !bench) return 2;
    setenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A", "1", 1);
    run(96, 65, 3, false);
    run(4096, 1024, 8, bench);
    run(8192, 1024, 4, bench);
    run_kslice(96, 5, 32);
    run_kslice(2407, 37, 1184);
    run_kslice(2048, 4096, 1024);
    run_kslice(8192, 4096, 4096);
    run_kslice(8192, 5120, 4096);
}
