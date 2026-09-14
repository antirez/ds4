#include "ds4_gpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t state = 593;
static uint32_t random_bits() {
    state = state * 1664525u + 1013904223u;
    return state;
}

static void check(uint32_t width, uint32_t hidden) {
    const size_t gu = (size_t)hidden * ((width + 31u) / 32u) * 34u;
    const size_t down = (size_t)width * ((hidden + 31u) / 32u) * 34u;
    std::vector<unsigned char> model(2u * gu + down);
    for (size_t b = 0; b < model.size(); b += 34u) {
        const uint16_t scale = 0x1800u + random_bits() % 1024u;
        memcpy(model.data() + b, &scale, 2);
        for (unsigned k = 2; k < 34; k++) model[b + k] = random_bits() >> 24;
    }
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_spark()) {
        fprintf(stderr, "CUDA shared-expert overlap requires Spark\n");
        exit(77);
    }
    CHECK(ds4_gpu_set_model_map(model.data(), model.size()));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(width * 4u);
    ds4_gpu_tensor *other = ds4_gpu_tensor_alloc(hidden * 4u);
    ds4_gpu_tensor *out[2], *gate[2], *up[2], *mid[2], *side[2];
    CHECK(x && other);
    for (unsigned i = 0; i < 2; i++) {
        out[i] = ds4_gpu_tensor_alloc((width + 8u) * 4u);
        gate[i] = ds4_gpu_tensor_alloc(hidden * 4u);
        up[i] = ds4_gpu_tensor_alloc(hidden * 4u);
        mid[i] = ds4_gpu_tensor_alloc(hidden * 4u);
        side[i] = ds4_gpu_tensor_alloc((width + 8u) * 4u);
        CHECK(out[i] && gate[i] && up[i] && mid[i] && side[i]);
    }
    auto start = [&](uint64_t offset) {
        return ds4_gpu_dsv41_shared_start(out[1], gate[1], up[1], mid[1], x,
            model.data(), model.size(), 0, gu, offset, width, hidden, 10.0f);
    };
    auto serial = [&]() {
        CHECK(ds4_gpu_matmul_q8_0_tensor(gate[0], model.data(), model.size(),
            0, width, hidden, x, 1));
        CHECK(ds4_gpu_dsv41_quantize(gate[0], hidden, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(up[0], model.data(), model.size(),
            gu, width, hidden, x, 1));
        CHECK(ds4_gpu_dsv41_quantize(up[0], hidden, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_swiglu_tensor(mid[0], gate[0], up[0], hidden, 10.0f, 1.0f));
        CHECK(ds4_gpu_dsv41_quantize(mid[0], hidden, 1, DS4_V41_BF16));
        CHECK(ds4_gpu_matmul_q8_0_tensor(out[0], model.data(), model.size(),
            2u * gu, hidden, width, mid[0], 1));
        CHECK(ds4_gpu_dsv41_quantize(out[0], width, 1, DS4_V41_BF16));
    };
    auto unrelated = [&](unsigned i) {
        CHECK(ds4_gpu_matmul_q8_0_tensor(side[i], model.data(), model.size(),
            2u * gu, hidden, width, other, 1));
    };
    std::vector<float> input(width), other_input(hidden);
    std::vector<float> expected(width + 8u), actual(width + 8u);
    ds4_decode_graph_key key = {};
    key.il = 0; key.island = 1; key.cur_hc = x;
    unsigned captures = 0, replays = 0;
    for (unsigned round = 0; round < 10; round++) {
        for (float &v : input) v = ((int32_t)(random_bits() >> 16) - 32768) / 8192.0f;
        for (float &v : other_input) v = ((int32_t)(random_bits() >> 16) - 32768) / 32768.0f;
        CHECK(ds4_gpu_tensor_write(x, 0, input.data(), width * 4u));
        CHECK(ds4_gpu_tensor_write(other, 0, other_input.data(), hidden * 4u));
        for (unsigned i = 0; i < 2; i++) {
            CHECK(ds4_gpu_tensor_fill_f32(out[i], NAN, width + 8u));
            CHECK(ds4_gpu_tensor_fill_f32(side[i], NAN, width + 8u));
        }
        serial(); unrelated(0);
        CHECK(ds4_gpu_add_tensor(out[0], out[0], side[0], width));
        CHECK(ds4_gpu_tensor_read(out[0], 0, expected.data(), expected.size() * 4u));
        if (round == 0) {
            CHECK(start(model.size()) == -1); /* Error after gate/up launch. */
            CHECK(ds4_gpu_dsv41_shared_join());
        }
        const int graph = round < 2 ? -1 : ds4_gpu_decode_graph_begin(&key);
        if (graph != 1) {
            CHECK(start(2u * gu) == 1);
            CHECK(start(2u * gu) == -1); /* No nested scratch owner. */
            unrelated(1);
            CHECK(ds4_gpu_dsv41_shared_join());
            CHECK(ds4_gpu_dsv41_shared_join());
            CHECK(ds4_gpu_add_tensor(out[1], out[1], side[1], width));
            if (graph == 0) { CHECK(ds4_gpu_decode_graph_end(&key) == 0); captures++; }
        } else replays++;
        CHECK(ds4_gpu_tensor_read(out[1], 0, actual.data(), actual.size() * 4u));
        CHECK(!memcmp(expected.data(), actual.data(), width * 4u));
        for (size_t i = width; i < actual.size(); i++) CHECK(std::isnan(actual[i]));
        CHECK(ds4_gpu_tensor_read(side[0], 0, expected.data(), expected.size() * 4u));
        CHECK(ds4_gpu_tensor_read(side[1], 0, actual.data(), actual.size() * 4u));
        CHECK(!memcmp(expected.data(), actual.data(), width * 4u));
        for (size_t i = width; i < actual.size(); i++) CHECK(std::isnan(actual[i]));
    }
    CHECK(captures == 1 && replays == 6);
    key.variant = 1;
    CHECK(ds4_gpu_decode_graph_begin(&key) == -1);
    CHECK(start(2u * gu) == 1 && ds4_gpu_dsv41_shared_join());
    CHECK(ds4_gpu_synchronize());
    CHECK(ds4_gpu_decode_graph_begin(&key) == 0);
    CHECK(start(model.size()) == -1);
    ds4_gpu_decode_graph_abort(&key);
    CHECK(start(2u * gu) == 1 && ds4_gpu_dsv41_shared_join());
    serial();
    CHECK(ds4_gpu_tensor_read(out[0], 0, expected.data(), width * 4u));
    CHECK(ds4_gpu_tensor_read(out[1], 0, actual.data(), width * 4u));
    CHECK(!memcmp(expected.data(), actual.data(), width * 4u));

    ds4_gpu_tensor *large_x = ds4_gpu_tensor_alloc(65536u * 4u);
    ds4_gpu_tensor *large_out = ds4_gpu_tensor_alloc(65536u * 4u);
    CHECK(large_x && large_out && ds4_gpu_tensor_fill_f32(large_out, NAN, 65536u));
    CHECK(ds4_gpu_dsv41_shared_start(large_out, gate[1], up[1], mid[1], large_x,
        model.data(), model.size(), 0, gu, 2u * gu, 65536u, 1u, 10.0f) == 0);
    CHECK(ds4_gpu_dsv41_shared_join());
    std::vector<float> untouched(65536u);
    CHECK(ds4_gpu_tensor_read(large_out, 0, untouched.data(), untouched.size() * 4u));
    for (float value : untouched) CHECK(std::isnan(value));
    ds4_gpu_tensor_free(large_out); ds4_gpu_tensor_free(large_x);
    CHECK(ds4_gpu_synchronize());
    ds4_gpu_decode_graphs_invalidate();
    for (unsigned i = 0; i < 2; i++) {
        ds4_gpu_tensor_free(out[i]); ds4_gpu_tensor_free(gate[i]);
        ds4_gpu_tensor_free(up[i]); ds4_gpu_tensor_free(mid[i]);
        ds4_gpu_tensor_free(side[i]);
    }
    ds4_gpu_tensor_free(other); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    printf("CUDA shared expert %u x %u: serial, concurrent, capture, replay, recovery exact PASS\n",
        width, hidden);
}

int main() {
    for (unsigned repeat = 0; repeat < 2; repeat++) {
        check(33, 65);
        check(128, 64);
        check(1280, 257);
        check(5120, 2304);
    }
    return 0;
}
