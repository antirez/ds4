/* Check rank-local artifact geometry and sparse execution against the raw
 * whole expert table and independent raw shards. No model download needed. */
#include "ds4_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
static uint32_t rng = 41;
static unsigned char byte(void) { rng = rng * 1664525u + 1013904223u; return rng >> 24; }
template<class T> static void append(std::vector<unsigned char> &v, T x) {
    const auto *p = reinterpret_cast<const unsigned char *>(&x);
    v.insert(v.end(), p, p + sizeof(x));
}
static void string(std::vector<unsigned char> &v, const char *s) {
    append<uint64_t>(v, std::strlen(s));
    v.insert(v.end(), s, s + std::strlen(s));
}
static ds4_gpu_tensor *tensor(uint64_t bytes) {
    auto *t = ds4_gpu_tensor_alloc(bytes);
    CHECK(t);
    return t;
}

static void run(unsigned dim, unsigned mid_dim, unsigned experts) {
    constexpr unsigned slots = 6, rows = 256;
    const uint64_t gate_row = dim / 256u * 66u, down_row = mid_dim / 256u * 84u;
    const uint64_t unit[] = {gate_row * mid_dim, gate_row * mid_dim, down_row * dim};
    const char *names[] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
    std::vector<unsigned char> model;
    append<uint32_t>(model, 0x46554747);
    append<uint32_t>(model, 3);
    append<uint64_t>(model, 4);
    append<uint64_t>(model, 0);
    uint64_t off[3], payload = 0;
    for (unsigned i = 0; i < 3; i++) {
        string(model, names[i]);
        append<uint32_t>(model, 3);
        append<uint64_t>(model, i == 2 ? mid_dim : dim);
        append<uint64_t>(model, i == 2 ? dim : mid_dim);
        append<uint64_t>(model, experts);
        append<uint32_t>(model, i == 2 ? 10 : 16);
        append<uint64_t>(model, payload);
        off[i] = payload;
        payload += unit[i] * experts;
    }
    string(model, "blk.1.engram_embd.weight");
    append<uint32_t>(model, 2);
    append<uint64_t>(model, 16);
    append<uint64_t>(model, 64);
    append<uint32_t>(model, 0);
    append<uint64_t>(model, payload);
    const size_t header = (model.size() + 31u) & ~size_t(31);
    const size_t mapped_size = header + payload;
    model.resize(mapped_size + 4096);
    for (unsigned i = 0; i < 3; i++) {
        off[i] += header;
        const size_t block = i == 2 ? 84 : 66;
        for (size_t p = 0; p < unit[i] * experts; p++) model[off[i] + p] = byte();
        for (size_t p = 0; p < unit[i] * experts; p += block) {
            const size_t d = off[i] + p + (i == 2 ? 80 : 0);
            model[d] = 0; model[d + 1] = 0x14;
            if (i == 2) { model[d + 2] = 0; model[d + 3] = 0x10; }
        }
    }
    char path[] = "/tmp/ds4-tp-repack-XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    FILE *f = fdopen(fd, "wb");
    CHECK(f && std::fwrite(model.data(), 1, model.size(), f) == model.size());
    CHECK(std::fclose(f) == 0);
    std::vector<float> input(rows * dim), weights(rows * slots), one(dim);
    std::vector<int32_t> ids(rows * slots);
    for (auto &v : input) v = ((int)byte() - 128) / 128.0f;
    for (unsigned r = 0; r < rows; r++) for (unsigned s = 0; s < slots; s++) {
        /* Include completely empty contributions from either rank. */
        ids[r * slots + s] = r % 3 == 0 ? s : r % 3 == 1 ? experts / 2 + s : (r * 7 + s * 3) % experts;
        weights[r * slots + s] = (1.0f + byte()) / (256 * slots);
    }
    const unsigned counts[] = {1, 2, 8, 9, 127, 128, 129, 256};
    std::vector<std::vector<float>> reference, partial, owned_reference[2];
    for (unsigned pass = 0; pass < 5; pass++) {
        const bool owned = pass != 0;
        const bool raw = pass == 0 || pass >= 3;
        const unsigned rank = owned ? (pass - 1) % 2 : 0;
        CHECK(ds4_gpu_init());
        CHECK(ds4_gpu_build_derived_artifacts_shard(model.data(), mapped_size, model.size(), path, 2) == 0);
        CHECK(ds4_gpu_build_derived_artifacts_shard(model.data(), header, model.size(), path, 0) == 0);
        if (!raw) CHECK(ds4_gpu_build_derived_artifacts_shard(
            model.data(), mapped_size, model.size(), path, rank) == 3);
        uint64_t offsets[3], sizes[3];
        for (unsigned i = 0; i < 3; i++) {
            sizes[i] = unit[i] * experts / (owned ? 2 : 1);
            offsets[i] = off[i] + rank * sizes[i];
            CHECK(!!ds4_gpu_model_range_replaced(model.data(), offsets[i], sizes[i]) == !raw);
            if (owned) {
                CHECK(!ds4_gpu_model_range_replaced(model.data(), off[i], unit[i] * experts));
                CHECK(!ds4_gpu_model_range_replaced(model.data(), off[i] + (rank == 0 ? sizes[i] : 0), sizes[i]));
            }
        }
        CHECK(ds4_gpu_set_model_map_spans(model.data(), mapped_size, offsets, sizes, 3, 0));
        for (unsigned i = 0; i < 3; i++)
            CHECK(ds4_gpu_cache_model_range(model.data(), model.size(), offsets[i], sizes[i], "owned"));
        auto *x = tensor(input.size() * 4), *sel = tensor(ids.size() * 4), *sw = tensor(weights.size() * 4);
        auto *out = tensor((rows + 1) * dim * 4);
        auto *gate = tensor(rows * slots * mid_dim * 4), *up = tensor(rows * slots * mid_dim * 4);
        auto *mid = tensor(rows * slots * mid_dim * 4), *down = tensor(rows * slots * dim * 4);
        auto invoke = [&](unsigned n) {
            bool half = false;
            const bool ok = owned ? ds4_gpu_routed_moe_batch_owned_tensor(out, gate, up, mid, down,
                model.data(), model.size(), off[0], off[1], off[2], 16, 10, unit[0], gate_row, unit[2], down_row,
                dim, mid_dim, dim, sel, sw, experts, slots, rank * (experts / 2), experts / 2,
                10.0f, x, 0, n, &half) : ds4_gpu_routed_moe_batch_tensor(out, gate, up, mid, down,
                model.data(), model.size(), off[0], off[1], off[2], 16, 10, unit[0], gate_row, unit[2], down_row,
                dim, mid_dim, dim, sel, sw, experts, slots, 10.0f, x, 0, n, &half, false);
            CHECK(ok && !half);
        };
        for (unsigned c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
            const unsigned n = counts[c];
            CHECK(ds4_gpu_tensor_write(x, 0, input.data(), n * dim * 4));
            CHECK(ds4_gpu_tensor_write(sel, 0, ids.data(), n * slots * 4));
            CHECK(ds4_gpu_tensor_write(sw, 0, weights.data(), n * slots * 4));
            CHECK(ds4_gpu_tensor_fill_f32(gate, NAN, n * slots * mid_dim));
            CHECK(ds4_gpu_tensor_fill_f32(up, NAN, n * slots * mid_dim));
            CHECK(ds4_gpu_tensor_fill_f32(mid, NAN, n * slots * mid_dim));
            CHECK(ds4_gpu_tensor_fill_f32(down, 12345, n * slots * dim));
            CHECK(ds4_gpu_tensor_fill_f32(out, 12345, (n + 1) * dim));
            invoke(n);
            std::vector<float> actual((n + 1) * dim);
            CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * 4));
            for (unsigned i = 0; i < dim; i++) CHECK(actual[n * dim + i] == 12345);
            actual.resize(n * dim);
            for (unsigned i = 0; i < n * dim; i++) {
                CHECK(std::isfinite(actual[i]));
                if (owned && (i / dim) % 3 == (rank == 0 ? 1 : 0)) CHECK(actual[i] == 0);
                if (pass == 2 && n > 8) {
                    const float expected = reference[c][i], sum = partial[c][i] + actual[i];
                    if (std::fabs(sum - expected) > 0.0002f + std::fabs(expected) * 0.00002f) {
                        std::fprintf(stderr, "aligned shard mismatch rows=%u i=%u %.9g != %.9g\n", n, i, sum, expected);
                        std::exit(1);
                    }
                }
            }
            if (pass == 0) reference.push_back(actual);
            if (pass == 1) partial.push_back(actual);
            if (owned && !raw) owned_reference[rank].push_back(actual);
            if (raw && owned) {
                double square = 0, reference_square = 0;
                float max_abs = 0;
                size_t different = 0;
                for (unsigned i = 0; i < n * dim; i++) {
                    const float expected = owned_reference[rank][c][i];
                    const double delta = (double)actual[i] - expected;
                    square += delta * delta;
                    reference_square += (double)expected * expected;
                    max_abs = std::max(max_abs, (float)std::fabs(delta));
                    different += std::memcmp(&actual[i], &expected, sizeof(float)) != 0;
                }
                std::printf("raw/aligned rank=%u rows=%u different=%zu max_abs=%.9g rel_rms=%.9g\n",
                    rank, n, different, max_abs, std::sqrt(square / std::max(reference_square, 1e-30)));
                if (n <= 8 || (n >= 128 && (dim == 5120 || experts == 384)))
                    CHECK(different == 0);
            }
            if (owned && n <= 8) for (unsigned r = 0; r < n; r++) {
                CHECK(ds4_gpu_tensor_write(x, 0, input.data() + r * dim, dim * 4));
                CHECK(ds4_gpu_tensor_write(sel, 0, ids.data() + r * slots, slots * 4));
                CHECK(ds4_gpu_tensor_write(sw, 0, weights.data() + r * slots, slots * 4));
                invoke(1);
                CHECK(ds4_gpu_tensor_read(out, 0, one.data(), dim * 4));
                CHECK(!std::memcmp(one.data(), actual.data() + r * dim, dim * 4));
            }
        }
        for (auto *t : {x, sel, sw, out, gate, up, mid, down}) ds4_gpu_tensor_free(t);
        ds4_gpu_cleanup();
    }
    CHECK(unlink(path) == 0);
    std::printf("aligned TP %u/%u experts=%u: owned geometry, empty ranks, scalar rows, prefill PASS\n", dim, mid_dim, experts);
}

int main(void) {
    run(1024, 256, 16);
    run(5120, 2304, 16);
    run(1024, 256, 384);
    return 0;
}
