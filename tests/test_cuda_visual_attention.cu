#include "ds4_gpu.h"
#include "ds4_image.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

struct shape {
    uint32_t nt, pos, nc, nh, dim, mask;
    bool quality;
};

static size_t mismatched_cases;
static bool benchmark;
static bool fractional;

static float value(size_t i, unsigned seed) {
    uint32_t x = (uint32_t)i * 747796405u + seed * 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    if (fractional) return ((int)(x % 65521u) - 32760) / 65537.0f;
    return ((int)(x % 31u) - 15) / 32.0f;
}

static ds4_gpu_tensor *upload(const std::vector<float> &v) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(v.size() * sizeof(float));
    CHECK(t);
    CHECK(ds4_gpu_tensor_write(t, 0, v.data(), v.size() * sizeof(float)));
    return t;
}

static void run_case(const shape &s, float *model, FILE *dump, FILE *oracle,
                     bool memory_check) {
    const uint32_t nr = s.nt + std::min(s.pos, 19u);
    const uint32_t cap = nr + 13u, start = cap - 7u;
    const uint32_t first = s.pos + s.nt - nr, nk = nr + s.nc;
    const uint32_t ratio = s.nc ? 4u : 0u, window = 256u, vocab = 128u;
    const size_t count = (size_t)s.nt * s.nh * s.dim;
    std::vector<float> q(count), raw((size_t)cap * s.dim);
    std::vector<float> comp((size_t)std::max(s.nc, 1u) * s.dim);
    std::vector<float> mask((size_t)s.nt * std::max(s.nc, 1u));
    std::vector<float> output(count + 32u, 12345.25f);
    for (size_t i = 0; i < q.size(); i++) q[i] = value(i, 3);
    for (size_t i = 0; i < raw.size(); i++) raw[i] = value(i, 5);
    for (size_t i = 0; i < comp.size(); i++) comp[i] = value(i, 7);
    for (size_t i = 0; i < mask.size(); i++)
        mask[i] = s.mask == 2 || i % 3u == 0 ? -INFINITY : -0.25f;
    std::vector<int32_t> tokens(s.nt, 11);
    if (s.nt >= 7u) {
        tokens[1] = vocab + DS4_DEEPSEEK4_IMAGE_PAD;
        tokens[2] = vocab + DS4_DEEPSEEK4_IMAGE_START;
        tokens[3] = vocab + DS4_DEEPSEEK4_IMAGE;
        tokens[4] = vocab + DS4_DEEPSEEK4_IMAGE_NEWLINE;
        tokens[5] = vocab + DS4_DEEPSEEK4_IMAGE;
        tokens[6] = vocab + DS4_DEEPSEEK4_IMAGE_END;
    }
    if (s.nt >= 17u) {
        tokens[s.nt-4u] = vocab + DS4_DEEPSEEK4_IMAGE_START;
        tokens[s.nt-3u] = vocab + DS4_DEEPSEEK4_IMAGE;
        tokens[s.nt-2u] = vocab + DS4_DEEPSEEK4_IMAGE_END;
    }
    ds4_gpu_tensor *dq = upload(q), *dr = upload(raw), *dc = upload(comp);
    ds4_gpu_tensor *dm = upload(mask), *storage = upload(output);
    ds4_gpu_tensor *dst = ds4_gpu_tensor_view(storage, 16u*sizeof(float), count*sizeof(float));
    CHECK(dst);
    ds4_gpu_set_quality(s.quality);
    size_t free_before = 0, total = 0, free_after = 0;
    CHECK(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
    cudaEvent_t begin, end;
    CHECK(cudaEventCreate(&begin) == cudaSuccess);
    CHECK(cudaEventCreate(&end) == cudaSuccess);
    auto infer = [&]() {
      CHECK(ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        dst, model, 4096, 0, dq, dr, s.nc ? dc : NULL, 0,
        s.mask ? dm : NULL, s.mask != 0, tokens.data(), vocab,
        s.nt, s.pos, nr, cap, start, s.nc, window, ratio, s.nh, s.dim));
    };
    CHECK(cudaEventRecord(begin) == cudaSuccess);
    infer();
    CHECK(cudaEventRecord(end) == cudaSuccess);
    CHECK(cudaEventSynchronize(end) == cudaSuccess);
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, begin, end) == cudaSuccess);
    if (benchmark) {
        for (int i = 0; i < 3; i++) infer();
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> samples;
        for (int i = 0; i < 10; i++) {
            CHECK(cudaEventRecord(begin) == cudaSuccess);
            infer();
            CHECK(cudaEventRecord(end) == cudaSuccess);
            CHECK(cudaEventSynchronize(end) == cudaSuccess);
            CHECK(cudaEventElapsedTime(&ms, begin, end) == cudaSuccess);
            samples.push_back(ms);
        }
        std::sort(samples.begin(), samples.end());
        ms = (samples[4] + samples[5]) / 2;
    }
    CHECK(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    CHECK(ds4_gpu_tensor_read(storage, 0, output.data(), output.size()*sizeof(float)));
    for (size_t i = 0; i < 16; i++) {
        CHECK(output[i] == 12345.25f);
        CHECK(output[count+16+i] == 12345.25f);
    }
    for (size_t i = 16; i < count+16; i++) CHECK(std::isfinite(output[i]));
    if (dump) CHECK(fwrite(output.data()+16, sizeof(float), count, dump) == count);
    if (oracle) {
        std::vector<float> expected(count);
        CHECK(fread(expected.data(), sizeof(float), count, oracle) == count);
        size_t differences = 0;
        double max_error = 0;
        for (size_t i = 0; i < count; i++) {
            differences += memcmp(&expected[i], &output[i+16], sizeof(float)) != 0;
            max_error = std::max(max_error, fabs((double)expected[i]-output[i+16]));
        }
        if (differences) fprintf(stderr, "nt=%u heads=%u dim=%u quality=%d: %zu/%zu differing floats, max %.9g\n",
            s.nt, s.nh, s.dim, s.quality, differences, count, max_error);
        if (differences) mismatched_cases++;
    }
    /* Independent double-precision oracle: every small output, and selected
     * rows/components across head-group/image boundaries for large calls.
     * Exact backend-to-backend comparison above separately covers TF32 mode. */
    if (s.quality) {
        const bool sampled = (uint64_t)s.nt*s.nh*nk*s.dim >= 12000000u;
        std::vector<uint32_t> bounds((size_t)s.nt*2u);
        CHECK(ds4_deepseek4_attention_bounds((const int *)tokens.data(), s.nt,
                                            vocab, s.pos, nr, window, bounds.data()));
        std::vector<double> scores(nk);
        for (uint32_t t = 0; t < s.nt; t++) for (uint32_t h = 0; h < s.nh; h++) {
            if (sampled && t != 0 && t != std::min(3u,s.nt-1u) && t != s.nt-1u) continue;
            if (sampled && h != 0 && h != std::min(7u,s.nh-1u) &&
                h != std::min(8u,s.nh-1u) && h != s.nh-1u) continue;
            double maximum = model[h];
            const uint32_t visible = ratio ? std::min(s.nc, (s.pos+t+1u)/ratio) : 0;
            for (uint32_t k = 0; k < nk; k++) {
                const uint32_t c = k-nr;
                const bool allowed = k < nr
                    ? first+k >= bounds[t*2u] && first+k <= bounds[t*2u+1u]
                    : c < visible && (!s.mask || mask[(size_t)t*s.nc+c] > -1e20f);
                double v = -INFINITY;
                if (allowed) {
                    const float *kv = k < nr ? raw.data() + (size_t)((start+k)%cap)*s.dim
                                            : comp.data() + (size_t)c*s.dim;
                    v = 0;
                    for (uint32_t d = 0; d < s.dim; d++)
                        v += (double)q[((size_t)t*s.nh+h)*s.dim+d] * kv[d];
                    v /= sqrt((double)s.dim);
                    if (k >= nr && s.mask) v += mask[(size_t)t*s.nc+c];
                }
                scores[k] = v;
                maximum = std::max(maximum, v);
            }
            double denom = exp(model[h]-maximum);
            for (uint32_t k = 0; k < nk; k++) denom += scores[k] = exp(scores[k]-maximum);
            for (uint32_t d = 0; d < s.dim; d++) {
                if (sampled && d != 0 && d != s.dim/2u && d != s.dim-1u) continue;
                double expected = 0;
                for (uint32_t k = 0; k < nk; k++) {
                    const float *kv = k < nr ? raw.data() + (size_t)((start+k)%cap)*s.dim
                                            : comp.data() + (size_t)(k-nr)*s.dim;
                    expected += scores[k] * kv[d] / denom;
                }
                CHECK(fabs(expected-output[16+((size_t)t*s.nh+h)*s.dim+d]) < 2e-5);
            }
        }
    }
    const double growth = free_before > free_after ? (free_before-free_after)/1048576.0 : 0;
    printf("nt=%u pos=%u comp=%u heads=%u dim=%u mask=%u quality=%d %s=%.4f scratch_growth_MiB=%.2f\n",
           s.nt,s.pos,s.nc,s.nh,s.dim,s.mask,s.quality,
           benchmark ? "warmed_median_ms" : "ms",ms,growth);
    fflush(stdout);
    if (memory_check) CHECK(growth < 160.0);
    CHECK(cudaEventDestroy(begin) == cudaSuccess);
    CHECK(cudaEventDestroy(end) == cudaSuccess);
    ds4_gpu_tensor_free(dst); ds4_gpu_tensor_free(storage);
    ds4_gpu_tensor_free(dm); ds4_gpu_tensor_free(dc);
    ds4_gpu_tensor_free(dr); ds4_gpu_tensor_free(dq);
}

int main(int argc, char **argv) {
    FILE *dump = NULL, *oracle = NULL;
    bool memory = false, large = false, boundaries = false;
    for (int i=1;i<argc;i++) {
        if (!strcmp(argv[i],"--dump") && i+1<argc) { dump=fopen(argv[++i],"wb"); CHECK(dump); }
        else if (!strcmp(argv[i],"--compare") && i+1<argc) { oracle=fopen(argv[++i],"rb"); CHECK(oracle); }
        else if (!strcmp(argv[i],"--memory")) memory=true;
        else if (!strcmp(argv[i],"--large")) large=true;
        else if (!strcmp(argv[i],"--boundaries")) boundaries=true;
        else if (!strcmp(argv[i],"--bench")) benchmark=true;
        else if (!strcmp(argv[i],"--fractional")) fractional=true;
        else { fprintf(stderr,"bad argument\n"); return 1; }
    }
    int devices=0;
    CHECK(cudaGetDeviceCount(&devices)==cudaSuccess && devices>0);
    CHECK(ds4_gpu_init());
    float *model=NULL;
    CHECK(cudaMallocHost((void **)&model,4096)==cudaSuccess);
    for (size_t i=0;i<1024;i++) model[i]=value(i,17);
    /* This synthetic model is only 4 KiB. Make its sinks device-resident,
     * avoiding unsupported read-only host registration on integrated GPUs. */
    CHECK(setenv("DS4_CUDA_COPY_MODEL", "1", 1)==0);
    CHECK(ds4_gpu_set_model_map(model,4096));
    CHECK(unsetenv("DS4_CUDA_COPY_MODEL")==0);
    if (memory) {
        /* Exclude one-time CUDA/cuBLAS initialization from the scratch delta. */
        run_case({7,0,0,1,32,0,false},model,NULL,NULL,false);
        run_case({257,32761,8192,64,512,1,false},model,dump,oracle,true);
    } else if (boundaries) {
        for (bool quality : {false,true}) {
            /* Immediately below, at, and above the 256 MiB score cutoff. */
            for (uint32_t nc : {32716u,32717u,32718u})
                run_case({32,129001,nc,64,512,1,quality},model,dump,oracle,false);
            /* Large calls must also cover incomplete final head groups. */
            for (uint32_t nh : {9u,17u,63u})
                run_case({257,129001,32768,nh,512,1,quality},model,dump,oracle,false);
        }
    } else if (benchmark) {
        run_case({1,32767,8192,64,512,1,false},model,dump,oracle,false);
        run_case({32,129001,32769,64,512,1,false},model,dump,oracle,false);
        run_case({469,130603,32768,64,512,1,false},model,dump,oracle,false);
        run_case({1579,137216,35000,64,512,1,false},model,dump,oracle,false);
        run_case({2048,260096,65536,64,512,1,false},model,dump,oracle,false);
    } else {
        for (bool quality : {false,true}) {
            for (uint32_t nh : {1u,7u,8u,9u,17u,63u,64u}) {
                run_case({7,0,0,nh,32,0,quality},model,dump,oracle,false);
                run_case({17,113,37,nh,32,1,quality},model,dump,oracle,false);
                run_case({31,2047,513,nh,32,2,quality},model,dump,oracle,false);
            }
            run_case({1,32767,8192,64,512,1,quality},model,dump,oracle,false);
            run_case({32,129001,32769,64,512,1,quality},model,dump,oracle,false);
            if (large) {
                run_case({469,130603,32768,64,512,1,quality},model,dump,oracle,false);
                run_case({1579,137216,35000,64,512,1,quality},model,dump,oracle,false);
                run_case({2048,260096,65536,64,512,1,quality},model,dump,oracle,false);
            }
        }
    }
    if (dump) CHECK(fclose(dump)==0);
    if (oracle) { CHECK(fgetc(oracle)==EOF); CHECK(fclose(oracle)==0); }
    ds4_gpu_cleanup();
    CHECK(cudaFreeHost(model)==cudaSuccess);
    CHECK(mismatched_cases == 0);
    puts("PASS: CUDA visual attention");
    return 0;
}
