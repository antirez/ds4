/* Inkling 2 Metal kernel checks that need no model file: a NaN router score must never select expert 255 (the old pick read past
 * the 128-expert tensors), and a NaN or infinity must never be stored in the K/V cache (the cache survives a session reset and
 * tiled attention multiplies masked rows by 0, so one bad row would poison every later request).  Metal builds only. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include "ds4.h"
#include "ds4_gpu.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static int finite_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x7f800000u) != 0x7f800000u; }

static void test_router_nan(void) {
    enum { N = 3, NEXP = 128 };
    float *lg = calloc(N * NEXP, sizeof(float));
    const uint32_t nanb = 0x7fc00000u;
    for (int e = 0; e < NEXP; e++) {
        lg[1 * NEXP + e] = ((e * 37) % 101) / 50.0f - 1.0f;
        lg[2 * NEXP + e] = ((e * 53) % 97) / 40.0f - 1.2f;
        memcpy(&lg[0 * NEXP + e], &nanb, 4);                                           /* token 0: every logit NaN */
    }
    memcpy(&lg[1 * NEXP + 5], &nanb, 4);
    memcpy(&lg[1 * NEXP + 77], &nanb, 4);                                              /* token 1: two NaN logits */
    const uint32_t infb = 0x7f800000u;
    memcpy(&lg[2 * NEXP + 9], &infb, 4);                                               /* token 2: one +inf logit */
    ds4_gpu_tensor *tl = ds4_gpu_tensor_alloc(N * NEXP * 4), *tp = ds4_gpu_tensor_alloc(N * 8 * 4), *tw = ds4_gpu_tensor_alloc(N * 8 * 4);
    CHECK(tl && tp && tw && ds4_gpu_tensor_write(tl, 0, lg, N * NEXP * 4), "alloc");
    ds4_gpu_ik_w wb = {0};   /* absent weight: the wrapper binds a zeroed dummy buffer, i.e. all router biases are 0 */
    ds4_gpu_ik_ref none = {0}, rp = { tp, 0 }, rw = { tw, 0 }, rl = { tl, 0 };
    CHECK(ds4_gpu_ik_router(rp, rw, rl, wb, none, 0, 0, 0, N), "router dispatch");
    int32_t picks[N * 8];
    float wts[N * 8];
    CHECK(ds4_gpu_tensor_read(tp, 0, picks, sizeof(picks)) && ds4_gpu_tensor_read(tw, 0, wts, sizeof(wts)), "read back");
    if (getenv("IK_TEST_VERBOSE")) for (int t = 0; t < N; t++) { for (int k = 0; k < 8; k++) fprintf(stderr, "%d(%.3f) ", picks[t * 8 + k], wts[t * 8 + k]); fputc(10, stderr); }
    for (int t = 0; t < N; t++) {
        for (int k = 0; k < 8; k++) {
            const int p = picks[t * 8 + k];
            CHECK(p >= 0 && p < NEXP, "token %d pick %d is %d (must be 0..127)", t, k, p);
            CHECK(finite_bits(wts[t * 8 + k]), "token %d weight %d is not finite", t, k);
            for (int j = 0; j < k; j++) CHECK(p != picks[t * 8 + j], "token %d picks expert %d twice", t, p);
            if (t == 1) CHECK(p != 5 && p != 77, "token 1 picked an expert whose logit is NaN (%d)", p);
        }
    }
    for (int k = 0; k < 8; k++) CHECK(picks[k] == k, "all-NaN token: pick %d is %d, want the lowest free id %d", k, picks[k], k);
    ds4_gpu_tensor_free(tl);
    ds4_gpu_tensor_free(tp);
    ds4_gpu_tensor_free(tw);
    free(lg);
}

static void test_kv_store_nan(void) {
    enum { N = 2, ROWS = 4 };
    float K[N * 1024], V[N * 1024];
    for (int i = 0; i < N * 1024; i++) {
        K[i] = 0.01f * (float)((i % 17) - 8);
        V[i] = 0.02f * (float)((i % 13) - 6) + 0.1f;
    }
    const uint32_t nanb = 0x7fc00000u, infb = 0x7f800000u;
    memcpy(&K[3], &nanb, 4);      /* token 0 K: NaN in kv head 0 */
    memcpy(&K[200], &nanb, 4);    /* and in kv head 1 */
    memcpy(&V[700], &infb, 4);    /* token 0 V: infinity */
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc(sizeof(K)), *tv = ds4_gpu_tensor_alloc(sizeof(V));
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc(ROWS * 1024 * 2), *vc = ds4_gpu_tensor_alloc(ROWS * 1024 * 2);
    static uint16_t zero[ROWS * 1024];
    CHECK(tk && tv && kc && vc && ds4_gpu_tensor_write(tk, 0, K, sizeof(K)) && ds4_gpu_tensor_write(tv, 0, V, sizeof(V)) &&
          ds4_gpu_tensor_write(kc, 0, zero, sizeof(zero)) && ds4_gpu_tensor_write(vc, 0, zero, sizeof(zero)), "alloc");
    ds4_gpu_ik_ref rk = { tk, 0 }, rv = { tv, 0 }, rkc = { kc, 0 }, rvc = { vc, 0 };
    CHECK(ds4_gpu_ik_kv_store(rk, rv, rkc, rvc, 0, ROWS, N), "kv store dispatch");
    static uint16_t hk[ROWS * 1024], hv[ROWS * 1024];
    CHECK(ds4_gpu_tensor_read(kc, 0, hk, sizeof(hk)) && ds4_gpu_tensor_read(vc, 0, hv, sizeof(hv)), "read back");
    int bad = 0, nonzero1 = 0;
    for (int i = 0; i < N * 1024; i++) {
        if ((hk[i] & 0x7c00u) == 0x7c00u) bad++;
        if ((hv[i] & 0x7c00u) == 0x7c00u) bad++;
        if (i >= 1024 && hk[i]) nonzero1++;
    }
    CHECK(bad == 0, "%d NaN/infinity half values reached the K/V cache", bad);
    CHECK(nonzero1 > 500, "the clean token 1 was not stored (%d non-zero K values)", nonzero1);
    ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tv);
    ds4_gpu_tensor_free(kc);
    ds4_gpu_tensor_free(vc);
}

/* Decode attention over a long context: the split-keys kernels (kernel_ik_attn_split + kernel_ik_attn_reduce, used for one token
 * above 128 keys) must give the same result as the plain one-token kernel (reference: part buffer absent), for a global slot and a
 * sliding-window slot, at several positions including the ring wrap. */
static void test_attn_split_matches_plain(void) {
    enum { ROWS_G = 6000, ROWS_L = 1536, NH = 24 };
    const uint32_t cases[][3] = { {200, 0, 0}, {1000, 0, 1}, {3000, 0, 1}, {5999, 0, 0}, {3000, 1, 0}, {1500, 1, 1}, {200, 1, 0}, {1023, 1, 1} };   /* p0, local, relative-bias dummy unused */
    uint32_t rng = 7;
    #define RNG() (rng = rng * 1664525u + 1013904223u, (float)((rng >> 8) & 0xffff) / 65536.0f - 0.5f)
    static _Float16 kc[ROWS_G * 1024], vc[ROWS_G * 1024];
    for (size_t i = 0; i < (size_t)ROWS_G * 1024; i++) { kc[i] = (_Float16)(RNG() * 2.0f); vc[i] = (_Float16)(RNG() * 2.0f); }
    float Q[NH * 128], R[NH * 16], sink[NH], gate[NH * 128];
    for (int i = 0; i < NH * 128; i++) { Q[i] = RNG() * 4.0f; gate[i] = RNG() * 4.0f; }
    for (int i = 0; i < NH * 16; i++) R[i] = RNG();
    for (int i = 0; i < NH; i++) sink[i] = RNG();
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc(sizeof(Q)), *tr = ds4_gpu_tensor_alloc(sizeof(R)), *ts = ds4_gpu_tensor_alloc(sizeof(sink));
    ds4_gpu_tensor *tg = ds4_gpu_tensor_alloc(sizeof(gate)), *tk = ds4_gpu_tensor_alloc(sizeof(kc)), *tv = ds4_gpu_tensor_alloc(sizeof(vc));
    ds4_gpu_tensor *o1 = ds4_gpu_tensor_alloc(NH * 128 * 4), *o2 = ds4_gpu_tensor_alloc(NH * 128 * 4), *tp = ds4_gpu_tensor_alloc((size_t)NH * 64 * 130 * 4);
    CHECK(tq && tr && ts && tg && tk && tv && o1 && o2 && tp, "alloc");
    CHECK(ds4_gpu_tensor_write(tq, 0, Q, sizeof(Q)) && ds4_gpu_tensor_write(tr, 0, R, sizeof(R)) && ds4_gpu_tensor_write(ts, 0, sink, sizeof(sink)) &&
          ds4_gpu_tensor_write(tg, 0, gate, sizeof(gate)) && ds4_gpu_tensor_write(tk, 0, kc, sizeof(kc)) && ds4_gpu_tensor_write(tv, 0, vc, sizeof(vc)), "write");
    /* relative bias table [1024][16] floats from a page-aligned block registered as the 'model' */
    const size_t mapsz = 1u << 20;
    float *bias_map = mmap(NULL, mapsz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    for (int i = 0; i < 1024 * 16; i++) bias_map[i] = RNG() * 0.5f;
    CHECK(ds4_gpu_set_model_map_range(bias_map, mapsz, 0, mapsz, mapsz), "register the bias block");
    ds4_gpu_ik_w none = { bias_map, mapsz, 0, 1024 * 16 * 4, 0, 0 };
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const uint32_t p0 = cases[c][0], local = cases[c][1], rows = local ? ROWS_L : ROWS_G;
        ds4_gpu_ik_ref rq = {tq, 0}, rr = {tr, 0}, rs = {ts, 0}, rg = {tg, 0}, rk = {tk, 0}, rv = {tv, 0}, ro1 = {o1, 0}, ro2 = {o2, 0}, rp = {tp, 0}, nop = {0};
        CHECK(ds4_gpu_ik_attn(ro1, rq, rr, rs, rg, none, rk, rv, p0, rows, local, 1, nop, 0), "plain attention dispatch");
        CHECK(ds4_gpu_ik_attn(ro2, rq, rr, rs, rg, none, rk, rv, p0, rows, local, 1, rp, 64), "split attention dispatch");
        float a[NH * 128], b[NH * 128];
        CHECK(ds4_gpu_tensor_read(o1, 0, a, sizeof(a)) && ds4_gpu_tensor_read(o2, 0, b, sizeof(b)), "read back");
        double worst = 0.0, mag = 0.0;
        for (int i = 0; i < NH * 128; i++) { worst = fmax(worst, fabs((double)a[i] - (double)b[i])); mag = fmax(mag, fabs((double)a[i])); }
        CHECK(mag > 1e-3 && worst < 1e-4 * (1.0 + mag), "p0 %u local %u: split vs plain max difference %.3g (outputs up to %.3g)", p0, local, worst, mag);
    }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tr); ds4_gpu_tensor_free(ts); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tk);
    ds4_gpu_tensor_free(tv); ds4_gpu_tensor_free(o1); ds4_gpu_tensor_free(o2); ds4_gpu_tensor_free(tp);
}

/* Child mode (IK_TEST_NOSRC=1, run in a directory whose metal/ has every shader but inkling2.metal): the Metal library of the other
 * families must still load, and only the Inkling 2 kernels report "not available". */
static int child_without_inkling_source(void) {
    if (!ds4_gpu_init()) { fprintf(stderr, "FAIL: Metal init stops without metal/inkling2.metal\n"); return 1; }
    if (ds4_gpu_ik_ready()) { fprintf(stderr, "FAIL: inkling2 kernels reported ready without their source\n"); return 1; }
    ds4_gpu_cleanup();
    return 0;
}

static void test_missing_inkling_source(const char *self) {
    char root[PATH_MAX], cwd[PATH_MAX], cmd[3 * PATH_MAX], path[PATH_MAX], real[PATH_MAX];
    if (!getcwd(cwd, sizeof(cwd)) || !realpath(self, real)) { CHECK(0, "paths"); return; }
    snprintf(root, sizeof(root), "%s/ik_nosrc_XXXXXX", getenv("TMPDIR") && getenv("TMPDIR")[0] ? getenv("TMPDIR") : "/tmp");
    if (!mkdtemp(root)) { CHECK(0, "mkdtemp"); return; }
    snprintf(path, sizeof(path), "%s/metal", root);
    mkdir(path, 0700);
    DIR *d = opendir("metal");
    struct dirent *de;
    while (d && (de = readdir(d))) {
        const size_t n = strlen(de->d_name);
        if (n < 7 || strcmp(de->d_name + n - 6, ".metal") || !strcmp(de->d_name, "inkling2.metal")) continue;
        char src[PATH_MAX], dst[PATH_MAX];
        snprintf(src, sizeof(src), "%s/metal/%s", cwd, de->d_name);
        snprintf(dst, sizeof(dst), "%s/metal/%s", root, de->d_name);
        symlink(src, dst);
    }
    if (d) closedir(d);
    snprintf(cmd, sizeof(cmd), "cd '%s' && IK_TEST_NOSRC=1 '%s' > child.log 2>&1", root, real);
    const int rc = system(cmd);
    CHECK(rc == 0, "Metal init without metal/inkling2.metal failed (see %s/child.log)", root);
    if (rc == 0) {
        snprintf(cmd, sizeof(cmd), "rm -rf '%s'", root);
        if (system(cmd)) {}
    }
}

int main(int argc, char **argv) {
    (void)argc;
    if (getenv("IK_TEST_NOSRC")) return child_without_inkling_source();
    test_missing_inkling_source(argv[0]);
    if (!ds4_gpu_init()) { fprintf(stderr, "no Metal device: skipped\n"); return 0; }
    test_router_nan();
    test_kv_store_nan();
    test_attn_split_matches_plain();
    ds4_gpu_cleanup();
    if (g_fail) { fprintf(stderr, "inkling2 metal tests: %d failure(s)\n", g_fail); return 1; }
    puts("inkling2 metal tests: ok");
    return 0;
}
