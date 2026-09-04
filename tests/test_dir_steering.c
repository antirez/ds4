/* Numeric test for the projection behind the steering hooks.
 *
 * The residual_stream_post_layer hook applies the shared row projector to the
 * post-fold hyper-connection stack out_hc, [n_hc][n_embd] per token and
 * contiguous, as rows = n_tok * n_hc independent rows -- i.e. each stream is
 * projected independently against the same per-layer direction.  That call is
 * what this file exercises, through the DS4_TEST_HOOKS wrapper, against a
 * hand-computed projection:
 *
 *   - a zero direction row is exactly a no-op (dot(0, x) == 0), which is what
 *     makes layers a GLP file does not cover unsteered;
 *   - a nonzero direction removes scale * dot(v, x) * v from every stream
 *     row, per stream, not once on a flattened sum.
 *
 * Model-free: the projector only needs the default shape's n_embd. */

#include "../ds4.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_EMBD 4096   /* default shape n_embd; the projector reads it live */
#define N_HC   4      /* hyper-connection streams per token */
#define N_LAYERS 8

static int g_failed = 0;

#define CHECK(cond, msg) do {                                                  \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

static double dot_row(const float *a, const float *b) {
    double s = 0.0;
    for (int i = 0; i < N_EMBD; i++) s += (double)a[i] * (double)b[i];
    return s;
}

/* Deterministic non-trivial fill, different per stream so the per-stream
 * dots differ and a flattened projection would not match. */
static void fill_stack(float *x) {
    for (int r = 0; r < N_HC; r++) {
        for (int i = 0; i < N_EMBD; i++) {
            x[r * N_EMBD + i] =
                0.001f * (float)(((i + 13 * r) % 23) - 11) * (1.0f + 0.25f * (float)r);
        }
    }
}

int main(void) {
    static float dirs[N_LAYERS * N_EMBD];
    static float x[N_HC * N_EMBD];
    static float ref[N_HC * N_EMBD];

    /* Layer 3 carries a unit direction; every other layer is a zero row. */
    for (int i = 0; i < N_EMBD; i++) {
        dirs[3 * N_EMBD + i] = (float)((i % 7) + 1) * ((i % 2) ? -1.0f : 1.0f);
    }
    const double dnorm = sqrt(dot_row(dirs + 3 * N_EMBD, dirs + 3 * N_EMBD));
    for (int i = 0; i < N_EMBD; i++) dirs[3 * N_EMBD + i] /= (float)dnorm;
    const float *dir = dirs + 3 * N_EMBD;

    fill_stack(x);
    memcpy(ref, x, sizeof(x));

    /* Zero direction at a covered-by-nothing layer: exact no-op. */
    ds4_test_directional_steering_project_rows(x, dirs, 4, N_HC, 1.5f);
    CHECK(memcmp(x, ref, sizeof(x)) == 0, "zero direction row is an exact no-op");

    /* Zero scale: exact no-op too. */
    ds4_test_directional_steering_project_rows(x, dirs, 3, N_HC, 0.0f);
    CHECK(memcmp(x, ref, sizeof(x)) == 0, "zero scale is an exact no-op");

    /* The zero-scale no-op must not even read the direction: with a NaN row
     * and scale 0 the stack is bit-identical.  This is what makes
     * alpha_default=0 -- "no steering by default" -- safe to honor. */
    {
        static float nan_dirs[N_LAYERS * N_EMBD];
        const uint32_t nan_bits = 0x7fc00000u;   /* quiet NaN, by bits: this
                                                    file builds with
                                                    -ffast-math, where the
                                                    NAN macro is UB */
        float nan_f;
        memcpy(&nan_f, &nan_bits, sizeof(nan_f));
        for (int i = 0; i < N_EMBD; i++) nan_dirs[3 * N_EMBD + i] = nan_f;
        ds4_test_directional_steering_project_rows(x, nan_dirs, 3, N_HC, 0.0f);
        CHECK(memcmp(x, ref, sizeof(x)) == 0,
              "zero scale does not read the direction (NaN row, no-op)");
    }

    /* Nonzero direction, scale 2: x -= 2 * dot(dir, row) * dir per stream. */
    ds4_test_directional_steering_project_rows(x, dirs, 3, N_HC, 2.0f);
    double max_abs = 0.0;
    int changed = 0;
    for (int r = 0; r < N_HC; r++) {
        const float *row_in = ref + r * N_EMBD;
        const float coeff = (float)(2.0 * dot_row(dir, row_in));
        for (int i = 0; i < N_EMBD; i++) {
            const float expect = row_in[i] - coeff * dir[i];
            const double d = fabs((double)x[r * N_EMBD + i] - (double)expect);
            if (d > max_abs) max_abs = d;
            if (x[r * N_EMBD + i] != row_in[i]) changed = 1;
        }
    }
    CHECK(changed, "nonzero direction changes the post-fold stack");
    CHECK(max_abs < 1e-4, "per-stream projection matches the hand computation");
    if (max_abs >= 1e-4) fprintf(stderr, "    max_abs=%g\n", max_abs);

    /* The projection is per stream: stream r must see dot(dir, row_r), not a
     * shared dot over the flattened stack.  The hand computation above only
     * passes if that is true, so this is the same assertion viewed from the
     * other side: at scale 1 the component along a unit direction is removed
     * exactly, so a second application must be a no-op. */
    memcpy(x, ref, sizeof(x));
    ds4_test_directional_steering_project_rows(x, dirs, 3, N_HC, 1.0f);
    ds4_test_directional_steering_project_rows(x, dirs, 3, N_HC, 1.0f);
    for (int r = 0; r < N_HC; r++) {
        const float *row_in = ref + r * N_EMBD;
        const float coeff = (float)dot_row(dir, row_in);
        double row_err = 0.0;
        for (int i = 0; i < N_EMBD; i++) {
            const float expect = row_in[i] - coeff * dir[i];
            row_err += fabs((double)x[r * N_EMBD + i] - (double)expect);
        }
        CHECK(row_err < 1e-2, "a second scale-1 projection is a no-op (idempotent)");
    }

    fprintf(stderr, "%s\n", g_failed == 0 ? "dir-steering projection tests: OK"
                                          : "dir-steering projection tests: FAILED");
    return g_failed == 0 ? 0 : 1;
}
