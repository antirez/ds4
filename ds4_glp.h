#ifndef DS4_GLP_H
#define DS4_GLP_H

/* GLP — GGUF Layer Projection: container for directional steering vectors.
 *
 * ds4 already performs the operation GLP describes.  --dir-steering-file
 * applies, per steered layer,
 *
 *     y = y - scale * v * dot(v, y)
 *
 * which is directional ablation: remove the component of the activation along
 * v.  What ds4 lacked was a container.  The raw ".f32" form is a headerless
 * blob of n_layers * n_embd floats: it carries no operation, no hook point, no
 * base-model pin, and no layer map.  Every one of those is silently wrong when
 * it is wrong.
 *
 *   - Operation.  llama.cpp has shipped control vectors since 2024 with the
 *     same tensor convention (direction.<N>, fp32, 1-D) but ADDS them:
 *     h <- h + v.  ds4 PROJECTS: h <- h - a(h.v)v.  Loading a projective
 *     direction into an additive consumer raises no error and produces wrong
 *     output -- instead of removing the refusal component it pushes every token
 *     along the refusal axis.  Right tensor names, right dtype, right shapes.
 *
 *   - Hook point.  Steering one of a layer's contributors is not the same
 *     intervention as steering the accumulated residual: the latter also
 *     removes whatever upstream layers contributed, so a layer left unsteered
 *     only re-adds its own share.  Measured at equal coverage and equal alpha,
 *     cleaning the accumulated stream reached 3.8% refusal on a cyber suite
 *     against 34.0% for cleaning the attention write alone -- roughly 9x.
 *     Two caveats travel with that pair.  It predates the current scorer and
 *     did not store completions, so it cannot be re-scored.  And it measured
 *     the ATTENTION contributor, whereas ds4's default hook is the FFN/MoE
 *     write -- a third site the pair says nothing about; measured since
 *     (2026-09-04, DSV4-Flash-0731, refusal32), the FFN writer at alpha 4-6
 *     outperformed the residual site with a residual-derived direction
 *     transferred to it, so the factor above is not a bound in either
 *     direction.  So the mechanism is the argument here, not the number.  A
 *     file cannot say where it was calibrated, so nothing can check.
 *
 *   - Layer map.  A one-layer shift does not fail, it degrades: adjacent
 *     layers' refusal directions have cosine similarity 0.555-0.979 (mean
 *     0.863), so a shifted stack still ablates and still passes a smoke test.
 *     Only a differential layer-mapping probe finds it.
 *
 *   - Base model.  A direction is tied to the exact checkpoint it was derived
 *     from.  Applying one to a different revision is undefined.
 *
 * GLP is standard GGUF v3 with llama.cpp's tensor convention unchanged plus a
 * small "glp.*" metadata block, of which one key may not be ignored:
 *
 *     glp.mode = "project" | "add"       absent means "add"
 *
 * ds4 implements "project" only, so an "add" file is refused rather than
 * misapplied.  Spec: https://github.com/msuiche/weightless/blob/main/spec/GLP.md
 *
 * This reader is deliberately independent of ds4.c's GGUF loader: it takes
 * (n_layers, n_embd) as plain arguments and returns errors instead of calling
 * ds4_die(), so tests/test_glp.c can exercise every refusal path without a
 * model. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Where the consumer applies the projection.  A GLP file names its own hook in
 * glp.hook_point; a reader whose hook does not match must refuse the file
 * rather than apply it somewhere else.
 *
 * ds4 hooks the block writers, before the residual/hyper-connection fold --
 * ffn_out = moe + shared, and the attention output -- and the post-layer
 * residual itself, where each of the n_hc hyper-connection streams is
 * projected independently.  The two kinds of site are not interchangeable: a
 * writer carries only what that block computed this layer, while the folded
 * residual carries the accumulated sum. */
typedef enum {
    DS4_GLP_HOOK_UNKNOWN = 0,
    /* "residual_stream_post_layer": llama.cpp's build_cvec() site and the
     * vLLM overlay's site.  ds4 --dir-steering-resid, applied per
     * hyper-connection stream after the FFN fold (out_hc). */
    DS4_GLP_HOOK_RESID_POST_LAYER,
    /* "ffn_out_pre_residual": ds4 --dir-steering-ffn. */
    DS4_GLP_HOOK_FFN_OUT,
    /* "attn_out_pre_residual": ds4 --dir-steering-attn. */
    DS4_GLP_HOOK_ATTN_OUT,
} ds4_glp_hook;

#define DS4_GLP_STR_MAX 160

typedef struct {
    /* Required keys. */
    char     mode[32];              /* verbatim glp.mode */
    uint32_t spec_version;          /* glp.spec_version */

    /* Apply parameters.  alpha is a separate parameter, never folded into the
     * data: projection is quadratic in the direction's norm, so a strength
     * baked into the tensor would not mean what a caller expects. */
    float    alpha_default;         /* glp.alpha_default, 0 if absent */
    int      has_alpha_default;
    uint32_t rank;                  /* glp.rank, 1 if absent */
    ds4_glp_hook hook;
    char     hook_name[DS4_GLP_STR_MAX];  /* verbatim, "" if absent */

    /* Where the direction was CAPTURED, which is not necessarily where it is
     * applied.  A transferred vector has derived_at != hook_name.  This is a
     * warning, never a refusal: the reader applies at hook_point regardless,
     * and alpha_default belongs to the apply site, not the derivation site. */
    char     derived_at[DS4_GLP_STR_MAX]; /* verbatim glp.derived_at, "" if absent */

    /* Shape, as resolved from the direction.<N> tensors. */
    uint32_t n_embd;
    uint32_t n_dirs;
    uint32_t layer_min;
    uint32_t layer_max;

    /* Provenance. */
    char base_model[DS4_GLP_STR_MAX];
    char base_org[DS4_GLP_STR_MAX];
    char base_version[DS4_GLP_STR_MAX];
    char base_repo_url[DS4_GLP_STR_MAX];
    char content_sha256[DS4_GLP_STR_MAX];
    char created[DS4_GLP_STR_MAX];
    char method[DS4_GLP_STR_MAX];
    char contrast[DS4_GLP_STR_MAX];

    /* Norm report.  ds4's op assumes a unit direction: at scale 1 the
     * component is removed exactly only when ||v|| == 1.  Directions that
     * arrive off-unit are rescaled (the projector v^v^T is invariant to
     * ||v||, so this cannot change the intended operation) and counted here so
     * a caller can say it happened. */
    float    norm_min;
    float    norm_max;
    uint32_t n_renormalized;
} ds4_glp_info;

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if path starts with the GGUF magic, 0 if it does not, -1 if it cannot be
 * read.  Used to route --dir-steering-file between the GLP and raw-f32 paths
 * without making the user say which one they have. */
int ds4_glp_is_gguf(const char *path);

/* "residual_stream_post_layer" etc.  Never NULL. */
const char *ds4_glp_hook_name(ds4_glp_hook hook);

/* Parse metadata and tensor directory only; no direction bytes are read and no
 * model shape is needed.  This is the first thing to reach for when a vector
 * behaves oddly: it answers "what is actually in this file" without loading
 * the several-gigabyte checkpoint it belongs to.  Our own exporter off-by-one
 * surfaced as an inspect dump reporting layers 11-39 for a file derived from
 * layers 10-38.
 *
 * Returns 0 on success, nonzero on failure with a message in err. */
int ds4_glp_read_info(const char *path,
                      ds4_glp_info *info,
                      char *err,
                      size_t errlen);

/* Fill dirs[n_layers * n_embd] from a GLP file.
 *
 * dirs is the dense, layer-indexed buffer ds4's steering path already uses:
 * direction.N lands at row N, with no offset, and layers the file does not
 * cover are zeroed.  A zero row is exactly a no-op under this operation
 * (dot(0, y) == 0), so the caller's unconditional per-layer apply stays
 * correct and branch-free.
 *
 * target_hook is where the caller will apply the projection.  If the file
 * names a different hook the load fails unless allow_hook_mismatch is set:
 * steering one contributor is a different intervention from steering the
 * accumulated stream, and it degrades rather than errors, which is the failure
 * this field exists to prevent.
 *
 * Returns 0 on success, nonzero on failure with a message in err.  On failure
 * dirs is left zeroed. */
int ds4_glp_load(const char *path,
                 ds4_glp_hook target_hook,
                 int allow_hook_mismatch,
                 float *dirs,
                 uint32_t n_layers,
                 uint32_t n_embd,
                 ds4_glp_info *info,
                 char *err,
                 size_t errlen);

/* Human-readable dump for --dir-steering-info and for the one-line startup
 * provenance record. */
void ds4_glp_print_info(FILE *out, const char *path, const ds4_glp_info *info);

/* The default --dir-steering-ffn for a file the user gave no scale for.
 *
 * A GLP file carries the strength it was calibrated at, so honour it -- but
 * only when it was calibrated for the hook this scale drives. alpha is not a
 * property of the direction: projection is quadratic in the norm, and a site
 * that steers one contributor rather than the accumulated stream needs a
 * different dose, so an alpha from elsewhere is not a better default than the
 * fallback, only a more confident wrong one.
 *
 * Returns fallback for a raw f32 blob, for a GLP file with no
 * glp.alpha_default, and for one whose hook is not ffn_out_pre_residual.
 * adopted, when given, is set to 1 exactly when the file's value is used --
 * the caller needs it because an adopted 0 ("no steering by default") is a
 * value, not an absence, and must not fall through to another site's
 * default.
 * Prints a line to stderr when it does adopt the file's value, because a
 * scale the user did not type should be visible in the log. Shared by ds4,
 * ds4-server and ds4-agent so the three cannot drift. */
float ds4_glp_default_ffn_scale(const char *path, float fallback, int *adopted);

/* The --dir-steering-resid sibling: adopts glp.alpha_default only for a file
 * whose hook is residual_stream_post_layer.  Same rules and same stderr line
 * as ds4_glp_default_ffn_scale. */
float ds4_glp_default_resid_scale(const char *path, float fallback, int *adopted);

/* --dir-steering-info: describe the vector at path on stdout and return the
 * process exit code (0 ok, 1 unreadable/refused, 2 not a GLP file).
 *
 * Deliberately reports on files this build would refuse to APPLY -- a wrong
 * hook, an unknown mode -- because "what is actually in this file" is the
 * question worth answering first, and refusing to answer it for a file that
 * needs diagnosing would defeat the purpose. Shared by ds4, ds4-server and
 * ds4-agent so the flag the help text advertises exists in all three. */
int ds4_glp_inspect_main(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* DS4_GLP_H */
