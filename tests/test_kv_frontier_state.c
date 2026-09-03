/* Regression test: compressor frontier-state occupancy serialization.
 *
 * Single-lane compressor layers (compress_ratio != 0 && != 4) now serialize
 * only the occupied partial = checkpoint_len % ratio rows of the in-progress
 * compressor block (plus a u32 count), instead of the full-capacity buffer.
 * Ratio-4 layers keep the full-capacity layout and are untouched by this.
 *
 * This test verifies, on the Metal backend:
 *   - Round-trip exactness (gate #1/#6): greedy-decode GEN_N tokens from a
 *     live session, then save + warm-restore the checkpoint and decode again.
 *     The two continuations must be byte-for-byte identical.  The restore is
 *     done warm (the session has already advanced past the checkpoint), so the
 *     tail-clear path is exercised: a missing clear would leave stale rows in
 *     [partial, ratio) and corrupt the next compressed row.
 *   - Size math (gate #2): the saved file size equals ds4_session_payload_bytes.
 *   - Both a ratio-aligned checkpoint (partial = 0, gate #5) and an un-aligned
 *     one (partial > 0), where generating GEN_N crosses a ratio-128 commit.
 *   - Version rejection: a payload whose version field is corrupted must be
 *     rejected (the ABI bump that retires pre-change .kv files).
 *
 * Requires DS4_TEST_MODEL.  Generation is kept short and ctx modest.
 */
#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GEN_N          8
#define TEST_CTX       2048
#define STATE_HEADER_BYTES 4u   /* u32 version lives at file offset 4 */

static void fail(const char *msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    exit(1);
}

#define CHECK(cond, msg) do { if (!(cond)) fail(msg); } while (0)

/* Greedy-decode exactly n tokens from the current session position. */
static void greedy_n(ds4_session *s, int *out, int n) {
    char err[256] = {0};
    for (int i = 0; i < n; i++) {
        int t = ds4_session_argmax(s);
        CHECK(t >= 0, "argmax");
        out[i] = t;
        CHECK(ds4_session_eval(s, t, err, sizeof(err)) == 0, "eval");
    }
}

/* One save -> warm-restore round-trip at target_len. Returns the saved byte count. */
static uint64_t run_target(ds4_engine *e, const ds4_tokens *prompt, int target_len) {
    char err[256] = {0};
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, TEST_CTX) == 0, "session create");
    CHECK(ds4_session_sync(s, prompt, err, sizeof(err)) == 0, "session sync");

    /* Greedy-eval forward to the target checkpoint length. */
    while (ds4_session_pos(s) < target_len) {
        int t = ds4_session_argmax(s);
        CHECK(t >= 0, "argmax to target");
        CHECK(ds4_session_eval(s, t, err, sizeof(err)) == 0, "eval to target");
    }
    CHECK(ds4_session_pos(s) == target_len, "reached target length");

    /* Save the payload at the target checkpoint. */
    FILE *fp = tmpfile();
    CHECK(fp != NULL, "tmpfile");
    CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0, "save payload");
    const uint64_t saved_bytes = (uint64_t)ftello(fp);
    const uint64_t predicted = ds4_session_payload_bytes(s);
    if (saved_bytes != predicted) {
        fprintf(stderr,
                "FAIL: size math target=%d file=%llu payload_bytes=%llu\n",
                target_len, (unsigned long long)saved_bytes,
                (unsigned long long)predicted);
        exit(1);
    }

    /* Baseline continuation from the live (un-checkpointed) session. */
    int baseline[GEN_N];
    greedy_n(s, baseline, GEN_N);

    /* Warm restore: reload the saved checkpoint into the now-advanced session,
     * resetting it to target_len.  Exercises the compressor tail-clear path. */
    rewind(fp);
    CHECK(ds4_session_load_payload(s, fp, saved_bytes, err, sizeof(err)) == 0,
          "load payload");
    CHECK(ds4_session_pos(s) == target_len, "restored to target length");

    int restored[GEN_N];
    greedy_n(s, restored, GEN_N);

    for (int i = 0; i < GEN_N; i++) {
        if (baseline[i] != restored[i]) {
            fprintf(stderr,
                    "FAIL: token divergence target=%d step=%d baseline=%d restored=%d\n",
                    target_len, i, baseline[i], restored[i]);
            exit(1);
        }
    }

    fclose(fp);
    ds4_session_free(s);
    return saved_bytes;
}

/* Compare two open files byte-for-byte from their current positions up to
 * `bytes`. Returns true if identical. Restores the file positions afterwards. */
static bool files_identical(FILE *a, FILE *b, uint64_t bytes) {
    rewind(a); rewind(b);
    uint8_t ba[8192], bb[8192];
    uint64_t left = bytes;
    while (left > 0) {
        size_t n = left > sizeof(ba) ? sizeof(ba) : (size_t)left;
        if (fread(ba, 1, n, a) != n || fread(bb, 1, n, b) != n) return false;
        if (memcmp(ba, bb, n) != 0) return false;
        left -= n;
    }
    rewind(a); rewind(b);
    return true;
}

/* Deep single-target check used by the fuzz sweep: save -> warm-restore ->
 * generate must match the live baseline exactly; re-saving after restore must
 * be byte-identical (derivation is deterministic); and a second warm restore
 * into the dirty session must again match (repeated tail-clear). */
static void run_fuzz_target(ds4_engine *e, const ds4_tokens *prompt, int target_len) {
    char err[256] = {0};
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, TEST_CTX) == 0, "fuzz session create");
    CHECK(ds4_session_sync(s, prompt, err, sizeof(err)) == 0, "fuzz sync");
    while (ds4_session_pos(s) < target_len) {
        int t = ds4_session_argmax(s);
        CHECK(t >= 0, "fuzz argmax to target");
        CHECK(ds4_session_eval(s, t, err, sizeof(err)) == 0, "fuzz eval to target");
    }
    CHECK(ds4_session_pos(s) == target_len, "fuzz reached target");

    FILE *fa = tmpfile(); CHECK(fa != NULL, "fuzz tmpfile a");
    CHECK(ds4_session_save_payload(s, fa, err, sizeof(err)) == 0, "fuzz save a");
    const uint64_t bytes_a = (uint64_t)ftello(fa);
    CHECK(bytes_a == ds4_session_payload_bytes(s), "fuzz size math");

    int baseline[GEN_N];
    greedy_n(s, baseline, GEN_N);

    /* 1st warm restore. */
    rewind(fa);
    CHECK(ds4_session_load_payload(s, fa, bytes_a, err, sizeof(err)) == 0, "fuzz load 1");
    CHECK(ds4_session_pos(s) == target_len, "fuzz pos after load 1");

    /* Re-save immediately (still at target_len) and confirm a byte-identical
     * payload: the partial derivation is deterministic and the state was
     * restored exactly, so the serialized bytes must match the original save. */
    FILE *fb = tmpfile(); CHECK(fb != NULL, "fuzz tmpfile b");
    CHECK(ds4_session_save_payload(s, fb, err, sizeof(err)) == 0, "fuzz save b");
    const uint64_t bytes_b = (uint64_t)ftello(fb);
    CHECK(bytes_b == bytes_a, "fuzz re-save size changed");
    CHECK(files_identical(fa, fb, bytes_a), "fuzz re-save bytes differ");
    fclose(fb);

    /* Generate from the restored checkpoint; must match the live baseline. */
    int restored[GEN_N];
    greedy_n(s, restored, GEN_N);
    for (int i = 0; i < GEN_N; i++)
        CHECK(baseline[i] == restored[i], "fuzz gen mismatch after load 1");

    /* 2nd warm restore into the now-dirty session + generate (repeated clear). */
    rewind(fa);
    CHECK(ds4_session_load_payload(s, fa, bytes_a, err, sizeof(err)) == 0, "fuzz load 2");
    CHECK(ds4_session_pos(s) == target_len, "fuzz pos after load 2");
    greedy_n(s, restored, GEN_N);
    for (int i = 0; i < GEN_N; i++)
        CHECK(baseline[i] == restored[i], "fuzz gen mismatch after load 2");

    fclose(fa);
    ds4_session_free(s);
    fprintf(stderr, "  fuzz target=%d (partial128=%d) OK\n",
            target_len, target_len % 128);
}

/* A payload whose version field is corrupted must be rejected on load. */
static void test_version_rejection(ds4_engine *e, const ds4_tokens *prompt) {
    char err[256] = {0};
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, TEST_CTX) == 0, "vr session create");
    CHECK(ds4_session_sync(s, prompt, err, sizeof(err)) == 0, "vr sync");

    FILE *fp = tmpfile();
    CHECK(fp != NULL, "vr tmpfile");
    CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0, "vr save");
    const uint64_t bytes = (uint64_t)ftello(fp);

    /* Overwrite the version u32 (header word 1, byte offset 4). */
    const uint32_t bad_version = 0xFFFFFFFFu;
    CHECK(fseek(fp, (long)STATE_HEADER_BYTES, SEEK_SET) == 0, "vr seek");
    CHECK(fwrite(&bad_version, sizeof(bad_version), 1, fp) == 1, "vr corrupt");
    rewind(fp);

    const int rc = ds4_session_load_payload(s, fp, bytes, err, sizeof(err));
    if (rc == 0) fail("version rejection: corrupt payload was accepted");

    fclose(fp);
    ds4_session_free(s);
    fprintf(stderr, "version-rejection OK (corrupt payload rejected)\n");
}

/* A payload whose single-lane compressor partial count is inconsistent with
 * the checkpoint length must be rejected (gate #3: no silent acceptance). */
static void test_partial_corruption(ds4_engine *e, const ds4_tokens *prompt,
                                    int unaligned_target) {
    char err[256] = {0};
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, TEST_CTX) == 0, "pc session create");
    CHECK(ds4_session_sync(s, prompt, err, sizeof(err)) == 0, "pc sync");
    while (ds4_session_pos(s) < unaligned_target) {
        int t = ds4_session_argmax(s);
        CHECK(t >= 0, "pc argmax");
        CHECK(ds4_session_eval(s, t, err, sizeof(err)) == 0, "pc eval");
    }
    CHECK(ds4_session_pos(s) == unaligned_target, "pc reached target");

    /* Locate the first single-lane layer's partial-count field in a payload
     * written at this checkpoint (reuses the engine's own layout accounting). */
    uint64_t off = 0;
    int found = -1;
    const int n_layers = ds4_engine_layer_count(e);
    for (int il = 0; il < n_layers; il++) {
        if (ds4_test_payload_partial_offset(s, (uint32_t)il, &off)) {
            found = il;
            break;
        }
    }
    if (found < 0) {
        fprintf(stderr, "partial-corruption: no single-lane layers; skipping\n");
        ds4_session_free(s);
        return;
    }

    FILE *fp = tmpfile();
    CHECK(fp != NULL, "pc tmpfile");
    CHECK(ds4_session_save_payload(s, fp, err, sizeof(err)) == 0, "pc save");
    const uint64_t bytes = (uint64_t)ftello(fp);

    /* Flip the partial count's low bit so it can no longer equal
     * checkpoint_len % ratio.  The load validates this right after reading the
     * count, before consuming any state rows, so the rejection is clean. */
    uint32_t partial = 0;
    CHECK(fseeko(fp, (off_t)off, SEEK_SET) == 0, "pc seek");
    CHECK(fread(&partial, sizeof(partial), 1, fp) == 1, "pc read partial");
    const uint32_t corrupted = partial ^ 1u;
    CHECK(fseeko(fp, (off_t)off, SEEK_SET) == 0, "pc seek2");
    CHECK(fwrite(&corrupted, sizeof(corrupted), 1, fp) == 1, "pc corrupt");
    rewind(fp);

    const int rc = ds4_session_load_payload(s, fp, bytes, err, sizeof(err));
    if (rc == 0) fail("partial-corruption: inconsistent partial was accepted");

    fclose(fp);
    ds4_session_free(s);
    fprintf(stderr,
            "partial-corruption OK (inconsistent partial rejected, layer %d)\n",
            found);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "FAIL: DS4_TEST_MODEL is not set\n");
        return 1;
    }

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_METAL;
    opt.n_threads = 1;
    opt.warm_weights = false;
    opt.quality = false;

    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0 || !e) fail("engine open");

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(e, /*system=*/NULL, "What is the capital of France?",
                           DS4_THINK_NONE, &prompt);
    CHECK(prompt.len > 0, "tokenize prompt");
    const int prompt_len = prompt.len;

    /* Opt-in broad sweep: many checkpoint lengths covering partial 0, edge
     * values (1, 127), mid values, and lengths crossing several ratio-128
     * commits, each with exact restore + byte-stable re-save + repeated warm
     * restore.  Default run (no env) stays at the two representative lengths. */
    if (getenv("DS4_TEST_FUZZ") && getenv("DS4_TEST_FUZZ")[0] != '0') {
        const int base = ((prompt_len + 64 + 127) / 128) * 128;  /* first mult of 128 */
        fprintf(stderr, "FUZZ mode: base=%d prompt_len=%d\n", base, prompt_len);
        const int aligned[] = { base, base * 2, base * 4 };
        const int offs[] = { 1, 2, 7, 32, 63, 64, 65, 100, 120, 127 };
        for (size_t i = 0; i < sizeof(aligned) / sizeof(aligned[0]); i++)
            run_fuzz_target(e, &prompt, aligned[i]);
        for (size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); i++)
            run_fuzz_target(e, &prompt, base + offs[i]);
        run_fuzz_target(e, &prompt, base - 1);   /* partial 127, n_comp one less */
        run_fuzz_target(e, &prompt, base * 4 - 12);  /* partial 116, crosses 3 commits */
        ds4_tokens_free(&prompt);
        ds4_engine_close(e);
        fprintf(stderr, "test_kv_frontier_state FUZZ PASS\n");
        return 0;
    }

    /* Aligned target: a multiple of 128 (hence also of 4) past the prompt. */
    const int margin = 192;
    int aligned_target = ((prompt_len + margin + 127) / 128) * 128;
    /* Un-aligned target: aligned - GEN_N so generating GEN_N crosses the
     * ratio-128 commit boundary at aligned_target. */
    int unaligned_target = aligned_target - GEN_N;
    while (unaligned_target <= prompt_len) {
        aligned_target += 128;
        unaligned_target = aligned_target - GEN_N;
    }

    fprintf(stderr, "prompt_len=%d aligned_target=%d unaligned_target=%d\n",
            prompt_len, aligned_target, unaligned_target);

    const uint64_t aligned_bytes = run_target(e, &prompt, aligned_target);
    const uint64_t unaligned_bytes = run_target(e, &prompt, unaligned_target);

    /* Occupancy varies with checkpoint length: at the un-aligned checkpoint the
     * single-lane layers hold partial > 0 occupied rows, so the payload must be
     * strictly larger than the aligned (partial = 0) one. */
    if (unaligned_bytes <= aligned_bytes) {
        fprintf(stderr,
                "FAIL: frontier-state size is not length-dependent "
                "(aligned=%llu unaligned=%llu)\n",
                (unsigned long long)aligned_bytes,
                (unsigned long long)unaligned_bytes);
        return 1;
    }
    fprintf(stderr,
            "payload bytes: aligned=%llu unaligned=%llu delta=%llu\n",
            (unsigned long long)aligned_bytes,
            (unsigned long long)unaligned_bytes,
            (unsigned long long)(unaligned_bytes - aligned_bytes));

    test_version_rejection(e, &prompt);
    test_partial_corruption(e, &prompt, unaligned_target);

    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    fprintf(stderr, "test_kv_frontier_state PASS\n");
    return 0;
}
