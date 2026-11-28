/* Model-free checks for the Inkling 2 CPU path: row dequantisation of every weight type the loader accepts
 * (NVFP4 is new) and the GPT-4o pre-tokenizer.  Full-model parity against llama.cpp is checked with the
 * DS4_INKLING2_* oracle (see docs/INKLING2.md). */
#define DS4_NO_GPU
#include "../ds4.c"
#include <assert.h>

static const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static void test_nvfp4_row(void) {
    ik_ue4m3_init();
    /* four sub-block scales: 1.0 (0x38), 2.0 (0x40), 0.5 (0x30), 0 (0x00); byte j holds element j (low nibble,
     * positive) and element j+8 (high nibble, same magnitude, negative) of its 16-element sub-block */
    uint8_t blk[36] = {0x38, 0x40, 0x30, 0x00};
    const float scale[4] = {1.0f, 2.0f, 0.5f, 0.0f};
    for (int s = 0; s < 4; s++)
        for (int j = 0; j < 8; j++) blk[4 + 8 * s + j] = (uint8_t)(((8 + j) << 4) | j);
    float out[64];
    float *got = (float *)ik_row_f32(DS4_TENSOR_NVFP4, blk, 64, out);
    for (int s = 0; s < 4; s++)
        for (int j = 0; j < 8; j++) {
            assert(got[s * 16 + j] == e2m1[j] * scale[s]);
            assert(got[s * 16 + 8 + j] == -e2m1[j] * scale[s]);
        }
    /* UE4M3 corners: subnormal 1 * 2^-9, 0x7f is NaN and reads as 0, max finite 0x7e = 1.75 * 2^8 */
    assert(ik_ue4m3_lut[1] == ldexpf(1.0f, -9));
    assert(ik_ue4m3_lut[0x7f] == 0.0f);
    assert(ik_ue4m3_lut[0x7e] == 448.0f);
}

static void test_other_rows(void) {
    uint8_t q8[34] = {0x00, 0x3c};   /* scale 1.0 in half precision */
    for (int j = 0; j < 32; j++) q8[2 + j] = (uint8_t)(int8_t)(j - 16);
    float out[64];
    const float *r = ik_row_f32(DS4_TENSOR_Q8_0, q8, 32, out);
    for (int j = 0; j < 32; j++) assert(r[j] == (float)(j - 16));

    uint8_t mx[17] = {128};          /* E8M0 128 = 2^1; byte j holds element j and element j+16 */
    for (int j = 0; j < 16; j++) mx[1 + j] = (uint8_t)(((j & 7) << 4) | (j & 7));
    r = ik_row_f32(DS4_TENSOR_MXFP4, mx, 32, out);
    for (int j = 0; j < 16; j++) assert(r[j] == 2.0f * e2m1[j & 7] && r[j + 16] == 2.0f * e2m1[j & 7]);

    const uint16_t bf[2] = {0x3fc0, 0xc000};   /* 1.5, -2 */
    r = ik_row_f32(DS4_TENSOR_BF16, (const uint8_t *)bf, 2, out);
    assert(r[0] == 1.5f && r[1] == -2.0f);
}

/* Pre-token boundaries of the GPT-4o regex; pieces are printed with '|' between them. */
static void split(const char *text, char *dst) {
    const uint64_t len = strlen(text);
    for (uint64_t pos = 0; pos < len;) {
        const uint64_t e = ik_next_piece(text, len, pos);
        assert(e > pos);
        if (pos) *dst++ = '|';
        memcpy(dst, text + pos, (size_t)(e - pos));
        dst += e - pos;
        pos = e;
    }
    *dst = '\0';
}

static void check_split(const char *text, const char *want) {
    char got[256];
    split(text, got);
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "split(\"%s\") = \"%s\", want \"%s\"\n", text, got, want);
        exit(1);
    }
}

static void test_pretokenizer(void) {
    check_split("Hello world", "Hello| world");
    check_split("don't stop", "don't| stop");
    check_split("I'LL go", "I'LL| go");
    check_split("ABC def", "ABC| def");
    check_split("12345", "123|45");
    check_split("foo.bar", "foo|.bar");
    check_split("a, b", "a|,| b");
    check_split("a\n\nb", "a|\n\n|b");
    check_split("a  b", "a| | b");              /* \s+(?!\S) leaves the last space for the next word */
    check_split("end  ", "end|  ");
    check_split("x = 1;\n", "x| =| |1|;\n");
    check_split("na\xc3\xafve", "na\xc3\xafve");   /* non-ASCII letters join lower and upper runs */
}

/* Regression: a malformed \u escape or a trailing backslash used to step over the terminating NUL and write
 * past the end of the buffer (heap overflow, found with AddressSanitizer).  Each of these must be rejected. */
static void test_json_string(void) {
    static const char *const bad[] = { "abc\\u\"", "abc\\u12", "abc\\u12\"", "abc\\", "abc\\ud83d\\u", "abc\\ud83d\\u12",
                                       "abc\\ud83d\\uzzzz\"", "abc\\uzzzz\"", "abc" };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        char *out = NULL;
        uint64_t len = 0;
        assert(!ik_json_string(bad[i], &out, &len));
        free(out);
    }
    char *out = NULL;
    uint64_t len = 0;
    assert(ik_json_string("a\\u00e9\\ud83d\\ude00\\n\\\"b\"tail", &out, &len));
    assert(len == 1 + 2 + 4 + 1 + 1 + 1 && memcmp(out, "a\xc3\xa9\xf0\x9f\x98\x80\n\"b", len) == 0);
    free(out);
}

/* Regression: DS4_INKLING2_EXACT=0 or empty used to be read as "on" by one place (variable exists) and "off" by another (value),
 * so the tiled kernels ran without their work list and the output was garbage.  One parser now serves both. */
static void test_exact_switch(void) {
    static const struct { const char *val; int want; } cases[] = {
        {NULL, 0}, {"", 0}, {"0", 0}, {"false", 0}, {"off", 0}, {"no", 0}, {"1", 1}, {"true", 1}, {"yes", 1}, {"2", 1},
        /* case and blanks do not matter (OFF, No, " 0" used to select the slow exact kernels) */
        {"OFF", 0}, {"No", 0}, {"FALSE", 0}, {" 0", 0}, {"0 ", 0}, {"  ", 0}, {"1 ", 1}, {"garbage", 1}, {"00", 1}, {"-1", 1}};
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (cases[i].val) setenv("DS4_INKLING2_EXACT", cases[i].val, 1); else unsetenv("DS4_INKLING2_EXACT");
        assert(ds4_inkling2_exact_mode() == cases[i].want);
    }
    unsetenv("DS4_INKLING2_EXACT");
}

/* Regression: the special-token splitter moved the fragment list with memmove for every match (quadratic) and compared every
 * position with memcmp (a 2.48 MB body took 2.2 s).  The new partition must cut exactly the same fragments as the old algorithm
 * (kept here as the reference) and stay linear. */
static ik_frags ik_ref_partition(const ds4_vocab *v, const char *text, uint64_t len) {
    ik_frag *f = xmalloc(sizeof(*f));
    int nf = 1;
    f[0] = (ik_frag){ 0, len, -1 };
    for (int si = 0; si < g_ik_special.n; si++) {
        const ds4_str sp = v->token[g_ik_special.id[si]];
        for (int i = 0; i < nf; i++) {
            if (f[i].token >= 0) continue;
            for (uint64_t p = f[i].off; p + sp.len <= f[i].off + f[i].len; p++) {
                if (memcmp(text + p, sp.ptr, (size_t)sp.len) != 0) continue;
                f = xrealloc(f, (size_t)(nf + 2) * sizeof(*f));
                memmove(f + i + 3, f + i + 1, (size_t)(nf - i - 1) * sizeof(*f));
                const ik_frag cur = f[i];
                f[i] = (ik_frag){ cur.off, p - cur.off, -1 };
                f[i + 1] = (ik_frag){ p, sp.len, g_ik_special.id[si] };
                f[i + 2] = (ik_frag){ p + sp.len, cur.off + cur.len - p - sp.len, -1 };
                nf += 2;
                i++;
                break;
            }
        }
    }
    ik_frags out = { f, nf, nf };
    return out;
}

static bool ik_frags_same(const ik_frags *a, const ik_frags *b) {
    int i = 0, j = 0;
    for (;;) {   /* empty raw fragments carry no text: skip them on both sides */
        while (i < a->n && a->v[i].token < 0 && !a->v[i].len) i++;
        while (j < b->n && b->v[j].token < 0 && !b->v[j].len) j++;
        if (i == a->n || j == b->n) return i == a->n && j == b->n;
        if (a->v[i].off != b->v[j].off || a->v[i].len != b->v[j].len || a->v[i].token != b->v[j].token) return false;
        i++; j++;
    }
}

static void test_special_partition(void) {
    static const char *const toks[] = { "<|end_message|>", "<|message_model|>", "<|a|>", "<|ab|>", "<|", "|>", "<|content_text|>" };
    const int nt = (int)(sizeof(toks) / sizeof(toks[0]));
    ds4_vocab v;
    memset(&v, 0, sizeof(v));
    v.n_vocab = nt;
    v.token = xcalloc((size_t)nt, sizeof(v.token[0]));
    for (int i = 0; i < nt; i++) v.token[i] = (ds4_str){ toks[i], strlen(toks[i]) };
    free(g_ik_special.id);
    g_ik_special.id = xmalloc((size_t)nt * sizeof(int));
    g_ik_special.n = nt;
    for (int i = 0; i < nt; i++) g_ik_special.id[i] = i;
    for (int i = 1; i < nt; i++) {   /* longest first, as ik_vocab_init */
        const int id = g_ik_special.id[i];
        int j = i;
        while (j > 0 && v.token[g_ik_special.id[j - 1]].len < v.token[id].len) { g_ik_special.id[j] = g_ik_special.id[j - 1]; j--; }
        g_ik_special.id[j] = id;
    }
    /* random texts built from marker pieces and filler, including overlaps like "<|a|>|>" and "<<|a|>" */
    static const char *const piece[] = { "<|end_message|>", "<|message_model|>", "<|a|>", "<|ab|>", "<|", "|>", "<", "|", ">", "ab", " ", "x\n", "<|content_text|>", "<|a" };
    const int np = (int)(sizeof(piece) / sizeof(piece[0]));
    uint32_t rng = 12345;
    for (int iter = 0; iter < 3000; iter++) {
        char text[512];
        size_t n = 0;
        const int pieces = 1 + (int)(iter % 40);
        for (int k = 0; k < pieces; k++) {
            rng = rng * 1664525u + 1013904223u;
            const char *s = piece[(rng >> 16) % (uint32_t)np];
            const size_t l = strlen(s);
            if (n + l >= sizeof(text)) break;
            memcpy(text + n, s, l);
            n += l;
        }
        text[n] = '\0';
        ik_frags a = ik_special_partition(&v, text, n), b = ik_ref_partition(&v, text, n);
        assert(ik_frags_same(&a, &b));
        free(a.v);
        free(b.v);
    }
    /* scaling: 2.5 MB of back-to-back markers and filler */
    const size_t big_n = 2480000;
    char *big = malloc(big_n + 1);
    size_t n = 0;
    while (n + 40 < big_n) n += (size_t)snprintf(big + n, 40, "<|message_model|>ab <|a|>");
    clock_t t0 = clock();
    ik_frags a = ik_special_partition(&v, big, n);
    const double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
    assert(a.n > 100000);
    fprintf(stderr, "inkling2 special-token partition, 2.48 MB: %.3f s CPU (quadratic version 3.4 s)\n", dt);
    assert(dt < 1.0);
    free(a.v);
    free(big);
    free(v.token);
    free(g_ik_special.id);
    g_ik_special.id = NULL;
    g_ik_special.n = 0;
}

/* The start check uses what is free now with a margin: 8 sessions of 262,144 tokens (87.4 GiB with the model) are refused beside
 * a busy machine (70% free of 256 GiB), 8 x 65,536 (about 40 GiB) start; the fixed budget still applies. */
static void test_memory_verdict(void) {
    const double G = 1073741824.0, ram = 256.0 * G;
    assert(ds4_inkling2_memory_verdict(87.37 * G, ram, 70, 40, 128.0 * G) == 2);
    assert(ds4_inkling2_memory_verdict(39.4 * G, ram, 70, 40, 128.0 * G) == 0);
    assert(ds4_inkling2_memory_verdict(87.37 * G, ram, 86, 40, 128.0 * G) == 0);   /* idle machine: it fits */
    assert(ds4_inkling2_memory_verdict(87.37 * G, ram, 100, 40, 64.0 * G) == 1);   /* fixed budget wins */
    assert(ds4_inkling2_memory_verdict(87.37 * G, ram, -1, 40, 128.0 * G) == 0);   /* free memory unknown: only the budget */
    assert(ds4_inkling2_memory_verdict(1.0 * G, ram, 35, 40, 0.0) == 2);           /* already under the floor */
    assert(ds4_inkling2_memory_verdict(87.37 * G, ram, 70, 0, 128.0 * G) == 0);    /* floor 0 turns the check off */
}

int main(void) {
    test_memory_verdict();
    test_special_partition();
    test_exact_switch();
    test_json_string();
    test_nvfp4_row();
    test_other_rows();
    test_pretokenizer();
    puts("inkling2 tests: ok");
    return 0;
}
