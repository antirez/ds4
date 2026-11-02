/* Tokenizer regression and speed test.
 *
 * The id-pair heap BPE, the id-pair merge table, the marker scan and the
 * decoded-token arena must reproduce the string-keyed implementation they
 * replaced exactly.  The reference copies below are that implementation.
 *
 *   tests/test_tokenizer                         synthetic vocab only
 *   tests/test_tokenizer VOCAB [CORPUS...]       also a real vocab
 *   tests/test_tokenizer --bench VOCAB [--shape SHAPE] CORPUS...
 *                                                throughput, new vs reference
 *   tests/test_tokenizer --export-vocab MODEL.gguf OUT
 *                                                write the GGUF's vocab as text
 *   tests/test_tokenizer --dump-ids VOCAB SHAPE FILE...
 *                                                print ids, one line per file
 *
 * VOCAB is a GGUF model or a text file: the token count, one token per line,
 * then one "left right" merge per line in rank order.  --export-vocab writes it
 * from a GGUF; tests/tokenizer_vocab.py from a Hugging Face tokenizer.json or
 * GPT-2 vocab.bpe. */
#include "../ds4.c"

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

/* ---------------------------------------------------------------------- */
/* Reference implementation, verbatim apart from names. */

typedef struct {
    char *ptr;
    uint64_t len;
} ref_owned_str;

static ref_owned_str ref_owned_copy(const char *ptr, uint64_t len) {
    ref_owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static uint32_t ref_gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

static int ref_gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static char *ref_byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, ref_gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int ref_bpe_rank(const str_i32_table *merge_rank,
                        const ref_owned_str *a, const ref_owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
    int rank = -1;
    table_get(merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void ref_bpe_emit_piece(const ds4_vocab *vocab, const str_i32_table *merge_rank,
                               ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = ref_byte_encode(raw_piece, &encoded_len);
    int n_sym = 0;
    int cap_sym = 32;
    ref_owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));
    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = ref_owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }
    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = ref_bpe_rank(merge_rank, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        ref_owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }
    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }
    free(sym);
    free(encoded);
}

static bool ref_special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"[gMASK]",                vocab->bos_id},
        {"<sop>",                  vocab->sop_id},
        {"<|system|>",             vocab->system_id},
        {"<｜System｜>",            DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41 ? vocab->system_id : -1},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<|user|>",               vocab->user_id},
        {"<|assistant|>",          vocab->assistant_id},
        {"<|observation|>",        vocab->observation_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"<tool_call>",            vocab->tool_call_start_id},
        {"</tool_call>",           vocab->tool_call_end_id},
        {"<tool_response>",        vocab->tool_response_start_id},
        {"</tool_response>",       vocab->tool_response_end_id},
        {"<arg_key>",              vocab->arg_key_start_id},
        {"</arg_key>",             vocab->arg_key_end_id},
        {"<arg_value>",            vocab->arg_value_start_id},
        {"</arg_value>",           vocab->arg_value_end_id},
        {"｜DSML｜",                vocab->dsml_id},
        {"<|im_start|>",           vocab->im_start_id},
        {"<|im_end|>",             vocab->im_end_id},
        {"<|endoftext|>",          vocab->endoftext_id},
    };
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

/* The marker scan of the reference; spans use the current BPE, which is
 * compared separately. */
static void ref_tokenize_rendered(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (ref_special_token_at(vocab, p, &token, &len)) {
            bpe_tokenize_span(vocab, span, (uint64_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    bpe_tokenize_span(vocab, span, (uint64_t)(p - span), out);
}

static char *ref_token_text(const ds4_vocab *vocab, int token, size_t *len) {
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }
    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }
    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = ref_gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

/* ---------------------------------------------------------------------- */
/* Vocab construction. */

typedef struct {
    ds4_engine engine;      /* only engine.vocab is populated */
    str_i32_table merge_rank;
    ds4_str *merges;
    uint64_t n_merges;
    char *file;             /* backing storage of a text vocab */
    ds4_model model;
    bool gguf;
} test_vocab;

static void clear_markers(ds4_vocab *v) {
    v->bos_id = v->eos_id = v->system_id = v->user_id = v->assistant_id = -1;
    v->observation_id = v->sop_id = v->think_start_id = v->think_end_id = -1;
    v->tool_call_start_id = v->tool_call_end_id = -1;
    v->tool_response_start_id = v->tool_response_end_id = -1;
    v->arg_key_start_id = v->arg_key_end_id = -1;
    v->arg_value_start_id = v->arg_value_end_id = -1;
    v->dsml_id = v->im_start_id = v->im_end_id = v->endoftext_id = -1;
}

static void finish_vocab(test_vocab *t) {
    ds4_vocab *v = &t->engine.vocab;
    vocab_index(v, t->merges, t->n_merges);
    table_init(&t->merge_rank, t->n_merges);
    for (uint64_t i = 0; i < t->n_merges; i++) table_put(&t->merge_rank, t->merges[i], (int)i);
}

static void free_test_vocab(test_vocab *t) {
    if (t->gguf) {
        vocab_free(&t->engine.vocab);
        model_close(&t->model);
    } else {
        vocab_free(&t->engine.vocab);
        free(t->file);
    }
    table_free(&t->merge_rank);
    free(t->merges);
    memset(t, 0, sizeof(*t));
}

static char *read_file(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "test_tokenizer: cannot open %s\n", path);
        exit(1);
    }
    size_t cap = 1 << 16, n = 0;
    char *buf = xmalloc(cap + 1);
    for (size_t r; (r = fread(buf + n, 1, cap - n, fp)) > 0;) {
        n += r;
        if (n == cap) buf = xrealloc(buf, (cap *= 2) + 1);
    }
    fclose(fp);
    buf[n] = '\0';
    *len = n;
    return buf;
}

static void load_text_vocab(test_vocab *t, const char *path) {
    size_t len = 0;
    t->file = read_file(path, &len);
    char *p = t->file;
    char *nl = strchr(p, '\n');
    if (!nl) ds4_die("bad vocab file");
    *nl = '\0';
    long n = strtol(p, NULL, 10);
    if (n <= 0 || n > INT32_MAX) ds4_die("bad vocab token count");
    p = nl + 1;

    ds4_vocab *v = &t->engine.vocab;
    v->n_vocab = (int)n;
    v->token = xcalloc((size_t)n, sizeof(v->token[0]));
    table_init(&v->token_to_id, (uint64_t)n);
    for (long i = 0; i < n; i++) {
        nl = strchr(p, '\n');
        if (!nl) ds4_die("truncated vocab file");
        v->token[i] = (ds4_str){p, (uint64_t)(nl - p)};
        table_put(&v->token_to_id, v->token[i], (int)i);
        p = nl + 1;
    }
    uint64_t cap = 1024;
    t->merges = xmalloc(cap * sizeof(t->merges[0]));
    while (*p) {
        nl = strchr(p, '\n');
        uint64_t l = nl ? (uint64_t)(nl - p) : strlen(p);
        if (t->n_merges == cap) t->merges = xrealloc(t->merges, (cap *= 2) * sizeof(t->merges[0]));
        t->merges[t->n_merges++] = (ds4_str){p, l};
        p += l + (nl ? 1 : 0);
    }
    clear_markers(v);
    /* Recognise the chat markers present so the marker scan is exercised. */
    v->bos_id = vocab_lookup_optional(v, "<｜begin▁of▁sentence｜>");
    v->eos_id = vocab_lookup_optional(v, "<｜end▁of▁sentence｜>");
    v->user_id = vocab_lookup_optional(v, "<｜User｜>");
    if (v->user_id < 0) v->user_id = vocab_lookup_optional(v, "<|user|>");
    v->assistant_id = vocab_lookup_optional(v, "<｜Assistant｜>");
    if (v->assistant_id < 0) v->assistant_id = vocab_lookup_optional(v, "<|assistant|>");
    v->system_id = vocab_lookup_optional(v, "<｜System｜>");
    if (v->system_id < 0) v->system_id = vocab_lookup_optional(v, "<|system|>");
    v->observation_id = vocab_lookup_optional(v, "<|observation|>");
    v->dsml_id = vocab_lookup_optional(v, "｜DSML｜");
    v->im_start_id = vocab_lookup_optional(v, "<|im_start|>");
    v->im_end_id = vocab_lookup_optional(v, "<|im_end|>");
    v->endoftext_id = vocab_lookup_optional(v, "<|endoftext|>");
    v->think_start_id = vocab_lookup_optional(v, "<think>");
    v->think_end_id = vocab_lookup_optional(v, "</think>");
    finish_vocab(t);
}

static void load_gguf_vocab(test_vocab *t, const char *path) {
    t->gguf = true;
    model_open(&t->model, path, false, false);
    config_validate_model(&t->model);
    vocab_load(&t->engine.vocab, &t->model);
    ds4_array_ref merges;
    if (!model_get_array(&t->model, "tokenizer.ggml.merges", &merges)) ds4_die("no merges");
    t->n_merges = merges.len;
    t->merges = xmalloc((size_t)(merges.len ? merges.len : 1) * sizeof(t->merges[0]));
    ds4_cursor c = cursor_at(&t->model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        if (!cursor_string(&c, &t->merges[i])) ds4_die(c.error);
    }
    table_init(&t->merge_rank, t->n_merges);
    for (uint64_t i = 0; i < t->n_merges; i++) table_put(&t->merge_rank, t->merges[i], (int)i);
}

static void load_vocab(test_vocab *t, const char *path) {
    size_t n = strlen(path);
    if (n > 5 && !strcmp(path + n - 5, ".gguf")) load_gguf_vocab(t, path);
    else load_text_vocab(t, path);
}

/* A tiny byte-level vocab with merges chosen to stress the merge order:
 * repeated pairs ("a a"), competing overlaps ("a b" vs "b c"), chains
 * that only become possible after earlier merges, a byte with no symbol, a
 * duplicate merge, a candidate made stale by its right symbol merging, and
 * merges that can never fire. */
static void build_synthetic_vocab(test_vocab *t) {
    static const char *extra[] = {
        "aa", "aaa", "aaaa", "ab", "bc", "abc", "ca", "cab", "bca", "Ġa", "Ġab",
        "Ġabc", "xy", "yx", "xyx", "yxy", "xyxy", "pq", "qr", "qrs", "pqr",
        "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    };
    static const char *merge_text[] = {
        "a a", "b c", "a b", "aa a", "aa aa", "a bc", "ab c", "c a", "c ab",
        "b ca", "Ġ a", "Ġa b", "Ġab c", "x y", "y x", "xy x", "y xy", "xy xy",
        /* "pqrs": the pending "p q" goes stale when q's symbol grows to "qr";
         * applying it anyway would produce "pqr" + "s" instead of "p" + "qrs". */
        "q r", "p q", "qr s", "p qr",
        "a a",            /* duplicate: the later rank wins */
        "q", "q r s",      /* malformed: never fire */
        "zz z",            /* side outside the vocab: never fires */
    };
    const int n_extra = (int)(sizeof(extra) / sizeof(extra[0]));
    ds4_vocab *v = &t->engine.vocab;
    v->n_vocab = 256 + n_extra;
    v->token = xcalloc((size_t)v->n_vocab, sizeof(v->token[0]));
    table_init(&v->token_to_id, (uint64_t)v->n_vocab);
    t->file = xmalloc(256 * 4);
    for (int b = 0; b < 256; b++) {
        char *p = t->file + b * 4;
        char *start = p;
        utf8_put(&p, ref_gpt2_byte_to_codepoint((uint8_t)b));
        /* Byte 0x07 has no symbol: it must act as a merge barrier and be
         * dropped, as the reference does. */
        v->token[b] = (ds4_str){start, b == 7 ? 0 : (uint64_t)(p - start)};
        if (b != 7) table_put(&v->token_to_id, v->token[b], b);
    }
    for (int i = 0; i < n_extra; i++) {
        v->token[256 + i] = (ds4_str){extra[i], strlen(extra[i])};
        table_put(&v->token_to_id, v->token[256 + i], 256 + i);
    }
    t->n_merges = sizeof(merge_text) / sizeof(merge_text[0]);
    t->merges = xmalloc((size_t)t->n_merges * sizeof(t->merges[0]));
    for (uint64_t i = 0; i < t->n_merges; i++) {
        t->merges[i] = (ds4_str){merge_text[i], strlen(merge_text[i])};
    }
    clear_markers(v);
    v->im_start_id = vocab_lookup(v, "<|im_start|>");
    v->im_end_id = vocab_lookup(v, "<|im_end|>");
    v->think_start_id = vocab_lookup(v, "<think>");
    v->think_end_id = vocab_lookup(v, "</think>");
    finish_vocab(t);
}

/* ---------------------------------------------------------------------- */
/* Comparisons. */

static bool same_tokens(const token_vec *a, const token_vec *b) {
    return a->len == b->len && !memcmp(a->v, b->v, (size_t)a->len * sizeof(int));
}

static uint64_t pieces_checked;

static void check_piece(test_vocab *t, const char *ptr, uint64_t len) {
    token_vec got = {0}, want = {0};
    bpe_emit_piece(&t->engine.vocab, (ds4_str){ptr, len}, &got);
    ref_bpe_emit_piece(&t->engine.vocab, &t->merge_rank, (ds4_str){ptr, len}, &want);
    if (!same_tokens(&got, &want)) {
        fprintf(stderr, "BPE mismatch on %llu-byte piece \"%.*s\"\n",
                (unsigned long long)len, (int)(len > 80 ? 80 : len), ptr);
        failures++;
    }
    pieces_checked++;
    token_vec_free(&got);
    token_vec_free(&want);
}

static void check_rendered(test_vocab *t, const char *text) {
    token_vec got = {0}, want = {0};
    tokenize_rendered_chat_vocab(&t->engine.vocab, text, &got);
    ref_tokenize_rendered(&t->engine.vocab, text, &want);
    if (!same_tokens(&got, &want)) {
        fprintf(stderr, "rendered-chat mismatch on \"%.80s\"\n", text);
        failures++;
    }
    token_vec_free(&got);
    token_vec_free(&want);
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;

static uint64_t rng(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* Every pre-tokenizer piece reaches bpe_emit_piece unchanged, so checking
 * pieces covers the pipeline: every line, every space-separated word, the
 * text as a whole (bounded, since the reference is quadratic), and random
 * substrings and alphabets. */
static void check_text(test_vocab *t, const char *text, size_t len) {
    const size_t whole = len < 4096 ? len : 4096;
    check_piece(t, text, whole);
    for (size_t i = 0; i < len;) {
        size_t j = i;
        while (j < len && text[j] != '\n') j++;
        if (j - i <= 4096) check_piece(t, text + i, j - i);
        size_t k = i;
        while (k < j) {
            size_t w = k + 1;
            while (w < j && text[w] != ' ') w++;
            /* Words are bounded like the whole-text check above: the
             * reference is quadratic, and an unwrapped base64 blob is a
             * single word of unbounded length. */
            if (w - k <= 4096) check_piece(t, text + k, w - k);
            k = w;
        }
        i = j + 1;
    }
    for (int i = 0; i < 200 && len; i++) {
        size_t off = (size_t)(rng() % len);
        size_t n = 1 + (size_t)(rng() % 300);
        if (off + n > len) n = len - off;
        check_piece(t, text + off, n);
    }
}

static void check_random(test_vocab *t, const char *alphabet, int rounds, int max_len) {
    const size_t an = strlen(alphabet);
    char buf[2048];
    for (int r = 0; r < rounds; r++) {
        int n = 1 + (int)(rng() % (uint64_t)max_len);
        for (int i = 0; i < n; i++) buf[i] = alphabet[rng() % an];
        check_piece(t, buf, (uint64_t)n);
    }
    for (int r = 0; r < rounds; r++) {
        int n = 1 + (int)(rng() % (uint64_t)max_len);
        for (int i = 0; i < n; i++) buf[i] = (char)(rng() & 0xff);
        check_piece(t, buf, (uint64_t)n);
    }
}

static void check_detokenize(test_vocab *t) {
    for (int id = -2; id < t->engine.vocab.n_vocab + 2; id++) {
        size_t got_len = 0, want_len = 0;
        char *got = ds4_token_text(&t->engine, id, &got_len);
        char *want = ref_token_text(&t->engine.vocab, id, &want_len);
        CHECK(got_len == want_len && !memcmp(got, want, want_len) && got[got_len] == '\0');
        free(got);
        free(want);
    }
}

static void check_markers(test_vocab *t) {
    static const char *texts[] = {
        "", "<", "<|im_start|>", "x<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "<think>a</think>b", "<<|im_end|><", "<|im_start", "<think><think></think>",
        "plain text without markers", "[gMASK]<sop><|user|>", "<｜User｜>q<｜Assistant｜>",
    };
    for (size_t i = 0; i < sizeof(texts) / sizeof(texts[0]); i++) check_rendered(t, texts[i]);
}

static void check_byte_map(void) {
    for (int b = 0; b < 256; b++) {
        CHECK(gpt2_byte_to_codepoint((uint8_t)b) == ref_gpt2_byte_to_codepoint((uint8_t)b));
    }
    for (uint32_t cp = 0; cp < 0x400; cp++) {
        CHECK(gpt2_codepoint_to_byte(cp) == ref_gpt2_codepoint_to_byte(cp));
    }
}

/* The ASCII fast path of qwen4_char_at must agree with the Unicode tables. */
static void check_qwen4_ascii_classes(void) {
    for (uint32_t cp = 1; cp < 128; cp++) {
        const char ch = (char)cp;
        const qwen4_char_info c = qwen4_char_at(&ch, 1, 0);
        CHECK(c.letter == DS4_QWEN4_CLASS(letter, cp));
        CHECK(c.mark == DS4_QWEN4_CLASS(mark, cp));
        CHECK(c.number == DS4_QWEN4_CLASS(number, cp));
        CHECK(c.space == DS4_QWEN4_CLASS(space, cp));
    }
}

static void check_tokens_append(void) {
    ds4_tokens a = {0}, b = {0};
    for (int i = 0; i < 100; i++) ds4_tokens_push(&b, i);
    ds4_tokens_push(&a, -1);
    ds4_tokens_append(&a, &b);
    ds4_tokens_append(&a, &b);
    CHECK(a.len == 201 && a.v[0] == -1 && a.v[1] == 0 && a.v[200] == 99 && a.cap >= a.len);
    ds4_tokens_copy(&a, &b);
    CHECK(a.len == 100 && a.v[99] == 99);
    ds4_tokens_free(&a);
    ds4_tokens_free(&b);
}

static void run_checks(test_vocab *t, int n_corpus, char **corpus) {
    check_detokenize(t);
    check_markers(t);
    check_random(t, "ab", 300, 200);
    check_random(t, "abcxy Ġ", 300, 200);
    check_random(t, "pqrs", 300, 50);
    check_piece(t, "pqrs", 4);
    check_random(t, "abcdefghijklmnopqrstuvwxyz0123456789+/=", 200, 400);
    for (int i = 0; i < n_corpus; i++) {
        size_t len = 0;
        char *text = read_file(corpus[i], &len);
        check_text(t, text, len);
        check_rendered(t, text);
        free(text);
    }
    /* A long letter run, the case the heap exists for. */
    static char run[4096];
    for (size_t i = 0; i < sizeof(run); i++) run[i] = "ab+/"[rng() % 2];
    check_piece(t, run, sizeof(run));
}

/* ---------------------------------------------------------------------- */
/* Benchmark. */

static double bench_rate(void (*fn)(test_vocab *, const char *, size_t, token_vec *),
                         test_vocab *t, const char *text, size_t len, int *n_tokens) {
    token_vec out = {0};
    int iters = 0;
    const double start = now_sec();
    double elapsed = 0;
    do {
        out.len = 0;
        fn(t, text, len, &out);
        iters++;
        elapsed = now_sec() - start;
    } while (elapsed < 0.5);
    *n_tokens = out.len;
    token_vec_free(&out);
    return elapsed / iters;
}

static void bench_tokenize(test_vocab *t, const char *text, size_t len, token_vec *out) {
    (void)len;
    tokenize_rendered_chat_vocab(&t->engine.vocab, text, out);
}

static void bench_piece(test_vocab *t, const char *text, size_t len, token_vec *out) {
    bpe_emit_piece(&t->engine.vocab, (ds4_str){text, len}, out);
}

static void bench_ref_piece(test_vocab *t, const char *text, size_t len, token_vec *out) {
    ref_bpe_emit_piece(&t->engine.vocab, &t->merge_rank, (ds4_str){text, len}, out);
}

/* Detokenize the tokens of the text back to bytes, as the server does for
 * every checkpoint after a job. */
static token_vec bench_detok_tokens;

static void bench_detokenize(test_vocab *t, const char *text, size_t len, token_vec *out) {
    (void)text;
    (void)len;
    size_t total = 0;
    for (int i = 0; i < bench_detok_tokens.len; i++) {
        size_t n = 0;
        ds4_token_bytes(&t->engine, bench_detok_tokens.v[i], &n);
        total += n;
    }
    token_vec_push(out, (int)total);
}

static int bench(test_vocab *t, int n_corpus, char **corpus) {
    printf("%-28s %12s %14s\n", "input", "MB/s", "tokens/s");
    for (int i = 0; i < n_corpus; i++) {
        size_t len = 0;
        char *text = read_file(corpus[i], &len);
        int n = 0;
        double sec = bench_rate(bench_tokenize, t, text, len, &n);
        const char *name = strrchr(corpus[i], '/');
        printf("%-28.28s %12.2f %14.0f\n", name ? name + 1 : corpus[i], len / sec / 1e6, n / sec);
        tokenize_rendered_chat_vocab(&t->engine.vocab, text, &bench_detok_tokens);
        sec = bench_rate(bench_detokenize, t, text, len, &n);
        printf("%-28s %12.2f %14.0f\n", "  detokenize", len / sec / 1e6,
               bench_detok_tokens.len / sec);
        token_vec_free(&bench_detok_tokens);
        free(text);
    }
    /* One unbroken letter run, as base64 or hex in tool output produces. */
    for (size_t n = 1000; n <= 20000; n *= 20) {
        char *run = xmalloc(n);
        for (size_t i = 0; i < n; i++) {
            run[i] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[rng() % 52];
        }
        int tokens = 0;
        double cur = bench_rate(bench_piece, t, run, n, &tokens);
        double ref = bench_rate(bench_ref_piece, t, run, n, &tokens);
        printf("%zu-letter run: current %.3f ms, reference %.3f ms (%d tokens)\n",
               n, cur * 1e3, ref * 1e3, tokens);
        free(run);
    }
    return 0;
}

/* Raw GGUF arrays, without model validation: the export must reflect exactly
 * what vocab_load reads, for any family. */
/* Print the engine's token ids for each file, one line of space-separated
 * ids per file, so an external tokenizer (Hugging Face) can be diffed
 * against ds4.  SHAPE picks the pre-tokenizer family: flash, glm53,
 * flash41 or qwen4.  Plain text tokenization, no chat markers. */
static ds4_shape parse_shape(const char *name) {
    const struct { const char *name; ds4_shape shape; } shapes[] = {
        {"flash", DS4_SHAPE_FLASH}, {"glm53", DS4_SHAPE_GLM53},
        {"flash41", DS4_SHAPE_FLASH41}, {"qwen4", DS4_SHAPE_QWEN4_EXP},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        if (!strcmp(shapes[i].name, name)) return shapes[i].shape;
    ds4_die("shape must be flash, glm53, flash41 or qwen4");
    return DS4_SHAPE_FLASH;
}

static int dump_ids(const char *vocab_path, const char *shape_name,
                    int n_files, char **files) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = parse_shape(shape_name);
    test_vocab t = {0};
    load_vocab(&t, vocab_path);
    for (int i = 0; i < n_files; i++) {
        size_t len = 0;
        char *text = read_file(files[i], &len);
        ds4_tokens tokens = {0};
        ds4_tokenize_text(&t.engine, text, &tokens);
        free(text);
        for (int j = 0; j < tokens.len; j++)
            printf(j ? " %d" : "%d", tokens.v[j]);
        printf("\n");
        ds4_tokens_free(&tokens);
    }
    free_test_vocab(&t);
    g_ds4_shape = saved;
    return 0;
}

static int export_vocab(const char *model_path, const char *out_path) {
    ds4_model model;
    model_open(&model, model_path, false, false);
    ds4_array_ref arrays[2];
    if (!model_get_array(&model, "tokenizer.ggml.tokens", &arrays[0]) ||
        !model_get_array(&model, "tokenizer.ggml.merges", &arrays[1]) ||
        arrays[0].type != GGUF_VALUE_STRING || arrays[1].type != GGUF_VALUE_STRING) {
        ds4_die("GGUF has no tokenizer token and merge tables");
    }
    FILE *fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "test_tokenizer: cannot write %s\n", out_path);
        return 1;
    }
    fprintf(fp, "%llu\n", (unsigned long long)arrays[0].len);
    for (int a = 0; a < 2; a++) {
        ds4_cursor c = cursor_at(&model, arrays[a].data_pos);
        for (uint64_t i = 0; i < arrays[a].len; i++) {
            ds4_str str;
            if (!cursor_string(&c, &str)) ds4_die(c.error);
            if (memchr(str.ptr, '\n', (size_t)str.len)) {
                fprintf(stderr, "test_tokenizer: entry %llu contains a newline\n",
                        (unsigned long long)i);
                fclose(fp);
                return 1;
            }
            fprintf(fp, "%.*s\n", (int)str.len, str.ptr);
        }
    }
    fclose(fp);
    printf("%s: %llu tokens, %llu merges\n", out_path,
           (unsigned long long)arrays[0].len, (unsigned long long)arrays[1].len);
    model_close(&model);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--export-vocab")) return export_vocab(argv[2], argv[3]);
    if (argc >= 5 && !strcmp(argv[1], "--dump-ids"))
        return dump_ids(argv[2], argv[3], argc - 4, argv + 4);
    const bool do_bench = argc > 1 && !strcmp(argv[1], "--bench");
    if (do_bench) {
        if (argc < 3) {
            fprintf(stderr, "usage: %s --bench VOCAB [--shape SHAPE] [CORPUS...]\n", argv[0]);
            return 2;
        }
        int first = 3;
        if (argc > 4 && !strcmp(argv[3], "--shape")) {
            g_ds4_shape = parse_shape(argv[4]);
            first = 5;
        }
        test_vocab t = {0};
        load_vocab(&t, argv[2]);
        int rc = bench(&t, argc - first, argv + first);
        free_test_vocab(&t);
        return rc;
    }

    check_byte_map();
    check_qwen4_ascii_classes();
    check_tokens_append();

    const ds4_shape saved = g_ds4_shape;
    const ds4_shape shapes[] = {
        DS4_SHAPE_FLASH, DS4_SHAPE_GLM53, DS4_SHAPE_FLASH41, DS4_SHAPE_QWEN4_EXP
    };
    test_vocab t = {0};
    build_synthetic_vocab(&t);
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        g_ds4_shape = shapes[i];
        run_checks(&t, 0, NULL);
    }
    free_test_vocab(&t);
    g_ds4_shape = saved;

    if (argc > 1) {
        load_vocab(&t, argv[1]);
        if (!t.gguf) {
            for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
                g_ds4_shape = shapes[i];
                run_checks(&t, argc - 2, argv + 2);
            }
            g_ds4_shape = saved;
        } else {
            run_checks(&t, argc - 2, argv + 2);
        }
        free_test_vocab(&t);
    }

    if (failures) {
        fprintf(stderr, "tokenizer tests: %d failures\n", failures);
        return 1;
    }
    printf("tokenizer tests: ok (%llu BPE pieces compared)\n", (unsigned long long)pieces_checked);
    return 0;
}
