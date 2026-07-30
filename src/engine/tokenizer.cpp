#include "pulsar_engine_internal.h"



static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}



static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = (str_i32_entry *)xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}



static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}



static void table_put(str_i32_table *t, pulsar_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (pulsar_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}



static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        pulsar_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}



void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = (int *)xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}



void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}



void pulsar_tokens_push(pulsar_tokens *tv, int token) {
    token_vec_push(tv, token);
}



void pulsar_tokens_free(pulsar_tokens *tv) {
    token_vec_free(tv);
}



void pulsar_tokens_copy(pulsar_tokens *dst, const pulsar_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}



bool pulsar_tokens_starts_with(const pulsar_tokens *tokens, const pulsar_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}



bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale) {
    return dirs && scale != 0.0f;
}



void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale) {
    if (!cpu_directional_steering_enabled(dirs, scale) || !x || rows == 0) return;

    const float *dir = dirs + (uint64_t)il * PULSAR_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * PULSAR_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < PULSAR_N_EMBD; i++) {
            dot += xr[i] * dir[i];
        }
        const float coeff = scale * dot;
        for (uint32_t i = 0; i < PULSAR_N_EMBD; i++) {
            xr[i] -= coeff * dir[i];
        }
    }
}






static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}



static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
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



/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(pulsar_str in, uint64_t *out_len) {
    char *out = (char *)xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}



static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}



static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = (char *)xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}



/* Look up the merge rank for two adjacent BPE symbols. */
int pulsar_vocab::bpe_rank(const owned_str *a, const owned_str *b) const {
    const auto *vocab = this;
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : (char *)xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}



/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
void pulsar_vocab::bpe_emit_piece(pulsar_str raw_piece, token_vec *out) const {
    const auto *vocab = this;
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = (owned_str *)xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = (owned_str *)xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = vocab->bpe_rank(&sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = (char *)xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
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



static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}



static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}



static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}



static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}



static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}



static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}



static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}



static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}



static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}



static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}



static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}



/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
/* JoyAI/DeepSeek pre-tokenization.  The split shape matters: different pieces
 * lead to different BPE merges even when the final text bytes are identical. */
void pulsar_vocab::bpe_tokenize_text(const char *text, token_vec *out) const {
    const auto *vocab = this;
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        vocab->bpe_emit_piece((pulsar_str){ text + start, pos - start }, out);
    }
}



int pulsar_vocab::vocab_lookup(const char *text) const {
    const auto *vocab = this;
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "pulsar: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}



/* Load token strings, special token ids, and merge ranks from GGUF metadata. */
void pulsar_vocab::vocab_load(const pulsar_model *model) {
    auto *vocab = this;
    memset(vocab, 0, sizeof(*vocab));

    pulsar_array_ref tokens;
    pulsar_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        pulsar_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        pulsar_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = (pulsar_str *)xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    pulsar_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) pulsar_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        pulsar_str merge;
        if (!cursor_string(&c, &merge)) pulsar_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    vocab->bos_id       = vocab->vocab_lookup("<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab->vocab_lookup("<｜end▁of▁sentence｜>");
    vocab->user_id      = vocab->vocab_lookup("<｜User｜>");
    vocab->assistant_id = vocab->vocab_lookup("<｜Assistant｜>");
    vocab->think_start_id = vocab->vocab_lookup("<think>");
    vocab->think_end_id = vocab->vocab_lookup("</think>");
    vocab->dsml_id = vocab->vocab_lookup("｜DSML｜");
}



void pulsar_vocab::vocab_free() {
    auto *vocab = this;
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}



/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void encode_chat_prompt(
        const pulsar_vocab *vocab,
        const char      *system,
        const char      *prompt,
        pulsar_think_mode   think_mode,
        token_vec       *out) {
    token_vec_push(out, vocab->bos_id);
    if (think_mode == PULSAR_THINK_MAX) {
        vocab->bpe_tokenize_text(PULSAR_REASONING_EFFORT_MAX_PREFIX, out);
    }
    if (system && system[0]) {
        vocab->bpe_tokenize_text(system, out);
    }
    token_vec_push(out, vocab->user_id);
    vocab->bpe_tokenize_text(prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (pulsar_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}



void pulsar_tokenize_text(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    e->vocab.bpe_tokenize_text(text ? text : "", out);
}



bool pulsar_vocab::special_token_at(const char *p, int *token, size_t *len) const {
    const auto *vocab = this;
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}



void pulsar_vocab::tokenize_span(const char *p, size_t n, token_vec *out) const {
    const auto *vocab = this;
    if (!n) return;
    char *tmp = (char *)xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    vocab->bpe_tokenize_text(tmp, out);
    free(tmp);
}



void pulsar_vocab::tokenize_rendered_chat_vocab(const char *text,
                                                token_vec *out) const {
    const auto *vocab = this;
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (vocab->special_token_at(p, &token, &len)) {
            vocab->tokenize_span(span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    vocab->tokenize_span(span, (size_t)(p - span), out);
}



void pulsar_tokenize_rendered_chat(pulsar_engine *e, const char *text, pulsar_tokens *out) {
    e->vocab.tokenize_rendered_chat_vocab(text, out);
}



void pulsar_chat_begin(pulsar_engine *e, pulsar_tokens *tokens) {
    token_vec_push(tokens, e->vocab.bos_id);
}



void pulsar_encode_chat_prompt(
        pulsar_engine *e,
        const char *system,
        const char *prompt,
        pulsar_think_mode think_mode,
        pulsar_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}



void pulsar_chat_append_max_effort_prefix(pulsar_engine *e, pulsar_tokens *tokens) {
    e->vocab.bpe_tokenize_text(PULSAR_REASONING_EFFORT_MAX_PREFIX, tokens);
}



void pulsar_vocab::bpe_tokenize_tool_result_text(const char *content, token_vec *out) {
    auto *vocab = this;
    /* Tool output is plain data inside <tool_result>...</tool_result>.
     * Preserve literal '<', '>' and '&' so shell output and file snippets stay
     * intact, but escape the exact closing sentinel so a malicious or accidental
     * tool payload cannot terminate the wrapper early. */
    const char *end = "</tool_result>";
    const size_t endlen = strlen(end);
    const char *span = content ? content : "";
    const char *p = span;
    while (*p) {
        if (!strncmp(p, end, endlen)) {
            vocab->tokenize_span(span, (size_t)(p - span), out);
            vocab->bpe_tokenize_text("&lt;", out);
            p++;
            span = p;
        } else {
            p++;
        }
    }
    vocab->tokenize_span(span, (size_t)(p - span), out);
}



void pulsar_chat_append_message(pulsar_engine *e, pulsar_tokens *tokens, const char *role, const char *content) {
    pulsar_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        vocab->bpe_tokenize_text(content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        vocab->bpe_tokenize_text(content, tokens);
    } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        token_vec_push(tokens, vocab->user_id);
        vocab->bpe_tokenize_text("<tool_result>", tokens);
        vocab->bpe_tokenize_tool_result_text(content, tokens);
        vocab->bpe_tokenize_text("</tool_result>", tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        vocab->bpe_tokenize_text(content, tokens);
    }
}



void pulsar_chat_append_assistant_prefix(pulsar_engine *e, pulsar_tokens *tokens, pulsar_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    token_vec_push(tokens, pulsar_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}



void dump_tokens_fp(FILE *fp, const pulsar_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}



void pulsar_vocab::dump_tokens(const token_vec *tokens) const {
    const auto *vocab = this;
    dump_tokens_fp(stdout, vocab, tokens);
}



static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}



static int gpt2_codepoint_to_byte(uint32_t cp) {
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



static bool vocab_token_is_literal_special(pulsar_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}



char *pulsar_token_text(pulsar_engine *e, int token, size_t *len) {
    pulsar_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = (char *)xmalloc(1);
        out[0] = '\0';
        return out;
    }

    pulsar_str s = vocab->token[token];
    char *out = (char *)xmalloc((size_t)s.len + 1);
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
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}



int pulsar_token_eos(pulsar_engine *e) {
    return e->vocab.eos_id;
}



int pulsar_token_user(pulsar_engine *e) {
    return e->vocab.user_id;
}



int pulsar_token_assistant(pulsar_engine *e) {
    return e->vocab.assistant_id;
}



int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = PULSAR_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}



static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}



static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}



/* IEEE-754 float32 -> uint32 whose UNSIGNED ASCENDING order is the float's
 * DESCENDING order: flip negatives entirely, set the sign bit on
 * non-negatives (the standard monotonic transform, giving float-ascending),
 * then complement to reverse it. Only finite values reach here — the callers
 * filter !isfinite — so NaN ordering is not a concern.
 *
 * -0.0f is canonicalized to +0.0f first: it is a distinct bit pattern but
 * compares EQUAL to +0.0f, so the comparator this replaces called them a
 * tie, and by the tie note below, ordering what it tied can change the
 * nucleus. Real, not hypothetical: logits arrive by memcpy from the GPU, and
 * device code is built --use_fast_math (FTZ), so it writes exactly this.
 *
 * Subnormals are deliberately NOT flushed, and the reasoning is worth
 * recording because it is a trap. -ffast-math would set FPCR.FZ (flushing
 * subnormal FCMP inputs, making the comparator tie them) — but ONLY via
 * crtfastmath.o, which the LINKER pulls in, and every binary carrying this
 * code (ds4, pulsar-server, pulsar_test) is linked by nvcc, not by gcc -ffast-math.
 * Measured in the real link config: FPCR = 0x0, FZ = 0, and the comparator
 * ORDERS subnormals. So keying them strictly is what matches. A gcc-linked
 * probe of the same source reports FPCR = 0x1000000 and the opposite answer —
 * do not test this outside the real linkage. tests/pulsar_test.c --sampler
 * covers it (shape "subnormals + zeros (FZ range)"). */
static inline uint32_t sample_desc_key(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if (u == 0x80000000u) u = 0u;   /* -0.0 == +0.0 to the comparator */
    const uint32_t asc = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ~asc;
}



/* Stable LSD radix sort of packed (sample_desc_key << 32 | id) on the key
 * half — 4 byte passes, O(n), no comparator calls. Replaces a qsort over the
 * whole 129k-entry vocab that cost ~10.6 ms per call (~97% of dist_build) and
 * ran once per accepted position in the sampled speculative walk.
 *
 * Order is bit-for-bit what the qsort produced, ties included.
 *
 * STABILITY IS LOAD-BEARING — do not swap in an unstable sort (parallel,
 * in-place, MSD) on the theory that ties are harmless. They are not. It is
 * true that permuting tied candidates permutes IDENTICAL prob values, so
 * `sum`, `min_prob`, `filtered`, `filtered_sum` and every out->probs entry
 * are bit-invariant under any tie order. But `filtered` is a COUNT, and a tie
 * group straddling the cutoff changes which ids fall inside it — i.e. nucleus
 * MEMBERSHIP, not just order within the nucleus. Concretely, all-equal logits
 * with top_p=0.5 give filtered=64640 of 129280: ascending ties yield
 * ids [0..64639], descending ties yield ids [64640..129279] — disjoint.
 * pulsar_sample_dist_prob returns 0 for an id outside the nucleus, so
 * pulsar_sample_dist_accept would then REJECT a draft the other order ACCEPTS,
 * changing the emitted token stream.
 *
 * So: the caller fills `a` in ascending-id order and this sort is stable,
 * making ascending-id the canonical tie order. That also matches what the
 * replaced qsort produced at this size (glibc takes its stable msort_with_tmp
 * path for a 129280 x 12B array -- verified), but nothing here depends on that
 * unspecified detail: this sort is stable by construction and libc-independent.
 * tests/pulsar_test.c --sampler pins the order explicitly (and catches a
 * tie-order flip on a realistic shape -- ties are common, not adversarial).
 *
 * Requires n >= 1. */
static void sample_radix_sort_desc(uint64_t *a, uint64_t *tmp, uint32_t n) {
    uint32_t hist[4][256];
    memset(hist, 0, sizeof(hist));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t k = (uint32_t)(a[i] >> 32);
        hist[0][k & 0xffu]++;
        hist[1][(k >> 8) & 0xffu]++;
        hist[2][(k >> 16) & 0xffu]++;
        hist[3][(k >> 24) & 0xffu]++;
    }
    uint64_t *src = a;
    uint64_t *dst = tmp;
    for (int pass = 0; pass < 4; pass++) {
        const int sh = 32 + pass * 8;
        /* Whole pass is a no-op when every element shares this key byte —
         * e.g. the all-equal-logits degenerate case skips all four. */
        if (hist[pass][(uint32_t)(src[0] >> sh) & 0xffu] == n) continue;
        uint32_t off[256];
        uint32_t run = 0;
        for (int b = 0; b < 256; b++) {
            off[b] = run;
            run += hist[pass][b];
        }
        for (uint32_t i = 0; i < n; i++)
            dst[off[(uint32_t)(src[i] >> sh) & 0xffu]++] = src[i];
        uint64_t *swap = src;
        src = dst;
        dst = swap;
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof(*a));
}



/* =====
 * min-p prefilter threshold (shared by sample_full_vocab and
 * pulsar_sample_dist_build's full-vocab paths).
 *
 * The min-p cutoff both paths apply post-sort keeps candidate i (i > 0) iff
 *
 *     fl(p_i / sum) >= min_prob,   min_prob = fl(fl(p_max / sum) * min_p)
 *
 * where p_max = expf((max - max)/T) = expf(0.0f) == 1.0f EXACTLY. In exact
 * arithmetic the sum cancels and the condition is p_i >= min_p. In float,
 * each of the three roundings perturbs by at most one unit roundoff
 * u = 2^-24, so:
 *   - any candidate the cutoff CAN keep satisfies
 *         p_i >= min_p * (1-u)^2 / (1+u) > min_p * (1 - 3u);
 *   - any candidate with p_i < min_p * (1 - 3u) is cut for EVERY value of
 *     `sum`, i.e. regardless of summation order.
 * A logit-side prefilter that keeps p_i >= min_p * (1 - 4e-6) (~67u of
 * slack, which also absorbs the rounding of this threshold product itself)
 * therefore keeps a SUPERSET of whatever the exact cutoff keeps, under any
 * sum. Because probs are monotone in the logit, that superset is a PREFIX of
 * the descending sort — so the byte-exact cutoff loop, run unchanged over
 * the sorted survivors, walks exactly the candidates it would have walked
 * over the full sorted vocab and trims the boundary with the SAME float
 * comparisons as before. Membership is never decided by the prefilter.
 *
 * The max candidate is kept unconditionally (mirrors the loop's i > 0
 * exemption; also covers min_p > 1 and a NaN temperature poisoning p).
 *
 * DOMAIN: the one-roundoff-per-operation bound only holds while min_prob and
 * pr are NORMAL floats; if min_prob lands in the subnormals (min_p on the
 * order of 1e-38 * sum), ties-to-even at a few ulps can round the cutoff
 * below what the slack covers and the superset claim fails (review finding,
 * 2026-07-17: min_p = 13*2^-149, sum = 2.0, p_i = 12*2^-149 flips). min_p is
 * NOT range-checked server-side, so both paths take the prefilter only for
 * min_p > SAMPLE_MINP_PREFILTER_MIN and fall back to the exact full sort
 * below it: sum <= n_vocab <= ~1.3e5 keeps min_prob >= min_p/sum > 7e-36 —
 * normal, with ~650x margin over FLT_MIN — and boundary p_i ~ min_p are
 * normal too. Requests with 0 < min_p <= 1e-30 are absurd but stay correct. */
#define SAMPLE_MINP_PREFILTER_SLACK (1.0f - 4e-6f)
#define SAMPLE_MINP_PREFILTER_MIN 1e-30f



/* NOTE: the free below also drops `qmap` (and its all-zero invariant with it),
 * because pulsar_sample_scratch_free clears the whole struct. That is safe only
 * because no caller holds live qmap state across a dist_build — the residual
 * draw scatters, reads and re-zeros within one call. Do not cache anything in
 * qmap across calls without decoupling this. */
static void sample_scratch_reserve(pulsar_sample_scratch *s, uint32_t cap) {
    if (s->cap >= cap) return;
    pulsar_sample_scratch_free(s);
    s->cand = (sample_candidate *)xmalloc((size_t)cap * sizeof(*s->cand));
    s->keys = (uint64_t *)xmalloc((size_t)cap * sizeof(*s->keys));
    s->tmp = (uint64_t *)xmalloc((size_t)cap * sizeof(*s->tmp));
    s->cand2 = (sample_candidate *)xmalloc((size_t)cap * sizeof(*s->cand2));
    s->cap = cap;
}



void pulsar_sample_scratch_free(pulsar_sample_scratch *s) {
    free(s->cand);
    free(s->keys);
    free(s->tmp);
    free(s->cand2);
    free(s->qmap);
    memset(s, 0, sizeof(*s));
}



/* Reserve the dense q map, zero-filled. Sized by token id, independent of
 * sample_scratch_reserve's `cap` (which is top_k on the preselect path). Grows
 * by free+calloc: callers restore every entry they set, so a fresh all-zero
 * buffer preserves the invariant. */
static void sample_qmap_reserve(pulsar_sample_scratch *s, uint32_t cap) {
    if (s->qmap_cap >= cap) return;
    free(s->qmap);
    s->qmap = (float *)xcalloc((size_t)cap, sizeof(*s->qmap));
    s->qmap_cap = cap;
}



static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    float max_logit = PULSAR_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    /* Same full-vocab qsort the speculative dist_build had (~10.6 ms a call),
     * on the PLAIN per-token path — reached whenever a client sends top_p < 1
     * without a top_k, which is a common shape. Radix-sorted for the same
     * reason and with the same byte-exactness argument.
     *
     * `sum` here accumulates in VOCAB order, BEFORE the sort (unlike
     * dist_build, which sums post-sort) — so this loop must stay exactly as it
     * was: it still adds EVERY finite candidate's prob, in the same order, so
     * `sum` is bit-identical with the prefilter on or off. probs are computed
     * once and carried THROUGH the sort by packing the compaction index as the
     * radix payload, rather than recomputed post-sort: a second expf loop
     * could be vectorized differently from this one under -ffast-math and
     * differ in the last ulp.
     *
     * With min_p > 0 only the prefilter's survivors (a superset of the min-p
     * cutoff's survivors — see SAMPLE_MINP_PREFILTER_SLACK) are collected for
     * sorting, which is what turns the 129k-candidate sort into a
     * tens-of-candidates sort at the default min_p. The cutoff loop below is
     * unchanged and trims the boundary byte-exactly, so this path's output
     * (token and rng stream) is bit-identical to the unfiltered build. */
    sample_candidate *cand = (sample_candidate *)xmalloc((size_t)finite * sizeof(cand[0]));
    uint64_t *keys = (uint64_t *)xmalloc((size_t)finite * sizeof(keys[0]));
    uint64_t *tmp = (uint64_t *)xmalloc((size_t)finite * sizeof(tmp[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    const float prefilter = min_p > SAMPLE_MINP_PREFILTER_MIN
        ? min_p * SAMPLE_MINP_PREFILTER_SLACK : -1.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        sum += p;
        if (p >= prefilter || (int)i == best) {
            keys[n] = ((uint64_t)sample_desc_key(v) << 32) | n;
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        free(keys);
        free(tmp);
        return best;
    }

    /* finite >= 1 here (checked above), so n >= 1 as sample_radix_sort_desc
     * requires. Stable over an ascending fill => ties keep ascending vocab id,
     * matching the qsort this replaces. */
    sample_radix_sort_desc(keys, tmp, n);
    sample_candidate *sorted = (sample_candidate *)xmalloc((size_t)n * sizeof(sorted[0]));
    for (uint32_t i = 0; i < n; i++) sorted[i] = cand[(uint32_t)keys[i]];
    free(cand);
    free(keys);
    free(tmp);
    cand = sorted;
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}



/* =====
 * Filtered-distribution object for speculative sampling.
 *
 * Builds the SAME final sampling distribution sample_top_p_min_p draws from
 * (temperature scaling, top-k preselect, min-p relative floor, top-p nucleus
 * with the crossing candidate included), but materialized so the speculative
 * path can (a) query p(token), (b) draw, and (c) draw excluding a rejected
 * token — the exact residual rule for deterministic-proposal speculative
 * sampling. Mirrors the sampler branch-for-branch; the spec_sampling harness
 * chi-square-checks the equivalence.
 *
 * NOTE on optimizing this: `sum` is taken over ALL n candidates BEFORE any
 * cutoff, and both cutoffs are relative to it (min_prob =
 * (cand[0].prob/sum)*min_p; the top_p test is filtered_sum/sum). Dropping
 * candidates from `sum` — the NAIVE min-p pre-filter — therefore changes
 * `sum`, hence `filtered`, hence `filtered_sum`, hence EVERY output
 * probability. It is not an equivalent rewrite; tests/pulsar_test.c --sampler
 * catches it (n 6 != 5). The full-vocab sum is load-bearing.
 *
 * What the top_k <= 0, min_p > 0 path DOES do (the LEGAL prefilter): the sum
 * still covers every finite candidate, but is accumulated in VOCAB-INDEX
 * order in the same pass that computes each prob once and collects only the
 * prefilter survivors (see SAMPLE_MINP_PREFILTER_SLACK: a strict superset of
 * the min-p cutoff's survivors under ANY sum, forming a prefix of the
 * descending sort). Only the survivors are sorted — tens instead of 129k at
 * the server-default min_p = 0.05 — and the byte-exact cutoff loop then
 * trims the boundary with the same comparisons as ever, evaluated against
 * that sum. Distribution-preserving by construction; NOT stream-preserving:
 * summing in index order instead of the old sorted-descending order rounds
 * differently (~1e-7 relative), so every output prob moves by that much and
 * rng draws near a bucket edge can flip. Survivor membership and order are
 * unchanged (tests/pulsar_test.c --sampler-prefilter pins set/order identity
 * against the old-sum reference and characterizes the prob delta; --sampler
 * is byte-exact against the re-derived index-order-sum reference). The
 * min_p <= 0 full sort and the top_k > 0 preselect keep the old sorted-order
 * sum and remain byte-identical to the pre-prefilter build.
 *
 * ALIASING CONTRACT: `out`'s ids/probs must never point into `scratch`. The
 * spec walk (session.c) holds one dist at a time while reusing the scratch
 * across accepted positions, so handing `out` a scratch pointer to save the
 * malloc would be silently wrong the moment two dists overlap — and the
 * --sampler gate builds one dist at a time, so it would NOT catch it.
 */
int pulsar_sample_dist_build(const float *logits, uint32_t n_vocab,
                          float temperature, int top_k, float top_p, float min_p,
                          pulsar_sample_scratch *scratch, pulsar_sample_dist *out) {
    memset(out, 0, sizeof(*out));
    if (temperature <= 0.0f) {
        out->ids = (int *)xmalloc(sizeof(int));
        out->probs = (float *)xmalloc(sizeof(float));
        out->ids[0] = sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0 || top_k > 1024) top_k = top_k <= 0 ? 0 : 1024;

    /* collect candidates: full vocab, or top-k preselect like the sampler.
     * Buffers come from the caller's scratch and are reused across calls —
     * the sampled speculative walk calls this per accepted position. */
    uint32_t cap = top_k > 0 ? (uint32_t)top_k : n_vocab;
    sample_scratch_reserve(scratch, cap);
    sample_candidate *cand = scratch->cand;
    /* `sc` is the descending-order candidate view the cutoff tail walks. The
     * prefilter path gathers into scratch->cand2 (probs and sum already
     * computed, index-order sum); the other paths sort `cand` in place and
     * compute probs + sum post-sort, in sorted order, exactly as before. */
    sample_candidate *sc = cand;
    /* LAZY-MATERIALIZATION view for the full-vocab (top_k<=0, min_p small)
     * path. That path used to materialize a 12-byte sample_candidate for
     * EVERY finite vocab entry (129,280 x 12 B = 1.55 MB of strided stores,
     * each pulling logits[id] through a random gather) and then walk the same
     * array AGAIN to fill in .prob — while the cutoff loop below almost always
     * breaks within a few dozen entries and only `filtered` entries are ever
     * read out. Instead it now makes ONE fused pass that writes only the
     * probabilities, PACKED (4 B/entry, into cand's storage — cand itself is
     * unused on that path), and leaves the ids where the radix sort already
     * put them: the low 32 bits of keys[i]. `pv`/`idk` are the resulting
     * lazy view; SC_PROB/SC_ID below read whichever representation is live.
     * Arithmetic is UNCHANGED: same max_logit (logits[keys[0]] IS cand[0].logit),
     * same expf inputs in the same sorted order, same `sum` accumulation
     * order, hence the same cutoff, `filtered` set and output probabilities. */
    const float *pv = NULL;         /* packed probs, sorted order (or NULL) */
    const uint64_t *idk = NULL;     /* sorted keys; id = low 32 bits (or NULL) */
#define SC_PROB(i) (pv ? pv[(i)] : sc[(i)].prob)
#define SC_ID(i)   (idk ? (int)(uint32_t)idk[(i)] : sc[(i)].id)
    uint32_t n = 0;
    int have_probs = 0;
    float sum = 0.0f;
    if (top_k > 0) {
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == (uint32_t)top_k && v <= cand[n - 1].logit) continue;
            uint32_t j = n < (uint32_t)top_k ? n++ : n - 1;
            while (j > 0 && cand[j - 1].logit < v) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j].id = (int)i;
            cand[j].logit = v;
        }
    } else if (min_p > SAMPLE_MINP_PREFILTER_MIN) {
        /* min-p prefilter path (see the block comment above; the MIN floor
         * keeps the cutoff arithmetic normal — below it the exact full sort
         * runs instead). Pass 1: the
         * true max over finite logits, first occurrence on ties — the same
         * candidate the stable descending sort put at cand[0]. */
        float max_logit = 0.0f;
        uint32_t max_id = 0;
        uint32_t finite = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            if (finite == 0 || v > max_logit) {
                max_logit = v;
                max_id = i;
            }
            finite++;
        }
        /* Pass 2: one prob per finite candidate, computed ONCE and carried
         * through the sort; `sum` over ALL of them, in this (index) order;
         * survivors collected in ascending-id order so the stable radix keeps
         * ascending-id as the canonical tie order. */
        if (finite > 0) {
            uint64_t *keys = scratch->keys;
            const float prefilter = min_p * SAMPLE_MINP_PREFILTER_SLACK;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                if (!isfinite(v)) continue;
                const float p = expf((v - max_logit) / temperature);
                sum += p;
                if (p >= prefilter || i == max_id) {
                    keys[n] = ((uint64_t)sample_desc_key(v) << 32) | n;
                    cand[n] = (sample_candidate){
                        .id = (int)i, .logit = v, .prob = p};
                    n++;
                }
            }
            /* n >= 1: the max candidate is always kept. */
            sample_radix_sort_desc(keys, scratch->tmp, n);
            for (uint32_t i = 0; i < n; i++)
                scratch->cand2[i] = cand[(uint32_t)keys[i]];
            sc = scratch->cand2;
            have_probs = 1;
        }
    } else {
        uint64_t *keys = scratch->keys;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            keys[n++] = ((uint64_t)sample_desc_key(v) << 32) | (uint32_t)i;
        }
        if (n) {
            sample_radix_sort_desc(keys, scratch->tmp, n);
            /* Fused prob pass over the packed view (see the note at `pv`).
             * Byte-identical to the old materialize-then-!have_probs-loop:
             * max_logit, the expf inputs and the `sum` order all match. */
            float *probs = (float *)cand;   /* n*4 <= cap*sizeof(sample_candidate) */
            const float max_logit = logits[(uint32_t)keys[0]];
            for (uint32_t i = 0; i < n; i++) {
                const float p =
                    expf((logits[(uint32_t)keys[i]] - max_logit) / temperature);
                probs[i] = p;
                sum += p;
            }
            pv = probs;
            idk = keys;
            have_probs = 1;
        }
    }
    if (n == 0) {
        out->ids = (int *)xmalloc(sizeof(int));
        out->probs = (float *)xmalloc(sizeof(float));
        out->ids[0] = sample_argmax(logits, n_vocab);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }

    if (!have_probs) {
        const float max_logit = sc[0].logit;
        for (uint32_t i = 0; i < n; i++) {
            sc[i].prob = expf((sc[i].logit - max_logit) / temperature);
            sum += sc[i].prob;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        out->ids = (int *)xmalloc(sizeof(int));
        out->probs = (float *)xmalloc(sizeof(float));
        out->ids[0] = SC_ID(0);
        out->probs[0] = 1.0f;
        out->n = 1;
        return 1;
    }
    const float min_prob = (SC_PROB(0) / sum) * min_p;
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = SC_PROB(i);
        const float pr = p / sum;
        if (i > 0 && pr < min_prob) break;
        filtered_sum += p;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) filtered = 1;
    out->ids = (int *)xmalloc((size_t)filtered * sizeof(int));
    out->probs = (float *)xmalloc((size_t)filtered * sizeof(float));
    out->n = filtered;
    for (uint32_t i = 0; i < filtered; i++) {
        out->ids[i] = SC_ID(i);
        out->probs[i] = SC_PROB(i) / filtered_sum;   /* renormalized nucleus */
    }
    return 1;
#undef SC_PROB
#undef SC_ID
}

void pulsar_sample_dist_free(pulsar_sample_dist *d) {
    free(d->ids);
    free(d->probs);
    memset(d, 0, sizeof(*d));
}

float pulsar_sample_dist_prob(const pulsar_sample_dist *d, int token) {
    for (uint32_t i = 0; i < d->n; i++)
        if (d->ids[i] == token) return d->probs[i];
    return 0.0f;
}

int pulsar_sample_dist_accept(const pulsar_sample_dist *d, int token, uint64_t *rng) {
    const float pd = pulsar_sample_dist_prob(d, token);
    if (pd >= 1.0f) return 1;
    if (pd <= 0.0f) return 0;
    return sample_rng_f32(rng) < pd;
}

int pulsar_sample_dist_draw(const pulsar_sample_dist *d, uint64_t *rng) {
    float r = sample_rng_f32(rng);
    for (uint32_t i = 0; i < d->n; i++) {
        r -= d->probs[i];
        if (r <= 0.0f) return d->ids[i];
    }
    return d->ids[d->n - 1];
}

/* =====
 * Sampled-proposal speculative rule (Leviathan/Chen). The pair above
 * (_accept / _draw_excluding) is the DETERMINISTIC-proposal rule: it accepts an
 * argmax draft with probability p(draft), so its acceptance is mathematically
 * capped at p(mode). Drawing the draft from a temperature-matched q instead and
 * accepting w.p. min(1, p/q) removes that cap; both rules emit exactly p.
 *
 * Two bugs in the reference implementation this is modelled on
 * (xangel82/DS4-GB10-GX10-DSpark-CUDA — technique only, no code taken) are
 * fixed here, and the unit gate tests/pulsar_test.c --spec-math pins both:
 *   (a) its `u <= ap` accept test can emit a token with p(x) == 0 (u==0 draws
 *       accept an impossible token). We reject p <= 0 outright and use a
 *       strict `<`, matching pulsar_sample_dist_accept's discipline.
 *   (b) its residual fallback can land on a token whose residual mass is zero.
 *       We track the last STRICTLY POSITIVE residual index instead.
 */
int pulsar_sample_dist_accept_pq(const pulsar_sample_dist *p, int token, float q, uint64_t *rng) {
    const float pd = pulsar_sample_dist_prob(p, token);
    /* bug (a): a token outside p's nucleus is impossible under the target and
     * must never be emitted, whatever u is. */
    if (pd <= 0.0f) return 0;
    /* Defensive: the token was drawn from q, so q(token) > 0 by construction.
     * A non-positive q here would mean the stored q lost sync with the draft;
     * accepting is the exact-preserving choice (p(token) > 0 is established). */
    if (q <= 0.0f) return 1;
    /* min(1, p/q) == 1: certain accept, and — like _accept's `pd >= 1` fast
     * path — consume no rng. Load-bearing for temperature<=0 byte-identity:
     * greedy makes p and q both point masses of 1.0 at the argmax, so this
     * returns 1 having touched neither the rng stream nor its ordering. */
    if (pd >= q) return 1;
    /* u < p/q, written without the divide. u in [0,1). */
    return sample_rng_f32(rng) * q < pd;
}

int pulsar_sample_dist_draw_residual(const pulsar_sample_dist *p, const pulsar_sample_dist *q,
                                  pulsar_sample_scratch *scratch, uint64_t *rng) {
    /* r(x) = max(0, p(x) - q(x)). r(x) > 0 requires p(x) > 0, so the union of
     * the two supports collapses to p's support: an x in q but not in p has
     * r = max(0, 0 - q(x)) = 0 and cannot be drawn. Iterating p alone is
     * therefore complete, and every id considered already has p(x) > 0. */
    uint32_t maxid = 0;
    for (uint32_t i = 0; i < q->n; i++) {
        if (q->ids[i] < 0) continue;
        if ((uint32_t)q->ids[i] > maxid) maxid = (uint32_t)q->ids[i];
    }
    sample_qmap_reserve(scratch, maxid + 1u);
    for (uint32_t i = 0; i < q->n; i++)
        if (q->ids[i] >= 0) scratch->qmap[q->ids[i]] = q->probs[i];

    float mass = 0.0f;
    for (uint32_t i = 0; i < p->n; i++) {
        const int id = p->ids[i];
        const float qv = (id >= 0 && (uint32_t)id <= maxid) ? scratch->qmap[id] : 0.0f;
        const float r = p->probs[i] - qv;
        if (r > 0.0f) mass += r;
    }

    int out;
    if (mass <= 0.0f) {
        /* Degenerate: q dominates p across p's whole support. Exactness is
         * already lost in the numerics here; emit SOMETHING p can produce
         * rather than an impossible token. */
        out = pulsar_sample_dist_draw(p, rng);
    } else {
        float r_acc = sample_rng_f32(rng) * mass;
        int last = -1;
        out = -1;
        for (uint32_t i = 0; i < p->n; i++) {
            const int id = p->ids[i];
            const float qv = (id >= 0 && (uint32_t)id <= maxid) ? scratch->qmap[id] : 0.0f;
            const float r = p->probs[i] - qv;
            if (r <= 0.0f) continue;
            /* bug (b): remember the last id with STRICTLY POSITIVE residual, so
             * the float-rounding overrun below cannot land on a zero-residual
             * token. Every candidate here has p(x) > 0 by construction. */
            last = id;
            r_acc -= r;
            if (r_acc <= 0.0f) { out = id; break; }
        }
        if (out < 0) out = last;   /* overrun: mass > 0 guarantees last >= 0 */
    }

    /* restore the all-zero invariant (only q's own ids were touched) */
    for (uint32_t i = 0; i < q->n; i++)
        if (q->ids[i] >= 0) scratch->qmap[q->ids[i]] = 0.0f;
    return out;
}

int pulsar_sample_dist_draw_excluding(const pulsar_sample_dist *d, int excluded, uint64_t *rng) {
    /* residual of a rejected deterministic proposal: p with `excluded`
     * removed, renormalized. If the nucleus is exactly {excluded}, there is
     * no residual mass — the caller treats that as accept-forced. */
    float mass = 0.0f;
    for (uint32_t i = 0; i < d->n; i++)
        if (d->ids[i] != excluded) mass += d->probs[i];
    if (mass <= 0.0f) return excluded;
    float r = sample_rng_f32(rng) * mass;
    int last = excluded;
    for (uint32_t i = 0; i < d->n; i++) {
        if (d->ids[i] == excluded) continue;
        last = d->ids[i];
        r -= d->probs[i];
        if (r <= 0.0f) return d->ids[i];
    }
    return last;
}



int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}



static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const pulsar_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "pulsar: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}






/* GPU generation entry point.  The model runs as one local whole-graph
 * pipeline: graph prefill followed by graph decode steps.  Streaming PRO may
 * use decode-style prefill for short prompts. */
int generate_gpu_graph_raw_swa(
        const pulsar_model   * model,
        const pulsar_vocab   * vocab,
        const pulsar_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        uint32_t            prefill_chunk,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        pulsar_token_emit_fn   emit,
        pulsar_generation_done_fn done,
        void              * emit_ud,
        pulsar_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "pulsar: using GPU graph generation with graph prefill\n");

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "pulsar: prompt is empty or exceeds context size\n");
        return 1;
    }

    const uint32_t prefill_cap =
        gpu_graph_prefill_cap_for_prompt(prompt->len, prefill_chunk);
    const uint32_t raw_cap = gpu_graph_raw_cap_for_context(ctx_size, prefill_cap);
    if (prefill_cap < (uint32_t)prompt->len) {
        fprintf(stderr,
                "pulsar: using chunked GPU prefill (%u-token chunks for %d prompt tokens)\n",
                prefill_cap,
                prompt->len);
    }
    pulsar_gpu_graph g;
    bool ok = gpu_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size, prefill_cap, false);
    if (!ok) {
        fprintf(stderr, "pulsar: failed to allocate GPU graph runtime\n");
        return 1;
    }
    g.quality = quality;
    if (!gpu_graph_load_directional_steering(&g,
                                               directional_steering_file,
                                               directional_steering_attn,
                                               directional_steering_ffn)) {
        gpu_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("PULSAR_CUDA_MEMORY_REPORT") != NULL;
    if (memory_report) pulsar_gpu_print_memory_report("after graph alloc");

    float *logits = (float *)xmalloc((size_t)PULSAR_N_VOCAB * sizeof(logits[0]));
    const bool trace_top = getenv("PULSAR_TRACE_TOP") != NULL;
    const bool token_timing = getenv("PULSAR_TOKEN_TIMING") != NULL;

    const double t_prefill0 = now_sec();
    if (prefill_cap < (uint32_t)prompt->len) {
        ok = gpu_graph_prefill_chunked(&g, model, weights, prompt,
                                         prompt->len, logits, false,
                                         progress, progress_ud,
                                         progress, progress_ud,
                                         NULL, NULL, NULL);
    } else {
        ok = gpu_graph_prefill_raw_swa(&g, model, weights, prompt,
                                         prompt->len, logits, true,
                                         progress, progress_ud,
                                         NULL, NULL, NULL);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) pulsar_gpu_print_memory_report("after prefill");

    if (!ok) {
        free(logits);
        gpu_graph_free(&g);
        return 1;
    }
    const char *dump_prefill_logits = getenv("PULSAR_CUDA_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, PULSAR_N_VOCAB)) {
            free(logits);
            gpu_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "pulsar: wrote GPU prefill logits to %s\n", dump_prefill_logits);
    }

    int pos = prompt->len;
    int n_generated = 0;
    int n_decode_eval = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, PULSAR_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, PULSAR_N_VOCAB);
        if (token == vocab->eos_id) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = gpu_graph_eval_token_raw_swa(&g,
                                            model,
                                            weights,
                                            (uint32_t)token,
                                            (uint32_t)pos,
                                            logits);
        if (!ok) break;
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "pulsar: gpu decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    pulsar_log(stderr,
            PULSAR_LOG_TIMING,
            "pulsar: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) pulsar_gpu_print_memory_report("before graph free");
    free(logits);
    gpu_graph_free(&g);
    return ok ? 0 : 1;
}




/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

const char *pulsar_backend_name(pulsar_backend backend) {
    switch (backend) {
    case PULSAR_BACKEND_CUDA:  return "cuda";
    }
    return "unknown";
}

