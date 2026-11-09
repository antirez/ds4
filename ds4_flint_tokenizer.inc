/* Recursive Flint (fork-owned): vocabulary ids and the pre-tokenizer.
 * Textually included into ds4.c just before bpe_tokenize_text, after the
 * UTF-8 helpers and bpe_emit_piece it uses.
 *
 * Flint's tokenizer.json is byte-level BPE behind three successive
 * "isolated" splits, applied in sequence to the pieces of the previous one
 * (llama.cpp's LLAMA_VOCAB_PRE_TYPE_FLINT lists the same three):
 *
 *   1. \p{N}{1,3}
 *   2. [一-龥぀-ゟ゠-ヿ]+
 *   3. [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *    | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
 *    |  ?[\p{P}\p{S}]+[\r\n]*
 *    | \s*[\r\n]+
 *    | \s+(?!\S)
 *    | \s+
 *
 * An isolated split keeps the matched spans and the gaps between them as
 * separate pieces, so text the third pattern does not match (digits, control
 * characters) still reaches BPE as its own piece.  Every piece is then
 * byte-encoded and merged by bpe_emit_piece.  The alternatives of the third
 * pattern are tried in order with regex backtracking semantics; the comments
 * at each step say which backtracking case they cover.
 * tests/flint/check_tokenizer.py compares the result with HF tokenizers. */

#include "ds4_flint_unicode.inc"

static int vocab_lookup(const ds4_vocab *vocab, const char *text);
static int vocab_lookup_optional(const ds4_vocab *vocab, const char *text);

/* <|developer|> has no ds4_vocab slot; it is a generation stop for Flint. */
static int g_flint_developer_id = -1;
/* Added tokens occupy [g_flint_first_added_id, n_vocab): the 144 role, tool,
 * think and reserved tokens, every one of which HF tokenizers matches
 * literally in text (added tokens are matched whether or not "special"). */
static int g_flint_first_added_id = -1;

static void flint_vocab_setup(ds4_vocab *vocab) {
    /* GLM-style role tokens, no BOS in prompts, <|end_of_text|> as EOS.
     * <|user|>, <|observation|> and <|developer|> also end generation
     * (config.json eos_token_id); the stop predicate handles those. */
    vocab->bos_id = -1;
    vocab->eos_id = vocab_lookup(vocab, "<|end_of_text|>");
    vocab->system_id = vocab_lookup(vocab, "<|system|>");
    vocab->user_id = vocab_lookup(vocab, "<|user|>");
    vocab->assistant_id = vocab_lookup(vocab, "<|assistant|>");
    vocab->observation_id = vocab_lookup(vocab, "<|observation|>");
    vocab->sop_id = -1;
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->tool_call_start_id = vocab_lookup(vocab, "<tool_call>");
    vocab->tool_call_end_id = vocab_lookup(vocab, "</tool_call>");
    vocab->tool_response_start_id = vocab_lookup(vocab, "<tool_response>");
    vocab->tool_response_end_id = vocab_lookup(vocab, "</tool_response>");
    vocab->arg_key_start_id = vocab_lookup(vocab, "<arg_key>");
    vocab->arg_key_end_id = vocab_lookup(vocab, "</arg_key>");
    vocab->arg_value_start_id = vocab_lookup(vocab, "<arg_value>");
    vocab->arg_value_end_id = vocab_lookup(vocab, "</arg_value>");
    vocab->dsml_id = -1;
    vocab->im_start_id = -1;
    vocab->im_end_id = -1;
    vocab->endoftext_id = -1;
    g_flint_developer_id = vocab_lookup(vocab, "<|developer|>");
    g_flint_first_added_id = vocab_lookup(vocab, "<|begin_of_text|>");
    (void)vocab_lookup_optional;
}

/* Replaces special_token_at for Flint: longest added token starting at p. */
static bool flint_special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    if (g_flint_first_added_id < 0 || *p != '<') return false;
    size_t best_len = 0;
    int best = -1;
    for (int id = g_flint_first_added_id; id < vocab->n_vocab; id++) {
        const ds4_str t = vocab->token[id];
        if (t.len > best_len && !strncmp(p, t.ptr, t.len)) {
            best_len = t.len;
            best = id;
        }
    }
    if (best < 0) return false;
    *token = best;
    *len = best_len;
    return true;
}

static bool flint_urange_has(const ds4_flint_urange *r, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid].lo) hi = mid;
        else if (cp > r[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

#define DS4_FLINT_CLASS(name_, cp_) \
    flint_urange_has(ds4_flint_##name_##_ranges, \
                     sizeof(ds4_flint_##name_##_ranges) / sizeof(ds4_flint_##name_##_ranges[0]), (cp_))

typedef struct {
    uint32_t cp;
    uint64_t next;
    bool valid;
    bool letter;    /* \p{L} */
    bool number;    /* \p{N} */
    bool space;     /* White_Space */
    bool upperish;  /* \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}: the first class of the word patterns */
    bool lowerish;  /* \p{Ll}\p{Lm}\p{Lo}\p{M}: the second class */
    bool punct;     /* \p{P}\p{S} */
    bool newline;   /* \r or \n */
} flint_char_info;

static flint_char_info flint_char_at(const char *s, uint64_t len, uint64_t pos) {
    flint_char_info c;
    memset(&c, 0, sizeof(c));
    if (pos >= len) return c;
    c.valid = true;
    c.cp = utf8_peek_one(s, len, pos, &c.next);
    if (c.next <= pos) c.next = pos + 1;
    c.letter = DS4_FLINT_CLASS(letter, c.cp);
    c.number = DS4_FLINT_CLASS(number, c.cp);
    c.space = DS4_FLINT_CLASS(space, c.cp);
    const bool mark = DS4_FLINT_CLASS(mark, c.cp);
    const bool lother = DS4_FLINT_CLASS(lother, c.cp);
    c.upperish = mark || lother || DS4_FLINT_CLASS(upper, c.cp);
    c.lowerish = mark || lother || DS4_FLINT_CLASS(lower, c.cp);
    c.punct = DS4_FLINT_CLASS(punct, c.cp);
    c.newline = c.cp == '\r' || c.cp == '\n';
    return c;
}

/* [U]*[Lw]+ from `start` with regex backtracking: U* takes its maximal
 * run, Lw+ continues greedily; if nothing follows the U run, U* gives
 * characters back until the last one it kept is in Lw.  Returns the end or
 * UINT64_MAX. */
static uint64_t flint_match_upper_lower(const char *s, uint64_t len, uint64_t start) {
    uint64_t u_end = start;
    uint64_t last_lw = UINT64_MAX;   /* start of the last Lw char inside the U run */
    while (u_end < len) {
        flint_char_info c = flint_char_at(s, len, u_end);
        if (!c.valid || !c.upperish) break;
        if (c.lowerish) last_lw = u_end;
        u_end = c.next;
    }
    uint64_t l_end = u_end;
    while (l_end < len) {
        flint_char_info c = flint_char_at(s, len, l_end);
        if (!c.valid || !c.lowerish) break;
        l_end = c.next;
    }
    if (l_end > u_end) return l_end;
    if (last_lw == UINT64_MAX) return UINT64_MAX;
    /* Lw+ restarts at last_lw; the characters after it up to u_end are
     * upper-only, so the match ends right after it. */
    return flint_char_at(s, len, last_lw).next;
}

/* [U]+[Lw]* from `start`: no backtracking needed. */
static uint64_t flint_match_upper_any(const char *s, uint64_t len, uint64_t start) {
    uint64_t u_end = start;
    while (u_end < len) {
        flint_char_info c = flint_char_at(s, len, u_end);
        if (!c.valid || !c.upperish) break;
        u_end = c.next;
    }
    if (u_end == start) return UINT64_MAX;
    uint64_t l_end = u_end;
    while (l_end < len) {
        flint_char_info c = flint_char_at(s, len, l_end);
        if (!c.valid || !c.lowerish) break;
        l_end = c.next;
    }
    return l_end;
}

/* (?i:'s|'t|'re|'ve|'m|'ll|'d)? after a word */
static uint64_t flint_contraction_end(const char *s, uint64_t len, uint64_t pos) {
    if (pos >= len || s[pos] != '\'') return pos;
    if (pos + 1 >= len) return pos;
    const uint32_t c1 = ascii_tolower_cp((uint8_t)s[pos + 1]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
    if (pos + 2 >= len) return pos;
    const uint32_t c2 = ascii_tolower_cp((uint8_t)s[pos + 2]);
    if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return pos + 3;
    return pos;
}

/* The third split's first match at `pos` within piece [0, len); UINT64_MAX
 * when no alternative matches there. */
static uint64_t flint_word_match(const char *s, uint64_t len, uint64_t pos) {
    const flint_char_info cur = flint_char_at(s, len, pos);
    if (!cur.valid) return UINT64_MAX;

    /* Alternatives 1 and 2, each first with the optional prefix character,
     * then without it (the prefix may itself be a mark, so both can apply). */
    const bool prefix_ok = !cur.newline && !cur.letter && !cur.number;
    uint64_t end;
    if (prefix_ok && (end = flint_match_upper_lower(s, len, cur.next)) != UINT64_MAX)
        return flint_contraction_end(s, len, end);
    if ((end = flint_match_upper_lower(s, len, pos)) != UINT64_MAX)
        return flint_contraction_end(s, len, end);
    if (prefix_ok && (end = flint_match_upper_any(s, len, cur.next)) != UINT64_MAX)
        return flint_contraction_end(s, len, end);
    if ((end = flint_match_upper_any(s, len, pos)) != UINT64_MAX)
        return flint_contraction_end(s, len, end);

    /*  ?[\p{P}\p{S}]+[\r\n]* */
    {
        uint64_t p = pos;
        flint_char_info c = cur;
        if (c.cp == ' ') {
            p = c.next;
            c = flint_char_at(s, len, p);
        }
        if (c.valid && c.punct) {
            while (p < len) {
                flint_char_info scan = flint_char_at(s, len, p);
                if (!scan.valid || !scan.punct) break;
                p = scan.next;
            }
            while (p < len) {
                flint_char_info scan = flint_char_at(s, len, p);
                if (!scan.valid || !scan.newline) break;
                p = scan.next;
            }
            return p;
        }
    }

    /* \s*[\r\n]+ | \s+(?!\S) | \s+ over the whitespace run at pos */
    if (cur.space) {
        uint64_t p = pos;
        uint64_t last_newline_end = 0;
        uint64_t last_ws_start = pos;
        int nspace = 0;
        while (p < len) {
            flint_char_info scan = flint_char_at(s, len, p);
            if (!scan.valid || !scan.space) break;
            last_ws_start = p;
            if (scan.newline) last_newline_end = scan.next;
            p = scan.next;
            nspace++;
        }
        if (last_newline_end) return last_newline_end;        /* \s*[\r\n]+ ends at the last newline */
        if (p >= len) return p;                                /* \s+(?!\S): nothing follows */
        if (nspace > 1) return last_ws_start;                  /* \s+(?!\S): keep one for the next word */
        return p;                                              /* \s+ */
    }
    return UINT64_MAX;
}

/* Third split over one piece: matches and the gaps between them. */
static void flint_split_words(const ds4_vocab *vocab, const char *s, uint64_t len, token_vec *out) {
    uint64_t pos = 0;
    uint64_t gap_start = UINT64_MAX;
    while (pos < len) {
        const uint64_t end = flint_word_match(s, len, pos);
        if (end == UINT64_MAX || end <= pos) {
            if (gap_start == UINT64_MAX) gap_start = pos;
            pos = next_utf8_char(s, len, pos);
            if (pos <= gap_start) pos = gap_start + 1;
            continue;
        }
        if (gap_start != UINT64_MAX) {
            bpe_emit_piece(vocab, (ds4_str){ s + gap_start, pos - gap_start }, out);
            gap_start = UINT64_MAX;
        }
        bpe_emit_piece(vocab, (ds4_str){ s + pos, end - pos }, out);
        pos = end;
    }
    if (gap_start != UINT64_MAX) {
        bpe_emit_piece(vocab, (ds4_str){ s + gap_start, len - gap_start }, out);
    }
}

/* Second split: CJK, hiragana and katakana runs. */
static void flint_split_cjk(const ds4_vocab *vocab, const char *s, uint64_t len, token_vec *out) {
    uint64_t pos = 0;
    uint64_t gap_start = UINT64_MAX;
    while (pos < len) {
        uint64_t next = pos;
        const uint32_t cp = utf8_peek_one(s, len, pos, &next);
        if (next <= pos) next = pos + 1;
        if (!utf8_is_cjk_hira_kata(cp)) {
            if (gap_start == UINT64_MAX) gap_start = pos;
            pos = next;
            continue;
        }
        if (gap_start != UINT64_MAX) {
            flint_split_words(vocab, s + gap_start, pos - gap_start, out);
            gap_start = UINT64_MAX;
        }
        uint64_t run_end = next;
        while (run_end < len) {
            uint64_t n2 = run_end;
            const uint32_t c2 = utf8_peek_one(s, len, run_end, &n2);
            if (!utf8_is_cjk_hira_kata(c2) || n2 <= run_end) break;
            run_end = n2;
        }
        flint_split_words(vocab, s + pos, run_end - pos, out);
        pos = run_end;
    }
    if (gap_start != UINT64_MAX) flint_split_words(vocab, s + gap_start, len - gap_start, out);
}

/* First split: one to three digits. */
static void bpe_tokenize_text_flint(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;
    uint64_t gap_start = UINT64_MAX;
    while (pos < len) {
        uint64_t next = pos;
        const uint32_t cp = utf8_peek_one(text, len, pos, &next);
        if (next <= pos) next = pos + 1;
        if (!DS4_FLINT_CLASS(number, cp)) {
            if (gap_start == UINT64_MAX) gap_start = pos;
            pos = next;
            continue;
        }
        if (gap_start != UINT64_MAX) {
            flint_split_cjk(vocab, text + gap_start, pos - gap_start, out);
            gap_start = UINT64_MAX;
        }
        uint64_t run_end = next;
        int n = 1;
        while (n < 3 && run_end < len) {
            uint64_t n2 = run_end;
            const uint32_t c2 = utf8_peek_one(text, len, run_end, &n2);
            if (!DS4_FLINT_CLASS(number, c2) || n2 <= run_end) break;
            run_end = n2;
            n++;
        }
        flint_split_cjk(vocab, text + pos, run_end - pos, out);
        pos = run_end;
    }
    if (gap_start != UINT64_MAX) flint_split_cjk(vocab, text + gap_start, len - gap_start, out);
}

/* ---- chat template (rec_mlx/chat_template.jinja), the parts step 2 needs ----
 * No BOS.  <|system|>text, <|developer|>text, <|user|>text, <|assistant|>text,
 * <|observation|><tool_response>text</tool_response>; the generation prompt is
 * <|assistant|>.  The tool preamble and thinking control follow in step 5. */
static void flint_encode_chat_prompt(const ds4_vocab *vocab, const char *system, const char *prompt, token_vec *out) {
    if (system && system[0]) {
        token_vec_push(out, vocab->system_id);
        bpe_tokenize_text_flint(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text_flint(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
}

static void flint_chat_append_message(const ds4_vocab *vocab, token_vec *out, const char *role, const char *content) {
    if (!strcmp(role, "system")) {
        token_vec_push(out, vocab->system_id);
        bpe_tokenize_text_flint(vocab, content, out);
    } else if (!strcmp(role, "developer")) {
        token_vec_push(out, g_flint_developer_id >= 0 ? g_flint_developer_id : vocab->system_id);
        bpe_tokenize_text_flint(vocab, content, out);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(out, vocab->assistant_id);
        bpe_tokenize_text_flint(vocab, content, out);
    } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        token_vec_push(out, vocab->observation_id);
        token_vec_push(out, vocab->tool_response_start_id);
        bpe_tokenize_text_flint(vocab, content, out);
        token_vec_push(out, vocab->tool_response_end_id);
    } else {
        token_vec_push(out, vocab->user_id);
        bpe_tokenize_text_flint(vocab, content, out);
    }
}

/* Generation prompt: <|assistant|> then <think> (rec's grammar forces the
 * first token to <think>, so prefilling it leaves the distribution of the
 * rest unchanged) or, with thinking off, the template's empty think block. */
static void flint_chat_append_assistant_prefix(const ds4_vocab *vocab, ds4_think_mode think_mode, token_vec *out) {
    token_vec_push(out, vocab->assistant_id);
    token_vec_push(out, vocab->think_start_id);
    if (!ds4_think_mode_enabled(think_mode)) token_vec_push(out, vocab->think_end_id);
}

/* config.json eos_token_id: <|end_of_text|>, <|user|>, <|observation|>, <|developer|>. */
static bool flint_token_is_generation_stop(const ds4_vocab *vocab, int token) {
    return token == vocab->eos_id || token == vocab->user_id || token == vocab->observation_id ||
           (g_flint_developer_id >= 0 && token == g_flint_developer_id);
}
