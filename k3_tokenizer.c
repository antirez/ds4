#define _POSIX_C_SOURCE 200809L

#include "k3_tokenizer.h"
#include "k3_safetensors.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <unicode/uregex.h>
#include <unicode/ustring.h>

enum {
    K3_BASE_VOCAB = 163584,
};

typedef struct {
    uint8_t *bytes;
    uint32_t length;
} k3_token_bytes;

typedef struct {
    const uint8_t *bytes;
    uint32_t length;
    int32_t rank;
    int used;
} k3_rank_entry;

struct k3_tokenizer {
    k3_token_bytes token[K3_BASE_VOCAB];
    k3_rank_entry *table;
    uint32_t table_capacity;
    URegularExpression *regex;
};

static void set_error(char *err, size_t cap, const char *fmt, ...) {
    if (!err || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, cap, fmt, ap);
    va_end(ap);
}

static uint64_t hash_bytes(const uint8_t *bytes, uint32_t length) {
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint32_t next_power_of_two(uint32_t value) {
    uint32_t result = 1;
    while (result < value && result <= UINT32_MAX / 2u) result <<= 1u;
    return result;
}

static int rank_table_put(k3_tokenizer *tokenizer,
                          const uint8_t *bytes,
                          uint32_t length,
                          int32_t rank) {
    const uint32_t mask = tokenizer->table_capacity - 1u;
    uint32_t slot = (uint32_t)hash_bytes(bytes, length) & mask;
    while (tokenizer->table[slot].used) {
        if (tokenizer->table[slot].length == length &&
            memcmp(tokenizer->table[slot].bytes, bytes, length) == 0) {
            return 0;
        }
        slot = (slot + 1u) & mask;
    }
    tokenizer->table[slot] = (k3_rank_entry){
        .bytes = bytes,
        .length = length,
        .rank = rank,
        .used = 1,
    };
    return 1;
}

static int32_t rank_table_get(const k3_tokenizer *tokenizer,
                              const uint8_t *bytes,
                              uint32_t length) {
    if (!length) return -1;
    const uint32_t mask = tokenizer->table_capacity - 1u;
    uint32_t slot = (uint32_t)hash_bytes(bytes, length) & mask;
    while (tokenizer->table[slot].used) {
        if (tokenizer->table[slot].length == length &&
            memcmp(tokenizer->table[slot].bytes, bytes, length) == 0) {
            return tokenizer->table[slot].rank;
        }
        slot = (slot + 1u) & mask;
    }
    return -1;
}

static int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static int decode_base64(const char *input,
                         size_t input_length,
                         uint8_t **out,
                         uint32_t *out_length) {
    if (!input || !out || !out_length || input_length % 4u != 0 ||
        input_length > (size_t)UINT32_MAX * 4u / 3u) {
        return 0;
    }
    const size_t capacity = input_length / 4u * 3u;
    uint8_t *bytes = malloc(capacity ? capacity : 1u);
    if (!bytes) return 0;
    size_t count = 0;
    for (size_t i = 0; i < input_length; i += 4u) {
        int value[4];
        for (uint32_t j = 0; j < 4; j++) {
            value[j] = input[i + j] == '='
                ? -2
                : base64_value((unsigned char)input[i + j]);
            if (value[j] < 0 && value[j] != -2) {
                free(bytes);
                return 0;
            }
        }
        if (value[0] < 0 || value[1] < 0 ||
            (value[2] == -2 && value[3] != -2)) {
            free(bytes);
            return 0;
        }
        const uint32_t packed =
            ((uint32_t)value[0] << 18u) |
            ((uint32_t)value[1] << 12u) |
            ((uint32_t)(value[2] < 0 ? 0 : value[2]) << 6u) |
            (uint32_t)(value[3] < 0 ? 0 : value[3]);
        bytes[count++] = (uint8_t)(packed >> 16u);
        if (value[2] != -2) bytes[count++] = (uint8_t)(packed >> 8u);
        if (value[3] != -2) bytes[count++] = (uint8_t)packed;
    }
    *out = bytes;
    *out_length = (uint32_t)count;
    return 1;
}

static int tokens_push(k3_tokens *tokens, int32_t token) {
    if (tokens->count == tokens->capacity) {
        uint32_t capacity = tokens->capacity ? tokens->capacity * 2u : 128u;
        if (capacity < tokens->capacity) return 0;
        int32_t *data = realloc(
            tokens->data, (size_t)capacity * sizeof(*data));
        if (!data) return 0;
        tokens->data = data;
        tokens->capacity = capacity;
    }
    tokens->data[tokens->count++] = token;
    return 1;
}

void k3_tokens_free(k3_tokens *tokens) {
    if (!tokens) return;
    free(tokens->data);
    memset(tokens, 0, sizeof(*tokens));
}

static int compile_regex(k3_tokenizer *tokenizer,
                         char *err,
                         size_t errcap) {
    static const char pattern_utf8[] =
        "[\\p{Han}]+|"
        "[^\\r\\n\\p{L}\\p{N}]?"
        "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        "[^\\r\\n\\p{L}\\p{N}]?"
        "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*"
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        "\\p{N}{1,3}|"
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|"
        "\\s+(?!\\S)|"
        "\\s+";
    UErrorCode status = U_ZERO_ERROR;
    int32_t pattern_length = 0;
    u_strFromUTF8(
        NULL, 0, &pattern_length,
        pattern_utf8, (int32_t)strlen(pattern_utf8), &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        set_error(err, errcap, "ICU could not size K3 tokenizer regex");
        return 0;
    }
    status = U_ZERO_ERROR;
    UChar *pattern = malloc(
        ((size_t)pattern_length + 1u) * sizeof(*pattern));
    if (!pattern) {
        set_error(err, errcap, "out of memory compiling K3 tokenizer regex");
        return 0;
    }
    u_strFromUTF8(
        pattern, pattern_length + 1, NULL,
        pattern_utf8, (int32_t)strlen(pattern_utf8), &status);
    UParseError parse_error;
    memset(&parse_error, 0, sizeof(parse_error));
    if (U_SUCCESS(status)) {
        tokenizer->regex = uregex_open(
            pattern, pattern_length, 0, &parse_error, &status);
    }
    free(pattern);
    if (U_FAILURE(status) || !tokenizer->regex) {
        set_error(err, errcap,
                  "ICU rejected K3 tokenizer regex at offset %d: %s",
                  parse_error.offset, u_errorName(status));
        return 0;
    }
    return 1;
}

int k3_tokenizer_open(k3_tokenizer **out,
                      const char *model_directory,
                      char *err,
                      size_t errcap) {
    if (out) *out = NULL;
    if (!out || !model_directory || !model_directory[0]) {
        set_error(err, errcap, "missing K3 tokenizer directory");
        return 0;
    }
    char path[4096];
    struct stat model_stat;
    const int is_gguf =
        stat(model_directory, &model_stat) == 0 &&
        S_ISREG(model_stat.st_mode);
    const size_t directory_length = strlen(model_directory);
    const char *slash =
        directory_length && model_directory[directory_length - 1u] == '/'
            ? ""
            : "/";
    const int n = is_gguf
        ? snprintf(path, sizeof(path), "%s", model_directory)
        : snprintf(
            path, sizeof(path), "%s%stiktoken.model",
            model_directory, slash);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        set_error(err, errcap, "K3 tokenizer path is too long");
        return 0;
    }
    uint8_t *gguf_tokenizer = NULL;
    size_t gguf_tokenizer_size = 0;
    FILE *fp = NULL;
    if (is_gguf) {
        if (!k3_st_gguf_read_string(
                path,
                "tokenizer.kimi-k3.tiktoken",
                &gguf_tokenizer,
                &gguf_tokenizer_size,
                err,
                errcap)) {
            return 0;
        }
        fp = fmemopen(gguf_tokenizer, gguf_tokenizer_size, "rb");
    } else {
        fp = fopen(path, "rb");
    }
    if (!fp) {
        set_error(err, errcap, "could not open %s: %s", path, strerror(errno));
        free(gguf_tokenizer);
        return 0;
    }
    k3_tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) {
        fclose(fp);
        free(gguf_tokenizer);
        set_error(err, errcap, "out of memory opening K3 tokenizer");
        return 0;
    }
    tokenizer->table_capacity =
        next_power_of_two(K3_BASE_VOCAB * 3u);
    tokenizer->table = calloc(
        tokenizer->table_capacity, sizeof(*tokenizer->table));
    if (!tokenizer->table) {
        fclose(fp);
        free(gguf_tokenizer);
        k3_tokenizer_close(tokenizer);
        set_error(err, errcap, "out of memory indexing K3 tokenizer");
        return 0;
    }

    char *line = NULL;
    size_t line_capacity = 0;
    ssize_t line_length;
    uint32_t count = 0;
    while ((line_length = getline(&line, &line_capacity, fp)) >= 0) {
        while (line_length > 0 &&
               (line[line_length - 1] == '\n' ||
                line[line_length - 1] == '\r')) {
            line[--line_length] = '\0';
        }
        char *space = strrchr(line, ' ');
        if (!space || count >= K3_BASE_VOCAB) {
            set_error(err, errcap, "malformed K3 tiktoken line %u", count + 1u);
            goto fail;
        }
        *space++ = '\0';
        char *rank_end = NULL;
        const long rank = strtol(space, &rank_end, 10);
        uint8_t *bytes = NULL;
        uint32_t length = 0;
        if (!rank_end || *rank_end != '\0' ||
            rank < 0 || rank >= K3_BASE_VOCAB ||
            !decode_base64(
                line, strlen(line), &bytes, &length) ||
            tokenizer->token[rank].bytes ||
            !rank_table_put(tokenizer, bytes, length, (int32_t)rank)) {
            free(bytes);
            set_error(err, errcap, "invalid K3 tiktoken rank on line %u",
                      count + 1u);
            goto fail;
        }
        tokenizer->token[rank] = (k3_token_bytes){
            .bytes = bytes,
            .length = length,
        };
        count++;
    }
    free(line);
    fclose(fp);
    free(gguf_tokenizer);
    gguf_tokenizer = NULL;
    line = NULL;
    fp = NULL;
    if (count != K3_BASE_VOCAB ||
        !compile_regex(tokenizer, err, errcap)) {
        if (count != K3_BASE_VOCAB) {
            set_error(err, errcap,
                      "K3 tokenizer has %u base tokens, expected %u",
                      count, K3_BASE_VOCAB);
        }
        goto fail;
    }
    *out = tokenizer;
    return 1;

fail:
    free(line);
    if (fp) fclose(fp);
    free(gguf_tokenizer);
    k3_tokenizer_close(tokenizer);
    return 0;
}

void k3_tokenizer_close(k3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->regex) uregex_close(tokenizer->regex);
    for (uint32_t i = 0; i < K3_BASE_VOCAB; i++) {
        free(tokenizer->token[i].bytes);
    }
    free(tokenizer->table);
    free(tokenizer);
}

static int emit_piece(const k3_tokenizer *tokenizer,
                      const uint8_t *bytes,
                      uint32_t length,
                      k3_tokens *out) {
    if (!length) return 1;
    const int32_t whole = rank_table_get(tokenizer, bytes, length);
    if (whole >= 0) return tokens_push(out, whole);

    uint32_t *boundaries = malloc(
        ((size_t)length + 1u) * sizeof(*boundaries));
    if (!boundaries) return 0;
    uint32_t parts = length;
    for (uint32_t i = 0; i <= length; i++) boundaries[i] = i;
    while (parts > 1u) {
        int32_t best_rank = INT32_MAX;
        uint32_t best = UINT32_MAX;
        for (uint32_t i = 0; i + 1u < parts; i++) {
            const uint32_t begin = boundaries[i];
            const uint32_t end = boundaries[i + 2u];
            const int32_t rank =
                rank_table_get(tokenizer, bytes + begin, end - begin);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best = i;
            }
        }
        if (best == UINT32_MAX) break;
        memmove(
            &boundaries[best + 1u],
            &boundaries[best + 2u],
            (size_t)(parts - best - 1u) * sizeof(*boundaries));
        parts--;
    }
    for (uint32_t i = 0; i < parts; i++) {
        const uint32_t begin = boundaries[i];
        const uint32_t end = boundaries[i + 1u];
        const int32_t rank =
            rank_table_get(tokenizer, bytes + begin, end - begin);
        if (rank < 0 || !tokens_push(out, rank)) {
            free(boundaries);
            return 0;
        }
    }
    free(boundaries);
    return 1;
}

static int encode_ordinary(k3_tokenizer *tokenizer,
                           const char *text,
                           k3_tokens *out,
                           char *err,
                           size_t errcap) {
    const size_t utf8_length = strlen(text);
    if (!utf8_length) return 1;
    if (utf8_length > INT32_MAX) {
        set_error(err, errcap, "K3 tokenizer input is too long");
        return 0;
    }
    UErrorCode status = U_ZERO_ERROR;
    int32_t utf16_length = 0;
    u_strFromUTF8WithSub(
        NULL, 0, &utf16_length, text, (int32_t)utf8_length,
        0xfffd, NULL, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR) {
        set_error(err, errcap, "ICU could not size K3 tokenizer input");
        return 0;
    }
    status = U_ZERO_ERROR;
    UChar *utf16 = malloc(
        ((size_t)utf16_length + 1u) * sizeof(*utf16));
    if (!utf16) {
        set_error(err, errcap, "out of memory converting K3 tokenizer input");
        return 0;
    }
    u_strFromUTF8WithSub(
        utf16, utf16_length + 1, NULL,
        text, (int32_t)utf8_length, 0xfffd, NULL, &status);
    if (U_FAILURE(status)) {
        free(utf16);
        set_error(err, errcap, "ICU could not convert K3 tokenizer input");
        return 0;
    }
    uregex_setText(tokenizer->regex, utf16, utf16_length, &status);
    int32_t covered = 0;
    while (U_SUCCESS(status) && uregex_findNext(tokenizer->regex, &status)) {
        const int32_t begin = uregex_start(tokenizer->regex, 0, &status);
        const int32_t end = uregex_end(tokenizer->regex, 0, &status);
        if (U_FAILURE(status) || begin != covered || end <= begin) break;
        int32_t piece_length = 0;
        UErrorCode convert_status = U_ZERO_ERROR;
        u_strToUTF8(
            NULL, 0, &piece_length, utf16 + begin, end - begin,
            &convert_status);
        if (convert_status != U_BUFFER_OVERFLOW_ERROR || piece_length <= 0) {
            status = U_INTERNAL_PROGRAM_ERROR;
            break;
        }
        convert_status = U_ZERO_ERROR;
        uint8_t *piece = malloc((size_t)piece_length);
        if (!piece) {
            status = U_MEMORY_ALLOCATION_ERROR;
            break;
        }
        u_strToUTF8(
            (char *)piece, piece_length, NULL,
            utf16 + begin, end - begin, &convert_status);
        const int ok = U_SUCCESS(convert_status) &&
                       emit_piece(
                           tokenizer, piece, (uint32_t)piece_length, out);
        free(piece);
        if (!ok) {
            status = U_MEMORY_ALLOCATION_ERROR;
            break;
        }
        covered = end;
    }
    static const UChar empty_text[1] = {0};
    UErrorCode reset_status = U_ZERO_ERROR;
    uregex_setText(tokenizer->regex, empty_text, 0, &reset_status);
    if (U_SUCCESS(status) && U_FAILURE(reset_status)) status = reset_status;
    free(utf16);
    if (U_FAILURE(status) || covered != utf16_length) {
        set_error(err, errcap, "K3 tokenizer regex failed: %s",
                  u_errorName(status));
        return 0;
    }
    return 1;
}

int k3_tokenizer_encode(k3_tokenizer *tokenizer,
                        const char *text,
                        k3_tokens *out,
                        char *err,
                        size_t errcap) {
    if (!tokenizer || !text || !out) {
        set_error(err, errcap, "invalid K3 tokenization request");
        return 0;
    }
    out->count = 0;
    return encode_ordinary(tokenizer, text, out, err, errcap);
}

int k3_tokenizer_encode_rendered(k3_tokenizer *tokenizer,
                                 const char *text,
                                 k3_tokens *out,
                                 char *err,
                                 size_t errcap) {
    struct marker {
        const char *text;
        int32_t token;
    };
    static const struct marker markers[] = {
        {"<|end_of_msg|>", K3_TOKEN_END_OF_MSG},
        {"<|open|>", K3_TOKEN_OPEN},
        {"<|close|>", K3_TOKEN_CLOSE},
        {"<|sep|>", K3_TOKEN_SEP},
        {"[BOS]", K3_TOKEN_BOS},
        {"[EOS]", K3_TOKEN_EOS},
        {"[EOT]", K3_TOKEN_EOT},
    };
    if (!tokenizer || !text || !out) {
        set_error(err, errcap, "invalid K3 rendered tokenization request");
        return 0;
    }
    out->count = 0;
    const size_t text_length = strlen(text);
    char *ordinary = malloc(text_length + 1u);
    if (!ordinary) {
        set_error(err, errcap,
                  "out of memory tokenizing rendered K3 prompt");
        return 0;
    }
    size_t ordinary_length = 0;
    const char *p = text;
    while (*p) {
        const struct marker *match = NULL;
        size_t match_length = 0;
        if (*p == '\\') {
            for (uint32_t i = 0;
                 i < sizeof(markers) / sizeof(markers[0]); i++) {
                const size_t n = strlen(markers[i].text);
                if (!strncmp(p + 1, markers[i].text, n)) {
                    memcpy(
                        ordinary + ordinary_length,
                        markers[i].text,
                        n);
                    ordinary_length += n;
                    p += n + 1u;
                    match_length = SIZE_MAX;
                    break;
                }
            }
            if (match_length == SIZE_MAX) continue;
        }
        for (uint32_t i = 0;
             i < sizeof(markers) / sizeof(markers[0]); i++) {
            const size_t n = strlen(markers[i].text);
            if (!strncmp(p, markers[i].text, n)) {
                match = &markers[i];
                match_length = n;
                break;
            }
        }
        if (!match) {
            ordinary[ordinary_length++] = *p++;
            continue;
        }
        ordinary[ordinary_length] = '\0';
        if (!encode_ordinary(
                tokenizer, ordinary, out, err, errcap) ||
            !tokens_push(out, match->token)) {
            free(ordinary);
            return 0;
        }
        ordinary_length = 0;
        p += match_length;
    }
    ordinary[ordinary_length] = '\0';
    const int ok =
        encode_ordinary(tokenizer, ordinary, out, err, errcap);
    free(ordinary);
    return ok;
}

static int encode_segment(k3_tokenizer *tokenizer,
                          const char *text,
                          k3_tokens *out,
                          char *err,
                          size_t errcap) {
    return encode_ordinary(tokenizer, text, out, err, errcap);
}

static int open_tag(k3_tokenizer *tokenizer,
                    const char *tag,
                    const char *role,
                    const char *type,
                    k3_tokens *out,
                    char *err,
                    size_t errcap) {
    if (!tokens_push(out, K3_TOKEN_OPEN) ||
        !encode_segment(tokenizer, tag, out, err, errcap)) {
        return 0;
    }
    if (role &&
        (!encode_segment(tokenizer, " role", out, err, errcap) ||
         !encode_segment(tokenizer, "=\"", out, err, errcap) ||
         !encode_segment(tokenizer, role, out, err, errcap) ||
         !encode_segment(tokenizer, "\"", out, err, errcap))) {
        return 0;
    }
    if (type &&
        (!encode_segment(tokenizer, " type", out, err, errcap) ||
         !encode_segment(tokenizer, "=\"", out, err, errcap) ||
         !encode_segment(tokenizer, type, out, err, errcap) ||
         !encode_segment(tokenizer, "\"", out, err, errcap))) {
        return 0;
    }
    return tokens_push(out, K3_TOKEN_SEP);
}

static int close_tag(k3_tokenizer *tokenizer,
                     const char *tag,
                     k3_tokens *out,
                     char *err,
                     size_t errcap) {
    return tokens_push(out, K3_TOKEN_CLOSE) &&
           encode_segment(tokenizer, tag, out, err, errcap) &&
           tokens_push(out, K3_TOKEN_SEP);
}

static int append_attribute(k3_tokenizer *tokenizer,
                            const char *name,
                            const char *value,
                            k3_tokens *out,
                            char *err,
                            size_t errcap) {
    const char *src = value ? value : "";
    const size_t length = strlen(src);
    if (length > (SIZE_MAX - 1u) / 6u) {
        set_error(err, errcap, "K3 XTML attribute is too long");
        return 0;
    }
    char *escaped = malloc(length * 6u + 1u);
    if (!escaped) {
        set_error(err, errcap, "out of memory encoding K3 XTML attribute");
        return 0;
    }
    size_t n = 0;
    for (size_t i = 0; i < length; i++) {
        const char *replacement = NULL;
        if (src[i] == '&') replacement = "&amp;";
        else if (src[i] == '"') replacement = "&quot;";
        if (replacement) {
            const size_t replacement_length = strlen(replacement);
            memcpy(escaped + n, replacement, replacement_length);
            n += replacement_length;
        } else {
            escaped[n++] = src[i];
        }
    }
    escaped[n] = '\0';
    const size_t name_length = strlen(name);
    char *key = malloc(name_length + 2u);
    if (!key) {
        free(escaped);
        set_error(err, errcap, "out of memory encoding K3 XTML attribute");
        return 0;
    }
    key[0] = ' ';
    memcpy(key + 1u, name, name_length + 1u);
    const int ok =
        encode_segment(tokenizer, key, out, err, errcap) &&
        encode_segment(tokenizer, "=\"", out, err, errcap) &&
        encode_segment(tokenizer, escaped, out, err, errcap) &&
        encode_segment(tokenizer, "\"", out, err, errcap);
    free(key);
    free(escaped);
    return ok;
}

int k3_tokenizer_append_text(k3_tokenizer *tokenizer,
                             const char *text,
                             k3_tokens *out,
                             char *err,
                             size_t errcap) {
    if (!tokenizer || !out) {
        set_error(err, errcap, "invalid K3 text append request");
        return 0;
    }
    return encode_ordinary(
        tokenizer, text ? text : "", out, err, errcap);
}

int k3_tokenizer_append_thinking_effort(k3_tokenizer *tokenizer,
                                        const char *effort,
                                        k3_tokens *out,
                                        char *err,
                                        size_t errcap) {
    static const char prefix[] =
        "`thinking_effort` guides on how much to think in your thinking "
        "channel (not including the response channel), supported values "
        "include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=";
    if (!tokenizer || !out || !effort ||
        (strcmp(effort, "low") && strcmp(effort, "medium") &&
         strcmp(effort, "high") && strcmp(effort, "max"))) {
        set_error(err, errcap, "invalid K3 thinking effort");
        return 0;
    }
    char body[sizeof(prefix) + 16];
    const int body_length = snprintf(
        body, sizeof(body), "%s%s`.", prefix, effort);
    if (body_length <= 0 || (size_t)body_length >= sizeof(body)) {
        set_error(err, errcap, "K3 thinking effort is too long");
        return 0;
    }
    return open_tag(tokenizer, "message", "system", "thinking-effort",
                    out, err, errcap) &&
           encode_segment(tokenizer, body, out, err, errcap) &&
           close_tag(tokenizer, "message", out, err, errcap) &&
           tokens_push(out, K3_TOKEN_END_OF_MSG);
}

int k3_tokenizer_append_message(k3_tokenizer *tokenizer,
                                const char *role,
                                const char *content,
                                k3_tokens *out,
                                char *err,
                                size_t errcap) {
    if (!tokenizer || !out) {
        set_error(err, errcap, "invalid K3 message append request");
        return 0;
    }
    if (!role || !role[0]) role = "user";
    if (!strcmp(role, "developer")) role = "system";
    if (!strcmp(role, "function")) role = "tool";
    if (!content) content = "";

    if (!open_tag(tokenizer, "message", role, NULL, out, err, errcap)) {
        return 0;
    }
    if (!strcmp(role, "assistant")) {
        if (!open_tag(tokenizer, "response", NULL, NULL, out, err, errcap) ||
            !encode_segment(tokenizer, content, out, err, errcap) ||
            !close_tag(tokenizer, "response", out, err, errcap)) {
            return 0;
        }
    } else if (!encode_segment(tokenizer, content, out, err, errcap)) {
        return 0;
    }
    return close_tag(tokenizer, "message", out, err, errcap) &&
           tokens_push(out, K3_TOKEN_END_OF_MSG);
}

int k3_tokenizer_append_tool_declaration(k3_tokenizer *tokenizer,
                                         const char *tools_json,
                                         k3_tokens *out,
                                         char *err,
                                         size_t errcap) {
    static const char prefix[] =
        "# Tools\n"
        "Here are the available tools, described in JSONSchema.\n\n"
        "```json\n";
    static const char suffix[] = "\n```";
    if (!tokenizer || !out || !tools_json) {
        set_error(err, errcap, "invalid K3 tool declaration append request");
        return 0;
    }
    const size_t body_length =
        strlen(prefix) + strlen(tools_json) + strlen(suffix);
    char *body = malloc(body_length + 1u);
    if (!body) {
        set_error(err, errcap, "out of memory encoding K3 tool declaration");
        return 0;
    }
    snprintf(body, body_length + 1u, "%s%s%s", prefix, tools_json, suffix);
    const int ok =
        open_tag(tokenizer, "message", "system", "tool-declare",
                 out, err, errcap) &&
        encode_segment(tokenizer, body, out, err, errcap) &&
        close_tag(tokenizer, "message", out, err, errcap) &&
        tokens_push(out, K3_TOKEN_END_OF_MSG);
    free(body);
    return ok;
}

int k3_tokenizer_append_tool_result(k3_tokenizer *tokenizer,
                                    const char *tool,
                                    uint32_t index,
                                    const char *content,
                                    k3_tokens *out,
                                    char *err,
                                    size_t errcap) {
    char index_text[32];
    const int index_length =
        snprintf(index_text, sizeof(index_text), "%u", index);
    if (!tokenizer || !out || !tool || !tool[0] || index == 0 ||
        index_length <= 0 || (size_t)index_length >= sizeof(index_text)) {
        set_error(err, errcap, "invalid K3 tool result append request");
        return 0;
    }
    return tokens_push(out, K3_TOKEN_OPEN) &&
           encode_segment(tokenizer, "message", out, err, errcap) &&
           append_attribute(tokenizer, "role", "tool", out, err, errcap) &&
           append_attribute(tokenizer, "tool", tool, out, err, errcap) &&
           append_attribute(
               tokenizer, "index", index_text, out, err, errcap) &&
           tokens_push(out, K3_TOKEN_SEP) &&
           encode_segment(
               tokenizer, content ? content : "", out, err, errcap) &&
           close_tag(tokenizer, "message", out, err, errcap) &&
           tokens_push(out, K3_TOKEN_END_OF_MSG);
}

int k3_tokenizer_append_assistant_close(k3_tokenizer *tokenizer,
                                        k3_tokens *out,
                                        char *err,
                                        size_t errcap) {
    if (!tokenizer || !out) {
        set_error(err, errcap, "invalid K3 assistant close append request");
        return 0;
    }
    return close_tag(tokenizer, "message", out, err, errcap);
}

int k3_tokenizer_append_assistant_prefix(k3_tokenizer *tokenizer,
                                         int thinking,
                                         k3_tokens *out,
                                         char *err,
                                         size_t errcap) {
    return tokenizer && out &&
           open_tag(tokenizer, "message", "assistant", NULL,
                    out, err, errcap) &&
           open_tag(tokenizer, thinking ? "think" : "response", NULL, NULL,
                    out, err, errcap);
}

int k3_tokenizer_encode_chat(k3_tokenizer *tokenizer,
                             const char *prompt,
                             int thinking,
                             k3_tokens *out,
                             char *err,
                             size_t errcap) {
    if (!tokenizer || !prompt || !out) {
        set_error(err, errcap, "invalid K3 chat tokenization request");
        return 0;
    }
    out->count = 0;
    if (thinking &&
        !k3_tokenizer_append_thinking_effort(
            tokenizer, "max", out, err, errcap)) {
        return 0;
    }
    if (!k3_tokenizer_append_message(
            tokenizer, "user", prompt, out, err, errcap) ||
        !k3_tokenizer_append_assistant_prefix(
            tokenizer, thinking, out, err, errcap)) {
        return 0;
    }
    return 1;
}

const uint8_t *k3_tokenizer_token_bytes(const k3_tokenizer *tokenizer,
                                        int32_t token,
                                        uint32_t *length) {
    static const uint8_t empty[] = "";
    if (length) *length = 0;
    static const uint8_t bos[] = "[BOS]";
    static const uint8_t eos[] = "[EOS]";
    static const uint8_t end_of_msg[] = "<|end_of_msg|>";
    static const uint8_t open[] = "<|open|>";
    static const uint8_t close[] = "<|close|>";
    static const uint8_t sep[] = "<|sep|>";
    static const uint8_t eot[] = "[EOT]";
    struct special {
        int32_t token;
        const uint8_t *bytes;
        uint32_t length;
    };
    static const struct special specials[] = {
        {K3_TOKEN_BOS, bos, sizeof(bos) - 1u},
        {K3_TOKEN_EOS, eos, sizeof(eos) - 1u},
        {K3_TOKEN_END_OF_MSG, end_of_msg, sizeof(end_of_msg) - 1u},
        {K3_TOKEN_OPEN, open, sizeof(open) - 1u},
        {K3_TOKEN_CLOSE, close, sizeof(close) - 1u},
        {K3_TOKEN_SEP, sep, sizeof(sep) - 1u},
        {K3_TOKEN_EOT, eot, sizeof(eot) - 1u},
    };
    if (!tokenizer || token < 0) return empty;
    if (token >= K3_BASE_VOCAB) {
        for (uint32_t i = 0;
             i < sizeof(specials) / sizeof(specials[0]); i++) {
            if (specials[i].token != token) continue;
            if (length) *length = specials[i].length;
            return specials[i].bytes;
        }
        return empty;
    }
    if (length) *length = tokenizer->token[token].length;
    return tokenizer->token[token].bytes;
}
