/* DeepSeek V4 ("joyai-llm") pre-tokenizer against the pieces the official HF
 * tokenizer produces; regenerate the vectors with
 * tests/gen_joyai_pretokenizer_vectors.py. */
#include "../ds4.c"

typedef struct {
    const char *text;
    const char *const *pieces;
} joyai_vector;

#include "joyai_pretokenizer_vectors.inc"

typedef struct {
    const char *const *expected;
    int n;
    bool ok;
} joyai_check;

static void check_piece(void *ud, ds4_str piece) {
    joyai_check *c = ud;
    const char *want = c->expected[c->n];
    if (!want || strlen(want) != piece.len || memcmp(want, piece.ptr, piece.len) != 0) c->ok = false;
    if (want) c->n++;
}

static void print_bytes(const char *s, uint64_t len) {
    putchar('"');
    for (uint64_t i = 0; i < len; i++) {
        const uint8_t b = (uint8_t)s[i];
        if (b >= 0x20 && b < 0x7f && b != '"' && b != '\\') putchar(b);
        else printf("\\x%02x", b);
    }
    putchar('"');
}

static void print_piece(void *ud, ds4_str piece) {
    (void)ud;
    putchar(' ');
    print_bytes(piece.ptr, piece.len);
}

int main(void) {
    const int n = (int)(sizeof(joyai_vectors) / sizeof(joyai_vectors[0]));
    int failed = 0;
    for (int i = 0; i < n; i++) {
        const joyai_vector *v = &joyai_vectors[i];
        joyai_check c = { v->pieces, 0, true };
        joyai_pretokenize(v->text, strlen(v->text), check_piece, &c);
        if (c.ok && v->pieces[c.n] == NULL) continue;

        failed++;
        printf("FAIL ");
        print_bytes(v->text, strlen(v->text));
        printf("\n  got:");
        joyai_pretokenize(v->text, strlen(v->text), print_piece, NULL);
        printf("\n  want:");
        for (int j = 0; v->pieces[j]; j++) {
            putchar(' ');
            print_bytes(v->pieces[j], strlen(v->pieces[j]));
        }
        putchar('\n');
    }
    printf("joyai pre-tokenizer: %d/%d vectors match\n", n - failed, n);
    return failed ? 1 : 0;
}
