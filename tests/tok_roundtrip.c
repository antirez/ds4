/* Does rendering a token prefix reproduce the prompt's bytes? (kv cache text keys) */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: model prompt_file [cut_tokens...]\n"); return 1; }
    ds4_engine_options opt = { .model_path = argv[1], .backend = DS4_BACKEND_METAL, .n_threads = 1,
                               .context_size = 1024, .ssd_streaming = true };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    FILE *fp = fopen(argv[2], "rb"); fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    char *text = malloc(n + 1); fread(text, 1, n, fp); text[n] = 0; fclose(fp);
    ds4_tokens t = {0};
    ds4_tokenize_text(e, text, &t);
    printf("bytes %ld tokens %d\n", n, t.len);
    for (int a = 3; a < argc; a++) {
        int cut = atoi(argv[a]); if (cut > t.len) cut = t.len;
        size_t off = 0; int first_bad = -1;
        for (int i = 0; i < cut; i++) {
            size_t len = 0; char *piece = ds4_token_text(e, t.v[i], &len);
            if (first_bad < 0 && (off + len > (size_t)n || memcmp(text + off, piece, len) != 0)) {
                first_bad = i;
                printf("cut %d: mismatch at token %d byte %zu: piece=%.*s| text=%.*s|\n", cut, i, off,
                       (int)len, piece, (int)len, text + off);
            }
            off += len; free(piece);
        }
        printf("cut %d: rendered %zu bytes, %s\n", cut, off, first_bad < 0 ? "byte-exact prefix" : "NOT a prefix");
    }
    return 0;
}
