#include "ds4.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) die("fseek failed");
    long n = ftell(fp);
    if (n < 0) die("ftell failed");
    if (fseek(fp, 0, SEEK_SET) != 0) die("fseek failed");
    char *buf = malloc((size_t)n + 1);
    if (!buf) die("out of memory");
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die("read failed");
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

static void strip_newline(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) s[--n] = '\0';
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s MODEL manifest.tsv OUT.tsv [ctx]\n", argv[0]);
        return 2;
    }

    const char *model_path = argv[1];
    const char *manifest_path = argv[2];
    const char *out_path = argv[3];
    int ctx_size = argc == 5 ? atoi(argv[4]) : 4096;
    if (ctx_size < 1024) ctx_size = 1024;

    ds4_engine_options opt = {
        .model_path = model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 0,
        .warm_weights = false,
        .quality = false,
    };

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) die("failed to create session");

    FILE *mf = fopen(manifest_path, "rb");
    if (!mf) {
        fprintf(stderr, "open %s: %s\n", manifest_path, strerror(errno));
        return 1;
    }
    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "open %s: %s\n", out_path, strerror(errno));
        return 1;
    }
    fprintf(out, "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\n");

    char line[8192];
    int case_n = 0;
    double total_nll = 0.0;
    long total_tokens = 0;
    long total_lcp = 0;
    long first_matches = 0;
    char err[256];

    while (fgets(line, sizeof(line), mf)) {
        strip_newline(line);
        if (!line[0] || line[0] == '#') continue;

        char *id = strtok(line, "\t");
        char *prompt_path = strtok(NULL, "\t");
        char *cont_path = strtok(NULL, "\t");
        if (!id || !prompt_path || !cont_path) die("bad manifest row");

        char *prompt_text = read_file(prompt_path);
        char *cont_text = read_file(cont_path);

        ds4_tokens prompt = {0};
        ds4_tokens target = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);
        ds4_tokenize_text(engine, cont_text, &target);

        if (prompt.len + target.len + 1 >= ctx_size) {
            fprintf(stderr, "%s exceeds ctx=%d\n", id, ctx_size);
            return 1;
        }
        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s sync failed: %s\n", id, err);
            return 1;
        }

        double nll = 0.0;
        int lcp = 0;
        bool still_matching = true;
        bool first_match = false;
        for (int i = 0; i < target.len; i++) {
            const int greedy = ds4_session_argmax(session);
            if (i == 0) first_match = (greedy == target.v[i]);
            if (still_matching && greedy == target.v[i]) lcp++;
            else still_matching = false;

            ds4_token_score score;
            if (!ds4_session_token_logprob(session, target.v[i], &score)) {
                fprintf(stderr, "%s logprob failed at target token %d\n", id, i);
                return 1;
            }
            nll += -(double)score.logprob;

            if (ds4_session_eval(session, target.v[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "%s eval failed at target token %d: %s\n", id, i, err);
                return 1;
            }
        }

        const double avg = target.len ? nll / (double)target.len : 0.0;
        fprintf(out, "%s\t%d\t%d\t%.9f\t%.9f\t%d\t%d\n",
                id, prompt.len, target.len, nll, avg, first_match ? 1 : 0, lcp);
        fflush(out);

        case_n++;
        total_nll += nll;
        total_tokens += target.len;
        total_lcp += lcp;
        first_matches += first_match ? 1 : 0;
        fprintf(stderr,
                "%s cases=%d prompt=%d target=%d avg_nll=%.6f lcp=%d\n",
                id, case_n, prompt.len, target.len, avg, lcp);

        ds4_tokens_free(&prompt);
        ds4_tokens_free(&target);
        free(prompt_text);
        free(cont_text);
    }

    fprintf(stderr,
            "summary cases=%d tokens=%ld avg_nll=%.9f first_match=%ld avg_lcp=%.3f\n",
            case_n,
            total_tokens,
            total_tokens ? total_nll / (double)total_tokens : 0.0,
            first_matches,
            case_n ? (double)total_lcp / (double)case_n : 0.0);

    fclose(out);
    fclose(mf);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
