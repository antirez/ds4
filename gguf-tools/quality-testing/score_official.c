#include "ds4.h"
#include "ds4_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s MODEL manifest.tsv OUT.tsv [ctx] "
            "[--ssd-streaming] [--ssd-streaming-cold] "
            "[--ssd-streaming-cache-experts N|NGB] "
            "[--ssd-streaming-preload-experts N]\n",
            prog);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_positive_int(const char *s, const char *opt) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || !end || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "%s must be a positive integer\n", opt);
        exit(2);
    }
    return (int)v;
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

typedef struct {
    unsigned char *bytes;
    int len;
    double logprob;
} api_alt;

typedef struct {
    double logprob;
    api_alt *alts;
    int n_alts;
    int cap_alts;
} api_pos;

typedef struct {
    api_pos *pos;
    int n_pos;
    int cap_pos;
} api_ref;

typedef struct {
    long target_count;
    double target_abs_delta;
    double target_signed_delta;

    long top_items;
    long top_mapped;
    long top_logprob_count;
    double top_abs_delta;
    double top_signed_delta;

    long top1_count;
    long top1_match;
    long topn_ref;
    long topn_hit;
    long pair_total;
    long pair_agree;
} api_metrics;

static const char *json_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *json_skip_string(const char *p) {
    if (*p != '"') return p;
    p++;
    while (*p) {
        if (*p == '\\') {
            p++;
            if (*p) p++;
            continue;
        }
        if (*p == '"') return p + 1;
        p++;
    }
    return p;
}

static const char *json_skip_value(const char *p) {
    p = json_ws(p);
    if (*p == '"') return json_skip_string(p);
    if (*p == '{') {
        p++;
        while (*p) {
            p = json_ws(p);
            if (*p == '}') return p + 1;
            if (*p == '"') p = json_skip_string(p);
            p = json_ws(p);
            if (*p == ':') p++;
            p = json_skip_value(p);
            p = json_ws(p);
            if (*p == ',') p++;
        }
        return p;
    }
    if (*p == '[') {
        p++;
        while (*p) {
            p = json_ws(p);
            if (*p == ']') return p + 1;
            p = json_skip_value(p);
            p = json_ws(p);
            if (*p == ',') p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != ']' && *p != '}') p++;
    return p;
}

static bool json_key(const char **pp, char *out, size_t cap) {
    const char *p = json_ws(*pp);
    if (*p != '"') return false;
    p++;
    size_t n = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\') {
            ch = *p ? *p++ : '\0';
            if (ch == 'n') ch = '\n';
            else if (ch == 'r') ch = '\r';
            else if (ch == 't') ch = '\t';
            else if (ch == 'b') ch = '\b';
            else if (ch == 'f') ch = '\f';
            else if (ch == 'u') {
                for (int i = 0; i < 4 && isxdigit((unsigned char)*p); i++) p++;
                ch = '?';
            }
        }
        if (n + 1 < cap) out[n++] = ch;
    }
    if (*p != '"') return false;
    out[n] = '\0';
    *pp = p + 1;
    return true;
}

static bool json_number(const char **pp, double *out) {
    const char *p = json_ws(*pp);
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return false;
    *out = v;
    *pp = end;
    return true;
}

static bool json_bytes_array(const char **pp, unsigned char **out, int *len) {
    const char *p = json_ws(*pp);
    if (*p != '[') return false;
    p++;
    int cap = 16;
    int n = 0;
    unsigned char *buf = malloc((size_t)cap);
    if (!buf) die("out of memory");
    while (1) {
        p = json_ws(p);
        if (*p == ']') {
            *out = buf;
            *len = n;
            *pp = p + 1;
            return true;
        }
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > 255) {
            free(buf);
            return false;
        }
        if (n == cap) {
            cap *= 2;
            unsigned char *tmp = realloc(buf, (size_t)cap);
            if (!tmp) die("out of memory");
            buf = tmp;
        }
        buf[n++] = (unsigned char)v;
        p = json_ws(end);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p != ']') {
            free(buf);
            return false;
        }
    }
}

static void api_alt_free(api_alt *alt) {
    free(alt->bytes);
    memset(alt, 0, sizeof(*alt));
}

static void api_pos_free(api_pos *pos) {
    for (int i = 0; i < pos->n_alts; i++) api_alt_free(&pos->alts[i]);
    free(pos->alts);
    memset(pos, 0, sizeof(*pos));
}

static void api_ref_free(api_ref *ref) {
    for (int i = 0; i < ref->n_pos; i++) api_pos_free(&ref->pos[i]);
    free(ref->pos);
    memset(ref, 0, sizeof(*ref));
}

static void api_pos_add_alt(api_pos *pos, api_alt *alt) {
    if (pos->n_alts == pos->cap_alts) {
        int cap = pos->cap_alts ? pos->cap_alts * 2 : 16;
        api_alt *tmp = realloc(pos->alts, (size_t)cap * sizeof(pos->alts[0]));
        if (!tmp) die("out of memory");
        pos->alts = tmp;
        pos->cap_alts = cap;
    }
    pos->alts[pos->n_alts++] = *alt;
    memset(alt, 0, sizeof(*alt));
}

static void api_ref_add_pos(api_ref *ref, api_pos *pos) {
    if (ref->n_pos == ref->cap_pos) {
        int cap = ref->cap_pos ? ref->cap_pos * 2 : 32;
        api_pos *tmp = realloc(ref->pos, (size_t)cap * sizeof(ref->pos[0]));
        if (!tmp) die("out of memory");
        ref->pos = tmp;
        ref->cap_pos = cap;
    }
    ref->pos[ref->n_pos++] = *pos;
    memset(pos, 0, sizeof(*pos));
}

static bool api_parse_alt(const char **pp, api_alt *alt) {
    memset(alt, 0, sizeof(*alt));
    alt->logprob = NAN;
    const char *p = json_ws(*pp);
    if (*p != '{') return false;
    p++;
    while (1) {
        p = json_ws(p);
        if (*p == '}') {
            *pp = p + 1;
            return isfinite(alt->logprob);
        }
        char key[64];
        if (!json_key(&p, key, sizeof(key))) return false;
        p = json_ws(p);
        if (*p != ':') return false;
        p++;
        if (strcmp(key, "bytes") == 0) {
            free(alt->bytes);
            alt->bytes = NULL;
            alt->len = 0;
            p = json_ws(p);
            if (strncmp(p, "null", 4) == 0) {
                p += 4;
            } else if (!json_bytes_array(&p, &alt->bytes, &alt->len)) {
                return false;
            }
        } else if (strcmp(key, "logprob") == 0) {
            if (!json_number(&p, &alt->logprob)) return false;
        } else {
            p = json_skip_value(p);
        }
        p = json_ws(p);
        if (*p == ',') p++;
    }
}

static bool api_parse_alt_array(const char **pp, api_pos *pos) {
    const char *p = json_ws(*pp);
    if (*p != '[') return false;
    p++;
    while (1) {
        p = json_ws(p);
        if (*p == ']') {
            *pp = p + 1;
            return true;
        }
        api_alt alt = {0};
        if (!api_parse_alt(&p, &alt)) {
            api_alt_free(&alt);
            return false;
        }
        api_pos_add_alt(pos, &alt);
        p = json_ws(p);
        if (*p == ',') p++;
    }
}

static bool api_parse_pos(const char **pp, api_pos *pos) {
    memset(pos, 0, sizeof(*pos));
    pos->logprob = NAN;
    const char *p = json_ws(*pp);
    if (*p != '{') return false;
    p++;
    while (1) {
        p = json_ws(p);
        if (*p == '}') {
            *pp = p + 1;
            return true;
        }
        char key[64];
        if (!json_key(&p, key, sizeof(key))) return false;
        p = json_ws(p);
        if (*p != ':') return false;
        p++;
        if (strcmp(key, "logprob") == 0) {
            if (!json_number(&p, &pos->logprob)) return false;
        } else if (strcmp(key, "top_logprobs") == 0) {
            if (!api_parse_alt_array(&p, pos)) return false;
        } else {
            p = json_skip_value(p);
        }
        p = json_ws(p);
        if (*p == ',') p++;
    }
}

static bool api_ref_load(const char *path, api_ref *ref) {
    memset(ref, 0, sizeof(*ref));
    if (!path || !path[0]) return false;
    char *json = read_file(path);
    const char *p = strstr(json, "\"logprobs\"");
    if (p) p = strstr(p, "\"content\"");
    if (p) p = strchr(p, '[');
    if (!p) {
        free(json);
        return false;
    }
    p++;
    while (1) {
        p = json_ws(p);
        if (*p == ']') {
            free(json);
            return ref->n_pos > 0;
        }
        api_pos pos = {0};
        if (!api_parse_pos(&p, &pos)) {
            api_pos_free(&pos);
            api_ref_free(ref);
            free(json);
            return false;
        }
        api_ref_add_pos(ref, &pos);
        p = json_ws(p);
        if (*p == ',') p++;
    }
}

static int api_alt_token_id(ds4_engine *engine, const api_alt *alt) {
    if (!alt || !alt->bytes || alt->len <= 0) return -1;
    for (int i = 0; i < alt->len; i++) {
        if (alt->bytes[i] == 0) return -1;
    }
    char *text = malloc((size_t)alt->len + 1);
    if (!text) die("out of memory");
    memcpy(text, alt->bytes, (size_t)alt->len);
    text[alt->len] = '\0';

    ds4_tokens tv = {0};
    ds4_tokenize_text(engine, text, &tv);
    int id = tv.len == 1 ? tv.v[0] : -1;
    if (id >= 0) {
        size_t got_len = 0;
        char *got = ds4_token_text(engine, id, &got_len);
        if (got_len != (size_t)alt->len || memcmp(got, alt->bytes, (size_t)alt->len) != 0) {
            id = -1;
        }
        free(got);
    }
    ds4_tokens_free(&tv);
    free(text);
    return id;
}

static bool local_logits(ds4_session *session, float *logits, int n_vocab,
                         double *logsum, int *argmax) {
    if (ds4_session_copy_logits(session, logits, n_vocab) != n_vocab) return false;
    float max_logit = -INFINITY;
    int best = -1;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v) && (best < 0 || v > max_logit)) {
            max_logit = v;
            best = i;
        }
    }
    if (best < 0) return false;
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    *logsum = (double)max_logit + log(sum);
    *argmax = best;
    return true;
}

static double local_logprob(const float *logits, int n_vocab, int token, double logsum) {
    if (token < 0 || token >= n_vocab || !isfinite(logits[token])) return -INFINITY;
    return (double)logits[token] - logsum;
}

static int local_top_ids(const float *logits, int n_vocab, int *ids, int k) {
    if (k <= 0) return 0;
    float vals[64];
    if (k > (int)(sizeof(vals) / sizeof(vals[0]))) k = (int)(sizeof(vals) / sizeof(vals[0]));
    for (int i = 0; i < k; i++) {
        ids[i] = -1;
        vals[i] = -INFINITY;
    }
    for (int i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (ids[k - 1] >= 0 && v <= vals[k - 1]) continue;
        int j = k - 1;
        while (j > 0 && (ids[j - 1] < 0 || v > vals[j - 1])) {
            ids[j] = ids[j - 1];
            vals[j] = vals[j - 1];
            j--;
        }
        ids[j] = i;
        vals[j] = v;
    }
    int n = 0;
    while (n < k && ids[n] >= 0) n++;
    return n;
}

static bool id_in_top(const int *ids, int n, int id) {
    for (int i = 0; i < n; i++) {
        if (ids[i] == id) return true;
    }
    return false;
}

static void api_metrics_accum(api_metrics *dst, const api_metrics *src) {
    dst->target_count += src->target_count;
    dst->target_abs_delta += src->target_abs_delta;
    dst->target_signed_delta += src->target_signed_delta;
    dst->top_items += src->top_items;
    dst->top_mapped += src->top_mapped;
    dst->top_logprob_count += src->top_logprob_count;
    dst->top_abs_delta += src->top_abs_delta;
    dst->top_signed_delta += src->top_signed_delta;
    dst->top1_count += src->top1_count;
    dst->top1_match += src->top1_match;
    dst->topn_ref += src->topn_ref;
    dst->topn_hit += src->topn_hit;
    dst->pair_total += src->pair_total;
    dst->pair_agree += src->pair_agree;
}

static double safe_avg(double sum, long n) {
    return n ? sum / (double)n : 0.0;
}

static double safe_ratio(long num, long den) {
    return den ? (double)num / (double)den : 0.0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 2;
    }

    const char *model_path = argv[1];
    const char *manifest_path = argv[2];
    const char *out_path = argv[3];
    int ctx_size = 4096;
    bool ctx_set = false;
    bool ssd_streaming = false;
    bool ssd_streaming_cold = false;
    uint32_t ssd_streaming_cache_experts = 0;
    uint64_t ssd_streaming_cache_bytes = 0;
    uint32_t ssd_streaming_preload_experts = 0;

    for (int i = 4; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--ssd-streaming")) {
            ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg),
                    &ssd_streaming_cache_experts,
                    &ssd_streaming_cache_bytes)) {
                fprintf(stderr,
                        "score_official: --ssd-streaming-cache-experts must be "
                        "a positive count or <number>GB\n");
                return 2;
            }
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            ssd_streaming_preload_experts =
                (uint32_t)parse_positive_int(need_arg(&i, argc, argv, arg), arg);
        } else if (arg[0] != '-' && !ctx_set) {
            ctx_size = parse_positive_int(arg, "ctx");
            ctx_set = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (ctx_size < 1024) ctx_size = 1024;

    ds4_engine_options opt = {
        .model_path = model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .n_threads = 0,
        .ssd_streaming_cache_experts = ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = ssd_streaming_cache_bytes,
        .ssd_streaming_preload_experts = ssd_streaming_preload_experts,
        .warm_weights = false,
        .quality = false,
        .ssd_streaming = ssd_streaming,
        .ssd_streaming_cold = ssd_streaming_cold,
    };

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) die("failed to open model");

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx_size) != 0) die("failed to create session");

    const int n_vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)n_vocab * sizeof(logits[0]));
    if (!logits) die("out of memory");

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
    fprintf(out,
            "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp"
            "\tapi_ref_tokens\tapi_target_tokens\tapi_target_mae\tapi_target_mean_delta"
            "\tapi_top_items\tapi_top_mapped\tapi_top_coverage"
            "\tapi_top1_count\tapi_top1_match\tapi_top1_rate"
            "\tapi_topn_ref\tapi_topn_hit\tapi_topn_recall"
            "\tapi_top_logprob_count\tapi_top_mae\tapi_top_mean_delta"
            "\tapi_pair_total\tapi_pair_agree\tapi_pair_rate\n");

    char line[8192];
    int case_n = 0;
    double total_nll = 0.0;
    long total_tokens = 0;
    long total_lcp = 0;
    long first_matches = 0;
    long total_api_ref_tokens = 0;
    api_metrics total_api = {0};
    char err[256];

    while (fgets(line, sizeof(line), mf)) {
        strip_newline(line);
        if (!line[0] || line[0] == '#') continue;

        char *id = strtok(line, "\t");
        char *prompt_path = strtok(NULL, "\t");
        char *cont_path = strtok(NULL, "\t");
        char *resp_path = strtok(NULL, "\t");
        if (!id || !prompt_path || !cont_path) die("bad manifest row");

        char *prompt_text = read_file(prompt_path);
        char *cont_text = read_file(cont_path);
        api_ref ref = {0};
        bool have_api = false;
        bool api_aligned = false;
        api_metrics cm = {0};

        ds4_tokens prompt = {0};
        ds4_tokens target = {0};
        ds4_encode_chat_prompt(engine, NULL, prompt_text, DS4_THINK_NONE, &prompt);
        ds4_tokenize_text(engine, cont_text, &target);

        if (prompt.len + target.len + 1 >= ctx_size) {
            fprintf(stderr, "%s exceeds ctx=%d\n", id, ctx_size);
            return 1;
        }
        if (resp_path && resp_path[0]) {
            have_api = api_ref_load(resp_path, &ref);
            if (!have_api) {
                fprintf(stderr, "%s warning: no API logprobs parsed from %s\n", id, resp_path);
            } else if (ref.n_pos != target.len) {
                fprintf(stderr,
                        "%s warning: API token count %d != local target tokens %d; "
                        "API logprob agreement skipped\n",
                        id, ref.n_pos, target.len);
            } else {
                api_aligned = true;
            }
        }
        total_api_ref_tokens += have_api ? ref.n_pos : 0;

        if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "%s sync failed: %s\n", id, err);
            return 1;
        }

        double nll = 0.0;
        int lcp = 0;
        bool still_matching = true;
        bool first_match = false;
        for (int i = 0; i < target.len; i++) {
            double logsum = 0.0;
            int greedy = -1;
            if (!local_logits(session, logits, n_vocab, &logsum, &greedy)) {
                fprintf(stderr, "%s logits failed at target token %d\n", id, i);
                return 1;
            }

            if (i == 0) first_match = (greedy == target.v[i]);
            if (still_matching && greedy == target.v[i]) lcp++;
            else still_matching = false;

            const double target_lp = local_logprob(logits, n_vocab, target.v[i], logsum);
            if (!isfinite(target_lp)) {
                fprintf(stderr, "%s logprob failed at target token %d\n", id, i);
                return 1;
            }
            nll += -target_lp;

            if (api_aligned) {
                const api_pos *ap = &ref.pos[i];
                if (isfinite(ap->logprob)) {
                    const double delta = target_lp - ap->logprob;
                    cm.target_count++;
                    cm.target_abs_delta += fabs(delta);
                    cm.target_signed_delta += delta;
                }
                if (ap->n_alts > 0) {
                    int top_ids[64];
                    int top_k = ap->n_alts;
                    if (top_k > (int)(sizeof(top_ids) / sizeof(top_ids[0]))) {
                        top_k = (int)(sizeof(top_ids) / sizeof(top_ids[0]));
                    }
                    const int top_n = local_top_ids(logits, n_vocab, top_ids, top_k);
                    int mapped_ids[64];
                    double api_lp[64];
                    double local_lp[64];
                    int mapped_n = 0;

                    cm.top_items += ap->n_alts;
                    for (int j = 0; j < ap->n_alts; j++) {
                        const int tok = api_alt_token_id(engine, &ap->alts[j]);
                        if (tok < 0) continue;
                        const double lp = local_logprob(logits, n_vocab, tok, logsum);
                        if (!isfinite(lp)) continue;
                        if (j == 0) {
                            cm.top1_count++;
                            if (tok == greedy) cm.top1_match++;
                        }
                        cm.top_mapped++;
                        cm.top_logprob_count++;
                        const double delta = lp - ap->alts[j].logprob;
                        cm.top_abs_delta += fabs(delta);
                        cm.top_signed_delta += delta;
                        cm.topn_ref++;
                        if (id_in_top(top_ids, top_n, tok)) cm.topn_hit++;

                        if (mapped_n < (int)(sizeof(mapped_ids) / sizeof(mapped_ids[0]))) {
                            mapped_ids[mapped_n] = tok;
                            api_lp[mapped_n] = ap->alts[j].logprob;
                            local_lp[mapped_n] = lp;
                            mapped_n++;
                        }
                    }

                    for (int a = 0; a < mapped_n; a++) {
                        for (int b = a + 1; b < mapped_n; b++) {
                            if (mapped_ids[a] == mapped_ids[b]) continue;
                            const double api_diff = api_lp[a] - api_lp[b];
                            const double local_diff = local_lp[a] - local_lp[b];
                            if (fabs(api_diff) < 1e-12 || fabs(local_diff) < 1e-12) continue;
                            cm.pair_total++;
                            if ((api_diff > 0.0) == (local_diff > 0.0)) cm.pair_agree++;
                        }
                    }
                }
            }

            if (ds4_session_eval(session, target.v[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "%s eval failed at target token %d: %s\n", id, i, err);
                return 1;
            }
        }

        const double avg = target.len ? nll / (double)target.len : 0.0;
        fprintf(out,
                "%s\t%d\t%d\t%.9f\t%.9f\t%d\t%d"
                "\t%d\t%ld\t%.9f\t%.9f"
                "\t%ld\t%ld\t%.9f"
                "\t%ld\t%ld\t%.9f"
                "\t%ld\t%ld\t%.9f"
                "\t%ld\t%.9f\t%.9f"
                "\t%ld\t%ld\t%.9f\n",
                id, prompt.len, target.len, nll, avg, first_match ? 1 : 0, lcp,
                have_api ? ref.n_pos : 0,
                cm.target_count,
                safe_avg(cm.target_abs_delta, cm.target_count),
                safe_avg(cm.target_signed_delta, cm.target_count),
                cm.top_items,
                cm.top_mapped,
                safe_ratio(cm.top_mapped, cm.top_items),
                cm.top1_count,
                cm.top1_match,
                safe_ratio(cm.top1_match, cm.top1_count),
                cm.topn_ref,
                cm.topn_hit,
                safe_ratio(cm.topn_hit, cm.topn_ref),
                cm.top_logprob_count,
                safe_avg(cm.top_abs_delta, cm.top_logprob_count),
                safe_avg(cm.top_signed_delta, cm.top_logprob_count),
                cm.pair_total,
                cm.pair_agree,
                safe_ratio(cm.pair_agree, cm.pair_total));
        fflush(out);

        case_n++;
        total_nll += nll;
        total_tokens += target.len;
        total_lcp += lcp;
        first_matches += first_match ? 1 : 0;
        api_metrics_accum(&total_api, &cm);
        fprintf(stderr,
                "%s cases=%d prompt=%d target=%d avg_nll=%.6f lcp=%d "
                "api_target_mae=%.6f api_top_coverage=%ld/%ld "
                "api_top1=%.3f api_topn=%.3f api_pair=%.3f\n",
                id, case_n, prompt.len, target.len, avg, lcp,
                safe_avg(cm.target_abs_delta, cm.target_count),
                cm.top_mapped,
                cm.top_items,
                safe_ratio(cm.top1_match, cm.top1_count),
                safe_ratio(cm.topn_hit, cm.topn_ref),
                safe_ratio(cm.pair_agree, cm.pair_total));

        api_ref_free(&ref);
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
    fprintf(stderr,
            "api_summary ref_tokens=%ld target_tokens=%ld target_mae=%.9f "
            "target_mean_delta=%.9f top_items=%ld top_mapped=%ld "
            "top_coverage=%.9f top1_match=%ld/%ld top1_rate=%.9f "
            "topn_hit=%ld/%ld topn_recall=%.9f top_logprob_count=%ld "
            "top_mae=%.9f top_mean_delta=%.9f pair_agree=%ld/%ld pair_rate=%.9f\n",
            total_api_ref_tokens,
            total_api.target_count,
            safe_avg(total_api.target_abs_delta, total_api.target_count),
            safe_avg(total_api.target_signed_delta, total_api.target_count),
            total_api.top_items,
            total_api.top_mapped,
            safe_ratio(total_api.top_mapped, total_api.top_items),
            total_api.top1_match,
            total_api.top1_count,
            safe_ratio(total_api.top1_match, total_api.top1_count),
            total_api.topn_hit,
            total_api.topn_ref,
            safe_ratio(total_api.topn_hit, total_api.topn_ref),
            total_api.top_logprob_count,
            safe_avg(total_api.top_abs_delta, total_api.top_logprob_count),
            safe_avg(total_api.top_signed_delta, total_api.top_logprob_count),
            total_api.pair_agree,
            total_api.pair_total,
            safe_ratio(total_api.pair_agree, total_api.pair_total));

    fclose(out);
    fclose(mf);
    free(logits);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
