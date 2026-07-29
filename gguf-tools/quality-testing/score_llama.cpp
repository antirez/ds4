#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void die(const char *msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static std::string read_file(const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "open %s: %s\n", path, std::strerror(errno));
        std::exit(1);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void strip_newline(std::string &s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
}

static std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
}

static std::vector<llama_token> tokenize(
        const llama_vocab *vocab,
        const std::string &text,
        bool add_special,
        bool parse_special) {
    int n = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                           nullptr, 0, add_special, parse_special);
    if (n < 0) n = -n;
    if (n == 0) return {};

    std::vector<llama_token> tokens((size_t)n);
    int got = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                             tokens.data(), n, add_special, parse_special);
    if (got < 0) die("llama_tokenize failed");
    tokens.resize((size_t)got);
    return tokens;
}

static std::string render_glm_ds4_prompt(const std::string &prompt) {
    return std::string("[gMASK]<sop><|user|>") + prompt +
           "<|assistant|><think></think>";
}

static std::string render_template_prompt(
        const char *tmpl,
        const std::string &prompt,
        bool *ok) {
    llama_chat_message msg = {"user", prompt.c_str()};
    int n = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (n < 0) {
        *ok = false;
        return {};
    }
    std::vector<char> buf((size_t)n + 1);
    int got = llama_chat_apply_template(tmpl, &msg, 1, true,
                                        buf.data(), (int32_t)buf.size());
    if (got < 0) {
        *ok = false;
        return {};
    }
    *ok = true;
    return std::string(buf.data(), (size_t)got);
}

static bool decode_chunk(
        llama_context *ctx,
        llama_batch &batch,
        const llama_token *tokens,
        int n_tokens,
        int pos,
        bool logits_last) {
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (logits_last && i == n_tokens - 1) ? 1 : 0;
    }
    return llama_decode(ctx, batch) == 0;
}

static bool decode_tokens(
        llama_context *ctx,
        llama_batch &batch,
        const std::vector<llama_token> &tokens,
        int start_pos,
        int n_batch,
        bool logits_last) {
    int off = 0;
    while (off < (int)tokens.size()) {
        int n = std::min(n_batch, (int)tokens.size() - off);
        bool want_logits = logits_last && off + n == (int)tokens.size();
        if (!decode_chunk(ctx, batch, tokens.data() + off, n,
                          start_pos + off, want_logits)) {
            return false;
        }
        off += n;
    }
    return true;
}

static double token_logprob(
        const float *logits,
        int n_vocab,
        llama_token token,
        llama_token *greedy_out) {
    float max_logit = -std::numeric_limits<float>::infinity();
    llama_token greedy = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            greedy = (llama_token)i;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum += std::exp((double)logits[i] - (double)max_logit);
    }
    *greedy_out = greedy;
    return (double)logits[token] - ((double)max_logit + std::log(sum));
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5 && argc != 6) {
        std::fprintf(stderr,
                     "usage: %s MODEL manifest.tsv OUT.tsv [ctx] [auto|glm-ds4]\n",
                     argv[0]);
        return 2;
    }

    const char *model_path = argv[1];
    const char *manifest_path = argv[2];
    const char *out_path = argv[3];
    int ctx_size = argc >= 5 ? std::atoi(argv[4]) : 4096;
    if (ctx_size < 1024) ctx_size = 1024;
    const std::string template_mode = argc == 6 ? argv[5] : "auto";
    if (template_mode != "auto" && template_mode != "glm-ds4") {
        die("template mode must be auto or glm-ds4");
    }

    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.use_mmap = true;

    llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) die("failed to open model");
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const char *tmpl = llama_model_chat_template(model, nullptr);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t)ctx_size;
    ctx_params.n_batch = 2048;
    ctx_params.n_ubatch = 512;
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) die("failed to create context");
    const int n_batch = std::min<int>((int)llama_n_batch(ctx), 2048);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    std::ifstream mf(manifest_path, std::ios::binary);
    if (!mf) {
        std::fprintf(stderr, "open %s: %s\n", manifest_path, std::strerror(errno));
        return 1;
    }
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "open %s: %s\n", out_path, std::strerror(errno));
        return 1;
    }
    out << "id\tprompt_tokens\ttarget_tokens\tnll\tavg_nll\tfirst_match\tgreedy_lcp\n";

    std::string line;
    int case_n = 0;
    double total_nll = 0.0;
    long total_tokens = 0;
    long total_lcp = 0;
    long first_matches = 0;
    bool warned_template_fallback = false;

    while (std::getline(mf, line)) {
        strip_newline(line);
        if (line.empty() || line[0] == '#') continue;

        std::vector<std::string> cols = split_tab(line);
        if (cols.size() < 3) die("bad manifest row");
        const std::string &id = cols[0];
        const std::string &prompt_path = cols[1];
        const std::string &cont_path = cols[2];

        std::string prompt_text = read_file(prompt_path.c_str());
        std::string cont_text = read_file(cont_path.c_str());

        std::string rendered;
        bool used_template = false;
        if (template_mode == "auto" && tmpl) {
            rendered = render_template_prompt(tmpl, prompt_text, &used_template);
        }
        if (!used_template) {
            if (template_mode == "auto" && !warned_template_fallback) {
                std::fprintf(stderr,
                             "score_llama: llama.cpp chat template unavailable; "
                             "using DS4 GLM prompt fallback\n");
                warned_template_fallback = true;
            }
            rendered = render_glm_ds4_prompt(prompt_text);
        }

        std::vector<llama_token> prompt =
            tokenize(vocab, rendered, false, true);
        std::vector<llama_token> target =
            tokenize(vocab, cont_text, false, false);

        if (prompt.empty()) die("empty prompt tokenization");
        if (target.empty()) die("empty continuation tokenization");
        if ((int)prompt.size() + (int)target.size() + 1 >= ctx_size) {
            std::fprintf(stderr, "%s exceeds ctx=%d\n", id.c_str(), ctx_size);
            return 1;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        if (!decode_tokens(ctx, batch, prompt, 0, n_batch, true)) {
            std::fprintf(stderr, "%s prompt decode failed\n", id.c_str());
            return 1;
        }

        double nll = 0.0;
        int lcp = 0;
        bool still_matching = true;
        bool first_match = false;

        for (int i = 0; i < (int)target.size(); i++) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) {
                std::fprintf(stderr, "%s logits unavailable at target token %d\n",
                             id.c_str(), i);
                return 1;
            }

            llama_token greedy = 0;
            double lp = token_logprob(logits, n_vocab, target[(size_t)i], &greedy);
            if (i == 0) first_match = (greedy == target[(size_t)i]);
            if (still_matching && greedy == target[(size_t)i]) lcp++;
            else still_matching = false;
            nll += -lp;

            if (!decode_chunk(ctx, batch, &target[(size_t)i], 1,
                              (int)prompt.size() + i, true)) {
                std::fprintf(stderr, "%s target decode failed at token %d\n",
                             id.c_str(), i);
                return 1;
            }
        }

        const double avg = nll / (double)target.size();
        out << id << '\t'
            << prompt.size() << '\t'
            << target.size() << '\t'
            << nll << '\t'
            << avg << '\t'
            << (first_match ? 1 : 0) << '\t'
            << lcp << '\n';
        out.flush();

        case_n++;
        total_nll += nll;
        total_tokens += (long)target.size();
        total_lcp += lcp;
        first_matches += first_match ? 1 : 0;
        std::fprintf(stderr,
                     "%s cases=%d prompt=%zu target=%zu avg_nll=%.6f lcp=%d\n",
                     id.c_str(), case_n, prompt.size(), target.size(), avg, lcp);
    }

    std::fprintf(stderr,
                 "summary cases=%d tokens=%ld avg_nll=%.9f first_match=%ld avg_lcp=%.3f\n",
                 case_n,
                 total_tokens,
                 total_tokens ? total_nll / (double)total_tokens : 0.0,
                 first_matches,
                 case_n ? (double)total_lcp / (double)case_n : 0.0);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
