#define _POSIX_C_SOURCE 200809L

/* ds4-launch.
 *
 * Small integration launcher for local agent tools. It starts ds4-server in the
 * background when needed, reuses an already-running ds4 process when the engine
 * lock says one exists, configures the selected client for the local API, then
 * execs the client so the terminal belongs to the tool. */

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define DS4_LAUNCH_MODEL_NAME "DeepSeek V4 Flash (ds4.c local)"
#define DS4_LAUNCH_API_KEY "dsv4-local"
#define DS4_LAUNCH_DEFAULT_HOST "127.0.0.1"
#define DS4_LAUNCH_DEFAULT_PORT 8000
#define DS4_LAUNCH_DEFAULT_CTX 32768
#define DS4_LAUNCH_DEFAULT_TOKENS 393216
#define DS4_LAUNCH_STARTUP_TIMEOUT_MS 600000

typedef enum {
    TOOL_PI,
    TOOL_COPILOT,
    TOOL_OPENCODE,
} launch_tool;

typedef struct {
    const char *host;
    int port;
    int ctx;
    int tokens;
} server_cli;

typedef struct {
    char *base_url;
    char *base_url_v1;
    int ctx;
    int tokens;
} client_config;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4-launch: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *xasprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    char *s = xmalloc((size_t)n + 1);
    vsnprintf(s, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    return s;
}

static bool str_eq_ci(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static bool str_has_slash(const char *s) {
    return strchr(s, '/') != NULL;
}

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        die("invalid value for %s: %s", opt, s);
    }
    return (int)v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) die("missing value for %s", opt);
    return argv[++(*i)];
}

static void sleep_ms(int ms) {
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {}
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: ds4-launch TOOL [ds4-server options]\n"
        "\n"
        "Tools:\n"
        "  pi\n"
        "      Configure and launch Pi.\n"
        "  copilot\n"
        "      Launch GitHub Copilot CLI through the Responses API.\n"
        "  opencode | open_code | open-code\n"
        "      Launch OpenCode with inline ds4 provider config.\n"
        "\n"
        "Server options after TOOL are passed to ds4-server when a new server\n"
        "is needed. If another ds4 process already holds the engine lock,\n"
        "ds4-launch discovers its listening port and uses that server instead.\n"
        "\n"
        "Examples:\n"
        "  ds4-launch pi --ctx 100000 --kv-disk-dir /tmp/ds4-kv\n"
        "  ds4-launch copilot --port 9000 --trace /tmp/ds4-trace.txt\n"
        "  ds4-launch opencode\n"
        "\n"
        "The client model is always presented as: %s\n",
        DS4_LAUNCH_MODEL_NAME);
}

static bool canonical_tool(int argc, char **argv, int *tool_argc, launch_tool *tool) {
    if (argc < 2) return false;
    if (str_eq_ci(argv[1], "pi")) {
        *tool = TOOL_PI;
        *tool_argc = 2;
        return true;
    }
    if (str_eq_ci(argv[1], "copilot")) {
        *tool = TOOL_COPILOT;
        *tool_argc = 2;
        return true;
    }
    if (str_eq_ci(argv[1], "opencode") || str_eq_ci(argv[1], "open-code") ||
        str_eq_ci(argv[1], "open_code"))
    {
        *tool = TOOL_OPENCODE;
        *tool_argc = 2;
        return true;
    }
    if (argc >= 3 && str_eq_ci(argv[1], "open") && str_eq_ci(argv[2], "code")) {
        *tool = TOOL_OPENCODE;
        *tool_argc = 3;
        return true;
    }
    return false;
}

static server_cli parse_server_cli(int argc, char **argv, int start) {
    server_cli cli = {
        .host = DS4_LAUNCH_DEFAULT_HOST,
        .port = DS4_LAUNCH_DEFAULT_PORT,
        .ctx = DS4_LAUNCH_DEFAULT_CTX,
        .tokens = DS4_LAUNCH_DEFAULT_TOKENS,
    };

    for (int i = start; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--host")) {
            cli.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            cli.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            cli.ctx = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            cli.tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model") ||
                   !strcmp(arg, "--mtp") || !strcmp(arg, "--mtp-draft") ||
                   !strcmp(arg, "--mtp-margin") || !strcmp(arg, "-t") ||
                   !strcmp(arg, "--threads") || !strcmp(arg, "--trace") ||
                   !strcmp(arg, "--kv-disk-dir") || !strcmp(arg, "--kv-disk-space-mb") ||
                   !strcmp(arg, "--kv-cache-min-tokens") ||
                   !strcmp(arg, "--kv-cache-cold-max-tokens") ||
                   !strcmp(arg, "--kv-cache-continued-interval-tokens") ||
                   !strcmp(arg, "--kv-cache-boundary-trim-tokens") ||
                   !strcmp(arg, "--kv-cache-boundary-align-tokens") ||
                   !strcmp(arg, "--tool-memory-max-ids") ||
                   !strcmp(arg, "--dir-steering-file") ||
                   !strcmp(arg, "--dir-steering-ffn") ||
                   !strcmp(arg, "--dir-steering-attn") ||
                   !strcmp(arg, "--backend"))
        {
            (void)need_arg(&i, argc, argv, arg);
        }
    }
    return cli;
}

static const char *client_host(const char *host) {
    if (!host || !host[0]) return DS4_LAUNCH_DEFAULT_HOST;
    if (!strcmp(host, "0.0.0.0") || !strcmp(host, "*") || !strcmp(host, "localhost")) {
        return DS4_LAUNCH_DEFAULT_HOST;
    }
    return host;
}

static char *path_dirname(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) return xstrdup(".");
    if (slash == path) return xstrdup("/");
    return xstrndup(path, (size_t)(slash - path));
}

static char *path_join(const char *a, const char *b) {
    size_t na = strlen(a);
    size_t nb = strlen(b);
    bool slash = na && a[na - 1] == '/';
    char *out = xmalloc(na + (slash ? 0 : 1) + nb + 1);
    memcpy(out, a, na);
    size_t pos = na;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, nb + 1);
    return out;
}

static char *home_join(const char *suffix) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) return NULL;
    return path_join(home, suffix);
}

static bool executable_ok(const char *path) {
    return path && access(path, X_OK) == 0;
}

static char *find_on_path(const char *name) {
    if (str_has_slash(name)) return executable_ok(name) ? xstrdup(name) : NULL;
    const char *path = getenv("PATH");
    if (!path) return NULL;
    const char *p = path;
    while (*p) {
        const char *colon = strchr(p, ':');
        size_t n = colon ? (size_t)(colon - p) : strlen(p);
        char *dir = n ? xstrndup(p, n) : xstrdup(".");
        char *candidate = path_join(dir, name);
        free(dir);
        if (executable_ok(candidate)) return candidate;
        free(candidate);
        if (!colon) break;
        p = colon + 1;
    }
    return NULL;
}

static char *find_ds4_server(const char *argv0) {
    if (argv0 && str_has_slash(argv0)) {
        char *dir = path_dirname(argv0);
        char *candidate = path_join(dir, "ds4-server");
        free(dir);
        if (executable_ok(candidate)) return candidate;
        free(candidate);
    }
    if (executable_ok("./ds4-server")) return xstrdup("./ds4-server");
    char *path = find_on_path("ds4-server");
    if (path) return path;
    die("ds4-server not found; build it first with make ds4-server");
    return NULL;
}

static char *find_tool_binary(launch_tool tool) {
    char *path = NULL;
    switch (tool) {
    case TOOL_PI:
        path = find_on_path("pi");
        if (!path) die("pi is not installed or not on PATH");
        return path;
    case TOOL_COPILOT:
        path = find_on_path("copilot");
        if (!path) path = home_join(".local/bin/copilot");
        if (!executable_ok(path)) die("copilot is not installed; install GitHub Copilot CLI first");
        return path;
    case TOOL_OPENCODE:
        path = find_on_path("opencode");
        if (!path) path = home_join(".opencode/bin/opencode");
        if (!executable_ok(path)) die("opencode is not installed; install it from https://opencode.ai");
        return path;
    }
    die("unknown tool");
    return NULL;
}

static bool mkdir_p(const char *path) {
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        free(tmp);
        return false;
    }
    free(tmp);
    return true;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) {
        free(s);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    s[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return s;
}

static bool write_file_if_changed(const char *path, const char *content) {
    size_t old_len = 0;
    char *old = read_file(path, &old_len);
    size_t new_len = strlen(content);
    if (old && old_len == new_len && memcmp(old, content, new_len) == 0) {
        free(old);
        return true;
    }
    if (old) {
        char *bak = xasprintf("%s.ds4-launch.bak", path);
        FILE *bfp = fopen(bak, "wb");
        if (bfp) {
            (void)fwrite(old, 1, old_len, bfp);
            fclose(bfp);
        }
        free(bak);
        free(old);
    }

    char *dir = path_dirname(path);
    bool ok = mkdir_p(dir);
    free(dir);
    if (!ok) return false;

    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    bool wrote = fwrite(content, 1, new_len, fp) == new_len;
    bool closed = fclose(fp) == 0;
    return wrote && closed;
}

static char *json_escape(const char *s) {
    size_t cap = strlen(s) + 16;
    char *out = xmalloc(cap);
    size_t len = 0;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        const char *esc = NULL;
        char hex[7];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        default:
            if (c < 0x20) {
                snprintf(hex, sizeof(hex), "\\u%04x", c);
                esc = hex;
            }
            break;
        }
        if (esc) {
            size_t n = strlen(esc);
            if (len + n + 1 > cap) {
                cap = (len + n + 1) * 2;
                out = xrealloc(out, cap);
            }
            memcpy(out + len, esc, n);
            len += n;
        } else {
            if (len + 2 > cap) {
                cap *= 2;
                out = xrealloc(out, cap);
            }
            out[len++] = (char)c;
        }
    }
    out[len] = '\0';
    return out;
}

static size_t skip_ws_pos(const char *s, size_t i) {
    while (s[i] && isspace((unsigned char)s[i])) i++;
    return i;
}

static bool skip_json_string(const char *s, size_t *i) {
    if (s[*i] != '"') return false;
    (*i)++;
    while (s[*i]) {
        unsigned char c = (unsigned char)s[(*i)++];
        if (c == '"') return true;
        if (c == '\\' && s[*i]) (*i)++;
    }
    return false;
}

static bool skip_json_value(const char *s, size_t *i) {
    *i = skip_ws_pos(s, *i);
    if (s[*i] == '"') return skip_json_string(s, i);
    if (s[*i] == '{' || s[*i] == '[') {
        char open = s[*i];
        char close = open == '{' ? '}' : ']';
        int depth = 1;
        (*i)++;
        while (s[*i]) {
            if (s[*i] == '"') {
                if (!skip_json_string(s, i)) return false;
                continue;
            }
            if (s[*i] == open) depth++;
            if (s[*i] == close) {
                depth--;
                (*i)++;
                if (depth == 0) return true;
                continue;
            }
            (*i)++;
        }
        return false;
    }
    while (s[*i] && !strchr(",}]\r\n\t ", s[*i])) (*i)++;
    return true;
}

static size_t json_object_close_pos(const char *obj) {
    size_t i = skip_ws_pos(obj, 0);
    if (obj[i] != '{') return SIZE_MAX;
    size_t start = i;
    if (!skip_json_value(obj, &i)) return SIZE_MAX;
    if (obj[start] != '{') return SIZE_MAX;
    return i ? i - 1 : SIZE_MAX;
}

static bool json_find_member(const char *obj, const char *key,
                             size_t *key_start_out, size_t *value_start_out,
                             size_t *value_end_out, size_t *close_out) {
    size_t close_pos = json_object_close_pos(obj);
    if (close_pos == SIZE_MAX) return false;
    if (close_out) *close_out = close_pos;

    size_t i = skip_ws_pos(obj, 0);
    if (obj[i] != '{') return false;
    i++;
    while (true) {
        i = skip_ws_pos(obj, i);
        if (i >= close_pos || obj[i] == '}') return false;
        if (obj[i] != '"') return false;
        size_t key_start = i;
        i++;
        size_t raw_start = i;
        while (obj[i] && obj[i] != '"') {
            if (obj[i] == '\\' && obj[i + 1]) i += 2;
            else i++;
        }
        if (obj[i] != '"') return false;
        size_t raw_end = i;
        i++;
        i = skip_ws_pos(obj, i);
        if (obj[i] != ':') return false;
        i++;
        size_t value_start = skip_ws_pos(obj, i);
        size_t value_end = value_start;
        if (!skip_json_value(obj, &value_end)) return false;
        if (strlen(key) == raw_end - raw_start &&
            memcmp(obj + raw_start, key, raw_end - raw_start) == 0)
        {
            if (key_start_out) *key_start_out = key_start;
            if (value_start_out) *value_start_out = value_start;
            if (value_end_out) *value_end_out = value_end;
            return true;
        }
        i = skip_ws_pos(obj, value_end);
        if (obj[i] == ',') i++;
    }
}

static char *json_object_set_raw(const char *obj, const char *key, const char *raw_value) {
    size_t key_start = 0, value_start = 0, value_end = 0, close_pos = 0;
    if (json_find_member(obj, key, &key_start, &value_start, &value_end, &close_pos)) {
        size_t prefix = value_start;
        size_t suffix = strlen(obj + value_end);
        size_t raw_len = strlen(raw_value);
        char *out = xmalloc(prefix + raw_len + suffix + 1);
        memcpy(out, obj, prefix);
        memcpy(out + prefix, raw_value, raw_len);
        memcpy(out + prefix + raw_len, obj + value_end, suffix + 1);
        return out;
    }

    close_pos = json_object_close_pos(obj);
    if (close_pos == SIZE_MAX) {
        char *escaped = json_escape(key);
        char *out = xasprintf("{\n  \"%s\": %s\n}\n", escaped, raw_value);
        free(escaped);
        return out;
    }
    bool has_member = false;
    for (size_t i = 1; i < close_pos; i++) {
        if (!isspace((unsigned char)obj[i])) {
            has_member = true;
            break;
        }
    }
    char *escaped = json_escape(key);
    const char *comma = has_member ? ",\n" : "\n";
    char *insert = xasprintf("%s  \"%s\": %s\n", comma, escaped, raw_value);
    free(escaped);
    size_t prefix = close_pos;
    size_t insert_len = strlen(insert);
    size_t suffix = strlen(obj + close_pos);
    char *out = xmalloc(prefix + insert_len + suffix + 1);
    memcpy(out, obj, prefix);
    memcpy(out + prefix, insert, insert_len);
    memcpy(out + prefix + insert_len, obj + close_pos, suffix + 1);
    free(insert);
    return out;
}

static char *json_object_set_string(const char *obj, const char *key, const char *value) {
    char *escaped = json_escape(value);
    char *raw = xasprintf("\"%s\"", escaped);
    char *out = json_object_set_raw(obj, key, raw);
    free(raw);
    free(escaped);
    return out;
}

static char *build_pi_provider_json(const client_config *cfg) {
    char *base = json_escape(cfg->base_url_v1);
    char *name = json_escape(DS4_LAUNCH_MODEL_NAME);
    char *out = xasprintf(
        "{\n"
        "      \"name\": \"ds4.c local\",\n"
        "      \"baseUrl\": \"%s\",\n"
        "      \"api\": \"openai-completions\",\n"
        "      \"apiKey\": \"%s\",\n"
        "      \"compat\": {\n"
        "        \"supportsStore\": false,\n"
        "        \"supportsDeveloperRole\": false,\n"
        "        \"supportsReasoningEffort\": true,\n"
        "        \"supportsUsageInStreaming\": true,\n"
        "        \"maxTokensField\": \"max_tokens\",\n"
        "        \"supportsStrictMode\": false,\n"
        "        \"thinkingFormat\": \"deepseek\",\n"
        "        \"requiresReasoningContentOnAssistantMessages\": true\n"
        "      },\n"
        "      \"models\": [\n"
        "        {\n"
        "          \"id\": \"%s\",\n"
        "          \"name\": \"%s\",\n"
        "          \"reasoning\": true,\n"
        "          \"thinkingLevelMap\": {\n"
        "            \"off\": null,\n"
        "            \"minimal\": \"low\",\n"
        "            \"low\": \"low\",\n"
        "            \"medium\": \"medium\",\n"
        "            \"high\": \"high\",\n"
        "            \"xhigh\": \"xhigh\"\n"
        "          },\n"
        "          \"input\": [\"text\"],\n"
        "          \"contextWindow\": %d,\n"
        "          \"maxTokens\": %d,\n"
        "          \"cost\": {\"input\": 0, \"output\": 0, \"cacheRead\": 0, \"cacheWrite\": 0}\n"
        "        }\n"
        "      ]\n"
        "    }",
        base, DS4_LAUNCH_API_KEY, name, name,
        cfg->ctx, cfg->tokens);
    free(name);
    free(base);
    return out;
}

static void configure_pi(const client_config *cfg) {
    char *models_path = home_join(".pi/agent/models.json");
    char *settings_path = home_join(".pi/agent/settings.json");
    if (!models_path || !settings_path) die("HOME is not set; cannot configure Pi");

    char *provider = build_pi_provider_json(cfg);
    size_t len = 0;
    char *models = read_file(models_path, &len);
    if (!models) models = xstrdup("{\n  \"providers\": {}\n}\n");

    size_t providers_value_start = 0, providers_value_end = 0;
    char *new_models = NULL;
    if (json_find_member(models, "providers", NULL, &providers_value_start, &providers_value_end, NULL) &&
        models[skip_ws_pos(models, providers_value_start)] == '{')
    {
        char *providers = xstrndup(models + providers_value_start,
                                   providers_value_end - providers_value_start);
        char *new_providers = json_object_set_raw(providers, "ds4", provider);
        new_models = json_object_set_raw(models, "providers", new_providers);
        free(new_providers);
        free(providers);
    } else {
        char *providers = xasprintf("{\n    \"ds4\": %s\n  }", provider);
        new_models = json_object_set_raw(models, "providers", providers);
        free(providers);
    }
    free(provider);
    free(models);

    if (!write_file_if_changed(models_path, new_models)) {
        die("failed to write %s: %s", models_path, strerror(errno));
    }
    free(new_models);

    char *settings = read_file(settings_path, NULL);
    if (!settings) settings = xstrdup("{}\n");
    char *s1 = json_object_set_string(settings, "defaultProvider", "ds4");
    char *s2 = json_object_set_string(s1, "defaultModel", DS4_LAUNCH_MODEL_NAME);
    if (!write_file_if_changed(settings_path, s2)) {
        die("failed to write %s: %s", settings_path, strerror(errno));
    }
    free(s2);
    free(s1);
    free(settings);
    free(models_path);
    free(settings_path);
}

static char *build_opencode_config_json(const client_config *cfg) {
    char *base = json_escape(cfg->base_url_v1);
    char *name = json_escape(DS4_LAUNCH_MODEL_NAME);
    char *out = xasprintf(
        "{"
        "\"$schema\":\"https://opencode.ai/config.json\"," 
        "\"provider\":{"
        "\"ds4\":{"
        "\"name\":\"ds4.c local\"," 
        "\"npm\":\"@ai-sdk/openai-compatible\"," 
        "\"options\":{\"baseURL\":\"%s\",\"apiKey\":\"%s\"},"
        "\"models\":{\"%s\":{\"name\":\"%s\",\"limit\":{\"context\":%d,\"output\":%d}}}"
        "}"
        "},"
        "\"model\":\"ds4/%s\"," 
        "\"agent\":{\"ds4\":{\"description\":\"DeepSeek V4 Flash served by local ds4-server\",\"model\":\"ds4/%s\",\"temperature\":0}}"
        "}",
        base, DS4_LAUNCH_API_KEY, name, name,
        cfg->ctx, cfg->tokens, name, name);
    free(name);
    free(base);
    return out;
}

static long read_lock_owner(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char buf[64];
    if (!fgets(buf, sizeof(buf), fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    char *end = NULL;
    long pid = strtol(buf, &end, 10);
    return pid > 0 ? pid : -1;
}

static bool pid_exists(long pid) {
    if (pid <= 0) return false;
    if (kill((pid_t)pid, 0) == 0) return true;
    return errno == EPERM;
}

static int parse_lsof_port_line(const char *line) {
    const char *tcp = strstr(line, "TCP ");
    const char *listen = strstr(line, "(LISTEN)");
    if (!tcp || !listen || listen <= tcp) return 0;
    const char *end = listen;
    while (end > tcp && !isdigit((unsigned char)end[-1])) end--;
    const char *start = end;
    while (start > tcp && isdigit((unsigned char)start[-1])) start--;
    if (start == end) return 0;
    char buf[16];
    size_t n = (size_t)(end - start);
    if (n >= sizeof(buf)) return 0;
    memcpy(buf, start, n);
    buf[n] = '\0';
    return parse_int_arg(buf, "lsof port");
}

static int lsof_port_for_pid(long pid) {
    char *cmd = xasprintf("lsof -Pan -p %ld -iTCP -sTCP:LISTEN 2>/dev/null", pid);
    FILE *fp = popen(cmd, "r");
    free(cmd);
    if (!fp) return 0;
    char line[1024];
    int port = 0;
    while (fgets(line, sizeof(line), fp)) {
        port = parse_lsof_port_line(line);
        if (port > 0) break;
    }
    pclose(fp);
    return port;
}

static long parse_lock_pid_from_text(const char *text) {
    const char *needle = "another ds4 process is already running (pid ";
    const char *p = strstr(text, needle);
    if (!p) return -1;
    p += strlen(needle);
    char *end = NULL;
    long pid = strtol(p, &end, 10);
    return pid > 0 ? pid : -1;
}

static bool tcp_connect_ok(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    const char *h = client_host(host);
    if (inet_pton(AF_INET, h, &sa.sin_addr) != 1) {
        close(fd);
        return false;
    }
    int rc = connect(fd, (struct sockaddr *)&sa, sizeof(sa));
    close(fd);
    return rc == 0;
}

static char **build_server_argv(const char *server_path, int argc, char **argv, int start) {
    int n = argc - start;
    char **out = xmalloc((size_t)(n + 2) * sizeof(char *));
    out[0] = (char *)server_path;
    for (int i = 0; i < n; i++) out[i + 1] = argv[start + i];
    out[n + 1] = NULL;
    return out;
}

static pid_t start_server_background(const char *server_path, char **server_argv, char **log_path_out) {
    char *log_path = xasprintf("/tmp/ds4-launch-server-%ld.log", (long)time(NULL));
    pid_t pid = fork();
    if (pid < 0) die("failed to fork ds4-server: %s", strerror(errno));
    if (pid == 0) {
        (void)setsid();
        int nullfd = open("/dev/null", O_RDONLY);
        if (nullfd >= 0) {
            dup2(nullfd, STDIN_FILENO);
            close(nullfd);
        }
        int logfd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (logfd >= 0) {
            dup2(logfd, STDOUT_FILENO);
            dup2(logfd, STDERR_FILENO);
            close(logfd);
        }
        execv(server_path, server_argv);
        fprintf(stderr, "ds4-launch: failed to exec %s: %s\n", server_path, strerror(errno));
        _exit(127);
    }
    *log_path_out = log_path;
    return pid;
}

static bool existing_server_from_lock(long *pid_out, int *port_out) {
    long pid = read_lock_owner();
    if (!pid_exists(pid)) return false;
    int port = lsof_port_for_pid(pid);
    if (port <= 0) return false;
    *pid_out = pid;
    *port_out = port;
    return true;
}

static void ensure_server(const char *argv0, int argc, char **argv, int server_start,
                          server_cli *cli, char **log_path_out) {
    long existing_pid = -1;
    int existing_port = 0;
    if (existing_server_from_lock(&existing_pid, &existing_port)) {
        cli->host = DS4_LAUNCH_DEFAULT_HOST;
        cli->port = existing_port;
        fprintf(stderr,
                "ds4-launch: found ds4-server already running (pid %ld) on port %d; using http://127.0.0.1:%d\n",
                existing_pid, existing_port, existing_port);
        return;
    }

    char *server_path = find_ds4_server(argv0);
    char **server_argv = build_server_argv(server_path, argc, argv, server_start);
    char *log_path = NULL;
    pid_t pid = start_server_background(server_path, server_argv, &log_path);
    free(server_argv);
    free(server_path);

    fprintf(stderr, "ds4-launch: starting ds4-server (pid %ld), log: %s\n",
            (long)pid, log_path);

    const int step_ms = 200;
    int waited = 0;
    while (waited < DS4_LAUNCH_STARTUP_TIMEOUT_MS) {
        if (tcp_connect_ok(cli->host, cli->port)) {
            fprintf(stderr, "ds4-launch: ds4-server ready on http://%s:%d\n",
                    client_host(cli->host), cli->port);
            if (log_path_out) *log_path_out = log_path;
            else free(log_path);
            return;
        }

        int status = 0;
        pid_t done = waitpid(pid, &status, WNOHANG);
        if (done == pid) {
            char *log = read_file(log_path, NULL);
            long lock_pid = log ? parse_lock_pid_from_text(log) : -1;
            free(log);
            if (pid_exists(lock_pid)) {
                int port = lsof_port_for_pid(lock_pid);
                if (port > 0) {
                    cli->host = DS4_LAUNCH_DEFAULT_HOST;
                    cli->port = port;
                    fprintf(stderr,
                            "ds4-launch: found ds4-server already running (pid %ld) on port %d; using http://127.0.0.1:%d\n",
                            lock_pid, port, port);
                    if (log_path_out) *log_path_out = log_path;
                    else free(log_path);
                    return;
                }
            }
            die("ds4-server exited before becoming ready; see %s", log_path);
        }

        sleep_ms(step_ms);
        waited += step_ms;
    }
    die("timed out waiting for ds4-server; see %s", log_path);
}

static client_config make_client_config(const server_cli *cli) {
    client_config cfg = {0};
    cfg.ctx = cli->ctx;
    cfg.tokens = cli->tokens;
    cfg.base_url = xasprintf("http://%s:%d", client_host(cli->host), cli->port);
    cfg.base_url_v1 = xasprintf("%s/v1", cfg.base_url);
    return cfg;
}

static void free_client_config(client_config *cfg) {
    free(cfg->base_url);
    free(cfg->base_url_v1);
    memset(cfg, 0, sizeof(*cfg));
}

static void configure_environment(launch_tool tool, const client_config *cfg) {
    switch (tool) {
    case TOOL_PI:
        configure_pi(cfg);
        break;
    case TOOL_COPILOT:
        setenv("COPILOT_PROVIDER_BASE_URL", cfg->base_url_v1, 1);
        setenv("COPILOT_PROVIDER_API_KEY", DS4_LAUNCH_API_KEY, 1);
        setenv("COPILOT_PROVIDER_WIRE_API", "responses", 1);
        setenv("COPILOT_MODEL", DS4_LAUNCH_MODEL_NAME, 1);
        break;
    case TOOL_OPENCODE: {
        char *content = build_opencode_config_json(cfg);
        setenv("OPENCODE_CONFIG_CONTENT", content, 1);
        free(content);
        break;
    }
    }
}

static void exec_tool(launch_tool tool, char *tool_path) {
    char *const argv_pi[] = { tool_path, NULL };
    char *const argv_opencode[] = { tool_path, NULL };
    char *const argv_with_model[] = {
        tool_path,
        "--model",
        DS4_LAUNCH_MODEL_NAME,
        NULL,
    };

    switch (tool) {
    case TOOL_PI:
        fprintf(stderr, "ds4-launch: launching Pi with %s\n", DS4_LAUNCH_MODEL_NAME);
        execv(tool_path, argv_pi);
        break;
    case TOOL_COPILOT:
        fprintf(stderr, "ds4-launch: launching Copilot CLI with %s\n", DS4_LAUNCH_MODEL_NAME);
        execv(tool_path, argv_with_model);
        break;
    case TOOL_OPENCODE:
        fprintf(stderr, "ds4-launch: launching OpenCode with %s\n", DS4_LAUNCH_MODEL_NAME);
        execv(tool_path, argv_opencode);
        break;
    }
    die("failed to exec %s: %s", tool_path, strerror(errno));
}

#ifdef DS4_LAUNCH_TEST

static int launch_test_failures = 0;

static void launch_test_assert(bool cond, const char *file, int line, const char *expr) {
    if (cond) return;
    fprintf(stderr, "%s:%d: launch assertion failed: %s\n", file, line, expr);
    launch_test_failures++;
}

#define LAUNCH_TEST_ASSERT(expr) launch_test_assert((expr), __FILE__, __LINE__, #expr)

static void launch_test_accept_tool(const char *name, launch_tool expected, int expected_argc) {
    char *argv[] = {"ds4-launch", (char *)name, NULL};
    launch_tool tool = TOOL_PI;
    int tool_argc = 0;
    LAUNCH_TEST_ASSERT(canonical_tool(2, argv, &tool_argc, &tool));
    LAUNCH_TEST_ASSERT(tool == expected);
    LAUNCH_TEST_ASSERT(tool_argc == expected_argc);
}

static void launch_test_reject_tool(const char *name) {
    char *argv[] = {"ds4-launch", (char *)name, NULL};
    launch_tool tool = TOOL_PI;
    int tool_argc = 0;
    LAUNCH_TEST_ASSERT(!canonical_tool(2, argv, &tool_argc, &tool));
}

static void launch_test_tool_names(void) {
    launch_test_accept_tool("pi", TOOL_PI, 2);
    launch_test_accept_tool("copilot", TOOL_COPILOT, 2);
    launch_test_reject_tool("claude");
    launch_test_reject_tool("Claude");
    launch_test_accept_tool("opencode", TOOL_OPENCODE, 2);
    launch_test_accept_tool("open-code", TOOL_OPENCODE, 2);
    launch_test_accept_tool("open_code", TOOL_OPENCODE, 2);
}

static void launch_test_parse_server_cli(void) {
    char *argv[] = {
        "ds4-launch", "pi",
        "--quality", "--metal", "--unknown", "value",
        "--host", "0.0.0.0",
        "--port", "9000",
        "-c", "100000",
        "--ctx", "200000",
        "-n", "123",
        "--tokens", "456",
        "--kv-disk-dir", "/tmp/ds4-kv",
        "--backend", "cpu",
        "--dir-steering-file", "dir.f32",
        "--dir-steering-ffn", "1.5",
        "--dir-steering-attn", "0.5",
        "--port", "8123",
        NULL,
    };
    int argc = 0;
    while (argv[argc]) argc++;
    server_cli cli = parse_server_cli(argc, argv, 2);
    LAUNCH_TEST_ASSERT(!strcmp(cli.host, "0.0.0.0"));
    LAUNCH_TEST_ASSERT(cli.port == 8123);
    LAUNCH_TEST_ASSERT(cli.ctx == 200000);
    LAUNCH_TEST_ASSERT(cli.tokens == 456);
}

static void launch_test_lock_pid_parse(void) {
    const char *msg = "ds4: another ds4 process is already running (pid 73679); refusing to start\n";
    LAUNCH_TEST_ASSERT(parse_lock_pid_from_text(msg) == 73679);
    LAUNCH_TEST_ASSERT(parse_lock_pid_from_text("ds4: another ds4 process is already running; refusing to start") == -1);
}

static void launch_test_lsof_port_parse(void) {
    LAUNCH_TEST_ASSERT(parse_lsof_port_line(
        "ds4-serve 73679 dchini 5u IPv4 0x0 0t0 TCP 127.0.0.1:8000 (LISTEN)") == 8000);
    LAUNCH_TEST_ASSERT(parse_lsof_port_line(
        "ds4-serve 73679 dchini 5u IPv4 0x0 0t0 TCP 0.0.0.0:9000 (LISTEN)") == 9000);
    LAUNCH_TEST_ASSERT(parse_lsof_port_line(
        "ds4-serve 73679 dchini 5u IPv6 0x0 0t0 TCP *:7777 (LISTEN)") == 7777);
}

int ds4_launch_unit_tests_run(void) {
    launch_test_failures = 0;
    launch_test_tool_names();
    launch_test_parse_server_cli();
    launch_test_lock_pid_parse();
    launch_test_lsof_port_parse();
    return launch_test_failures;
}

#else

int main(int argc, char **argv) {
    if (argc == 1 || (argc == 2 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")))) {
        usage(stdout);
        return argc == 1 ? 2 : 0;
    }

    launch_tool tool;
    int server_start = 0;
    if (!canonical_tool(argc, argv, &server_start, &tool)) {
        fprintf(stderr, "ds4-launch: unknown tool: %s\n", argv[1]);
        usage(stderr);
        return 2;
    }

    server_cli cli = parse_server_cli(argc, argv, server_start);
    char *log_path = NULL;
    ensure_server(argv[0], argc, argv, server_start, &cli, &log_path);

    client_config cfg = make_client_config(&cli);
    configure_environment(tool, &cfg);
    char *tool_path = find_tool_binary(tool);
    free_client_config(&cfg);
    free(log_path);

    exec_tool(tool, tool_path);
    return 1;
}

#endif
