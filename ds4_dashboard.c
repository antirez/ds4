/* ds4_dashboard.c - Real-time dashboard for ds4-server
 *
 * Self-contained module.  No dependency on ds4_server.c internals.
 * Uses only POSIX socket I/O and standard C.
 */
#include "ds4_dashboard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

/* ─── Internal structures ─── */

struct ds4_dashboard {
    ds4_dashboard_config cfg;            /* startup config (copied) */
    ds4_dashboard_status st;             /* live status */
    ds4_dashboard_metrics m;             /* cumulative metrics */
};

/* ─── Buf helper (simple growable string buffer) ─── */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} dash_buf;

static void dash_buf_init(dash_buf *b) {
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
}

static void dash_buf_reserve(dash_buf *b, size_t add) {
    if (b->len + add > b->cap) {
        size_t newcap = b->cap ? b->cap : 256;
        while (newcap < b->len + add) newcap *= 2;
        char *p = realloc(b->ptr, newcap);
        if (!p) return;
        b->ptr = p;
        b->cap = newcap;
    }
}

static void dash_buf_putc(dash_buf *b, char c) {
    dash_buf_reserve(b, 1);
    if (b->ptr) b->ptr[b->len++] = c;
}

static void dash_buf_puts(dash_buf *b, const char *s) {
    if (!s) return;
    size_t n = strlen(s);
    dash_buf_reserve(b, n);
    if (b->ptr) {
        memcpy(b->ptr + b->len, s, n);
        b->len += n;
    }
}

static void dash_buf_printf(dash_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    dash_buf_reserve(b, (size_t)n);
    if (b->ptr) {
        va_start(ap, fmt);
        vsnprintf(b->ptr + b->len, (size_t)n + 1, fmt, ap);
        va_end(ap);
        b->len += (size_t)n;
    }
}

static void dash_buf_free(dash_buf *b) {
    free(b->ptr);
    b->ptr = NULL;
    b->len = 0;
    b->cap = 0;
}

/* ─── JSON string escaping ─── */

static void dash_json_escape(dash_buf *b, const char *s) {
    if (!s) { dash_buf_puts(b, "\"\""); return; }
    dash_buf_putc(b, '"');
    for (; *s; s++) {
        switch (*s) {
            case '"':  dash_buf_puts(b, "\\\""); break;
            case '\\': dash_buf_puts(b, "\\\\"); break;
            case '\n': dash_buf_puts(b, "\\n");  break;
            case '\t': dash_buf_puts(b, "\\t");  break;
            default:   dash_buf_putc(b, *s);     break;
        }
    }
    dash_buf_putc(b, '"');
}

/* ─── HTTP response helper ─── */

static bool dash_send_all(int fd, const void *p, size_t n) {
    if (!p || n == 0) return true;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, (const char *)p + off, n - off);
        if (w <= 0) return false;
        off += (size_t)w;
    }
    return true;
}

static void dash_http_response(int fd, bool cors, int code,
                               const char *type, const char *body) {
    const char *reason = code == 200 ? "OK" :
                         code == 204 ? "No Content" :
                         code == 400 ? "Bad Request" :
                         code == 404 ? "Not Found" :
                         code == 500 ? "Internal Server Error" : "Error";
    size_t body_len = body ? strlen(body) : 0;
    dash_buf h;
    dash_buf_init(&h);
    dash_buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, reason, body_len);
    if (type && type[0]) {
        dash_buf_puts(&h, "Content-Type: ");
        dash_buf_puts(&h, type);
        dash_buf_puts(&h, "\r\n");
    }
    if (cors) dash_buf_puts(&h, "Access-Control-Allow-Origin: *\r\n");
    dash_buf_puts(&h, "Connection: close\r\n\r\n");
    dash_send_all(fd, h.ptr, h.len);
    if (body_len) dash_send_all(fd, body, body_len);
    dash_buf_free(&h);
}

/* ─── Metrics JSON serialization ─── */

static void dash_write_metrics_json(dash_buf *b, const ds4_dashboard_metrics *m) {
    dash_buf_puts(b, "{\n");
    dash_buf_printf(b, "  \"total_prompt_tokens\": %llu,\n",
                    (unsigned long long)m->total_prompt_tokens);
    dash_buf_printf(b, "  \"total_completion_tokens\": %llu,\n",
                    (unsigned long long)m->total_completion_tokens);
    dash_buf_printf(b, "  \"total_tokens\": %llu,\n",
                    (unsigned long long)(m->total_prompt_tokens + m->total_completion_tokens));
    dash_buf_printf(b, "  \"total_requests\": %llu,\n",
                    (unsigned long long)m->total_requests);
    dash_buf_printf(b, "  \"total_cache_hit_tokens\": %llu,\n",
                    (unsigned long long)m->total_cache_hit_tokens);
    dash_buf_printf(b, "  \"total_cache_hit_requests\": %llu,\n",
                    (unsigned long long)m->total_cache_hit_requests);
    dash_buf_printf(b, "  \"cache_hit_rate_all\": %.1f,\n",
                    m->total_prompt_tokens > 0
                    ? (double)m->total_cache_hit_tokens / (double)m->total_prompt_tokens * 100.0
                    : 0.0);
    dash_buf_printf(b, "  \"last_prefill_tps\": %.2f,\n", m->last_prefill_tps);
    dash_buf_printf(b, "  \"last_incremental_tps\": %.2f,\n", m->last_incremental_tps);
    dash_buf_printf(b, "  \"last_gen_tps\": %.2f,\n", m->last_gen_tps);
    dash_buf_printf(b, "  \"last_prompt_tokens\": %d,\n", m->last_prompt_tokens);
    dash_buf_printf(b, "  \"last_completion_tokens\": %d,\n", m->last_completion_tokens);
    dash_buf_printf(b, "  \"last_cache_hit_tokens\": %d,\n", m->last_cache_hit_tokens);
    dash_buf_printf(b, "  \"last_cache_hit_rate\": %.1f,\n",
                    m->last_cache_hit_rate * 100.0);
    dash_buf_printf(b, "  \"last_prefill_sec\": %.4f,\n", m->last_prefill_sec);
    dash_buf_printf(b, "  \"last_gen_sec\": %.4f\n", m->last_gen_sec);
    dash_buf_puts(b, "}\n");
}

/* ─── Status JSON serialization ─── */

static void dash_write_status_json(dash_buf *b, const ds4_dashboard_status *st) {
    dash_buf_puts(b, "{\n");
    dash_buf_printf(b, "  \"state\": %d,\n", st->state);
    dash_buf_printf(b, "  \"current_tokens\": %d,\n", st->current_tokens);
    dash_buf_printf(b, "  \"target_tokens\": %d,\n", st->target_tokens);
    dash_buf_printf(b, "  \"current_speed\": %.2f,\n", st->current_speed);
    dash_buf_printf(b, "  \"elapsed_sec\": %.4f,\n", st->elapsed_sec);
    dash_buf_printf(b, "  \"last_text\": ");
    dash_json_escape(b, st->last_text);
    dash_buf_printf(b, ",\n");
    dash_buf_printf(b, "  \"prefill_tokens\": %d,\n", st->prefill_tokens);
    dash_buf_printf(b, "  \"prefill_current\": %d,\n", st->prefill_current);
    dash_buf_printf(b, "  \"prefill_total\": %d,\n", st->prefill_total);
    dash_buf_printf(b, "  \"prefill_elapsed\": %.4f,\n", st->prefill_elapsed);
    dash_buf_printf(b, "  \"prefill_speed\": %.2f,\n", st->prefill_speed);
    dash_buf_printf(b, "  \"gen_begin\": %.4f\n", st->gen_begin);
    dash_buf_puts(b, "}\n");
}

/* ─── Config JSON serialization ─── */

static void dash_write_config_json(dash_buf *b, const ds4_dashboard_config *cfg) {
    dash_buf_puts(b, "{\n");
    dash_buf_printf(b, "  \"ctx_size\": %d,\n", cfg->ctx_size);
    dash_buf_printf(b, "  \"default_tokens\": %d,\n", cfg->default_tokens);
    dash_buf_printf(b, "  \"kv_disk_dir\": ");
    dash_json_escape(b, cfg->kv_disk_dir ? cfg->kv_disk_dir : "");
    dash_buf_printf(b, ",\n");
    dash_buf_printf(b, "  \"kv_disk_space_mb\": %llu,\n",
                    (unsigned long long)cfg->kv_disk_space_mb);
    dash_buf_printf(b, "  \"kv_cache_min_tokens\": %d,\n", cfg->kv_cache_min_tokens);
    dash_buf_printf(b, "  \"kv_cache_cold_max_tokens\": %d,\n", cfg->kv_cache_cold_max_tokens);
    dash_buf_printf(b, "  \"kv_cache_continued_interval_tokens\": %d,\n",
                    cfg->kv_cache_continued_interval_tokens);
    dash_buf_printf(b, "  \"kv_cache_boundary_trim_tokens\": %d,\n",
                    cfg->kv_cache_boundary_trim_tokens);
    dash_buf_printf(b, "  \"kv_cache_boundary_align_tokens\": %d,\n",
                    cfg->kv_cache_boundary_align_tokens);
    dash_buf_printf(b, "  \"backend\": ");
    dash_json_escape(b, cfg->backend_name ? cfg->backend_name : "");
    dash_buf_puts(b, "\n}\n");
}

/* ─── The dashboard HTML page ─── */

static const char *dash_html_page =
"<!DOCTYPE html><html><head>"
"<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>DS4 Dashboard</title>"
"<style>"
"*{margin:0;padding:0;box-sizing:border-box}"
"body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f0f;color:#eee;padding:20px}"
"h1{font-size:22px;font-weight:600;margin:0 0 16px 0;color:#fff}"
".card{background:#1a1a1a;border-radius:10px;padding:18px;margin-bottom:14px}"
".card h2{font-size:14px;font-weight:500;color:#888;text-transform:uppercase;letter-spacing:.5px;margin:0 0 8px 0}"
".stat{font-size:28px;font-weight:700;color:#fff}"
".stat-detail{font-size:13px;color:#888;margin-top:4px}"
".row{display:flex;gap:14px;flex-wrap:wrap}"
".row .card{flex:1;min-width:160px}"
".tps{color:#4ae}"
".green{color:#4c4}"
".yellow{color:#ea4}"
".refresh{font-size:11px;color:#666;margin-top:16px}"
".config-grid{display:flex;flex-wrap:wrap;gap:4px 12px;margin-bottom:14px;padding:10px 18px;background:#1a1a1a;border-radius:10px;font-size:12px}"
".config-item{color:#888;white-space:nowrap}"
".config-item span{color:#eee}"
".status-bar{display:flex;align-items:center;gap:8px;margin-bottom:14px;padding:10px 18px;background:#1a1a1a;border-radius:10px}"
".status-indicator{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}"
".status-idle .status-indicator{background:#666}"
".status-prefill .status-indicator{background:#4ae;animation:pulse 1s infinite}"
".status-think .status-indicator{background:#ea4;animation:pulse 1s infinite}"
".status-gen .status-indicator{background:#4c4;animation:pulse 1s infinite}"
"@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}"
".progress-bar{background:#333;border-radius:6px;height:8px;margin-top:8px;overflow:hidden;width:100%}"
".progress-fill{background:#4ae;height:100%;border-radius:6px;transition:width .3s ease}"
".progress-fill-green{background:#4c4;height:100%;border-radius:6px;transition:width .3s ease}"
".live-text{font-size:13px;color:#aaa;margin-top:8px;padding:8px;background:#222;border-radius:6px;max-height:80px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;font-family:monospace}"
".live-text:empty{display:none}"
".eta-text{font-size:24px;font-weight:700;color:#4ae}"
".eta-label{font-size:11px;color:#666;margin-top:2px}"
"@media(prefers-color-scheme:light){body{background:#f5f5f5;color:#222}"
".card{background:#fff}.card h2{color:#666}.stat{color:#111}"
".refresh{color:#999}.status-bar{background:#fff}"
".config-grid{background:#fff}.config-item{color:#666}.config-item span{color:#111}"
".live-text{background:#eee;color:#333}.eta-text{color:#48a}}"
"</style></head><body>"
"<h1>⚡ DS4 Server</h1>"
"<div class='status-bar' id='status-bar'>"
"<span class='status-indicator'></span>"
"<span id='status-label'>Loading...</span>"
"</div>"
"<div class='config-grid' id='config-grid'>"
"<div class='config-item'>ctx: <span id='cfg-ctx'>—</span></div>"
"<div class='config-item'>backend: <span id='cfg-backend'>—</span></div>"
"<div class='config-item'>max tokens: <span id='cfg-tokens'>—</span></div>"
"<div class='config-item' style='white-space:normal'>kv disk: <span id='cfg-kv-dir'>—</span></div>"
"<div class='config-item'>kv space: <span id='cfg-kv-mb'>—</span> MB</div>"
"<div class='config-item'>cold max: <span id='cfg-cold'>—</span></div>"
"<div class='config-item'>continued: <span id='cfg-continued'>—</span></div>"
"</div>"
"<div class='row'>"
"<div class='card'><h2>Total Tokens</h2><div class='stat' id='total'>—</div><div class='stat-detail'>prompt + completion</div></div>"
"<div class='card'><h2>Requests</h2><div class='stat' id='requests'>—</div><div class='stat-detail'>since startup</div></div>"
"</div>"
"<div class='row'>"
"<div class='card'><h2>Cache Hit Rate</h2><div class='stat green' id='cache-rate'>—</div><div class='stat-detail'><span id='cache-hit-tokens'>—</span> cached / <span id='cache-total-tokens'>—</span> total</div></div>"
"<div class='card'><h2>Cache Hit Requests</h2><div class='stat green' id='cache-hit-req'>—</div><div class='stat-detail'>of <span id='cache-total-req'>—</span> requests</div></div>"
"</div>"
"<div class='row'>"
"<div class='card'><h2>Prefill Speed</h2><div class='stat tps' id='prefill-tps'>—</div><div class='stat-detail'>total &middot; <span id='prefill-tokens'>—</span> tokens</div><div class='stat-detail' style='font-size:12px;color:#888'>incremental: <span class='tps' id='incremental-tps'>—</span></div></div>"
"<div class='card'><h2>Generation Speed</h2><div class='stat tps' id='gen-tps'>—</div><div class='stat-detail'>last request &middot; <span id='gen-tokens'>—</span> tokens</div></div>"
"</div>"
"<div class='card'><h2>Last Request</h2>"
"<div class='stat' id='last-prompt'>—</div>"
"<div class='stat-detail'>prompt tokens &rarr; <span id='last-completion'>—</span> completion tokens</div>"
"</div>"
"<div class='card' id='prefill-card' style='display:none'>"
"<h2>Prefill Progress</h2>"
"<div class='progress-bar'><div class='progress-fill' id='prefill-progress-fill' style='width:0%'></div></div>"
"<div class='stat-detail'><span id='prefill-current'>0</span> / <span id='prefill-total'>0</span> tokens &middot; <span id='prefill-speed'>0.0</span> t/s</div>"
"<div class='eta-text' id='prefill-eta'>—</div>"
"<div class='eta-label'>estimated time remaining</div>"
"</div>"
"<div class='card' id='live-card' style='display:none'>"
"<h2>Live Output</h2>"
"<div class='progress-bar'><div class='progress-fill-green' id='progress-fill' style='width:0%'></div></div>"
"<div class='stat-detail'><span id='live-tokens'>0</span> tokens &middot; <span id='live-speed'>0.0</span> t/s &middot; <span id='live-elapsed'>0s</span></div>"
"<div class='live-text' id='live-text'></div>"
"</div>"
"<div class='refresh'>Refreshing every 1s</div>"
"<script>"
"const stateNames=['Idle','Prefilling','Thinking','Generating'];"
"async function fetchMetrics(){try{const r=await fetch('/metrics');const d=await r.json();"
"document.getElementById('total').textContent=d.total_tokens.toLocaleString();"
"document.getElementById('requests').textContent=d.total_requests.toLocaleString();"
"document.getElementById('prefill-tps').textContent=d.last_prefill_tps.toFixed(1)+' t/s';"
"document.getElementById('incremental-tps').textContent=d.last_incremental_tps.toFixed(1)+' t/s (uncached only)';"
"document.getElementById('prefill-tokens').textContent=d.last_prompt_tokens.toLocaleString();"
"document.getElementById('gen-tps').textContent=d.last_gen_tps.toFixed(1)+' t/s';"
"document.getElementById('gen-tokens').textContent=d.last_completion_tokens.toLocaleString();"
"document.getElementById('last-prompt').textContent=d.last_prompt_tokens.toLocaleString()+' prompt tokens';"
"document.getElementById('last-completion').textContent=d.last_completion_tokens.toLocaleString();"
"document.getElementById('cache-rate').textContent=d.cache_hit_rate_all.toFixed(1)+'%';"
"document.getElementById('cache-hit-tokens').textContent=d.total_cache_hit_tokens.toLocaleString();"
"document.getElementById('cache-total-tokens').textContent=d.total_prompt_tokens.toLocaleString();"
"document.getElementById('cache-hit-req').textContent=d.total_cache_hit_requests.toLocaleString();"
"document.getElementById('cache-total-req').textContent=d.total_requests.toLocaleString();"
"}catch(e){document.getElementById('total').textContent='error'}}"
"async function fetchStatus(){try{const r=await fetch('/status');const d=await r.json();"
"const sb=document.getElementById('status-bar');"
"const label=document.getElementById('status-label');"
"const state=d.state||0;"
"sb.className='status-bar status-'+['idle','prefill','think','gen'][state];"
"label.textContent=stateNames[state]||'Unknown';"
"const pc=document.getElementById('prefill-card');"
"const lc=document.getElementById('live-card');"
"if(state===1){"
"pc.style.display='';lc.style.display='none';"
"const cur=d.prefill_current||0;const tot=d.prefill_total||1;"
"const pct=tot>0?Math.min(100,cur/tot*100):0;"
"document.getElementById('prefill-progress-fill').style.width=pct+'%';"
"document.getElementById('prefill-current').textContent=cur.toLocaleString();"
"document.getElementById('prefill-total').textContent=tot.toLocaleString();"
"const sp=d.prefill_speed||0;document.getElementById('prefill-speed').textContent=sp.toFixed(1);"
"const eta=sp>0.1?(tot-cur)/sp:0;"
"if(eta>0){const m=Math.floor(eta/60);const s=Math.floor(eta%60);document.getElementById('prefill-eta').textContent=(m>0?m+'m ':'')+s+'s'}else{document.getElementById('prefill-eta').textContent='—'}"
"}else if(state===2||state===3){"
"pc.style.display='none';lc.style.display='';"
"document.getElementById('live-tokens').textContent=d.current_tokens.toLocaleString();"
"document.getElementById('live-speed').textContent=d.current_speed.toFixed(1);"
"const sec=d.elapsed_sec||0;const ts=Math.floor(sec);"
"document.getElementById('live-elapsed').textContent=(ts<60?ts+'s':Math.floor(ts/60)+'m '+ts%60+'s');"
"const gpct=d.target_tokens>0?Math.min(100,(d.current_tokens/d.target_tokens)*100):0;"
"document.getElementById('progress-fill').style.width=gpct+'%';"
"const lt=document.getElementById('live-text');"
"if(d.last_text&&d.last_text.length>0){lt.textContent=d.last_text;lt.style.display=''}else{lt.textContent='';lt.style.display='none'}"
"}else{"
"pc.style.display='none';lc.style.display='none'"
"}"
"}catch(e){document.getElementById('status-label').textContent='Status unavailable'}}"
"async function fetchConfig(){try{const r=await fetch('/config');const d=await r.json();"
"document.getElementById('cfg-ctx').textContent=d.ctx_size.toLocaleString();"
"document.getElementById('cfg-backend').textContent=d.backend;"
"document.getElementById('cfg-tokens').textContent=d.default_tokens.toLocaleString();"
"document.getElementById('cfg-kv-dir').textContent=d.kv_disk_dir||'none';"
"document.getElementById('cfg-kv-mb').textContent=d.kv_disk_space_mb.toLocaleString();"
"document.getElementById('cfg-cold').textContent=d.kv_cache_cold_max_tokens.toLocaleString();"
"document.getElementById('cfg-continued').textContent=d.kv_cache_continued_interval_tokens.toLocaleString();"
"}catch(e){}}"
"setInterval(fetchMetrics,1000);fetchMetrics();"
"setInterval(fetchStatus,1000);fetchStatus();"
"fetchConfig()"
"</script></body></html>";

/* ─── Metrics persistence ─── */

static void dash_persist(const ds4_dashboard_metrics *m, const char *kv_dir) {
    if (!kv_dir || !kv_dir[0]) return;
    size_t path_len = strlen(kv_dir) + 32;
    char *path = malloc(path_len);
    if (!path) return;
    snprintf(path, path_len, "%s/metrics.json", kv_dir);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "{\n"
                "\"total_prompt_tokens\": %llu,\n"
                "\"total_completion_tokens\": %llu,\n"
                "\"total_requests\": %llu,\n"
                "\"total_cache_hit_tokens\": %llu,\n"
                "\"total_cache_hit_requests\": %llu\n"
                "}\n",
                (unsigned long long)m->total_prompt_tokens,
                (unsigned long long)m->total_completion_tokens,
                (unsigned long long)m->total_requests,
                (unsigned long long)m->total_cache_hit_tokens,
                (unsigned long long)m->total_cache_hit_requests);
        fclose(fp);
    }
    free(path);
}

static void dash_load(ds4_dashboard_metrics *m, const char *kv_dir) {
    if (!kv_dir || !kv_dir[0]) return;
    memset(m, 0, sizeof(*m));
    size_t path_len = strlen(kv_dir) + 32;
    char *path = malloc(path_len);
    if (!path) return;
    snprintf(path, path_len, "%s/metrics.json", kv_dir);
    FILE *fp = fopen(path, "r");
    if (!fp) { free(path); return; }
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    const char *keys[5] = {
        "\"total_prompt_tokens\"",
        "\"total_completion_tokens\"",
        "\"total_requests\"",
        "\"total_cache_hit_tokens\"",
        "\"total_cache_hit_requests\""
    };
    uint64_t v[5] = {0};
    for (int i = 0; i < 5; i++) {
        const char *k = strstr(buf, keys[i]);
        if (k) {
            k += strlen(keys[i]);
            while (*k && (*k == ':' || *k == ' ' || *k == '\n' || *k == '\r' || *k == '\t')) k++;
            if (*k) sscanf(k, "%llu", (unsigned long long *)&v[i]);
        }
    }
    m->total_prompt_tokens = v[0];
    m->total_completion_tokens = v[1];
    m->total_requests = v[2];
    m->total_cache_hit_tokens = v[3];
    m->total_cache_hit_requests = v[4];
    free(path);
}

/* ─── Public API ─── */

void *ds4_dashboard_init(const ds4_dashboard_config *cfg) {
    struct ds4_dashboard *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    if (cfg) {
        d->cfg = *cfg;
        /* Copy string fields so caller can free their originals */
        if (cfg->kv_disk_dir) {
            d->cfg.kv_disk_dir = strdup(cfg->kv_disk_dir);
        }
        if (cfg->backend_name) {
            d->cfg.backend_name = strdup(cfg->backend_name);
        }
    }
    return d;
}

void ds4_dashboard_free(void *dash) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d) return;
    free((void *)d->cfg.kv_disk_dir);
    free((void *)d->cfg.backend_name);
    free(d);
}

void ds4_dashboard_set_status(void *dash, const ds4_dashboard_status *st) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d || !st) return;
    d->st = *st;
}

void ds4_dashboard_update_metrics(void *dash, const ds4_dashboard_metrics *m,
                                  int cached_tokens, int prompt_tokens) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d || !m) return;
    d->m.total_prompt_tokens += m->total_prompt_tokens;
    d->m.total_completion_tokens += m->total_completion_tokens;
    d->m.total_requests += m->total_requests;
    d->m.total_cache_hit_tokens += m->total_cache_hit_tokens;
    d->m.total_cache_hit_requests += m->total_cache_hit_requests;
    d->m.last_prompt_tokens = m->last_prompt_tokens;
    d->m.last_completion_tokens = m->last_completion_tokens;
    d->m.last_cache_hit_tokens = cached_tokens > 0 ? cached_tokens : 0;
    d->m.last_cache_hit_rate = prompt_tokens > 0
        ? (double)cached_tokens / (double)prompt_tokens : 0.0;
    d->m.last_prefill_sec = m->last_prefill_sec;
    d->m.last_gen_sec = m->last_gen_sec;
    d->m.last_prefill_tps = m->last_prefill_tps;
    d->m.last_gen_tps = m->last_gen_tps;
    /* Compute incremental tps */
    int uncached = prompt_tokens - (cached_tokens > 0 ? cached_tokens : 0);
    if (uncached < 0) uncached = 0;
    d->m.last_incremental_tps = m->last_prefill_sec > 0.001 && uncached > 0
        ? (double)uncached / m->last_prefill_sec : 0.0;
}

void ds4_dashboard_persist(void *dash, const char *kv_dir) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d) return;
    dash_persist(&d->m, kv_dir);
}

void ds4_dashboard_load(void *dash, const char *kv_dir) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d) return;
    dash_load(&d->m, kv_dir);
}

bool ds4_dashboard_handle_http(void *dash, const char *method, const char *path,
                               int fd, bool cors) {
    struct ds4_dashboard *d = (struct ds4_dashboard *)dash;
    if (!d || !method || !path) return false;
    if (strcmp(method, "GET") != 0) return false;

    /* /config */
    if (!strcmp(path, "/config")) {
        dash_buf b;
        dash_buf_init(&b);
        dash_write_config_json(&b, &d->cfg);
        dash_http_response(fd, cors, 200, "application/json", b.ptr);
        dash_buf_free(&b);
        return true;
    }

    /* /status */
    if (!strcmp(path, "/status")) {
        dash_buf b;
        dash_buf_init(&b);
        dash_write_status_json(&b, &d->st);
        dash_http_response(fd, cors, 200, "application/json", b.ptr);
        dash_buf_free(&b);
        return true;
    }

    /* /metrics or /dashboard */
    if (!strcmp(path, "/metrics")) {
        dash_buf b;
        dash_buf_init(&b);
        dash_write_metrics_json(&b, &d->m);
        dash_http_response(fd, cors, 200, "application/json", b.ptr);
        dash_buf_free(&b);
        return true;
    }

    if (!strcmp(path, "/dashboard") || !strcmp(path, "/")) {
        dash_http_response(fd, cors, 200, "text/html", dash_html_page);
        return true;
    }

    return false;
}
