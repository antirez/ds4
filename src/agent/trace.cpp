#include "ds4_agent_internal.h"



/* ============================================================================
 * Trace Logging
 * ============================================================================
 */

static void agent_trace_time(FILE *fp) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(fp, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}



void agent_trace(agent_worker *w, const char *fmt, ...) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fputs(" ", w->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(w->trace, fmt, ap);
    va_end(ap);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}



static void agent_trace_escaped(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '\\': fputs("\\\\", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        case '"': fputs("\\\"", fp); break;
        default:
            if (c < 32 || c == 127) fprintf(fp, "\\x%02x", c);
            else fputc(c, fp);
            break;
        }
    }
}



void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " token index=%d id=%d bytes=%zu text=\"",
            index, token, text_len);
    agent_trace_escaped(w->trace, text ? text : "", text_len);
    fputs("\" hex=", w->trace);
    for (size_t i = 0; i < text_len; i++)
        fprintf(w->trace, "%02x", (unsigned char)text[i]);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}



void agent_trace_tokens(agent_worker *w, const char *label,
                               const ds4_tokens *tokens, int start) {
    if (!w || !w->trace || !tokens) return;
    if (start < 0) start = 0;
    if (start > tokens->len) start = tokens->len;
    agent_trace(w, "tokens label=%s start=%d len=%d", label ? label : "",
                start, tokens->len);
    for (int i = start; i < tokens->len; i++) {
        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, tokens->v[i], &text_len);
        agent_trace_token(w, tokens->v[i], text, text_len, i);
        free(text);
    }
}



void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " %s=\"", label ? label : "text");
    agent_trace_escaped(w->trace, text ? text : "", len);
    fputs("\"\n", w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

