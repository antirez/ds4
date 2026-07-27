#include "pulsar_engine_internal.h"

/* Fatal-exit and colorized logging. Split out of util.c in the C++ port:
 * everything here is about reporting, nothing about compute. */

void pulsar_die(const char *msg) {
    fprintf(stderr, "pulsar: %s\n", msg);
    exit(1);
}



void pulsar_die_errno(const char *what, const char *path) {
    fprintf(stderr, "pulsar: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}



static const char *pulsar_log_color_code(pulsar_log_type type) {
    switch (type) {
    case PULSAR_LOG_PREFILL:
    case PULSAR_LOG_TIMING:
        return "\x1b[36m";
    case PULSAR_LOG_GENERATION:
    case PULSAR_LOG_OK:
        return "\x1b[32m";
    case PULSAR_LOG_KVCACHE:
        return "\x1b[33m";
    case PULSAR_LOG_TOOL:
        return "\x1b[90m";
    case PULSAR_LOG_WARNING:
        return "\x1b[38;5;208m";
    case PULSAR_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}



bool pulsar_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}



static void pulsar_vlog(FILE *fp, pulsar_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != PULSAR_LOG_DEFAULT && pulsar_log_is_tty(fp);
    if (colorize) fputs(pulsar_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}



void pulsar_log(FILE *fp, pulsar_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    pulsar_vlog(fp, type, fmt, ap);
    va_end(ap);
}
