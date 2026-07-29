/* ds4_gpu_args.c — shared parser for --gpu-vram and --gpu-devices.
 *
 * See ds4_gpu_args.h. The actual CUDA probing for the "auto" branch
 * lives in ds4_cuda.cu so this .c file never includes <cuda_runtime.h>
 * directly (mirroring the rest of the tree's separation between .c and
 * .cu compilation units).
 */
#include "ds4_gpu_args.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_GPU_ARGS_DEFAULT_SAFETY_MARGIN ((size_t)512 * 1024 * 1024)

/* Helper: copy at most outlen-1 bytes of msg into out and NUL-terminate. */
static void errbuf_set(char *out, size_t outlen, const char *msg) {
    if (!out || outlen == 0) return;
    size_t n = strlen(msg);
    if (n >= outlen) n = outlen - 1;
    memcpy(out, msg, n);
    out[n] = '\0';
}

static void errbuf_setf(char *out, size_t outlen, const char *fmt, ...) {
    if (!out || outlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(out, outlen, fmt, ap);
    va_end(ap);
}

/* Maximum per-entry values. Generous, but small enough to keep all
 * downstream arithmetic safely inside size_t/int. */
#define DS4_GPU_ARGS_MAX_VRAM_GB        16384   /* 16 TiB per device */
#define DS4_GPU_ARGS_MAX_DEVICE_INDEX   1023    /* CUDA visible-device ID cap */

/* Parse a comma-separated list of nonnegative integers into out_count
 * entries of out[]. Returns 0 on success, nonzero on parse error.
 * Empty entries (",,") or trailing/leading commas error. Per-entry
 * upper bound enforced via max_value (use LONG_MAX for "no cap"). */
static int parse_csv_int_list(const char *s,
                              long       *out,
                              int        *out_count,
                              int         max_count,
                              long        max_value,
                              char       *errbuf,
                              size_t      errbuflen,
                              const char *what) {
    *out_count = 0;
    if (!s || !*s) {
        errbuf_setf(errbuf, errbuflen, "%s: empty value", what);
        return 1;
    }
    const char *p = s;
    while (*p) {
        /* Disallow leading whitespace inside the list. */
        while (*p == ' ' || *p == '\t') p++;
        char *end = NULL;
        errno = 0;
        long v = strtol(p, &end, 10);
        if (end == p) {
            errbuf_setf(errbuf, errbuflen, "%s: not a number: '%s'", what, p);
            return 1;
        }
        if (errno != 0 || v < 0 || v > max_value) {
            errbuf_setf(errbuf, errbuflen,
                        "%s: out-of-range value: '%s' (allowed 0..%ld)",
                        what, p, max_value);
            return 1;
        }
        if (*out_count >= max_count) {
            errbuf_setf(errbuf, errbuflen, "%s: too many entries (max %d)", what, max_count);
            return 1;
        }
        out[*out_count] = v;
        (*out_count)++;
        while (*end == ' ' || *end == '\t') end++;
        if (*end == ',') {
            end++;
            if (*end == '\0' || *end == ',') {
                errbuf_setf(errbuf, errbuflen, "%s: empty entry near '%s'", what, end);
                return 1;
            }
        } else if (*end != '\0') {
            errbuf_setf(errbuf, errbuflen, "%s: junk after value: '%s'", what, end);
            return 1;
        }
        p = end;
    }
    if (*out_count == 0) {
        errbuf_setf(errbuf, errbuflen, "%s: no values parsed", what);
        return 1;
    }
    return 0;
}

int parse_gpu_vram_arg(const char     *vram_arg,
                       const char     *devices_arg,
                       ds4_gpu_config *out,
                       bool           *out_skip_cuda,
                       char           *errbuf,
                       size_t          errbuflen) {
    if (!out || !out_skip_cuda) {
        errbuf_set(errbuf, errbuflen, "internal: NULL out parameter");
        return 1;
    }
    *out_skip_cuda = false;
    /* Caller zeroed *out — just confirm. */
    out->n_gpus = 0;
    out->safety_margin_bytes = DS4_GPU_ARGS_DEFAULT_SAFETY_MARGIN;

    /* Parse devices_arg first so we have the filter when handling "auto". */
    long dev_list[DS4_MAX_GPUS] = {0};
    int  dev_count = 0;
    if (devices_arg) {
        if (parse_csv_int_list(devices_arg, dev_list, &dev_count,
                                DS4_MAX_GPUS,
                                DS4_GPU_ARGS_MAX_DEVICE_INDEX,
                                errbuf, errbuflen,
                                "--gpu-devices") != 0) {
            return 1;
        }
    }

    /* Case 1: vram_arg unset. */
    if (vram_arg == NULL) {
        if (devices_arg == NULL) {
            errbuf_set(errbuf, errbuflen, "internal: parse_gpu_vram_arg called with both args NULL");
            return 1;
        }
        /* --gpu-devices given without --gpu-vram: implicit auto with filter. */
        vram_arg = "auto";
    }

    /* Case 2: --gpu-vram 0 — short-circuit. */
    if (!strcmp(vram_arg, "0")) {
        if (devices_arg) {
            errbuf_set(errbuf, errbuflen,
                       "--gpu-vram 0 conflicts with --gpu-devices");
            return 1;
        }
        *out_skip_cuda = true;
        return 0;
    }

    /* Case 3: --gpu-vram auto. */
    if (!strcmp(vram_arg, "auto")) {
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
        int filter_buf[DS4_MAX_GPUS];
        const int *filter_ptr = NULL;
        if (dev_count > 0) {
            for (int i = 0; i < dev_count; i++) filter_buf[i] = (int)dev_list[i];
            filter_ptr = filter_buf;
        }
        if (ds4_gpu_args_probe_auto_cuda(filter_ptr, dev_count, out,
                                          out->safety_margin_bytes,
                                          errbuf, errbuflen) != 0) {
            return 1;
        }
        return 0;
#else
        (void)dev_list; (void)dev_count;
        errbuf_set(errbuf, errbuflen,
                   "--gpu-vram auto requires a CUDA build");
        return 1;
#endif
    }

    /* Case 4: explicit GB budgets — "N[,N...]" */
    long vram_list[DS4_MAX_GPUS] = {0};
    int  vram_count = 0;
    if (parse_csv_int_list(vram_arg, vram_list, &vram_count,
                            DS4_MAX_GPUS,
                            DS4_GPU_ARGS_MAX_VRAM_GB,
                            errbuf, errbuflen,
                            "--gpu-vram") != 0) {
        return 1;
    }
    if (dev_count > 0 && dev_count != vram_count) {
        errbuf_setf(errbuf, errbuflen,
                    "--gpu-devices count (%d) does not match --gpu-vram count (%d)",
                    dev_count, vram_count);
        return 1;
    }
    out->n_gpus = vram_count;
    for (int i = 0; i < vram_count; i++) {
        int dev_id = (dev_count > 0) ? (int)dev_list[i] : i;
        out->device_indices[i] = dev_id;
        out->vram_bytes[i] = (size_t)vram_list[i] * (size_t)1024 * 1024 * 1024;
    }
    return 0;
}

int format_gpu_layout_line(const ds4_gpu_config *cfg,
                           bool                  was_auto,
                           char                 *out,
                           size_t                outlen) {
    if (!out || outlen == 0 || !cfg) return -1;
    /* "ds4: GPU config: N devices [d0,d1,...] requested, budgets G0,G1,... GB; auto=<bool>" */
    int written = snprintf(out, outlen,
                           "ds4: GPU config: %d device%s [",
                           cfg->n_gpus, cfg->n_gpus == 1 ? "" : "s");
    if (written < 0 || (size_t)written >= outlen) return written;
    for (int i = 0; i < cfg->n_gpus; i++) {
        int n = snprintf(out + written, outlen - (size_t)written,
                         "%s%d", i ? "," : "", cfg->device_indices[i]);
        if (n < 0 || (size_t)(written + n) >= outlen) return written + n;
        written += n;
    }
    int n = snprintf(out + written, outlen - (size_t)written,
                     "] requested, budgets ");
    if (n < 0 || (size_t)(written + n) >= outlen) return written + n;
    written += n;
    for (int i = 0; i < cfg->n_gpus; i++) {
        double gb = (double)cfg->vram_bytes[i] / (1024.0 * 1024.0 * 1024.0);
        int m = snprintf(out + written, outlen - (size_t)written,
                         "%s%.0f", i ? "," : "", gb);
        if (m < 0 || (size_t)(written + m) >= outlen) return written + m;
        written += m;
    }
    int t = snprintf(out + written, outlen - (size_t)written,
                     " GB; auto=%s", was_auto ? "true" : "false");
    if (t < 0) return t;
    written += t;
    return written;
}
