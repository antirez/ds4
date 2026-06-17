/* ds4_dashboard.h - Real-time dashboard for ds4-server
 *
 * This module provides a self-contained HTML dashboard with real-time
 * status monitoring, metrics, and startup configuration display.
 * It can be extracted and reused by other projects that embed the
 * ds4-server HTTP interface.
 *
 * Usage:
 *   1. Call ds4_dashboard_init() to initialize
 *   2. Call ds4_dashboard_handle_http() from your HTTP handler
 *   3. Call ds4_dashboard_set_status() during generation to update live status
 *   4. Call ds4_dashboard_update_metrics() after each request completes
 *   5. Call ds4_dashboard_persist() / ds4_dashboard_load() for disk persistence
 */
#ifndef DS4_DASHBOARD_H
#define DS4_DASHBOARD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Data structures ─── */

/* Live status updated during generation */
typedef struct {
    int           state;              /* 0=idle, 1=prefilling, 2=thinking, 3=generating */
    int           current_tokens;     /* tokens generated so far */
    int           target_tokens;      /* max tokens budget */
    double        current_speed;      /* current generation speed (t/s) */
    double        elapsed_sec;        /* seconds since generation started */
    char          last_text[256];     /* last ~200 chars of generated text */
    double        prefill_begin;      /* prefill start timestamp */
    double        prefill_end;        /* prefill end timestamp */
    int           prefill_tokens;     /* tokens prefilled */
    double        gen_begin;          /* generation start timestamp */
    int           prefill_current;    /* tokens processed during prefill */
    int           prefill_total;      /* total tokens to process */
    double        prefill_elapsed;    /* prefill elapsed seconds */
    double        prefill_speed;      /* current prefill speed (t/s) */
} ds4_dashboard_status;

/* Cumulative metrics */
typedef struct {
    uint64_t total_prompt_tokens;
    uint64_t total_completion_tokens;
    uint64_t total_requests;
    uint64_t total_cache_hit_tokens;
    uint64_t total_cache_hit_requests;
    double   last_prefill_tps;
    double   last_incremental_tps;
    double   last_gen_tps;
    int      last_prompt_tokens;
    int      last_completion_tokens;
    double   last_prefill_sec;
    double   last_gen_sec;
    int      last_cache_hit_tokens;
    double   last_cache_hit_rate;
} ds4_dashboard_metrics;

/* Startup configuration (read once, immutable) */
typedef struct {
    int      ctx_size;
    int      default_tokens;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    int      kv_cache_min_tokens;
    int      kv_cache_cold_max_tokens;
    int      kv_cache_continued_interval_tokens;
    int      kv_cache_boundary_trim_tokens;
    int      kv_cache_boundary_align_tokens;
    const char *backend_name;
} ds4_dashboard_config;

/* ─── API ─── */

/* Initialize a dashboard handle.  Returns an opaque pointer.
 * The config structure is copied internally; caller can free it after init. */
void *ds4_dashboard_init(const ds4_dashboard_config *cfg);

/* Handle an HTTP GET request for /dashboard, /metrics, /status, /config.
 * Returns true if the path was handled (response written to fd).
 * Pass the opaque handle returned by ds4_dashboard_init(). */
bool ds4_dashboard_handle_http(void *dash, const char *method, const char *path,
                               int fd, bool cors);

/* Update live status (called during generation, under the caller's mutex). */
void ds4_dashboard_set_status(void *dash, const ds4_dashboard_status *st);

/* Update cumulative metrics (call after each request completes).
 * The cached_tokens and prompt_tokens are used to compute incremental tps
 * and cache hit rate. */
void ds4_dashboard_update_metrics(void *dash, const ds4_dashboard_metrics *m,
                                  int cached_tokens, int prompt_tokens);

/* Persist cumulative metrics to a JSON file in kv_dir.
 * Call at shutdown and periodically. */
void ds4_dashboard_persist(void *dash, const char *kv_dir);

/* Load cumulative metrics from a JSON file in kv_dir.
 * Call at startup after init. */
void ds4_dashboard_load(void *dash, const char *kv_dir);

/* Destroy the dashboard handle. */
void ds4_dashboard_free(void *dash);

#ifdef __cplusplus
}
#endif

#endif /* DS4_DASHBOARD_H */
