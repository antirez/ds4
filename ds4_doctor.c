#include "ds4.h"
#include "ds4_help.h"

/* ds4-doctor self-diagnostic command.
 *
 * Runs read-only checks against the local installation: backend compile,
 * model file integrity, environment sanity. No GPU loads beyond what the
 * probe functions already do, no network calls, no filesystem writes.
 * Intended to be safe to run in CI without GPU hardware.
 *
 * U1 is the skeleton: parse options, dispatch --help, print a placeholder.
 * U2 ships the backend compile check.
 * U3 (model), U4 (environment) follow.
 * JSON / human output and exit-code policy land in U5.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *kv_disk_dir;
    bool json_output;
    bool color_output;
    int timeout_ms;
} doctor_config;

static doctor_config parse_options(int argc, char **argv) {
    doctor_config c = {0};
    c.timeout_ms = 5000;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            ds4_help_print(stdout, DS4_HELP_DOCTOR, NULL);
            exit(0);
        } else if (!strcmp(arg, "--json")) {
            c.json_output = true;
        } else if (!strcmp(arg, "--color")) {
            c.color_output = true;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ds4-doctor: %s requires a path argument\n", arg);
                exit(2);
            }
            c.model_path = argv[++i];
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ds4-doctor: %s requires a path argument\n", arg);
                exit(2);
            }
            c.kv_disk_dir = argv[++i];
        } else if (!strcmp(arg, "--timeout")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "ds4-doctor: --timeout requires a value in ms\n");
                exit(2);
            }
            char *end = NULL;
            long v = strtol(argv[++i], &end, 10);
            if (end == argv[i] || v < 0 || v > 600000) {
                fprintf(stderr, "ds4-doctor: --timeout must be 0..600000 ms\n");
                exit(2);
            }
            c.timeout_ms = (int)v;
        } else {
            fprintf(stderr, "ds4-doctor: unknown option: %s\n", arg);
            ds4_help_print(stderr, DS4_HELP_DOCTOR, NULL);
            exit(2);
        }
    }
    return c;
}

/* Canonical check status. Mirrors KTD-2 in the plan:
 *   OK    check passed
 *   SKIP  not run (deferred, disabled, missing dependency)
 *   WARN  check ran but flagged a soft problem
 *   FAIL  check ran and found a hard problem
 *
 * Exit-code mapping lands in U5.
 */
typedef enum {
    DOCTOR_OK,
    DOCTOR_SKIP,
    DOCTOR_WARN,
    DOCTOR_FAIL,
} doctor_status;

static const char *doctor_status_name(doctor_status s) {
    switch (s) {
    case DOCTOR_OK:   return "ok";
    case DOCTOR_WARN: return "warn";
    case DOCTOR_FAIL: return "fail";
    case DOCTOR_SKIP: return "skip";
    }
    return "skip";
}

typedef struct {
    const char *id;       /* short stable identifier, e.g. "backend" */
    const char *title;    /* human-readable label */
    doctor_status status; /* canonical status */
    int duration_ms;      /* wall time spent in the check */
    const char *message;  /* one-line explanation or hint */
} doctor_check;

static uint64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/* Forward declarations for checks added in U3 and U4. */
static doctor_check doctor_check_backend(void);
static doctor_check doctor_check_model(const char *model_path, int timeout_ms);
static doctor_check doctor_check_port(void);
static doctor_check doctor_check_memory(void);
static doctor_check doctor_check_kv_disk(const char *kv_disk_dir);

static doctor_check doctor_check_backend(void) {
    doctor_check c = {0};
    c.id = "backend";
    c.title = "Backend compile";
    c.status = DOCTOR_OK;
#ifdef DS4_NO_GPU
    c.message = "CPU-only build (DS4_NO_GPU)";
#elif defined(__APPLE__)
    c.message = "Metal backend compiled in (Apple Silicon)";
#elif defined(DS4_ROCM_BUILD)
    c.message = "ROCm/HIP backend compiled in (Strix Halo / gfx1151)";
#else
    c.message = "CUDA backend compiled in (Linux/Windows CUDA)";
#endif
    return c;
}

static void print_human(const doctor_check *checks, size_t n, bool color) {
    (void)color; /* color rendering lands in U5 */
    for (size_t i = 0; i < n; i++) {
        const doctor_check *c = &checks[i];
        printf("[%s] %s: %s\n",
               doctor_status_name(c->status), c->id, c->message);
    }
}

int main(int argc, char **argv) {
    doctor_config cfg = parse_options(argc, argv);

    /* U2: backend check only. U3/U4 add model + env checks. U5 wires the
     * final JSON renderer + exit-code policy across all of them. */
    doctor_check checks[1];
    uint64_t t0 = now_ms();
    checks[0] = doctor_check_backend();
    checks[0].duration_ms = (int)(now_ms() - t0);

    if (cfg.json_output) {
        /* JSON renderer lands in U5; keep the empty array alive for now. */
        fputs("{\"status\":\"ok\",\"checks\":[]}\n", stdout);
        return 0;
    }
    print_human(checks, sizeof(checks) / sizeof(checks[0]), cfg.color_output);
    puts("ds4-doctor: skeleton — implementation pending in U3..U5");
    return 0;
}
