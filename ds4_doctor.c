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
 * U3 ships the model file integrity check.
 * U4 ships the port / memory / kv-disk checks.
 * JSON / human output and exit-code policy land in U5.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <netinet/in.h>
#include <arpa/inet.h>

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
    char message[256];    /* one-line explanation or hint */
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
    snprintf(c.message, sizeof(c.message), "CPU-only build (DS4_NO_GPU)");
#elif defined(__APPLE__)
    snprintf(c.message, sizeof(c.message), "Metal backend compiled in (Apple Silicon)");
#elif defined(DS4_ROCM_BUILD)
    snprintf(c.message, sizeof(c.message), "ROCm/HIP backend compiled in (Strix Halo / gfx1151)");
#else
    snprintf(c.message, sizeof(c.message), "CUDA backend compiled in (Linux/Windows CUDA)");
#endif
    return c;
}

/* Read a little-endian u32 / u64 with strict short-read detection.
 * Returns false if the stream ends before the requested width. */
static bool read_u32_le(FILE *fp, uint32_t *out) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, fp) != 4) return false;
    *out = (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
    return true;
}

static bool read_u64_le(FILE *fp, uint64_t *out) {
    uint8_t buf[8];
    if (fread(buf, 1, 8, fp) != 8) return false;
    *out = (uint64_t)buf[0]
         | ((uint64_t)buf[1] << 8)
         | ((uint64_t)buf[2] << 16)
         | ((uint64_t)buf[3] << 24)
         | ((uint64_t)buf[4] << 32)
         | ((uint64_t)buf[5] << 40)
         | ((uint64_t)buf[6] << 48)
         | ((uint64_t)buf[7] << 56);
    return true;
}

#define DOCTOR_GGUF_MAGIC 0x46554747u /* "GGUF" little-endian; matches
                                        DS4_GGUF_MAGIC in ds4.c. */
#define DOCTOR_GGUF_VERSION 3u

static doctor_check doctor_check_model(const char *model_path, int timeout_ms) {
    (void)timeout_ms; /* timeout-bounded read lands in U5 */
    doctor_check c = {0};
    c.id = "model";
    c.title = "Model file integrity";
    c.status = DOCTOR_OK;

    if (model_path == NULL) {
        c.status = DOCTOR_SKIP;
        snprintf(c.message, sizeof(c.message),
                 "no model path provided (use -m FILE or DS4_GGUF)");
        return c;
    }

    FILE *fp = fopen(model_path, "rb");
    if (!fp) {
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "cannot open %s", model_path);
        return c;
    }

    struct stat st;
    if (fstat(fileno(fp), &st) != 0) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "cannot stat %s", model_path);
        return c;
    }

    uint32_t magic = 0;
    if (!read_u32_le(fp, &magic)) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: header too short to read magic (< 4 bytes)", model_path);
        return c;
    }

    if (magic != DOCTOR_GGUF_MAGIC) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: bad magic 0x%08x (expected GGUF)", model_path, magic);
        return c;
    }

    uint32_t version = 0;
    if (!read_u32_le(fp, &version)) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: header truncated at version field", model_path);
        return c;
    }

    if (version != DOCTOR_GGUF_VERSION) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: unsupported GGUF version %u (expected %u)",
                 model_path, version, DOCTOR_GGUF_VERSION);
        return c;
    }

    uint64_t n_tensors = 0;
    if (!read_u64_le(fp, &n_tensors)) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: header truncated at tensor count", model_path);
        return c;
    }

    uint64_t n_kv = 0;
    if (!read_u64_le(fp, &n_kv)) {
        fclose(fp);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: header truncated at metadata kv count", model_path);
        return c;
    }

    fclose(fp);

    double gib = (double)st.st_size / (1024.0 * 1024.0 * 1024.0);
    snprintf(c.message, sizeof(c.message),
             "%.2f GiB, v%u, %llu tensors, %llu metadata kvs",
             gib, version, (unsigned long long)n_tensors,
             (unsigned long long)n_kv);
    return c;
}

/* Default port ds4-server binds to when --port is not provided. Mirrors
 * ds4_server.c; if the project changes the default, update both. */
#define DOCTOR_DEFAULT_PORT 8000
#define DOCTOR_DEFAULT_HOST "127.0.0.1"

static doctor_check doctor_check_port(void) {
    doctor_check c = {0};
    c.id = "port";
    c.title = "Default server port";
    c.status = DOCTOR_OK;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "cannot create socket: %s", strerror(errno));
        return c;
    }

    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)DOCTOR_DEFAULT_PORT);
    if (inet_pton(AF_INET, DOCTOR_DEFAULT_HOST, &sa.sin_addr) != 1) {
        close(fd);
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "inet_pton failed for %s", DOCTOR_DEFAULT_HOST);
        return c;
    }

    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        int e = errno;
        close(fd);
        if (e == EADDRINUSE) {
            c.status = DOCTOR_WARN;
            snprintf(c.message, sizeof(c.message),
                     "port %d is in use (ds4-server may already be running)",
                     DOCTOR_DEFAULT_PORT);
            return c;
        }
        if (e == EACCES) {
            c.status = DOCTOR_WARN;
            snprintf(c.message, sizeof(c.message),
                     "port %d requires elevated privileges",
                     DOCTOR_DEFAULT_PORT);
            return c;
        }
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "bind to %s:%d failed: %s",
                 DOCTOR_DEFAULT_HOST, DOCTOR_DEFAULT_PORT, strerror(e));
        return c;
    }

    close(fd);
    snprintf(c.message, sizeof(c.message),
             "%s:%d bindable (default for ds4-server)",
             DOCTOR_DEFAULT_HOST, DOCTOR_DEFAULT_PORT);
    return c;
}

#define DOCTOR_MEM_WARN_GIB 8.0
#define DOCTOR_MEM_FAIL_GIB 4.0

static doctor_check doctor_check_memory(void) {
    doctor_check c = {0};
    c.id = "memory";
    c.title = "Total system RAM";
    c.status = DOCTOR_OK;

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) {
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "sysconf(_SC_PHYS_PAGES/_SC_PAGESIZE) unavailable");
        return c;
    }

    double gib = (double)pages * (double)page_size
               / (1024.0 * 1024.0 * 1024.0);

    if (gib < DOCTOR_MEM_FAIL_GIB) {
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%.2f GiB total RAM (below %.0f GiB minimum)",
                 gib, DOCTOR_MEM_FAIL_GIB);
        return c;
    }
    if (gib < DOCTOR_MEM_WARN_GIB) {
        c.status = DOCTOR_WARN;
        snprintf(c.message, sizeof(c.message),
                 "%.2f GiB total RAM (below %.0f GiB recommended)",
                 gib, DOCTOR_MEM_WARN_GIB);
        return c;
    }
    snprintf(c.message, sizeof(c.message),
             "%.2f GiB total RAM", gib);
    return c;
}

static doctor_check doctor_check_kv_disk(const char *kv_disk_dir) {
    doctor_check c = {0};
    c.id = "kv-disk";
    c.title = "KV-disk directory";
    c.status = DOCTOR_OK;

    if (kv_disk_dir == NULL) {
        c.status = DOCTOR_SKIP;
        snprintf(c.message, sizeof(c.message),
                 "no --kv-disk-dir provided");
        return c;
    }

    struct stat st;
    if (stat(kv_disk_dir, &st) != 0) {
        int e = errno;
        c.status = DOCTOR_WARN;
        snprintf(c.message, sizeof(c.message),
                 "%s: %s (server will create it on demand)",
                 kv_disk_dir, strerror(e));
        return c;
    }
    if (!S_ISDIR(st.st_mode)) {
        c.status = DOCTOR_FAIL;
        snprintf(c.message, sizeof(c.message),
                 "%s: exists but is not a directory", kv_disk_dir);
        return c;
    }

    struct statvfs vfs;
    double free_gib = 0.0;
    if (statvfs(kv_disk_dir, &vfs) == 0 && vfs.f_frsize > 0) {
        free_gib = (double)vfs.f_bavail * (double)vfs.f_frsize
                 / (1024.0 * 1024.0 * 1024.0);
    }

    snprintf(c.message, sizeof(c.message),
             "%s: %.2f GiB free", kv_disk_dir, free_gib);
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

    /* U2: backend. U3: model. U4: port, memory, kv-disk.
     * U5 (final JSON renderer + exit-code policy) is still pending. */
    doctor_check checks[5];
    uint64_t t0;

    t0 = now_ms();
    checks[0] = doctor_check_backend();
    checks[0].duration_ms = (int)(now_ms() - t0);

    t0 = now_ms();
    checks[1] = doctor_check_model(cfg.model_path, cfg.timeout_ms);
    checks[1].duration_ms = (int)(now_ms() - t0);

    t0 = now_ms();
    checks[2] = doctor_check_port();
    checks[2].duration_ms = (int)(now_ms() - t0);

    t0 = now_ms();
    checks[3] = doctor_check_memory();
    checks[3].duration_ms = (int)(now_ms() - t0);

    t0 = now_ms();
    checks[4] = doctor_check_kv_disk(cfg.kv_disk_dir);
    checks[4].duration_ms = (int)(now_ms() - t0);

    if (cfg.json_output) {
        /* JSON renderer lands in U5; keep the empty array alive for now. */
        fputs("{\"status\":\"ok\",\"checks\":[]}\n", stdout);
        return 0;
    }
    print_human(checks, sizeof(checks) / sizeof(checks[0]), cfg.color_output);
    puts("ds4-doctor: skeleton — implementation pending in U5");
    return 0;
}
