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
 * Real checks land in U2 (backend), U3 (model), U4 (environment).
 * JSON / human output and exit-code policy land in U5.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv) {
    doctor_config cfg = parse_options(argc, argv);
    (void)cfg;
    /* Skeleton — real checks land in U2..U5. */
    if (cfg.json_output) {
        fputs("{\"status\":\"ok\",\"checks\":[]}\n", stdout);
        return 0;
    }
    puts("ds4-doctor: skeleton — implementation pending in U2..U5");
    return 0;
}
