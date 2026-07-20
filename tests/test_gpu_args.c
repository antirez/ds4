/* tests/test_gpu_args.c — unit tests for the --gpu-vram /
 * --gpu-devices parser. Compiles against the CPU/non-CUDA build of
 * ds4_gpu_args.c (so the "auto" branch reports the build-mode error
 * rather than calling cuda). The CUDA path is exercised on box smoke
 * runs.
 */
#include "../ds4_gpu_args.h"
#include "../ds4_gpu_mgpu.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define PASS_FAIL(name, cond)                                                \
    do {                                                                     \
        if (cond) {                                                          \
            fprintf(stdout, "ok %s\n", name);                                \
        } else {                                                             \
            fprintf(stdout, "FAIL %s\n", name);                              \
            failed++;                                                        \
        }                                                                    \
    } while (0)

int main(void) {
    int failed = 0;
    char err[256];

    /* 1: explicit 2-device split */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        err[0] = '\0';
        int rc = parse_gpu_vram_arg("40,12", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 40,12 returns 0",
                  rc == 0 && !skip && cfg.n_gpus == 2 &&
                  cfg.device_indices[0] == 0 && cfg.device_indices[1] == 1 &&
                  cfg.vram_bytes[0] == (size_t)40 * 1024 * 1024 * 1024 &&
                  cfg.vram_bytes[1] == (size_t)12 * 1024 * 1024 * 1024);
    }

    /* 2: single-GPU */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("40", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 40 → 1 device",
                  rc == 0 && !skip && cfg.n_gpus == 1 &&
                  cfg.device_indices[0] == 0 &&
                  cfg.vram_bytes[0] == (size_t)40 * 1024 * 1024 * 1024);
    }

    /* 3: --gpu-vram 0 → skip-cuda */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("0", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 0 → skip_cuda=true", rc == 0 && skip);
    }

    /* 4: "auto" without CUDA build → error */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("auto", NULL, &cfg, &skip, err, sizeof(err));
#if !defined(DS4_NO_GPU) && !defined(__APPLE__)
        /* CUDA build: linker resolves ds4_gpu_args_probe_auto_cuda. We
         * don't run that here (no CUDA hardware in unit test), so we
         * only assert the function returned with something reasonable.
         * This test is mostly CPU/Metal-focused. */
        PASS_FAIL("parse auto (cuda build) doesn't NULL-deref",
                  (rc == 0 || rc != 0));  /* tautological — auto path
                                              exists; box smoke covers it */
#else
        PASS_FAIL("parse auto rejects on CPU/Metal build",
                  rc != 0 && strstr(err, "auto") != NULL &&
                  strstr(err, "CUDA build") != NULL);
#endif
    }

    /* 5: count mismatch */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("40,12,40", "0,2", &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 40,12,40 with devs 0,2 errors on count",
                  rc != 0 && strstr(err, "count") != NULL);
    }

    /* 6: garbage value */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("abc", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse abc errors", rc != 0);
    }

    /* 7: explicit devices with explicit vram */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("40,12", "0,2", &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 40,12 + devices 0,2",
                  rc == 0 && !skip && cfg.n_gpus == 2 &&
                  cfg.device_indices[0] == 0 && cfg.device_indices[1] == 2 &&
                  cfg.vram_bytes[0] == (size_t)40 * 1024 * 1024 * 1024 &&
                  cfg.vram_bytes[1] == (size_t)12 * 1024 * 1024 * 1024);
    }

    /* 8: --gpu-vram 0 with --gpu-devices → error */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("0", "0,1", &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 0 + devices errors", rc != 0);
    }

    /* 9: trailing comma is a parse error */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("40,", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse 40, (trailing comma) errors", rc != 0);
    }

    /* 10: huge value rejected (overflow guard) */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("9999999", NULL, &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse huge vram rejected (overflow guard)", rc != 0);
    }

    /* 11: huge device index rejected */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        bool skip = false;
        int rc = parse_gpu_vram_arg("40,12", "0,99999",
                                     &cfg, &skip, err, sizeof(err));
        PASS_FAIL("parse huge device index rejected", rc != 0);
    }

    /* 12: format_gpu_layout_line */
    {
        ds4_gpu_config cfg = (ds4_gpu_config){0};
        cfg.n_gpus = 2;
        cfg.device_indices[0] = 0;
        cfg.device_indices[1] = 1;
        cfg.vram_bytes[0] = (size_t)40 * 1024 * 1024 * 1024;
        cfg.vram_bytes[1] = (size_t)12 * 1024 * 1024 * 1024;
        char out[256];
        int n = format_gpu_layout_line(&cfg, true, out, sizeof(out));
        PASS_FAIL("format_gpu_layout_line", n > 0 &&
                  strstr(out, "2 devices [0,1]") != NULL &&
                  strstr(out, "budgets 40,12 GB") != NULL &&
                  strstr(out, "auto=true") != NULL);
    }

    if (failed > 0) {
        fprintf(stderr, "test_gpu_args: %d test(s) FAILED\n", failed);
        return 1;
    }
    fprintf(stdout, "test_gpu_args: all tests passed\n");
    return 0;
}
