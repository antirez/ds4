#include "ds4.h"
#include "ds4_image.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double wall_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s MAIN.gguf VISION.gguf IMAGE OUTPUT.f32 [PATCHES.f32]\n", argv[0]);
        return 2;
    }
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.vision_path = argv[2];
#ifdef __APPLE__
    options.backend = DS4_BACKEND_METAL;
#else
    options.backend = DS4_BACKEND_CUDA;
#endif
    options.inspect_only = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    char error[256] = {0};
    ds4_vision_embedding embedding = {0};
    unsigned repeats = 1u;
    const char *repeat_env = getenv("DS4_TEST_VISION_REPEATS");
    if (repeat_env && repeat_env[0]) {
        char *end = NULL;
        unsigned long parsed = strtoul(repeat_env, &end, 10);
        if (!end || *end != '\0' || parsed == 0ul || parsed > 100ul) {
            fprintf(stderr, "invalid DS4_TEST_VISION_REPEATS: %s\n",
                    repeat_env);
            ds4_engine_close(engine);
            return 2;
        }
        repeats = (unsigned)parsed;
    }
    for (unsigned i = 0; i < repeats; i++) {
        ds4_vision_embedding next = {0};
        const double start = wall_seconds();
        if (!ds4_engine_vision_encode_file(engine, argv[3], &next,
                                           error, sizeof(error))) {
            fprintf(stderr, "vision encode failed: %s\n", error);
            ds4_vision_embedding_free(&embedding);
            ds4_engine_close(engine);
            return 1;
        }
        if (repeats > 1u) {
            fprintf(stderr, "vision encode %u/%u: %.3f s\n",
                    i + 1u, repeats, wall_seconds() - start);
        }
        ds4_vision_embedding_free(&embedding);
        embedding = next;
    }
    FILE *fp = fopen(argv[4], "wb");
    if (!fp) {
        fprintf(stderr, "cannot open %s: %s\n", argv[4], strerror(errno));
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    size_t values = (size_t)embedding.token_count * 4096u;
    if (fwrite(embedding.data, sizeof(float), values, fp) != values ||
        fclose(fp) != 0) {
        fprintf(stderr, "cannot write %s\n", argv[4]);
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    printf("%ux%u -> %ux%u, %u image tokens\n",
           embedding.width, embedding.height,
           embedding.content_width, embedding.content_height,
           embedding.token_count);
    if (argc == 6) {
        ds4_image image = {0};
        ds4_image_patches patches = {0};
        if (!ds4_image_decode_file(&image, argv[3], error, sizeof(error)) ||
            !ds4_image_preprocess_glm53(&patches, &image, 16u, 8000u,
                                        error, sizeof(error))) {
            fprintf(stderr, "patch dump failed: %s\n", error);
            ds4_image_free(&image);
            ds4_vision_embedding_free(&embedding);
            ds4_engine_close(engine);
            return 1;
        }
        fp = fopen(argv[5], "wb");
        values = (size_t)patches.patch_count * 1176u;
        if (!fp || fwrite(patches.patches, sizeof(float), values, fp) != values ||
            fclose(fp) != 0) {
            fprintf(stderr, "cannot write %s\n", argv[5]);
            if (fp) fclose(fp);
            ds4_image_patches_free(&patches);
            ds4_image_free(&image);
            ds4_vision_embedding_free(&embedding);
            ds4_engine_close(engine);
            return 1;
        }
        ds4_image_patches_free(&patches);
        ds4_image_free(&image);
    }
    ds4_vision_embedding_free(&embedding);
    ds4_engine_close(engine);
    return 0;
}
