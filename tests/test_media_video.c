/* test_media_video VIDEO [FPS] — sample frames with ds4_media_video_frames and report. */
#include <stdio.h>
#include <stdlib.h>
#include "../ds4_media.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s VIDEO [FPS]\n", argv[0]); return 2; }
    ds4_media_video_opts o = {0};
    o.fps = argc > 2 ? atof(argv[2]) : 1.0;
    ds4_media_frame *f = NULL;
    int n = 0, dropped = 0;
    double dur = 0;
    char err[256] = {0};
    if (!ds4_media_video_frames("ffmpeg", argv[1], NULL, 0, &o, &f, &n, &dropped, &dur, err, sizeof err)) {
        printf("FAIL: %s\n", err);
        return 1;
    }
    printf("kept %d frames, dropped %d, duration %.1f s:", n, dropped, dur);
    for (int i = 0; i < n; i++) printf(" t=%.1f(%zuB)", f[i].t_sec, f[i].len);
    printf("\n");
    o.dedupe = -1;
    ds4_media_frames_free(f, n);
    if (!ds4_media_video_frames("ffmpeg", argv[1], NULL, 0, &o, &f, &n, &dropped, &dur, err, sizeof err)) return 1;
    printf("dedupe off: %d frames\n", n);
    ds4_media_frames_free(f, n);
    return 0;
}
