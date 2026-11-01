/* Media inputs: data:/file:// URL loading and ffmpeg-based video frame sampling.
 * See ds4_media.h. */
#include "ds4_media.h"
#include "ds4_image.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define DS4_MEDIA_MAX_IMAGE_BYTES (512u * 1024u * 1024u)
#define DS4_MEDIA_MAX_INLINE_VIDEO (1024u * 1024u * 1024u)

static void media_err(char *err, size_t errlen, const char *fmt, const char *arg) {
    if (err && errlen) snprintf(err, errlen, fmt, arg ? arg : "");
}

static int b64v(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return 26 + c - 'a';
    if (c >= '0' && c <= '9') return 52 + c - '0';
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static bool b64_decode(const char *s, uint8_t **out, size_t *out_len, size_t max_bytes) {
    const size_t n = strlen(s);
    if (n / 4 * 3 > max_bytes + 3) return false;
    uint8_t *buf = malloc(n / 4 * 3 + 3);
    if (!buf) return false;
    size_t o = 0;
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '=') break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int v = b64v(c);
        if (v < 0) { free(buf); return false; }
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[o++] = (uint8_t)(acc >> bits);
        }
    }
    *out = buf;
    *out_len = o;
    return o > 0;
}

static const char *mime_from_ext(const char *path) {
    const char *dot = strrchr(path, '.');
    if (!dot) return "application/octet-stream";
    if (!strcasecmp(dot, ".png")) return "image/png";
    if (!strcasecmp(dot, ".jpg") || !strcasecmp(dot, ".jpeg")) return "image/jpeg";
    if (!strcasecmp(dot, ".mp4") || !strcasecmp(dot, ".m4v")) return "video/mp4";
    if (!strcasecmp(dot, ".mov")) return "video/quicktime";
    if (!strcasecmp(dot, ".webm")) return "video/webm";
    if (!strcasecmp(dot, ".mkv")) return "video/x-matroska";
    if (!strcasecmp(dot, ".avi")) return "video/x-msvideo";
    if (!strcasecmp(dot, ".gif")) return "image/gif";
    return "application/octet-stream";
}

/* file:///abs/path -> realpath under media_root. */
bool ds4_media_resolve_file_url(const char *url, const char *media_root, char out[PATH_MAX],
                             char *err, size_t errlen) {
    if (!media_root || !media_root[0]) {
        media_err(err, errlen, "file:// URLs need ds4-server --media-root DIR%s", NULL);
        return false;
    }
    const char *p = url + 7;
    if (*p != '/') {
        media_err(err, errlen, "file:// URL must be absolute: %s", url);
        return false;
    }
    char root[PATH_MAX];
    if (!realpath(media_root, root) || !realpath(p, out)) {
        media_err(err, errlen, "cannot resolve %s", p);
        return false;
    }
    const size_t rl = strlen(root);
    if (strncmp(out, root, rl) != 0 || (out[rl] != '/' && out[rl] != '\0')) {
        media_err(err, errlen, "%s is outside --media-root", p);
        return false;
    }
    return true;
}

static bool read_file(const char *path, uint8_t **out, size_t *out_len, size_t max_bytes,
                      char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { media_err(err, errlen, "cannot open %s", path); return false; }
    struct stat st;
    if (fstat(fileno(fp), &st) != 0 || st.st_size <= 0 || (uint64_t)st.st_size > max_bytes) {
        fclose(fp);
        media_err(err, errlen, "%s is empty or too large", path);
        return false;
    }
    uint8_t *buf = malloc((size_t)st.st_size);
    const bool ok = buf && fread(buf, 1, (size_t)st.st_size, fp) == (size_t)st.st_size;
    fclose(fp);
    if (!ok) { free(buf); media_err(err, errlen, "cannot read %s", path); return false; }
    *out = buf;
    *out_len = (size_t)st.st_size;
    return true;
}

bool ds4_media_load_url(const char *url, const char *media_root,
                        uint8_t **out, size_t *out_len, char mime[64],
                        char *err, size_t errlen) {
    *out = NULL;
    *out_len = 0;
    mime[0] = '\0';
    if (!url) { media_err(err, errlen, "missing media URL%s", NULL); return false; }
    if (!strncmp(url, "data:", 5)) {
        const char *semi = strchr(url, ';');
        const char *comma = strchr(url, ',');
        if (!semi || !comma || semi > comma || strncmp(semi, ";base64,", 8) != 0) {
            media_err(err, errlen, "unsupported data URI (base64 required)%s", NULL);
            return false;
        }
        const size_t ml = (size_t)(semi - (url + 5));
        snprintf(mime, 64, "%.*s", (int)(ml < 63 ? ml : 63), url + 5);
        const size_t cap = !strncmp(mime, "video/", 6) ? DS4_MEDIA_MAX_INLINE_VIDEO
                                                       : DS4_MEDIA_MAX_IMAGE_BYTES;
        if (!b64_decode(comma + 1, out, out_len, cap)) {
            media_err(err, errlen, "invalid or oversized base64 payload%s", NULL);
            return false;
        }
        return true;
    }
    if (!strncmp(url, "file://", 7)) {
        char path[PATH_MAX];
        if (!ds4_media_resolve_file_url(url, media_root, path, err, errlen)) return false;
        snprintf(mime, 64, "%s", mime_from_ext(path));
        return read_file(path, out, out_len,
                         !strncmp(mime, "video/", 6) ? DS4_MEDIA_MAX_INLINE_VIDEO
                                                     : DS4_MEDIA_MAX_IMAGE_BYTES,
                         err, errlen);
    }
    media_err(err, errlen, "unsupported media URL scheme: %.32s", url);
    return false;
}

static bool run_quiet(const char *exe, char *const argv[], const char *log_path) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, log_path ? log_path : "/dev/null",
                                     O_WRONLY | O_CREAT | O_TRUNC, 0600);
    pid_t pid;
    const int rc = posix_spawnp(&pid, exe, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return false;
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

/* Container duration in seconds via the ffprobe beside ffmpeg; 0 if unknown. */
static double media_probe_duration(const char *ffmpeg, const char *input, const char *dir) {
    char probe[PATH_MAX], outp[PATH_MAX];
    const char *slash = strrchr(ffmpeg, '/');
    if (slash) snprintf(probe, sizeof probe, "%.*sffprobe", (int)(slash - ffmpeg + 1), ffmpeg);
    else snprintf(probe, sizeof probe, "ffprobe");
    snprintf(outp, sizeof outp, "%s/duration.txt", dir);
    char *const argv[] = {probe, "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", (char *)input, NULL};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, outp, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid;
    const int rc = posix_spawnp(&pid, probe, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    double d = 0;
    if (rc == 0) {
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        FILE *f = fopen(outp, "r");
        if (f) {
            if (fscanf(f, "%lf", &d) != 1 || !(d > 0)) d = 0;
            fclose(f);
        }
    }
    unlink(outp);
    return d;
}

bool ds4_media_ffmpeg_available(const char *ffmpeg) {
    static int cached = -1;
    if (cached < 0) {
        char *argv[] = { (char *)ffmpeg, "-hide_banner", "-version", NULL };
        cached = run_quiet(ffmpeg, argv, NULL) ? 1 : 0;
    }
    return cached == 1;
}

/* 32x18 gray thumbnail of a decoded frame (box average). */
static void thumb32x18(const ds4_image *img, uint8_t t[32 * 18]) {
    for (int ty = 0; ty < 18; ty++) {
        for (int tx = 0; tx < 32; tx++) {
            const uint32_t x0 = (uint32_t)((uint64_t)tx * img->width / 32);
            const uint32_t x1 = (uint32_t)((uint64_t)(tx + 1) * img->width / 32);
            const uint32_t y0 = (uint32_t)((uint64_t)ty * img->height / 18);
            const uint32_t y1 = (uint32_t)((uint64_t)(ty + 1) * img->height / 18);
            uint64_t sum = 0, n = 0;
            for (uint32_t y = y0; y < y1; y += 2) {
                for (uint32_t x = x0; x < x1; x += 2) {
                    const uint8_t *px = img->rgb + ((uint64_t)y * img->width + x) * 3u;
                    sum += (uint32_t)px[0] * 77u + (uint32_t)px[1] * 150u + (uint32_t)px[2] * 29u;
                    n++;
                }
            }
            t[ty * 32 + tx] = n ? (uint8_t)(sum / n / 256u) : 0;
        }
    }
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

bool ds4_media_video_frames(const char *ffmpeg, const char *path,
                            const uint8_t *data, size_t len,
                            const ds4_media_video_opts *opts,
                            ds4_media_frame **frames, int *count, int *dropped,
                            double *duration_sec, char *err, size_t errlen) {
    *frames = NULL;
    *count = 0;
    if (dropped) *dropped = 0;
    if (duration_sec) *duration_sec = 0;
    double fps = opts && opts->fps > 0 ? opts->fps : 1.0;
    const int max_frames = opts && opts->max_frames > 0 ? opts->max_frames : 256;
    const int max_side = opts && opts->max_side > 0 ? opts->max_side : 1722;
    const double dedupe = !opts || opts->dedupe == 0 ? 2.0 : opts->dedupe;
    if (!ffmpeg || !ffmpeg[0]) ffmpeg = "ffmpeg";
    if (!ds4_media_ffmpeg_available(ffmpeg)) {
        media_err(err, errlen, "video input needs ffmpeg (%s not runnable)", ffmpeg);
        return false;
    }
    char dir[] = "/tmp/ds4-video-XXXXXX";
    if (!mkdtemp(dir)) { media_err(err, errlen, "mkdtemp failed%s", NULL); return false; }
    char input[PATH_MAX], logp[PATH_MAX], pattern[PATH_MAX];
    snprintf(logp, sizeof logp, "%s/ffmpeg.log", dir);
    snprintf(pattern, sizeof pattern, "%s/f_%%06d.jpg", dir);
    bool ok = true;
    if (path) {
        snprintf(input, sizeof input, "%s", path);
    } else {
        snprintf(input, sizeof input, "%s/input.bin", dir);
        FILE *fp = fopen(input, "wb");
        ok = fp && fwrite(data, 1, len, fp) == len;
        if (fp) fclose(fp);
        if (!ok) media_err(err, errlen, "cannot stage the video%s", NULL);
    }
    /* Cover the whole clip: when duration * fps would exceed max_frames, lower
     * the rate so the frames spread over the full video instead of stopping
     * after its first max_frames / fps seconds. */
    if (ok) {
        double span = opts && opts->duration_sec > 0 ? opts->duration_sec
                                                     : media_probe_duration(ffmpeg, input, dir);
        if (opts && opts->start_sec > 0 && !(opts->duration_sec > 0)) span -= opts->start_sec;
        if (span > 0 && span * fps > max_frames) fps = max_frames / span;
    }
    char vf[256], nframes[32], ss[32], tt[32];
    snprintf(vf, sizeof vf,
             "fps=%.6g,scale=w='min(iw,%d)':h='min(ih,%d)':force_original_aspect_ratio=decrease",
             fps, max_side, max_side);
    snprintf(nframes, sizeof nframes, "%d", max_frames);
    char *argv[24];
    int a = 0;
    argv[a++] = (char *)ffmpeg; argv[a++] = "-nostdin"; argv[a++] = "-hide_banner";
    argv[a++] = "-loglevel"; argv[a++] = "error";
    if (opts && opts->start_sec > 0) {
        snprintf(ss, sizeof ss, "%.3f", opts->start_sec);
        argv[a++] = "-ss"; argv[a++] = ss;
    }
    argv[a++] = "-i"; argv[a++] = input;
    if (opts && opts->duration_sec > 0) {
        snprintf(tt, sizeof tt, "%.3f", opts->duration_sec);
        argv[a++] = "-t"; argv[a++] = tt;
    }
    argv[a++] = "-vf"; argv[a++] = vf;
    argv[a++] = "-frames:v"; argv[a++] = nframes;
    argv[a++] = "-q:v"; argv[a++] = "3";
    argv[a++] = "-f"; argv[a++] = "image2";
    argv[a++] = pattern;
    argv[a] = NULL;
    if (ok && !run_quiet(ffmpeg, argv, logp)) {
        char msg[200] = "ffmpeg failed";
        FILE *lf = fopen(logp, "r");
        if (lf) {
            size_t n = fread(msg, 1, sizeof msg - 1, lf);
            msg[n] = '\0';
            char *nl = strchr(msg, '\n');
            if (nl) *nl = '\0';
            fclose(lf);
        }
        media_err(err, errlen, "ffmpeg: %s", msg);
        ok = false;
    }

    char **names = NULL;
    int nnames = 0, cap = 0;
    DIR *d = ok ? opendir(dir) : NULL;
    if (d) {
        struct dirent *de;
        while ((de = readdir(d))) {
            if (strncmp(de->d_name, "f_", 2) != 0) continue;
            if (nnames == cap) {
                cap = cap ? cap * 2 : 64;
                names = realloc(names, (size_t)cap * sizeof(*names));
            }
            names[nnames++] = strdup(de->d_name);
        }
        closedir(d);
        qsort(names, (size_t)nnames, sizeof(*names), cmp_str);
    }
    ds4_media_frame *out = nnames ? calloc((size_t)nnames, sizeof(*out)) : NULL;
    uint8_t last[32 * 18];
    bool have_last = false;
    int kept = 0, skipped = 0;
    for (int i = 0; ok && i < nnames; i++) {
        char fp[PATH_MAX];
        snprintf(fp, sizeof fp, "%s/%s", dir, names[i]);
        uint8_t *buf = NULL;
        size_t bl = 0;
        if (!read_file(fp, &buf, &bl, DS4_MEDIA_MAX_IMAGE_BYTES, NULL, 0)) continue;
        bool keep = true;
        if (dedupe > 0) {
            ds4_image img = {0};
            char e2[128];
            if (ds4_image_decode_memory(&img, buf, bl, e2, sizeof e2)) {
                uint8_t th[32 * 18];
                thumb32x18(&img, th);
                if (have_last && i != nnames - 1) {
                    uint64_t diff = 0;
                    for (int j = 0; j < 32 * 18; j++) diff += (uint64_t)abs((int)th[j] - (int)last[j]);
                    if ((double)diff / (32.0 * 18.0) < dedupe) keep = false;
                }
                if (keep) { memcpy(last, th, sizeof last); have_last = true; }
                ds4_image_free(&img);
            }
        }
        if (!keep) { free(buf); skipped++; continue; }
        out[kept].data = buf;
        out[kept].len = bl;
        out[kept].t_sec = (opts && opts->start_sec > 0 ? opts->start_sec : 0) + (double)i / fps;
        kept++;
    }
    for (int i = 0; i < nnames; i++) {
        char fp[PATH_MAX];
        snprintf(fp, sizeof fp, "%s/%s", dir, names[i]);
        unlink(fp);
        free(names[i]);
    }
    free(names);
    if (!path) unlink(input);
    unlink(logp);
    rmdir(dir);
    if (ok && kept == 0) {
        media_err(err, errlen, "no frames could be extracted from the video%s", NULL);
        ok = false;
    }
    if (!ok) {
        ds4_media_frames_free(out, kept);
        return false;
    }
    *frames = out;
    *count = kept;
    if (dropped) *dropped = skipped;
    if (duration_sec) *duration_sec = (double)nnames / fps;
    return true;
}

bool ds4_media_is_png_or_jpeg(const uint8_t *data, size_t len) {
    return (len >= 8 && !memcmp(data, "\x89PNG\r\n\x1a\n", 8)) ||
           (len >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff);
}

bool ds4_media_image_to_png(const char *ffmpeg, const uint8_t *data, size_t len,
                            uint8_t **out, size_t *out_len, char *err, size_t errlen) {
    *out = NULL;
    *out_len = 0;
    if (!ffmpeg || !ffmpeg[0]) ffmpeg = "ffmpeg";
    if (!ds4_media_ffmpeg_available(ffmpeg)) {
        media_err(err, errlen, "image is not PNG/JPEG and ffmpeg (%s) is not runnable to convert it", ffmpeg);
        return false;
    }
    char dir[] = "/tmp/ds4-image-XXXXXX";
    if (!mkdtemp(dir)) { media_err(err, errlen, "mkdtemp failed%s", NULL); return false; }
    char in[PATH_MAX], outp[PATH_MAX];
    snprintf(in, sizeof in, "%s/in.bin", dir);
    snprintf(outp, sizeof outp, "%s/out.png", dir);
    FILE *fp = fopen(in, "wb");
    bool ok = fp && fwrite(data, 1, len, fp) == len;
    if (fp) ok = fclose(fp) == 0 && ok;
    if (ok) {
        char *const argv[] = {(char *)ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                              "-i", in, "-frames:v", "1", outp, NULL};
        ok = run_quiet(ffmpeg, argv, NULL) && read_file(outp, out, out_len, DS4_MEDIA_MAX_IMAGE_BYTES, err, errlen);
        if (!ok && err && !err[0]) media_err(err, errlen, "ffmpeg could not decode the image%s", NULL);
    } else {
        media_err(err, errlen, "cannot stage the image%s", NULL);
    }
    unlink(in);
    unlink(outp);
    rmdir(dir);
    return ok;
}

void ds4_media_frames_free(ds4_media_frame *frames, int count) {
    if (!frames) return;
    for (int i = 0; i < count; i++) free(frames[i].data);
    free(frames);
}
