#ifndef DS4_MEDIA_H
#define DS4_MEDIA_H

/* Media inputs for multimodal requests: data:/file:// URL loading and video frame
 * sampling. Video decoding runs an external ffmpeg (posix_spawn, no shell) so ds4
 * gains no link-time dependency; it is opt-in (ds4-server --video). */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint8_t *data;          /* encoded JPEG */
    size_t len;
    double t_sec;           /* presentation time of the frame */
} ds4_media_frame;

typedef struct {
    double fps;             /* sampling rate; <= 0: 1 fps */
    int max_frames;         /* <= 0: 256 */
    int max_side;           /* longest side of extracted frames in px; <= 0: 1722 */
    double dedupe;          /* drop a frame whose 32x18 gray thumbnail differs from the last
                             * kept frame by less than this mean abs difference (0..255);
                             * < 0: off; 0: default 2.0 */
    double start_sec, duration_sec;   /* optional window; duration <= 0: whole video */
} ds4_media_video_opts;

/* Load the bytes behind a URL. Accepts data:<mime>;base64,... and, when media_root is
 * non-NULL, file:///abs/path whose realpath lies under media_root. mime receives the
 * media type (data:) or one guessed from the extension (file://). */
bool ds4_media_load_url(const char *url, const char *media_root,
                        uint8_t **out, size_t *out_len, char mime[64],
                        char *err, size_t errlen);

/* Sample frames from an encoded video held in memory (written to a private temp file)
 * or from a local path (path != NULL wins). ffmpeg is the executable name or path.
 * Frames are returned in time order; *dropped counts dedupe-skipped frames. */
bool ds4_media_video_frames(const char *ffmpeg, const char *path,
                            const uint8_t *data, size_t len,
                            const ds4_media_video_opts *opts,
                            ds4_media_frame **frames, int *count, int *dropped,
                            double *duration_sec, char *err, size_t errlen);

void ds4_media_frames_free(ds4_media_frame *frames, int count);

/* file:///abs/path -> realpath, which must lie under media_root. */
#include <limits.h>
bool ds4_media_resolve_file_url(const char *url, const char *media_root, char out[PATH_MAX],
                                char *err, size_t errlen);

/* True when `ffmpeg` can be executed (cached after the first probe). */
bool ds4_media_ffmpeg_available(const char *ffmpeg);

/* PNG or JPEG magic: the formats the image decoder reads natively. */
bool ds4_media_is_png_or_jpeg(const uint8_t *data, size_t len);

/* Convert any ffmpeg-decodable still image (WebP, GIF, BMP, TIFF, HEIC, ...)
 * to PNG; the first frame of an animation. Caller frees *out. */
bool ds4_media_image_to_png(const char *ffmpeg, const uint8_t *data, size_t len,
                            uint8_t **out, size_t *out_len, char *err, size_t errlen);

#endif
