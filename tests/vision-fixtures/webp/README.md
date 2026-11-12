# WebP decoder fixtures

Small WebP files with the PNG that libwebp's `dwebp` decodes each one to.
`tests/test_image_decode` checks that `ds4_webp.h` produces exactly the same
pixels as the reference PNG, alpha included, then cuts each bitstream short
and corrupts the container to check that bad files are rejected (and that a
still image carrying an `ANIM` chunk is still decoded).

Generated with libwebp 1.6.0 (`cwebp`, `dwebp`) from synthetic images. `cwebp`
leaves small alpha planes unfiltered, so the script derives the vertical and
gradient variants by filtering the raw alpha plane itself; `dwebp` still
produces their reference PNGs. Re-run `./generate.sh` if you have libwebp on
`PATH`.

| File | Size | What it covers |
| --- | --- | --- |
| `lossy.webp` | 61x37 | VP8 with the normal loop filter, 16x16 and 4x4 intra modes, odd size. |
| `lossy_simple_filter.webp` | 61x37 | VP8 with the simple loop filter, sharpness 3 and two segments. |
| `lossy_alpha.webp` | 33x21 | Extended format: `VP8X`, lossless `ALPH` chunk with the horizontal filter, then `VP8 `. |
| `lossy_alpha_raw.webp` | 33x21 | Uncompressed `ALPH`, no filter. |
| `lossy_alpha_vertical.webp` | 33x21 | Uncompressed `ALPH` with the vertical filter. |
| `lossy_alpha_gradient.webp` | 33x21 | Uncompressed `ALPH` with the gradient filter. |
| `lossless.webp` | 61x37 | VP8L with the predictor and cross-color transforms. |
| `lossless_subtract_green.webp` | 61x37 | VP8L with the subtract-green and predictor transforms. |
| `lossless_palette.webp` | 23x11 | VP8L color indexing with four colors bundled per pixel. |
| `lossless_alpha.webp` | 33x21 | VP8L with alpha, encoded with `-exact` so hidden RGB is kept. |

`make test` runs them through `tests/test_image_decode`.
