# JPEG decoder fixtures

The two `prog_ac_refine_zrl_*` images exercise successive-approximation AC refinement (`Ss=1, Se=63,
Ah=2, Al=1`) with restart markers. The unpatched Iris decoder mishandles ZRL
(`run=15, size=0`) in that scan: it keeps consuming correction bits after the
16th currently-zero coefficient.

Generated with libjpeg-turbo 3.2.0 `cjpeg`. Re-run `./generate.sh` if you have
`cjpeg` on `PATH`.

The `*_luma_subsampled` fixtures were added later with libjpeg-turbo 2.1.5
`cjpeg`, which regenerates the two ZRL files byte for byte. Their luma is sampled
below Cb (`-sample 1x1,2x2,1x1`), so the Y plane is smaller than the image.

| File | Size | Why it is here |
| --- | --- | --- |
| `prog_ac_refine_zrl_gray.jpg` | 24x16 grayscale | Unpatched decode succeeds but pixels differ from libjpeg-turbo. Patched decode is bit-exact with `djpeg`. |
| `prog_ac_refine_zrl_420.jpg` | 64x48 YCbCr 4:2:0 | Unpatched `jpeg_load` returns NULL. Patched decode is bit-exact with `djpeg`. |
| `base_luma_subsampled.jpg` | 40x24 YCbCr baseline, sampled 1x1,2x2,1x1 | Unpatched decode reads the 24x16 Y plane as if it were 40x24, past its end (AddressSanitizer: heap-buffer-overflow). Patched decode is bit-exact with `djpeg`. |
| `prog_luma_subsampled.jpg` | 40x24 YCbCr progressive, sampled 1x1,2x2,1x1 | Same read past the Y plane on the progressive path. Patched decode is bit-exact with `djpeg`. |

`make test` covers all of them through `tests/test_image_decode`.
