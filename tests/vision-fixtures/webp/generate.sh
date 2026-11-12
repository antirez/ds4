#!/bin/sh
# Recreate the WebP fixtures and their reference PNGs with libwebp's cwebp
# and dwebp. Each *.ref.png is what dwebp decodes the matching *.webp to.
set -e
cd "$(dirname "$0")"
for tool in cwebp dwebp python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "generate.sh: $tool not found (install libwebp)" >&2
        exit 1
    fi
done
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

python3 - "$tmp" << 'PY'
import sys
from pathlib import Path
out = Path(sys.argv[1])

def lcg(x, y, salt):
    return ((x * 1103515245 + y * 12345 + salt) & 0xFFFFFFFF) >> 16

def photo(x, y):
    # Smooth gradients, a hard diagonal edge and some noise, so the encoder
    # uses both 16x16 and 4x4 intra modes.
    n = lcg(x, y, 42) & 31
    r = (x * 4 + n) & 255
    g = (y * 6 + (x * y >> 3)) & 255
    b = 230 if x > y + 5 else (x * 3 + y * 2 + n) & 255
    return r, g, b

def palette(x, y):
    colors = [(255, 0, 0), (0, 128, 255), (20, 20, 20), (250, 240, 200)]
    return colors[((x // 3) + (y // 2)) % 4]

def write_ppm(name, w, h, fn):
    data = bytearray()
    for y in range(h):
        for x in range(w):
            data.extend(fn(x, y))
    (out / name).write_bytes(f"P6\n{w} {h}\n255\n".encode() + bytes(data))

def write_pam(name, w, h, fn):
    data = bytearray()
    for y in range(h):
        for x in range(w):
            r, g, b = fn(x, y)
            a = 0 if (x - w // 2) ** 2 + (y - h // 2) ** 2 < 60 else (x * 9 + y * 5) & 255
            data.extend((r, g, b, a))
    header = f"P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n"
    (out / name).write_bytes(header.encode() + bytes(data))

write_ppm("photo.ppm", 61, 37, photo)
write_ppm("palette.ppm", 23, 11, palette)
write_pam("alpha.pam", 33, 21, photo)
PY

enc() { out=$1; shift; cwebp -quiet "$@" -o "$out.webp"; dwebp -quiet "$out.webp" -o "$out.ref.png"; }
enc lossy -q 60 "$tmp/photo.ppm"
enc lossy_simple_filter -q 30 -nostrong -sharpness 3 -segments 2 "$tmp/photo.ppm"
enc lossy_alpha -q 70 -alpha_filter best "$tmp/alpha.pam"
enc lossy_alpha_raw -q 70 -alpha_method 0 -alpha_filter fast "$tmp/alpha.pam"
enc lossless -lossless -z 9 "$tmp/photo.ppm"
enc lossless_subtract_green -lossless -z 0 "$tmp/photo.ppm"
enc lossless_palette -lossless "$tmp/palette.ppm"
enc lossless_alpha -lossless -exact "$tmp/alpha.pam"

# cwebp keeps small alpha planes unfiltered, so derive the vertical and
# gradient filtered variants from the raw one by filtering its ALPH payload
# (RFC 9649 Section 2.7.1.2). dwebp still decodes the reference PNGs.
python3 - << 'PY'
from pathlib import Path

src = Path("lossy_alpha_raw.webp").read_bytes()
at = src.index(b"ALPH")
size = int.from_bytes(src[at + 4:at + 8], "little")
header = at + 8
assert src[header] & 3 == 0 and (src[header] >> 2) & 3 == 0, "expected raw, unfiltered alpha"
w, h = 33, 21
alpha = src[header + 1:header + 1 + w * h]
assert size == 1 + w * h

def a(x, y):
    return alpha[y * w + x]

def predict(kind, x, y):
    if x == 0 and y == 0:
        return 0
    if y == 0:
        return a(x - 1, 0)          # first row: left
    if x == 0:
        return a(0, y - 1)          # first column: above
    if kind == "vertical":
        return a(x, y - 1)
    g = a(x - 1, y) + a(x, y - 1) - a(x - 1, y - 1)
    return min(255, max(0, g))      # gradient

for kind, code in (("vertical", 2), ("gradient", 3)):
    deltas = bytes((a(x, y) - predict(kind, x, y)) & 255 for y in range(h) for x in range(w))
    out = bytearray(src)
    out[header] = code << 2
    out[header + 1:header + 1 + w * h] = deltas
    Path(f"lossy_alpha_{kind}.webp").write_bytes(bytes(out))
PY
for kind in vertical gradient; do
    dwebp -quiet "lossy_alpha_$kind.webp" -o "lossy_alpha_$kind.ref.png"
done
echo "wrote $(ls *.webp | wc -l | tr -d ' ') WebP fixtures with reference PNGs"
