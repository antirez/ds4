#!/bin/sh
# Recreate the progressive and sampling-factor JPEG fixtures with libjpeg-turbo cjpeg.
set -e
cd "$(dirname "$0")"
if ! command -v cjpeg >/dev/null 2>&1; then
    echo "generate.sh: cjpeg not found (install libjpeg-turbo)" >&2
    exit 1
fi

python3 - << 'PY'
from pathlib import Path

def write_pgm(path, w, h):
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            v = (x * 1103515245 + y * 12345 + 42) & 0xFFFFFFFF
            row.append((v >> 16) & 255)
        rows.append(bytes(row))
    Path(path).write_bytes(f"P5\n{w} {h}\n255\n".encode() + b"".join(rows))

def write_ppm(path, w, h):
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            v = (x * 1103515245 + y * 12345 + 42) & 0xFFFFFFFF
            row.extend(((v >> 16) & 255, (v >> 8) & 255, v & 255))
        rows.append(bytes(row))
    Path(path).write_bytes(f"P6\n{w} {h}\n255\n".encode() + b"".join(rows))

write_pgm("/tmp/ds4_zrl_gray.pgm", 24, 16)
write_ppm("/tmp/ds4_zrl_420.ppm", 64, 48)
write_ppm("/tmp/ds4_dc_420.ppm", 48, 32)
write_pgm("/tmp/ds4_gray22.pgm", 40, 24)
PY

cat > /tmp/ds4_zrl_gray.scans << 'EOF'
0: 0-0, 0, 1 ;
0: 1-63, 0, 2 ;
0: 1-63, 2, 1 ;
0: 1-63, 1, 0 ;
0: 0-0, 1, 0 ;
EOF

cat > /tmp/ds4_zrl_color.scans << 'EOF'
0,1,2: 0-0, 0, 1 ;
0: 1-63, 0, 2 ;
1: 1-63, 0, 1 ;
2: 1-63, 0, 1 ;
0: 1-63, 2, 1 ;
0: 1-63, 1, 0 ;
0,1,2: 0-0, 1, 0 ;
1: 1-63, 1, 0 ;
2: 1-63, 1, 0 ;
EOF

cjpeg -grayscale -quality 75 -restart 1 -scans /tmp/ds4_zrl_gray.scans \
    -outfile prog_ac_refine_zrl_gray.jpg < /tmp/ds4_zrl_gray.pgm
cjpeg -quality 75 -restart 1 -scans /tmp/ds4_zrl_color.scans \
    -outfile prog_ac_refine_zrl_420.jpg < /tmp/ds4_zrl_420.ppm

# One DC scan per component: non-interleaved DC scans, first and refinement.
cat > /tmp/ds4_dc_split.scans << 'EOF'
0: 0-0, 0, 1 ;
1: 0-0, 0, 1 ;
2: 0-0, 0, 1 ;
0: 1-63, 0, 0 ;
1: 1-63, 0, 0 ;
2: 1-63, 0, 0 ;
0: 0-0, 1, 0 ;
1: 0-0, 1, 0 ;
2: 0-0, 1, 0 ;
EOF

cjpeg -quality 75 -sample 2x2,1x1,1x1 -scans /tmp/ds4_dc_split.scans \
    -outfile prog_dc_per_component_420.jpg < /tmp/ds4_dc_420.ppm
# Single-component frame with 2x2 sampling factors: blocks stay in raster order.
cjpeg -grayscale -quality 75 -sample 2x2 \
    -outfile base_gray_2x2.jpg < /tmp/ds4_gray22.pgm
echo "wrote prog_ac_refine_zrl_gray.jpg, prog_ac_refine_zrl_420.jpg,"
echo "      prog_dc_per_component_420.jpg and base_gray_2x2.jpg"
