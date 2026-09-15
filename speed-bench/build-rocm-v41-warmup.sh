#!/usr/bin/env bash
# Build the published TP warmup adapter against existing ROCm engine objects.
# The engine and ordinary ds4-bench source/binary remain untouched.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
work=$(mktemp -d ./.rocm-v41-warmup.XXXXXX)
trap 'rm -rf -- "$work"' EXIT
patch --silent --output "$work/ds4_bench.c" ds4_bench.c < speed-bench/rocm-v41-warmup.patch
objects=(ds4_help.o ds4_gpu_args.o ds4.o ds4_image.o ds4_distributed.o ds4_tp.o
  ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o
  ds4_engram.o cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o
  cuda/mmq/quantize.rocm.o cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o
  cuda/mmq/d2r_stubs.rocm.o)
for file in "${objects[@]}"; do
  test -f "$file" || { echo "Missing $file; run make strix-halo first." >&2; exit 1; }
done
"${CC:-cc}" -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 \
  -D_GNU_SOURCE -fno-finite-math-only -fPIC -DDS4_ROCM_BUILD -I. \
  -c "$work/ds4_bench.c" -o "$work/ds4_bench.o"
verbs=()
if nm -u ds4_distributed.o ds4_tp.o | grep 'ibv_' >/dev/null; then verbs=(-libverbs); fi
"${HIPCC:-/opt/rocm/bin/hipcc}" -O3 -ffast-math -g -fno-finite-math-only \
  -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=gfx1151 \
  -o ds4-bench-warm "$work/ds4_bench.o" "${objects[@]}" \
  -lm -pthread -lhipblas -lhipblaslt -lrocblas "${verbs[@]}"
./ds4-bench-warm --help >/dev/null
printf '%s\n' 'Built ./ds4-bench-warm: TP only, excluded 256-token/128-output warmup, fresh measured session.'
