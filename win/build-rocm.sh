#!/usr/bin/env bash
# build-rocm.sh — native Windows ROCm/HIP build of ds4-bench.exe for gfx1151.
#
# Builds DS4's GPU (HIP) backend natively on Windows with the AMD HIP SDK — no
# WSL, no MSVC on PATH, no full Visual Studio install required at the command
# line (hipcc auto-discovers the VS build tools' headers). Produces a gfx1151
# ROCm binary: ds4-bench.exe.
#
# Why a script instead of pure Make: hipcc.exe's .bat wrapper splits arguments
# on spaces, so paths like "C:/Program Files/AMD/ROCm/7.1" break -I/-L flags.
# This script relies on the SDK's default include/lib search (hipcc adds the SDK
# include via -idirafter automatically) and a space-free path for the hipblas
# import lib, sidestepping the quoting problem.
#
# Usage:
#   win/build-rocm.sh                 # uses defaults below
#   ROCM_PATH="C:/Program Files/AMD/ROCm/7.1" ROCM_ARCH=gfx1151 win/build-rocm.sh
#
# Requirements:
#   - AMD HIP SDK (default C:/Program Files/AMD/ROCm/7.1) with hipcc.exe + clang.
#   - Vendored rocWMMA version header at win/third_party/rocwmma (in-tree).
#   - hipblas.lib (MSVC import lib). Generated on the fly from libhipblas.dll if
#     win/third_party/hipblas.lib is absent (needs llvm-dlltool on PATH or in the
#     scoop LLVM install).
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-C:/Program Files/AMD/ROCm/7.1}"
ROCM_ARCH="${ROCM_ARCH:-gfx1151}"

# Resolve the repo root (this script lives in win/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

HIPCC="$ROCM_PATH/bin/hipcc.exe"
CLANG="$ROCM_PATH/bin/clang.exe"
THIRD="win/third_party"
ROCWMMA_INC="$THIRD/rocwmma"

if [ ! -x "$HIPCC" ]; then
    echo "error: hipcc.exe not found at '$HIPCC' (set ROCM_PATH)" >&2
    exit 2
fi

# --- ensure MSVC-style import libs exist ------------------------------------
# The Windows HIP SDK ships only MinGW-style lib*.dll.a import libs, which the
# MSVC linker (lld-link) cannot consume. Synthesize MSVC .lib files from each
# DLL's export table once and cache them in win/third_party/. main's ROCm path
# links BOTH hipBLAS and hipBLASLt (see Makefile ROCM_LDLIBS); the old port
# needed only hipBLAS.
DLLTOOL="$(command -v llvm-dlltool.exe 2>/dev/null || true)"
if [ -z "$DLLTOOL" ] && [ -x "$HOME/scoop/apps/llvm/current/bin/llvm-dlltool.exe" ]; then
    DLLTOOL="$HOME/scoop/apps/llvm/current/bin/llvm-dlltool.exe"
fi
OBJDUMP="$ROCM_PATH/bin/llvm-objdump.exe"

gen_import_lib() {
    # $1 = dll base name (e.g. libhipblas) ; produces $THIRD/<short>.lib
    local dll="$1.dll" out="$THIRD/${1#lib}.lib"
    [ -f "$out" ] && return 0
    if [ -z "$DLLTOOL" ]; then
        echo "error: need llvm-dlltool to build $out (install LLVM or scoop llvm)" >&2
        exit 2
    fi
    if [ ! -f "$ROCM_PATH/bin/$dll" ]; then
        echo "error: $ROCM_PATH/bin/$dll not found (HIP SDK incomplete?)" >&2
        exit 2
    fi
    echo "==> generating $out from $dll"
    local def; def="$(mktemp)"
    { echo "LIBRARY $dll"; echo "EXPORTS"; \
      "$OBJDUMP" -p "$ROCM_PATH/bin/$dll" \
        | awk '/Export Table:/{f=1} f&&/^[[:space:]]+[0-9]+[[:space:]]+0x[0-9a-f]+[[:space:]]+/{print $NF}' \
        | sort -u; } > "$def"
    "$DLLTOOL" -m i386:x86-64 -d "$def" -l "$out" -D "$dll"
    rm -f "$def"
}

gen_import_lib libhipblas
gen_import_lib libhipblaslt

# --- common flags ----------------------------------------------------------
# Host C files are compiled in the MSVC ABI (clang --target=...-windows-msvc) so
# they link against the hipcc-produced (MSVC-ABI) ds4_cuda.o. DS4_WIN_PTHREAD
# selects the Win32 pthread shim (MSVC has no <pthread.h>).
# DS4_ROCM_BUILD matches the Makefile `strix-halo` target's CFLAGS so the host
# C files take the ROCm code paths. DS4_WIN_PTHREAD selects the Win32 pthread
# shim (MSVC has no <pthread.h>).
HOSTFLAGS="--target=x86_64-pc-windows-msvc -O3 -ffast-math -fno-finite-math-only \
  -DDS4_ROCM_BUILD -DDS4_WIN_PTHREAD -D_CRT_SECURE_NO_WARNINGS \
  -Wno-deprecated-declarations -Wno-unused-command-line-argument"

# -std=c++17 is required by ROCm 7.1's hipcub/rocprim headers (std::visit and
# constexpr_value_variant), which main's ds4_rocm.h pulls in via <hipcub>.
GPUFLAGS="--offload-arch=$ROCM_ARCH -std=c++17 -O3 -fno-finite-math-only \
  -D__HIP_PLATFORM_AMD__ -D_CRT_SECURE_NO_WARNINGS \
  -Wno-deprecated-declarations -Wno-unused-command-line-argument -I$ROCWMMA_INC"

# main split the GPU backend: the ROCm path compiles ds4_rocm.cu (which pulls in
# the rocm/*.cuh kernel/launch units), NOT ds4_cuda.cu (that is the CUDA/nvcc
# unit). ds4-bench links: ds4_bench.o ds4_help.o + CORE_OBJS, where the ROCm
# CORE_OBJS = ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o (see Makefile).
echo "==> compiling ds4_rocm.cu (HIP, $ROCM_ARCH)"
"$HIPCC" $GPUFLAGS -c ds4_rocm.cu -o ds4_rocm.o

# Host C translation units (MSVC ABI, to match the hipcc-built ds4_rocm.o).
for src in ds4 ds4_bench ds4_help ds4_distributed ds4_ssd; do
    echo "==> compiling $src.c (host, MSVC ABI)"
    "$CLANG" $HOSTFLAGS -c "$src.c" -o "$src.o"
done

echo "==> linking ds4-bench.exe"
"$HIPCC" --offload-arch="$ROCM_ARCH" \
    ds4_bench.o ds4_help.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o \
    -o ds4-bench.exe -L"$THIRD" -lhipblas -lhipblaslt -lws2_32

echo "==> done: ds4-bench.exe"
echo "    Run with the SDK bin on PATH, e.g.:"
echo "      PATH=\"$ROCM_PATH/bin:\$PATH\" ./ds4-bench.exe --prompt-file FILE -m MODEL.gguf"
