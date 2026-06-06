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
HIPBLAS_LIB="$THIRD/hipblas.lib"

if [ ! -x "$HIPCC" ]; then
    echo "error: hipcc.exe not found at '$HIPCC' (set ROCM_PATH)" >&2
    exit 2
fi

# --- ensure an MSVC-style hipblas import lib exists -------------------------
# The Windows HIP SDK ships only the MinGW-style libhipblas.dll.a, which the
# MSVC linker (lld-link) cannot consume. Generate hipblas.lib from the DLL's
# export table once and cache it in win/third_party/.
if [ ! -f "$HIPBLAS_LIB" ]; then
    echo "==> generating $HIPBLAS_LIB from libhipblas.dll"
    DLLTOOL="$(command -v llvm-dlltool.exe 2>/dev/null || true)"
    if [ -z "$DLLTOOL" ] && [ -x "$HOME/scoop/apps/llvm/current/bin/llvm-dlltool.exe" ]; then
        DLLTOOL="$HOME/scoop/apps/llvm/current/bin/llvm-dlltool.exe"
    fi
    if [ -z "$DLLTOOL" ]; then
        echo "error: need llvm-dlltool to build hipblas.lib (install LLVM or scoop llvm)" >&2
        exit 2
    fi
    OBJDUMP="$ROCM_PATH/bin/llvm-objdump.exe"
    DEF="$(mktemp)"
    { echo "LIBRARY libhipblas.dll"; echo "EXPORTS"; \
      "$OBJDUMP" -p "$ROCM_PATH/bin/libhipblas.dll" \
        | awk '/Export Table:/{f=1} f&&/^[[:space:]]+[0-9]+[[:space:]]+0x[0-9a-f]+[[:space:]]+/{print $NF}' \
        | sort -u; } > "$DEF"
    "$DLLTOOL" -m i386:x86-64 -d "$DEF" -l "$HIPBLAS_LIB" -D libhipblas.dll
    rm -f "$DEF"
fi

# --- common flags ----------------------------------------------------------
# Host C files are compiled in the MSVC ABI (clang --target=...-windows-msvc) so
# they link against the hipcc-produced (MSVC-ABI) ds4_cuda.o. DS4_WIN_PTHREAD
# selects the Win32 pthread shim (MSVC has no <pthread.h>).
HOSTFLAGS="--target=x86_64-pc-windows-msvc -O3 -ffast-math -fno-finite-math-only \
  -DDS4_WIN_PTHREAD -D_CRT_SECURE_NO_WARNINGS \
  -Wno-deprecated-declarations -Wno-unused-command-line-argument"

GPUFLAGS="--offload-arch=$ROCM_ARCH -O3 -fno-finite-math-only \
  -D__HIP_PLATFORM_AMD__ -D_CRT_SECURE_NO_WARNINGS \
  -Wno-deprecated-declarations -Wno-unused-command-line-argument -I$ROCWMMA_INC"

echo "==> compiling ds4_cuda.cu (HIP, $ROCM_ARCH)"
"$HIPCC" $GPUFLAGS -c ds4_cuda.cu -o ds4_cuda.o

echo "==> compiling ds4.c (host, MSVC ABI)"
"$CLANG" $HOSTFLAGS -c ds4.c -o ds4.o

echo "==> compiling ds4_bench.c (host, MSVC ABI)"
"$CLANG" $HOSTFLAGS -c ds4_bench.c -o ds4_bench.o

echo "==> linking ds4-bench.exe"
"$HIPCC" --offload-arch="$ROCM_ARCH" ds4_bench.o ds4.o ds4_cuda.o \
    -o ds4-bench.exe -L"$THIRD" -lhipblas

echo "==> done: ds4-bench.exe"
echo "    Run with the SDK bin on PATH, e.g.:"
echo "      PATH=\"$ROCM_PATH/bin:\$PATH\" ./ds4-bench.exe --prompt-file FILE -m MODEL.gguf"
