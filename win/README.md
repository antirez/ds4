# Native Windows build (experimental)

DS4 assumes a POSIX environment. `ds4_win.h` (top level) is a small,
dependency-free compatibility shim that lets the **CPU backend** build with
native MinGW-w64 GCC (no WSL, no Cygwin, no MSVC). It supplies the POSIX surface
MinGW/UCRT lacks: `mmap`/`munmap`/`madvise`, `sysconf`, `flock`/`fcntl`/`pread`/
`dprintf`, and a temp-file-backed `fmemopen`. MinGW already provides `pthread`,
`clock_gettime`, and `ftruncate`.

The shim is wired in-tree: `ds4.c` includes `ds4_win.h` in place of
`<sys/mman.h>` (and the other POSIX-only headers) behind `#ifdef _WIN32`. The
header's entire body is guarded by `_WIN32`, so POSIX builds are byte-for-byte
unchanged. No special include/search-path flags are needed.

## Build `ds4-bench.exe` (CPU)

### With make (MinGW/MSYS)

```sh
make windows-cpu
```

`uname -s` on MinGW/MSYS reports `MINGW64_NT*` / `MSYS_NT*`; the Makefile detects
this and selects a Windows branch that defaults `CC` to `gcc`.

### Direct gcc (no make)

```sh
CF="-O3 -ffast-math -march=native -std=c99 -D_GNU_SOURCE -fno-finite-math-only \
    -DDS4_NO_GPU -D_CRT_SECURE_NO_WARNINGS"
gcc $CF -c ds4.c       -o ds4_cpu.o
gcc $CF -c ds4_bench.c -o ds4_bench_cpu.o
gcc $CF -o ds4-bench.exe ds4_bench_cpu.o ds4_cpu.o -lm
```

Toolchain used: `x86_64-w64-mingw32` GCC 15.2.0.

### Verify it runs

```sh
$ ./ds4-bench.exe
ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file
```

(Exit code 2 — argument validation fires, proving the binary executes.)

## Build `ds4-bench.exe` (ROCm / HIP, AMD GPU — gfx1151)

The GPU backend (`ds4_cuda.cu`, the unified CUDA/HIP source) builds **natively on
Windows** with the AMD HIP SDK — no WSL, no MSVC on `PATH`, and no full Visual
Studio install needed at the command line. Target: AMD Strix Halo (gfx1151).

### Prerequisites

- **AMD HIP SDK** (default `C:/Program Files/AMD/ROCm/7.1`), providing
  `bin/hipcc.exe`, `bin/clang.exe`, `include/hipblas/…`, `bin/libhipblas.dll`,
  and the gfx1151 device bitcode (`amdgcn/bitcode/oclc_isa_version_1151.bc`).
- **`llvm-dlltool`** reachable on `PATH` (or a scoop LLVM install at
  `~/scoop/apps/llvm/current`). Used **once** to synthesize an MSVC-style
  `hipblas.lib` from the SDK's `libhipblas.dll` (see "Why a generated import
  lib" below). After first run the lib is cached in `win/third_party/`.
- **Vendored rocWMMA header** (in-tree, already committed). The Windows HIP SDK
  does **not** ship the header-only rocWMMA library, but `ds4_rocm.h` includes
  `<rocwmma/rocwmma-version.hpp>`. A faithful copy of the CMake-configured
  version header (rocWMMA 2.2.1, the release that ships with ROCm 7.x) lives at
  `win/third_party/rocwmma/rocwmma/rocwmma-version.hpp`. rocWMMA is MIT-licensed
  and header-only; only the version header is needed because DS4's WMMA kernel
  path is CUDA-only (guarded by `__CUDA_ARCH__`) and is not compiled for HIP.

### With make (MinGW/MSYS)

```sh
make windows-rocm
# overridable:
make windows-rocm ROCM_PATH="C:/Program Files/AMD/ROCm/7.1" ROCM_ARCH=gfx1151
```

### Direct (the script make calls)

```sh
ROCM_PATH="C:/Program Files/AMD/ROCm/7.1" ROCM_ARCH=gfx1151 win/build-rocm.sh
```

The build:

1. Generates `win/third_party/hipblas.lib` from `libhipblas.dll` (first run only).
2. Compiles `ds4_cuda.cu` with `hipcc.exe --offload-arch=gfx1151
   -D__HIP_PLATFORM_AMD__ -I win/third_party/rocwmma`.
3. Compiles the host C files (`ds4.c`, `ds4_bench.c`) with
   `clang --target=x86_64-pc-windows-msvc … -DDS4_WIN_PTHREAD` so they share the
   **MSVC ABI** with the hipcc-built `ds4_cuda.o` (mixing the MinGW and MSVC C
   runtimes across the `FILE*`/heap boundary is unsafe, so the GPU build uses
   MSVC ABI throughout — unlike the MinGW CPU build above).
4. Links `ds4-bench.exe` with `hipcc.exe`, pulling in `amdhip64_7.dll` and
   `libhipblas.dll`.

### Verify it runs

```sh
$ PATH="C:/Program Files/AMD/ROCm/7.1/bin:$PATH" \
  DS4_LOCK_FILE="$TEMP/ds4.lock" ./ds4-bench.exe -m model.gguf --prompt-file p.txt
ds4-bench: context buffers 753.89 MiB (ctx=32897, backend=cuda, …)
```

The SDK `bin` must be on `PATH` at runtime for the HIP/hipBLAS DLLs. The binary
initializes the GPU backend (reported as `backend=cuda`, the unified GPU path
name) before touching the model file. Full inference is not yet runtime-verified
here — the ~80 GB model needs a BIOS UMA memory re-split + reboot — but the HIP
runtime and hipBLAS load and the gfx1151 binary links and starts cleanly.

### Why a generated import lib

The Windows HIP SDK ships hipBLAS with only the **MinGW-style**
`lib/libhipblas.dll.a`, which the MSVC linker (`lld-link`, used by `hipcc` on
Windows) cannot consume. The HIP runtime itself has a proper MSVC import lib
(`lib/amdhip64.lib`), but hipBLAS does not. `win/build-rocm.sh` therefore dumps
`libhipblas.dll`'s export table and synthesizes a COFF `hipblas.lib` with
`llvm-dlltool`. The generated lib is git-ignored and regenerated on demand.

### Native-Windows portability shims (all behind `#ifdef _WIN32`)

- `ds4_win.h` — the existing POSIX shim, extended for the MSVC-ABI GPU build
  with `STDERR_FILENO`/`ssize_t`/`SSIZE_MAX`/`off_t`, `clock_gettime` +
  `CLOCK_MONOTONIC`, and `ftruncate` (MinGW already supplies these; the
  additions are guarded `!defined(__MINGW32__)` so the CPU build is unchanged).
- `win/ds4_pthread_win.h` — a header-only Win32 pthread shim (threads, mutex,
  condition variable, once) used when `DS4_WIN_PTHREAD` is defined, because the
  MSVC toolchain has no `<pthread.h>`. The MinGW CPU build keeps winpthreads.
- `st_blksize` (a Linux `O_DIRECT` alignment hint) is skipped on Windows; the
  whole direct-I/O block is already `#if defined(__linux__) && defined(O_DIRECT)`.

All edits to `ds4.c`, `ds4_bench.c`, and `ds4_cuda.cu` are behind `_WIN32`
(further sub-guarded by `__MINGW32__` / `DS4_WIN_PTHREAD`), so the POSIX, macOS,
CUDA, and Linux-ROCm builds produce byte-identical preprocessor output.

## Status

| Target               | Native Windows build | Notes |
|----------------------|----------------------|-------|
| `ds4-bench` (CPU)    | builds & runs        | MinGW-w64, no terminal/socket deps |
| `ds4-bench` (ROCm)   | builds, links, starts| HIP/clang-MSVC, gfx1151; full inference pending model + UMA re-split |
| `ds4` (CLI)          | not yet              | `linenoise.c` uses POSIX `termios`, plus `sigaction` in `ds4_cli.c` (needs Win console raw-mode port) |
| `ds4-server`         | not yet              | BSD sockets / `poll` / `arpa/inet.h` (needs Winsock port) |

## Runtime note

Set `DS4_LOCK_FILE` to a Windows path (the default is `/tmp/ds4.lock`, which does
not exist on Windows) before running, e.g.:

```sh
export DS4_LOCK_FILE="$TEMP/ds4.lock"
```

For the **ROCm** build, also:

- Put the HIP SDK `bin` on `PATH` at runtime (`amdhip64_7.dll`, `libhipblas.dll`).
- Set `DS4_CUDA_MANAGED=1` to use the full UMA pool (managed memory) once a model
  is being loaded — required for the large Strix Halo UMA allocation.

## Deferred work

- **CLI (`ds4`)**: port `linenoise.c` to the Windows console (raw mode via
  `SetConsoleMode`) and replace the `sigaction`/`SIGINT` handling in `ds4_cli.c`.
- **Server (`ds4-server`)**: port the BSD sockets / `poll(2)` event loop to
  Winsock 2 (`WSAStartup`, `WSAPoll`, `closesocket`).
