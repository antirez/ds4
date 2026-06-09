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

The `windows-cpu` target is self-contained: it sets its own `CC=gcc` (override
with `WIN_CPU_CC=`) and CPU flags, so it builds the same way regardless of which
`uname -s` branch the Makefile takes. It compiles the GPU-less core
(`-DDS4_NO_GPU`) and links the Winsock import libs the distributed runtime now
needs (see below).

### Direct gcc (no make)

main moved `ds4_distributed.c` and `ds4_ssd.c` into the shared core, so the CPU
bench now links them too — `ds4_distributed.c` pulls in the Winsock shim, so the
link needs `-lws2_32 -liphlpapi` (the shim's `#pragma comment(lib,…)` is a no-op
under gcc). Full TU set: `ds4.c`, `ds4_bench.c`, `ds4_help.c`,
`ds4_distributed.c`, `ds4_ssd.c`.

```sh
CF="-O3 -ffast-math -march=native -std=c99 -D_GNU_SOURCE -fno-finite-math-only \
    -DDS4_NO_GPU"
gcc $CF -c ds4.c             -o ds4_cpu.o
gcc $CF -c ds4_bench.c       -o ds4_bench_cpu.o
gcc $CF -c ds4_help.c        -o ds4_help.o
gcc $CF -c ds4_distributed.c -o ds4_distributed.o
gcc $CF -c ds4_ssd.c         -o ds4_ssd.o
gcc $CF -o ds4-bench.exe ds4_bench_cpu.o ds4_help.o ds4_cpu.o \
    ds4_distributed.o ds4_ssd.o -lm -lpthread -lws2_32 -liphlpapi
```

Toolchain used: `x86_64-w64-mingw32` GCC 15.2.0. MinGW supplies
pthread/`clock_gettime`/`ftruncate`/`sleep` natively, so the MSVC pthread shim
(`win/ds4_pthread_win.h`) is not used here (it is `!__MINGW32__`-guarded).

### Verify it runs

```sh
$ ./ds4-bench.exe
ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file
```

(Exit code 2 — argument validation fires, proving the binary executes.)

## Build `ds4-bench.exe` (ROCm / HIP, AMD GPU — gfx1151)

The GPU backend builds **natively on Windows** with the AMD HIP SDK — no WSL, no
MSVC on `PATH`, and no full Visual Studio install needed at the command line.
Target: AMD Strix Halo (gfx1151).

> **main structure note.** Upstream split the GPU backend: the ROCm path now
> compiles **`ds4_rocm.cu`** (which `#include`s the `rocm/*.cuh` kernel/launch
> units), *not* `ds4_cuda.cu` (that is the CUDA/nvcc unit). The distributed
> runtime (`ds4_distributed.c`) and SSD-cache helper (`ds4_ssd.c`) moved into
> the shared `CORE_OBJS`, so they now link into **every** binary — including
> `ds4-bench`. The full ROCm `ds4-bench` translation-unit set is therefore:
> `ds4_rocm.cu` + host C `ds4.c`, `ds4_bench.c`, `ds4_help.c`,
> `ds4_distributed.c`, `ds4_ssd.c`. `ds4_distributed.c` is a TCP
> coordinator/worker transport written against BSD sockets, so the Windows build
> needs a Winsock shim (see below) even though the bench never serves.

### Prerequisites

- **AMD HIP SDK** (default `C:/Program Files/AMD/ROCm/7.1`), providing
  `bin/hipcc.exe`, `bin/clang.exe`, `include/hipblas/…`, `bin/libhipblas.dll`,
  and the gfx1151 device bitcode (`amdgcn/bitcode/oclc_isa_version_1151.bc`).
- **`llvm-dlltool`** reachable on `PATH` (or a scoop LLVM install at
  `~/scoop/apps/llvm/current`). Used **once** to synthesize MSVC-style
  `hipblas.lib` **and** `hipblaslt.lib` from the SDK's `libhipblas.dll` /
  `libhipblaslt.dll` (see "Why a generated import lib" below). After first run
  the libs are cached in `win/third_party/`.
- **Vendored rocWMMA headers** (in-tree, already committed). The Windows HIP SDK
  does **not** ship the header-only rocWMMA library, and on main `ds4_rocm.h`
  includes the **full** `<rocwmma/rocwmma.hpp>` — the ROCm MoE / Q8 / indexer
  kernels (`rocm/ds4_rocm_moe.cuh`, `rocm/ds4_rocm_q8.cuh`,
  `rocm/ds4_rocm_indexer.cuh`) use `rocwmma::fragment`/`mma_sync` directly, so
  the WMMA path is now mandatory for HIP (it is no longer CUDA-only as in the
  old port). The complete header tree (rocWMMA `rocm-7.1.0`, MIT-licensed,
  header-only) is vendored under `win/third_party/rocwmma/rocwmma/`, with the
  CMake-generated `rocwmma-version.hpp` (rocWMMA 2.2.1, matching the installed
  ROCm 7.1 SDK) kept from the SDK.

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

1. Generates `win/third_party/hipblas.lib` and `hipblaslt.lib` from the SDK DLLs
   (first run only).
2. Compiles `ds4_rocm.cu` with `hipcc.exe --offload-arch=gfx1151 -std=c++17
   -D__HIP_PLATFORM_AMD__ -I win/third_party/rocwmma`. (`-std=c++17` is required
   by ROCm 7.1's hipcub/rocprim headers — `std::visit` / `constexpr_value_variant`
   — which `ds4_rocm.h` pulls in via `<hipcub>`.)
3. Compiles the host C files (`ds4.c`, `ds4_bench.c`, `ds4_help.c`,
   `ds4_distributed.c`, `ds4_ssd.c`) with
   `clang --target=x86_64-pc-windows-msvc … -DDS4_ROCM_BUILD -DDS4_WIN_PTHREAD`
   so they share the **MSVC ABI** with the hipcc-built `ds4_rocm.o` (mixing the
   MinGW and MSVC C runtimes across the `FILE*`/heap boundary is unsafe, so the
   GPU build uses MSVC ABI throughout — unlike the MinGW CPU build above).
   `-DDS4_ROCM_BUILD` matches the Makefile `strix-halo` target's `CFLAGS`.
4. Links `ds4-bench.exe` with `hipcc.exe` + `-lhipblas -lhipblaslt -lws2_32`,
   pulling in `amdhip64_7.dll`, `libhipblas.dll`, `libhipblaslt.dll` and Winsock.

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

The Windows HIP SDK ships hipBLAS and hipBLASLt with only the **MinGW-style**
`lib/libhipblas.dll.a` / `lib/libhipblaslt.dll.a`, which the MSVC linker
(`lld-link`, used by `hipcc` on Windows) cannot consume. The HIP runtime itself
has a proper MSVC import lib (`lib/amdhip64.lib`), but the BLAS libs do not.
`win/build-rocm.sh` therefore dumps each DLL's export table and synthesizes COFF
`hipblas.lib` / `hipblaslt.lib` with `llvm-dlltool`. The generated libs are
git-ignored and regenerated on demand. (main's ROCm path uses hipBLASLt for the
MoE matmul descriptors, so both libs are required now — the old port linked only
hipBLAS.)

### Native-Windows portability shims (all behind `#ifdef _WIN32`)

- `ds4_win.h` — the existing POSIX shim, extended for the MSVC-ABI GPU build
  with `STDERR_FILENO`/`ssize_t`/`SSIZE_MAX`/`off_t`, `clock_gettime` +
  `CLOCK_MONOTONIC`, and `ftruncate`, **plus the surface main newly requires**:
  anonymous `mmap` (`MAP_ANONYMOUS`/`MAP_ANON`, `VirtualAlloc`-backed) +
  `mlock`/`munlock` (for `ds4_ssd.c --simulate-used-memory`), and
  `nanosleep`/`sleep`/`getpagesize`/`mkstemp`/`ftello`/`fseeko`/`PATH_MAX`
  (used by `ds4.c`, `ds4_bench.c`, `ds4_distributed.c`). All additions are
  guarded `!defined(__MINGW32__)` so the MinGW CPU build is unchanged.
- `win/ds4_pthread_win.h` — a header-only Win32 pthread shim (threads, mutex,
  condition variable, once, **`pthread_detach`** added for the distributed
  runtime) used when `DS4_WIN_PTHREAD` is defined, because the MSVC toolchain
  has no `<pthread.h>`. The MinGW CPU build keeps winpthreads.
- `win/ds4_sockets_win.h` — **new**. A minimal BSD-sockets-over-Winsock2 shim
  for `ds4_distributed.c` (which main links into every binary, including the
  bench). Supplies the POSIX socket headers' surface mapped onto Winsock2:
  `poll`→`WSAPoll`, `close`→`closesocket` (with a non-socket `_close` fallback),
  errno translation on `recv`/`send`/`accept`/`connect`/`socket`/`getaddrinfo`,
  lazy `WSAStartup`, `SIGPIPE` no-op, `if_nametoindex` via `<iphlpapi.h>`. Its
  body is `_WIN32`-only.
- `st_blksize` (a Linux `O_DIRECT` alignment hint) is skipped on Windows in
  `rocm/ds4_rocm_runtime.cuh`; the direct-I/O block itself is already
  `#if defined(__linux__) && defined(O_DIRECT)`.

All edits to `ds4.c`, `ds4_bench.c`, `ds4_help.c`, `ds4_ssd.c`,
`ds4_distributed.c`, `ds4_rocm.cu`, and `rocm/ds4_rocm_runtime.cuh` are behind
`_WIN32` (further sub-guarded by `__MINGW32__` / `DS4_WIN_PTHREAD`), so the
POSIX, macOS, CUDA, and Linux-ROCm builds produce byte-identical preprocessor
output.

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
- **Distributed serving on Windows**: `ds4_distributed.c` now compiles and links
  into the bench via `win/ds4_sockets_win.h`, but the shim is sized for
  *compile/link* of the (non-serving) bench, not full `--role coordinator/worker`
  fidelity. Two runtime gaps to close before distributed serving works on
  Windows: (1) `dup()` of a socket maps to `_dup`, which does not duplicate a
  Winsock socket — needs `WSADuplicateSocket`; (2) `SO_RCVTIMEO`/`SO_SNDTIMEO`
  take a `DWORD` milliseconds value on Winsock, not `struct timeval`. Both are
  on coordinator-only paths the bench never reaches.
- **Server (`ds4-server`)**: links `rax.c`, `ds4_kvstore.c`, `ds4_server.c`;
  reuse `win/ds4_sockets_win.h` and audit its larger socket surface.

## rocWMMA version caveat

The vendored rocWMMA tree is the upstream `rocm-7.1.0` tag (its CMake
`VERSION_STRING` reads `2.0.0`), while the installed SDK's generated
`rocwmma-version.hpp` reads `2.2.1`. The kernels compile cleanly against the
`rocm-7.1.0` headers for gfx1151, but if a future SDK bump changes the rocWMMA
fragment API, re-vendor from the matching `rocm-<sdk-version>` tag.
