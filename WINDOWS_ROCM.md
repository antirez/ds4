# DS4 on Windows — ROCm / Strix Halo Port

This document describes every change made to build and run DS4 natively on Windows
with an AMD Strix Halo GPU (Radeon 8060S, gfx1151, RDNA4) using ROCm 7.1 HIP SDK.

## Machine Configuration

| Component | Details |
|---|---|
| OS | Windows 11 |
| CPU/GPU | AMD Strix Halo (Ryzen AI Max, Radeon 8060S gfx1151) |
| RAM | 96 GB LPDDR5X (unified with GPU, 107.87 GiB GPU-visible) |
| Build toolchain | MSYS2 / MinGW64, ROCm 7.1 HIP SDK (`hipcc` wrapping Clang 21) |
| ABI target | `x86_64-pc-windows-msvc` |
| Model | DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf (80.76 GiB) |

## Architecture Decision: Unified MSVC ABI

All compilation goes through `hipcc` (which wraps Clang targeting MSVC), not
`gcc`/`g++`. This avoids ABI mismatches between MinGW-compiled C objects and
MSVC-compiled GPU kernels (hipblaslt, hipblas, amdhip64 all use the MSVC ABI).

## Files Added

### `compat/win32/` — POSIX Shim Layer

A complete compatibility layer mapping POSIX APIs to Win32 equivalents.

| File | Purpose |
|---|---|
| `unistd.h` | POSIX unistd shim: `close()`, `pread()`, `sysconf()`, `usleep()`, `clock_gettime()`, `PATH_MAX`, `SIGPIPE`, `ssize_t`, `off_t`, `useconds_t` |
| `sys/file.h` | `flock()`, `fcntl()`, `dprintf()`, `nanosleep()`, `sleep()`, `sigaction` stub, `winsize`, `ioctl()`, `mkdir()` |
| `sys/socket.h` | Winsock2 wrapper: `socket()` with auto-`WSAStartup`, `poll()` via `WSAPoll`, `if_nametoindex()` stub, POSIX shutdown constants |
| `sys/time.h` | `gettimeofday()` via `GetSystemTimeAsFileTime` |
| `sys/mman.h` + `sys/mman.c` | `mmap()`/`munmap()`/`msync()`/`mlock()` via `CreateFileMapping`/`MapViewOfFile` |
| `dirent.h` | `opendir()`/`readdir()`/`closedir()` via `FindFirstFileA`/`FindNextFileA`; `ino_t` typedef |
| `strings.h` | `strcasecmp`/`strncasecmp` macros → `_stricmp`/`_strnicmp` |
| `poll.h` | Forward include of `sys/socket.h` |

### `create_import_libs.py`

Script to generate `.lib` import libraries for ROCm DLLs (hipblas, hipblaslt)
when `.dll.a` files aren't available.

## Changes to Source Files

### `Makefile`

**Added `strix-halo-windows` build target.** Key points:

- Uses `hipcc` (from ROCm 7.1) as CC for all C sources, with `-x c -std=c11`
  to force C mode.
- Uses `hipcc` for C++ GPU kernels with `--offload-arch=gfx1151` (Strix Halo /
  RDNA4).
- Links against `amdhip64.lib`, `libhipblas.dll.a`, `libhipblaslt.dll.a` from
  `C:/PROGRA~1/AMD/ROCm/7.1/lib/`.
- Links `-lws2_32 -liphlpapi` for Winsock2 and IP helper.
- Adds `-I./compat/win32` to the include path so POSIX headers resolve to our
  shims.
- Adds `-D_CRT_SECURE_NO_WARNINGS -Wno-deprecated-declarations` to silence MSVC
  deprecation noise.
- Compiles `compat/win32/sys/mman.c` to `compat/win32/sys/mman.o` and includes
  it in `CORE_OBJS`.

### `ds4.c`

**64-bit stat for large files.** The model file is 86 GB. Windows MSVC defaults
`stat`/`fstat` to 32-bit `off_t` which cannot represent file sizes > 4 GB.
Added a `#ifdef _WIN32` block replacing `fstat()` with `_fstat64()` and
`struct stat` with `struct _stat64` in the model-open path.

```c
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
#else
    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
#endif
```

### `ds4_server.c`

**Two fixes:**

1. **Include `sys/socket.h` on Windows.** The original code had:
   ```c
   #ifdef _WIN32
     /* sys/socket.h on Windows is handled via winsock2.h */
   #else
     #include <sys/socket.h>
   #endif
   ```
   This skipped our compat shim entirely, meaning `WSAStartup()` was never
   called before `socket()`. Without `WSAStartup`, every socket call returns
   `WSANOTINITIALISED` (errno 132, "value too large"). Fix: include
   `sys/socket.h` unconditionally (the compat version handles Winsock init).

2. **`setsockopt` timeout type.** On Windows, `SO_RCVTIMEO`/`SO_SNDTIMEO`
   already use `struct timeval` via the compat layer (no change needed since
   winsock2 accepts it), but the `(const char*)` cast was already present.

### `ds4_bench.c`

Added `#include <sys/file.h>` to pull in `clock_gettime`, `nanosleep`,
`PATH_MAX`, and `CLOCK_MONOTONIC` on Windows.

### `ds4_cli.c` / `ds4_distributed.c` / `ds4_eval.c` / `ds4_kvstore.c` / `ds4_rocm.cu` / `ds4_ssd.c` / `ds4_web.c` / `linenoise.c`

These files compiled with only minor warning fixes (unused variable suppression,
type casts) and no functional changes on Windows.

### `rocm/ds4_rocm_runtime.cuh`

**Three changes:**

1. **64-bit `off_t` cast in `cuda_pread_full`.** The inner loop casts the file
   offset to `off_t` before passing to `pread()`. On Windows MSVC, `off_t` is
   32-bit (`typedef long off_t`) even though our compat layer defines `off_t` as
   `__int64`. At offsets > 4 GB (which this 86 GB model has many of), the cast
   silently truncated to a negative value, causing every read past 4 GB to
   fail with EIO. Fix: use `(int64_t)` instead of `(off_t)`.

   ```c
   // Before:
   ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
   // After:
   ssize_t n = pread(fd, (char *)buf + done, n_req, (int64_t)(offset + done));
   ```

2. **Windows `_WIN32` reserve in Q8→F16 cache budget.** After loading the
   80.76 GiB model, the ROCm driver on Windows has ~12 GiB of overhead
   (staging buffers, driver bookkeeping, HipBLAS handles) that Linux does not.
   The existing reserve of `5% of total = 5.39 GiB` was not enough to leave
   room for inference computation tensors after the Q8→F16 cache allocated
   10+ GiB. Added an extra `+12 GiB` reserve on Windows so the Q8 cache stops
   early, leaving GPU memory for the KV cache and inference buffers.

   ```c
   #ifdef _WIN32
   reserve += 12ull * 1024ull * 1048576ull;
   #endif
   ```

3. **`int64_t` fix** as noted above (item 1).

## Changes to Compatibility Layer (compat/win32/)

### `unistd.h` — `pread()` Implementation

The original `pread` used a plain `OVERLAPPED` struct on a synchronous HANDLE.
The key fixes:

- **`ReOpenFile` for thread safety.** The ROCm streaming engine uses 18 worker
  threads that all call `pread()` on the same file descriptor concurrently.
  Concurrent `ReadFile` on a synchronous HANDLE from multiple threads is
  undefined on Windows. Fixed by calling `ReOpenFile()` to obtain a private
  `FILE_FLAG_OVERLAPPED` HANDLE for each read, paired with a manual-reset event
  for async completion. This gives true parallel positional I/O without races.

- **`close()` handles sockets.** Windows `_close()` only works on CRT file
  descriptors, not Winsock `SOCKET` handles. Added a fallback: if `_close(fd)`
  returns `EBADF`, try `closesocket(fd)` via `GetProcAddress("ws2_32.dll",
  "closesocket")` to avoid the `include <winsock2.h>` header conflict.

### `sys/socket.h` — `WSAStartup` Auto-Init

Added `ds4_wsa_ensure_init()` called from a `ds4_socket()` wrapper that replaces
`socket()` via `#define`. This ensures Winsock is initialized lazily on the
first `socket()` call in any compilation unit that includes this header.

```c
static inline int ds4_wsa_ensure_init(void) {
    static int s_inited = 0;
    if (!s_inited) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        s_inited = 1;
    }
    return 1;
}
```

The ROCm runtime (`hipblaslt`, `amdhip64`) does **not** call `WSAStartup` on
behalf of the process. Without this, every socket call in `ds4-server` fails.

### `sys/file.h` — `sigaction` Stub, `winsize`, `clock_gettime`

- Provided a no-op `sigaction()` + `sigemptyset()` because `ds4_cli.c` calls
  `sigaction(SIGINT, ...)` to handle Ctrl+C.
- Provided `struct winsize` and `TIOCGWINSZ` via `GetConsoleScreenBufferInfo`
  for terminal width detection (`linenoise`).
- Provided `clock_gettime(CLOCK_MONOTONIC)` via `QueryPerformanceCounter` for
  `ds4_bench.c` and `ds4_eval.c`.

### `sys/mman.h` + `sys/mman.c` — 64-bit `mmap`

The mmap implementation properly splits the 64-bit file size and offset into
`DWORD maxHigh`/`maxLow` for `CreateFileMapping`, and passes the full `SIZE_T`
(64-bit on 64-bit Windows) to `MapViewOfFile`. This is required to map the
86 GB GGUF file into a single contiguous virtual address range.

`OffsetType` is `int64_t` on `_WIN64` to match the 64-bit offset requirements.

## Runtime Configuration

### Lock File

DS4 defaults to `/tmp/ds4.lock`. Windows does not have `/tmp`. Set:
```
DS4_LOCK_FILE=C:/Windows/Temp/ds4.lock
```

### ROCm DLLs on PATH

The ROCm runtime DLLs must be on `PATH`:
```
PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
```

### Q8 Cache (no longer needed)

The `DS4_CUDA_NO_Q8_F16_CACHE=1` env var was the initial workaround before the
`+12 GiB` reserve fix was applied. It is no longer required with the patched
build.

## Memory Budget (107.87 GiB unified pool)

| Component | GiB |
|---|---|
| Model weights (Q2/IQ2/Q8 mix) | 80.76 |
| Q8→F16 partial cache | 2.04 |
| ROCm driver overhead (Win) | ~12.00 |
| Available for context | ~13.0 |

## Tested Context Sizes

| `--ctx` | Context buffers | Status |
|---|---|---|
| 32,768 (32k) | 1.05 GiB | ✅ |
| 131,072 (128k) | 3.11 GiB | ✅ |
| 196,608 (192k) | 4.38 GiB | ✅ |
| 229,376 (224k) | 4.84 GiB | ✅ recommended max |
| 237,568 (232k) | 5.21 GiB | ✅ |
| 245,760 (240k) | 5.38 GiB | ❌ KV alloc OOM |
| 262,144 (256k) | ~6.3 GiB | ❌ model load OOM |

**Recommended**: `--ctx 229376` for maximum stable context.

## Performance

Measured on AMD Radeon 8060S (Strix Halo), `--ctx 32768`, `--nothink`:

- **Prefill**: ~15–18 t/s
- **Generation**: ~15–16 t/s

This matches the expected range for Strix Halo with a 2-bit quantized model.

## Build Instructions

### Prerequisites

1. [MSYS2](https://www.msys2.org/) with MinGW64
2. [AMD ROCm 7.1 HIP SDK for Windows](https://rocm.docs.amd.com/en/latest/)
3. ROCm 7.1 `bin` on PATH (for `hipcc`)

### Build

```bash
# In MSYS2 bash terminal:
export PATH=/c/PROGRA~1/AMD/ROCm/7.1/bin:/mingw64/bin:/usr/bin:$PATH
cd /c/path/to/ds4
make strix-halo-windows
```

### Run

```bash
export DS4_LOCK_FILE="C:/Windows/Temp/ds4.lock"
export PATH="/c/PROGRA~1/AMD/ROCm/7.1/bin:$PATH"

# Interactive CLI:
./ds4 --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
      --ctx 229376

# HTTP API server:
./ds4-server --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
             --ctx 229376 \
             --host 127.0.0.1 \
             --port 8000
```

## Known Remaining Issues

- **SSD streaming (`--ssd-streaming`) crashes** at ~8 GiB during model loading.
  The cause is the fallback `cudaMemcpy(dst, mmap_ptr+large_offset, n,
  HostToDevice)` path in `cuda_model_range_ptr`. On Windows, the ROCm driver
  cannot DMA from demand-paged mmap'd memory at arbitrary large offsets when
  pread fails. Not needed for full-fit machines (model fits in 107.87 GiB), but
  blocks use on machines with less RAM.

- **`isLargeBar: 0`** in `hipInfo`. Enabling Resizable BAR in BIOS may improve
  DMA performance and potentially allow larger context windows.

- **pread "range read failed" warnings** during model loading: minor, non-fatal,
  handled by the `cudaHostRegister` fallback path. Occur because some tensor
  spans are loaded via the mmap fallback rather than direct pread. Does not
  affect inference correctness.
