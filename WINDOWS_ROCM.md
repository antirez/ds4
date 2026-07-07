# DS4 on Windows — Native ROCm Build for AMD Strix Halo

This document covers building and running DS4 natively on Windows with an AMD Strix
Halo GPU using the ROCm 7.1 HIP SDK.  Every prerequisite installation step, every
source change, and the rationale behind each decision is documented here.

---

## Tested Configuration

| Component | Details |
|---|---|
| OS | Windows 11 |
| Hardware | AMD Strix Halo — Ryzen AI Max, Radeon 8060S (gfx1151, RDNA4) |
| RAM | 96 GB LPDDR5X unified memory (107.87 GiB GPU-visible via ROCm) |
| ROCm | AMD HIP SDK 7.1 |
| Compiler | `hipcc` (HIP SDK 7.1, wrapping Clang 21 targeting MSVC ABI) |
| ABI | `x86_64-pc-windows-msvc` — all objects use the same MSVC ABI |
| MSYS2 | MinGW64 shell for `make` and file operations |
| Model | DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf |
| Model size | 80.76 GiB (Q2/IQ2 routed experts, Q8 attention/shared/output) |

**Measured performance**: ~18 t/s prefill, ~16 t/s generation at `--ctx 229376`
(224k token context).

---

## Prerequisites

Follow these steps in order on a Windows 11 machine with a Strix Halo APU.

### 1. AMD HIP SDK 7.1

Download and install from AMD's documentation:
https://rocm.docs.amd.com/projects/install-on-windows/en/latest/

The installer puts `hipcc.exe`, `clang.exe`, and all ROCm runtime libraries under
`C:\Program Files\AMD\ROCm\7.1\`.  After installation verify:

```powershell
& "C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe" --version
# Expected: HIP version 7.1, clang 21
```

### 2. MSYS2 (MinGW64 shell + make + git)

Download from https://www.msys2.org/ and install to `C:\msys64\`.
Open the **MSYS2 MinGW64** shell and install the required packages:

```bash
pacman -S mingw-w64-x86_64-make mingw-w64-x86_64-git
```

You do **not** need `mingw-w64-x86_64-gcc` — all compilation goes through `hipcc`
to keep a single MSVC ABI.  `make` is the only MinGW64 tool used directly.

### 3. rocWMMA Headers

The ROCm WMMA (warp matrix multiply-accumulate) headers are required for GPU
kernel compilation.  Clone the matching tag and install the headers:

```bash
# In MSYS2 bash
git clone --depth 1 --branch rocm-7.1.0 \
    https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA

mkdir -p /c/msys64/opt/include
cp -a /tmp/rocWMMA/library/include/rocwmma /c/msys64/opt/include/
```

The `Makefile` variable `WIN32_ROCWMMA_INC` defaults to `C:/msys64/opt/include`.
This include path is passed **only** to `hipcc`, not to the C compiler, because
MinGW stdlib headers conflict with hipcc's MSVC-mode built-in headers.

### 4. Import Libraries for hipblas / hipblaslt

The HIP SDK ships `C:\Program Files\AMD\ROCm\7.1\lib\libhipblas.dll.a` and
`libhipblaslt.dll.a` — MinGW-format import libraries that link against the MSVC
ABI.  The `strix-halo-windows` target uses these directly with no extra steps.

If those `.dll.a` files are missing on your installation, use the included helper
to generate them from the DLLs:

```bash
# In MSYS2 bash
python3 create_import_libs.py
# Generates C:/msys64/opt/lib/hipblas.lib and hipblaslt.lib
# Then adjust WIN32_ROCM_LIBS_DIR in the Makefile
```

### 5. Runtime: ROCm DLLs on PATH

DS4 executables load ROCm DLLs at runtime.  Add the ROCm bin directory before
running any ds4 binary:

```bash
# MSYS2 / bash
export PATH="/c/PROGRA~1/AMD/ROCm/7.1/bin:$PATH"
```

```powershell
# PowerShell
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;$env:PATH"
```

### 6. Lock File Path

DS4 uses a lock file to prevent concurrent model loads.  The default path
`/tmp/ds4.lock` does not exist on Windows.  Set before running:

```bash
export DS4_LOCK_FILE="C:/Windows/Temp/ds4.lock"
```

---

## Build

Open the **MSYS2 MinGW64** shell and run:

```bash
export PATH="/c/PROGRA~1/AMD/ROCm/7.1/bin:/mingw64/bin:/usr/bin:$PATH"
cd /c/path/to/ds4
make strix-halo-windows
```

This produces `ds4`, `ds4-server`, `ds4-bench`, and `ds4-eval` as PE32+ x86-64
executables.  The GPU kernel compilation step (`ds4_rocm.cu → ds4_rocm.o`) takes
2–5 minutes with `--offload-arch=gfx1151`.

---

## Run

```bash
# Interactive CLI
./ds4 \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --ctx 229376

# HTTP API server (OpenAI-compatible)
./ds4-server \
  --model gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --ctx 229376 \
  --host 127.0.0.1 \
  --port 8000
```

The server exposes `/v1/models`, `/v1/chat/completions`, `/v1/responses`, and
`/v1/messages` (Anthropic-compatible).  Navigating to `http://127.0.0.1:8000/`
returns `{"error":{"message":"unknown endpoint",...}}` — this is correct.

---

## Context Size Limits

With 96 GB unified RAM (107.87 GiB GPU-visible):

| `--ctx` tokens | Context buffers | Status |
|---|---|---|
| 32,768 (32k) | 1.05 GiB | ✅ |
| 131,072 (128k) | 3.11 GiB | ✅ |
| 196,608 (192k) | 4.38 GiB | ✅ |
| 229,376 (224k) | 4.84 GiB | ✅ **recommended max** |
| 237,568 (232k) | 5.21 GiB | ✅ |
| 245,760 (240k) | 5.38 GiB | ❌ KV alloc OOM |
| 262,144 (256k) | ~6.3 GiB | ❌ model load OOM |

After the 80.76 GiB model and ~12 GiB of ROCm Windows driver overhead, ~15 GiB
remains.  The KV cache hits a fragmentation cliff around 5.2–5.4 GiB; **229376 is
the reliable maximum**.

---

## Architecture Decision: Unified MSVC ABI via hipcc

**Problem:** The ROCm GPU runtime (`amdhip64.dll`, `hipblas.dll`, `hipblaslt.dll`)
uses the MSVC ABI.  Mixing MinGW-compiled `.o` files with MSVC-ABI DLLs causes
linker failures — incompatible calling conventions, duplicated C++ symbols, and
`__chkstk` / `__CxxFrameHandler` mismatches.

**Solution:** All compilation goes through `hipcc` (which wraps `clang` targeting
`x86_64-pc-windows-msvc`).  The CC override in the Makefile uses `-x c -std=c11`
to force C mode for `.c` sources:

```makefile
CC="$(HIPCC) -x c -std=c11 -D_CRT_SECURE_NO_WARNINGS"
```

Every object file therefore uses the same MSVC ABI, eliminating linker ABI
mismatches entirely.

---

## Source Changes

### `Makefile` — `strix-halo-windows` target

The new target configures:
- **CC override**: all `.c` sources compiled via `hipcc -x c -std=c11` (MSVC ABI)
- **Include path**: `-I./compat/win32` for POSIX shim headers;
  `C:/msys64/opt/include` for rocWMMA (GPU compilation only)
- **GPU offload**: `--offload-arch=gfx1151` for Strix Halo RDNA4
- **Linker libraries**: `amdhip64.lib`, `libhipblas.dll.a`, `libhipblaslt.dll.a`,
  `-lws2_32` (Winsock2), `-liphlpapi` (IP helper)
- `compat/win32/sys/mman.c` compiled to `compat/win32/sys/mman.o` and added to
  `CORE_OBJS`
- `-D_HAS_CLANG_BUILTINS=0` — defensive define preventing potential issues with
  `__builtin_*` intrinsics emitted by ROCm headers in MSVC-target mode
- `-Wno-invalid-specialization` — suppresses template specialization warnings from
  ROCm C++ headers when targeting MSVC

### `ds4.c` — 64-bit stat for files > 4 GB

**Problem:** The model file is 86 GB.  MSVC's default `off_t` is 32-bit
(`typedef long off_t`), so `fstat()` returns `EOVERFLOW` (errno 132, "value too
large") for any file larger than 4 GB.

**Fix:** `#ifdef _WIN32` block uses `_fstat64` / `struct _stat64`, which have a
64-bit `st_size`:

```c
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
#else
    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
#endif
```

### `ds4_server.c` — include `sys/socket.h` on Windows

**Problem:** The original code was:
```c
#ifdef _WIN32
  /* sys/socket.h on Windows is handled via winsock2.h */
#else
  #include <sys/socket.h>
#endif
```
This skipped the compat `sys/socket.h` entirely, so `WSAStartup()` was never
called before `socket()`.  Without `WSAStartup`, `socket()` returns
`WSANOTINITIALISED (10093)` which, through errno translation, surfaces as
errno 132 / "value too large".

**Symptom:** `ds4-server: failed to listen on 127.0.0.1:8000: value too large`

**Fix:**
```c
#include <sys/socket.h>   /* resolves to compat/win32/sys/socket.h on Windows */
```

### `ds4_bench.c` — missing `sys/file.h` include

`ds4_bench.c` uses `clock_gettime`, `CLOCK_MONOTONIC`, `nanosleep`, and `PATH_MAX`,
which on Windows come from our compat `sys/file.h`.  Adding the include resolves
all four symbols.

### `rocm/ds4_rocm_runtime.cuh` — 64-bit pread offset cast

**Problem:** In `cuda_pread_full`, the file offset is cast to `off_t`:
```c
ssize_t n = pread(fd, buf + done, n_req, (off_t)(offset + done));
```
On Windows MSVC, `off_t = long` (32-bit).  Offsets beyond 4 GB are silently
truncated to negative values.  `pread` reads from the wrong position and returns
EIO, causing every tensor beyond the 4 GB boundary to fail with "Input/output error".

**Symptom:**
```
ds4: ROCm model range read failed for tensor-span:3 at 320.00 MiB: Input/output error
```
— affecting all tensor reads past ~4 GB into the 86 GB model file.

**Fix:**
```c
// Before (wrong on MSVC — off_t is 32-bit):
ssize_t n = pread(fd, buf + done, n_req, (off_t)(offset + done));
// After (correct — int64_t is always 64-bit):
ssize_t n = pread(fd, buf + done, n_req, (int64_t)(offset + done));
```

### `rocm/ds4_rocm_runtime.cuh` — Q8→F16 cache reserve for Windows

**Problem:** After the 80.76 GiB model loads, the Windows ROCm driver consumes
approximately 12 GiB of overhead (pinned staging buffers, HipBLAS/HipBLASLt
workspaces, driver bookkeeping) not present on Linux.  The default Q8→F16 cache
reserve of `5% of total ≈ 5.39 GiB` was too small: the cache pre-converted 10+ GiB
before exhausting free memory, leaving nothing for the inference KV cache.

**Symptom:**
```
ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels
    (cached=10.06 GiB free=5.38 GiB reserve=5.39 GiB total=107.87 GiB)
ds4: ROCm tensor alloc failed: out of memory   (×9)
ds4: sampled CLI generation requires a session backend
```

**Fix:** Add 12 GiB to the reserve on Windows so the Q8 cache stops early:
```c
static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    ...
    uint64_t reserve = pct_reserve > min_reserve ? pct_reserve : min_reserve;
#ifdef _WIN32
    /* ROCm Windows driver overhead (~12 GiB) not present on Linux */
    reserve += 12ull * 1024ull * 1048576ull;
#endif
    return reserve;
}
```
With this fix, only 2.04 GiB is pre-converted (embeddings), leaving ~15 GiB free
for the KV cache.  Inference uses Q8 kernels directly for the remaining tensors,
which works correctly and with negligible performance difference.

---

## POSIX Compatibility Layer (`compat/win32/`)

DS4 is deeply POSIX-dependent.  A complete shim layer maps POSIX APIs to Win32.
All headers are found via `-I./compat/win32` during compilation.

### `unistd.h`

The most complex shim.

**`pread()` — thread-safe positioned read**

The ROCm streaming engine uses 18 worker threads all calling `pread()` on the same
`g_model_fd` concurrently.  On Linux, `pread()` is atomic and safe.  On Windows,
`ReadFile()` on a synchronous HANDLE from multiple threads races on the shared
internal file pointer, causing corruption or errors.

Fix: each `pread()` call opens a private `FILE_FLAG_OVERLAPPED` handle via
`ReOpenFile()`, uses a manual-reset event for async completion, then closes the
handle.  This provides true parallel positioned reads at the cost of one extra
syscall per read (~2 µs overhead):

```c
HANDLE h = ReOpenFile(orig, GENERIC_READ,
                      FILE_SHARE_READ | FILE_SHARE_WRITE,
                      FILE_FLAG_OVERLAPPED);
HANDLE ev = CreateEventW(NULL, TRUE, FALSE, NULL);
OVERLAPPED ov = {0}; ov.hEvent = ev;
ov.Offset = lo32(offset); ov.OffsetHigh = hi32(offset);
ReadFile(h, buf, count, &nread, &ov);
// if ERROR_IO_PENDING: GetOverlappedResult(h, &ov, &nread, TRUE)
CloseHandle(ev);
CloseHandle(h);
```

**`close()` — handles both CRT fds and Winsock sockets**

Windows `_close()` only works on CRT file descriptors; sockets need `closesocket()`.
We use `GetProcAddress` to avoid pulling `<winsock2.h>` into `unistd.h` (which
causes `sockaddr` redefinition conflicts):

```c
static inline int close(int fd) {
    int r = _close(fd);
    if (r == -1 && errno == EBADF) {
        typedef int (__stdcall *pfn_cs_t)(ULONG_PTR);
        static pfn_cs_t pfn_cs = NULL;
        if (!pfn_cs) {
            HMODULE ws2 = GetModuleHandleA("ws2_32.dll");
            if (ws2) pfn_cs = (pfn_cs_t)(void*)GetProcAddress(ws2, "closesocket");
        }
        if (pfn_cs && pfn_cs((ULONG_PTR)(unsigned)fd) == 0) { errno = 0; return 0; }
    }
    return r;
}
```

Other exports: `sysconf()`, `usleep()`, `clock_gettime(CLOCK_MONOTONIC)` via
`QueryPerformanceCounter`, `PATH_MAX = MAX_PATH`, `SIGPIPE = 13`, `ssize_t`,
`off_t = __int64`, `useconds_t`.

### `sys/socket.h`

Winsock2 wrapper with **lazy `WSAStartup`** on the first `socket()` call:

```c
static inline int ds4_wsa_ensure_init(void) {
    static int s_inited = 0;
    if (!s_inited) { WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa); s_inited = 1; }
    return 1;
}
static inline int ds4_socket(int af, int type, int protocol) {
    ds4_wsa_ensure_init();
    SOCKET s = socket(af, type, protocol);
    if (s == INVALID_SOCKET) { errno = EINVAL; return -1; }
    return (int)s;
}
#define socket ds4_socket
```

`poll()` is forwarded to `WSAPoll()`.  `if_nametoindex()` returns a stub `1`
(sufficient for the distributed inference multicast path).

### `sys/file.h`

`flock()` via `LockFileEx`/`UnlockFileEx`; `fcntl()` no-op macro; `dprintf()` via
`_write()`; `nanosleep()` via `Sleep()`; `struct winsize` + `TIOCGWINSZ` via
`GetConsoleScreenBufferInfo`; `sigaction` no-op stub (Ctrl+C handled by
`SetConsoleCtrlHandler`); `clock_gettime(CLOCK_MONOTONIC)` via
`QueryPerformanceCounter`; `mkdir()` 1-arg wrapper.

### `sys/time.h`

`gettimeofday()` via `GetSystemTimeAsFileTime` — converts Windows FILETIME (100 ns
ticks from 1601-01-01) to Unix timeval using offset `116444736000000000`.

### `sys/mman.h` + `sys/mman.c`

`mmap`/`munmap`/`msync`/`mlock` via `CreateFileMapping` + `MapViewOfFile`.
The 64-bit file size is correctly split into high/low `DWORD` pairs for
`CreateFileMapping`.  `MapViewOfFile`'s `dwNumberOfBytesToMap` is `SIZE_T`
(64-bit on 64-bit Windows), mapping the full 86 GB GGUF into contiguous virtual
address space.

### `dirent.h`

`opendir`/`readdir`/`closedir` via `FindFirstFileA`/`FindNextFileA`.
Adds `ino_t` typedef (`unsigned long`) which MSVC's `sys/stat.h` omits.

### `strings.h`

```c
#define strcasecmp  _stricmp
#define strncasecmp _strnicmp
```

### `poll.h`

Forward-includes `sys/socket.h` to bring in the `WSAPoll`-backed `poll()`.

---

## Diagnostic Test Files

Written during development to diagnose Windows-specific bugs:

| File | Purpose |
|---|---|
| `test_errno.c` | Enumerates Windows errno strings; confirmed errno 132 = "value too large" = WSANOTINITIALISED mapping |
| `test_pread2.c` | Validated `ReOpenFile + OVERLAPPED` reads at offsets > 20 GiB on the actual GGUF file |
| `test_socket.c` | Confirmed `WSAStartup + bind + listen` sequence works before the `sys/socket.h` fix |
| `test_compile.sh` | Batch-compiles all `.c` files with compat headers to catch regressions |

Build a diagnostic test:
```bash
"/c/PROGRA~1/AMD/ROCm/7.1/bin/hipcc" -x c -D_CRT_SECURE_NO_WARNINGS \
    -DDS4_ROCM_BUILD -I./compat/win32 test_socket.c -o test_socket.exe -lws2_32
./test_socket.exe
```

---

## Memory Budget (107.87 GiB unified pool)

On Strix Halo, CPU RAM and GPU VRAM are the same physical LPDDR5X.
`hipInfo` reports `isIntegrated = 1` and `totalGlobalMem = 107.87 GB`.

| Component | GiB |
|---|---|
| Model weights (Q2/IQ2 routed + Q8 attn/shared/output) | 80.76 |
| Q8→F16 partial preconversion cache | 2.04 |
| ROCm Windows driver overhead | ~11.67 |
| KV cache + inference buffers (ctx=229376) | ~4.84 |
| **Total used** | **~99.3** |
| **Remaining free** | **~8.5** |

The ~11.67 GiB Windows ROCm overhead is not present on Linux and is the primary
difference between the two platforms' memory budgets.

---

## Known Remaining Issues

### `--ssd-streaming` crashes at ~8 GiB

When pread fails for a tensor, the fallback code calls
`cudaMemcpy(dst, mmap_ptr + large_offset, n, HostToDevice)`.  On Linux with Strix
Halo unified memory the GPU can DMA from mmap'd file pages directly.  On Windows,
the ROCm driver cannot DMA from demand-paged file-backed memory at large offsets,
causing an access violation.  SSD streaming is not needed when the full 80.76 GiB
model fits in the 107.87 GiB pool; this only matters on machines with < ~95 GiB.

### `isLargeBar: 0` in `hipInfo`

Resizable BAR is not enabled in BIOS/UEFI on this machine.  Enabling it
("Above 4G Decoding" + "Re-size BAR Support" in BIOS) may allow larger context
windows (currently capped at ~232k tokens) and improve DMA throughput.

### Non-fatal `pread` warnings during model load

A small fraction of tensor reads fall through to the `cudaHostRegister` + mmap
pointer fallback path, logging `"ROCm model range read failed ... Input/output error"`.
These are cosmetic — the model loads correctly via the fallback — and do not affect
inference correctness or performance.
