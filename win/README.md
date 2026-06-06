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

## Status

| Target      | Native Windows CPU build | Notes |
|-------------|--------------------------|-------|
| `ds4-bench` | builds & runs            | no terminal/socket deps |
| `ds4` (CLI) | not yet                  | `linenoise.c` uses POSIX `termios`, plus `sigaction` in `ds4_cli.c` (needs Win console raw-mode port) |
| `ds4-server`| not yet                  | BSD sockets / `poll` / `arpa/inet.h` (needs Winsock port) |

## Runtime note

Set `DS4_LOCK_FILE` to a Windows path (the default is `/tmp/ds4.lock`, which does
not exist on Windows) before running, e.g.:

```sh
export DS4_LOCK_FILE="$TEMP/ds4.lock"
```

## Deferred work

- **CLI (`ds4`)**: port `linenoise.c` to the Windows console (raw mode via
  `SetConsoleMode`) and replace the `sigaction`/`SIGINT` handling in `ds4_cli.c`.
- **Server (`ds4-server`)**: port the BSD sockets / `poll(2)` event loop to
  Winsock 2 (`WSAStartup`, `WSAPoll`, `closesocket`).
