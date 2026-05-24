# Building DS4

DS4 uses a Nix flake for reproducible builds across platforms. You can also build directly with `make`.

## Prerequisites

- **Nix** with flakes enabled (`--extra-experimental-features 'nix-command flakes'`)
- A supported GPU backend:
  - **ROCm**: AMD GPU (MI300, RX 7900, etc.) — tested on Linux
  - **CUDA**: NVIDIA GPU with Blackwell support (DGX Spark / GB10)
  - **Metal**: Apple Silicon Mac (M-series)
  - **CPU**: No GPU required (diagnostics / CPU-only)

## Quick start (Nix)

```bash
# Build for your platform (auto-detected):
nix build

# Or pick a specific backend:
nix build .#ds4-rocm    # Linux + AMD ROCm
nix build .#ds4-cuda    # Linux + NVIDIA CUDA (requires --impure for unfree license)
nix build .#ds4-metal   # macOS + Metal
nix build .#ds4-cpu     # CPU-only (no GPU)
```

> **CUDA note:** CUDA packages have an unfree license. Use:
> ```bash
> NIXPKGS_ALLOW_UNFREE=1 nix build .#ds4-cuda --impure
> ```

Output goes to `result` (a symlink to `/nix/store/…`). Binaries are in `result/bin/`.

## Development shell

```bash
nix develop
```

This drops you into a shell with `make`, `gcc`, `hipcc`, and ROCm libraries available. On macOS it provides the Metal SDK headers; on Linux it provides CUDA or ROCm as appropriate.

## Building without Nix (direct Makefile)

```bash
# ROCm (Linux)
make rocm ROCM_ARCH=gfx1151

# CUDA (Linux)
make cuda-generic

# Metal (macOS)
make

# CPU-only
make cpu
```

