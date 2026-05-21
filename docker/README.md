# Docker

The Docker setup builds and serves `ds4-server` with the Linux CUDA backend. 
It currently does not support the ds4 Metal backend, although this should be easy enought to add.

## Requirements

- Docker with Compose v2
- NVIDIA driver compatible with CUDA 12/13 containers
- NVIDIA Container Toolkit configured for Docker GPU access
- Enough disk space for the selected GGUF model and disk KV cache

The default image uses CUDA 13, which is what most DGX Spark system are currently targeting:

- Build stage: `nvidia/cuda:13.0.3-devel-ubuntu24.04`
- Runtime stage: `nvidia/cuda:13.0.3-runtime-ubuntu24.04`

The CUDA and Ubuntu image versions are build-time parameters. The defaults are
`CUDA_VERSION=13.0.3` and `UBUNTU_VERSION=24.04`.

## Start

From the repository root:

```sh
docker compose up --build
```

On first startup the container downloads the selected model into the weights
volume, then starts `ds4-server` on port `8000`. Which model? See `DS4_MODEL` later.

The server exposes the same API as the native binary, including:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`

## Volumes

Compose bind-mounts the GGUF weights directory and disk KV cache directory:

- `${DS4_WEIGHTS_HOST_DIR:-./gguf}` mounted at `/models` for GGUF weights
- `${DS4_VOLUMES_HOST_DIR:-./volumes}/kv-cache` mounted at `/kv-cache` for disk KV checkpoints

The model downloader resumes partial downloads and skips files that are already
present in `/models`. The default weights mount is `./gguf`, matching the native
`download_model.sh` default, so models downloaded on the host are reused by the
container instead of downloaded again.

## Configuration

Compose reads a root `.env` file by default for variable interpolation. These are
the main knobs:

```env
DS4_MODEL=q2
DS4_ENABLE_MTP=0
DS4_CTX=100000
DS4_KV_DISK_SPACE_MB=8192
DS4_MTP_DRAFT=2
DS4_MTP_MARGIN=
DS4_THREADS=
DS4_EXTRA_ARGS=
DS4_WEIGHTS_HOST_DIR=./gguf
DS4_VOLUMES_HOST_DIR=./volumes
HF_TOKEN=
CUDA_VERSION=13.0.3
UBUNTU_VERSION=24.04
DS4_BUILD_TARGET=cuda-generic
CUDA_ARCH=
```

`DS4_MODEL` is passed to `download_model.sh`. Supported values at the time of writing are:

- `q2-imatrix`
- `q4-imatrix`
- `q2`
- `q4`
- `none`

Use `none` only when you provide a model path yourself with `DS4_EXTRA_ARGS` or
by overriding the container command.

`DS4_ENABLE_MTP=1` downloads the optional MTP GGUF and starts the server with
`--mtp /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.

`DS4_CTX` maps to `--ctx`.

`DS4_KV_DISK_SPACE_MB` maps to `--kv-disk-space-mb`.

`DS4_EXTRA_ARGS` is appended to the generated `ds4-server` command for advanced
server flags such as `--quality` or cache policy tuning.

`DS4_WEIGHTS_HOST_DIR` selects the host directory mounted at `/models`. The
default is `./gguf`, matching the native model downloader.

`DS4_VOLUMES_HOST_DIR` selects the host directory used for remaining persistent
bind mounts. The disk KV cache uses the `kv-cache` subdirectory below it.

`DS4_BUILD_TARGET` selects the Makefile target for the Linux CUDA build:

| Value | Effect |
|---|---|
| `cuda-generic` (default) | `make CUDA_ARCH=native` — targets the visible GPU via `nvcc -arch=native` |
| `cuda-spark` | `make CUDA_ARCH=` — omits `-arch` for DGX Spark / GB10 |
| `cuda` | Use with `CUDA_ARCH=sm_N` for an explicit architecture override |

`CUDA_ARCH` is passed alongside `DS4_BUILD_TARGET` and is only consumed when
`DS4_BUILD_TARGET=cuda`. For the other targets the variable is silently ignored.

`CUDA_VERSION` and `UBUNTU_VERSION` select the NVIDIA CUDA base images used for
both build and runtime stages. Keep `CUDA_VERSION` compatible with the host
NVIDIA driver. A host where `nvidia-smi` reports `CUDA Version: 13.0` should use
a CUDA `13.0.x` container, not `13.1.x`.

## Examples

Use the default q2 model:

```sh
docker compose up --build
```

Enable MTP and use a larger disk KV cache:

```sh
DS4_ENABLE_MTP=1 DS4_KV_DISK_SPACE_MB=32768 docker compose up --build
```

Use q4 weights for the lucky RAM owners:

```sh
DS4_MODEL=q4 docker compose up --build
```

Set a larger context window:

```sh
DS4_CTX=250000 docker compose up --build
```

Store weights and disk KV cache under different host directories:

```sh
DS4_WEIGHTS_HOST_DIR=/data/ds4/gguf DS4_VOLUMES_HOST_DIR=/data/ds4 docker compose up --build
```

Pass extra server flags:

```sh
DS4_EXTRA_ARGS="--quality --kv-cache-min-tokens 1024" docker compose up --build
```

Build for DGX Spark / GB10 (omits explicit `nvcc -arch`):

```sh
DS4_BUILD_TARGET=cuda-spark docker compose up --build
```

Build with an explicit CUDA architecture override:

```sh
DS4_BUILD_TARGET=cuda CUDA_ARCH=sm_120 docker compose up --build
```

Build against a different compatible CUDA container version:

```sh
CUDA_VERSION=13.0.3 UBUNTU_VERSION=24.04 docker compose build
```

## Authentication

Public downloads normally do not require authentication. If Hugging Face requires
a token, set `HF_TOKEN` in the environment or in the root `.env` file.

## Notes

The first startup can take a long time because the q2 model is roughly 81 GB and
the q4 model is roughly 153 GB. Keep the weights volume mounted so subsequent
starts reuse the existing GGUF files.
