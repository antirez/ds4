ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

ARG DS4_BUILD_TARGET=cuda-generic
ARG CUDA_ARCH=
RUN make ${DS4_BUILD_TARGET} CUDA_ARCH="${CUDA_ARCH}"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /src/ds4-server /app/ds4-server
COPY download_model.sh /app/download_model.sh
COPY docker/entrypoint.sh /usr/local/bin/ds4-entrypoint

RUN chmod +x /app/download_model.sh /usr/local/bin/ds4-entrypoint \
    && mkdir -p /models /kv-cache

ENV DS4_GGUF_DIR=/models \
    DS4_MODEL=q2-imatrix \
    DS4_ENABLE_MTP=0 \
    DS4_CTX=100000 \
    DS4_KV_DISK_DIR=/kv-cache \
    DS4_KV_DISK_SPACE_MB=8192 \
    DS4_HOST=0.0.0.0 \
    DS4_PORT=8000

EXPOSE 8000
VOLUME ["/models", "/kv-cache"]

ENTRYPOINT ["/usr/local/bin/ds4-entrypoint"]
