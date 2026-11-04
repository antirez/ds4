# syntax=docker/dockerfile:1
#
# ds4 (DwarfStar) engine for DeepSeek V4 Flash, built for Strix Halo (gfx1151).
#
# Produces a flat /opt/ds4 layout holding all five binaries, mirroring the
# /opt/llama layout the llama.cpp images use:
#   /opt/ds4/ds4          coordinator / worker / CLI
#   /opt/ds4/ds4-server   OpenAI + Anthropic compatible HTTP server
#   /opt/ds4/ds4-bench  /opt/ds4/ds4-eval  /opt/ds4/ds4-agent
#
# Build from the ds4 source tree (it is the build context):
#   cd ~/ds4
#   sudo docker build -t ds4:rocm-gfx1151 .
#
# The .dockerignore beside this file MUST exclude gguf/ -- it holds ~159 GB of
# model files and would otherwise be sent to the daemon as build context.

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full

# ---------------------------------------------------------------- build stage
FROM ${ROCM_IMAGE} AS build

ARG ROCM_ARCH=gfx1151
# Host CPU target for the C code.  -march=native suits a build on the machine
# that will run the image; CI runners have a different CPU, so the workflow
# passes an explicit Strix Halo compatible target (-march=znver4: Zen 5 runs
# every Zen 4 instruction, and this image's GCC 13 predates -march=znver5).
ARG NATIVE_CPU_FLAG=-march=native

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}

WORKDIR /src
COPY . /src

# `strix-halo` is the ROCm target (`rocm` is an alias for it). ROCM_ARCH already
# defaults to gfx1151 in the Makefile; it is passed explicitly so the image can
# be retargeted with --build-arg without editing the Makefile.
#
# STRIXHALO.md documents installing a complete rocWMMA header tree by hand
# because Ubuntu's librocwmma-dev omits rocwmma/internal/. That workaround is
# NOT needed here: this ROCm image already ships rocwmma/internal.
RUN make strix-halo -j"$(nproc)" ROCM_ARCH=${ROCM_ARCH} NATIVE_CPU_FLAG="${NATIVE_CPU_FLAG}"

RUN mkdir -p /opt/ds4 && \
    cp ds4 ds4-server ds4-bench ds4-eval ds4-agent /opt/ds4/ && \
    cp -a dir-steering /opt/ds4/dir-steering && \
    test -x /opt/ds4/ds4-server && test -x /opt/ds4/ds4

# -------------------------------------------------------------- runtime stage
FROM ${ROCM_IMAGE}

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/ds4 /opt/ds4

ENV ROCM_PATH=/opt/rocm
# The binaries link against libhipblas / libhipblaslt in /opt/rocm/lib.
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/ds4:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ds4 resolves a few relative paths (e.g. dir-steering/) from the working
# directory, so keep it at the install root.
WORKDIR /opt/ds4

ENTRYPOINT ["/opt/ds4/ds4-server"]
