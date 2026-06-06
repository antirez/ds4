CC ?= cc
UNAME_S := $(shell uname -s)

# On MinGW/MSYS `uname -s` is e.g. MINGW64_NT-10.0 or MSYS_NT-10.0.
IS_WINDOWS := $(filter MINGW% MSYS%,$(UNAME_S))

# MinGW has no `cc`; default the compiler to gcc there (still overridable).
ifneq ($(IS_WINDOWS),)
ifeq ($(origin CC),default)
CC := gcc
endif
endif

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

# Native Windows (MinGW-w64) CPU build flags. ds4.c pulls in the dependency-free
# POSIX shim (ds4_win.h) behind #ifdef _WIN32; no extra -I/-include is needed.
WIN_CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -std=c99 -D_GNU_SOURCE \
	-fno-finite-math-only -DDS4_NO_GPU -D_CRT_SECURE_NO_WARNINGS
WIN_LDLIBS ?= -lm

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o
else

CFLAGS += -D_GNU_SOURCE -fno-finite-math-only

ifeq ($(GPU_BACKEND),rocm)
ROCM_PATH ?= /opt/rocm
GPU_CC = $(ROCM_PATH)/bin/hipcc
ROCM_ARCH ?= gfx1151

GPU_CFLAGS ?= -O3 -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
GPU_LDLIBS = -lm -pthread -L$(ROCM_PATH)/lib -lhipblas

@echo "ROCM_ARCH: $(ROCM_ARCH)"

EXTRA_DEPS = ds4_rocm.h

else

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas

GPU_CC = $(NVCC)
GPU_CFLAGS = $(NVCCFLAGS)
GPU_LDLIBS = $(CUDA_LDLIBS)

endif

CORE_OBJS = ds4.o ds4_cuda.o
EXTRA_DEPS =
CPU_CORE_OBJS = ds4_cpu.o
METAL_LDLIBS := $(LDLIBS)

endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression windows-cpu

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"

else ifneq ($(IS_WINDOWS),)
# ---- Native Windows (MinGW-w64 / HIP-clang) -------------------------------
# CPU bench builds with MinGW. The GPU (ROCm/HIP) bench builds with the AMD HIP
# SDK for gfx1151. The CLI (linenoise/termios + sigaction) and server (BSD
# sockets/poll) still need Windows ports; see win/README.md.
#
# Windows ROCm/HIP build settings. hipcc.exe's .bat wrapper splits args on
# spaces, so the actual compile/link is delegated to win/build-rocm.sh, which
# relies on the SDK's default include search and a space-free import-lib dir.
ROCM_PATH ?= C:/Program Files/AMD/ROCm/7.1
ROCM_ARCH ?= gfx1151

all: help

help:
	@echo "DS4 build targets (native Windows):"
	@echo "  make windows-cpu    Build native Windows CPU   ./ds4-bench.exe (MinGW)"
	@echo "  make windows-rocm   Build native Windows ROCm  ./ds4-bench.exe (HIP, gfx1151)"
	@echo "  make clean          Remove build outputs"
	@echo ""
	@echo "  windows-rocm uses the AMD HIP SDK (ROCM_PATH=$(ROCM_PATH),"
	@echo "  ROCM_ARCH=$(ROCM_ARCH)). See win/README.md for the rocWMMA vendoring"
	@echo "  step and run caveats."
	@echo ""
	@echo "  ds4 (CLI) and ds4-server are not yet ported to Windows."

windows-cpu: ds4-bench.exe

ds4-bench.exe: ds4_bench.c ds4.c ds4.h ds4_gpu.h ds4_win.h
	$(CC) $(WIN_CFLAGS) -c -o ds4_cpu.o ds4.c
	$(CC) $(WIN_CFLAGS) -c -o ds4_bench_cpu.o ds4_bench.c
	$(CC) $(WIN_CFLAGS) -o $@ ds4_bench_cpu.o ds4_cpu.o $(WIN_LDLIBS)

# Native Windows ROCm/HIP ds4-bench.exe (gfx1151). Delegates to the build
# script to work around hipcc.exe's space-splitting argument wrapper.
.PHONY: windows-rocm
windows-rocm:
	ROCM_PATH="$(ROCM_PATH)" ROCM_ARCH="$(ROCM_ARCH)" bash win/build-rocm.sh

else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, and ./ds4-bench"
	@echo "  make rocm                Build ROCm"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH=

cuda-generic:
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench CUDA_ARCH="$(CUDA_ARCH)"

rocm:
	@if [ -z "$(strip $(ROCM_ARCH))" ]; then \
		echo "error: specify ROCM_ARCH, for example: make rocm ROCM_ARCH=gfx1151"; \
		exit 2; \
	fi
	$(MAKE) ds4 ds4-server ds4-bench GPU_BACKEND=rocm ROCM_ARCH=$(ROCM_ARCH)


ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(GPU_CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(GPU_CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(GPU_CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(EXTRA_DEPS)
	$(GPU_CC) $(GPU_CFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(GPU_CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4_cpu ds4_native ds4_server_test ds4_test *.o *.exe tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o

