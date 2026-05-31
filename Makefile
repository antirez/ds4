CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

# --- Toolchain selection ---
ifeq ($(UNAME_S),Darwin)
LINK      ?= $(CC)
LINKFLAGS ?= $(CFLAGS)
CORE_OBJS  = ds4.o ds4_distributed.o ds4_metal.o
CPU_CORE   = ds4_cpu.o ds4_distributed.o
LDFLAGS   := $(LDLIBS) -framework Foundation -framework Metal
else

CFLAGS += -D_GNU_SOURCE -fno-finite-math-only

ifeq ($(GPU_BACKEND),rocm)
ROCM_PATH ?= /opt/rocm
GPU_CC = $(ROCM_PATH)/bin/hipcc
ROCM_ARCH ?= gfx1151
GPU_CFLAGS ?= -O3 -ffast-math -fno-finite-math-only -pthread -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)

GPU_LDLIBS = -lm -pthread -L$(ROCM_PATH)/lib -lhipblas
EXTRA_DEPS = ds4_rocm.h
else

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
GPU_CC = $(NVCC)
GPU_CFLAGS = $(NVCCFLAGS)
GPU_LDLIBS = $(CUDA_LDLIBS)
EXTRA_DEPS =
endif

CORE_OBJS = ds4.o ds4_distributed.o ds4_cuda.o
CPU_CORE  = ds4_cpu.o ds4_distributed.o
LDFLAGS   := $(GPU_LDLIBS)
LINK      ?= $(GPU_CC)
LINKFLAGS ?= $(GPU_CFLAGS)
endif

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression rocm

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent
help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"
else
all: help
help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make rocm                Build ROCm"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"
endif

# --- Convenience GPU build targets ---
ifneq ($(UNAME_S),Darwin)
cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

rocm:
	@if [ -z "$(strip $(ROCM_ARCH))" ]; then \
		echo "error: specify ROCM_ARCH, for example: make rocm ROCM_ARCH=gfx1151"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4_test ds4-eval ds4-agent GPU_BACKEND=rocm ROCM_ARCH=$(ROCM_ARCH)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

# --- Binary targets (single set of rules, toolchain chosen above) ---
ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

# --- CPU-only build ---
cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_kvstore.o rax.o $(CPU_CORE) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o $(CPU_CORE) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE) $(LDLIBS)

# --- Object compilation rules ---
ds4.o: ds4.c ds4.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(EXTRA_DEPS)
	$(GPU_CC) $(GPU_CFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(GPU_CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(LINK) $(LINKFLAGS) -o $@ $^ $(LDFLAGS)

test:
	@if [ ! -x ds4_test ]; then echo "ds4_test not found — rebuild with the appropriate GPU target first"; exit 2; fi
	@if [ ! -x ds4-eval ]; then echo "ds4-eval not found — rebuild with the appropriate GPU target first"; exit 2; fi
	./ds4-eval --self-test-extractors
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test tests/test_q4k_dot *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
