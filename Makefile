CC ?= cc
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_distributed.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test cpu cpu-avx2 cpu-avx512 cpu-avx512-vnni cpu-simd-build cuda cuda-spark cuda-generic cuda-regression

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu-avx2     Build CPU-only with AVX2 (x86_64 only)"
	@echo "  make cpu-avx512   Build CPU-only with AVX512BW (x86_64 only)"
	@echo "  make cpu-avx512-vnni Build CPU-only with AVX512BW+VNNI (x86_64 only)"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu-avx2            Build CPU-only with AVX2"
	@echo "  make cpu-avx512          Build CPU-only with AVX512BW"
	@echo "  make cpu-avx512-vnni     Build CPU-only with AVX512BW+VNNI"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

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

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

# --- SIMD-specific CPU builds (x86_64 only, shared across Darwin/Linux) ---
X86_64_HOST := $(filter x86_64 amd64,$(UNAME_M))

ifneq ($(X86_64_HOST),)
cpu-avx2:
	$(MAKE) cpu-simd-build NATIVE_CPU_FLAG= SUFFIX=-avx2 SIMDFLAGS="-mavx2"

cpu-avx512:
	$(MAKE) cpu-simd-build NATIVE_CPU_FLAG= SUFFIX=-avx512 SIMDFLAGS="-mavx2 -mavx512f -mavx512bw"

cpu-avx512-vnni:
	$(MAKE) cpu-simd-build NATIVE_CPU_FLAG= SUFFIX=-avx512-vnni SIMDFLAGS="-mavx2 -mavx512f -mavx512bw -mavx512vnni"
else
cpu-avx2 cpu-avx512 cpu-avx512-vnni:
	@echo "error: $$@ requires an x86_64 host (detected: $(UNAME_M))"
	@false
endif

BDIR = build/cpu$(SUFFIX)

cpu-simd-build:
	@mkdir -p $(BDIR)
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_cpu.o ds4.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_cli_cpu.o ds4_cli.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_server_cpu.o ds4_server.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_bench_cpu.o ds4_bench.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_eval_cpu.o ds4_eval.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -DDS4_NO_GPU -c -o $(BDIR)/ds4_agent_cpu.o ds4_agent.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/ds4_distributed.o ds4_distributed.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/ds4_help.o ds4_help.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/ds4_web.o ds4_web.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/ds4_kvstore.o ds4_kvstore.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/linenoise.o linenoise.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -c -o $(BDIR)/rax.o rax.c
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o ds4$(SUFFIX) $(BDIR)/ds4_cli_cpu.o $(BDIR)/ds4_help.o $(BDIR)/linenoise.o $(BDIR)/ds4_cpu.o $(BDIR)/ds4_distributed.o $(LDLIBS)
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o ds4-server$(SUFFIX) $(BDIR)/ds4_server_cpu.o $(BDIR)/ds4_help.o $(BDIR)/ds4_kvstore.o $(BDIR)/rax.o $(BDIR)/ds4_cpu.o $(BDIR)/ds4_distributed.o $(LDLIBS)
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o ds4-bench$(SUFFIX) $(BDIR)/ds4_bench_cpu.o $(BDIR)/ds4_help.o $(BDIR)/ds4_cpu.o $(BDIR)/ds4_distributed.o $(LDLIBS)
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o ds4-eval$(SUFFIX) $(BDIR)/ds4_eval_cpu.o $(BDIR)/ds4_help.o $(BDIR)/ds4_cpu.o $(BDIR)/ds4_distributed.o $(LDLIBS)
	$(CC) $(CFLAGS) $(SIMDFLAGS) -o ds4-agent$(SUFFIX) $(BDIR)/ds4_agent_cpu.o $(BDIR)/ds4_help.o $(BDIR)/ds4_web.o $(BDIR)/ds4_kvstore.o $(BDIR)/linenoise.o $(BDIR)/ds4_cpu.o $(BDIR)/ds4_distributed.o $(LDLIBS)

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

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4-eval quant-dot-test q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_test

quant-dot-test: tests/test_quant_dot.c
	$(CC) $(CFLAGS) -o tests/test_quant_dot tests/test_quant_dot.c -lm -pthread
	./tests/test_quant_dot

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-avx2 ds4-avx512 ds4-avx512-vnni ds4-server ds4-server-avx2 ds4-server-avx512 ds4-server-avx512-vnni ds4-bench ds4-bench-avx2 ds4-bench-avx512 ds4-bench-avx512-vnni ds4-eval ds4-eval-avx2 ds4-eval-avx512 ds4-eval-avx512-vnni ds4-agent ds4-agent-avx2 ds4-agent-avx512 ds4-agent-avx512-vnni ds4_cpu ds4_native ds4_server_test ds4_test tests/test_quant_dot tests/test_q4k_dot *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
	rm -rf build/
