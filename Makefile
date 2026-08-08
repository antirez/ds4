CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
SAMPLING_TEST :=
else
NATIVE_CPU_FLAG ?= -march=native
SAMPLING_TEST := tests/test_sampling
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_CUDA_HAVE_MXF4=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test test-metal-session-batch test-mxfp4-cuda test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.c ds4.h $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_official.c $(CORE_OBJS) rax.o ds4_gpu_args.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-mxfp4-metal: tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=sm_121

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_args.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	$(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
ifneq ($(UNAME_S),Darwin)
	./tests/test_sampling
endif

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_metal tests/test_mxfp4_cuda tests/test_metal_session_batch tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
