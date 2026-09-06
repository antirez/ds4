CC ?= cc
UNAME_S := $(shell uname -s)
.DEFAULT_GOAL := all

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif
SAMPLING_TEST := tests/test_sampling
GLM53_KDA_TEST := tests/test_glm53_kda
GLM53_KDA_ROCM_TEST := tests/test_glm53_kda_rocm

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
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
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
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/ds4_mmq_q4_16warp.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_HOST_CFLAGS ?= -fPIC
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lrocblas
ROCM_MMQ_Y ?= 64
ROCM_MMQ_FLAGS := $(ROCM_CFLAGS) -std=c++17 -DGGML_USE_HIP -DDS4_HIP_MMQ_Y=$(ROCM_MMQ_Y) $(MMQ_INCLUDES)
ROCM_MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o cuda/mmq/quantize.rocm.o cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o cuda/mmq/d2r_stubs.rocm.o
ROCM_CORE_OBJS := ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o $(ROCM_MMQ_OBJS)
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test test-ssd environment-docs test-quantizer-indexer-q4 test-rocm test-glm53-kda-rocm test-metal-session-batch test-metal-session-batch-ssd test-metal-q4-streams test-metal-q4-prefill-pair test-metal-indexer-q4 test-metal-q4-attn-exactn test-metal-q4-attn-out-a-direct test-metal-q4-qb-f16-cache test-metal-q4-qb-f16-cache-timing test-metal-exactn-oracle test-metal-dspark-capture test-metal-argmax-top1 bench-metal-argmax-top1 test-metal-iq2-midonly test-metal-iq2-ssd-grouped-mm test-metal-iq2-live-index test-mxfp4-metal test-mxfp4-cuda test-mxfp4-rocm test-mmq-parity-cuda test-mmq-q4-grouped-q81-cuda test-mmq-q4-16warp-cuda test-rocm-q4-parity test-rocm-q4-dense test-rocm-q4-pair test-rocm-q4-prefill test-strix-rocm-q4-parity test-strix-rocm-q4-prefill test-strix-rocm-q4-prefill-long test-cuda-session-batch test-cuda-mixed-batch dspark-acceptance dspark-verify-depth rocm-dspark-acceptance rocm-dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm cuda-iq2-moe-prefill-bench cuda-q4-prefill-bench rocm-iq2-moe-prefill-bench rocm-q4-prefill-bench

gguf-tools/deepseek4-quantize: gguf-tools/deepseek4-quantize.c gguf-tools/quants.c gguf-tools/quants.h
	$(MAKE) -C gguf-tools deepseek4-quantize

tests/test_quantizer_indexer_q4: tests/test_quantizer_indexer_q4.c gguf-tools/quants.c gguf-tools/quants.h
	$(CC) -O2 -Wall -Wextra -std=c99 -Igguf-tools -o $@ tests/test_quantizer_indexer_q4.c gguf-tools/quants.c $(LDLIBS)

test-quantizer-indexer-q4: gguf-tools/deepseek4-quantize tests/test_quantizer_indexer_q4
	./tests/test_quantizer_indexer_q4 ./gguf-tools/deepseek4-quantize

.PHONY: test-q8-quantize-host test-cuda-q8-quantize bench-cuda-q8-prefill-quantize
.PHONY: test-q4-epilogue-host test-cuda-q4-epilogue
tests/test_q4_epilogue_host: tests/test_cuda_q4_epilogue.cpp cuda/mmq/ds4_q4_mmvq_epilogue.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

test-q4-epilogue-host: tests/test_q4_epilogue_host
	./tests/test_q4_epilogue_host

tests/test_q8_quantize_host: tests/test_cuda_q8_quantize.cpp cuda/ds4_q8_prefill_layout.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

test-q8-quantize-host: tests/test_q8_quantize_host
	./tests/test_q8_quantize_host

tests/test_cuda_q8_quantize: tests/test_cuda_q8_quantize.cpp cuda/ds4_q8_quantize.cuh cuda/ds4_q8_prefill_layout.h
	@if [ -z "$(NVCC)" ] || ! command -v "$(NVCC)" >/dev/null 2>&1; then \
		echo "error: CUDA q8 quantizer tests require nvcc and a CUDA host"; exit 1; fi
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu -o $@ $<

test-cuda-q8-quantize: tests/test_cuda_q8_quantize
	./tests/test_cuda_q8_quantize

bench-cuda-q8-prefill-quantize: tests/test_cuda_q8_quantize
	./tests/test_cuda_q8_quantize --bench

.PHONY: test-q4-prefill-dequant-host test-cuda-q4-prefill-dequant test-rocm-q4-prefill-dequant bench-cuda-q4-prefill-dequant bench-rocm-q4-prefill-dequant
Q4_PREFILL_DEQUANT_HEADERS := cuda/ds4_q4_dequant_layout.h cuda/ds4_q4_dequant_vec.cuh

tests/test_q4_prefill_dequant_host: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	$(CXX) -O2 -Wall -Wextra -std=c++17 -o $@ $<

test-q4-prefill-dequant-host: tests/test_q4_prefill_dequant_host
	./tests/test_q4_prefill_dequant_host

tests/test_cuda_q4_prefill_dequant: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	@if [ -z "$(NVCC)" ] || ! command -v "$(NVCC)" >/dev/null 2>&1; then \
		echo "error: native Q4 dequant tests require nvcc"; exit 1; fi
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu -o $@ $<

tests/test_rocm_q4_prefill_dequant: tests/test_q4_prefill_dequant.cpp $(Q4_PREFILL_DEQUANT_HEADERS)
	@if [ -z "$(HIPCC)" ] || ! command -v "$(HIPCC)" >/dev/null 2>&1; then \
		echo "error: native Q4 dequant tests require hipcc"; exit 1; fi
	$(HIPCC) $(ROCM_CFLAGS) -std=c++17 -x hip -o $@ $<

test-cuda-q4-prefill-dequant: tests/test_cuda_q4_prefill_dequant
	./tests/test_cuda_q4_prefill_dequant

test-rocm-q4-prefill-dequant: tests/test_rocm_q4_prefill_dequant
	./tests/test_rocm_q4_prefill_dequant

bench-cuda-q4-prefill-dequant: tests/test_cuda_q4_prefill_dequant
	./tests/test_cuda_q4_prefill_dequant --bench

bench-rocm-q4-prefill-dequant: tests/test_rocm_q4_prefill_dequant
	./tests/test_rocm_q4_prefill_dequant --bench

.PHONY: test-metal-raw-kv-ring test-cuda-raw-kv-ring test-rocm-raw-kv-ring test-metal-moe-contracts test-rocm-iq2-prefill test-rocm-mxfp4-lds-ownership test-cuda-kernel-contracts test-gpu-raw-kv-ownership

tests/test_metal_raw_kv_ring.o: tests/test_gpu_raw_kv_ring.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_raw_kv_ring: tests/test_metal_raw_kv_ring.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-raw-kv-ring: tests/test_metal_raw_kv_ring
	./tests/test_metal_raw_kv_ring

tests/test_cuda_raw_kv_ring.o: tests/test_gpu_raw_kv_ring.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_cuda_raw_kv_ring: tests/test_cuda_raw_kv_ring.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-raw-kv-ring: tests/test_cuda_raw_kv_ring
	./tests/test_cuda_raw_kv_ring

tests/test_rocm_raw_kv_ring.o: tests/test_gpu_raw_kv_ring.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_raw_kv_ring: tests/test_rocm_raw_kv_ring.o ds4_image.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-raw-kv-ring: tests/test_rocm_raw_kv_ring
	./tests/test_rocm_raw_kv_ring

test-metal-moe-contracts: tests/test_metal_ssd_decode_kernels
	./tests/test_metal_ssd_decode_kernels --moe-contracts

tests/test_rocm_iq2_prefill.o: tests/test_rocm_iq2_prefill.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_iq2_prefill: tests/test_rocm_iq2_prefill.o ds4_image.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-iq2-prefill: tests/test_rocm_iq2_prefill
	./tests/test_rocm_iq2_prefill

test-rocm-mxfp4-lds-ownership:
	python3 tests/test_rocm_mxfp4_lds_ownership.py

test-cuda-kernel-contracts:
	python3 tests/test_cuda_kernel_contracts.py

test-gpu-raw-kv-ownership:
	python3 tests/test_gpu_raw_kv_ownership.py

ifeq ($(UNAME_S),Darwin)
.PHONY: metal-decode-schedule-bench metal-prefill-variant-bench metal-q4-dense-pair-bench metal-q4-prefill-pair-bench metal-q4-mm-tail-cull-bench metal-q4-attn-out-a-direct-bench metal-iq2-moe-tail-cull-bench metal-iq2-moe-top8-pair-bench check-mxfp4-half-lut test-mxfp4-metal
.PHONY: test-metal-moe-prefill test-metal-dense-mpp

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make test-ssd     Run the model suite with cold SSD streaming"
	@echo "  make test-quantizer-indexer-q4  Check direct F16-to-Q4_K indexer conversion"
	@echo "  make environment-docs  Generate and verify the environment variable inventory"
	@echo "  make test-metal-session-batch-ssd  Exact-logit Metal SSD union control/candidate oracle"
	@echo "  make test-metal-q4-streams  Check resident Q4 Metal stream overlap"
	@echo "  make test-metal-q4-prefill-pair  Runtime oracle for the M1-M4 Q4 prefill pair"
	@echo "  make test-metal-indexer-q4  Check the production-shape Q4_K indexer projection"
	@echo "  make test-metal-q4-attn-exactn  Bitwise/canary oracle for M1-M4 SSD-prefill Q4 attention output"
	@echo "  make test-metal-q4-attn-out-a-direct  Production-shape oracle for M1-M4 Q4 output-A direct routing"
	@echo "  make test-metal-q4-qb-f16-cache  Oracle for M1-M4 Q4 q_b sidecar and transient F16 paths"
	@echo "  make test-metal-q4-qb-f16-cache-timing  Compare Q4 direct, sidecar, and transient production at N=4096"
	@echo "  make test-metal-dspark-capture  Check fused DSpark HC capture bitwise"
	@echo "  make test-metal-argmax-top1  Check the resident production-shape Metal decode argmax"
	@echo "  make bench-metal-argmax-top1  Time full argsort versus resident top-1 without GGUF/SSD"
	@echo "  make test-metal-iq2-midonly  Check M1 IQ2 addr mid-only output and sentinels"
	@echo "  make test-metal-ssd-decode-kernels  Bitwise Q4/Q8 SSD decode kernel oracle (no model)"
	@echo "  make bench-metal-q8-mv  Bitwise checks and kernel-only Q8 decode reduction A/B"
	@echo "  make bench-metal-q4-token-pair  Bitwise checks and kernel-only Q4 Q-b token-pair A/B"
	@echo "  make test-metal-q4-qb-token-pair  Check the M1 Q-b token-pair runtime and opt-out"
	@echo "  make test-q4-epilogue-host  Check CUDA Q4 epilogue bits and admission without a GPU"
	@echo "  make test-q4-prefill-reduce-host  Check the CUDA Q4 RMS reduction tree without a GPU"
	@echo "  make test-cuda-q4-prefill-reduce  Check Q4 RMS reduction rounding/barriers on CUDA"
	@echo "  make test-rocm-q4-dot-host  Check production Q4 dot and staged pair sums without HIP"
	@echo "  make test-metal-iq2-live-index  Check IQ2 SSD live-cache index policy and fallback"
	@echo "  make test-rocm-q4-parity  Run ROCm Q4_K dense/pair/prefill oracle (or SKIP without HIP)"
	@echo "  make test-rocm-q4-prefill Run ROCm Q4 tiled-prefill parity/canary oracle"
	@echo "  make test-rocm-q4-lds-host Check Q4 prefill LDS addressing and fence schedule without HIP"
	@echo "  make test-rocm-q4-lds-aligned-host Check aligned Q8_K staging fields/ownership without HIP"
	@echo "  make test-rocm-q4-wmma-load-host Check Q4 K64 float4 loader policy/addressing without HIP"
	@echo "  make test-rocm-q4-qb-epilogue-host Check Q4 F32 epilogue mapping/reduction without HIP"
	@echo "  make test-rocm-q4-qb-epilogue Run the gfx1151 F32 RMSNorm/RoPE parity oracle"
	@echo "  make bench-rocm-q4-qb-epilogue Time the F32 epilogue default/rollback with HIP events"
	@echo "  make metal-decode-schedule-bench  Build the balanced Metal decode schedule benchmark"
	@echo "  make metal-prefill-variant-bench  Build the balanced Metal prefill variant benchmark"
	@echo "  make metal-q4-dense-pair-bench  Build the resident Q4 decode pair kernel benchmark"
	@echo "  make metal-q4-prefill-pair-bench  Build the resident Q4 prefill pair F16-RHS benchmark"
	@echo "  make metal-q4-mm-tail-cull-bench  Build the resident Q4 prefill tail-cull kernel benchmark"
	@echo "  make metal-q4-attn-out-a-direct-bench  Build the resident Q4 attention output-A direct benchmark"
	@echo "  make metal-iq2-moe-tail-cull-bench  Build the resident IQ2 pair MoE tail-cull benchmark"
	@echo "  make metal-iq2-moe-top8-pair-bench  Build the resident GLM-shape IQ2 top-8 pair-fusion benchmark"
	@echo "  make check-mxfp4-half-lut  Verify the checked-in MXFP4 half LUT matches the generator"
	@echo "  make test-mxfp4-metal  Check the MXFP4 half LUT, then run Metal MXFP4 exactness tests"
	@echo "  make test-metal-exactn-oracle  Compare Metal exact-N state with sequential decode"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.c ds4.h ds4_distributed.h ds4_tp.h $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(CC) $(QUALITY_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_official.c $(CORE_OBJS) rax.o ds4_gpu_args.o $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_tp_spec.o: tests/test_metal_tp_spec.c ds4.h ds4_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_tp_spec: tests/test_metal_tp_spec.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

test-metal-session-batch-ssd: tests/test_metal_session_batch
	env -u DS4_METAL_ENABLE_Q4_SSD_SESSION_UNION \
		-u DS4_METAL_REQUIRE_EXACT_ROWS_PERSISTENT_CACHE \
		-u DS4_TEST_SSD_UNION_POLICY_SWITCH \
		DS4_METAL_DISABLE_Q4_SSD_SESSION_UNION=1 \
		DS4_METAL_REQUIRE_Q4_SSD_SESSION_UNION=1 \
		DS4_METAL_DISABLE_EXACT_ROWS_PERSISTENT_CACHE=1 \
		DS4_TEST_SSD_STREAMING=1 DS4_TEST_SESSION_COUNT=5 \
		DS4_TEST_SSD_CACHE_EXPERTS="$(DS4_TEST_SSD_CACHE_EXPERTS)" \
		DS4_TEST_SESSION_BATCH_TIMING=1 \
		DS4_TEST_SESSION_BATCH_ARM=control \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
		./tests/test_metal_session_batch
	env -u DS4_METAL_DISABLE_Q4_SSD_SESSION_UNION \
		-u DS4_METAL_ENABLE_Q4_SSD_SESSION_UNION \
		-u DS4_METAL_DISABLE_EXACT_ROWS_PERSISTENT_CACHE \
		DS4_METAL_REQUIRE_Q4_SSD_SESSION_UNION=1 \
		DS4_METAL_REQUIRE_EXACT_ROWS_PERSISTENT_CACHE=1 \
		DS4_TEST_SSD_STREAMING=1 DS4_TEST_SESSION_COUNT=5 \
		DS4_TEST_SSD_UNION_POLICY_SWITCH=1 \
		DS4_TEST_SSD_CACHE_EXPERTS="$(DS4_TEST_SSD_CACHE_EXPERTS)" \
		DS4_TEST_SESSION_BATCH_TIMING=1 \
		DS4_TEST_SESSION_BATCH_ARM=candidate \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
		./tests/test_metal_session_batch

tests/test_metal_q4_streams.o: tests/test_metal_q4_streams.c ds4.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_metal_q4_streams: tests/test_metal_q4_streams.o ds4_metal_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-streams: tests/test_metal_q4_streams
	env -u DS4_METAL_MODEL_UNTRACKED ./tests/test_metal_q4_streams
	DS4_METAL_MODEL_UNTRACKED=1 ./tests/test_metal_q4_streams

tests/test_metal_q4_prefill_pair.o: tests/test_metal_q4_prefill_pair.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_prefill_pair: tests/test_metal_q4_prefill_pair.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-prefill-pair: tests/test_metal_q4_prefill_pair
	env -u DS4_METAL_ENABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_DENSE_PAIR \
		-u DS4_METAL_DISABLE_CONTIG_F32_F16_COPY \
		-u DS4_METAL_MODEL_UNTRACKED \
		-u DS4_METAL_UNRETAINED_COMMAND_BUFFERS \
		./tests/test_metal_q4_prefill_pair
	env -u DS4_METAL_ENABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_PREFILL_PAIR_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_DENSE_PAIR \
		-u DS4_METAL_DISABLE_CONTIG_F32_F16_COPY \
		-u DS4_METAL_MODEL_UNTRACKED \
		DS4_METAL_UNRETAINED_COMMAND_BUFFERS=1 \
		./tests/test_metal_q4_prefill_pair

tests/test_metal_indexer_q4.o: tests/test_metal_indexer_q4.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_indexer_q4: tests/test_metal_indexer_q4.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-indexer-q4: tests/test_metal_indexer_q4
	./tests/test_metal_indexer_q4

tests/test_metal_q4_attn_exactn.o: tests/test_metal_q4_attn_exactn.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_attn_exactn: tests/test_metal_q4_attn_exactn.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-attn-exactn: tests/test_metal_q4_attn_exactn
	env -u DS4_METAL_ENABLE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN \
		-u DS4_METAL_DISABLE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN \
		-u DS4_METAL_REQUIRE_Q4_SSD_PREFILL_ATTN_OUT_EXACTN \
		-u DS4_METAL_DISABLE_Q4_SSD_PREFILL_ATTN_OUT_SCALE_META \
		-u DS4_METAL_REQUIRE_Q4_SSD_PREFILL_ATTN_OUT_SCALE_META \
		-u DS4_METAL_DISABLE_Q4_MV_CLASSIC \
		./tests/test_metal_q4_attn_exactn

tests/test_metal_q4_attn_out_a_direct.o: tests/test_metal_q4_attn_out_a_direct.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_attn_out_a_direct: tests/test_metal_q4_attn_out_a_direct.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-attn-out-a-direct: tests/test_metal_q4_attn_out_a_direct
	env -u DS4_METAL_DISABLE_Q4_ATTN_OUT_A_DIRECT \
		-u DS4_METAL_REQUIRE_Q4_ATTN_OUT_A_DIRECT \
		-u DS4_METAL_DISABLE_Q4_ATTN_OUT_B_F16_RHS \
		-u DS4_METAL_REQUIRE_Q4_ATTN_OUT_B_F16_RHS \
		./tests/test_metal_q4_attn_out_a_direct

tests/test_metal_q4_qb_f16_cache.o: tests/test_metal_q4_qb_f16_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_q4_qb_f16_cache: tests/test_metal_q4_qb_f16_cache.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-q4-qb-f16-cache: tests/test_metal_q4_qb_f16_cache
	env -u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE \
		-u DS4_METAL_ENABLE_Q4_ATTN_Q_B_F16_CACHE_WITH_SSD_STREAMING \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16 \
		-u DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS \
		-u DS4_METAL_UNRETAINED_COMMAND_BUFFERS \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS \
		DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS=32 \
		DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE=1 \
		./tests/test_metal_q4_qb_f16_cache
	env -u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE \
		-u DS4_METAL_ENABLE_Q4_ATTN_Q_B_F16_CACHE_WITH_SSD_STREAMING \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16 \
		-u DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING \
		-u DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS \
		DS4_METAL_UNRETAINED_COMMAND_BUFFERS=1 \
		DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS=32 \
		DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE=1 \
		./tests/test_metal_q4_qb_f16_cache

test-metal-q4-qb-f16-cache-timing: tests/test_metal_q4_qb_f16_cache
	env -u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE \
		-u DS4_METAL_ENABLE_Q4_ATTN_Q_B_F16_CACHE_WITH_SSD_STREAMING \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_RHS \
		-u DS4_METAL_DISABLE_Q4_ATTN_Q_B_TRANSIENT_F16 \
		-u DS4_METAL_Q4_ATTN_Q_B_TRANSIENT_F16_MIN_TOKENS \
		DS4_METAL_Q4_ATTN_Q_B_F16_CACHE_MIN_TOKENS=32 \
		DS4_METAL_REQUIRE_Q4_ATTN_Q_B_F16_CACHE=1 \
		DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING=1 \
		DS4_TEST_METAL_Q4_QB_F16_CACHE_TIMING_TOKENS=4096 \
		./tests/test_metal_q4_qb_f16_cache

tests/test_metal_dspark_capture.o: tests/test_metal_dspark_capture.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_dspark_capture: tests/test_metal_dspark_capture.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-dspark-capture: tests/test_metal_dspark_capture
	./tests/test_metal_dspark_capture

tests/test_metal_argmax_top1.o: tests/test_metal_argmax_top1.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_argmax_top1: tests/test_metal_argmax_top1.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-argmax-top1: tests/test_metal_argmax_top1
	env -u DS4_TEST_METAL_ARGMAX_TOP1_TIMING ./tests/test_metal_argmax_top1

bench-metal-argmax-top1: tests/test_metal_argmax_top1
	DS4_TEST_METAL_ARGMAX_TOP1_TIMING=1 ./tests/test_metal_argmax_top1

tests/test_metal_iq2_midonly.o: tests/test_metal_iq2_midonly.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_iq2_midonly: tests/test_metal_iq2_midonly.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-iq2-midonly: tests/test_metal_iq2_midonly
	./tests/test_metal_iq2_midonly

tests/test_metal_iq2_ssd_grouped_mm.o: tests/test_metal_iq2_ssd_grouped_mm.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_iq2_ssd_grouped_mm: tests/test_metal_iq2_ssd_grouped_mm.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-iq2-ssd-grouped-mm: tests/test_metal_iq2_ssd_grouped_mm
	./tests/test_metal_iq2_ssd_grouped_mm

tests/test_metal_iq2_live_index.o: tests/test_metal_iq2_live_index.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_metal_iq2_live_index: tests/test_metal_iq2_live_index.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-iq2-live-index: tests/test_metal_iq2_live_index
	./tests/test_metal_iq2_live_index

speed-bench/metal_decode_schedule_bench.o: speed-bench/metal_decode_schedule_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_decode_schedule_bench: speed-bench/metal_decode_schedule_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-decode-schedule-bench: speed-bench/metal_decode_schedule_bench

speed-bench/metal_prefill_variant_bench.o: speed-bench/metal_prefill_variant_bench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_prefill_variant_bench: speed-bench/metal_prefill_variant_bench.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-prefill-variant-bench: speed-bench/metal_prefill_variant_bench

ds4_metal_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_image.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_metal_exactn_oracle.o: tests/test_metal_exactn_oracle.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_metal_exactn_oracle: tests/test_metal_exactn_oracle.o ds4_metal_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-exactn-oracle: tests/test_metal_exactn_oracle
	DS4_TEST_REQUIRE_MODEL=1 \
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
	./tests/test_metal_exactn_oracle

speed-bench/metal_q4_dense_pair_bench: speed-bench/metal_q4_dense_pair_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

.PHONY: test-metal-ssd-decode-kernels bench-metal-q8-mv bench-metal-q4-token-pair bench-metal-q4-single-vec
.PHONY: test-metal-q4-qb-token-pair
tests/test_metal_q4_qb_token_pair: tests/test_metal_q4_qb_token_pair.m ds4_gpu.h ds4_image.o ds4_metal.o
	$(CC) $(OBJCFLAGS) -I. -o $@ $< ds4_image.o ds4_metal.o $(METAL_LDLIBS)

test-metal-q4-qb-token-pair: tests/test_metal_q4_qb_token_pair
	./tests/test_metal_q4_qb_token_pair

tests/test_metal_ssd_decode_kernels: tests/test_metal_ssd_decode_kernels.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

test-metal-ssd-decode-kernels: tests/test_metal_ssd_decode_kernels
	./tests/test_metal_ssd_decode_kernels

bench-metal-q8-mv: tests/test_metal_ssd_decode_kernels
	./tests/test_metal_ssd_decode_kernels --bench-q8-mv

bench-metal-q4-token-pair: tests/test_metal_ssd_decode_kernels
	./tests/test_metal_ssd_decode_kernels --bench-q4-token-pair

bench-metal-q4-single-vec: tests/test_metal_ssd_decode_kernels
	./tests/test_metal_ssd_decode_kernels --bench-q4-single-vec

metal-q4-dense-pair-bench: speed-bench/metal_q4_dense_pair_bench

speed-bench/metal_q4_prefill_pair_bench: speed-bench/metal_q4_prefill_pair_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-prefill-pair-bench: speed-bench/metal_q4_prefill_pair_bench

speed-bench/metal_q4_mm_tail_cull_bench: speed-bench/metal_q4_mm_tail_cull_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-mm-tail-cull-bench: speed-bench/metal_q4_mm_tail_cull_bench

speed-bench/metal_q4_attn_out_a_direct_bench: speed-bench/metal_q4_attn_out_a_direct_bench.m $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -o $@ $< $(METAL_LDLIBS)

metal-q4-attn-out-a-direct-bench: speed-bench/metal_q4_attn_out_a_direct_bench

speed-bench/metal_iq2_moe_tail_cull_bench.o: speed-bench/metal_iq2_moe_tail_cull_bench.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_iq2_moe_tail_cull_bench: speed-bench/metal_iq2_moe_tail_cull_bench.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-iq2-moe-tail-cull-bench: speed-bench/metal_iq2_moe_tail_cull_bench

speed-bench/metal_iq2_moe_top8_pair_bench.o: speed-bench/metal_iq2_moe_top8_pair_bench.c speed-bench/metal_iq2_moe_tail_cull_bench.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

speed-bench/metal_iq2_moe_top8_pair_bench: speed-bench/metal_iq2_moe_top8_pair_bench.o ds4_image.o ds4_metal.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

metal-iq2-moe-top8-pair-bench: speed-bench/metal_iq2_moe_top8_pair_bench

tests/test_mxfp4_metal.o: tests/test_mxfp4_metal.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_mxfp4_metal: tests/test_mxfp4_metal.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

check-mxfp4-half-lut:
	python3 metal/generate_mxfp4_half_lut.py --check

test-mxfp4-metal: check-mxfp4-half-lut tests/test_mxfp4_metal
	./tests/test_mxfp4_metal

tests/test_metal_moe_prefill.o: tests/test_metal_moe_prefill.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_moe_prefill: tests/test_metal_moe_prefill.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-moe-prefill: tests/test_metal_moe_prefill
	./tests/test_metal_moe_prefill

tests/test_metal_dense_mpp.o: tests/test_metal_dense_mpp.c ds4_gpu.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ $<

tests/test_metal_dense_mpp: tests/test_metal_dense_mpp.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-dense-mpp: tests/test_metal_dense_mpp
	./tests/test_metal_dense_mpp

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make test-mmq-parity-cuda CUDA_ARCH=sm_N  Run quantized CUDA kernel parity tests"
	@echo "  make test-mmq-q4-grouped-q81-cuda CUDA_ARCH=sm_N  Run focused grouped Q8_1 byte-parity tests"
	@echo "  make test-mmq-q4-16warp-cuda CUDA_ARCH=sm_N  Run focused Q4 16-warp bitwise/canary oracle"
	@echo "  make test-cuda-q4-epilogue CUDA_ARCH=sm_N  Check Q4 decode epilogue parity, rollback and graph nodes"
	@echo "  make test-rocm-q4-parity        Run ROCm Q4_K dense/pair/prefill oracle"
	@echo "  make test-strix-rocm-q4-prefill Require gfx1151 and run tiled-prefill oracle"
	@echo "  make test-strix-rocm-q4-parity  Require a visible gfx1151 device and run the Q4 tests"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make rocm-dspark-acceptance   Build ROCm and run the DSpark acceptance fixture"
	@echo "  make rocm-dspark-verify-depth Build ROCm and run the DSpark verifier invariant"
	@echo "  make test-mxfp4-rocm     Build and run the synthetic ROCm MXFP4 MoE test"
	@echo "  make rocm-iq2-moe-prefill-bench  Build the resident ROCm IQ2/Q2 WMMA A/B harness"
	@echo "  make rocm-q4-prefill-bench  Build the resident ROCm Q4 projection/WMMA A/B harness"
	@echo "  make cuda-iq2-moe-prefill-bench CUDA_ARCH=sm_N  Build the resident CUDA IQ2/Q2 profiling harness"
	@echo "  make cuda-q4-prefill-bench CUDA_ARCH=sm_N  Build the resident CUDA Q4 dense/pair/q_b/output-A/output-B harness"
	@echo "  make test-rocm           Core regression suite on ROCm-only hosts"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make test-ssd            Run the model suite with cold SSD streaming"
	@echo "  make environment-docs  Generate and verify the environment variable inventory"
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
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test \
		CORE_OBJS="$(ROCM_CORE_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

# Core regression suite for ROCm-only hosts: the CUDA-specific binaries
# (tests/test_sampling, the CUDA session/mixed-batch oracles) are not part
# of this target; run them through `make test` / `make cuda-regression` on
# CUDA hosts.  Everything else mirrors `make test`.
test-rocm:
	$(MAKE) -B ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test \
		test-session-state \
		tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args tests/test_prompt_prefix \
		ds4 ds4-server ds4-bench ds4-agent \
		CORE_OBJS="$(ROCM_CORE_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix

rocm-dspark-acceptance:
	@if [ ! -f "$(DS4_DSPARK_MODEL)" ]; then \
		echo "rocm-dspark-acceptance: missing model $(DS4_DSPARK_MODEL)" >&2; \
		exit 1; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "rocm-dspark-acceptance: missing DSpark support $(DS4_DSPARK_SUPPORT)" >&2; \
		exit 1; \
	fi
	$(MAKE) -B ds4 \
		CORE_OBJS="$(ROCM_CORE_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	DS4_DSPARK_FIXTURE_BACKEND=rocm \
	sh tests/dspark_acceptance_fixture.sh

rocm-dspark-verify-depth:
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "rocm-dspark-verify-depth: missing model $(DS4_TEST_MODEL)" >&2; \
		exit 1; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "rocm-dspark-verify-depth: missing DSpark support $(DS4_DSPARK_SUPPORT)" >&2; \
		exit 1; \
	fi
	$(MAKE) -B ds4_test \
		CORE_OBJS="$(ROCM_CORE_OBJS)" \
		CFLAGS="$(CFLAGS) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" \
	DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" \
	./ds4_test --dspark-verify-depth

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) $(ROCM_HOST_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch.o: tests/test_cuda_q8_scratch.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_q8_scratch: tests/test_cuda_q8_scratch.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-scratch
test-cuda-q8-scratch: tests/test_cuda_q8_scratch
	./tests/test_cuda_q8_scratch

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda

cuda/mmq/test/test_mmq_parity: cuda/mmq/test/test_mmq_parity.cu cuda/mmq/ds4_mmq.h $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $< $(MMQ_OBJS) $(CUDA_LDLIBS)

test-mmq-parity-cuda: cuda/mmq/test/test_mmq_parity
	./cuda/mmq/test/test_mmq_parity

test-mmq-q4-grouped-q81-cuda: cuda/mmq/test/test_mmq_parity
	./cuda/mmq/test/test_mmq_parity --q4-grouped-q81

test-mmq-q4-16warp-cuda: cuda/mmq/test/test_mmq_parity
	./cuda/mmq/test/test_mmq_parity --q4-16warp

tests/test_cuda_q4_epilogue.o: tests/test_cuda_q4_epilogue.cpp cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu $(MMQ_INCLUDES) -c -o $@ $<

tests/test_cuda_q4_epilogue: tests/test_cuda_q4_epilogue.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-q4-epilogue: tests/test_cuda_q4_epilogue
	./tests/test_cuda_q4_epilogue

speed-bench/gpu_iq2_moe_prefill_bench_rocm.o: speed-bench/gpu_iq2_moe_prefill_bench.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -std=c11 -DDS4_ROCM_BUILD -DDS4_BENCH_ROCM -I. -c -o $@ $<

speed-bench/gpu_iq2_moe_prefill_bench_rocm: speed-bench/gpu_iq2_moe_prefill_bench_rocm.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

rocm-iq2-moe-prefill-bench:
	$(MAKE) --no-print-directory -B speed-bench/gpu_iq2_moe_prefill_bench_rocm ROCM_ARCH="$(ROCM_ARCH)"

speed-bench/rocm_q4_prefill_bench.o: speed-bench/rocm_q4_prefill_bench.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

speed-bench/rocm_q4_prefill_bench: speed-bench/rocm_q4_prefill_bench.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

rocm-q4-prefill-bench:
	$(MAKE) --no-print-directory -B speed-bench/rocm_q4_prefill_bench ROCM_ARCH="$(ROCM_ARCH)"

speed-bench/gpu_iq2_moe_prefill_bench_cuda.o: speed-bench/gpu_iq2_moe_prefill_bench.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -std=c11 -DDS4_BENCH_CUDA -I. -c -o $@ $<

speed-bench/gpu_iq2_moe_prefill_bench_cuda: speed-bench/gpu_iq2_moe_prefill_bench_cuda.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

cuda-iq2-moe-prefill-bench:
	$(MAKE) --no-print-directory -B speed-bench/gpu_iq2_moe_prefill_bench_cuda CUDA_ARCH="$(CUDA_ARCH)"

speed-bench/cuda_q4_prefill_bench.o: speed-bench/cuda_q4_prefill_bench.cu ds4_gpu.h cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_q4_16warp.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 -DDS4_BENCH_CUDA -I. -c -o $@ $<

speed-bench/cuda_q4_prefill_bench: speed-bench/cuda_q4_prefill_bench.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

cuda-q4-prefill-bench:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda-q4-prefill-bench CUDA_ARCH=sm_121"; \
		exit 2; \
	fi
	$(MAKE) --no-print-directory -B speed-bench/cuda_q4_prefill_bench CUDA_ARCH="$(CUDA_ARCH)"
endif

environment-docs:
	python3 scripts/generate_environment_variables.py
	python3 scripts/generate_environment_variables.py --check

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_image.h ds4_linux_memory.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_prompt_prefix.o: ds4_prompt_prefix.c ds4_prompt_prefix.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_prompt_prefix.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_eval_cases.o: ds4_eval_cases.c ds4_eval_cases.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval_cases.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_image.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h ds4_image.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

tests/test_glm53_kda.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_kda.c

tests/test_glm53_vision_engine.o: tests/test_glm53_vision_engine.c ds4.h ds4_image.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_glm53_vision_engine.c

tests/test_glm53_vision_engine: tests/test_glm53_vision_engine.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_glm53_vision_prompt.o: tests/test_glm53_vision_prompt.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_vision_prompt.c

tests/test_glm53_vision_prompt: tests/test_glm53_vision_prompt.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifeq ($(UNAME_S),Darwin)
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm53-kda
test-glm53-kda: $(GLM53_KDA_TEST)
	./$(GLM53_KDA_TEST)

tests/test_glm53_kda_rocm.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

$(GLM53_KDA_ROCM_TEST): tests/test_glm53_kda_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-glm53-kda-rocm: $(GLM53_KDA_ROCM_TEST)
	./$(GLM53_KDA_ROCM_TEST)

tests/test_glm_attention_rocm.o: tests/test_glm_attention.c ds4.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_glm_attention_rocm: tests/test_glm_attention_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

.PHONY: test-glm-attention-rocm
test-glm-attention-rocm: tests/test_glm_attention_rocm
	./tests/test_glm_attention_rocm

tests/test_linux_memory: tests/test_linux_memory.c ds4_linux_memory.h
	$(CC) $(CFLAGS) -I. -o $@ $<

tests/test_rocm_memory: tests/test_rocm_memory.cu ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -o $@ $<

.PHONY: test-linux-memory test-rocm-memory
test-linux-memory: tests/test_linux_memory
	./tests/test_linux_memory

test-rocm-memory: tests/test_rocm_memory
	./tests/test_rocm_memory

tests/test_glm_attention.o: tests/test_glm_attention.c ds4.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

ifeq ($(UNAME_S),Darwin)
tests/test_glm_attention: tests/test_glm_attention.o ds4_metal.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
tests/test_glm_attention: tests/test_glm_attention.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)
endif

.PHONY: test-glm-attention
test-glm-attention: tests/test_glm_attention
	./tests/test_glm_attention

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h cuda/ds4_q8_quantize.cuh cuda/ds4_q8_prefill_layout.h cuda/ds4_q4_prefill_reduce.h $(Q4_PREFILL_DEQUANT_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/ds4_mmq_q4_16warp.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/mmvq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_q4_16warp.o: cuda/mmq/ds4_mmq_q4_16warp.cu cuda/mmq/ds4_mmq_q4_16warp.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh cuda/mmq/ds4_q4_mmvq_epilogue.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

ds4_rocm.o: ds4_rocm.cu ds4_rocm.h ds4_rocm_memory.h ds4_linux_memory.h ds4_gpu.h ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_image.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS) $(Q4_PREFILL_DEQUANT_HEADERS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

cuda/mmq/ds4_ggml_stubs.rocm.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh cuda/mmq/vendors/hip.h ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/ds4_mmq.rocm.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh cuda/mmq/vendors/hip.h cuda/mmq/ds4_q4_mmvq_epilogue.h cuda/mmq/mmvq.cuh
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/quantize.rocm.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmid.rocm.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/mmvq.rocm.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh cuda/mmq/vendors/hip.h cuda/mmq/ds4_q4_mmvq_epilogue.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

cuda/mmq/d2r_stubs.rocm.o: cuda/mmq/test/d2r_stubs.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/vendors/hip.h
	$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<

tests/test_mxfp4_rocm.o: tests/test_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_mxfp4_rocm: tests/test_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/bench_mxfp4_rocm.o: tests/bench_mxfp4_rocm.c ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(ROCM_HOST_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/bench_mxfp4_rocm: tests/bench_mxfp4_rocm.o ds4_rocm.o ds4_image.o $(ROCM_MMQ_OBJS)
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-mxfp4-rocm: tests/test_mxfp4_rocm
	./tests/test_mxfp4_rocm

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_args.h ds4_rocm_memory.h ds4_linux_memory.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

.PHONY: test-q4-prefill-reduce-host test-cuda-q4-prefill-reduce bench-cuda-q4-prefill-reduce
tests/test_q4_prefill_reduce_host: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -o $@ $<

tests/test_q4_prefill_reduce_host_fast: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -o $@ $<

test-q4-prefill-reduce-host: tests/test_q4_prefill_reduce_host tests/test_q4_prefill_reduce_host_fast
	./tests/test_q4_prefill_reduce_host
	./tests/test_q4_prefill_reduce_host_fast

tests/test_cuda_q4_prefill_reduce: tests/test_cuda_q4_prefill_reduce.cpp cuda/ds4_q4_prefill_reduce.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -x cu -o $@ $<

CUDA_Q4_PREFILL_REDUCE_TEST_ARGS ?=
test-cuda-q4-prefill-reduce:
	@reduce_nvcc="$(strip $(NVCC))"; \
	if [ -z "$$reduce_nvcc" ]; then reduce_nvcc="$$(command -v nvcc 2>/dev/null || true)"; fi; \
	reduce_probe="$${reduce_nvcc%% *}"; \
	if [ -z "$$reduce_probe" ] || ! command -v "$$reduce_probe" >/dev/null 2>&1; then \
		echo "CUDA Q4 prefill reduction: FAIL (nvcc and CUDA device required)"; exit 1; \
	fi; \
	$(MAKE) --no-print-directory tests/test_cuda_q4_prefill_reduce NVCC="$$reduce_nvcc" || exit $$?; \
	./tests/test_cuda_q4_prefill_reduce $(CUDA_Q4_PREFILL_REDUCE_TEST_ARGS)

bench-cuda-q4-prefill-reduce:
	$(MAKE) test-cuda-q4-prefill-reduce CUDA_Q4_PREFILL_REDUCE_TEST_ARGS=--bench

.PHONY: test-rocm-q4-dot-host
ROCM_Q4_DOT_HEADERS = rocm/ds4_rocm_q4_dot.cuh rocm/ds4_rocm_q4_lds.cuh
tests/test_rocm_q4_dot_host: tests/test_rocm_q4_dot_host.cpp $(ROCM_Q4_DOT_HEADERS)
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -I. -o $@ $<

tests/test_rocm_q4_dot_host_fast: tests/test_rocm_q4_dot_host.cpp $(ROCM_Q4_DOT_HEADERS)
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-dot-host: tests/test_rocm_q4_dot_host tests/test_rocm_q4_dot_host_fast
	./tests/test_rocm_q4_dot_host
	./tests/test_rocm_q4_dot_host_fast

.PHONY: test-rocm-q4-lds-host
tests/test_rocm_q4_lds_host: tests/test_rocm_q4_lds_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -I. -o $@ $<

test-rocm-q4-lds-host: tests/test_rocm_q4_lds_host
	./tests/test_rocm_q4_lds_host

.PHONY: test-rocm-q4-lds-aligned-host
tests/test_rocm_q4_lds_aligned_host: tests/test_rocm_q4_lds_aligned_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -I. -o $@ $<

tests/test_rocm_q4_lds_aligned_host_fast: tests/test_rocm_q4_lds_aligned_host.cpp rocm/ds4_rocm_q4_lds.cuh
	$(CXX) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-lds-aligned-host: tests/test_rocm_q4_lds_aligned_host tests/test_rocm_q4_lds_aligned_host_fast
	./tests/test_rocm_q4_lds_aligned_host
	./tests/test_rocm_q4_lds_aligned_host_fast

.PHONY: test-rocm-q4-lds-aligned bench-rocm-q4-lds-aligned
tests/test_rocm_q4_lds_aligned.o: tests/test_rocm_q4_lds_aligned.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_lds_aligned: tests/test_rocm_q4_lds_aligned.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ROCM_Q4_LDS_ALIGNED_TEST_ARGS ?=
test-rocm-q4-lds-aligned:
	@lds_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$lds_hipcc" ]; then lds_hipcc="$$(command -v hipcc 2>/dev/null || true)"; fi; \
	lds_probe="$${lds_hipcc%% *}"; \
	if [ -z "$$lds_probe" ] || ! command -v "$$lds_probe" >/dev/null 2>&1; then \
		echo "ROCm Q4 aligned LDS: FAIL (hipcc and gfx1151 device required)"; exit 1; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_lds_aligned HIPCC="$$lds_hipcc" || exit $$?; \
	./tests/test_rocm_q4_lds_aligned $(ROCM_Q4_LDS_ALIGNED_TEST_ARGS)

bench-rocm-q4-lds-aligned:
	$(MAKE) test-rocm-q4-lds-aligned ROCM_Q4_LDS_ALIGNED_TEST_ARGS="--bench $(ROCM_Q4_LDS_ALIGNED_TEST_ARGS)"

.PHONY: test-rocm-q4-wmma-load-host
tests/test_rocm_q4_wmma_load_host: tests/test_rocm_q4_wmma_load_host.cpp rocm/ds4_rocm_q4_wmma_load.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -I. -o $@ $<

test-rocm-q4-wmma-load-host: tests/test_rocm_q4_wmma_load_host
	./tests/test_rocm_q4_wmma_load_host

.PHONY: test-rocm-q4-qb-epilogue-host test-rocm-q4-qb-epilogue bench-rocm-q4-qb-epilogue
tests/test_rocm_q4_qb_epilogue_host: tests/test_rocm_q4_qb_epilogue_host.cpp rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -ffp-contract=off -I. -o $@ $<

# ROCm uses Clang FP pragmas; exercise them under fast-math even without HIP.
ROCM_Q4_EPILOGUE_HOST_CLANG ?= clang++
tests/test_rocm_q4_qb_epilogue_host_fast: tests/test_rocm_q4_qb_epilogue_host.cpp rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(ROCM_Q4_EPILOGUE_HOST_CLANG) -O3 -Wall -Wextra -std=c++17 -ffast-math -fno-finite-math-only -I. -o $@ $<

test-rocm-q4-qb-epilogue-host: tests/test_rocm_q4_qb_epilogue_host tests/test_rocm_q4_qb_epilogue_host_fast
	./tests/test_rocm_q4_qb_epilogue_host
	./tests/test_rocm_q4_qb_epilogue_host_fast

tests/test_rocm_q4_qb_epilogue.o: tests/test_rocm_q4_qb_epilogue.cpp ds4_gpu.h rocm/ds4_rocm_q4_qb_epilogue_layout.cuh
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_qb_epilogue: tests/test_rocm_q4_qb_epilogue.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ROCM_Q4_EPILOGUE_TEST_ARGS ?=
test-rocm-q4-qb-epilogue:
	@epilogue_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$epilogue_hipcc" ]; then epilogue_hipcc="$$(command -v hipcc 2>/dev/null || true)"; fi; \
	epilogue_probe="$${epilogue_hipcc%% *}"; \
	if [ -z "$$epilogue_probe" ] || ! command -v "$$epilogue_probe" >/dev/null 2>&1; then \
		if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
			echo "ROCm Q4 F32 epilogue: FAIL (hipcc not found, device required)"; exit 1; \
		fi; \
		echo "ROCm Q4 F32 epilogue: SKIP (hipcc not found)"; exit 0; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_qb_epilogue HIPCC="$$epilogue_hipcc" || exit $$?; \
	DS4_TEST_REQUIRE_ROCM_DEVICE="$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" \
		./tests/test_rocm_q4_qb_epilogue $(ROCM_Q4_EPILOGUE_TEST_ARGS); \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then echo "ROCm Q4 F32 epilogue: SKIP (no HIP device)"; exit 0; fi; \
	exit $$rc

bench-rocm-q4-qb-epilogue:
	$(MAKE) test-rocm-q4-qb-epilogue ROCM_Q4_EPILOGUE_TEST_ARGS="--bench $(ROCM_Q4_EPILOGUE_TEST_ARGS)"

.PHONY: test-rocm-q4-decode-host
tests/test_rocm_q4_decode_host: tests/test_rocm_q4_decode_host.cpp rocm/ds4_rocm_q4_decode.cuh
	$(CXX) -O2 -Wall -Wextra -std=c++17 -fno-fast-math -I. -o $@ $<

test-rocm-q4-decode-host: tests/test_rocm_q4_decode_host
	./tests/test_rocm_q4_decode_host

.PHONY: test-rocm-q4-decode-lane4 bench-rocm-q4-decode-lane4
tests/test_rocm_q4_decode_lane4.o: tests/test_rocm_q4_decode_lane4.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_decode_lane4: tests/test_rocm_q4_decode_lane4.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ROCM_Q4_DECODE_TEST_ARGS ?=
test-rocm-q4-decode-lane4:
	@lane4_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$lane4_hipcc" ]; then lane4_hipcc="$$(command -v hipcc 2>/dev/null || true)"; fi; \
	lane4_probe="$${lane4_hipcc%% *}"; \
	if [ -z "$$lane4_probe" ] || ! command -v "$$lane4_probe" >/dev/null 2>&1; then \
		if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
			echo "ROCm Q4 lane4: FAIL (hipcc not found, device required)"; exit 1; \
		fi; \
		echo "ROCm Q4 lane4: SKIP (hipcc not found)"; exit 0; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_decode_lane4 HIPCC="$$lane4_hipcc" || exit $$?; \
	DS4_TEST_REQUIRE_ROCM_DEVICE="$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" \
		./tests/test_rocm_q4_decode_lane4 $(ROCM_Q4_DECODE_TEST_ARGS); \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then echo "ROCm Q4 lane4: SKIP (no HIP device)"; exit 0; fi; \
	exit $$rc

bench-rocm-q4-decode-lane4:
	$(MAKE) test-rocm-q4-decode-lane4 ROCM_Q4_DECODE_TEST_ARGS="--bench $(ROCM_Q4_DECODE_TEST_ARGS)"

tests/test_rocm_q4_dense_pair.o: tests/test_rocm_q4_dense_pair.cpp ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -std=c++17 -fno-fast-math -I. -c -o $@ $<

tests/test_rocm_q4_dense_pair: tests/test_rocm_q4_dense_pair.o ds4_image.o ds4_rocm.o $(ROCM_MMQ_OBJS) ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

# Keep the public test target usable on development hosts without ROCm.  The
# binary itself exits 77 when HIP is installed but no device is visible; an
# explicitly required Strix run converts that condition into a hard failure.
ROCM_Q4_TEST_ARGS ?= --all
test-rocm-q4-parity:
	@rocm_test_hipcc="$(strip $(HIPCC))"; \
	if [ -z "$$rocm_test_hipcc" ]; then \
		rocm_test_hipcc="$$(command -v hipcc 2>/dev/null || true)"; \
	fi; \
	rocm_test_probe="$${rocm_test_hipcc%% *}"; \
	if [ -z "$$rocm_test_probe" ] || ! command -v "$$rocm_test_probe" >/dev/null 2>&1; then \
		if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
			echo "ROCm Q4 dense/pair/prefill oracle: FAIL (hipcc not found, device required)"; \
			exit 1; \
		fi; \
		echo "ROCm Q4 dense/pair/prefill oracle: SKIP (hipcc not found)"; exit 0; \
	fi; \
	$(MAKE) --no-print-directory tests/test_rocm_q4_dense_pair HIPCC="$$rocm_test_hipcc" || exit $$?; \
	if [ -n "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" ] && [ "$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" != "0" ]; then \
		DS4_TEST_REQUIRE_ROCM_DEVICE="$(strip $(DS4_TEST_REQUIRE_ROCM_DEVICE))" \
			./tests/test_rocm_q4_dense_pair $(ROCM_Q4_TEST_ARGS); \
	else \
		env -u DS4_TEST_REQUIRE_ROCM_DEVICE \
			./tests/test_rocm_q4_dense_pair $(ROCM_Q4_TEST_ARGS); \
	fi; \
	rc=$$?; \
	if [ $$rc -eq 77 ]; then \
		echo "ROCm Q4 dense/pair/prefill oracle: SKIP (no visible HIP device)"; \
		exit 0; \
	fi; \
	exit $$rc

test-rocm-q4-dense:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--dense

test-rocm-q4-pair:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--pair

test-rocm-q4-prefill:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--prefill

.PHONY: test-rocm-q4-prefill-load4
test-rocm-q4-prefill-load4:
	$(MAKE) --no-print-directory test-rocm-q4-parity ROCM_Q4_TEST_ARGS=--prefill-wmma-load4

test-strix-rocm-q4-parity:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1

test-strix-rocm-q4-prefill:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1 ROCM_Q4_TEST_ARGS=--prefill
	$(MAKE) --no-print-directory test-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1
	$(MAKE) --no-print-directory test-rocm-q4-lds-aligned ROCM_ARCH=gfx1151

test-strix-rocm-q4-prefill-long:
	$(MAKE) --no-print-directory -B test-rocm-q4-parity ROCM_ARCH=gfx1151 \
		DS4_TEST_REQUIRE_ROCM_DEVICE=1 ROCM_Q4_TEST_ARGS=--prefill-long

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_session_state: tests/test_session_state.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state_gpu.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_session_state_gpu: tests/test_session_state_gpu.o $(filter-out ds4.o,$(CORE_OBJS))
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
endif

tests/test_tp_commands.o: tests/test_tp_commands.c ds4_tp.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_commands: tests/test_tp_commands.o $(filter-out ds4_tp.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-session-state
test-session-state: tests/test_session_state tests/test_tp_commands
	./tests/test_session_state
	./tests/test_tp_commands

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_image.o ds4_cuda.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_image.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

tests/test_prompt_prefix.o: tests/test_prompt_prefix.c ds4_prompt_prefix.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_prompt_prefix: tests/test_prompt_prefix.o ds4_prompt_prefix.o
	$(CC) $(CFLAGS) -o $@ $^

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test test-session-state test-linux-memory \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_quantizer_indexer_q4 gguf-tools/deepseek4-quantize \
	tests/test_deepseek4_vision_image tests/test_prompt_prefix $(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --validate-cases
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	# Avoid adding the Q4 resident sidecar's 2.69 GiB to this broad suite.
	# This does not enable SSD streaming; use `make test-ssd` for oversized models.
	DS4_METAL_DISABLE_Q4_ATTN_Q_B_F16_CACHE=1 ./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix
	./tests/test_sampling
	./tests/test_quantizer_indexer_q4 ./gguf-tools/deepseek4-quantize
	./tests/test_deepseek4_vision_image

test-ssd:
	DS4_TEST_SSD_STREAMING=1 \
	DS4_TEST_SSD_STREAMING_COLD=1 \
	$(MAKE) test

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

.PHONY: test-quality-api
# Only the scorer's JSON parser is needed; discard the unused engine entry point.
test-quality-api: tests/test_quality_api.c gguf-tools/quality-testing/score_official.c
	$(CC) $(QUALITY_CFLAGS) -I. -ffunction-sections -fdata-sections -o tests/test_quality_api tests/test_quality_api.c -Wl,$(if $(filter Darwin,$(UNAME_S)),-dead_strip,--gc-sections) -lm
	./tests/test_quality_api

clean:
	rm -f tests/test_metal_raw_kv_ring tests/test_cuda_raw_kv_ring tests/test_rocm_raw_kv_ring tests/test_rocm_iq2_prefill
	rm -f tests/test_q4_prefill_reduce_host tests/test_q4_prefill_reduce_host_fast tests/test_cuda_q4_prefill_reduce
	rm -f tests/test_rocm_q4_dot_host tests/test_rocm_q4_dot_host_fast
	rm -f tests/test_rocm_q4_qb_epilogue_host tests/test_rocm_q4_qb_epilogue_host_fast tests/test_rocm_q4_qb_epilogue
	rm -f tests/test_rocm_q4_decode_host
	rm -f tests/test_rocm_q4_decode_lane4
	rm -f tests/test_rocm_q4_lds_host
	rm -f tests/test_rocm_q4_lds_aligned_host tests/test_rocm_q4_lds_aligned_host_fast tests/test_rocm_q4_lds_aligned
	rm -f tests/test_rocm_q4_wmma_load_host
	rm -f tests/test_metal_q4_qb_token_pair
	rm -f tests/test_q4_epilogue_host tests/test_cuda_q4_epilogue
	rm -f tests/test_q8_quantize_host tests/test_cuda_q8_quantize
	rm -f tests/test_q4_prefill_dequant_host tests/test_cuda_q4_prefill_dequant tests/test_rocm_q4_prefill_dequant
	rm -f tests/test_metal_ssd_decode_kernels
	rm -f speed-bench/metal_iq2_moe_top8_pair_bench
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o speed-bench/metal_decode_schedule_bench speed-bench/metal_prefill_variant_bench speed-bench/metal_q4_dense_pair_bench speed-bench/metal_q4_prefill_pair_bench speed-bench/metal_q4_mm_tail_cull_bench speed-bench/metal_q4_attn_out_a_direct_bench speed-bench/metal_iq2_moe_tail_cull_bench speed-bench/gpu_iq2_moe_prefill_bench_rocm speed-bench/gpu_iq2_moe_prefill_bench_cuda speed-bench/rocm_q4_prefill_bench speed-bench/cuda_q4_prefill_bench speed-bench/*.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_quantizer_indexer_q4 tests/test_mxfp4_metal tests/test_mxfp4_rocm tests/bench_mxfp4_rocm tests/test_mxfp4_cuda tests/test_rocm_q4_dense_pair tests/test_metal_session_batch tests/test_metal_q4_streams tests/test_metal_q4_prefill_pair tests/test_metal_indexer_q4 tests/test_metal_q4_attn_exactn tests/test_metal_q4_attn_out_a_direct tests/test_metal_q4_qb_f16_cache tests/test_metal_exactn_oracle tests/test_metal_dspark_capture tests/test_metal_argmax_top1 tests/test_metal_iq2_midonly tests/test_metal_iq2_ssd_grouped_mm tests/test_metal_iq2_live_index tests/test_glm53_kda tests/test_glm53_kda_rocm tests/test_glm53_vision_engine tests/test_glm53_vision_prompt tests/test_deepseek4_vision_image tests/test_prompt_prefix tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o cuda/mmq/*.o cuda/mmq/test/*.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
	rm -f tests/test_cuda_q8_scratch
	rm -f tests/test_quality_api
	rm -f tests/test_linux_memory tests/test_rocm_memory
	rm -f tests/test_glm_attention tests/test_glm_attention_rocm
	rm -f tests/test_session_state tests/test_session_state_gpu tests/test_tp_commands
	rm -f tests/test_metal_tp_spec
	rm -f tests/test_metal_moe_prefill tests/test_metal_dense_mpp
