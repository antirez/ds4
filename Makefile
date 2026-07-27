CC ?= cc
NATIVE_CPU_FLAG ?= -march=native

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only

# C++ port (pulsar): host TUs migrate .c -> .cpp one at a time. The FP and
# optimization flags MUST stay identical to CFLAGS so a ported TU generates
# the same math as its C predecessor (the per-TU bit-exact gate depends on
# it). -fno-exceptions/-fno-rtti is the project style: hot paths never throw,
# fatal errors keep the pulsar_die() contract.
CXX ?= g++
CXXFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c++23
CXXFLAGS += -D_GNU_SOURCE -fno-finite-math-only -fno-exceptions -fno-rtti
# Partial designated initializers (fields filled right after, or {0} zero-init)
# are a pervasive idiom here; C's -Wextra accepts them silently, C++'s warns.
# Keep the warning surface identical to the C build.
CXXFLAGS += -Wno-missing-field-initializers
CXXFLAGS += -DPULSAR_VERSION_STR='"$(PULSAR_VERSION_STR)"'

# Version string reported by /version, /health and the startup banner. Derived
# from git so it never goes stale (e.g. "v0.2.3-8-gec51fb2", "-dirty" if the
# tree has uncommitted changes); falls back to "unknown" outside a git checkout.
PULSAR_VERSION_STR := $(shell git describe --tags --dirty --always 2>/dev/null || echo unknown)
CFLAGS += -DPULSAR_VERSION_STR='"$(PULSAR_VERSION_STR)"'

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math --default-stream per-thread $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread

# HC residual-carrier storage precision (task #62). HC_F32=1 restores f32
# carriers (the fallback, and the control build for the byte-exact gate).
# CRITICAL: this MUST reach BOTH the host (CFLAGS) and device (NVCCFLAGS) TUs.
# Defining it on only one half compiles and links cleanly, the in-TU
# static_assert does NOT fire, and every residual read/write silently strides
# the wrong element size -> total activation corruption. Never pass
# -DPULSAR_HC_F32 by hand; use HC_F32=1.
HC_F32 ?= 0
ifeq ($(HC_F32),1)
CFLAGS += -DPULSAR_HC_F32
NVCCFLAGS += -DPULSAR_HC_F32
endif

CUTLASS_DIR ?= $(CURDIR)/cutlass
CUTLASS_INC ?= -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcublasLt

PULSAR_INC = -Isrc -Isrc/lib -Isrc/vendor

ENGINE_SRCS = $(wildcard src/engine/*.c) $(wildcard src/engine/*.cpp)
ENGINE_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(ENGINE_SRCS)))
AGENT_SRCS = $(wildcard src/agent/*.c) $(wildcard src/agent/*.cpp)
AGENT_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(AGENT_SRCS)))
SERVER_SRCS = $(wildcard src/server/*.c) $(wildcard src/server/*.cpp)
SERVER_OBJS = $(patsubst %.cpp,%.o,$(patsubst %.c,%.o,$(SERVER_SRCS)))
# CUTLASS TUs need the CUTLASS include path + c++17; they build via dedicated rules below,
# so keep them out of the generic src/cuda/%.o rule.
CUTLASS_CUDA_OBJS = src/cuda/pulsar_mxfp4_cutlass.o
CUDA_SRCS = $(filter-out src/cuda/pulsar_mxfp4_cutlass.cu,$(wildcard src/cuda/*.cu))
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
LIB_HDRS = src/lib/pulsar_help.h src/lib/pulsar_kvstore.h
CORE_OBJS = $(ENGINE_OBJS) $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
PULSAR_LINK ?= $(NVCC) $(NVCCFLAGS)
PULSAR_LINK_LIBS ?= $(CUDA_LDLIBS)

.PHONY: all help clean test cuda-spark cuda-regression cuda-frontier-gate cuda-multiseq-gate cuda-multiseq-gate-nodspark cuda-bank-spec-gate cuda-accounting-gate cuda-evict-restore-gate cuda-fork-gate cuda-algo-stability-gate cuda-mixed-prefill-gate cuda-mixed-neutrality-gate cuda-prefill-gate cuda-prefill-gate-baseline cuda-spec-sampling-gate warm-fork-3way warm-partial-fork-3way

all: help

help:
	@echo "Pulsar build targets (CUDA-only fork for DGX Spark / GB10):"
	@echo "  make cuda-spark          Build for the DGX Spark / GB10 (sm_120f)"
	@echo "  make test                Build and run tests"
	@echo "  make cuda-regression     Kernel smokes vs synthetic slabs (modelless)"
	@echo "  make cuda-frontier-gate  Multiseq frontier-isolation gate (needs the model;"
	@echo "                           FRONTIER_MODEL=./ds4flash.gguf by default)"
	@echo "  make cuda-multiseq-gate  Multiseq-vs-solo token-stream gate + aggregate"
	@echo "                           throughput at N=1..3 (needs the model)"
	@echo "  make cuda-multiseq-gate-nodspark"
	@echo "                           The same gate with speculation disabled"
	@echo "                           (--no-dspark config; needs the model)"
	@echo "  make cuda-prefill-gate   Prefill bit-exactness gate: full-vocab frontier"
	@echo "                           logits byte-compared against a baseline build"
	@echo "                           (needs the model + a baseline blob)"
	@echo "  make cuda-prefill-gate-baseline"
	@echo "                           Build the baseline blob from PREFILL_BASELINE_REF"
	@echo "                           in a git worktree (needs the model)"
	@echo "  make cuda-spec-sampling-gate"
	@echo "                           Speculative-sampling chi-square exactness"
	@echo "                           oracle + acceptance alpha (needs the model)"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B pulsar-server CUDA_ARCH=sm_120f

pulsar-server: $(SERVER_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

# Development tools, not part of the shipped release. The release is just
# pulsar-server (the HTTP API). Source is kept; build these by name.
pulsar: src/cli/pulsar_cli.o src/lib/pulsar_help.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-bench: src/cli/pulsar_bench.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-eval: src/cli/pulsar_eval.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

pulsar-agent: $(AGENT_OBJS) src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(PULSAR_LINK) -o $@ $^ $(PULSAR_LINK_LIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

# Multiseq frontier-isolation gate (engine-side wrong-bank wiring; see the
# header of tests/multiseq_frontier_gate.c).  MODEL-DEPENDENT — run manually
# on the GB10, not part of `make test`.  Discipline before running: no
# pulsar-server/pulsar_test process, `sync; echo 3 > /proc/sys/vm/drop_caches`,
# check `free -g` headroom (the model is ~87 GB).
FRONTIER_MODEL ?= ./ds4flash.gguf
cuda-frontier-gate: tests/multiseq_frontier_gate
	DS4_MSEQ_BANKS=2 ./tests/multiseq_frontier_gate $(FRONTIER_MODEL)

# Multiseq-vs-solo token-stream gate + first aggregate-throughput measurement
# (see the header of tests/multiseq_decode_gate.c).  MODEL-DEPENDENT — run
# manually on the GB10 with the same memory discipline as the frontier gate.
cuda-multiseq-gate: tests/multiseq_decode_gate
	DS4_MSEQ_BANKS=3 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 3 512

# The same gate with speculation DISABLED — the pulsar-bench/pulsar-eval/agent and
# `pulsar-server --no-dspark` config, and a different allocation shape (no DSpark
# graph state).  The driver must work there; it used to reject every step.
# Shorter (N=2, 64 steps): this is a config gate, not a throughput run.
cuda-multiseq-gate-nodspark: tests/multiseq_decode_gate
	DS4_MSEQ_BANKS=2 DS4_GATE_NO_DSPARK=1 ./tests/multiseq_decode_gate $(FRONTIER_MODEL) 2 64

# Tier-2 PATH A / Option F: fused DSpark speculation on a BANK (N=1 spec-on-
# bank == classic, N=2 spec-time-slice with warm per-bank rings, mseq_dirty
# cheap resume).  See tests/bank_spec_gate.c.  MODEL-DEPENDENT, drafter-merged
# model, same memory discipline as the gates above; hold temp/gpu.lock.
cuda-bank-spec-gate: tests/bank_spec_gate
	DS4_MSEQ_BANKS=2 ./tests/bank_spec_gate $(FRONTIER_MODEL) 128

# Tier-2 overcommit accounting-exactness gate (task #55, increment 1): the
# exact-frontier touched-KV number the eviction guard triggers on must track the
# real cudaMemGetInfo physical delta of the demand-paged comp/index growth.  See
# tests/accounting_gate.c.  MODEL-DEPENDENT, same memory discipline as the gates
# above (hold temp/gpu.lock, drop_caches, no other pulsar process); fills stay
# modest (peak ~64k tokens) and do not exercise eviction.
cuda-accounting-gate: tests/accounting_gate
	DS4_MSEQ_BANKS=2 ./tests/accounting_gate $(FRONTIER_MODEL)

# Tier-2 increment 2b bank evict/restore bit-identity + reclaim gate (the
# memory-safety core; no OOM risk). See tests/bank_evict_restore_gate.c.
cuda-evict-restore-gate: tests/bank_evict_restore_gate tests/bank_fork_gate
	DS4_MSEQ_BANKS=2 ./tests/bank_evict_restore_gate $(FRONTIER_MODEL)

# Tier-2 PATH-A full-prefix fork gate (plan-33 inc A): fork==cold oracle. See
# tests/bank_fork_gate.c. MODEL-DEPENDENT, needs DS4_MSEQ_BANKS>=3.
cuda-fork-gate: tests/bank_fork_gate
	DS4_MSEQ_BANKS=3 ./tests/bank_fork_gate $(FRONTIER_MODEL)

# plan-34 phase-2 inc 2: cuBLASLt algo-stability. A decode bank's step logits must
# be byte-identical across batched-step widths M (incl. the M=4->5 custom->cuBLASLt
# boundary) so a co-scheduled big prefill (inc 4) cannot perturb it. MODEL-DEPENDENT,
# needs DS4_MSEQ_BANKS>=8. Run pack on/off + idx-fp4 on/off under GPU discipline.
cuda-algo-stability-gate: tests/algo_stability_gate
	DS4_MSEQ_BANKS=8 ./tests/algo_stability_gate $(FRONTIER_MODEL)

# plan-34 phase-2 inc 3: K-row single-bank prefill through the mixed entry —
# coherence vs classic, K>ratio boundary, tensor-core speed. MODEL-DEPENDENT.
cuda-mixed-prefill-gate: tests/mixed_prefill_gate
	DS4_MSEQ_BANKS=2 ./tests/mixed_prefill_gate $(FRONTIER_MODEL)

# plan-34 phase-2 inc 4: TRUE mixed step — decode banks + one K-row prefill run
# fused. Gate 4 co-scheduling neutrality (decode logits byte-identical with/without
# a co-scheduled prefill), gate 2 prefill correctness, gate 3 MoE two-pass split.
cuda-mixed-neutrality-gate: tests/mixed_neutrality_gate
	DS4_MSEQ_BANKS=3 ./tests/mixed_neutrality_gate $(FRONTIER_MODEL)

# plan-33 inc B: 3-way output-equality harness (server-level; see the script).
warm-fork-3way: pulsar-server
	bash tests/warm_fork_3way.sh $(FRONTIER_MODEL)

# plan-33 inc D: partial-prefix fork 3-way output-equality harness (server-level).
warm-partial-fork-3way: pulsar-server
	bash tests/warm_partial_fork_3way.sh $(FRONTIER_MODEL)

# Prefill bit-exactness gate (the D2R acceptance gate; see the header of
# tests/prefill_bitexact_gate.c).  MODEL-DEPENDENT — run manually on the GB10,
# not part of `make test`.  Discipline before running: no pulsar-server/pulsar_test
# process, `sync; echo 3 > /proc/sys/vm/drop_caches`, check `free -g` headroom
# (the model is ~87 GB).  The GPU is shared: hold temp/gpu.lock.
#
# Two-step, and the baseline is the slow half:
#   1. `make cuda-prefill-gate-baseline`  — ONCE per baseline ref.  Checks out
#      PREFILL_BASELINE_REF into a detached worktree, copies THIS tree's gate
#      source + Makefile into it (the harness must be identical; only the
#      engine/kernels may differ), builds there, and dumps the blob.
#   2. `make cuda-prefill-gate`           — after every D2R increment.
# Each step loads the model once (~35 s) and prefills 2*(512+2048+4096) tokens.
# Re-baselined 2026-07-26 to the v0.3.1 shipped commit: the old 8aa9d35 blob
# predated the v0.2.3 type-40/MXFP8_LT model repack, so its header no longer
# matched the shipped ds4flash.gguf ("baseline header mismatch — different
# model") AND the 8aa9d35 engine cannot even run the repacked tensors. The
# baseline now protects drift from the current shipped line.
PREFILL_BASELINE_REF ?= 536466c
PREFILL_BASELINE     ?= temp/prefill_bitexact_baseline.bin
PREFILL_BASELINE_WT  ?= temp/wt-prefill-baseline
# The blob stamps `git rev-parse --short HEAD` as resolved INSIDE the baseline
# worktree; normalise the expected ref through the same abbreviation so the
# provenance compare is not defeated by a differing --short length.
PREFILL_BASELINE_REF_SHORT := $(shell git rev-parse --short $(PREFILL_BASELINE_REF) 2>/dev/null || echo $(PREFILL_BASELINE_REF))

# Both sides of a byte-compare must be built for the SAME arch, so pin sm_120f
# here exactly as the baseline sub-make does.  CUDA_ARCH ?= defaults to EMPTY
# (top of this file), so only an explicit sm_120f produces the objects the gate
# must measure; passing it on the sub-make command line also overrides any
# CUDA_ARCH inherited from the caller's environment.
#
# -B COVERS THE WHOLE LINK, NOT JUST THE HARNESS, AND THAT IS THE POINT.  There
# is no -MMD/-MP anywhere in this Makefile: src/cuda/%.o hand-lists its headers.
# That list is complete today, but D2R is expected to land a new header (e.g.
# src/cuda/pulsar_cuda_d2r.cuh) included by pulsar_cuda_moe.cu; make would then see
# moe.o as up to date after a .cuh-only edit, relink the OLD kernel, and print
# PASS — the gate would certify a kernel it never compiled.  Forcing the whole
# tree costs a rebuild per run, which is acceptable: the gate is manual and
# spends ~3 GPU-minutes regardless, and an acceptance gate must never certify
# stale objects.  Forcing the link also rebuilds every object at the sm_120f
# pinned above, which retires the old "make clean first" arch caveat FOR THIS
# TARGET: a tree previously built at another arch can no longer leak a
# mismatched object into the compare.
#
# Project-wide `-MMD -MP` dependency tracking is the better long-term fix and is
# deliberately OUT OF SCOPE here (this Makefile is shared with a parallel branch;
# restructuring it would collide).  Until it lands, -B is what keeps the gate
# honest.
cuda-prefill-gate:
	$(MAKE) -B tests/prefill_bitexact_gate CUDA_ARCH=sm_120f
	./tests/prefill_bitexact_gate $(FRONTIER_MODEL) --check $(PREFILL_BASELINE) \
		$(PREFILL_BASELINE_REF_SHORT)

# NOTE: `git worktree add` does not populate submodules, so the baseline build
# is pointed at THIS tree's CUTLASS pin (a header-only include path).
cuda-prefill-gate-baseline:
	git worktree remove --force $(PREFILL_BASELINE_WT) 2>/dev/null || true
	rm -rf $(PREFILL_BASELINE_WT)
	git worktree prune
	git worktree add --detach $(PREFILL_BASELINE_WT) $(PREFILL_BASELINE_REF)
	cp tests/prefill_bitexact_gate.cpp $(PREFILL_BASELINE_WT)/tests/
	cp Makefile $(PREFILL_BASELINE_WT)/
	cp tests/long_context_story_prompt.txt $(PREFILL_BASELINE_WT)/tests/
	$(MAKE) -C $(PREFILL_BASELINE_WT) tests/prefill_bitexact_gate \
		CUDA_ARCH=sm_120f CUTLASS_DIR=$(abspath $(CUTLASS_DIR))
	cd $(PREFILL_BASELINE_WT) && ./tests/prefill_bitexact_gate $(abspath $(FRONTIER_MODEL)) \
		--dump $(abspath $(PREFILL_BASELINE))
	git worktree remove --force $(PREFILL_BASELINE_WT)
	@echo "baseline ($(PREFILL_BASELINE_REF)) -> $(PREFILL_BASELINE)"
# Speculative-sampling exactness oracle: chi-square of plain-sampled vs
# speculative per-position marginals, plus the greedy prefix gate and the
# acceptance rate alpha (see the header of tests/spec_sampling_gate.c).
# Proposal-agnostic, so it gates both the deterministic and the
# temperature-matched (p/q) accept rules.  MODEL-DEPENDENT — same memory
# discipline as the gates above; not part of `make test`.
SPEC_GATE_MODEL ?= gguf/model.gguf
SPEC_GATE_ARGS ?= 0.95
cuda-spec-sampling-gate: tests/spec_sampling_gate
	./tests/spec_sampling_gate $(SPEC_GATE_MODEL) $(SPEC_GATE_ARGS)

src/engine/%.o: src/engine/%.c src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

# Ported C++ TUs (pulsar). One rule per source dir, mirroring the .c rules.
src/engine/%.o: src/engine/%.cpp src/engine/pulsar_engine_internal.h src/engine/cursor.hpp src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/server/%.o: src/server/%.cpp src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.cpp src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.cpp src/pulsar.h src/lib/pulsar_help.h src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.cpp src/pulsar.h $(LIB_HDRS) src/lib/sha1.hpp
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ $<

src/agent/%.o: src/agent/%.c src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/server/%.o: src/server/%.c src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/cli/%.o: src/cli/%.c src/pulsar.h src/lib/pulsar_help.h src/vendor/linenoise.h
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/lib/%.o: src/lib/%.c src/pulsar.h $(LIB_HDRS)
	$(CC) $(CFLAGS) $(PULSAR_INC) -c -o $@ $<

src/vendor/%.o: src/vendor/%.cpp src/vendor/linenoise.h src/vendor/rax.h src/vendor/rax_malloc.h
	$(CXX) $(CXXFLAGS) -Wno-write-strings -c -o $@ $<

tests/pulsar_test.o: tests/pulsar_test.cpp $(SERVER_SRCS) src/server/pulsar_server_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/rax.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Wno-unused-function -c -o $@ tests/pulsar_test.cpp

tests/pulsar_agent_test.o: tests/pulsar_agent_test.cpp $(AGENT_SRCS) src/agent/pulsar_agent_internal.h src/pulsar.h $(LIB_HDRS) src/vendor/linenoise.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Wno-unused-function -c -o $@ tests/pulsar_agent_test.cpp

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.cpp src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) -Isrc -c -o $@ tests/cuda_long_context_smoke.cpp

tests/multiseq_frontier_gate.o: tests/multiseq_frontier_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/multiseq_frontier_gate.cpp

tests/multiseq_decode_gate.o: tests/multiseq_decode_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/multiseq_decode_gate.cpp

tests/bank_spec_gate.o: tests/bank_spec_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_spec_gate.cpp

tests/accounting_gate.o: tests/accounting_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/accounting_gate.cpp

tests/bank_evict_restore_gate.o: tests/bank_evict_restore_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_evict_restore_gate.cpp

tests/bank_fork_gate.o: tests/bank_fork_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/bank_fork_gate.cpp

tests/algo_stability_gate.o: tests/algo_stability_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/algo_stability_gate.cpp

tests/mixed_prefill_gate.o: tests/mixed_prefill_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mixed_prefill_gate.cpp

tests/mixed_neutrality_gate.o: tests/mixed_neutrality_gate.cpp src/engine/pulsar_engine_internal.h src/pulsar.h src/pulsar_gpu.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -Isrc/engine -c -o $@ tests/mixed_neutrality_gate.cpp

# Public-API only (pulsar.h): the gate must build unchanged against the baseline
# ref's tree, so it must not depend on engine internals that may have drifted.
# PULSAR_GATE_BUILD_REF stamps the blob with the git HEAD that built the dumper, so
# a baseline re-dumped from the tree under test cannot pass vacuously (see the
# file header).  cuda-prefill-gate force-rebuilds this object so the stamp is
# never stale.
GATE_BUILD_REF := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
tests/prefill_bitexact_gate.o: tests/prefill_bitexact_gate.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -DPULSAR_GATE_BUILD_REF='"$(GATE_BUILD_REF)"' \
		-c -o $@ tests/prefill_bitexact_gate.cpp
tests/spec_sampling_gate.o: tests/spec_sampling_gate.cpp src/pulsar.h
	$(CXX) $(CXXFLAGS) $(PULSAR_INC) -c -o $@ tests/spec_sampling_gate.cpp

src/cuda/%.o: src/cuda/%.cu src/cuda/pulsar_cuda_internal.h src/pulsar_gpu.h src/cuda/pulsar_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -Isrc -c -o $@ $<

# CUTLASS MXFP4 tensor-core expert FFN (GB10/sm_120f). Requires -arch=sm_120f (family mode) for the
# mxf4 block-scale MMA; build the whole engine with CUDA_ARCH=sm_120f so all objects match arch.
src/cuda/pulsar_mxfp4_cutlass.o: src/cuda/pulsar_mxfp4_cutlass.cu src/pulsar_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -diag-suppress 20012 -diag-suppress 177 -Isrc $(CUTLASS_INC) -c -o $@ src/cuda/pulsar_mxfp4_cutlass.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o $(CUDA_OBJS) $(CUTLASS_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_frontier_gate: tests/multiseq_frontier_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/multiseq_decode_gate: tests/multiseq_decode_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_spec_gate: tests/bank_spec_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/accounting_gate: tests/accounting_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_evict_restore_gate: tests/bank_evict_restore_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/bank_fork_gate: tests/bank_fork_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/algo_stability_gate: tests/algo_stability_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mixed_prefill_gate: tests/mixed_prefill_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/mixed_neutrality_gate: tests/mixed_neutrality_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/prefill_bitexact_gate: tests/prefill_bitexact_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/spec_sampling_gate: tests/spec_sampling_gate.o src/lib/pulsar_help.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

pulsar_test: tests/pulsar_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/rax.o $(CORE_OBJS) $(CUDA_LDLIBS)

pulsar_agent_test: tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/pulsar_agent_test.o src/lib/pulsar_help.o src/lib/pulsar_kvstore.o src/vendor/linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)

test: pulsar_test
	./pulsar_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test ds4_agent_test
	rm -f pulsar pulsar-server pulsar-bench pulsar-eval pulsar-agent pulsar_test pulsar_agent_test src/engine/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cli/*.o src/lib/*.o src/vendor/*.o tests/*.o tests/cuda_long_context_smoke tests/multiseq_frontier_gate tests/multiseq_decode_gate tests/prefill_bitexact_gate tests/bank_spec_gate tests/accounting_gate tests/bank_evict_restore_gate tests/bank_fork_gate
	rm -f pulsar pulsar-server pulsar-bench pulsar-eval pulsar-agent pulsar_test pulsar_agent_test src/engine/*.o src/agent/*.o src/server/*.o src/cuda/*.o src/cli/*.o src/lib/*.o src/vendor/*.o tests/*.o tests/cuda_long_context_smoke tests/multiseq_frontier_gate tests/multiseq_decode_gate tests/spec_sampling_gate tests/accounting_gate tests/bank_evict_restore_gate tests/bank_fork_gate

# --- Transitional aliases: old DwarfStar/ds4 target names (remove after ops
# migrate). Each alias depends on the new pulsar target and leaves an old-name
# symlink next to it so existing ops scripts and muscle memory keep working.
.PHONY: ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_test ds4_agent_test
ds4: pulsar
	ln -sf pulsar ds4
ds4-server: pulsar-server
	ln -sf pulsar-server ds4-server
ds4-bench: pulsar-bench
	ln -sf pulsar-bench ds4-bench
ds4-eval: pulsar-eval
	ln -sf pulsar-eval ds4-eval
ds4-agent: pulsar-agent
	ln -sf pulsar-agent ds4-agent
ds4_test: pulsar_test
	ln -sf pulsar_test ds4_test
ds4_agent_test: pulsar_agent_test
	ln -sf pulsar_agent_test ds4_agent_test
