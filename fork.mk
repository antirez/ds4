# Fork-owned build rules (Recursive Flint).  Included once from the upstream
# Makefile right after the platform object lists are defined and before any
# target that expands $(CORE_OBJS) as a prerequisite.  Keep everything fork
# specific here so the upstream Makefile diff stays at one line.

# Rules below must not steal the upstream default goal ("all").
FORK_SAVED_DEFAULT_GOAL := $(.DEFAULT_GOAL)

# SSD streaming of Flint's routed experts (plain C, every backend links it).
CORE_OBJS += ds4_flint_stream.o
CPU_CORE_OBJS += ds4_flint_stream.o
ds4_flint_stream.o: ds4_flint_stream.c ds4_flint_stream.h
	$(CC) $(CFLAGS) -c -o $@ ds4_flint_stream.c

# Model-free test of the expert cache over a sparse temporary file.
.PHONY: test-flint-stream
test-flint-stream: tests/flint/test_flint_stream
	./tests/flint/test_flint_stream
tests/flint/test_flint_stream: tests/flint/test_flint_stream.c ds4_flint_stream.c ds4_flint_stream.h tests/flint/flint_pread_probe.h
	$(CC) -O2 -Wall -Wextra -std=c99 -D_GNU_SOURCE -D_DARWIN_C_SOURCE -include tests/flint/flint_pread_probe.h -o $@ tests/flint/test_flint_stream.c ds4_flint_stream.c -lpthread

ifeq ($(UNAME_S),Darwin)
CORE_OBJS += ds4_metal_flint.o
FORK_METAL_OBJS := ds4_metal_flint.o

# metal/flint.metal is compiled at run time by ds4_metal_flint.m into its own
# strict-math library; it is deliberately not part of the shared library.
ds4_metal_flint.o: ds4_metal_flint.m ds4_metal_internal.h ds4_flint_gpu.h ds4_flint_stream.h ds4_gpu.h
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal_flint.m

# ds4_metal.m includes the internal header (trampolines for out-of-tree kernel files).
ds4_metal.o: ds4_metal_internal.h

# Model-free kernel tests for the prefill shader (CPU references, random data).
.PHONY: test-flint-kernels
test-flint-kernels: tests/flint/test_flint_kernels
	./tests/flint/test_flint_kernels metal/flint_prefill.metal
tests/flint/test_flint_kernels: tests/flint/test_flint_kernels.m
	$(CC) -fobjc-arc -O2 -framework Metal -framework Foundation -o $@ tests/flint/test_flint_kernels.m
# Model-free micro-benchmarks of the prefill attention and GEMM kernels
# (GPU timestamps; run them alone on the GPU, on mains, in High Power Mode).
.PHONY: bench-flint-kernels
bench-flint-kernels: tests/flint/bench_flint_attention tests/flint/bench_flint_gemm
tests/flint/bench_flint_attention: tests/flint/bench_flint_attention.m
	$(CC) -fobjc-arc -O2 -framework Metal -framework Foundation -o $@ tests/flint/bench_flint_attention.m
tests/flint/bench_flint_gemm: tests/flint/bench_flint_gemm.m
	$(CC) -fobjc-arc -O2 -framework Metal -framework Foundation -o $@ tests/flint/bench_flint_gemm.m
endif

# ds4_server.c (and the server tests inside ds4_test) include the Flint renderer.
ds4_server.o ds4_server_cpu.o ds4_test.o: ds4_flint_server.inc ds4_flint_server_test.inc

# ds4.c textually includes the fork engine code.
ds4.o ds4_cpu.o: ds4_flint_shape.inc ds4_flint_weights.inc ds4_flint_model.inc ds4_flint_tokenizer.inc ds4_flint_unicode.inc ds4_flint.inc ds4_flint_session.inc ds4_flint_gpu.h ds4_flint_stream.h

.PHONY: fork-clean
clean: fork-clean
fork-clean:
	rm -f ds4_metal_flint.o ds4_flint_stream.o tests/flint/test_flint_kernels tests/flint/test_flint_stream \
	      tests/flint/bench_flint_attention tests/flint/bench_flint_gemm

.DEFAULT_GOAL := $(FORK_SAVED_DEFAULT_GOAL)
