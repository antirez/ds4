CC ?= cc
UNAME_S := $(shell uname -s)

# Default backend selection
ifeq ($(UNAME_S),Darwin)
    BACKEND ?= metal
else
    # On Linux, try to detect ROCm or CUDA if BACKEND is not set.
    # Default to 'cpu' if neither is found.
    ifeq ($(BACKEND),)
        ifneq ($(wildcard /opt/rocm/bin/hipcc),)
            BACKEND = rocm
        else ifneq ($(shell which nvcc 2>/dev/null),)
            BACKEND = cuda
        else
            BACKEND = cpu
        endif
    endif
endif

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

CORE_OBJS = ds4.o
CPU_CORE_OBJS = ds4_cpu.o

# Backend specific settings
ifeq ($(BACKEND),metal)
    METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
    CORE_OBJS += ds4_metal.o
    LDLIBS_BIN = $(METAL_LDLIBS)
    CC_BIN = $(CC)
endif

ifeq ($(BACKEND),cuda)
    CUDA_HOME ?= /usr/local/cuda
    NVCC ?= $(CUDA_HOME)/bin/nvcc
    CUDA_ARCH ?= native
    ifneq ($(strip $(CUDA_ARCH)),)
        NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
    endif
    NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
    CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
    CORE_OBJS += ds4_cuda.o
    CFLAGS += -DDS4_HAVE_CUDA
    LDLIBS_BIN = $(CUDA_LDLIBS)
    CC_BIN = $(NVCC)
    CC_BIN_FLAGS = $(NVCCFLAGS)
    REGRESSION_TEST = tests/cuda_long_context_smoke
endif

ifeq ($(BACKEND),rocm)
    ROCM_HOME ?= /opt/rocm
    HIPCC ?= $(ROCM_HOME)/bin/hipcc
    HIP_ARCH ?= native
    ifneq ($(strip $(HIP_ARCH)),)
        HIP_ARCH_FLAGS := --offload-arch=$(HIP_ARCH)
    endif
    HIPCCFLAGS ?= -O3 -ffast-math $(HIP_ARCH_FLAGS) $(NATIVE_CPU_FLAG) -pthread -Wno-unused-result
    HIP_LDLIBS ?= -lm -pthread -L$(ROCM_HOME)/lib -lhipblas -lamdhip64
    CORE_OBJS += ds4_hip.o
    CFLAGS += -DDS4_HAVE_ROCM
    LDLIBS_BIN = $(HIP_LDLIBS)
    CC_BIN = $(HIPCC)
    CC_BIN_FLAGS = $(HIPCCFLAGS)
    REGRESSION_TEST = tests/rocm_long_context_smoke
endif

ifeq ($(BACKEND),cpu)
    CFLAGS += -DDS4_NO_GPU
    CORE_OBJS = $(CPU_CORE_OBJS)
    LDLIBS_BIN = $(LDLIBS)
    CC_BIN = $(CC)
endif

ifeq ($(UNAME_S),Linux)
    CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
endif

.PHONY: all clean test cpu cuda-regression rocm-regression

all: ds4 ds4-server ds4-bench

ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC_BIN) $(CC_BIN_FLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS_BIN)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC_BIN) $(CC_BIN_FLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS_BIN)

ds4-bench: ds4_bench.o $(CORE_OBJS)
	$(CC_BIN) $(CC_BIN_FLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS_BIN)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

rocm-regression: tests/rocm_long_context_smoke
	./tests/rocm_long_context_smoke

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

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_hip.o: ds4_hip.cpp ds4_gpu.h ds4_iq2_tables_hip.inc
	$(HIPCC) $(HIPCCFLAGS) -c -o $@ ds4_hip.cpp

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/rocm_long_context_smoke.o: tests/rocm_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/rocm_long_context_smoke.c

tests/rocm_long_context_smoke: tests/rocm_long_context_smoke.o ds4_hip.o
	$(HIPCC) $(HIPCCFLAGS) -o $@ $^ $(HIP_LDLIBS)

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(CC_BIN) $(CC_BIN_FLAGS) $(CFLAGS) -o $@ $^ $(LDLIBS_BIN)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-bench ds4_cpu ds4_native ds4_server_test ds4_test *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/rocm_long_context_smoke tests/rocm_long_context_smoke.o
