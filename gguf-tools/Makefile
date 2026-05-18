CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_TARGETS := ds4.o ds4_metal.o
QUALITY_OBJS := ../ds4.o ../ds4_metal.o
QUALITY_LINK := $(CC) $(CFLAGS) -I.. -o
else
NATIVE_CPU_FLAG ?= -march=native
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_TARGETS := ds4.o ds4_cuda.o
QUALITY_OBJS := ../ds4.o ../ds4_cuda.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I.. -o
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official
