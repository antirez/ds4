CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)
RAM_TEST_MB ?= 16384
RAM_TEST_STREAM_CACHE ?= 32
RAM_TEST_STREAM_WINDOW_MB ?= 8
RAM_TEST_PIN_MAX_MB ?= 1
RAM_TEST_COMPACT_CACHE_MB ?= 4096
RAM_TEST_CTX ?= 256
RAM_TEST_TOKENS ?= 2
RAM_TEST_PROMPT ?= Hi
RAM_TEST_ARGS ?= --stream-weights --nothink --temp 0

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all clean test-constrained-ram test-constrained-ram-matrix

all: ds4 ds4-server

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

test-constrained-ram: ds4
	env DS4_METAL_STREAM_WEIGHTS=1 \
	    DS4_METAL_STREAM_CACHE=$(RAM_TEST_STREAM_CACHE) \
	    DS4_METAL_STREAM_WINDOW_MB=$(RAM_TEST_STREAM_WINDOW_MB) \
	    DS4_METAL_STREAM_RAM_MB=$(RAM_TEST_MB) \
	    DS4_METAL_STREAM_PIN_MAX_MB=$(RAM_TEST_PIN_MAX_MB) \
	    DS4_METAL_COMPACT_EXPERT_CACHE_MB=$(RAM_TEST_COMPACT_CACHE_MB) \
	    DS4_METAL_NO_RESIDENCY=1 \
	    DS4_METAL_MEMORY_REPORT=1 \
	    /usr/bin/time -l ./ds4 $(RAM_TEST_ARGS) \
	        --ctx $(RAM_TEST_CTX) \
	        --tokens $(RAM_TEST_TOKENS) \
	        -p '$(RAM_TEST_PROMPT)'

test-constrained-ram-matrix: ds4
	@for cache in 1 8 16; do \
	    for window in 1 8 16; do \
	        echo "== cache=$$cache window=$$window MiB =="; \
	        $(MAKE) test-constrained-ram \
	            RAM_TEST_STREAM_CACHE=$$cache \
	            RAM_TEST_STREAM_WINDOW_MB=$$window \
	            RAM_TEST_TOKENS=$(RAM_TEST_TOKENS) \
	            RAM_TEST_PROMPT='$(RAM_TEST_PROMPT)' || exit $$?; \
	    done; \
	done

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test *.o
