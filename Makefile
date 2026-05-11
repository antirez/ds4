CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o ds4_rpc.o
NATIVE_CORE_OBJS = ds4_native.o ds4_rpc.o
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o ds4_rpc.o
NATIVE_CORE_OBJS = ds4_native.o ds4_rpc.o
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all clean test

all: ds4 ds4-server ds4-rpc-worker

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-rpc-worker: ds4_rpc_worker.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_rpc_worker.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-rpc-worker: ds4_rpc_worker.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_rpc.o: ds4_rpc.c ds4_rpc.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_rpc.c

ds4_rpc_worker.o: ds4_rpc_worker.c ds4_rpc.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_rpc_worker.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4-rpc-worker ds4_native ds4_server_test ds4_test *.o
