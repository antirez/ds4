# Fork-owned build rules (Recursive Flint).  Included once from the upstream
# Makefile right after the platform object lists are defined and before any
# target that expands $(CORE_OBJS) as a prerequisite.  Keep everything fork
# specific here so the upstream Makefile diff stays at one line.

# Rules below must not steal the upstream default goal ("all").
FORK_SAVED_DEFAULT_GOAL := $(.DEFAULT_GOAL)

ifeq ($(UNAME_S),Darwin)
CORE_OBJS += ds4_metal_flint.o
FORK_METAL_OBJS := ds4_metal_flint.o

# metal/flint.metal is compiled at run time by ds4_metal_flint.m into its own
# strict-math library; it is deliberately not part of the shared library.
ds4_metal_flint.o: ds4_metal_flint.m ds4_metal_internal.h ds4_flint_gpu.h ds4_gpu.h
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal_flint.m

# ds4_metal.m includes the internal header (trampolines for out-of-tree kernel files).
ds4_metal.o: ds4_metal_internal.h
endif

# ds4.c textually includes the fork engine code.
ds4.o ds4_cpu.o: ds4_flint_shape.inc ds4_flint_weights.inc ds4_flint_model.inc ds4_flint_tokenizer.inc ds4_flint_unicode.inc ds4_flint.inc ds4_flint_gpu.h

.PHONY: fork-clean
clean: fork-clean
fork-clean:
	rm -f ds4_metal_flint.o

.DEFAULT_GOAL := $(FORK_SAVED_DEFAULT_GOAL)
