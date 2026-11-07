# Fork-owned build rules (Recursive Flint).  Included once from the upstream
# Makefile right after the platform object lists are defined and before any
# target that expands $(CORE_OBJS) as a prerequisite.  Keep everything fork
# specific here so the upstream Makefile diff stays at one line.

# Rules below must not steal the upstream default goal ("all").
FORK_SAVED_DEFAULT_GOAL := $(.DEFAULT_GOAL)

ifeq ($(UNAME_S),Darwin)
# ds4_metal.m includes the internal header (trampolines for out-of-tree
# kernel files); metal/flint.metal joins the library through the upstream
# wildcard once it exists.  ds4_metal_flint.o is added here in step 1.
ds4_metal.o: ds4_metal_internal.h
endif

# ds4.c textually includes the fork engine code.
ds4.o ds4_cpu.o: ds4_flint_shape.inc ds4_flint_model.inc

.PHONY: fork-clean
clean: fork-clean
fork-clean:
	rm -f ds4_metal_flint.o

.DEFAULT_GOAL := $(FORK_SAVED_DEFAULT_GOAL)
