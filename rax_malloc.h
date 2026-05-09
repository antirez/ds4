/* Rax -- A radix tree implementation.
 *
 * Copyright (c) 2017, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * This file is part of a radix tree implementation used for efficient
 * prefix-based key indexing and compressed trie structures.
 *
 * The implementation is widely used in high-performance systems such as Redis
 * for sorted key traversal, autocomplete, and memory-efficient string mapping.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted under the BSD 3-Clause License.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

/*
 * ---------------------------------------------------------------------------
 * Allocator Abstraction Layer (Compile-Time Swappable Memory Backend)
 * ---------------------------------------------------------------------------
 *
 * This header defines the memory allocation strategy used internally by Rax.
 *
 * The design allows swapping the underlying allocator without modifying core
 * radix-tree logic, enabling integration with custom memory systems such as:
 *
 *   - jemalloc (high-performance production allocator)
 *   - tcmalloc (Google allocator optimized for concurrency)
 *   - custom arena allocators (embedded / ML inference engines)
 *   - debug allocators (memory tracking / leak detection)
 *
 * This abstraction is critical for:
 *
 *   1. Memory profiling and instrumentation
 *   2. Reducing fragmentation in long-running services
 *   3. Embedding Rax into constrained runtime environments
 *   4. Replacing libc allocator without touching core logic
 *
 * ---------------------------------------------------------------------------
 * Design Principle:
 *   Keep radix tree logic allocator-agnostic and fully portable.
 * ---------------------------------------------------------------------------
 */

#ifndef RAX_ALLOC_H
#define RAX_ALLOC_H

/*
 * Default allocator mapping to standard C runtime (libc).
 *
 * These macros can be overridden at compile time to redirect memory
 * operations to a custom allocator implementation.
 *
 * Example:
 *   -Drax_malloc=my_malloc
 *   -Drax_realloc=my_realloc
 *   -Drax_free=my_free
 */

#ifndef rax_malloc
#define rax_malloc  malloc
#endif

#ifndef rax_realloc
#define rax_realloc realloc
#endif

#ifndef rax_free
#define rax_free    free
#endif

/*
 * ---------------------------------------------------------------------------
 * Extension Notes (Optional Enhancements)
 * ---------------------------------------------------------------------------
 *
 * Advanced systems may extend this abstraction with:
 *
 *   - allocation tagging (debug builds)
 *   - memory arenas per radix subtree
 *   - NUMA-aware allocation strategies
 *   - zero-allocation mode for embedded inference graphs
 *
 * Example future extension:
 *
 *   #define rax_malloc(size) arena_alloc(global_arena, size)
 *
 * ---------------------------------------------------------------------------
 */

#endif /* RAX_ALLOC_H */
