#ifndef DS4_LINUX_MEMORY_H
#define DS4_LINUX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/* CMA pages can back movable allocations, but not ordinary pinned GPU pages. */
static inline bool ds4_linux_nonmovable_memory_parse(FILE *fp, uint64_t *bytes) {
    char line[256];
    uint64_t available = 0, cma = 0;
    bool found = false;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kib;
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            available = (uint64_t)kib * 1024u;
            found = true;
        } else if (sscanf(line, "CmaFree: %llu kB", &kib) == 1) {
            if (kib > UINT64_MAX / 1024u) return false;
            cma = (uint64_t)kib * 1024u;
        }
    }
    if (!found || ferror(fp)) return false;
    *bytes = available > cma ? available - cma : 0;
    return true;
}

static inline bool ds4_linux_nonmovable_memory(uint64_t *bytes) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return false;
    const bool ok = ds4_linux_nonmovable_memory_parse(fp, bytes);
    fclose(fp);
    return ok;
}

#endif
