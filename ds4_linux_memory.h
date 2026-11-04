#ifndef DS4_LINUX_MEMORY_H
#define DS4_LINUX_MEMORY_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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

/* A cgroup memory limit binds before MemAvailable does. "max", the v1
 * LONG_MAX sentinel, and unreadable files mean no finite limit applies. */
static inline bool ds4_linux_cgroup_headroom_parse(FILE *limit_fp, FILE *usage_fp,
                                                   uint64_t *bytes) {
    unsigned long long limit, usage;
    if (!limit_fp || !usage_fp) return false;
    if (fscanf(limit_fp, "%llu", &limit) != 1 || limit >= (1ull << 62)) return false;
    if (fscanf(usage_fp, "%llu", &usage) != 1) return false;
    *bytes = limit > usage ? limit - usage : 0;
    return true;
}

/* Headroom of this process's own cgroup: v2 memory.max/memory.current, or the
 * v1 memory controller's limit_in_bytes/usage_in_bytes. */
static inline bool ds4_linux_cgroup_headroom(uint64_t *bytes) {
    FILE *fp = fopen("/proc/self/cgroup", "r");
    if (!fp) return false;
    char line[512], limit_path[640] = "", usage_path[640] = "";
    while (fgets(line, sizeof(line), fp)) {
        char *controllers = strchr(line, ':'), *path;
        if (!controllers || !(path = strchr(controllers + 1, ':'))) continue;
        *path++ = '\0';
        path[strcspn(path, "\n")] = '\0';
        controllers++;
        if (controllers[0] == '\0') {
            snprintf(limit_path, sizeof(limit_path), "/sys/fs/cgroup%s/memory.max", path);
            snprintf(usage_path, sizeof(usage_path), "/sys/fs/cgroup%s/memory.current", path);
            break;
        }
        const char *memory = strstr(controllers, "memory");
        if (memory && (memory == controllers || memory[-1] == ',') &&
            (memory[6] == '\0' || memory[6] == ',')) {
            snprintf(limit_path, sizeof(limit_path),
                     "/sys/fs/cgroup/memory%s/memory.limit_in_bytes", path);
            snprintf(usage_path, sizeof(usage_path),
                     "/sys/fs/cgroup/memory%s/memory.usage_in_bytes", path);
            break;
        }
    }
    fclose(fp);
    if (!limit_path[0]) return false;
    FILE *limit_fp = fopen(limit_path, "r"), *usage_fp = fopen(usage_path, "r");
    const bool ok = ds4_linux_cgroup_headroom_parse(limit_fp, usage_fp, bytes);
    if (limit_fp) fclose(limit_fp);
    if (usage_fp) fclose(usage_fp);
    return ok;
}

static inline bool ds4_linux_nonmovable_memory(uint64_t *bytes) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return false;
    const bool ok = ds4_linux_nonmovable_memory_parse(fp, bytes);
    fclose(fp);
    uint64_t cgroup;
    if (ok && ds4_linux_cgroup_headroom(&cgroup) && cgroup < *bytes) *bytes = cgroup;
    return ok;
}

#endif
