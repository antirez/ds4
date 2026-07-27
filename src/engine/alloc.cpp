#include "pulsar_engine_internal.h"

/* Checked allocators and the CPU-decode allocation guard. Split out of
 * util.c in the C++ port. The guard is process-wide state armed around
 * phases that must not allocate (CPU decode reuses preallocated scratch). */

namespace pulsar {

class AllocGuard {
public:
    void begin(const char *phase) {
        phase_ = phase;
        enabled_ = true;
    }

    void end() {
        enabled_ = false;
        phase_ = nullptr;
    }

    void check(const char *op, size_t size) const {
        if (!enabled_) return;
        fprintf(stderr,
                "pulsar: internal allocation during %s: %s(%zu). "
                "CPU decode is expected to reuse preallocated scratch buffers.\n",
                phase_ ? phase_ : "guarded phase",
                op,
                size);
        exit(1);
    }

private:
    bool enabled_ = false;
    const char *phase_ = nullptr;
};

static AllocGuard g_alloc_guard;

} // namespace pulsar



void pulsar_alloc_guard_begin(const char *phase) {
    pulsar::g_alloc_guard.begin(phase);
}



void pulsar_alloc_guard_end(void) {
    pulsar::g_alloc_guard.end();
}



void *xcalloc(size_t n, size_t size) {
    pulsar::g_alloc_guard.check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xmalloc(size_t size) {
    pulsar::g_alloc_guard.check("malloc", size);
    void *p = malloc(size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xrealloc(void *ptr, size_t size) {
    pulsar::g_alloc_guard.check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) pulsar_die("out of memory");
    return p;
}



void *xmalloc_zeroed(size_t n, size_t size) {
    if (size != 0 && n > SIZE_MAX / size) pulsar_die("allocation size overflow");
    const size_t total = n * size;
    void *p = xmalloc(total ? total : 1);
    /*
     * This is intentionally not calloc(). Large untouched calloc ranges may be
     * represented by the VM through shared zero-page bookkeeping. The CPU decode
     * KV cache grows one token at a time, so using calloc here can move thousands
     * of first-touch faults into generation. On Darwin we have observed this end
     * in a kernel cpt_mapcnt_inc overflow panic instead of a user-space error.
     *
     * Explicitly writing the zeroes while the cache is allocated keeps those VM
     * faults out of the token loop and gives the cache private resident pages.
     */
    memset(p, 0, total);
    return p;
}



char *pulsar_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = static_cast<char *>(xmalloc(n + 1));
    memcpy(p, s, n + 1);
    return p;
}
