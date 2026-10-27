#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ds4_ssd.h"

int main(void) {
    const uint64_t gib = 1ull << 30;
    ds4_ssd_cache_plan p;
    unsetenv("DS4_SSD_AUTO_CACHE_PCT");
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 0, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 86 * gib && p.cache_experts == 76);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 80 * gib && p.cache_experts == 70);
    /* A large context reduces the combined static/cache budget. */
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 65 * gib && p.cache_experts == 55);
    assert(p.effective_cache_bytes == 55 * gib);
    setenv("DS4_SSD_AUTO_CACHE_PCT", "95", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 65 * gib && p.cache_experts == 55);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 8, &p));
    assert(p.cache_experts == 8 && p.effective_cache_bytes == 8 * gib);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 5 * gib, 10 * gib, gib, 1000, &p));
    assert(p.cache_experts == 1);
    assert(!ds4_ssd_auto_cache_plan(0, 86, 0, 0, gib, 0, &p));
    assert(!ds4_ssd_auto_cache_plan(gib, 86, 0, 0, 0, 0, &p));
    assert(!ds4_ssd_auto_cache_plan(gib, 100, 0, 0, gib, 0, &p));
    unsetenv("DS4_SSD_AUTO_CACHE_PCT");
    puts("SSD cache sizing: PASS");
    return 0;
}
