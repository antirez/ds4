#include <assert.h>
#include <stdio.h>
#include <string.h>
static FILE *cgroup_fopen(const char *path, const char *mode);
#define fopen cgroup_fopen
#include "ds4_linux_memory.h"
#undef fopen

static const struct cgroup_fixture {
    bool v1;
    const char *limit, *usage, *stat;
} *cgroup_fixture;
static unsigned fixture_count;

static void memory_case(const char *text, bool valid, uint64_t expected) {
    fixture_count++;
    FILE *fp = tmpfile();
    assert(fp);
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    rewind(fp);
    uint64_t bytes = 123;
    assert(ds4_linux_nonmovable_memory_parse(fp, &bytes) == valid);
    if (valid) assert(bytes == expected);
    fclose(fp);
}

static FILE *text_file(const char *text) {
    if (!text) return NULL;
    FILE *fp = tmpfile();
    assert(fp);
    assert(fwrite(text, 1, strlen(text), fp) == strlen(text));
    rewind(fp);
    return fp;
}

/* Exercise the real membership/path reader with real streams, including the
 * choice of v1 total_* counters and a missing memory.stat. */
static FILE *cgroup_fopen(const char *path, const char *mode) {
    if (!cgroup_fixture) return fopen(path, mode);
    if (!strcmp(path, "/proc/self/cgroup")) {
        return text_file(cgroup_fixture->v1 ? "5:cpu,memory:/fixture\n"
                                           : "0::/fixture\n");
    }
    if (!strcmp(path, "/proc/meminfo")) return fopen(path, mode);
    const char *prefix = cgroup_fixture->v1 ? "/sys/fs/cgroup/memory/fixture/"
                                           : "/sys/fs/cgroup/fixture/";
    assert(!strncmp(path, prefix, strlen(prefix)));
    const char *name = path + strlen(prefix);
    if (!strcmp(name, cgroup_fixture->v1 ? "memory.limit_in_bytes" : "memory.max"))
        return text_file(cgroup_fixture->limit);
    if (!strcmp(name, cgroup_fixture->v1 ? "memory.usage_in_bytes" : "memory.current"))
        return text_file(cgroup_fixture->usage);
    assert(!strcmp(name, "memory.stat"));
    return text_file(cgroup_fixture->stat);
}

static void cgroup_cache_case(bool v1, const char *stat, const char *limit,
                              const char *usage, bool valid, uint64_t expected) {
    fixture_count++;
    const struct cgroup_fixture fixture = {v1, limit, usage, stat};
    cgroup_fixture = &fixture;
    uint64_t bytes = 123;
    assert(ds4_linux_cgroup_headroom(&bytes) == valid);
    if (valid && bytes != expected) {
        fprintf(stderr, "v%d memory.stat headroom: got %llu, expected %llu bytes\n",
                v1 ? 1 : 2, (unsigned long long)bytes, (unsigned long long)expected);
    }
    if (valid) assert(bytes == expected);
    cgroup_fixture = NULL;
}

static void cgroup_case(const char *limit, const char *usage, bool valid,
                        uint64_t expected) {
    fixture_count++;
    FILE *lfp = text_file(limit), *ufp = text_file(usage);
    uint64_t bytes = 123;
    assert(ds4_linux_cgroup_headroom_parse(lfp, ufp, NULL, false, &bytes) == valid);
    if (valid) assert(bytes == expected);
    if (lfp) fclose(lfp);
    if (ufp) fclose(ufp);
}

int main(void) {
    /* v2: memory.max / memory.current. */
    cgroup_case("12884901888\n", "1073741824\n", true, 11811160064ull);
    cgroup_case("12884901888\n", "12884901888\n", true, 0);
    cgroup_case("1024\n", "4096\n", true, 0);
    cgroup_case("max\n", "1073741824\n", false, 0);
    /* v1: memory.limit_in_bytes / memory.usage_in_bytes, LONG_MAX page-rounded
     * when unlimited. */
    cgroup_case("9223372036854771712\n", "1073741824\n", false, 0);
    cgroup_case("-1\n", "1073741824\n", false, 0);
    /* Missing or malformed files keep the host number. */
    cgroup_case(NULL, "1073741824\n", false, 0);
    cgroup_case("12884901888\n", NULL, false, 0);
    cgroup_case("", "1073741824\n", false, 0);
    cgroup_case("12884901888\n", "", false, 0);
    cgroup_case("twelve\n", "1073741824\n", false, 0);
    cgroup_case("12884901888\n", "some\n", false, 0);

    /* 24 GiB limit - (20 GiB usage - 14 GiB file cache) = 18 GiB. */
    const char *limit = "25769803776\n", *usage = "21474836480\n";
    cgroup_cache_case(false, "inactive_file 8589934592\nactive_file 6442450944\n",
                      limit, usage, true, 19327352832ull);
    cgroup_cache_case(true, "inactive_file 1\nactive_file 2\n"
                            "total_inactive_file 8589934592\n"
                            "total_active_file 6442450944\n",
                      limit, usage, true, 19327352832ull);
    cgroup_cache_case(false, "active_file 6442450944\n", limit, usage,
                      true, 10737418240ull);
    cgroup_cache_case(false, "inactive_file 8589934592\n", limit, usage,
                      true, 12884901888ull);
    cgroup_cache_case(true, "inactive_file 8589934592\nactive_file 6442450944\n",
                      limit, usage, true, 4294967296ull);
    cgroup_cache_case(false, "total_inactive_file 8589934592\n"
                             "total_active_file 6442450944\n",
                      limit, usage, true, 4294967296ull);
    cgroup_cache_case(false, "file 20000000000\nshmem 1000000000\n",
                      limit, usage, true, 4294967296ull);
    cgroup_cache_case(false, "active_file 21474836481\n", limit, usage,
                      true, 25769803776ull);
    cgroup_cache_case(false, "active_file 21474836480\n", limit, usage,
                      true, 25769803776ull);
    cgroup_cache_case(false, "active_file 14\n", "10\n", "30\n", true, 0);
    cgroup_cache_case(false, "active_file 25\n", "10\n", "30\n", true, 5);

    /* Dirty and writeback pages are not room: of 14 GiB on the file lists,
     * 2 GiB dirty and 1 GiB under writeback leave 11 GiB, so 15 GiB of room. */
    cgroup_cache_case(false, "inactive_file 8589934592\nactive_file 6442450944\n"
                             "file_dirty 2147483648\nfile_writeback 1073741824\n",
                      limit, usage, true, 16106127360ull);
    cgroup_cache_case(true, "dirty 1\nwriteback 1\n"
                            "total_inactive_file 8589934592\n"
                            "total_active_file 6442450944\n"
                            "total_dirty 2147483648\ntotal_writeback 1073741824\n",
                      limit, usage, true, 16106127360ull);
    cgroup_cache_case(false, "inactive_file 8589934592\nactive_file 6442450944\n"
                             "file_dirty 2147483648\n",
                      limit, usage, true, 17179869184ull);
    cgroup_cache_case(false, "inactive_file 8589934592\nactive_file 6442450944\n"
                             "file_writeback 1073741824\n",
                      limit, usage, true, 18253611008ull);
    cgroup_cache_case(false, "inactive_file 8589934592\nactive_file 6442450944\n"
                             "total_dirty 2147483648\ntotal_writeback 1073741824\n",
                      limit, usage, true, 19327352832ull);
    cgroup_cache_case(false, "inactive_file 1073741824\nfile_dirty 2147483648\n",
                      limit, usage, true, 4294967296ull);

    /* Any malformed stat disables the entire subtraction, even after a valid
     * counter. The valid finite limit must still bind. */
    const char *bad_stats[] = {
        NULL, "", "active_file nope\n", "active_file -1\n", "active_file +1\n",
        "active_file 18446744073709551616\n", "active_file 3junk\n",
        "active_file\n", "active_file 3 extra\n",
        "inactive_file 8589934592\nactive_file nope\n",
        "active_file 6442450944\nactive_file 6442450944\n",
        "inactive_file 8589934592\ninactive_file 8589934592\n",
        "active_file 18446744073709551615\ninactive_file 1\n",
        "active_file 6442450944\nanon nope\n",
        /* A line with the wrong number of fields after a valid counter. */
        "inactive_file 8589934592\nbroken\n",
        "inactive_file 8589934592\nanon 1 2\n",
        /* Overflows that would wrap to 6 GiB of credit, not to zero. */
        "active_file 18446744073709551615\ninactive_file 6442450945\n",
        "active_file 18446744080152002560\n",
        "inactive_file 8589934592\nfile_dirty 1\nfile_dirty 2\n",
        "inactive_file 8589934592\nfile_writeback 1\nfile_writeback 2\n"
    };
    for (size_t i = 0; i < sizeof(bad_stats) / sizeof(bad_stats[0]); i++) {
        cgroup_cache_case(false, bad_stats[i], limit, usage, true, 4294967296ull);
    }
    cgroup_cache_case(true, "total_active_file -1\n", limit, usage,
                      true, 4294967296ull);
    cgroup_cache_case(true, "total_inactive_file 8589934592\n"
                            "total_active_file nope\n",
                      limit, usage, true, 4294967296ull);
    cgroup_cache_case(true, "total_active_file 1\ntotal_active_file 2\n",
                      limit, usage, true, 4294967296ull);
    cgroup_cache_case(false, "active_file 6442450944", limit, usage,
                      true, 10737418240ull);
    cgroup_cache_case(false, "active_file 6442450944\n", "max\n", usage, false, 0);
    cgroup_cache_case(false, "active_file 6442450944\n", limit, NULL, false, 0);
    cgroup_cache_case(false, "active_file 6442450944\n", NULL, usage, false, 0);

    memory_case("MemAvailable: 100 kB\n", true, 100 * 1024);
    memory_case("MemAvailable: 100 kB\nCmaFree: 30 kB\n", true, 70 * 1024);
    memory_case("CmaTotal: 0 kB\nCmaFree: 30 kB\nMemAvailable: 100 kB\n",
                true, 70 * 1024);
    memory_case("MemAvailable: 10 kB\nCmaFree: 30 kB\n", true, 0);
    memory_case("MemAvailable: 0 kB\n", true, 0);
    memory_case("CmaFree: 30 kB\n", false, 0);
    memory_case("MemAvailable: 18446744073709551615 kB\n", false, 0);

    /* The clamp itself: the live /proc/meminfo against a fixture cgroup with
     * 4096 bytes left. */
    {
        fixture_count++;
        const struct cgroup_fixture tiny = {false, "4096\n", "0\n", NULL};
        cgroup_fixture = &tiny;
        uint64_t clamped = 0;
        if (ds4_linux_nonmovable_memory(&clamped)) assert(clamped <= 4096);
        cgroup_fixture = NULL;
    }

    /* Printed, not asserted: two reads of a live cgroup are two snapshots of
     * a moving number, and comparing them fails when usage rises in between. */
    uint64_t host = 0, cgroup = 0;
    if (ds4_linux_nonmovable_memory(&host)) {
        printf("live probe: %.2f GiB usable", (double)host / 1073741824.0);
        if (ds4_linux_cgroup_headroom(&cgroup)) {
            printf(", cgroup headroom %.2f GiB", (double)cgroup / 1073741824.0);
        }
        putchar('\n');
    }
    printf("Linux nonmovable memory: PASS (%u fixtures)\n", fixture_count);
    return 0;
}
