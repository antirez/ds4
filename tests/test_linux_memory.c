#include <assert.h>
#include <string.h>
#include "ds4_linux_memory.h"

static void memory_case(const char *text, bool valid, uint64_t expected) {
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

static void cgroup_case(const char *limit, const char *usage, bool valid,
                        uint64_t expected) {
    FILE *lfp = text_file(limit), *ufp = text_file(usage);
    uint64_t bytes = 123;
    assert(ds4_linux_cgroup_headroom_parse(lfp, ufp, &bytes) == valid);
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

    memory_case("MemAvailable: 100 kB\n", true, 100 * 1024);
    memory_case("MemAvailable: 100 kB\nCmaFree: 30 kB\n", true, 70 * 1024);
    memory_case("CmaTotal: 0 kB\nCmaFree: 30 kB\nMemAvailable: 100 kB\n",
                true, 70 * 1024);
    memory_case("MemAvailable: 10 kB\nCmaFree: 30 kB\n", true, 0);
    memory_case("MemAvailable: 0 kB\n", true, 0);
    memory_case("CmaFree: 30 kB\n", false, 0);
    memory_case("MemAvailable: 18446744073709551615 kB\n", false, 0);

    uint64_t host = 0, cgroup = 0;
    if (ds4_linux_nonmovable_memory(&host)) {
        printf("live probe: %.2f GiB usable", (double)host / 1073741824.0);
        if (ds4_linux_cgroup_headroom(&cgroup)) {
            printf(", cgroup headroom %.2f GiB", (double)cgroup / 1073741824.0);
            assert(host <= cgroup);
        }
        putchar('\n');
    }
    puts("Linux nonmovable memory: PASS");
    return 0;
}
