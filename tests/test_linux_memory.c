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

int main(void) {
    memory_case("MemAvailable: 100 kB\n", true, 100 * 1024);
    memory_case("MemAvailable: 100 kB\nCmaFree: 30 kB\n", true, 70 * 1024);
    memory_case("CmaTotal: 0 kB\nCmaFree: 30 kB\nMemAvailable: 100 kB\n",
                true, 70 * 1024);
    memory_case("MemAvailable: 10 kB\nCmaFree: 30 kB\n", true, 0);
    memory_case("MemAvailable: 0 kB\n", true, 0);
    memory_case("CmaFree: 30 kB\n", false, 0);
    memory_case("MemAvailable: 18446744073709551615 kB\n", false, 0);
    puts("Linux nonmovable memory: PASS");
    return 0;
}
