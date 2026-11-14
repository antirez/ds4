/* Forced into tests/flint/test_flint_stream's build of ds4_flint_stream.c:
 * routes its pread calls through the test so reads can be counted and made
 * to fail.  Include the system declaration first, then rename. */
#include <unistd.h>
ssize_t ds4_test_pread(int fd, void *dst, size_t bytes, off_t offset);
#define pread ds4_test_pread
