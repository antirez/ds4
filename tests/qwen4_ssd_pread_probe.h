/* Only forced into the SSD test's backend object. Include the system
 * declaration before renaming calls, leaving Darwin's symbol aliases intact. */
#include <unistd.h>
ssize_t ds4_test_pread(int fd, void *dst, size_t bytes, off_t offset);
#define pread ds4_test_pread
