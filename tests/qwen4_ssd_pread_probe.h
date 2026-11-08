/* Only forced into the SSD test's backend object. Include the system
 * declaration before renaming calls, leaving Darwin's symbol aliases intact. */
#include <unistd.h>
ssize_t ds4_test_pread(int fd, void *dst, size_t bytes, off_t offset);
#define pread ds4_test_pread

/* Observe the selected production dispatch only in this test backend. */
#include <stdint.h>
void ds4_test_qwen4_ssd_mm(uint32_t n_tokens, uint32_t weight_type,
                          int k32, int half, int streaming, int legacy_moe);
#define DS4_TEST_QWEN4_SSD_MM ds4_test_qwen4_ssd_mm
