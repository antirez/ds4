#include "cursor.hpp"

/* C facade over pulsar::Cursor for the still-C GGUF loader. */

void cursor_error(pulsar_cursor *c, const char *msg) {
    pulsar::Cursor(*c).set_error(msg);
}



bool cursor_read(pulsar_cursor *c, void *dst, uint64_t n) {
    return pulsar::Cursor(*c).read(dst, n);
}



bool cursor_skip(pulsar_cursor *c, uint64_t n) {
    return pulsar::Cursor(*c).skip(n);
}



bool cursor_u32(pulsar_cursor *c, uint32_t *v) {
    return pulsar::Cursor(*c).u32(v);
}



bool cursor_u64(pulsar_cursor *c, uint64_t *v) {
    return pulsar::Cursor(*c).u64(v);
}



bool cursor_string(pulsar_cursor *c, pulsar_str *s) {
    return pulsar::Cursor(*c).string(s);
}
