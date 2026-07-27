#ifndef PULSAR_CURSOR_HPP
#define PULSAR_CURSOR_HPP

#include "pulsar_engine_internal.h"

/* Bounds-checked byte cursor over the mmapped GGUF file (C++ port of the
 * pulsar_cursor free functions). The state stays in the C pulsar_cursor struct so
 * the still-C gguf.c can hold and pass cursors; this class is a typed view
 * over one. When gguf.c ports, the struct folds into the class. */

namespace pulsar {

class Cursor {
public:
    explicit Cursor(pulsar_cursor &c) : c_(c) {}

    void set_error(const char *msg) {
        if (c_.error[0] == '\0') {
            snprintf(c_.error, sizeof(c_.error), "%s at byte %" PRIu64, msg, c_.pos);
        }
    }

    bool has(uint64_t n) {
        if (n > c_.size || c_.pos > c_.size - n) {
            set_error("truncated GGUF file");
            return false;
        }
        return true;
    }

    bool read(void *dst, uint64_t n) {
        if (!has(n)) return false;
        memcpy(dst, c_.base + c_.pos, (size_t)n);
        c_.pos += n;
        return true;
    }

    bool skip(uint64_t n) {
        if (!has(n)) return false;
        c_.pos += n;
        return true;
    }

    bool u32(uint32_t *v) { return read(v, sizeof(*v)); }
    bool u64(uint64_t *v) { return read(v, sizeof(*v)); }

    bool string(pulsar_str *s) {
        uint64_t len;
        if (!u64(&len)) return false;
        if (!has(len)) return false;
        s->ptr = (const char *)(c_.base + c_.pos);
        s->len = len;
        c_.pos += len;
        return true;
    }

private:
    pulsar_cursor &c_;
};

} // namespace pulsar

#endif /* PULSAR_CURSOR_HPP */
