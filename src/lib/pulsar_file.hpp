#pragma once

#include <cstdio>

/* RAII owner for a C FILE* (C++ port). Closes the handle on scope exit so that
 * early-return error paths cannot leak it — the recurring bug in the hand-rolled
 * "fclose only on the success path" idiom (e.g. `ok = nw == n && fclose(fp) == 0`,
 * which short-circuits past fclose on a short read/write).
 *
 * For files whose close-time flush error must be observed (write paths), call
 * close() explicitly and check its return — that returns fclose()'s status and
 * disarms the destructor, so behaviour is identical to the manual fclose while
 * the destructor remains a leak safety-net on the paths that forget it.
 *
 * Move-only; no exceptions (the project builds -fno-exceptions). */

namespace pulsar {

class FileHandle {
public:
    FileHandle() = default;
    explicit FileHandle(FILE *fp) : fp_(fp) {}
    ~FileHandle() { if (fp_) fclose(fp_); }

    FileHandle(const FileHandle &) = delete;
    FileHandle &operator=(const FileHandle &) = delete;
    FileHandle(FileHandle &&o) noexcept : fp_(o.fp_) { o.fp_ = nullptr; }
    FileHandle &operator=(FileHandle &&o) noexcept {
        if (this != &o) { if (fp_) fclose(fp_); fp_ = o.fp_; o.fp_ = nullptr; }
        return *this;
    }

    FILE *get() const { return fp_; }
    explicit operator bool() const { return fp_ != nullptr; }

    /* Explicit close; returns fclose()'s status (0 on success). Idempotent: the
     * destructor will not double-close after this. */
    int close() {
        if (!fp_) return 0;
        int rc = fclose(fp_);
        fp_ = nullptr;
        return rc;
    }

private:
    FILE *fp_ = nullptr;
};

} // namespace pulsar
