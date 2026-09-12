"""Host-only Linux regression: python3 tests/test_cuda_source_pages.py.

Uses CXX (default: c++); no CUDA toolkit or GPU is needed. The production
guard trusts the fd association's immutable-file contract, not arbitrary COW
mappings maliciously registered as immutable. Negative provenance tests do
not associate their anonymous or modified private mappings with the model fd.

Caller tests extract complete production functions and page-advice helpers;
CUDA, staging, and cache infrastructure are host stubs, not GPU integration.
Caller syscall arguments/order are recorded (not issued); the original helper
test above exercises real mmap/madvise/refault behavior. Caller fixtures that
assert page-aligned ranges require 4 KiB host pages and fail explicitly otherwise.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


PREAMBLE = r"""
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)

static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static uint64_t g_model_file_size;
static unsigned calls;
static void *advised_address;
static size_t advised_length;
static int advised_kind;

static int record_madvise(void *address, size_t length, int advice) {
    ++calls;
    advised_address = address;
    advised_length = length;
    advised_kind = advice;
    int result = madvise(address, length, advice);
    CHECK(result == 0);
    return result;
}
#define madvise record_madvise
"""

HARNESS = r"""
#undef madvise

// Locate by address, not pathname: the temporary file has no stable name.
static long rss_kib(const void *address) {
    FILE *smaps = fopen("/proc/self/smaps", "r");
    if (!smaps) return -1;
    char line[512];
    bool selected = false;
    long rss = -1;
    while (fgets(line, sizeof(line), smaps)) {
        unsigned long begin, end;
        if (sscanf(line, "%lx-%lx", &begin, &end) == 2)
            selected = (uintptr_t)address >= begin && (uintptr_t)address < end;
        else if (selected && sscanf(line, "Rss: %ld kB", &rss) == 1)
            break;
    }
    fclose(smaps);
    return rss;
}

int main() {
    CHECK(unsetenv("DS4_CUDA_KEEP_MODEL_PAGES") == 0);
    const long page_long = sysconf(_SC_PAGESIZE);
    CHECK(page_long > 0);
    const size_t page = (size_t)page_long;
    const size_t size = 65536 + page / 2;
    CHECK(size % page != 0);
    std::vector<unsigned char> expected(size);
    for (size_t i = 0; i < size; ++i)
        expected[i] = (unsigned char)((i * 37 + i / 251) % 256);
    FILE *file = tmpfile();
    CHECK(file != NULL);
    CHECK(fwrite(expected.data(), 1, size, file) == size);
    CHECK(fflush(file) == 0);
    int fd = fileno(file);
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(map != MAP_FAILED);
    g_model_file_size = size;
    g_model_fd_host_base = map;

    auto no_call = [&](const void *base, uint64_t model_size,
                       uint64_t offset, uint64_t bytes, bool force) {
        unsigned before = calls;
        cuda_model_discard_source_pages_impl(base, model_size, offset, bytes, force);
        CHECK(calls == before);
    };
    auto discard = [&](uint64_t offset, uint64_t bytes, bool force,
                       size_t start, size_t length) {
        unsigned before = calls;
        cuda_model_discard_source_pages_impl(map, size, offset, bytes, force);
        CHECK(calls == before + 1);
        CHECK(advised_address == (const unsigned char *)map + start);
        CHECK(advised_length == length);
        CHECK(advised_kind == MADV_DONTNEED);
    };

    // A matching base is insufficient without a live fd association.
    no_call(map, size, 0, size, false);
    no_call(map, size, 0, size, true);
    g_model_fd = fd;
    for (bool force : {false, true}) {
        no_call(NULL, size, 0, size, force);
        no_call(map, 0, 0, 1, force);
        no_call(map, size + 1, 0, size, force);
        no_call(map, size, 0, 0, force);
        no_call(map, size, size, 1, force);
        no_call(map, size, size + 1, 1, force);
        no_call(map, size, UINT64_MAX, UINT64_MAX, force);
        g_model_fd_host_base = NULL;
        no_call(map, size, 0, size, force);
        g_model_fd_host_base = map;
    }

    void *anon = mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *cow = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    CHECK(anon != MAP_FAILED && cow != MAP_FAILED);
    memset(anon, 0xa5, size);
    memset(cow, 0x5a, size);
    for (bool force : {false, true}) {
        no_call(anon, size, 0, size, force);
        no_call(cow, size, 0, size, force);
        for (size_t i = 0; i < size; ++i) {
            CHECK(((unsigned char *)anon)[i] == 0xa5);
            CHECK(((unsigned char *)cow)[i] == 0x5a);
        }
    }

    // Synthetic addresses exercise arithmetic guards without dereferencing.
    g_model_fd_host_base = (const unsigned char *)map + 1;
    no_call(g_model_fd_host_base, size, 0, 1, true);
    uintptr_t last_page = UINTPTR_MAX - UINTPTR_MAX % page;
    g_model_fd_host_base = (const void *)last_page;
    no_call(g_model_fd_host_base, page, 0, 1, true);
    g_model_fd_host_base = map;

    CHECK(memcmp(map, expected.data(), size) == 0); // Prefault every file page.
    CHECK(setenv("DS4_CUDA_KEEP_MODEL_PAGES", "", 1) == 0);
    no_call(map, size, 0, size, false);
    long before = rss_kib(map);
    discard(0, size, true, 0, size); // Force overrides even an empty KEEP value.
    long after = rss_kib(map);
    if (before > 0 && after >= 0) {
        CHECK(after < before);
        printf("file RSS: %ld -> %ld KiB\n", before, after);
    } else {
        puts("RSS unavailable; syscall and refault checks still enforced");
    }
    CHECK(memcmp(map, expected.data(), size) == 0);
    CHECK(unsetenv("DS4_CUDA_KEEP_MODEL_PAGES") == 0);

    discard(page + 17, page + 31, false, page, page + 48);
    CHECK(memcmp(map, expected.data(), size) == 0);
    size_t final_start = size - size % page;
    discard(size - 7, UINT64_MAX, false, final_start, size - final_start);
    CHECK(memcmp(map, expected.data(), size) == 0);
    discard(0, UINT64_MAX, false, 0, size);
    CHECK(memcmp(map, expected.data(), size) == 0);

    CHECK(munmap(cow, size) == 0);
    CHECK(munmap(anon, size) == 0);
    CHECK(munmap(map, size) == 0);
    CHECK(fclose(file) == 0);
    puts("source-page guards, syscall arguments, and refault bytes passed");
}
"""


CALLER_PREAMBLE = r"""
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <map>
#include <vector>
#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #expr); exit(1); \
} } while (0)
struct Event { char kind; uint64_t offset, bytes; };
std::vector<Event> events;
void (*advice_observer)() = nullptr;
alignas(65536) char model[65536], device[65536];
int g_model_fd = 17;
const void *g_model_fd_host_base = model;
uint64_t g_model_file_size = sizeof(model), g_model_registered_size = 1;
int record_madvise(void *p, size_t n, int advice) {
    if (advice_observer) advice_observer();
    CHECK(advice == MADV_DONTNEED);
    events.push_back({'m', (uint64_t)((char *)p - model), n});
    return 0;
}
int record_fadvise(int fd, off_t off, off_t n, int advice) {
    if (advice_observer) advice_observer();
    CHECK(fd == g_model_fd && fd >= 0 && advice == POSIX_FADV_DONTNEED);
    events.push_back({'f', (uint64_t)off, (uint64_t)n});
    return 0;
}
#define madvise record_madvise
#define posix_fadvise record_fadvise
void env(const char *name, const char *value) {
    CHECK((value ? setenv(name, value, 1) : unsetenv(name)) == 0);
}
void expect(size_t i, char kind, uint64_t offset, uint64_t bytes) {
    CHECK(i < events.size());
    if (kind == 'm') {
        const long page = sysconf(_SC_PAGESIZE);
        CHECK(page > 0 && 65536 % page == 0);
        bytes += offset % page;
        offset -= offset % page;
    }
    if (events[i].kind != kind || events[i].offset != offset || events[i].bytes != bytes)
        fprintf(stderr, "event %zu: expected %c(%llu,%llu), got %c(%llu,%llu)\n",
                i, kind, (unsigned long long)offset, (unsigned long long)bytes,
                events[i].kind, (unsigned long long)events[i].offset,
                (unsigned long long)events[i].bytes);
    CHECK(events[i].kind == kind);
    CHECK(events[i].offset == offset && events[i].bytes == bytes);
}
"""

UPLOAD_STUBS = r"""
using cudaError_t = int;
const int cudaSuccess = 0, cudaMemcpyHostToDevice = 1;
uint64_t g_model_direct_align = 1;
char *g_stream_selected_stage[4], *g_model_stage[4];
uint64_t g_stream_selected_stage_bytes, g_model_stage_bytes;
int g_stream_selected_stage_event[4], g_model_stage_event[4];
int g_stream_selected_upload_stream, g_model_upload_stream;
uint64_t cuda_model_copy_chunk_bytes() { return 4096; }
int cuda_stream_selected_stage_pool_alloc(uint64_t n) {
    g_stream_selected_stage_bytes = n; return 1;
}
int cuda_model_stage_pool_alloc(uint64_t n) { g_model_stage_bytes = n; return 1; }
int cuda_model_stage_read(char *, uint64_t, uint64_t off, uint64_t n,
                          const char **payload) {
    CHECK(off + n <= sizeof(model));
    *payload = model + off; return 1;
}
int cudaMemcpy(char *dst, const char *src, size_t n, int) {
    events.push_back({'c', (uint64_t)(src - model), n});
    memcpy(dst, src, n); return 0;
}
int cudaMemcpyAsync(char *dst, const char *src, size_t n, int kind, int) {
    return cudaMemcpy(dst, src, n, kind);
}
int cudaEventSynchronize(int) { events.push_back({'w', 0, 0}); return 0; }
int cudaEventRecord(int, int) { events.push_back({'e', 0, 0}); return 0; }
int cudaStreamSynchronize(int) { events.push_back({'s', 0, 0}); return 0; }
int cudaGetLastError() { return 0; }
const char *cudaGetErrorString(int) { return "stub error"; }
int cuda_ok(int err, const char *) { return err == cudaSuccess; }
uint64_t g_model_range_bytes;
struct Range {
    const void *map; uint64_t offset, bytes; char *dev;
    void *a, *b; int c, d, e;
};
std::vector<Range> g_model_ranges;
std::map<uint64_t, size_t> g_model_range_by_offset;
uint64_t cuda_model_cache_limit_bytes() { return sizeof(device); }
const char *cuda_model_ptr(const void *p, uint64_t off) { return (const char *)p + off; }
char *cuda_model_arena_alloc(uint64_t, const char *) { return device; }
void cuda_model_load_progress_note(uint64_t) {}
"""

STREAM_MAIN = r"""
int main() {
    for (const char *drop : {(const char *)nullptr, "", "0", "1"}) {
        for (const char *keep : {(const char *)nullptr, "", "0", "1"}) {
            env("DS4_CUDA_DROP_PERSISTENT_SOURCE_PAGES", drop);
            env("DS4_CUDA_KEEP_MODEL_PAGES", keep);
            events.clear();
            CHECK(cuda_model_copy_to_device_streamed(
                device, model, sizeof(model), 4096, 5 * 4096 + 7, "selected"));
            size_t i = 0;
            for (unsigned chunk = 0; chunk < 6; ++chunk) {
                if (chunk >= 4) expect(i++, 'w', 0, 0);
                uint64_t off = 4096 + chunk * 4096, n = chunk == 5 ? 7 : 4096;
                expect(i++, 'c', off, n);
                expect(i++, 'e', 0, 0);
                if (!keep) expect(i++, 'f', off, n);
            }
            expect(i++, 's', 0, 0);
            CHECK(events.size() == i); // No madvise, including when DROP is set.
        }
    }
}
"""

RANGE_MAIN = r"""
int main() {
    for (const char *keep : {(const char *)nullptr, "", "0"}) {
        env("DS4_CUDA_KEEP_MODEL_PAGES", keep);
        env("DS4_CUDA_WEIGHT_CACHE_VERBOSE", nullptr);
        events.clear(); g_model_ranges.clear(); g_model_range_by_offset.clear();
        g_model_range_bytes = 0;
        CHECK(g_model_registered_size < 8192);
        CHECK(cuda_model_range_ptr_from_fd(model, 8192, 8192, "range") == device);
        size_t i = 0;
        for (uint64_t off : {8192u, 12288u}) {
            expect(i++, 'c', off, 4096); expect(i++, 'e', 0, 0);
            if (!keep) {
                expect(i++, 'm', off, 4096); expect(i++, 'f', off, 4096);
            }
        }
        expect(i++, 's', 0, 0); CHECK(events.size() == i);
        CHECK(g_model_ranges.size() == 1 && g_model_range_by_offset.at(8192) == 0);
        CHECK(g_model_range_bytes == 8192);
    }
}
"""

PERSISTENT_STUBS = r"""
struct ds4_gpu_stream_expert_table {
    const void *model_map; uint64_t model_size;
    uint64_t gate_offset, up_offset, down_offset, gate_expert_bytes, down_expert_bytes;
};
using cuda_stream_hot_key = uint32_t;
struct cuda_stream_hot_entry {
    cuda_stream_hot_key key; uint64_t hotness, last_used; bool valid;
};
struct cuda_stream_hot_cache {
    std::map<cuda_stream_hot_key, uint32_t> index;
    cuda_stream_hot_entry entries[1] = {};
    char *slab = device;
    uint64_t expert_bytes = 16384, hits = 0, misses = 0, resident_count = 0, evictions = 0;
    bool allocation_failed = false;
};
cuda_stream_hot_cache *active;
int copy_count, fail_copy;
cuda_stream_hot_key cuda_stream_hot_key_make(const ds4_gpu_stream_expert_table *, uint32_t e) {
    return e;
}
void cuda_stream_hot_cache_touch_locked(cuda_stream_hot_cache *, cuda_stream_hot_entry *e,
                                       uint32_t weight) {
    e->hotness += weight; events.push_back({'t', 0, 0});
}
uint32_t cuda_stream_hot_cache_victim_locked(cuda_stream_hot_cache *) { return 0; }
void cuda_stream_hot_cache_release_locked(cuda_stream_hot_cache *, bool) { CHECK(false); }
int cuda_model_copy_to_device_streamed(char *dst, const void *map, uint64_t size,
                                       uint64_t off, uint64_t n, const char *) {
    CHECK(map == model && size >= off + n);
    CHECK(!active->entries[0].valid && active->index.empty());
    CHECK(active->resident_count == 0 && active->misses == 0);
    CHECK(dst == device + (copy_count == 0 ? 0 : copy_count == 1 ? 4096 : 8192));
    events.push_back({'c', off, n});
    return ++copy_count != fail_copy;
}
"""

PERSISTENT_MAIN = r"""
int main() {
    for (const char *drop : {(const char *)nullptr, "", "0", "1"})
    for (const char *keep : {(const char *)nullptr, "", "0", "1"})
    for (int association = 0; association < 5; ++association)
    for (int failure = 0; failure <= 3; ++failure) {
        env("DS4_CUDA_DROP_PERSISTENT_SOURCE_PAGES", drop);
        env("DS4_CUDA_KEEP_MODEL_PAGES", keep);
        g_model_fd = association == 2 ? -1 : 17;
        g_model_fd_host_base = association == 1 ? model + 4096 :
                               association == 3 ? nullptr : model;
        g_model_file_size = association == 4 ? sizeof(model) - 1 : sizeof(model);
        ds4_gpu_stream_expert_table table = {model, sizeof(model), 0, 16384, 32768, 4096, 8192};
        cuda_stream_hot_cache cache;
        active = &cache; copy_count = 0; fail_copy = failure; events.clear();
        advice_observer = [] {
            CHECK(copy_count == 3);
            CHECK(!active->entries[0].valid && active->index.empty());
            CHECK(active->resident_count == 0 && active->misses == 0);
        };
        uint32_t slot = 99;
        CHECK(cuda_stream_hot_cache_load_locked(&cache, &table, 1, 7, &slot) == !failure);
        const uint64_t offsets[] = {4096, 20480, 40960}, sizes[] = {4096, 4096, 8192};
        size_t i = 0;
        int copies = failure ? failure : 3;
        for (int j = 0; j < copies; ++j) expect(i++, 'c', offsets[j], sizes[j]);
        CHECK(copy_count == copies);
        if (failure) {
            CHECK(events.size() == i && slot == 99);
            CHECK(!cache.entries[0].valid && cache.index.empty());
            CHECK(cache.resident_count == 0 && cache.misses == 0 && cache.hits == 0);
            continue;
        }
        if (drop && association == 0) {
            for (int j = 0; j < 3; ++j) expect(i++, 'm', offsets[j], sizes[j]);
            for (int j = 0; j < 3; ++j) expect(i++, 'f', offsets[j], sizes[j]);
        }
        expect(i++, 't', 0, 0); CHECK(events.size() == i);
        CHECK(slot == 0 && cache.entries[0].valid && cache.entries[0].key == 1);
        CHECK(cache.entries[0].hotness == 7 && cache.index.at(1) == 0);
        CHECK(cache.resident_count == 1 && cache.misses == 1 && cache.hits == 0);
        events.clear(); slot = 99;
        CHECK(cuda_stream_hot_cache_load_locked(&cache, &table, 1, 3, &slot));
        CHECK(slot == 0 && cache.hits == 1 && cache.misses == 1 && copy_count == 3);
        CHECK(cache.entries[0].hotness == 10 && cache.resident_count == 1);
        expect(0, 't', 0, 0); CHECK(events.size() == 1);
    }
}
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux mmap")
class CudaSourcePagesTest(unittest.TestCase):
    def caller_source(self, name, next_name):
        source = (Path(__file__).resolve().parents[1] / "ds4_cuda.cu").read_text()
        # Match a definition, never the forward declaration; require the exact
        # next definition as well as an unindented closing function brace.
        pattern = (
            r"^static [^\n]*\b" + re.escape(name)
            + r"\([^;{}]*\) \{\n.*?^\}\n"
            + r"(?=\s*^static [^\n]*\b" + re.escape(next_name) + r"\()"
        )
        matches = list(re.finditer(pattern, source, re.MULTILINE | re.DOTALL))
        self.assertEqual(len(matches), 1, f"production boundaries missing/ambiguous: {name}")
        block = matches[0].group()
        self.assertEqual(len(re.findall(r"^static .*\(", block, re.MULTILINE)), 1,
                         f"extraction crossed a function boundary: {name}")
        return block

    def run_caller(self, name, next_name, stubs, main):
        helpers = "".join(self.caller_source(a, b) for a, b in [
            ("cuda_model_discard_source_pages_impl", "cuda_model_discard_source_pages"),
            ("cuda_model_discard_source_pages", "cuda_model_drop_file_pages_impl"),
            ("cuda_model_drop_file_pages_impl", "cuda_model_drop_file_pages"),
            ("cuda_model_drop_file_pages", "cuda_round_down"),
        ])
        caller = self.caller_source(name, next_name)
        with tempfile.TemporaryDirectory(prefix="ds4-source-callers-") as directory:
            cpp = Path(directory) / "test.cpp"
            binary = Path(directory) / "test"
            cpp.write_text(CALLER_PREAMBLE + helpers + stubs + caller + main)
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++"))
                + ["-std=c++11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                   str(cpp), "-o", str(binary)],
                check=True, timeout=60,
            )
            subprocess.run([str(binary)], check=True, timeout=30)

    def test_actual_selected_streaming_upload(self):
        self.run_caller("cuda_model_copy_to_device_streamed", "cuda_stream_hot_cache_enabled",
                        UPLOAD_STUBS, STREAM_MAIN)

    def test_actual_persistent_cache_load(self):
        self.run_caller("cuda_stream_hot_cache_load_locked", "cuda_stream_hot_cache_materialize",
                        PERSISTENT_STUBS, PERSISTENT_MAIN)

    def test_actual_fd_range_upload(self):
        self.run_caller("cuda_model_range_ptr_from_fd", "cuda_model_copy_chunked",
                        UPLOAD_STUBS, RANGE_MAIN)

    def test_actual_discard_function(self):
        source = (Path(__file__).resolve().parents[1] / "ds4_cuda.cu").read_text()
        self.assertRegex(source, r"(?m)^\s*#\s*include\s*<sys/mman\.h>")
        match = re.search(
            r"^static void cuda_model_discard_source_pages_impl\(.*?"
            r"(?=^static void cuda_model_discard_source_pages\()",
            source,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match, "production discard function boundaries missing")
        with tempfile.TemporaryDirectory(prefix="ds4-source-pages-") as directory:
            cpp = Path(directory) / "test.cpp"
            binary = Path(directory) / "test"
            cpp.write_text(PREAMBLE + match.group() + HARNESS)
            subprocess.run(
                shlex.split(os.environ.get("CXX", "c++"))
                + ["-std=c++11", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
                check=True,
                timeout=60,
            )
            subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
