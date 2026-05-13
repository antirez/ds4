#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct mapped_file {
    int fd = -1;
    const uint8_t *data = nullptr;
    uint64_t size = 0;
};

struct tensor_span {
    uint64_t off = 0;
    uint64_t end = 0;
};

struct owned_range {
    std::string model_id;
    uint64_t model_size = 0;
    uint64_t off = 0;
    uint64_t bytes = 0;
    void *dev = nullptr;
    cudaIpcMemHandle_t handle{};
};

static volatile sig_atomic_t g_stop;

static void on_signal(int) {
    g_stop = 1;
}

static bool read_u32(const mapped_file &m, uint64_t &pos, uint32_t &out) {
    if (pos > m.size || m.size - pos < 4) return false;
    memcpy(&out, m.data + pos, 4);
    pos += 4;
    return true;
}

static bool read_u64(const mapped_file &m, uint64_t &pos, uint64_t &out) {
    if (pos > m.size || m.size - pos < 8) return false;
    memcpy(&out, m.data + pos, 8);
    pos += 8;
    return true;
}

static bool skip_bytes(const mapped_file &m, uint64_t &pos, uint64_t n) {
    if (pos > m.size || n > m.size - pos) return false;
    pos += n;
    return true;
}

static bool read_string(const mapped_file &m, uint64_t &pos, std::string &out) {
    uint64_t len = 0;
    if (!read_u64(m, pos, len) || len > m.size || pos > m.size || len > m.size - pos) return false;
    out.assign((const char *)m.data + pos, (size_t)len);
    pos += len;
    return true;
}

static uint64_t align_up(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    const uint64_t r = v % a;
    return r ? v + (a - r) : v;
}

static uint64_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case 0: return 1;
    case 1: return 1;
    case 2: return 2;
    case 3: return 2;
    case 4: return 4;
    case 5: return 4;
    case 6: return 4;
    case 7: return 1;
    case 10: return 8;
    case 11: return 8;
    case 12: return 8;
    default: return 0;
    }
}

static bool skip_metadata_value(const mapped_file &m, uint64_t &pos, uint32_t type);

static bool skip_array(const mapped_file &m, uint64_t &pos) {
    uint32_t elem_type = 0;
    uint64_t len = 0;
    if (!read_u32(m, pos, elem_type) || !read_u64(m, pos, len)) return false;
    if (elem_type == 8) {
        for (uint64_t i = 0; i < len; i++) {
            std::string tmp;
            if (!read_string(m, pos, tmp)) return false;
        }
        return true;
    }
    const uint64_t elem_size = gguf_scalar_size(elem_type);
    if (elem_size == 0 || len > UINT64_MAX / elem_size) return false;
    return skip_bytes(m, pos, len * elem_size);
}

static bool skip_metadata_value(const mapped_file &m, uint64_t &pos, uint32_t type) {
    if (type == 8) {
        std::string tmp;
        return read_string(m, pos, tmp);
    }
    if (type == 9) return skip_array(m, pos);
    const uint64_t n = gguf_scalar_size(type);
    return n != 0 && skip_bytes(m, pos, n);
}

static bool tensor_type_info(uint32_t type, uint64_t &block_elems, uint64_t &block_bytes) {
    switch (type) {
    case 0: block_elems = 1; block_bytes = 4; return true;
    case 1: block_elems = 1; block_bytes = 2; return true;
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 3: block_elems = 32; block_bytes = 20; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 7: block_elems = 32; block_bytes = 24; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 9: block_elems = 32; block_bytes = 40; return true;
    case 10: block_elems = 256; block_bytes = 84; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 15: block_elems = 256; block_bytes = 292; return true;
    case 16: block_elems = 256; block_bytes = 66; return true;
    case 17: block_elems = 256; block_bytes = 74; return true;
    case 18: block_elems = 256; block_bytes = 98; return true;
    case 19: block_elems = 256; block_bytes = 110; return true;
    case 20: block_elems = 256; block_bytes = 50; return true;
    case 21: block_elems = 256; block_bytes = 110; return true;
    case 22: block_elems = 256; block_bytes = 82; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 24: block_elems = 1; block_bytes = 1; return true;
    case 25: block_elems = 1; block_bytes = 2; return true;
    case 26: block_elems = 1; block_bytes = 4; return true;
    case 27: block_elems = 1; block_bytes = 8; return true;
    case 28: block_elems = 1; block_bytes = 8; return true;
    case 29: block_elems = 256; block_bytes = 56; return true;
    case 30: block_elems = 1; block_bytes = 2; return true;
    default: return false;
    }
}

static bool map_file(const char *path, mapped_file &m) {
    m.fd = open(path, O_RDONLY);
    if (m.fd < 0) {
        fprintf(stderr, "ds4_weight_server: open failed %s: %s\n", path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(m.fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "ds4_weight_server: stat failed %s\n", path);
        close(m.fd);
        m.fd = -1;
        return false;
    }
    m.size = (uint64_t)st.st_size;
    void *p = mmap(NULL, (size_t)m.size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "ds4_weight_server: mmap failed %s: %s\n", path, strerror(errno));
        close(m.fd);
        m.fd = -1;
        return false;
    }
    m.data = (const uint8_t *)p;
    return true;
}

static void unmap_file(mapped_file &m) {
    if (m.data) munmap((void *)m.data, (size_t)m.size);
    if (m.fd >= 0) close(m.fd);
    m = {};
}

static bool collect_tensor_spans(const mapped_file &m, std::vector<tensor_span> &spans) {
    uint64_t pos = 0;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!read_u32(m, pos, magic) || magic != 0x46554747u ||
        !read_u32(m, pos, version) || version != 3 ||
        !read_u64(m, pos, n_tensors) ||
        !read_u64(m, pos, n_kv)) {
        fprintf(stderr, "ds4_weight_server: unsupported or invalid GGUF\n");
        return false;
    }

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key;
        uint32_t type = 0;
        uint64_t value_pos = 0;
        if (!read_string(m, pos, key) || !read_u32(m, pos, type)) return false;
        value_pos = pos;
        if (!skip_metadata_value(m, pos, type)) return false;
        if (key == "general.alignment" && type == 4) {
            uint32_t v = 0;
            memcpy(&v, m.data + value_pos, 4);
            if (v > 0) alignment = v;
        }
    }

    struct tensor_info {
        uint64_t rel = 0;
        uint64_t bytes = 0;
    };
    std::vector<tensor_info> tensors;
    tensors.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        std::string name;
        uint32_t ndim = 0;
        if (!read_string(m, pos, name) || !read_u32(m, pos, ndim) || ndim > 8) return false;
        uint64_t elems = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim = 0;
            if (!read_u64(m, pos, dim)) return false;
            if (dim != 0 && elems > UINT64_MAX / dim) return false;
            elems *= dim;
        }
        uint32_t type = 0;
        uint64_t rel = 0;
        if (!read_u32(m, pos, type) || !read_u64(m, pos, rel)) return false;
        uint64_t block_elems = 0, block_bytes = 0;
        if (!tensor_type_info(type, block_elems, block_bytes)) {
            fprintf(stderr, "ds4_weight_server: unsupported tensor type %u for %s\n", type, name.c_str());
            return false;
        }
        const uint64_t blocks = (elems + block_elems - 1u) / block_elems;
        if (blocks > UINT64_MAX / block_bytes) return false;
        tensors.push_back({rel, blocks * block_bytes});
    }

    const uint64_t tensor_data_pos = align_up(pos, alignment);
    spans.clear();
    spans.reserve(tensors.size());
    for (const tensor_info &t : tensors) {
        if (t.rel > UINT64_MAX - tensor_data_pos) return false;
        const uint64_t off = tensor_data_pos + t.rel;
        if (off > m.size || t.bytes > m.size - off) return false;
        if (t.bytes != 0) spans.push_back({off, off + t.bytes});
    }
    return true;
}

static uint64_t parse_mib(const char *s, uint64_t fallback) {
    if (!s || !s[0]) return fallback;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s || v == 0) return fallback;
    return (uint64_t)v * 1048576ull;
}

static uint64_t parse_gib(const char *s, uint64_t fallback) {
    if (!s || !s[0]) return fallback;
    char *end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s) return fallback;
    return (uint64_t)v * 1073741824ull;
}

static void build_range_plan(std::vector<tensor_span> &spans, uint64_t span_bytes,
                             std::vector<tensor_span> &ranges) {
    std::sort(spans.begin(), spans.end(), [](const tensor_span &a, const tensor_span &b) {
        if (a.off != b.off) return a.off < b.off;
        return a.end < b.end;
    });

    ranges.clear();
    for (size_t i = 0; i < spans.size();) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < spans.size() && spans[i].off <= end + 65536u && spans[i].end - off <= span_bytes) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        while (off < end) {
            uint64_t chunk_end = end;
            if (chunk_end - off > span_bytes) chunk_end = off + span_bytes;
            ranges.push_back({off, chunk_end});
            off = chunk_end;
        }
    }
}

static uint64_t range_plan_bytes(const std::vector<tensor_span> &ranges) {
    uint64_t total = 0;
    for (const tensor_span &r : ranges) {
        if (r.end >= r.off && UINT64_MAX - total >= r.end - r.off) total += r.end - r.off;
    }
    return total;
}

static bool inspect_model_plan(const char *id, const char *path, uint64_t span_bytes,
                               uint64_t *model_size_out, uint64_t *bytes_out,
                               uint64_t *ranges_out) {
    mapped_file m;
    if (!map_file(path, m)) return false;
    std::vector<tensor_span> spans;
    if (!collect_tensor_spans(m, spans)) {
        unmap_file(m);
        return false;
    }
    std::vector<tensor_span> ranges;
    build_range_plan(spans, span_bytes, ranges);
    const uint64_t planned = range_plan_bytes(ranges);
    fprintf(stderr,
            "ds4_weight_server: %s plan model=%.2f GiB raw_tensor_ranges=%.2f GiB ranges=%zu\n",
            id,
            (double)m.size / 1073741824.0,
            (double)planned / 1073741824.0,
            ranges.size());
    if (model_size_out) *model_size_out = m.size;
    if (bytes_out) *bytes_out = planned;
    if (ranges_out) *ranges_out = (uint64_t)ranges.size();
    unmap_file(m);
    return true;
}

static bool cuda_memory_preflight(const char *what, uint64_t planned_bytes, uint64_t reserve_bytes) {
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: cudaMemGetInfo failed for %s: %s\n",
                what, cudaGetErrorString(err));
        return false;
    }
    fprintf(stderr,
            "ds4_weight_server: memory preflight %s need=%.2f GiB reserve=%.2f GiB free=%.2f GiB total=%.2f GiB\n",
            what,
            (double)planned_bytes / 1073741824.0,
            (double)reserve_bytes / 1073741824.0,
            (double)free_b / 1073741824.0,
            (double)total_b / 1073741824.0);
    if (reserve_bytes > 0 &&
        planned_bytes <= UINT64_MAX - reserve_bytes &&
        (uint64_t)free_b < planned_bytes + reserve_bytes) {
        fprintf(stderr,
                "ds4_weight_server: refusing upload; not enough free CUDA memory for %s plus reserve. "
                "Stop other model processes or lower --reserve-gb explicitly.\n",
                what);
        return false;
    }
    return true;
}

static bool pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t req = bytes - done > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        done += (uint64_t)n;
    }
    return true;
}

struct upload_stage_pool {
    cudaStream_t stream = nullptr;
    void *stage[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t event[4] = {nullptr, nullptr, nullptr, nullptr};
    uint64_t bytes = 0;
};

static void upload_stage_pool_destroy(upload_stage_pool &pool) {
    for (int i = 0; i < 4; i++) {
        if (pool.event[i]) cudaEventDestroy(pool.event[i]);
        if (pool.stage[i]) cudaFreeHost(pool.stage[i]);
        pool.event[i] = nullptr;
        pool.stage[i] = nullptr;
    }
    if (pool.stream) cudaStreamDestroy(pool.stream);
    pool.stream = nullptr;
    pool.bytes = 0;
}

static bool upload_stage_pool_init(upload_stage_pool &pool, uint64_t bytes) {
    if (pool.bytes >= bytes) return true;
    upload_stage_pool_destroy(pool);
    cudaError_t err = cudaStreamCreateWithFlags(&pool.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: upload stream create failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    for (int i = 0; i < 4; i++) {
        err = cudaMallocHost(&pool.stage[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: pinned staging alloc failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        err = cudaEventCreateWithFlags(&pool.event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: staging event create failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }
    pool.bytes = bytes;
    return true;
}

static bool upload_range_chunked(int fd, uint64_t file_off, void *dev, uint64_t bytes,
                                 upload_stage_pool &pool, uint64_t chunk_bytes) {
    if (!upload_stage_pool_init(pool, chunk_bytes)) return false;
    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = bytes - copied < chunk_bytes ? bytes - copied : chunk_bytes;
        const int bi = (int)(chunk_idx % 4u);
        cudaError_t err = cudaSuccess;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(pool.event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4_weight_server: staging wait failed: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        if (!pread_full(fd, pool.stage[bi], n, file_off + copied)) {
            fprintf(stderr, "ds4_weight_server: model read failed at off=%llu: %s\n",
                    (unsigned long long)(file_off + copied), strerror(errno));
            return false;
        }
        err = cudaMemcpyAsync((char *)dev + copied, pool.stage[bi], (size_t)n,
                              cudaMemcpyHostToDevice, pool.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: async upload failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        err = cudaEventRecord(pool.event[bi], pool.stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: staging record failed: %s\n", cudaGetErrorString(err));
            return false;
        }
#if defined(POSIX_FADV_DONTNEED)
        (void)posix_fadvise(fd, (off_t)(file_off + copied), (off_t)n, POSIX_FADV_DONTNEED);
#endif
        copied += n;
        chunk_idx++;
    }
    cudaError_t err = cudaStreamSynchronize(pool.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: upload sync failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

static void hex_encode(const void *data, size_t bytes, std::string &out) {
    static const char *hex = "0123456789abcdef";
    const uint8_t *p = (const uint8_t *)data;
    out.resize(bytes * 2u);
    for (size_t i = 0; i < bytes; i++) {
        out[2u * i] = hex[p[i] >> 4];
        out[2u * i + 1u] = hex[p[i] & 15u];
    }
}

static bool upload_model(const char *id, const char *path, uint64_t span_bytes,
                         uint64_t copy_chunk_bytes, std::vector<owned_range> &ranges) {
    mapped_file m;
    if (!map_file(path, m)) return false;
    std::vector<tensor_span> spans;
    if (!collect_tensor_spans(m, spans)) {
        unmap_file(m);
        return false;
    }
    std::vector<tensor_span> plan;
    build_range_plan(spans, span_bytes, plan);
    if (m.data) {
        munmap((void *)m.data, (size_t)m.size);
        m.data = nullptr;
    }

    upload_stage_pool pool;
    uint64_t uploaded = 0;
    uint64_t count = 0;
    for (const tensor_span &planned_range : plan) {
        const uint64_t off = planned_range.off;
        const uint64_t bytes = planned_range.end - planned_range.off;
        void *dev = nullptr;
        cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaMalloc failed for %s %.2f MiB: %s\n",
                    id, (double)bytes / 1048576.0, cudaGetErrorString(err));
            unmap_file(m);
            return false;
        }
        if (!upload_range_chunked(m.fd, off, dev, bytes, pool, copy_chunk_bytes)) {
            cudaFree(dev);
            upload_stage_pool_destroy(pool);
            unmap_file(m);
            return false;
        }
        cudaIpcMemHandle_t handle;
        err = cudaIpcGetMemHandle(&handle, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_weight_server: cudaIpcGetMemHandle failed for %s: %s\n",
                    id, cudaGetErrorString(err));
            cudaFree(dev);
            upload_stage_pool_destroy(pool);
            unmap_file(m);
            return false;
        }
        owned_range r;
        r.model_id = id;
        r.model_size = m.size;
        r.off = off;
        r.bytes = bytes;
        r.dev = dev;
        r.handle = handle;
        ranges.push_back(r);
        uploaded += bytes;
        count++;
        if (count == 1 || uploaded / (8ull * 1073741824ull) != (uploaded - bytes) / (8ull * 1073741824ull)) {
            fprintf(stderr, "ds4_weight_server: %s uploaded %.2f GiB\r", id, (double)uploaded / 1073741824.0);
            fflush(stderr);
        }
    }
    upload_stage_pool_destroy(pool);
    fprintf(stderr, "ds4_weight_server: %s uploaded %.2f GiB across %llu ranges\n",
            id, (double)uploaded / 1073741824.0, (unsigned long long)count);
    unmap_file(m);
    return true;
}

static bool write_manifest(const char *path, const std::vector<owned_range> &ranges) {
    std::string tmp = std::string(path) + ".tmp";
    FILE *fp = fopen(tmp.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ds4_weight_server: manifest open failed %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    fprintf(fp, "DS4_WEIGHT_SERVER_IPC_V1\n");
    fprintf(fp, "# range <model-id> <model-size> <offset> <bytes> <cuda-ipc-handle-hex>\n");
    for (const owned_range &r : ranges) {
        std::string hex;
        hex_encode(&r.handle, sizeof(r.handle), hex);
        fprintf(fp, "range %s %llu %llu %llu %s\n",
                r.model_id.c_str(),
                (unsigned long long)r.model_size,
                (unsigned long long)r.off,
                (unsigned long long)r.bytes,
                hex.c_str());
    }
    if (fclose(fp) != 0) return false;
    if (rename(tmp.c_str(), path) != 0) {
        fprintf(stderr, "ds4_weight_server: manifest rename failed: %s\n", strerror(errno));
        return false;
    }
    return true;
}

static void usage(FILE *fp) {
    fprintf(fp,
            "Usage: ds4_weight_server --base FILE [--mtp FILE] --manifest FILE [options]\n"
            "\n"
            "Options:\n"
            "  --device N        CUDA device ordinal. Default: 0\n"
            "  --span-mb N       Maximum exported raw tensor span. Default: 1024\n"
            "  --copy-chunk-mb N Pinned staged upload chunk. Default: 256\n"
            "  --reserve-gb N    Free CUDA memory to keep unused. Default: 32\n"
            "  --dry-run         Parse GGUFs, print upload plan, and exit before allocation\n");
}

int main(int argc, char **argv) {
    const char *base = nullptr;
    const char *mtp = nullptr;
    const char *manifest = nullptr;
    int device = 0;
    uint64_t span_bytes = 1024ull * 1048576ull;
    uint64_t copy_chunk_bytes = 256ull * 1048576ull;
    uint64_t reserve_bytes = 32ull * 1073741824ull;
    bool dry_run = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!strcmp(argv[i], "--mtp") && i + 1 < argc) mtp = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--span-mb") && i + 1 < argc) span_bytes = parse_mib(argv[++i], span_bytes);
        else if (!strcmp(argv[i], "--copy-chunk-mb") && i + 1 < argc) copy_chunk_bytes = parse_mib(argv[++i], copy_chunk_bytes);
        else if (!strcmp(argv[i], "--reserve-gb") && i + 1 < argc) reserve_bytes = parse_gib(argv[++i], reserve_bytes);
        else if (!strcmp(argv[i], "--dry-run")) dry_run = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "ds4_weight_server: unknown or incomplete option: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }
    if (!base || (!manifest && !dry_run)) {
        usage(stderr);
        return 2;
    }
    if (span_bytes < 64ull * 1048576ull) span_bytes = 64ull * 1048576ull;
    if (span_bytes > 4096ull * 1048576ull) span_bytes = 4096ull * 1048576ull;
    if (copy_chunk_bytes < 16ull * 1048576ull) copy_chunk_bytes = 16ull * 1048576ull;
    if (copy_chunk_bytes > 1024ull * 1048576ull) copy_chunk_bytes = 1024ull * 1048576ull;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_weight_server: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    uint64_t total_upload_bytes = 0;
    uint64_t base_bytes = 0;
    if (!inspect_model_plan("base", base, span_bytes, nullptr, &base_bytes, nullptr)) return 1;
    total_upload_bytes += base_bytes;
    if (mtp) {
        uint64_t mtp_bytes = 0;
        if (!inspect_model_plan("mtp", mtp, span_bytes, nullptr, &mtp_bytes, nullptr)) return 1;
        if (UINT64_MAX - total_upload_bytes < mtp_bytes) {
            fprintf(stderr, "ds4_weight_server: upload plan size overflow\n");
            return 1;
        }
        total_upload_bytes += mtp_bytes;
    }
    if (!cuda_memory_preflight("full upload plan", total_upload_bytes, reserve_bytes)) return 1;
    if (dry_run) {
        fprintf(stderr, "ds4_weight_server: dry-run complete; no allocations or manifest were created\n");
        return 0;
    }

    std::vector<owned_range> ranges;
    if (!upload_model("base", base, span_bytes, copy_chunk_bytes, ranges)) return 1;
    if (mtp && !upload_model("mtp", mtp, span_bytes, copy_chunk_bytes, ranges)) return 1;
    if (!write_manifest(manifest, ranges)) return 1;

    fprintf(stderr,
            "ds4_weight_server: ready manifest=%s ranges=%zu. Keep this process alive while workers run.\n",
            manifest,
            ranges.size());
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    while (!g_stop) sleep(1);

    fprintf(stderr, "ds4_weight_server: shutting down\n");
    for (owned_range &r : ranges) {
        if (r.dev) cudaFree(r.dev);
    }
    return 0;
}
