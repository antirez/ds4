/* ds4_repack: shared aligned-artifact layout library.  See ds4_repack.h for
 * the API surface and tools/ds4_weight_server.cu for the original home of
 * this code (extracted 2026-07 so self-load boots build the same artifacts
 * in-process).  Every byte-movement and kernel here must stay bit-identical
 * between the weight-server and in-process producers — the FNV repack-hash
 * lines are the gate. */
#include "ds4_repack.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double repack_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static uint64_t repack_align_up(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    const uint64_t r = v % a;
    return r ? v + (a - r) : v;
}

static uint64_t repack_align_down(uint64_t v, uint64_t a) {
    if (a <= 1) return v;
    return (v / a) * a;
}

static void *repack_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    const uintptr_t p = (uintptr_t)ptr;
    const uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

/* ---- file mapping + GGUF catalog ---------------------------------------- */

bool ds4_repack_map_file(const char *log_prefix, const char *path, ds4_repack_file &m) {
    m.fd = open(path, O_RDONLY);
    if (m.fd < 0) {
        fprintf(stderr, "%s: open failed %s: %s\n", log_prefix, path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(m.fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "%s: stat failed %s\n", log_prefix, path);
        close(m.fd);
        m.fd = -1;
        return false;
    }
    m.size = (uint64_t)st.st_size;
    if (st.st_blksize > 1) m.direct_align = (uint64_t)st.st_blksize;
#if defined(__linux__) && defined(O_DIRECT)
    if (getenv("DS4_CUDA_NO_DIRECT_IO") == nullptr) {
        char proc_path[64];
        snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", m.fd);
        int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
        if (direct_fd >= 0) {
            m.direct_fd = direct_fd;
            if (m.direct_align < 512) m.direct_align = 512;
            fprintf(stderr, "%s: direct I/O enabled for %s align=%llu\n",
                    log_prefix,
                    path,
                    (unsigned long long)m.direct_align);
        }
    }
#endif
    void *p = mmap(NULL, (size_t)m.size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "%s: mmap failed %s: %s\n", log_prefix, path, strerror(errno));
        close(m.fd);
        m.fd = -1;
        return false;
    }
    m.data = (const uint8_t *)p;
    return true;
}

void ds4_repack_unmap_file(ds4_repack_file &m) {
    if (m.data) munmap((void *)m.data, (size_t)m.size);
    if (m.direct_fd >= 0) close(m.direct_fd);
    if (m.fd >= 0) close(m.fd);
    m = {};
}

static bool read_u32(const ds4_repack_file &m, uint64_t &pos, uint32_t &out) {
    if (pos > m.size || m.size - pos < 4) return false;
    memcpy(&out, m.data + pos, 4);
    pos += 4;
    return true;
}

static bool read_u64(const ds4_repack_file &m, uint64_t &pos, uint64_t &out) {
    if (pos > m.size || m.size - pos < 8) return false;
    memcpy(&out, m.data + pos, 8);
    pos += 8;
    return true;
}

static bool skip_bytes(const ds4_repack_file &m, uint64_t &pos, uint64_t n) {
    if (pos > m.size || n > m.size - pos) return false;
    pos += n;
    return true;
}

static bool read_string(const ds4_repack_file &m, uint64_t &pos, std::string &out) {
    uint64_t len = 0;
    if (!read_u64(m, pos, len) || len > m.size || pos > m.size || len > m.size - pos) return false;
    out.assign((const char *)m.data + pos, (size_t)len);
    pos += len;
    return true;
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

static bool skip_metadata_value(const ds4_repack_file &m, uint64_t &pos, uint32_t type);

static bool skip_array(const ds4_repack_file &m, uint64_t &pos) {
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

static bool skip_metadata_value(const ds4_repack_file &m, uint64_t &pos, uint32_t type) {
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

bool ds4_repack_collect_catalog(const char *log_prefix,
                                const ds4_repack_file &m,
                                std::vector<ds4_repack_span> *spans,
                                std::vector<ds4_repack_tensor> *records) {
    uint64_t pos = 0;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!read_u32(m, pos, magic) || magic != 0x46554747u ||
        !read_u32(m, pos, version) || version != 3 ||
        !read_u64(m, pos, n_tensors) ||
        !read_u64(m, pos, n_kv)) {
        fprintf(stderr, "%s: unsupported or invalid GGUF\n", log_prefix);
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

    std::vector<ds4_repack_tensor> tensors;
    tensors.reserve((size_t)n_tensors);
    for (uint64_t i = 0; i < n_tensors; i++) {
        ds4_repack_tensor t;
        uint32_t ndim = 0;
        if (!read_string(m, pos, t.name) || !read_u32(m, pos, ndim) || ndim > 8) return false;
        t.ndim = ndim;
        uint64_t elems = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint64_t dim = 0;
            if (!read_u64(m, pos, dim)) return false;
            if (dim != 0 && elems > UINT64_MAX / dim) return false;
            t.dims[d] = dim;
            elems *= dim;
        }
        t.elements = elems;
        uint32_t type = 0;
        uint64_t rel = 0;
        if (!read_u32(m, pos, type) || !read_u64(m, pos, rel)) return false;
        t.type = type;
        uint64_t block_elems = 0, block_bytes = 0;
        if (!tensor_type_info(type, block_elems, block_bytes)) {
            fprintf(stderr, "%s: unsupported tensor type %u for %s\n",
                    log_prefix, type, t.name.c_str());
            return false;
        }
        const uint64_t blocks = (elems + block_elems - 1u) / block_elems;
        if (blocks > UINT64_MAX / block_bytes) return false;
        t.off = rel;
        t.bytes = blocks * block_bytes;
        tensors.push_back(t);
    }

    const uint64_t tensor_data_pos = repack_align_up(pos, alignment);
    if (spans) {
        spans->clear();
        spans->reserve(tensors.size());
    }
    if (records) records->clear();
    if (records) records->reserve(tensors.size());
    for (ds4_repack_tensor &t : tensors) {
        if (t.off > UINT64_MAX - tensor_data_pos) return false;
        const uint64_t off = tensor_data_pos + t.off;
        if (off > m.size || t.bytes > m.size - off) return false;
        t.off = off;
        if (spans && t.bytes != 0) spans->push_back({off, off + t.bytes});
        if (records) records->push_back(t);
    }
    return true;
}

/* ---- staged file reads --------------------------------------------------- */

bool ds4_repack_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
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

bool ds4_repack_read_stage(const ds4_repack_file &m, void *stage, uint64_t stage_bytes,
                           uint64_t file_off, uint64_t bytes, const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (m.direct_fd >= 0 && m.direct_align > 1 && m.size != 0) {
        const uint64_t aligned_off = repack_align_down(file_off, m.direct_align);
        const uint64_t delta = file_off - aligned_off;
        const uint64_t read_size = repack_align_up(delta + bytes, m.direct_align);
        if (aligned_off <= m.size && read_size <= stage_bytes && read_size <= m.size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (ds4_repack_pread_full(m.direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return true;
            }
            const int direct_errno = errno;
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return ds4_repack_pread_full(m.fd, stage, bytes, file_off);
}

/* ---- aligned repack kernels ---------------------------------------------- */

/* One thread per (block, uint2 pair); lane p==0 additionally splits out the
 * block scale.  Source raw block_iq2_xxs is 66 bytes = [half d][64B codes]. */
__global__ static void repack_iq2_xxs_aligned_kernel(
        __half *dq,
        uint2 *qs,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 8ull) return;
    const uint64_t blk = i >> 3;
    const uint32_t p = (uint32_t)(i & 7u);
    const unsigned char *src = raw + blk * 66ull;
    if (p == 0u) {
        uint16_t h;
        memcpy(&h, src, 2u);
        dq[blk] = __ushort_as_half(h);
    }
    uint2 v;
    memcpy(&v, src + 2u + (uint64_t)p * 8u, 8u);
    qs[blk * 8ull + p] = v;
}

/* Row-pair-SoA Q2_K repack.  Source raw block_q2_K is 84 bytes =
 * [u8 scales[16]][u8 qs[64]][half d][half dmin]; the pair block for rows
 * (2p, 2p+1) at column-block b interleaves the two rows so the decode twin
 * does one 8B qs load / 16B scales load / 8B dm load per lane-iteration:
 *   dm2[pblk]         = {row0 dm word, row1 dm word}
 *   sc4[pblk*2 + h]   = {row0 scales[8h..8h+3], row0 [8h+4..8h+7],
 *                        row1 [8h..8h+3], row1 [8h+4..8h+7]}
 *   qs2[pblk*16 + i]  = {row0 qs int[i], row1 qs int[i]}
 * One thread per (raw block, qs int); p < 4 additionally writes a scales
 * word, p == 0 the dm word.  The caller chunks on whole row pairs, so g0
 * (the absolute index of the chunk's first raw block) is always row-pair
 * aligned and the pair index math stays global. */
__global__ static void repack_q2_k_aligned_kernel(
        uint32_t *dm2,          // section base, uint32 view (2 words / pair blk)
        uint32_t *sc4,          // section base, uint32 view (8 words / pair blk)
        uint32_t *qs2,          // section base, uint32 view (32 words / pair blk)
        const unsigned char *raw,
        uint64_t g0,            // absolute raw-block index of raw[0]
        uint64_t cblk,          // raw blocks in this chunk
        uint32_t nb_row,        // blocks per row = K/256
        uint32_t nrows) {       // rows per expert = M
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cblk * 16ull) return;
    const uint64_t j = i >> 4;
    const uint32_t p = (uint32_t)(i & 15u);
    const uint64_t g = g0 + j;
    const uint32_t b = (uint32_t)(g % nb_row);
    const uint32_t r = (uint32_t)((g / nb_row) % nrows);
    const uint64_t e = g / ((uint64_t)nb_row * nrows);
    const uint64_t pblk = ((uint64_t)e * (nrows/2u) + r/2u) * nb_row + b;
    const uint32_t parity = r & 1u;
    const unsigned char *src = raw + j * 84ull;
    uint32_t w;
    memcpy(&w, src + 16u + (uint64_t)p * 4u, 4u);
    qs2[pblk * 32ull + (uint64_t)p * 2ull + parity] = w;
    if (p < 4u) {
        memcpy(&w, src + (uint64_t)p * 4u, 4u);
        /* scales word p covers bytes [4p, 4p+4) = window h = p>>1, half p&1 */
        sc4[pblk * 8ull + (uint64_t)(p >> 1) * 4ull + parity * 2ull + (p & 1u)] = w;
    }
    if (p == 0u) {
        memcpy(&w, src + 80u, 4u);
        dm2[pblk * 2ull + parity] = w;
    }
}

/* One thread per (block, 16B half of the 32B code payload); p==0 additionally
 * splits out the block scale.  Source raw block_q8_0 is 34 bytes =
 * [half d][32 x int8 codes]. */
__global__ static void repack_q8_0_aligned_kernel(
        __half *dq,
        unsigned char *qs,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 2ull) return;
    const uint64_t blk = i >> 1;
    const uint32_t p = (uint32_t)(i & 1u);
    const unsigned char *src = raw + blk * 34ull;
    if (p == 0u) {
        uint16_t h;
        memcpy(&h, src, 2u);
        dq[blk] = __ushort_as_half(h);
    }
    memcpy(qs + blk * 32ull + p * 16ull, src + 2u + p * 16ull, 16u);
}

/* One thread per dequantized element of the staged chunk.  Math is the
 * engine's lazy dequant_q8_0_to_f16_kernel (ds4_cuda.cu) verbatim -- the
 * first-touch cache and this prebuild must stay bit-identical.  The full
 * tensor's f16 element order is exactly (raw block index)*32 + code index,
 * so chunking at 34-byte block granularity preserves every element's
 * inputs. */
__global__ static void dequant_q8_0_f16_colmajor_kernel(
        __half *out,
        const unsigned char *raw,
        uint64_t nblk) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nblk * 32ull) return;
    const uint64_t b = gid >> 5;
    const uint64_t j = gid & 31u;
    const unsigned char *blk = raw + b * 34ull;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

/* ---- candidate predicates ------------------------------------------------ */

/* --repack-iq2-aligned candidates: routed-expert gate/up stacks in IQ2_XXS.
 * dims[0] % 1024: the aligned decode kernel covers 4 blocks per warp pass. */
bool ds4_repack_iq2_candidate(const ds4_repack_tensor &t) {
    if (t.type != 16u || t.ndim != 3u) return false; /* GGML_TYPE_IQ2_XXS */
    if (t.dims[0] == 0 || t.dims[1] == 0 || t.dims[2] == 0 || t.dims[2] > UINT32_MAX) return false;
    if (t.dims[0] % 1024u != 0) return false;
    if (t.bytes == 0 || t.bytes % 66u != 0) return false;
    static const char gate_sfx[] = ".ffn_gate_exps.weight";
    static const char up_sfx[] = ".ffn_up_exps.weight";
    const size_t n = t.name.size();
    const size_t gl = sizeof(gate_sfx) - 1u;
    const size_t ul = sizeof(up_sfx) - 1u;
    if (n > gl && t.name.compare(n - gl, gl, gate_sfx) == 0) return true;
    if (n > ul && t.name.compare(n - ul, ul, up_sfx) == 0) return true;
    return false;
}

/* --repack-q2k-aligned candidates: routed-expert down stacks in Q2_K.
 * dims[0] % 256 (whole superblocks per row) and dims[1] % 2 (the layout pairs
 * adjacent rows). */
bool ds4_repack_q2k_candidate(const ds4_repack_tensor &t) {
    if (t.type != 10u || t.ndim != 3u) return false; /* GGML_TYPE_Q2_K */
    if (t.dims[0] == 0 || t.dims[1] == 0 || t.dims[2] == 0 || t.dims[2] > UINT32_MAX) return false;
    if (t.dims[0] % 256u != 0 || t.dims[1] % 2u != 0) return false;
    if (t.bytes == 0 || t.bytes % 84u != 0) return false;
    static const char down_sfx[] = ".ffn_down_exps.weight";
    const size_t n = t.name.size();
    const size_t dl = sizeof(down_sfx) - 1u;
    return n > dl && t.name.compare(n - dl, dl, down_sfx) == 0;
}

/* Aligned-SoA Q8_0 dense candidates (--repack-q8-aligned): every 2D Q8_0
 * tensor big enough to matter whose row length satisfies the decode kernel's
 * K % 1024 constraint.  token_embd is excluded: it is consumed by row-gather,
 * never by the dense GEMV.  Unlike the IQ2 expert repack these artifacts are
 * ADDITIVE (raw stays served). */
bool ds4_repack_q8_candidate(const ds4_repack_tensor &t) {
    if (t.type != 8u || t.ndim != 2u) return false; /* GGML_TYPE_Q8_0 */
    if (t.dims[0] == 0 || t.dims[1] == 0) return false;
    if (t.dims[0] % 1024u != 0) return false;
    /* 2 MiB floor: attn_kv (512 x 4096, 2.2 MiB) is an Inc4 pair-kernel
     * consumer; anything smaller isn't worth an artifact. */
    if (t.bytes < 2u * 1024u * 1024u || t.bytes % 34u != 0) return false;
    if (t.name.find("token_embd") != std::string::npos) return false;
    return true;
}

/* Q8_0 -> f16 colmajor dequant candidates (engine prebuild of the lazy
 * cuda_q8_f16_ptr cache): the packed-out_a prefill weights.  The runtime
 * admission gate admits more labels, but attn_output_a is the set the ship
 * config actually first-touch-builds on the fast paths. */
bool ds4_repack_q8_f16_candidate(const ds4_repack_tensor &t) {
    if (t.type != 8u || t.ndim != 2u) return false; /* GGML_TYPE_Q8_0 */
    if (t.dims[0] == 0 || t.dims[1] == 0) return false;
    if (t.dims[0] % 32u != 0) return false;
    if (t.bytes == 0 || t.bytes % 34u != 0) return false;
    return t.name.find("attn_output_a.weight") != std::string::npos;
}

/* ---- S5 parallel repack driver -------------------------------------------
 * The serial builders moved ~79 GiB through a fully serialized pipeline
 * (buffered pread -> sync H2D -> kernel -> device sync, one chunk at a time,
 * ~1.25 GiB/s = 63 s of every WS boot).  The tax is pipeline serialization,
 * not disk: per-TENSOR workers, each with its own non-blocking stream, pinned
 * staging and device scratch, reading via the O_DIRECT read_stage path, run
 * the NVMe at queue depth = nthreads while uploads/kernels overlap other
 * workers' reads.  Allocation (VMM reservation order) and artifact push stay
 * on the caller thread so range layout is thread-count-invariant; only the
 * byte movement fans out.  --repack-threads N / DS4_WS_REPACK_THREADS
 * overrides (1 = the old serial order through the new plumbing);
 * --repack-hash / DS4_WS_REPACK_HASH=1 prints a per-artifact FNV-1a for the
 * bit-identity gate. */
static int g_repack_threads = 0;   /* 0 = auto (min(6, hw)) */
static int g_repack_hash = -1;     /* -1 = read env on first use */

void ds4_repack_set_threads(int n) {
    g_repack_threads = n;
}

void ds4_repack_set_hash(bool on) {
    g_repack_hash = on ? 1 : 0;
}

static bool repack_hash_enabled(void) {
    if (g_repack_hash < 0) {
        const char *rh = getenv("DS4_WS_REPACK_HASH");
        g_repack_hash = (rh && rh[0] == '1' && rh[1] == '\0') ? 1 : 0;
    }
    return g_repack_hash == 1;
}

static int repack_thread_count(size_t njobs) {
    int n = g_repack_threads;
    if (n <= 0) {
        const char *e = getenv("DS4_WS_REPACK_THREADS");
        n = e && e[0] ? atoi(e) : 0;
    }
    if (n <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        n = hw > 6u ? 6 : (hw ? (int)hw : 1);
    }
    if (n > 16) n = 16;
    if ((size_t)n > njobs) n = (int)(njobs ? njobs : 1);
    return n;
}

struct repack_job {
    ds4_repack_artifact art;
    uint64_t chunk;        /* kind-specific rounded chunk bytes */
};

uint64_t ds4_repack_fnv1a(const unsigned char *p, uint64_t n, uint64_t h) {
    for (uint64_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

/* Run fn(job, stream, stage, stage_bytes, scratch) for every job across the
 * worker pool.  stage is pinned host (>= max chunk + 2*direct_align, aligned
 * for O_DIRECT), scratch is device (>= max chunk).  Returns false if any job
 * failed; caller releases the job artifacts. */
template <typename FN>
static bool run_repack_jobs(const char *log_prefix, const char *what, const ds4_repack_file &m,
                            int device, std::vector<repack_job> &jobs, FN fn) {
    if (jobs.empty()) return true;
    uint64_t max_chunk = 0;
    for (const repack_job &j : jobs) if (j.chunk > max_chunk) max_chunk = j.chunk;
    const uint64_t stage_align = (m.direct_fd >= 0 && m.direct_align > 1) ? m.direct_align : 1;
    const uint64_t stage_bytes = max_chunk + 2u * stage_align;
    const int nthreads = repack_thread_count(jobs.size());
    const bool hash = repack_hash_enabled();
    std::atomic<size_t> next{0};
    std::atomic<bool> fail{false};
    auto worker = [&]() {
        if (cudaSetDevice(device) != cudaSuccess) { fail = true; return; }
        cudaStream_t stream = nullptr;
        void *raw = nullptr;
        unsigned char *scratch = nullptr;
        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err == cudaSuccess) err = cudaMallocHost(&raw, (size_t)(stage_bytes + stage_align));
        if (err == cudaSuccess) err = cudaMalloc((void **)&scratch, (size_t)max_chunk);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s repack worker alloc failed: %s\n",
                    log_prefix, what, cudaGetErrorString(err));
            fail = true;
        } else {
            unsigned char *stage = (unsigned char *)repack_align_ptr(raw, stage_align);
            while (!fail.load(std::memory_order_relaxed)) {
                const size_t i = next.fetch_add(1u);
                if (i >= jobs.size()) break;
                if (!fn(jobs[i], stream, stage, stage_bytes, scratch)) { fail = true; break; }
                if (hash) {
                    const ds4_repack_artifact &r = jobs[i].art;
                    uint64_t h = 14695981039346656037ull;
                    for (uint64_t off = 0; off < r.bytes && !fail; off += max_chunk) {
                        const uint64_t nb = r.bytes - off < max_chunk ? r.bytes - off : max_chunk;
                        cudaError_t herr = cudaMemcpyAsync(stage, (const char *)r.dev + off, (size_t)nb,
                                                           cudaMemcpyDeviceToHost, stream);
                        if (herr == cudaSuccess) herr = cudaStreamSynchronize(stream);
                        if (herr != cudaSuccess) {
                            fprintf(stderr, "%s: %s repack hash readback failed: %s\n",
                                    log_prefix, what, cudaGetErrorString(herr));
                            fail = true;
                            break;
                        }
                        h = ds4_repack_fnv1a(stage, nb, h);
                    }
                    if (!fail)
                        fprintf(stderr, "%s: repack-hash %s %s bytes=%llu fnv=%016llx\n",
                                log_prefix, what, r.t->name.c_str(),
                                (unsigned long long)r.bytes, (unsigned long long)h);
                }
            }
        }
        if (scratch) (void)cudaFree(scratch);
        if (raw) (void)cudaFreeHost(raw);
        if (stream) (void)cudaStreamDestroy(stream);
    };
    std::vector<std::thread> pool;
    pool.reserve((size_t)nthreads);
    for (int i = 0; i < nthreads; i++) pool.emplace_back(worker);
    for (std::thread &th : pool) th.join();
    return !fail;
}

/* One chunk of one tensor: O_DIRECT read into stage, async H2D into scratch on
 * the worker stream; the caller launches its kind's kernel then stream-syncs. */
static bool repack_read_upload(const char *log_prefix, const ds4_repack_file &m, const char *what,
                               const ds4_repack_tensor &t,
                               unsigned char *stage, uint64_t stage_bytes, unsigned char *scratch,
                               uint64_t done, uint64_t nb, cudaStream_t stream) {
    const char *payload = nullptr;
    if (!ds4_repack_read_stage(m, stage, stage_bytes, t.off + done, nb, &payload)) {
        fprintf(stderr, "%s: %s repack read failed for %s at off=%llu: %s\n",
                log_prefix, what, t.name.c_str(), (unsigned long long)(t.off + done), strerror(errno));
        return false;
    }
    cudaError_t err = cudaMemcpyAsync(scratch, payload, (size_t)nb, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s repack upload failed for %s: %s\n",
                log_prefix, what, t.name.c_str(), cudaGetErrorString(err));
        return false;
    }
    return true;
}

static bool repack_alloc_artifact(const ds4_repack_build_args &a, const char *what,
                                  ds4_repack_artifact *art) {
    if (a.alloc_fn) return a.alloc_fn(a.alloc_ctx, art);
    void *dev = nullptr;
    cudaError_t err = cudaMalloc(&dev, (size_t)art->bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc failed for %s repack %s %.2f MiB: %s\n",
                a.log_prefix, what, art->t->name.c_str(),
                (double)art->bytes / 1048576.0, cudaGetErrorString(err));
        return false;
    }
    art->dev = dev;
    return true;
}

static void repack_free_artifact(const ds4_repack_build_args &a, ds4_repack_artifact *art) {
    if (!art->dev) return;
    if (a.free_fn) a.free_fn(a.alloc_ctx, art);
    else (void)cudaFree(art->dev);
    art->dev = nullptr;
}

/* ---- builders ------------------------------------------------------------ */

/* Aligned-SoA Q8_0 dense artifacts: [__half dq[nblk]][pad to 64B]
 * [int8 qs[nblk*32]], nblk = out_dim * (in_dim/32), block order identical to
 * the raw tensor byte order.  Layout contract shared with
 * ds4_mmq_q8_0_aligned_dense_vec (cuda/mmq/ds4_mmq.h).  ADDITIVE: the raw
 * spans stay served, so only the n_tokens==1 decode GEMV dispatches on the
 * artifact; every other consumer reads raw as before. */
bool ds4_repack_build_q8_aligned(const ds4_repack_build_args &a,
                                 std::vector<ds4_repack_artifact> &out,
                                 uint64_t *repacked_bytes_out) {
    if (repacked_bytes_out) *repacked_bytes_out = 0;
    ds4_repack_file m;
    if (!ds4_repack_map_file(a.log_prefix, a.path, m)) return false;
    uint64_t chunk = a.copy_chunk_bytes / 34u * 34u;
    if (chunk < 34u * 32768u) chunk = 34u * 32768u;

    const double t0 = repack_now_sec();
    std::vector<repack_job> jobs;
    bool ok = true;
    for (const ds4_repack_tensor &t : *a.records) {
        if (!ds4_repack_q8_candidate(t)) continue;
        const uint64_t nblk = t.bytes / 34u;
        const uint64_t expect_blk = (t.dims[0] / 32u) * t.dims[1];
        if (nblk != expect_blk || t.off > m.size || t.bytes > m.size - t.off) {
            fprintf(stderr,
                    "%s: q8 repack skipped %s: geometry mismatch (nblk=%llu expect=%llu)\n",
                    a.log_prefix,
                    t.name.c_str(),
                    (unsigned long long)nblk,
                    (unsigned long long)expect_blk);
            ok = false;
            break;
        }
        const uint64_t dq_bytes = repack_align_up(nblk * 2u, 64u);
        const uint64_t art_bytes = dq_bytes + nblk * 32u;

        repack_job j;
        j.chunk = chunk;
        j.art.t = &t;
        j.art.kind = DS4_REPACK_Q8_0_ALIGNED_DENSE;
        j.art.bytes = art_bytes;
        j.art.in_dim = t.dims[0];
        j.art.out_dim = t.dims[1];
        j.art.group_count = 1u;
        if (!repack_alloc_artifact(a, "q8", &j.art)) {
            ok = false;
            break;
        }
        jobs.push_back(j);
    }

    const int nthreads = repack_thread_count(jobs.size());
    if (ok)
        ok = run_repack_jobs(a.log_prefix, "q8", m, a.device, jobs,
            [&m, &a](const repack_job &j, cudaStream_t stream, unsigned char *stage,
                     uint64_t stage_bytes, unsigned char *scratch) -> bool {
        const ds4_repack_tensor &t = *j.art.t;
        const uint64_t nblk = t.bytes / 34u;
        const uint64_t dq_bytes = repack_align_up(nblk * 2u, 64u);
        __half *dq = (__half *)j.art.dev;
        unsigned char *qs = (unsigned char *)j.art.dev + dq_bytes;
        for (uint64_t done = 0; done < t.bytes; done += j.chunk) {
            const uint64_t nb = t.bytes - done < j.chunk ? t.bytes - done : j.chunk;
            if (!repack_read_upload(a.log_prefix, m, "q8", t, stage, stage_bytes, scratch, done, nb, stream))
                return false;
            const uint64_t cblk = nb / 34u;
            const uint64_t blk0 = done / 34u;
            repack_q8_0_aligned_kernel<<<(unsigned)((cblk * 2u + 255u) / 256u), 256, 0, stream>>>(
                dq + blk0, qs + blk0 * 32u, scratch, cblk);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: q8 repack kernel failed for %s: %s\n",
                        a.log_prefix, t.name.c_str(), cudaGetErrorString(err));
                return false;
            }
        }
        return true;
    });

    if (!ok) {
        for (repack_job &j : jobs) repack_free_artifact(a, &j.art);
        ds4_repack_unmap_file(m);
        return false;
    }

    uint64_t total_bytes = 0;
    uint32_t count = 0;
    for (repack_job &j : jobs) {
        out.push_back(j.art);
        total_bytes += j.art.bytes;
        count++;
    }

    ds4_repack_unmap_file(m);
    if (count == 0) {
        fprintf(stderr, "%s: --repack-q8-aligned found no candidate tensors in %s\n",
                a.log_prefix, a.model_id);
    }
    fprintf(stderr,
            "%s: q8 aligned repack %s: %u tensors %.2f GiB in %.1fs (threads=%d)\n",
            a.log_prefix,
            a.model_id,
            count,
            (double)total_bytes / 1073741824.0,
            repack_now_sec() - t0,
            nthreads);
    if (repacked_bytes_out) *repacked_bytes_out = total_bytes;
    return true;
}

/* Q8_0 -> f16 column-major dequant artifacts: in_dim*out_dim __half, element
 * order (raw block index)*32 + code index.  Kind and layout shared with the
 * weight server's --derive-q8-f16 (DERIVED_Q8_0_F16_COLMAJOR) and the
 * engine's lazy cuda_q8_f16_ptr cache.  ADDITIVE: raw stays served. */
bool ds4_repack_build_q8_f16(const ds4_repack_build_args &a,
                             std::vector<ds4_repack_artifact> &out,
                             uint64_t *repacked_bytes_out) {
    if (repacked_bytes_out) *repacked_bytes_out = 0;
    ds4_repack_file m;
    if (!ds4_repack_map_file(a.log_prefix, a.path, m)) return false;
    uint64_t chunk = a.copy_chunk_bytes / 34u * 34u;
    if (chunk < 34u * 32768u) chunk = 34u * 32768u;

    const double t0 = repack_now_sec();
    std::vector<repack_job> jobs;
    bool ok = true;
    for (const ds4_repack_tensor &t : *a.records) {
        if (!ds4_repack_q8_f16_candidate(t)) continue;
        const uint64_t nblk = t.bytes / 34u;
        const uint64_t expect_blk = (t.dims[0] / 32u) * t.dims[1];
        if (nblk != expect_blk || t.off > m.size || t.bytes > m.size - t.off) {
            fprintf(stderr,
                    "%s: q8 f16 prebuild skipped %s: geometry mismatch (nblk=%llu expect=%llu)\n",
                    a.log_prefix,
                    t.name.c_str(),
                    (unsigned long long)nblk,
                    (unsigned long long)expect_blk);
            ok = false;
            break;
        }

        repack_job j;
        j.chunk = chunk;
        j.art.t = &t;
        j.art.kind = DS4_REPACK_Q8_0_F16_COLMAJOR;
        j.art.bytes = nblk * 64u; /* nblk*32 __half elements */
        j.art.in_dim = t.dims[0];
        j.art.out_dim = t.dims[1];
        j.art.group_count = 0u;
        if (!repack_alloc_artifact(a, "q8_f16", &j.art)) {
            ok = false;
            break;
        }
        jobs.push_back(j);
    }

    const int nthreads = repack_thread_count(jobs.size());
    if (ok)
        ok = run_repack_jobs(a.log_prefix, "q8_f16", m, a.device, jobs,
            [&m, &a](const repack_job &j, cudaStream_t stream, unsigned char *stage,
                     uint64_t stage_bytes, unsigned char *scratch) -> bool {
        const ds4_repack_tensor &t = *j.art.t;
        __half *dst = (__half *)j.art.dev;
        for (uint64_t done = 0; done < t.bytes; done += j.chunk) {
            const uint64_t nb = t.bytes - done < j.chunk ? t.bytes - done : j.chunk;
            if (!repack_read_upload(a.log_prefix, m, "q8_f16", t, stage, stage_bytes, scratch, done, nb, stream))
                return false;
            const uint64_t cblk = nb / 34u;
            const uint64_t blk0 = done / 34u;
            dequant_q8_0_f16_colmajor_kernel<<<(unsigned)((cblk * 32u + 255u) / 256u), 256, 0, stream>>>(
                dst + blk0 * 32u, scratch, cblk);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: q8 f16 prebuild kernel failed for %s: %s\n",
                        a.log_prefix, t.name.c_str(), cudaGetErrorString(err));
                return false;
            }
        }
        return true;
    });

    if (!ok) {
        for (repack_job &j : jobs) repack_free_artifact(a, &j.art);
        ds4_repack_unmap_file(m);
        return false;
    }

    uint64_t total_bytes = 0;
    uint32_t count = 0;
    for (repack_job &j : jobs) {
        out.push_back(j.art);
        total_bytes += j.art.bytes;
        count++;
    }

    ds4_repack_unmap_file(m);
    if (count == 0) {
        fprintf(stderr, "%s: q8 f16 prebuild found no candidate tensors in %s\n",
                a.log_prefix, a.model_id);
    }
    fprintf(stderr,
            "%s: q8 f16 colmajor prebuild %s: %u tensors %.2f GiB in %.1fs (threads=%d)\n",
            a.log_prefix,
            a.model_id,
            count,
            (double)total_bytes / 1073741824.0,
            repack_now_sec() - t0,
            nthreads);
    if (repacked_bytes_out) *repacked_bytes_out = total_bytes;
    return true;
}

/* Aligned-SoA IQ2_XXS routed-expert artifacts.  Byte-neutral: [__half
 * dq[nblk]][pad to 64B][uint2 qs[nblk*8]], block order identical to the raw
 * tensor byte order.  Layout contract shared with
 * ds4_mmq_iq2_xxs_aligned_moe_vec (cuda/mmq/ds4_mmq.h).  These REPLACE the
 * raw residency of their source tensors (the producer stops serving/pinning
 * the raw spans); raw-layout consumers fall back to the host mmap. */
bool ds4_repack_build_iq2_aligned(const ds4_repack_build_args &a,
                                  std::vector<ds4_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out) {
    if (repacked_bytes_out) *repacked_bytes_out = 0;
    ds4_repack_file m;
    if (!ds4_repack_map_file(a.log_prefix, a.path, m)) return false;
    uint64_t chunk = a.copy_chunk_bytes / 66u * 66u;
    if (chunk < 66u * 16384u) chunk = 66u * 16384u;

    const double t0 = repack_now_sec();
    std::vector<repack_job> jobs;
    bool ok = true;
    for (const ds4_repack_tensor &t : *a.records) {
        if (!ds4_repack_iq2_candidate(t)) continue;
        const uint64_t nblk = t.bytes / 66u;
        const uint64_t expect_blk = (t.dims[0] / 256u) * t.dims[1] * t.dims[2];
        if (nblk != expect_blk || t.off > m.size || t.bytes > m.size - t.off) {
            fprintf(stderr,
                    "%s: iq2 repack skipped %s: geometry mismatch (nblk=%llu expect=%llu)\n",
                    a.log_prefix,
                    t.name.c_str(),
                    (unsigned long long)nblk,
                    (unsigned long long)expect_blk);
            ok = false;
            break;
        }
        const uint64_t dq_bytes = repack_align_up(nblk * 2u, 64u);
        const uint64_t art_bytes = dq_bytes + nblk * 64u;

        repack_job j;
        j.chunk = chunk;
        j.art.t = &t;
        j.art.kind = DS4_REPACK_IQ2_XXS_ALIGNED_MOE;
        j.art.bytes = art_bytes;
        j.art.in_dim = t.dims[0];
        j.art.out_dim = t.dims[1];
        j.art.group_count = (uint32_t)t.dims[2];
        if (!repack_alloc_artifact(a, "iq2", &j.art)) {
            ok = false;
            break;
        }
        jobs.push_back(j);
    }

    const int nthreads = repack_thread_count(jobs.size());
    if (ok)
        ok = run_repack_jobs(a.log_prefix, "iq2", m, a.device, jobs,
            [&m, &a](const repack_job &j, cudaStream_t stream, unsigned char *stage,
                     uint64_t stage_bytes, unsigned char *scratch) -> bool {
        const ds4_repack_tensor &t = *j.art.t;
        const uint64_t nblk = t.bytes / 66u;
        const uint64_t dq_bytes = repack_align_up(nblk * 2u, 64u);
        __half *dq = (__half *)j.art.dev;
        uint2 *qs = (uint2 *)((char *)j.art.dev + dq_bytes);
        for (uint64_t done = 0; done < t.bytes; done += j.chunk) {
            const uint64_t nb = t.bytes - done < j.chunk ? t.bytes - done : j.chunk;
            if (!repack_read_upload(a.log_prefix, m, "iq2", t, stage, stage_bytes, scratch, done, nb, stream))
                return false;
            const uint64_t cblk = nb / 66u;
            const uint64_t blk0 = done / 66u;
            repack_iq2_xxs_aligned_kernel<<<(unsigned)((cblk * 8u + 255u) / 256u), 256, 0, stream>>>(
                dq + blk0, qs + blk0 * 8u, scratch, cblk);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: iq2 repack kernel failed for %s: %s\n",
                        a.log_prefix, t.name.c_str(), cudaGetErrorString(err));
                return false;
            }
#if defined(POSIX_FADV_DONTNEED)
            (void)posix_fadvise(m.fd, (off_t)(t.off + done), (off_t)nb, POSIX_FADV_DONTNEED);
#endif
        }
        return true;
    });

    if (!ok) {
        for (repack_job &j : jobs) repack_free_artifact(a, &j.art);
        ds4_repack_unmap_file(m);
        return false;
    }

    uint64_t total_bytes = 0;
    uint32_t count = 0;
    for (repack_job &j : jobs) {
        out.push_back(j.art);
        total_bytes += j.art.bytes;
        count++;
    }

    ds4_repack_unmap_file(m);
    if (count == 0) {
        fprintf(stderr, "%s: --repack-iq2-aligned found no candidate tensors in %s\n",
                a.log_prefix, a.model_id);
    }
    fprintf(stderr,
            "%s: iq2 aligned repack %s: %u tensors %.2f GiB in %.1fs (threads=%d)\n",
            a.log_prefix,
            a.model_id,
            count,
            (double)total_bytes / 1073741824.0,
            repack_now_sec() - t0,
            nthreads);
    if (repacked_bytes_out) *repacked_bytes_out = total_bytes;
    return true;
}

/* Row-pair-SoA Q2_K down artifacts.  Mirror of the iq2 builder: byte-neutral
 * REPLACE artifacts for the .ffn_down_exps stacks; with npair = nblk/2
 * raw-order row-pair blocks (rows 2p, 2p+1 of an expert),
 *   [uint2 dm2[npair]][pad64][int4 sc4[npair*2]][pad64][uint2 qs2[npair*16]]
 * Layout contract shared with ds4_mmq_q2_K_aligned_moe_vec
 * (cuda/mmq/ds4_mmq.h).  Chunks are whole row pairs so the repack kernel's
 * global pair index math holds across chunk boundaries. */
bool ds4_repack_build_q2k_aligned(const ds4_repack_build_args &a,
                                  std::vector<ds4_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out) {
    if (repacked_bytes_out) *repacked_bytes_out = 0;
    ds4_repack_file m;
    if (!ds4_repack_map_file(a.log_prefix, a.path, m)) return false;

    const double t0 = repack_now_sec();
    std::vector<repack_job> jobs;
    bool ok = true;
    for (const ds4_repack_tensor &t : *a.records) {
        if (!ds4_repack_q2k_candidate(t)) continue;
        const uint64_t nblk = t.bytes / 84u;
        const uint64_t expect_blk = (t.dims[0] / 256u) * t.dims[1] * t.dims[2];
        if (nblk != expect_blk || t.off > m.size || t.bytes > m.size - t.off) {
            fprintf(stderr,
                    "%s: q2k repack skipped %s: geometry mismatch (nblk=%llu expect=%llu)\n",
                    a.log_prefix,
                    t.name.c_str(),
                    (unsigned long long)nblk,
                    (unsigned long long)expect_blk);
            ok = false;
            break;
        }
        const uint64_t nb_row = t.dims[0] / 256u;
        const uint64_t pair_bytes = 2u * nb_row * 84u;
        uint64_t chunk = a.copy_chunk_bytes / pair_bytes * pair_bytes;
        if (chunk < pair_bytes * 4096u) chunk = pair_bytes * 4096u;
        if (chunk > t.bytes) chunk = t.bytes;
        const uint64_t npair = nblk / 2u;
        const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
        const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
        const uint64_t art_bytes = dm_bytes + sc_bytes + npair * 128u;

        repack_job j;
        j.chunk = chunk;
        j.art.t = &t;
        j.art.kind = DS4_REPACK_Q2_K_ALIGNED_MOE;
        j.art.bytes = art_bytes;
        j.art.in_dim = t.dims[0];
        j.art.out_dim = t.dims[1];
        j.art.group_count = (uint32_t)t.dims[2];
        if (!repack_alloc_artifact(a, "q2k", &j.art)) {
            ok = false;
            break;
        }
        jobs.push_back(j);
    }

    const int nthreads = repack_thread_count(jobs.size());
    if (ok)
        ok = run_repack_jobs(a.log_prefix, "q2k", m, a.device, jobs,
            [&m, &a](const repack_job &j, cudaStream_t stream, unsigned char *stage,
                     uint64_t stage_bytes, unsigned char *scratch) -> bool {
        const ds4_repack_tensor &t = *j.art.t;
        const uint64_t nblk = t.bytes / 84u;
        const uint64_t nb_row = t.dims[0] / 256u;
        const uint64_t npair = nblk / 2u;
        const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
        const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
        uint32_t *dm2 = (uint32_t *)j.art.dev;
        uint32_t *sc4 = (uint32_t *)((char *)j.art.dev + dm_bytes);
        uint32_t *qs2 = (uint32_t *)((char *)j.art.dev + dm_bytes + sc_bytes);
        for (uint64_t done = 0; done < t.bytes; done += j.chunk) {
            const uint64_t nb = t.bytes - done < j.chunk ? t.bytes - done : j.chunk;
            if (!repack_read_upload(a.log_prefix, m, "q2k", t, stage, stage_bytes, scratch, done, nb, stream))
                return false;
            const uint64_t cblk = nb / 84u;
            const uint64_t blk0 = done / 84u;
            repack_q2_k_aligned_kernel<<<(unsigned)((cblk * 16u + 255u) / 256u), 256, 0, stream>>>(
                dm2, sc4, qs2, scratch, blk0, cblk,
                (uint32_t)nb_row, (uint32_t)t.dims[1]);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: q2k repack kernel failed for %s: %s\n",
                        a.log_prefix, t.name.c_str(), cudaGetErrorString(err));
                return false;
            }
#if defined(POSIX_FADV_DONTNEED)
            (void)posix_fadvise(m.fd, (off_t)(t.off + done), (off_t)nb, POSIX_FADV_DONTNEED);
#endif
        }
        return true;
    });

    if (!ok) {
        for (repack_job &j : jobs) repack_free_artifact(a, &j.art);
        ds4_repack_unmap_file(m);
        return false;
    }

    uint64_t total_bytes = 0;
    uint32_t count = 0;
    for (repack_job &j : jobs) {
        out.push_back(j.art);
        total_bytes += j.art.bytes;
        count++;
    }

    ds4_repack_unmap_file(m);
    if (count == 0) {
        fprintf(stderr, "%s: --repack-q2k-aligned found no candidate tensors in %s\n",
                a.log_prefix, a.model_id);
    }
    fprintf(stderr,
            "%s: q2k aligned repack %s: %u tensors %.2f GiB in %.1fs (threads=%d)\n",
            a.log_prefix,
            a.model_id,
            count,
            (double)total_bytes / 1073741824.0,
            repack_now_sec() - t0,
            nthreads);
    if (repacked_bytes_out) *repacked_bytes_out = total_bytes;
    return true;
}
