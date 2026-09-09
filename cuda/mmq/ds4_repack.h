/* ds4_repack: the aligned-SoA derived-weight layout library shared by the
 * weight server (tools/ds4_weight_server.cu) and the engine's in-process
 * self-load build (ds4_cuda.cu).  The GGUF tensor catalog walk, the repack
 * candidate predicates, the S5 parallel repack driver, the FNV-1a artifact
 * hash, and the three aligned repack builds (IQ2_XXS gate/up, Q2_K down,
 * Q8_0 dense) live here so both producers emit bit-identical artifacts
 * (gate: --repack-hash / DS4_WS_REPACK_HASH FNV lines).  Layout contracts
 * are shared with the consuming kernels in cuda/mmq/ds4_mmq.h. */
#ifndef DS4_REPACK_H
#define DS4_REPACK_H

#include <cstdint>
#include <string>
#include <vector>

struct ds4_repack_file {
    int fd = -1;
    int direct_fd = -1;
    const uint8_t *data = nullptr;
    uint64_t size = 0;
    uint64_t direct_align = 1;
};

struct ds4_repack_span {
    uint64_t off = 0;
    uint64_t end = 0;
};

struct ds4_repack_tensor {
    std::string name;
    uint32_t type = 0;
    uint32_t ndim = 0;
    uint64_t dims[8] = {};
    uint64_t elements = 0;
    uint64_t off = 0;
    uint64_t bytes = 0;
};

/* Aligned-artifact kinds.  Numbering is shared with the weight-server
 * manifest format and the engine's cuda_derived_kind (ds4_cuda.cu); the
 * layout comments live on the builders in ds4_repack.cu. */
enum {
    DS4_REPACK_Q8_0_F16_COLMAJOR = 2,
    DS4_REPACK_IQ2_XXS_ALIGNED_MOE = 4,
    DS4_REPACK_Q8_0_ALIGNED_DENSE = 5,
    DS4_REPACK_Q2_K_ALIGNED_MOE = 6,
};

struct ds4_repack_artifact {
    const ds4_repack_tensor *t = nullptr; /* borrowed from the caller's catalog */
    uint32_t kind = 0;                    /* DS4_REPACK_* */
    uint64_t bytes = 0;                   /* artifact byte size */
    uint64_t in_dim = 0;                  /* dims[0] */
    uint64_t out_dim = 0;                 /* dims[1] */
    uint32_t group_count = 0;             /* dims[2] for MoE kinds, 1 for dense,
                                             0 for f16 colmajor (the engine's
                                             lazy-cache lookup passes 0) */
    void *dev = nullptr;                  /* device artifact buffer */
};

/* Optional device-allocation hooks (weight server VMM backend).  alloc must
 * fill art->dev with at least art->bytes of device memory and return true;
 * free releases an allocation made by alloc.  Leave both NULL for plain
 * cudaMalloc/cudaFree. */
typedef bool (*ds4_repack_alloc_fn)(void *ctx, ds4_repack_artifact *art);
typedef void (*ds4_repack_free_fn)(void *ctx, ds4_repack_artifact *art);

struct ds4_repack_build_args {
    const char *log_prefix = "ds4";  /* stderr line prefix ("ds4_weight_server" | "ds4") */
    const char *model_id = "base";   /* log label */
    const char *path = nullptr;      /* GGUF path (each builder re-maps it) */
    const std::vector<ds4_repack_tensor> *records = nullptr;
    int device = 0;
    uint64_t copy_chunk_bytes = 0;   /* staged read chunk (weight server --copy-chunk-mb) */
    ds4_repack_alloc_fn alloc_fn = nullptr;
    ds4_repack_free_fn free_fn = nullptr;
    void *alloc_ctx = nullptr;
};

bool ds4_repack_map_file(const char *log_prefix, const char *path, ds4_repack_file &m);
void ds4_repack_unmap_file(ds4_repack_file &m);

/* Parse the GGUF header + tensor table into whole-tensor spans and records
 * with absolute file offsets.  Either out pointer may be NULL. */
bool ds4_repack_collect_catalog(const char *log_prefix,
                                const ds4_repack_file &m,
                                std::vector<ds4_repack_span> *spans,
                                std::vector<ds4_repack_tensor> *records);

bool ds4_repack_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset);
/* Read [file_off, file_off+bytes) into stage, preferring the O_DIRECT fd;
 * *payload points at the requested bytes inside stage. */
bool ds4_repack_read_stage(const ds4_repack_file &m, void *stage, uint64_t stage_bytes,
                           uint64_t file_off, uint64_t bytes, const char **payload);

/* Candidate predicates for the three aligned repacks. */
bool ds4_repack_iq2_candidate(const ds4_repack_tensor &t);
bool ds4_repack_q2k_candidate(const ds4_repack_tensor &t);
bool ds4_repack_q8_candidate(const ds4_repack_tensor &t);
bool ds4_repack_q8_f16_candidate(const ds4_repack_tensor &t);

/* S5 driver configuration.  Unset (0 / false) defers to the env knobs
 * DS4_WS_REPACK_THREADS and DS4_WS_REPACK_HASH. */
void ds4_repack_set_threads(int n);
void ds4_repack_set_hash(bool on);

uint64_t ds4_repack_fnv1a(const unsigned char *p, uint64_t n, uint64_t h);

/* Build every candidate's aligned artifact.  On success appends the built
 * artifacts to out, adds their byte total to *repacked_bytes_out (may be
 * NULL), and returns true; on any failure releases every artifact this call
 * allocated and returns false without touching out. */
bool ds4_repack_build_q8_aligned(const ds4_repack_build_args &a,
                                 std::vector<ds4_repack_artifact> &out,
                                 uint64_t *repacked_bytes_out);
bool ds4_repack_build_iq2_aligned(const ds4_repack_build_args &a,
                                  std::vector<ds4_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out);
bool ds4_repack_build_q2k_aligned(const ds4_repack_build_args &a,
                                  std::vector<ds4_repack_artifact> &out,
                                  uint64_t *repacked_bytes_out);
bool ds4_repack_build_q8_f16(const ds4_repack_build_args &a,
                             std::vector<ds4_repack_artifact> &out,
                             uint64_t *repacked_bytes_out);

#endif /* DS4_REPACK_H */
