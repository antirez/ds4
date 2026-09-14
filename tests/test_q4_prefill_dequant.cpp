// SPDX-License-Identifier: MIT
// CPU layout oracle or standalone CUDA/HIP scalar/old-vector/paired parity and timing.
// This includes the new production kernel; the scalar oracle below mirrors
// the unchanged backend kernels. Full-backend/model parity is a separate gate.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define GPU(name) hip##name
using DeviceProp = hipDeviceProp_t;
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define GPU(name) cuda##name
using DeviceProp = cudaDeviceProp;
#endif
#include "../cuda/ds4_q4_dequant_layout.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dq = ds4_q4_dequant;
static void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "Q4 prefill dequant: FAIL %s\n", what); std::exit(1); }
}
struct BlockQ4 { uint16_t d, dmin; uint8_t scales[12], qs[128]; };
static_assert(sizeof(BlockQ4) == 144, "GGUF Q4_K ABI");
static uint32_t random_u32(uint32_t &s) { return s = s * 1664525u + 1013904223u; }
[[maybe_unused]] static bool half_equal(uint16_t a, uint16_t b) {
    const bool both_nan = (a & 0x7c00u) == 0x7c00u && (a & 0x3ffu) &&
                          (b & 0x7c00u) == 0x7c00u && (b & 0x3ffu);
    return a == b || both_nan;
}

static void host_test() {
    size_t policies = 0, mapped = 0, codes = 0;
    constexpr uintptr_t src = 0x10000000u, dst = 0x20000000u;
    for (uint64_t k : {0u,256u,1024u,7168u})
    for (uint64_t m : {0u,1u,32767u,32768u,32769u})
    for (uint64_t n : {0u,1u,8u,255u,256u,2048u,4096u,8192u,8193u})
    for (unsigned flags = 0; flags < 16; ++flags)
    for (unsigned skew = 0; skew < 16; ++skew) {
        const bool device = flags & 1, quality = flags & 2;
        const bool ssd = flags & 4, disabled = flags & 8;
        const bool expected = k == 1024 && m == 32768 && n >= 256 && n <= 8192 &&
            device && !quality && !ssd && !disabled && skew == 0;
        require(dq::select(k,m,n,src+skew,dst,device,quality,ssd,disabled) == expected,
                "source alignment or admission");
        require(dq::select(k,m,n,src,dst+skew,device,quality,ssd,disabled) == expected,
                "destination alignment or admission");
        ++policies;
    }
    constexpr uintptr_t wb = 1024u / 256u * 32768u * 144u, ob = 1024u * 32768u * 2u;
    for (uintptr_t output : {src, src+16u, src+wb-16u, src+wb, src-ob, src-ob+16u}) {
        const bool disjoint = output >= src ? output-src >= wb : src-output >= ob;
        require(dq::select(1024,32768,4096,src,output,true,false,false,false) == disjoint,
                "overlap rejection / exact adjacent spans");
    }
    require(!dq::select(1024,32768,4096,0,dst,true,false,false,false), "null source");
    require(!dq::select(1024,32768,4096,src,0,true,false,false,false), "null output");
    require(!dq::select(1024,32768,4096,UINTPTR_MAX-15u,dst,true,false,false,false), "source overflow");
    require(!dq::select(1024,32768,4096,src,UINTPTR_MAX-15u,true,false,false,false), "output overflow");

    for (uint64_t k : {256u,512u,1024u,7168u}) for (uint64_t m : {1u,17u,33u}) {
        const uint64_t chunks = m*k/16u;
        std::vector<uint8_t> writes(m*k);
        for (uint64_t c = 0; c < chunks; ++c) {
            const uint64_t row = c / (k/16u), col = (c % (k/16u))*16u;
            require(dq::block(c) == row*(k/256u)+col/256u, "flat block != strided block");
            require(dq::within(c) == col%256u && dq::group(c) == (col%256u)/32u, "group mapping");
            const uint32_t offset = dq::packed_offset(c);
            require(offset%16u == 0 && offset+16u <= 128u, "vector input overread/alignment");
            for (unsigned j = 0; j < 16; ++j) {
                const unsigned within = unsigned(col%256u)+j, g = within/32u;
                require(offset+j == (g/2u)*32u + within%32u, "packed byte address");
                ++writes[c*16u+j];
            }
            ++mapped;
        }
        for (uint8_t count : writes) require(count == 1, "unique complete output ownership");
    }
    for (uint64_t blocks : {0u,1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,1024u}) {
        std::vector<uint8_t> writes(blocks*256u), reads(blocks*128u);
        for (uint64_t task = 0; task < blocks*8u; ++task) {
            const uint64_t low = dq::low_chunk(task);
            const uint64_t block = task/8u;
            const uint32_t offset = dq::packed_offset(low);
            require(dq::block(low) == block && dq::block(low+2u) == block,
                    "paired groups stay in their block");
            require(dq::group(low)%2u == 0 && dq::group(low+2u) == dq::group(low)+1u,
                    "paired groups select low then high nibble");
            require(offset%16u == 0 && offset+16u <= 128u &&
                    offset == dq::packed_offset(low+2u), "paired vector alignment and bounds");
            for (uint32_t j = 0; j < 16; ++j) ++reads[block*128u+offset+j];
            for (uint32_t hi = 0; hi < 2; ++hi) for (uint32_t j = 0; j < 16; ++j) {
                const uint64_t out = (low+2u*hi)*16u+j;
                const uint32_t within = out%256u, g = within/32u;
                require(offset+j == (g/2u)*32u+within%32u,"paired packed byte address");
                ++writes[out];
            }
        }
        for (uint8_t count : writes) require(count == 1,"paired unique complete output ownership");
        for (uint8_t count : reads) require(count == 1,"one logical packed-byte read per block");
    }
    // Independently pack all 6-bit scale/min combinations, then decode them.
    for (unsigned sc = 0; sc < 64; ++sc) for (unsigned mn = 0; mn < 64; ++mn) {
        uint8_t scales[8], minima[8], packed[12]{};
        for (unsigned g = 0; g < 8; ++g) {
            scales[g] = (sc + g*7u) & 63u; minima[g] = (mn + g*11u) & 63u;
        }
        for (unsigned g = 0; g < 4; ++g) {
            packed[g] = scales[g] | ((scales[g+4] >> 4u) << 6u);
            packed[g+4] = minima[g] | ((minima[g+4] >> 4u) << 6u);
            packed[g+8] = (scales[g+4] & 15u) | ((minima[g+4] & 15u) << 4u);
        }
        for (unsigned g = 0; g < 8; ++g) {
            uint8_t s, m; dq::scale_min(g,packed,&s,&m);
            require(s == scales[g] && m == minima[g], "six-bit metadata"); ++codes;
        }
    }
    uint32_t seed = 0x14526281;
    for (uint32_t byte = 0; byte < 256; ++byte)
    for (uint32_t position = 0; position < 4; ++position)
    for (uint32_t g = 0; g < 8; ++g) {
        const uint32_t word = (random_u32(seed) & ~(255u << (8u*position))) |
                              (byte << (8u*position));
        require(dq::nibble(word,position,g) == ((g & 1u) ? byte >> 4u : byte & 15u), "nibbles");
    }
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint16_t hi = uint16_t(h ^ 0xa52bu);
        const uint32_t pair = dq::half_pair(uint16_t(h),hi);
        require(uint16_t(pair) == h && uint16_t(pair >> 16u) == hi, "half-bit packing");
    }
    std::printf("PASS host: %zu policies, %zu mapped chunks, %zu metadata codes, "
                "all nibbles/half-bit patterns. No GPU or F16 arithmetic proof.\n", policies,mapped,codes);
}

// CPU-only execution of the actual vector kernel body. This catches packed
// stores/arithmetic errors under ASan/UBSan, but emulates neither GPU codegen
// nor CUDA's FTZ behavior. It is explicitly NOT a CUDA/HIP compile test.
#if !defined(__HIPCC__) && !defined(__CUDACC__) && defined(__FLT16_MANT_DIG__)
using __half = _Float16;
struct alignas(16) uint4 { uint32_t x,y,z,w; };
static uint4 make_uint4(uint32_t a,uint32_t b,uint32_t c,uint32_t d) { return {a,b,c,d}; }
static __half __ushort_as_half(uint16_t b) { __half h; std::memcpy(&h,&b,2); return h; }
static uint16_t __half_as_ushort(__half h) { uint16_t b; std::memcpy(&b,&h,2); return b; }
static float __half2float(__half h) { return static_cast<float>(h); }
static __half __float2half_rn(float f) { return static_cast<__half>(f); }
static struct { uint32_t x; } threadIdx, blockIdx, blockDim;
#define __global__
#endif

#if defined(__HIPCC__) || defined(__CUDACC__) || defined(__FLT16_MANT_DIG__)
#include "../cuda/ds4_q4_dequant_vec.cuh"

// Frozen ae60899 vec16 algorithm and addressing: no candidate helpers.
// Original header SHA256: 8f3ca3f9bbc54c8ed42391bad8349e5f21f0c8b1e3cfe178f900f3146fd994a9.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define FROZEN_DQ_INLINE __host__ __device__ __forceinline__
#else
#define FROZEN_DQ_INLINE inline
#endif
namespace frozen_vec16_layout {
enum { block_bytes = 144 };
FROZEN_DQ_INLINE uint64_t block(uint64_t chunk) { return chunk >> 4u; }
FROZEN_DQ_INLINE uint32_t within(uint64_t chunk) { return (chunk & 15u) * 16u; }
FROZEN_DQ_INLINE uint32_t group(uint64_t chunk) { return (chunk >> 1u) & 7u; }
FROZEN_DQ_INLINE uint32_t packed_offset(uint64_t chunk) {
    return ((chunk >> 2u) & 3u) * 32u + (chunk & 1u) * 16u;
}
FROZEN_DQ_INLINE void scale_min(uint32_t g, const uint8_t *s,
                                uint8_t *scale, uint8_t *minimum) {
    if (g < 4u) {
        *scale = s[g] & 63u;
        *minimum = s[g + 4u] & 63u;
    } else {
        *scale = (s[g + 4u] & 15u) | ((s[g - 4u] >> 6u) << 4u);
        *minimum = (s[g + 4u] >> 4u) | ((s[g] >> 6u) << 4u);
    }
}
FROZEN_DQ_INLINE uint32_t nibble(uint32_t word, uint32_t byte, uint32_t g) {
    return (word >> (8u * byte + ((g & 1u) * 4u))) & 15u;
}
FROZEN_DQ_INLINE uint32_t half_pair(uint16_t lo, uint16_t hi) {
    return (uint32_t)lo | ((uint32_t)hi << 16u);
}
}
#undef FROZEN_DQ_INLINE
template<typename BlockQ4>
__global__ static void frozen_vec16_reference(
        __half *dst, const BlockQ4 *src, uint64_t total_blocks) {
    namespace dq = frozen_vec16_layout;
    static_assert(sizeof(BlockQ4) == dq::block_bytes &&
                  offsetof(BlockQ4, d) == 0 && offsetof(BlockQ4, dmin) == 2 &&
                  offsetof(BlockQ4, scales) == 4 && offsetof(BlockQ4, qs) == 16,
                  "Q4_K GGUF ABI");
    const uint64_t chunk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (dq::block(chunk) >= total_blocks) return;
    // Complete Q4_K rows are contiguous: no runtime 64-bit row division.
    const BlockQ4 *xb = src + dq::block(chunk);
    const uint32_t g = dq::group(chunk);
    uint8_t scale, minimum;
    dq::scale_min(g, xb->scales, &scale, &minimum);
    const float d = __half2float(__ushort_as_half(xb->d));
    const float dmin = __half2float(__ushort_as_half(xb->dmin));

    // One aligned 16-byte input vector and two 16-byte output vectors replace
    // the scalar byte/half accesses. Each chunk stays inside one Q4 group;
    // the final load ends inside qs, never in the next block's metadata.
    const uint4 packed = *reinterpret_cast<const uint4 *>(xb->qs + dq::packed_offset(chunk));
    const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
    for (uint32_t h = 0; h < 2u; ++h) {
        uint32_t pairs[4];
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (uint32_t j = 0; j < 4u; ++j) {
            const uint32_t k = h * 8u + j * 2u;
            const uint32_t q0 = dq::nibble(words[k / 4u], k & 3u, g);
            const uint32_t q1 = dq::nibble(words[k / 4u], (k & 3u) + 1u, g);
            // Same F32 algebra and single RN-to-F16 boundary as the scalar
            // kernels. Do not replace with half arithmetic or change FMA flags.
            const __half a = __float2half_rn(
                (d * (float)scale) * (float)q0 - dmin * (float)minimum);
            const __half b = __float2half_rn(
                (d * (float)scale) * (float)q1 - dmin * (float)minimum);
            pairs[j] = dq::half_pair(__half_as_ushort(a), __half_as_ushort(b));
        }
        *reinterpret_cast<uint4 *>(dst + chunk * 16u + h * 8u) =
            make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
    }
}
#endif

#if !defined(__HIPCC__) && !defined(__CUDACC__) && defined(__FLT16_MANT_DIG__)
#undef __global__

static void host_kernel_test() {
    alignas(16) std::array<BlockQ4,65> input;
    alignas(16) std::array<uint16_t,65*256+32> output, previous;
    constexpr uint16_t special[] = {0,0x8000,1,0x3ff,0x400,0x3c00,0xbc00,0x7bff,0x7c00,0xfc00,0x7e11};
    uint32_t seed = 0x9832a831;
    size_t results = 0;
    for (unsigned fixture = 0; fixture < 128; ++fixture) {
        for (size_t b = 0; b < input.size(); ++b) {
            input[b].d = fixture < 11 ? special[fixture] : uint16_t(random_u32(seed));
            input[b].dmin = fixture < 11 ? special[(fixture+b)%11] : uint16_t(random_u32(seed));
            for (auto &v : input[b].scales) v = uint8_t(random_u32(seed) >> 16);
            for (auto &v : input[b].qs) v = uint8_t(random_u32(seed) >> 16);
        }
        for (uint64_t nblocks : {0u,1u,15u,16u,17u,31u,32u,33u,63u,64u,65u}) {
            const auto original = input;
            output.fill(0xa5a5u);
            previous.fill(0xa5a5u);
            blockDim.x = 256;
            for (blockIdx.x = 0; blockIdx.x < (nblocks+31u)/32u+1u; ++blockIdx.x)
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
                    ds4_q4_dequant_f16_pair32_kernel(
                        reinterpret_cast<__half *>(output.data()+16),input.data(),nblocks);
            for (blockIdx.x = 0; blockIdx.x < (nblocks+15u)/16u+1u; ++blockIdx.x)
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)
                    frozen_vec16_reference(
                        reinterpret_cast<__half *>(previous.data()+16),input.data(),nblocks);
            for (uint64_t i = 0; i < nblocks*256u; ++i) {
                const BlockQ4 &b = input[i/256u];
                const uint32_t within = i%256u, g = within/32u;
                const uint8_t s = g < 4 ? b.scales[g]&63u :
                    (b.scales[g+4]&15u) | ((b.scales[g-4]>>6u)<<4u);
                const uint8_t m = g < 4 ? b.scales[g+4]&63u :
                    (b.scales[g+4]>>4u) | ((b.scales[g]>>6u)<<4u);
                const uint8_t byte = b.qs[(g/2u)*32u+within%32u];
                const uint32_t q = (g&1u) ? byte>>4u : byte&15u;
                const float d = __half2float(__ushort_as_half(b.d));
                const float dm = __half2float(__ushort_as_half(b.dmin));
                const uint16_t expected = __half_as_ushort(__float2half_rn(
                    (d*(float)s)*(float)q-dm*(float)m));
                require(half_equal(output[16+i],expected),"CPU paired body vs scalar F16");
                require(half_equal(previous[16+i],expected),"CPU frozen vec16 vs scalar F16");
                ++results;
            }
            require(!std::memcmp(input.data(),original.data(),sizeof(input)),"CPU input unchanged");
            for (size_t i = 0; i < 16; ++i)
                require(output[i] == 0xa5a5u && previous[i] == 0xa5a5u,"CPU prefix guards");
            for (size_t i = 16+nblocks*256u; i < output.size(); ++i)
                require(output[i] == 0xa5a5u && previous[i] == 0xa5a5u,"CPU suffix guards / inactive threads");
        }
    }
    std::printf("PASS CPU paired/frozen kernel-body simulation: %zu half outputs per kernel, including exceptional values and guards. "
                "Not GPU arithmetic/codegen proof.\n",results);
}
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
static void check(GPU(Error_t) e) {
    if (e != GPU(Success)) { std::fprintf(stderr,"%s\n",GPU(GetErrorString)(e)); std::exit(1); }
}
// Intentionally independent scalar copy of the original backend algorithm.
// Do not use the candidate's indexing or metadata helpers in this oracle.
__global__ static void scalar_reference(__half *dst, const BlockQ4 *src,
                                        uint64_t k, uint64_t m) {
    const uint64_t c = uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (c >= m*(k/16u)) return;
    const uint64_t row = c/(k/16u), col = (c-row*(k/16u))*16u;
    const BlockQ4 *b = src+row*(k/256u)+col/256u;
    const float d = __half2float(__ushort_as_half(b->d));
    const float dm = __half2float(__ushort_as_half(b->dmin));
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) {
        const unsigned within = unsigned(col%256u)+j, g = within/32u;
        const uint8_t s = g < 4u ? b->scales[g] & 63u :
            (b->scales[g+4] & 15u) | ((b->scales[g-4] >> 6u) << 4u);
        const uint8_t mn = g < 4u ? b->scales[g+4] & 63u :
            (b->scales[g+4] >> 4u) | ((b->scales[g] >> 6u) << 4u);
        const uint8_t p = b->qs[(g/2u)*32u+within%32u];
        const uint32_t q = (g & 1u) ? p >> 4u : p & 15u;
        dst[row*k+col+j] = __float2half_rn((d*(float)s)*(float)q-dm*(float)mn);
    }
}
struct Buffer {
    unsigned char *base = nullptr;
    size_t bytes, prefix;
    explicit Buffer(size_t n, unsigned skew = 0) : bytes(n), prefix(256u+skew) {
        check(GPU(Malloc)(reinterpret_cast<void **>(&base),prefix+bytes+256u));
        check(GPU(Memset)(base,0xa5,prefix+bytes+256u));
    }
    ~Buffer() { GPU(Free)(base); }
    Buffer(const Buffer &) = delete;
    template<typename T> T *ptr() { return reinterpret_cast<T *>(base+prefix); }
    std::vector<unsigned char> read() {
        std::vector<unsigned char> all(prefix+bytes+256u);
        check(GPU(Memcpy)(all.data(),base,all.size(),GPU(MemcpyDeviceToHost)));
        for (size_t i = 0; i < prefix; ++i) require(all[i] == 0xa5,"prefix guard");
        for (size_t i = prefix+bytes; i < all.size(); ++i) require(all[i] == 0xa5,"suffix guard");
        return {all.begin()+prefix,all.begin()+prefix+bytes};
    }
};
static void parity(const std::vector<unsigned char> &a, const std::vector<unsigned char> &b) {
    require(a.size() == b.size(),"output size");
    for (size_t i = 0; i < a.size(); i += 2) {
        uint16_t x,y; std::memcpy(&x,a.data()+i,2); std::memcpy(&y,b.data()+i,2);
        // All finite values, signed zeros and infinities are bitwise exact.
        // Only NaN payload differences are allowed in exceptional fixtures.
        require(half_equal(x,y),"F16 parity (finite bits / nonfinite class)");
    }
}
static std::vector<BlockQ4> weights(size_t blocks, bool exceptional) {
    std::vector<BlockQ4> w(blocks);
    uint32_t seed = 0x853928ab;
    const uint16_t special[] = {0,0x8000,1,0x3ff,0x400,0x3c00,0xbc00,0x7bff,0x7c00,0xfc00,0x7e11};
    for (size_t b = 0; b < blocks; ++b) {
        w[b].d = uint16_t(random_u32(seed)) & 0xfbffu;
        w[b].dmin = uint16_t(random_u32(seed)) & 0xfbffu;
        if (b < 11 && (exceptional || b < 8)) {
            w[b].d = special[b]; w[b].dmin = special[(b+3)%8];
        }
        for (auto &v : w[b].scales) v = uint8_t(random_u32(seed) >> 16);
        for (size_t i = 0; i < 128; ++i) w[b].qs[i] = uint8_t(i+b); // all 256 packed byte values
    }
    return w;
}
static void run_case(uint64_t k, uint64_t m, unsigned skew, bool graph, bool bench) {
    const uint64_t blocks = (k/256u)*m, chunks = blocks*16u;
    auto input = weights(size_t(blocks),!bench);
    if (bench) for (size_t b = 0; b < input.size(); ++b) {
        // Time ordinary finite weights; exceptional values belong to parity.
        input[b].d = uint16_t(0x2400u | ((b*7u) & 0x3ffu));
        input[b].dmin = uint16_t(0x1800u | ((b*11u) & 0x3ffu));
    }
    Buffer x(input.size()*sizeof(BlockQ4),skew), y(size_t(k*m*2u),skew);
    if (bench) require(dq::select(k,m,4096,reinterpret_cast<uintptr_t>(x.ptr<BlockQ4>()),
                                  reinterpret_cast<uintptr_t>(y.ptr<__half>()),true,false,false,false),
                       "benchmark allocation must pass production admission");
    check(GPU(Memcpy)(x.ptr<BlockQ4>(),input.data(),x.bytes,GPU(MemcpyHostToDevice)));
    GPU(Stream_t) stream;
    check(GPU(StreamCreateWithFlags)(&stream,GPU(StreamNonBlocking)));
    auto launch = [&](int variant) {
        if (variant == 2) ds4_q4_dequant_f16_pair32_kernel<<<unsigned((chunks/2u+255u)/256u),256,0,stream>>>(
            y.ptr<__half>(),x.ptr<BlockQ4>(),blocks);
        else if (variant == 1) frozen_vec16_reference<<<unsigned((chunks+255u)/256u),256,0,stream>>>(
            y.ptr<__half>(),x.ptr<BlockQ4>(),blocks);
        else scalar_reference<<<unsigned((chunks+255u)/256u),256,0,stream>>>(
            y.ptr<__half>(),x.ptr<BlockQ4>(),k,m);
        check(GPU(GetLastError)());
    };
    launch(0); check(GPU(StreamSynchronize)(stream));
    const auto ref = y.read();
    for (int variant = 1; variant < 3; ++variant) {
        check(GPU(MemsetAsync)(y.ptr<__half>(),0xa5,y.bytes,stream));
        launch(variant); check(GPU(StreamSynchronize)(stream)); parity(ref,y.read());
    }
    if (graph || bench) {
        GPU(Graph_t) graphs[2]{}; GPU(GraphExec_t) execs[2]{};
        const int repeats = bench ? 4 : 1;
        for (int v = 0; v < 2; ++v) {
            for (int warm = 0; warm < 8; ++warm) launch(v+1);
            check(GPU(StreamSynchronize)(stream));
            check(GPU(StreamBeginCapture)(stream,GPU(StreamCaptureModeGlobal)));
            for (int i = 0; i < repeats; ++i) launch(v+1);
            check(GPU(StreamEndCapture)(stream,&graphs[v]));
            check(GPU(GraphInstantiate)(&execs[v],graphs[v],nullptr,nullptr,0));
            check(GPU(GraphLaunch)(execs[v],stream)); check(GPU(StreamSynchronize)(stream));
        }
        GPU(Event_t) start,end;
        check(GPU(EventCreate)(&start)); check(GPU(EventCreate)(&end));
        std::array<std::vector<float>,2> samples;
        // ABBA/BAAB order across pairs; immutable weights in every arm.
        for (int s = 0; s < (bench ? 12 : 3); ++s) for (int a = 0; a < 2; ++a) {
            const int v = (s & 1) ? 1-a : a;
            check(GPU(MemsetAsync)(y.ptr<__half>(),0xa5,y.bytes,stream));
            check(GPU(EventRecord)(start,stream));
            check(GPU(GraphLaunch)(execs[v],stream));
            check(GPU(EventRecord)(end,stream)); check(GPU(EventSynchronize)(end));
            float ms; check(GPU(EventElapsedTime)(&ms,start,end));
            require(ms >= 0.f && std::isfinite(ms),"event time");
            samples[v].push_back(ms*1000.f/repeats);
            parity(ref,y.read());
        }
        if (bench) for (int v = 0; v < 2; ++v) {
            const auto &raw = samples[v];
            auto sorted = raw; std::sort(sorted.begin(),sorted.end());
            double mean = 0, variance = 0;
            for (float us : raw) mean += us;
            mean /= raw.size();
            for (float us : raw) variance += (us-mean)*(us-mean);
            variance /= raw.size()-1u;
            std::printf("K=%llu M=%llu %s median_us=%.3f mean_us=%.3f sd_us=%.3f "
                        "min_us=%.3f max_us=%.3f samples_us=",
                (unsigned long long)k,(unsigned long long)m,v ? "pair32" : "old_vec16",
                (sorted[5]+sorted[6])*0.5f,mean,std::sqrt(variance),sorted.front(),sorted.back());
            for (size_t i = 0; i < raw.size(); ++i) std::printf("%s%.3f",i ? "," : "",raw[i]);
            std::putchar('\n');
        }
        for (int v = 0; v < 2; ++v) {
            check(GPU(GraphExecDestroy)(execs[v])); check(GPU(GraphDestroy)(graphs[v]));
        }
        check(GPU(EventDestroy)(start)); check(GPU(EventDestroy)(end));
    }
    const auto after = x.read();
    require(!std::memcmp(after.data(),input.data(),x.bytes),"input modified");
    check(GPU(StreamDestroy)(stream));
}
static void gpu_test(bool bench) {
    int count=0,device=0; check(GPU(GetDeviceCount)(&count)); require(count>0,"no GPU");
    check(GPU(GetDevice)(&device)); DeviceProp prop{}; check(GPU(GetDeviceProperties)(&prop,device));
#if defined(__HIPCC__)
    const bool target = prop.warpSize == 32 && !std::strncmp(prop.gcnArchName,"gfx1151",7) &&
        (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':');
#else
    const bool target = prop.major == 12 && prop.minor == 1 && prop.integrated;
#endif
    std::printf("GPU %s: target_admitted=%d\n",prop.name,int(target));
    require(!bench || target,"timing requires the production target (GB10/gfx1151)");
    for (uint64_t blocks : {1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,1024u})
        for (unsigned skew : {0u,16u,32u,48u}) run_case(256,blocks,skew,blocks==33,false);
    run_case(1024,32768,0,true,false);
    std::puts("PASS GPU: scalar/old-vector/paired F16 parity, guards, CTA tails, offsets, immutable weights and graph replay.");
    if (bench) {
        run_case(1024,32768,0,true,true);
        std::puts("Kernel-only timing; not prefill TPS or proof of closing the Q4/Q8 gap.");
    }
}
#endif

int main(int argc,char **argv) {
    const bool bench = argc==2 && !std::strcmp(argv[1],"--bench");
    require(argc==1 || bench,"usage: test_q4_prefill_dequant [--bench]");
#if !defined(__HIPCC__) && !defined(__CUDACC__)
    require(!bench,"--bench requires a native CUDA/HIP build and GPU");
#endif
    host_test();
#if defined(__HIPCC__) || defined(__CUDACC__)
    gpu_test(bench);
#else
#if defined(__FLT16_MANT_DIG__)
    host_kernel_test();
#endif
    std::puts("Host-only: CUDA/HIP compilation, numerical parity and speed remain unverified.");
#endif
}
