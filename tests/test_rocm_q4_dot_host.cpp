// SPDX-License-Identifier: MIT
// Compile the actual production dot with host FP16/dp4a shims. This tests
// staging and integer/FP32 operand order, not GPU instructions or scheduling.
#include "rocm/ds4_rocm_q4_lds.cuh"
#include <type_traits>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define __device__
#define __forceinline__ inline
struct alignas(16) int4 { int32_t x,y,z,w; };
struct cuda_block_q8_K { float d; int8_t qs[256]; int16_t bsums[16]; };
struct alignas(4) cuda_block_q4_K { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
enum { ROCM_Q4_PREFILL_TOKEN_TILE=8 };
static float dev_f16_to_f32(uint16_t bits) { _Float16 f; std::memcpy(&f,&bits,2); return (float)f; }
static int32_t __dp4a(int32_t a,int32_t b,int32_t acc) {
  for (uint32_t i=0;i<4;i++) acc += (int32_t)(int8_t)((uint32_t)a>>(8*i))*(int32_t)(int8_t)((uint32_t)b>>(8*i));
  return acc;
}
static uint32_t rng=0x486fabd9u;
static uint32_t next() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
static void halfbits(uint16_t &u) { const _Float16 f=(_Float16)((int32_t)(next()%2049)-1024)/(_Float16)32;std::memcpy(&u,&f,2); }
#include "rocm/ds4_rocm_q4_dot.cuh"

static uint64_t test_scale_pairs() {
  uint64_t cases = 0;
  // The final metadata byte ends at the allocation boundary. An odd offset
  // also verifies that the shared memcpy helper imposes no alignment promise
  // beyond the aligned production Q4_K layout tested by test_blocks().
  std::vector<uint8_t> storage(15u, 0xa5u);
  uint8_t *const s = storage.data() + 3u;
  for (uint32_t g = 0u; g < 8u; ++g) {
    const uint32_t limit = g < 4u ? (1u << 16u) : (1u << 24u);
    for (uint32_t packed = 0u; packed < limit; ++packed) {
      const uint8_t a = static_cast<uint8_t>(packed);
      const uint8_t b = static_cast<uint8_t>(packed >> 8u);
      const uint8_t c = static_cast<uint8_t>(packed >> 16u);
      uint32_t scale, minimum;
      if (g < 4u) {
        s[g] = a; s[g + 4u] = b;
        s[g ^ 1u] = static_cast<uint8_t>(~a);
        s[(g + 4u) ^ 1u] = static_cast<uint8_t>(~b);
        scale = a & 63u;
        minimum = b & 63u;
      } else {
        s[g - 4u] = a; s[g] = b; s[g + 4u] = c;
        s[(g - 4u) ^ 1u] = static_cast<uint8_t>(~a);
        s[g ^ 1u] = static_cast<uint8_t>(~b);
        s[(g + 4u) ^ 1u] = static_cast<uint8_t>(~c);
        scale = (c & 15u) | ((a >> 6u) << 4u);
        minimum = (c >> 4u) | ((b >> 6u) << 4u);
      }
      const ds4_rocm_q4_scales::pair result =
        ds4_rocm_q4_scales::load_pair(s, g / 2u);
      const uint32_t shift = (g & 1u) * 8u;
      if (((result.scales >> shift) & 255u) != scale ||
          ((result.minima >> shift) & 255u) != minimum) {
        std::fprintf(stderr, "FAIL paired Q4 metadata g=%u bytes=%06x\n", g, packed);
        std::exit(1);
      }
      ++cases;
    }
  }
  if (storage[0] != 0xa5u || storage[1] != 0xa5u || storage[2] != 0xa5u) {
    std::fputs("FAIL Q4 metadata prefix canary\n", stderr);
    std::exit(1);
  }
  return cases;
}

// Independently enumerate Q4 logical groups, nibble indices and Q8 elements.
// Integer products/sums fit int32 even for the deliberately malformed int16
// sum extrema below: 8 * 63 * 65536 < INT32_MAX. Preserve the reference FP32
// block order; an F16 GEMM approximation would not be an exact oracle.
static float scalar_block(const cuda_block_q4_K &w,
                          const cuda_block_q8_K &x, float acc) {
  int32_t dot=0, mins=0;
  for (unsigned g=0;g<8;g++) {
    const unsigned sc=g<4 ? w.scales[g]&63u :
      (w.scales[g+4]&15u)|((w.scales[g-4]>>6u)<<4u);
    const unsigned mn=g<4 ? w.scales[g+4]&63u :
      (w.scales[g+4]>>4u)|((w.scales[g]>>6u)<<4u);
    int32_t group=0;
    for(unsigned k=0;k<32;k++) {
      const unsigned q=(w.qs[(g/2)*32+k]>>((g%2)*4))&15u;
      group+=(int32_t)q*(int32_t)x.qs[g*32+k];
    }
    dot+=(int32_t)sc*group;
    mins+=(int32_t)mn*((int32_t)x.bsums[g*2]+(int32_t)x.bsums[g*2+1]);
  }
  const float xd=dev_f16_to_f32(w.d), xmin=dev_f16_to_f32(w.dmin), yd=x.d;
  acc += yd*xd*(float)dot - yd*xmin*(float)mins;
  return acc;
}

static unsigned test_blocks() {
  unsigned cases=0;
  for (uint32_t trial=0;trial<1600;trial++) {
    cuda_block_q4_K w[8];
    cuda_block_q8_K packed[8][8];
    alignas(16) ds4_rocm_q4_lds::aligned_q8_K aligned[8][8];
    for (auto &weight:w) {
      halfbits(weight.d);halfbits(weight.dmin);
      if (trial%17==0) weight.d=0x7bffu; // maximum finite half
      if (trial%13==0) weight.dmin=0x0001u; // half subnormal
      for (auto &s:weight.scales) s=(uint8_t)next();
      for (auto &q:weight.qs) q=(uint8_t)next();
    }
    for (auto &token:packed) for (auto &block:token) {
      block.d=(float)((int32_t)(next()%4097)-2048)/512.0f;
      for (auto &q:block.qs) q=trial%19==0?-128:trial%23==0?127:(int8_t)next();
      for (uint32_t s=0;s<16;s++) {
        int32_t sum=0; for(uint32_t i=0;i<16;i++) sum+=block.qs[s*16+i];
        block.bsums[s]=trial%29==0 ? INT16_MIN : trial%31==0 ? INT16_MAX : (int16_t)sum;
      }
    }
    for (uint32_t n=1;n<=8;n++) {
      std::memset(aligned,0xab,sizeof(aligned));
      for(uint32_t tid=0;tid<256;tid++) ds4_rocm_q4_lds::copy_thread_aligned<8>(
        reinterpret_cast<uint32_t*>(aligned), reinterpret_cast<const uint32_t*>(packed),tid,256,n,8,8*73);
      float a[8],b[8],ref[8];
      for(uint32_t p=0;p<8;p++) a[p]=b[p]=ref[p]=(float)((int32_t)(next()%4097)-2048)/256.0f;
      for (uint32_t lane=0;lane<8;lane++) {
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w+lane,packed[0]+lane,packed[1]+lane,packed[2]+lane,packed[3]+lane,packed[4]+lane,packed[5]+lane,packed[6]+lane,packed[7]+lane,n,a);
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w+lane,aligned[0]+lane,aligned[1]+lane,aligned[2]+lane,aligned[3]+lane,aligned[4]+lane,aligned[5]+lane,aligned[6]+lane,aligned[7]+lane,n,b);
        for(uint32_t p=0;p<n;p++) ref[p]=scalar_block(w[lane],packed[p][lane],ref[p]);
        if(std::memcmp(a,b,sizeof(a)) || std::memcmp(a,ref,sizeof(a))) {
          std::fprintf(stderr,"FAIL trial=%u n=%u lane=%u\n",trial,n,lane);std::exit(1);
        }
        cases++;
      }
    }
  }
  return cases;
}

namespace lds = ds4_rocm_q4_lds;
using token_acc = std::array<float, 8>;

static void require(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "Q4 ROWS32/64 host: FAIL %s\n", what);
    std::exit(1);
  }
}

// Model the synchronous width-eight shfl_down stages in quarter_warp_sum_f32.
// Each stage reads a snapshot, including the inactive values of other lanes.
// Only lane zero is eventually stored by the production kernel.
static float reduced_lane0(token_acc lanes) {
  for (unsigned offset : {4u, 2u, 1u}) {
    const token_acc before = lanes;
    for (unsigned lane = 0; lane < 8; ++lane)
      lanes[lane] = before[lane] + before[lane + offset < 8 ? lane + offset : lane];
  }
  return lanes[0];
}

template<uint32_t ROWS>
static std::vector<float> tile8_outputs(
    const cuda_block_q4_K *weights, const cuda_block_q8_K *x,
    uint32_t m, uint32_t n, uint32_t group, uint64_t x_stride,
    uint64_t out_stride, uint64_t out_start, const std::vector<float> &canary) {
  using Geometry = lds::tile_geometry<ROWS>;
  std::vector<float> out = canary;
  std::vector<unsigned> stores(out.size(), 0);
  for (uint32_t row_tile = 0; row_tile < (m + ROWS - 1u) / ROWS; ++row_tile)
  for (uint32_t tok0 = 0; tok0 < n; tok0 += 8u) {
    const uint32_t nt = std::min(8u, n - tok0);
    std::vector<token_acc> acc(Geometry::threads, token_acc{});
    alignas(16) lds::aligned_q8_K staged[8][8];
    for (uint32_t b0 = 0; b0 < 32u; b0 += 8u) {
      std::memset(staged, 0xab, sizeof(staged));
      const cuda_block_q8_K *src = x + (uint64_t)tok0 * x_stride + group * 32u + b0;
      for (uint32_t tid = 0; tid < Geometry::threads; ++tid)
        lds::copy_thread_aligned<8>(reinterpret_cast<uint32_t *>(staged),
          reinterpret_cast<const uint32_t *>(src), tid, Geometry::threads,
          nt, 8u, x_stride * 73u);
      // All writers finish before any dot consumes the shared tile.
      for (uint32_t tid = 0; tid < Geometry::threads; ++tid) {
        const uint32_t row = Geometry::row(row_tile, tid);
        const uint32_t lane = Geometry::lane(tid);
        if (row >= m) continue;
        const cuda_block_q4_K *w = weights + ((uint64_t)group * m + row) * 32u + b0 + lane;
        rocm_dot_q4_K_q8_K_block8_reuse_weights(w,
          staged[0]+lane, staged[1]+lane, staged[2]+lane, staged[3]+lane,
          staged[4]+lane, staged[5]+lane, staged[6]+lane, staged[7]+lane,
          nt, acc[tid].data());
      }
    }
    for (uint32_t tid = 0; tid < Geometry::threads; ++tid) {
      const uint32_t row = Geometry::row(row_tile, tid);
      if (row >= m || Geometry::lane(tid) != 0u) continue;
      for (uint32_t p = 0; p < nt; ++p) {
        token_acc lanes;
        for (uint32_t lane = 0; lane < 8u; ++lane) lanes[lane] = acc[tid + lane][p];
        const uint64_t index = out_start + (uint64_t)(tok0 + p) * out_stride + group * m + row;
        require(index < out.size() && stores[index]++ == 0u,
                "output bounds and unique lane-zero stores");
        out[index] = reduced_lane0(lanes);
      }
    }
  }
  for (uint32_t tok = 0; tok < n; ++tok)
  for (uint32_t row = 0; row < m; ++row)
    require(stores[out_start + (uint64_t)tok * out_stride + group * m + row] == 1u,
            "every active output is stored once");
  return out;
}

static unsigned test_row_tiles() {
  unsigned cases = 0;
  for (uint32_t m : {63u, 64u, 65u})
  for (uint32_t tail = 1u; tail <= 8u; ++tail)
  for (uint32_t groups : {1u, 3u}) {
    const uint32_t n = 8u + tail, group = groups - 1u;
    const uint64_t x_start = 3u, w_start = 5u, out_start = 11u;
    const uint64_t x_stride = groups * 32u + 5u;
    const uint64_t out_stride = groups * m + 3u;
    // Exact last source/weight ends make an invalid tail load visible to ASan.
    // Offsets and inter-token gaps also expose accidentally contiguous reads.
    std::vector<cuda_block_q4_K> weights(w_start + groups * m * 32u);
    std::vector<cuda_block_q8_K> packed(x_start + (n - 1u) * x_stride + groups * 32u);
    for (auto &w : weights) {
      halfbits(w.d); halfbits(w.dmin);
      for (auto &s : w.scales) s = (uint8_t)next();
      for (auto &q : w.qs) q = (uint8_t)next();
    }
    for (auto &x : packed) {
      x.d = (float)((int32_t)(next() % 4097u) - 2048) / 512.0f;
      for (auto &q : x.qs) q = (int8_t)next();
      for (uint32_t s = 0; s < 16u; ++s) {
        int32_t sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += x.qs[s * 16u + i];
        x.bsums[s] = (int16_t)sum;
      }
    }
    const auto weights_before = weights;
    const auto packed_before = packed;
    std::vector<float> canary(out_start + n * out_stride + 13u);
    for (size_t i = 0; i < canary.size(); ++i) {
      const uint32_t bits = 0x4e000000u + (uint32_t)i;
      std::memcpy(&canary[i], &bits, sizeof(bits));
    }
    const auto out32 = tile8_outputs<32u>(weights.data() + w_start, packed.data() + x_start,
      m, n, group, x_stride, out_stride, out_start, canary);
    auto expected = canary;
    for (uint32_t tok = 0; tok < n; ++tok)
    for (uint32_t row = 0; row < m; ++row) {
      token_acc lanes{};
      // Independent scalar enumeration: lane L accumulates L,L+8,L+16,L+24.
      // Do not replace this with a sequential sum over all 32 K blocks.
      for (uint32_t lane = 0; lane < 8u; ++lane)
      for (uint32_t b = lane; b < 32u; b += 8u)
        lanes[lane] = scalar_block(weights[w_start + (group * m + row) * 32u + b],
          packed[x_start + (uint64_t)tok * x_stride + group * 32u + b], lanes[lane]);
      expected[out_start + (uint64_t)tok * out_stride + group * m + row] = reduced_lane0(lanes);
    }
    const size_t bytes = expected.size() * sizeof(float);
    if (std::memcmp(out32.data(), expected.data(), bytes)) {
      std::fprintf(stderr, "FAIL ROWS32/scalar m=%u n=%u groups=%u\n", m, n, groups);
      std::exit(1);
    }
    require(std::memcmp(weights.data(), weights_before.data(), weights.size() * sizeof(weights[0])) == 0 &&
            std::memcmp(packed.data(), packed_before.data(), packed.size() * sizeof(packed[0])) == 0,
            "weight and activation sources remain immutable");
    ++cases;
  }
  return cases;
}

int main() {
  const uint64_t scale_pairs = test_scale_pairs();
  const unsigned blocks = test_blocks();
  const unsigned tiles = test_row_tiles();
  std::printf("PASS paired Q4 metadata: %llu exhaustive group cases "
              "(all contributing bytes, both pair positions, unaligned exact-end view).\n",
              static_cast<unsigned long long>(scale_pairs));
  std::printf("PASS production dot packed/aligned: %u block cases, %u ROWS32/scalar K8192 cases "
              "(M63/64/65, N tails 1..8, offsets, group strides, output canaries).\n", blocks, tiles);
  std::puts("Host-only: HIP compilation, GPU shuffle/barrier parity and timing remain required.");
}
