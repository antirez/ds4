// ds4_mxfp4_cutlass.cu — CUTLASS MXFP4 tensor-core expert FFN for the ds4 MoE (sm_120f).
// Weights arrive pre-packed in CUTLASS B layout (from the offline converter); activations are
// packed to MXFP4 on-device at runtime. Path: pack(x) -> gate/up GEMM -> SwiGLU -> pack(mid) -> down GEMM.
// Build (standalone test):  nvcc -std=c++17 -arch=sm_120f --expt-relaxed-constexpr --expt-extended-lambda
//                           -DDS4_MXFP4_STANDALONE -I cutlass/include -I cutlass/tools/util/include ...
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "ds4_gpu.h"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

// ---- GEMM: W4A8 mixed block-scaled — A=activation MXFP8 (E4M3, RowMajor) x B=weight MXFP4
// (E2M1, ColumnMajor) -> f32 D. This is the native DeepSeek-V4-Flash expert scheme (expert_dtype
// fp4 x activation e4m3, dynamic UE8M0 block scale). The sm120 CollectiveBuilder selects the
// kind::f8f6f4 (MX_F4F6F8) instruction (cute atom SM120_16x8x32_TN<e2m1,e4m3,f32>, mma_sm120.hpp:157).
// The weight-side SF (SFB) swizzle is BYTE-IDENTICAL to the former all-fp4 (mxf4nvf4) config
// (verified: same tile_atom_to_shape_SFB shape/stride/size), so the type-40 weight repack and the
// shipped blk 0-2 stay valid unchanged; only the activation element (fp4->fp8) and its packer change. ----
using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementD   = float;
using ElementC   = float;
using ElementAcc = float;
using LayoutA    = cutlass::layout::RowMajor;
using LayoutB    = cutlass::layout::ColumnMajor;
using LayoutC    = cutlass::layout::RowMajor;
using LayoutD    = cutlass::layout::RowMajor;
constexpr int AlignA = 32, AlignB = 32;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;
using TileShape    = Shape<_128,_128,_128>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, LayoutC, AlignC, ElementD, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignA, ElementB, LayoutB, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF = cutlass::float_ue8m0_t;

// ---- GROUPED (ptr-array) MXFP4 GEMM: one launch runs every active expert's GEMM. ----
// Same element/tile config as the per-expert path above, but the builders take POINTER
// layout tags (LayoutA*, ...), which selects the SM120 blockscaled *array* collective and
// the KernelPtrArrayTmaWarpSpecializedCooperative schedule (KernelScheduleAuto -> cooperative
// for grouped; verified accepted by sm120_blockscaled_mma_builder.inl:226-230). Per-group
// {M,N,K}, A/B/D + SFA/SFB pointer arrays, and per-group strides/SF-layouts are all built on
// DEVICE from the sorted-pairs/offsets buffers -- no host readback, zero host sync.
using GProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using GCollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, LayoutC*, AlignC, ElementD, LayoutD*, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using GCollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA*, AlignA, ElementB, LayoutB*, AlignB, ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename GCollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GGemmKernel = cutlass::gemm::kernel::GemmUniversal<GProblemShape, GCollectiveMainloop, GCollectiveEpilogue>;
using GGemm = cutlass::gemm::device::GemmUniversalAdapter<GGemmKernel>;

using GProbElem  = typename GProblemShape::UnderlyingProblemShape;   // Shape<int,int,int>
using GStrideA   = typename GGemmKernel::InternalStrideA;
using GStrideB   = typename GGemmKernel::InternalStrideB;
using GStrideC   = typename GGemmKernel::InternalStrideC;
using GStrideD   = typename GGemmKernel::InternalStrideD;
using GLayoutSFA = typename GGemmKernel::CollectiveMainloop::InternalLayoutSFA;
using GLayoutSFB = typename GGemmKernel::CollectiveMainloop::InternalLayoutSFB;
using GSm1xxBlkScaledConfig = typename GGemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using GElemA  = typename GGemm::ElementA;
using GElemB  = typename GGemm::ElementB;
using GElemSF = typename GGemmKernel::CollectiveMainloop::ElementSF;
using GElemC  = typename GGemm::ElementC;
using GElemD  = typename GGemm::EpilogueOutputOp::ElementOutput;

// ---- device activation packer: f32 [M,K] RowMajor -> A data (E4M3, 1 byte/elem) + SF (swizzled UE8M0).
// W4A8: activations are MXFP8 (E4M3) with a DYNAMIC per-32-element UE8M0 block scale, matching the
// DeepSeek-V4-Flash source scheme (activation_scheme dynamic, fmt e4m3, scale_fmt ue8m0). The scale
// exponent convention mirrors the engine's cuBLASLt MXFP8 packer (mxfp8_quant_act_kernel): se =
// floor(log2(amax)) - 7, data = v * 2^-se, SF byte = se+127; CUTLASS reconstructs v = data*2^(se-127+127-... )
// i.e. data * 2^(SF-127) = v exactly. The SF is written through the CUTLASS tile-atom SFA layout object
// (identical swizzle to the weight SFB), NOT the cuBLASLt VEC32 swizzle. ----
template<class TSFA>
__global__ void pack_act_e4m3_rowmajor(uint8_t *A_data, TSFA tSFA, const float *act, int M, int K){
  int nblk=K/32; long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*nblk) return;
  int m=(int)(idx/nblk), kb=(int)(idx%nblk);
  const float *x=act+(size_t)m*K+(size_t)kb*32;
  float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(x[i]));
  int se=-127; if(mx>0.f){ int e=(int)floorf(log2f(mx)); se=e-7; }
  if(se<-127)se=-127; if(se>127)se=127;
  float inv=exp2f((float)-se);
  cutlass::float_e4m3_t *outb=reinterpret_cast<cutlass::float_e4m3_t*>(A_data)+(size_t)m*K+(size_t)kb*32;
  for(int i=0;i<32;i++) outb[i]=cutlass::float_e4m3_t(x[i]*inv);
  tSFA(m, kb*32, 0)=ElementSF::bitcast((uint8_t)(se+127));
}

/* Vectorized twin (2026-07-26): the scalar kernel above is one thread per 32-block
 * doing 32 serial scalar LDG.32 for the amax + 32 scalar STG.8 — issue/latency
 * bound (the roofline named it 444 ms = 6.6% of prefill, the biggest dense/MLA
 * sub-cost). This reads the block as 8x float4 (LDG.128) and writes as 2x int4
 * (STG.128). BIT-EXACT to the scalar kernel: one thread still owns the whole
 * 32-block, the amax folds in the SAME sequential order 0..31 (.x/.y/.z/.w per
 * float4), and each element's `cutlass::float_e4m3_t(v*inv)` encode is byte-for-
 * byte the same conversion on the same float. Only the load/store WIDTH changes.
 * Alignment: per-expert K is a 128-multiple, kb*32 floats = 128 B, so
 * act+m*K+kb*32 is 16-B aligned (float4) and the e4m3 out is 16-B aligned (int4).
 * DS4_ACT_PACK_SCALAR=1 selects the scalar path for the bit-exact A/B. */
template<class TSFA>
__global__ void pack_act_e4m3_rowmajor_vec(uint8_t *A_data, TSFA tSFA, const float *act, int M, int K){
  int nblk=K/32; long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(long)M*nblk) return;
  int m=(int)(idx/nblk), kb=(int)(idx%nblk);
  const float4 *x4=reinterpret_cast<const float4*>(act+(size_t)m*K+(size_t)kb*32);
  float4 v[8];
  #pragma unroll
  for(int i=0;i<8;i++) v[i]=x4[i];
  float mx=0.f;
  #pragma unroll
  for(int i=0;i<8;i++){ mx=fmaxf(mx,fabsf(v[i].x)); mx=fmaxf(mx,fabsf(v[i].y));
                        mx=fmaxf(mx,fabsf(v[i].z)); mx=fmaxf(mx,fabsf(v[i].w)); }
  int se=-127; if(mx>0.f){ int e=(int)floorf(log2f(mx)); se=e-7; }
  if(se<-127)se=-127; if(se>127)se=127;
  float inv=exp2f((float)-se);
  cutlass::float_e4m3_t ob[32];
  #pragma unroll
  for(int i=0;i<8;i++){ ob[i*4+0]=cutlass::float_e4m3_t(v[i].x*inv);
                        ob[i*4+1]=cutlass::float_e4m3_t(v[i].y*inv);
                        ob[i*4+2]=cutlass::float_e4m3_t(v[i].z*inv);
                        ob[i*4+3]=cutlass::float_e4m3_t(v[i].w*inv); }
  int4 *outp=reinterpret_cast<int4*>(reinterpret_cast<cutlass::float_e4m3_t*>(A_data)+(size_t)m*K+(size_t)kb*32);
  const int4 *obp=reinterpret_cast<const int4*>(ob);
  outp[0]=obp[0]; outp[1]=obp[1];
  tSFA(m, kb*32, 0)=ElementSF::bitcast((uint8_t)(se+127));
}
// LOSSY dequant->fp4 weight packer still needs the E2M1 nearest-value helper (below); keep it.
__device__ __constant__ float d_kE2M1[16] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
__device__ __forceinline__ uint8_t d_to_e2m1(float v){ float best=1e30f; uint8_t bn=0; for(uint8_t n=0;n<16;n++){ float d=fabsf(v-d_kE2M1[n]); if(d<best){best=d;bn=n;} } return bn; }
// mid = silu(clamp(gate)) * clamp(up) * routing_weight  — matches engine ds4_cuda.cu:10827-10835
__global__ void swiglu_kernel(float *mid, const float *gate, const float *up, const float *w, float clamp, int mid_dim, long n){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  float g=gate[i], u=up[i];
  if(clamp>1.0e-6f){ if(g>clamp)g=clamp; if(u>clamp)u=clamp; if(u<-clamp)u=-clamp; }
  float s=g/(1.f+expf(-g));
  mid[i]=s*u*w[i/mid_dim];
}

// A_data is E4M3: M*K bytes (1 byte/elem), NOT M*K/2. SFA via the CUTLASS tile-atom layout.
static void pack_activation(uint8_t *A_data, ElementSF *A_sf, const float *x, int M, int K){
  auto layoutSF = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 0, K, 1));
  auto tSFA = make_tensor(make_gmem_ptr(A_sf), layoutSF);
  int nb=M*(K/32), t=128, b=(nb+t-1)/t;
  /* Vectorized pack is default; DS4_ACT_PACK_SCALAR=1 forces the scalar twin
   * (bit-exact) for the A/B. Env read ONCE (init-time static), not per launch. */
  static int scalar = -1;
  if (scalar < 0) { const char *e = getenv("DS4_ACT_PACK_SCALAR"); scalar = (e && e[0]=='1') ? 1 : 0; }
  if (scalar) pack_act_e4m3_rowmajor<<<b,t>>>(A_data, tSFA, x, M, K);
  else        pack_act_e4m3_rowmajor_vec<<<b,t>>>(A_data, tSFA, x, M, K);
}

static typename Gemm::Arguments make_gemm_args(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K){
  auto strideA = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M,K,1});
  auto strideB = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto strideC = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M,N,1});
  auto strideD = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M,N,1});
  auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  return typename Gemm::Arguments{
    cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    { reinterpret_cast<const ElementA::DataType*>(A_data), strideA,
      reinterpret_cast<const ElementB::DataType*>(B_data), strideB,
      A_sf, lSFA, B_sf, lSFB },
    { {1.0f, 0.0f}, D, strideC, D, strideD } };   // C=D ptr (beta=0, unused) to keep epilogue happy
}

// Workspace bytes CUTLASS needs for a GEMM of this shape (queried once per distinct
// (M,N,K) at scratch-sizing time; NOT malloc'd here -- caller folds this into one
// scratch allocation so the hot path never touches the CUDA allocator).
static size_t gemm_workspace_bytes(int M, int N, int K){
  auto args = make_gemm_args(nullptr,nullptr,nullptr,nullptr,nullptr,M,N,K);
  return Gemm::get_workspace_size(args);
}

// One MXFP4 GEMM: D[M,N] = A[M,K](act,packed) . B[N,K](weight,packed). A_sf/B_sf = swizzled SF
// buffers. workspace must be at least gemm_workspace_bytes(M,N,K) bytes, caller-owned.
static int run_gemm(float *D, const uint8_t *A_data, const ElementSF *A_sf,
                    const uint8_t *B_data, const ElementSF *B_sf, int M, int N, int K,
                    void *workspace){
  auto args = make_gemm_args(D,A_data,A_sf,B_data,B_sf,M,N,K);
  Gemm gemm;
  /* can_implement is pure host-side argument validation and is deterministic
   * per (M,N,K); the expert loop re-ran it for every expert's every GEMM. Cache
   * validated shapes (single-threaded GPU submission thread; the shape set is
   * tiny -- a few (T,mid,in)/(T,out,mid) combos). */
  {
    static uint64_t seen[64];
    static int n_seen = 0;
    const uint64_t key = ((uint64_t)(uint32_t)M << 42) ^ ((uint64_t)(uint32_t)N << 21) ^ (uint32_t)K;
    bool hit = false;
    for (int i = 0; i < n_seen; i++) if (seen[i] == key) { hit = true; break; }
    if (!hit) {
      if (gemm.can_implement(args)!=cutlass::Status::kSuccess) return 1;
      if (n_seen < 64) seen[n_seen++] = key;
    }
  }
  if (gemm.initialize(args, workspace)!=cutlass::Status::kSuccess) return 2;
  auto st = gemm.run();
  return st==cutlass::Status::kSuccess ? 0 : 3;
}

// ---- Scratch layout shared by the sizing and execution paths (must stay in lock-step). ----
// Everything the FFN needs beyond the weight/activation pointers it's called with: packed
// activation buffers, their SF tables, the three GEMMs' float outputs, and the three GEMMs'
// (usually zero-size, but not guaranteed) CUTLASS workspaces.
struct ds4_cutlass_ffn_scratch_layout {
  size_t xA_off, xSF_off, midA_off, midSF_off, gate_off, up_off, mid_off;
  size_t ws_gate_off, ws_up_off, ws_down_off;
  size_t xSF_n, midSF_n;
  size_t ws_gate_bytes, ws_up_bytes, ws_down_bytes;
  size_t total_bytes;
};

static size_t align_up_bytes(size_t n, size_t a){ return (n + a - 1) / a * a; }

static ds4_cutlass_ffn_scratch_layout cutlass_ffn_scratch_layout(int T, int in_dim, int mid_dim, int out_dim){
  ds4_cutlass_ffn_scratch_layout L{};
  const size_t align = 256;
  size_t off = 0;
  L.xA_off = off; off = align_up_bytes(off + (size_t)T*in_dim, align);   /* E4M3: 1 byte/elem */
  L.xSF_n = (size_t)((T+127)/128*128)*((in_dim/32+3)/4*4);
  L.xSF_off = off; off = align_up_bytes(off + L.xSF_n*sizeof(ElementSF), align);
  L.midA_off = off; off = align_up_bytes(off + (size_t)T*mid_dim, align);  /* E4M3: 1 byte/elem */
  L.midSF_n = (size_t)((T+127)/128*128)*((mid_dim/32+3)/4*4);
  L.midSF_off = off; off = align_up_bytes(off + L.midSF_n*sizeof(ElementSF), align);
  L.gate_off = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.up_off   = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.mid_off  = off; off = align_up_bytes(off + (size_t)T*mid_dim*sizeof(float), align);
  L.ws_gate_bytes = gemm_workspace_bytes(T, mid_dim, in_dim);
  L.ws_up_bytes   = L.ws_gate_bytes;   // identical (M,N,K) shape to gate
  L.ws_down_bytes = gemm_workspace_bytes(T, out_dim, mid_dim);
  L.ws_gate_off = off; off = align_up_bytes(off + L.ws_gate_bytes, align);
  L.ws_up_off   = off; off = align_up_bytes(off + L.ws_up_bytes, align);
  L.ws_down_off = off; off = align_up_bytes(off + L.ws_down_bytes, align);
  L.total_bytes = off;
  return L;
}

// Scratch bytes ds4_cutlass_expert_ffn_scratch() needs for the worst case of T tokens routed
// to a single expert. Callers size one buffer for their layer's (in_dim,mid_dim,out_dim) shape
// at T=max_tokens_per_expert once (e.g. via cuda_tmp_alloc) and reuse it across every expert
// and every layer that shares that shape -- this is deliberately NOT malloc'd per call.
extern "C" size_t ds4_cutlass_expert_ffn_scratch_bytes(int T, int in_dim, int mid_dim, int out_dim){
  return cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim).total_bytes;
}

// Core FFN, no allocation and no synchronization: out[T,out_dim] = down(swiglu(x.Wg^T, x.Wu^T)).Wd^T.
// Weights pre-packed (data+sf) in B layout; x is f32 [T,in_dim]. `scratch` must be at least
// ds4_cutlass_expert_ffn_scratch_bytes(T,in_dim,mid_dim,out_dim) bytes, 256-byte aligned.
// Launches into the caller's current stream (legacy default stream); the caller is responsible
// for ordering/synchronizing against its own subsequent work, same as every other kernel in
// this engine's decode/prefill graphs.
// SF weight pointers are typed as raw bytes at this extern "C" boundary (not ElementSF*) so
// callers outside this TU -- e.g. ds4_cuda_moe.cu, which has no CUTLASS header visibility --
// can call this without depending on CUTLASS types. ElementSF (cutlass::float_ue8m0_t) is a
// 1-byte POD; the reinterpret below is exact.
extern "C" int ds4_cutlass_expert_ffn_scratch(
    float *out, const float *x,
    const uint8_t *Wg_d, const uint8_t *Wg_sf,
    const uint8_t *Wu_d, const uint8_t *Wu_sf,
    const uint8_t *Wd_d, const uint8_t *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim,
    uint8_t *scratch, size_t scratch_bytes){
  ds4_cutlass_ffn_scratch_layout L = cutlass_ffn_scratch_layout(T, in_dim, mid_dim, out_dim);
  if (!scratch || scratch_bytes < L.total_bytes) return -1;
  const ElementSF *Wg_sf_e = reinterpret_cast<const ElementSF*>(Wg_sf);
  const ElementSF *Wu_sf_e = reinterpret_cast<const ElementSF*>(Wu_sf);
  const ElementSF *Wd_sf_e = reinterpret_cast<const ElementSF*>(Wd_sf);
  uint8_t *xA = scratch + L.xA_off;
  ElementSF *xSF = reinterpret_cast<ElementSF*>(scratch + L.xSF_off);
  uint8_t *midA = scratch + L.midA_off;
  ElementSF *midSF = reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  float *gate = reinterpret_cast<float*>(scratch + L.gate_off);
  float *up   = reinterpret_cast<float*>(scratch + L.up_off);
  float *mid  = reinterpret_cast<float*>(scratch + L.mid_off);
  void *ws_gate = L.ws_gate_bytes ? scratch + L.ws_gate_off : nullptr;
  void *ws_up   = L.ws_up_bytes   ? scratch + L.ws_up_off   : nullptr;
  void *ws_down = L.ws_down_bytes ? scratch + L.ws_down_off : nullptr;
  /* Async: blocking cudaMemset serializes the host against ALL in-flight device
   * work -- in the per-expert FFN loop that was 2 full device syncs per expert
   * per rich layer, dominating small-batch verify encode time. The null-stream
   * async memset is still ordered before the pack/GEMM launches below. */
  if (L.xSF_n) cudaMemsetAsync(xSF,0,L.xSF_n*sizeof(ElementSF));
  if (L.midSF_n) cudaMemsetAsync(midSF,0,L.midSF_n*sizeof(ElementSF));
  int rc=0;
  pack_activation(xA,xSF,x,T,in_dim);
  rc|=run_gemm(gate,xA,xSF,Wg_d,Wg_sf_e,T,mid_dim,in_dim,ws_gate);
  rc|=run_gemm(up,  xA,xSF,Wu_d,Wu_sf_e,T,mid_dim,in_dim,ws_up);
  { long n=(long)T*mid_dim; int t=256,b=(int)((n+t-1)/t); swiglu_kernel<<<b,t>>>(mid,gate,up,weights,clamp,mid_dim,n); }
  pack_activation(midA,midSF,mid,T,mid_dim);
  rc|=run_gemm(out, midA,midSF,Wd_d,Wd_sf_e,T,out_dim,mid_dim,ws_down);
  return rc;
}

// ---- Single-projection W4A8 GEMM (for MIXED type-40 + iq2 layers). One logical matmul:
// out[T,out_dim] = x[T,in_dim] . W[out_dim,in_dim]^T, where W is a type-40 CUTLASS-packed MXFP4
// weight for ONE expert (data at W_d, swizzled SFB at W_sf) and x is f32 activations for the T
// tokens routed to that expert (gathered contiguously by the caller). Activations are packed to
// E4M3 (dynamic UE8M0) exactly as the full FFN's projections -- so this projection is bit-for-bit
// the same function the uniform grouped path computes for a single GEMM. No allocation, no sync;
// launches into the caller's current stream. SF pointer typed as raw bytes at the boundary. ----
static size_t proj_scratch_layout(int T, int in_dim, int out_dim,
                                  size_t *xA_off, size_t *xSF_off, size_t *xSF_n, size_t *ws_off, size_t *ws_bytes){
  const size_t a=256; size_t off=0;
  *xA_off=off; off=align_up_bytes(off+(size_t)T*in_dim, a);                 /* E4M3: 1 byte/elem */
  *xSF_n=(size_t)((T+127)/128*128)*((in_dim/32+3)/4*4);
  *xSF_off=off; off=align_up_bytes(off+(*xSF_n)*sizeof(ElementSF), a);
  *ws_bytes=gemm_workspace_bytes(T, out_dim, in_dim);
  *ws_off=off; off=align_up_bytes(off+(*ws_bytes), a);
  return off;
}
extern "C" size_t ds4_cutlass_proj_scratch_bytes(int T, int in_dim, int out_dim){
  size_t a,b,c,d,e; return proj_scratch_layout(T,in_dim,out_dim,&a,&b,&c,&d,&e);
}
extern "C" int ds4_cutlass_proj_scratch(float *out, const float *x,
    const uint8_t *W_d, const uint8_t *W_sf, int T, int in_dim, int out_dim,
    uint8_t *scratch, size_t scratch_bytes){
  size_t xA_off,xSF_off,xSF_n,ws_off,ws_bytes;
  size_t need=proj_scratch_layout(T,in_dim,out_dim,&xA_off,&xSF_off,&xSF_n,&ws_off,&ws_bytes);
  if(!scratch || scratch_bytes<need) return -1;
  uint8_t *xA=scratch+xA_off;
  ElementSF *xSF=reinterpret_cast<ElementSF*>(scratch+xSF_off);
  void *ws=ws_bytes?(void*)(scratch+ws_off):nullptr;
  if(xSF_n) cudaMemsetAsync(xSF,0,xSF_n*sizeof(ElementSF));
  pack_activation(xA,xSF,x,T,in_dim);
  return run_gemm(out, xA, xSF, W_d, reinterpret_cast<const ElementSF*>(W_sf), T, out_dim, in_dim, ws);
}

// ---- extern-C expert FFN (standalone/test convenience): allocates+frees its own scratch and
// synchronizes before returning. The engine never calls this -- it calls the scratch variant
// above with a pre-sized, reused buffer, since a per-call cudaMalloc/cudaFree/cudaDeviceSynchronize
// here is exactly the hot-path cost the engine path exists to avoid. ----
extern "C" int ds4_cutlass_expert_ffn(
    float *out, const float *x,
    const uint8_t *Wg_d, const ElementSF *Wg_sf,
    const uint8_t *Wu_d, const ElementSF *Wu_sf,
    const uint8_t *Wd_d, const ElementSF *Wd_sf,
    const float *weights, float clamp,
    int T, int in_dim, int mid_dim, int out_dim){
  size_t scratch_bytes = ds4_cutlass_expert_ffn_scratch_bytes(T, in_dim, mid_dim, out_dim);
  uint8_t *scratch=nullptr;
  cudaMalloc(&scratch, scratch_bytes);
  int rc = ds4_cutlass_expert_ffn_scratch(out,x,Wg_d,reinterpret_cast<const uint8_t*>(Wg_sf),
                                          Wu_d,reinterpret_cast<const uint8_t*>(Wu_sf),
                                          Wd_d,reinterpret_cast<const uint8_t*>(Wd_sf),
                                          weights,clamp,
                                          T,in_dim,mid_dim,out_dim,scratch,scratch_bytes);
  cudaDeviceSynchronize();
  cudaFree(scratch);
  return rc;
}

// ============================================================================================
// GROUPED MXFP4 PREFILL FFN — single ptr-array grouped GEMM per logical matmul (gate/up/down),
// replacing routed_moe_launch_cutlass's blocking per-expert host loop. All per-group problem
// shapes, pointers, strides and SF-layouts are built on DEVICE from the sorted-pairs/offsets
// buffers; there is NO host readback and NO host sync.
//
// SF-LAYOUT INVARIANT (the highest-risk part): the SM120 blockscaled SF atom spans 128 rows
// (Blk_MN). CUTLASS grouped GEMM reads each group's A-side SF starting at ptr_SFA[g] with its
// own tile_atom_to_shape_SFA(M_g,K) layout, so every group's rows MUST start on a 128-row
// boundary in the packed activation buffer. We therefore GATHER each expert's tokens to a
// 128-padded row offset (padded_offsets[e], a multiple of 128) and pack the whole padded buffer
// with ONE tile_atom_to_shape_SFA(padded_total,K). Because tile_to_shape(..., Step<_2,_1>) makes
// the M-tile the outer (slowest) dimension, M-tile t begins at exactly t*per_Mtile_sf elements,
// so ptr_SFA[e] = SFA_base + (padded_offsets[e]/128)*per_Mtile_sf indexes group e's SF exactly,
// and the per-group tile_atom_to_shape_SFA(count[e],K) view reads its real rows [0,count[e]).
// Getting this offset/stride wrong reads scrambled scales — the standalone self-check below
// exercises a multi-expert problem list against the same double-precision oracle to catch it.
// ============================================================================================

static int grouped_sm_count(){
  static int sc = -1;
  if (sc < 0){ int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); sc = v > 0 ? v : 1; }
  return sc;
}

// Physical SF-A element count for exactly one 128-row M-tile at inner dim K (host + device).
static __host__ __device__ long grouped_per_mtile_sfA(int K){
  return (long)cute::size(cute::filter_zeros(GSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(128, 0, K, 1))));
}

// One thread per expert e in [0,n_total). Builds every device array the grouped GEMM needs for
// ONE logical matmul of constant (N,K) across groups (M_g = counts[e]). ptr_A/ptr_SFA index the
// shared packed-activation buffer at the group's 128-padded row; ptr_B/ptr_SFB index the model
// weight block for expert e; ptr_D indexes the padded output at the same padded row.
__global__ static void g_build_arrays(
    GProbElem *prob,
    const GElemA **ptrA, GStrideA *dA, const GElemSF **ptrSFA, GLayoutSFA *lSFA,
    const GElemB **ptrB, GStrideB *dB, const GElemSF **ptrSFB, GLayoutSFB *lSFB,
    const GElemC **ptrC, GStrideC *dC, GElemD **ptrD, GStrideD *dD,
    const uint32_t *counts, const uint32_t *padded_off,
    const uint8_t *A_data, const uint8_t *A_sf, long per_mtile_sfA,
    const uint8_t *B_base, uint64_t B_stride, uint64_t B_data_bytes,
    GElemD *D_base, int N, int K, int n_total){
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_total) return;
  int M = (int)counts[e];
  uint32_t roff = padded_off[e];
  prob[e] = make_shape(M, N, K);
  dA[e] = cutlass::make_cute_packed_stride(GStrideA{}, cute::make_shape(M, K, 1));
  dB[e] = cutlass::make_cute_packed_stride(GStrideB{}, cute::make_shape(N, K, 1));
  dC[e] = cutlass::make_cute_packed_stride(GStrideC{}, cute::make_shape(M, N, 1));
  dD[e] = cutlass::make_cute_packed_stride(GStrideD{}, cute::make_shape(M, N, 1));
  lSFA[e] = GSm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  lSFB[e] = GSm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  ptrA[e]   = reinterpret_cast<const GElemA*>(A_data + (size_t)roff * K);   /* E4M3 A: K bytes/row */
  ptrSFA[e] = reinterpret_cast<const GElemSF*>(A_sf + (size_t)(roff / 128u) * per_mtile_sfA);
  ptrB[e]   = reinterpret_cast<const GElemB*>(B_base + (size_t)e * B_stride);
  ptrSFB[e] = reinterpret_cast<const GElemSF*>(B_base + (size_t)e * B_stride + B_data_bytes);
  ptrC[e]   = nullptr;                                    // beta = 0, C unused
  ptrD[e]   = D_base + (size_t)roff * N;
}

// Per-group device array set for one logical GEMM shape.
struct GArrays {
  GProbElem   *prob;
  const GElemA **ptrA;  GStrideA *dA;
  const GElemB **ptrB;  GStrideB *dB;
  const GElemSF **ptrSFA; GLayoutSFA *lSFA;
  const GElemSF **ptrSFB; GLayoutSFB *lSFB;
  const GElemC **ptrC;  GStrideC *dC;
  GElemD      **ptrD;   GStrideD *dD;
};

static size_t g_arrays_bytes(int n_total){
  size_t a = 256, off = 0;
  auto add = [&](size_t n){ size_t r = off; off = align_up_bytes(off + n, a); (void)r; return r; };
  add(sizeof(GProbElem)*n_total);
  add(sizeof(GElemA*)*n_total); add(sizeof(GStrideA)*n_total);
  add(sizeof(GElemB*)*n_total); add(sizeof(GStrideB)*n_total);
  add(sizeof(GElemSF*)*n_total); add(sizeof(GLayoutSFA)*n_total);
  add(sizeof(GElemSF*)*n_total); add(sizeof(GLayoutSFB)*n_total);
  add(sizeof(GElemC*)*n_total); add(sizeof(GStrideC)*n_total);
  add(sizeof(GElemD*)*n_total); add(sizeof(GStrideD)*n_total);
  return off;
}

static GArrays g_arrays_place(uint8_t *base, int n_total){
  GArrays g{}; size_t a = 256, off = 0;
  auto add = [&](size_t n)->uint8_t*{ uint8_t *p = base + off; off = align_up_bytes(off + n, a); return p; };
  g.prob   = (GProbElem*)  add(sizeof(GProbElem)*n_total);
  g.ptrA   = (const GElemA**)add(sizeof(GElemA*)*n_total); g.dA = (GStrideA*)add(sizeof(GStrideA)*n_total);
  g.ptrB   = (const GElemB**)add(sizeof(GElemB*)*n_total); g.dB = (GStrideB*)add(sizeof(GStrideB)*n_total);
  g.ptrSFA = (const GElemSF**)add(sizeof(GElemSF*)*n_total); g.lSFA = (GLayoutSFA*)add(sizeof(GLayoutSFA)*n_total);
  g.ptrSFB = (const GElemSF**)add(sizeof(GElemSF*)*n_total); g.lSFB = (GLayoutSFB*)add(sizeof(GLayoutSFB)*n_total);
  g.ptrC   = (const GElemC**)add(sizeof(GElemC*)*n_total); g.dC = (GStrideC*)add(sizeof(GStrideC)*n_total);
  g.ptrD   = (GElemD**)     add(sizeof(GElemD*)*n_total); g.dD = (GStrideD*)add(sizeof(GStrideD)*n_total);
  return g;
}

static size_t grouped_gemm_workspace_bytes(int n_total, int sm_count){
  cutlass::KernelHardwareInfo hw; hw.device_id = 0; hw.sm_count = sm_count;
  typename GGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = {n_total, nullptr, nullptr};
  args.hw_info = hw;
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  return GGemm::get_workspace_size(args);
}

// Launch one grouped blockscaled MXFP4 GEMM over n_total groups (inactive experts carry M=0 and
// contribute zero M-tiles). Deterministic per problem list; caller owns ordering.
static int run_grouped_gemm(int n_total, const GArrays &g, void *workspace, int sm_count){
  cutlass::KernelHardwareInfo hw; hw.device_id = 0; hw.sm_count = sm_count;
  typename GGemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGrouped,
    {n_total, g.prob, nullptr},
    {g.ptrA, g.dA, g.ptrB, g.dB, g.ptrSFA, g.lSFA, g.ptrSFB, g.lSFB},
    {{}, g.ptrC, g.dC, g.ptrD, g.dD},
    hw};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta  = 0.0f;
  GGemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (gemm.initialize(args, workspace) != cutlass::Status::kSuccess) return 2;
  return gemm.run() == cutlass::Status::kSuccess ? 0 : 3;
}

// ---- Scratch layout for the grouped FFN (sizing and execution stay in lock-step). ----
struct ds4_grouped_scratch_layout {
  size_t xA_off, xSF_off, gate_off, up_off, mid_off, midA_off, midSF_off;
  size_t gu_arr_off, dn_arr_off, ws_gu_off, ws_dn_off;
  size_t xSF_bytes, midSF_bytes, ws_bytes, total_bytes;
};

static ds4_grouped_scratch_layout grouped_scratch_layout(int padded_total, int n_total,
                                                         int in_dim, int mid_dim, int out_dim){
  ds4_grouped_scratch_layout L{};
  const size_t a = 256; size_t off = 0;
  int sm = grouped_sm_count();
  int tiles = padded_total / 128;
  L.xA_off = off;  off = align_up_bytes(off + (size_t)padded_total*in_dim, a);   /* E4M3: 1 byte/elem */
  L.xSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(in_dim) * sizeof(ElementSF);
  L.xSF_off = off; off = align_up_bytes(off + L.xSF_bytes, a);
  L.gate_off = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
  L.up_off   = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
  L.mid_off  = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim*sizeof(float), a);
  L.midA_off = off; off = align_up_bytes(off + (size_t)padded_total*mid_dim, a);   /* E4M3: 1 byte/elem */
  L.midSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(mid_dim) * sizeof(ElementSF);
  L.midSF_off = off; off = align_up_bytes(off + L.midSF_bytes, a);
  L.gu_arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total), a);
  L.dn_arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total), a);
  L.ws_bytes = grouped_gemm_workspace_bytes(n_total, sm);
  L.ws_gu_off = off; off = align_up_bytes(off + L.ws_bytes, a);
  L.ws_dn_off = off; off = align_up_bytes(off + L.ws_bytes, a);
  L.total_bytes = off;
  return L;
}

extern "C" size_t ds4_cutlass_grouped_moe_scratch_bytes(
    int padded_total, int n_total_expert, int in_dim, int mid_dim, int out_dim){
  return grouped_scratch_layout(padded_total, n_total_expert, in_dim, mid_dim, out_dim).total_bytes;
}

// Grouped MXFP4 FFN. x_gathered/w_gathered are the per-slot activations gathered to 128-padded
// row offsets (padding rows must be pre-zeroed by the caller). Writes ffn_out[padded_total,out_dim]
// (the caller scatters the real rows into the flat down buffer, then moe_sum reduces). No host sync.
extern "C" int ds4_cutlass_grouped_moe(
    float *ffn_out, const float *x_gathered, const float *w_gathered,
    const uint8_t *gate_w, const uint8_t *up_w, const uint8_t *down_w,
    uint64_t gate_stride, uint64_t gate_data_bytes,
    uint64_t down_stride, uint64_t down_data_bytes,
    float clamp, int n_total_expert,
    int in_dim, int mid_dim, int out_dim,
    const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
    uint8_t *scratch, size_t scratch_bytes){
  ds4_grouped_scratch_layout L = grouped_scratch_layout(padded_total, n_total_expert, in_dim, mid_dim, out_dim);
  if (!scratch || scratch_bytes < L.total_bytes) return -1;
  int sm = grouped_sm_count();
  uint8_t   *xA   = scratch + L.xA_off;
  ElementSF *xSF  = reinterpret_cast<ElementSF*>(scratch + L.xSF_off);
  float     *gate = reinterpret_cast<float*>(scratch + L.gate_off);
  float     *up   = reinterpret_cast<float*>(scratch + L.up_off);
  float     *mid  = reinterpret_cast<float*>(scratch + L.mid_off);
  uint8_t   *midA = scratch + L.midA_off;
  ElementSF *midSF= reinterpret_cast<ElementSF*>(scratch + L.midSF_off);
  GArrays gu = g_arrays_place(scratch + L.gu_arr_off, n_total_expert);
  GArrays dn = g_arrays_place(scratch + L.dn_arr_off, n_total_expert);
  void *ws_gu = L.ws_bytes ? (void*)(scratch + L.ws_gu_off) : nullptr;
  void *ws_dn = L.ws_bytes ? (void*)(scratch + L.ws_dn_off) : nullptr;

  cudaMemsetAsync(xSF, 0, L.xSF_bytes);
  cudaMemsetAsync(midSF, 0, L.midSF_bytes);

  // Pack the whole padded activation buffer once (one global SF layout; per-group slices below).
  pack_activation(xA, xSF, x_gathered, padded_total, in_dim);

  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  long pmt_in  = grouped_per_mtile_sfA(in_dim);
  long pmt_mid = grouped_per_mtile_sfA(mid_dim);

  // gate arrays (D = gate) and up arrays (share A/SFA, D = up) — built as two calls.
  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      gate_w,gate_stride,gate_data_bytes, gate, mid_dim, in_dim, n_total_expert);
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  g_build_arrays<<<bb,bt>>>(gu.prob, gu.ptrA,gu.dA,gu.ptrSFA,gu.lSFA, gu.ptrB,gu.dB,gu.ptrSFB,gu.lSFB,
      gu.ptrC,gu.dC,gu.ptrD,gu.dD, counts,padded_offsets, xA,(const uint8_t*)xSF,pmt_in,
      up_w,gate_stride,gate_data_bytes, up, mid_dim, in_dim, n_total_expert);
  if (run_grouped_gemm(n_total_expert, gu, ws_gu, sm) != 0) return 3;

  { long n = (long)padded_total*mid_dim; int t = 256, b = (int)((n+t-1)/t);
    swiglu_kernel<<<b,t>>>(mid, gate, up, w_gathered, clamp, mid_dim, n); }

  pack_activation(midA, midSF, mid, padded_total, mid_dim);
  g_build_arrays<<<bb,bt>>>(dn.prob, dn.ptrA,dn.dA,dn.ptrSFA,dn.lSFA, dn.ptrB,dn.dB,dn.ptrSFB,dn.lSFB,
      dn.ptrC,dn.dC,dn.ptrD,dn.dD, counts,padded_offsets, midA,(const uint8_t*)midSF,pmt_mid,
      down_w,down_stride,down_data_bytes, ffn_out, out_dim, mid_dim, n_total_expert);
  if (run_grouped_gemm(n_total_expert, dn, ws_dn, sm) != 0) return 3;
  return 0;
}

// ---- Grouped SINGLE-projection W4A8 GEMM (for MIXED type-40 + iq2/q2k layers). ----
// One device-built ptr-array grouped GEMM over the 128-padded gathered activations, for ONE
// logical matmul out[padded_total,out_dim] = x_gathered . W^T (W = type-40 expert weight, data at
// W_base+e*W_stride, swizzled SFB at +W_data_bytes). This is exactly the gate/up/down sub-step of
// ds4_cutlass_grouped_moe factored out: same pack + same g_build_arrays gather order + same
// run_grouped_gemm, so it is bit-identical to the per-expert single-projection path it replaces,
// with NO host readback and NO per-expert sync. Padding rows (padded_offsets leave <128-row gaps)
// must be pre-zeroed by the caller. `ws` is folded into scratch (shape-independent grouped WS). ----
static size_t grouped_proj_layout(int padded_total, int n_total_expert, int in_dim,
                                  size_t *xA_off, size_t *xSF_off, size_t *xSF_bytes,
                                  size_t *arr_off, size_t *ws_off, size_t *ws_bytes) {
  const size_t a = 256; size_t off = 0;
  int tiles = padded_total / 128;
  *xA_off = off;  off = align_up_bytes(off + (size_t)padded_total * in_dim, a);   /* E4M3: 1 byte/elem */
  *xSF_bytes = (size_t)tiles * grouped_per_mtile_sfA(in_dim) * sizeof(ElementSF);
  *xSF_off = off; off = align_up_bytes(off + *xSF_bytes, a);
  *arr_off = off; off = align_up_bytes(off + g_arrays_bytes(n_total_expert), a);
  *ws_bytes = grouped_gemm_workspace_bytes(n_total_expert, grouped_sm_count());
  *ws_off = off;  off = align_up_bytes(off + *ws_bytes, a);
  return off;
}
extern "C" size_t ds4_cutlass_grouped_proj_scratch_bytes(int padded_total, int n_total_expert, int in_dim, int out_dim){
  (void)out_dim;   /* grouped WS is shape-independent; D buffer is caller-owned */
  size_t a,b,c,d,e,f; return grouped_proj_layout(padded_total, n_total_expert, in_dim, &a,&b,&c,&d,&e,&f);
}
extern "C" int ds4_cutlass_grouped_proj(
    float *out, const float *x_gathered,
    const uint8_t *W_base, uint64_t W_stride, uint64_t W_data_bytes,
    int n_total_expert, int in_dim, int out_dim,
    const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
    uint8_t *scratch, size_t scratch_bytes){
  size_t xA_off,xSF_off,xSF_bytes,arr_off,ws_off,ws_bytes;
  size_t need = grouped_proj_layout(padded_total, n_total_expert, in_dim,
                                    &xA_off,&xSF_off,&xSF_bytes,&arr_off,&ws_off,&ws_bytes);
  if (!scratch || scratch_bytes < need) return -1;
  int sm = grouped_sm_count();
  uint8_t   *xA  = scratch + xA_off;
  ElementSF *xSF = reinterpret_cast<ElementSF*>(scratch + xSF_off);
  GArrays g = g_arrays_place(scratch + arr_off, n_total_expert);
  void *ws = ws_bytes ? (void*)(scratch + ws_off) : nullptr;
  cudaMemsetAsync(xSF, 0, xSF_bytes);
  pack_activation(xA, xSF, x_gathered, padded_total, in_dim);
  long pmt = grouped_per_mtile_sfA(in_dim);
  const int bt = 128, bb = (n_total_expert + bt - 1) / bt;
  g_build_arrays<<<bb,bt>>>(g.prob, g.ptrA,g.dA,g.ptrSFA,g.lSFA, g.ptrB,g.dB,g.ptrSFB,g.lSFB,
      g.ptrC,g.dC,g.ptrD,g.dD, counts, padded_offsets, xA,(const uint8_t*)xSF, pmt,
      W_base, W_stride, W_data_bytes, out, out_dim, in_dim, n_total_expert);
  return run_grouped_gemm(n_total_expert, g, ws, sm) == 0 ? 0 : 3;
}

// ---- Small-batch expert FFN via direct fp4 GEMV (spec-decode verify, n_tokens 2..4). ----
// The grouped CUTLASS path costs ~2.8 ms per rich layer at n_tokens=3: per-expert GEMM
// launches at M<=3 run far off roofline, behind a blocking per-layer offsets readback.
// These GEMVs read the packed weights directly: one launch for gate+up+swiglu, one for
// down, no readback, no sort, and f32 activations (no fp4 activation quant -- tighter
// numerics than the GEMM path's fp4 x fp4).
// Data layout (see ds4_cutlass_pack_source): B is ColumnMajor packed E2M1 -- logical
// (n,k) lives at nibble n + k*N, so byte (n + k*N)/2. A thread owning row-pair
// (2p, 2p+1) owns whole bytes, and a warp reads 32 consecutive bytes at each k ->
// coalesced. SF is the swizzled tile-atom layout, indexed with the same layout object
// the packers use (callable on device).

__device__ __constant__ static float kE2M1_GEMV[16] =
    {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ static float gemv_sf_val(uint8_t b) {
  return __int_as_float((uint32_t)b << 23);   /* 2^(e-127) */
}

// v3: the CUTLASS B data section is ROW-MAJOR K-contiguous packed nibbles --
// verified empirically (temp/fp4gemv_test.cu): ds4_cutlass_pack_source's data
// section is an IDENTITY COPY of the source's row-major e2m1 bytes, i.e. logical
// (n,k) lives at nibble k + n*K, byte n*(K/2) + k/2. (pack_weight_f32's manual
// "n + k*N" math -- and its byte-for-byte comment -- does NOT match pack_source;
// v1/v2 of these kernels inherited that assumption and read scrambled weights on
// real models while passing a self-consistent synthetic test.) Row-major K lets
// each warp own one output row with lanes reading consecutive uint32s (8 nibbles)
// along k -- fully coalesced, no k-split, no partial buffers: one launch for
// gate+up+swiglu, one for down. SF stays in the swizzled tile-atom layout,
// indexed with the same layout object the packers use (SF sections of both
// packers agree byte-for-byte).

template <class SFL>
__global__ static void expert_gemv_gu_swiglu_kernel(
    float *mid,               // [n_slots, N]
    const float *xq,          // [n_tokens, K] fp4-roundtripped activations
    const int32_t *sel,       // [n_slots] expert ids
    const float *rw,          // [n_slots] routing weights
    const uint8_t *gate_base, const uint8_t *up_base,
    uint64_t stride, uint64_t data_bytes, SFL sfl, float clampv,
    int n_expert, unsigned n_total, int K, int N) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = kE2M1_GEMV[threadIdx.x];
  __syncthreads();
  const int slot = (int)blockIdx.y;
  const int lane = (int)(threadIdx.x & 31u);
  const int n = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
  if (n >= N) return;
  const int e = sel[slot];
  float *m = mid + (size_t)slot * N;
  if (e < 0 || (unsigned)e >= n_total) { if (lane == 0) m[n] = 0.f; return; }
  const uint8_t *ge = gate_base + (size_t)e * stride;
  const uint8_t *ue = up_base + (size_t)e * stride;
  const uint8_t *gd = ge + (size_t)n * (K / 2);
  const uint8_t *ud = ue + (size_t)n * (K / 2);
  const uint8_t *gsf = ge + data_bytes;
  const uint8_t *usf = ue + data_bytes;
  const float *xt = xq + (size_t)(slot / n_expert) * K;
  float g = 0.f, u = 0.f;
  for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
    const uint32_t wg = *(const uint32_t *)(gd + (k0 >> 1));
    const uint32_t wu = *(const uint32_t *)(ud + (k0 >> 1));
    const float sg = gemv_sf_val(gsf[sfl(n, k0 & ~31, 0)]);
    const float su = gemv_sf_val(usf[sfl(n, k0 & ~31, 0)]);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float xv = xt[k0 + j];
      g += lut[(wg >> (4 * j)) & 0xFu] * sg * xv;
      u += lut[(wu >> (4 * j)) & 0xFu] * su * xv;
    }
  }
  for (int sh = 16; sh > 0; sh >>= 1) {
    g += __shfl_xor_sync(0xffffffffu, g, sh);
    u += __shfl_xor_sync(0xffffffffu, u, sh);
  }
  if (lane == 0) {
    /* swiglu identical to swiglu_kernel above (clamp then silu(gate)*up*rweight) */
    if (clampv > 1.0e-6f) {
      if (g > clampv) g = clampv;
      if (u > clampv) u = clampv;
      if (u < -clampv) u = -clampv;
    }
    m[n] = (g / (1.f + expf(-g))) * u * rw[slot];
  }
}

template <class SFL>
__global__ static void expert_gemv_down_kernel(
    float *down_out,          // [n_slots, N]
    const float *midq,        // [n_slots, K] fp4-roundtripped mid
    const int32_t *sel,       // [n_slots] expert ids
    const uint8_t *down_base,
    uint64_t stride, uint64_t data_bytes, SFL sfl,
    unsigned n_total, int K, int N) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = kE2M1_GEMV[threadIdx.x];
  __syncthreads();
  const int slot = (int)blockIdx.y;
  const int lane = (int)(threadIdx.x & 31u);
  const int n = (int)(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
  if (n >= N) return;
  const int e = sel[slot];
  float *o = down_out + (size_t)slot * N;
  if (e < 0 || (unsigned)e >= n_total) { if (lane == 0) o[n] = 0.f; return; }
  const uint8_t *de = down_base + (size_t)e * stride;
  const uint8_t *dd = de + (size_t)n * (K / 2);
  const uint8_t *dsf = de + data_bytes;
  const float *xt = midq + (size_t)slot * K;
  float a = 0.f;
  for (int k0 = lane * 8; k0 < K; k0 += 32 * 8) {
    const uint32_t w = *(const uint32_t *)(dd + (k0 >> 1));
    const float sc = gemv_sf_val(dsf[sfl(n, k0 & ~31, 0)]);
    #pragma unroll
    for (int j = 0; j < 8; j++)
      a += lut[(w >> (4 * j)) & 0xFu] * sc * xt[k0 + j];
  }
  for (int sh = 16; sh > 0; sh >>= 1) a += __shfl_xor_sync(0xffffffffu, a, sh);
  if (lane == 0) o[n] = a;
}

/* Round-trip activations through the EXACT fp4 quantizer the GEMM path applies
 * (pack_act_rowmajor: per-32-group e=ceil(log2(max/6)) clamped to +-30, nearest
 * e2m1) and hand the dequantized f32 back to the GEMVs. Without this the GEMV
 * computes a numerically different function than the grouped path (f32 vs fp4
 * activations on every rich layer) and greedy output drifts -- the fp4
 * activation quant is part of the model's effective, quality-gated inference
 * function, not an implementation detail. */
__global__ static void fp4_act_roundtrip_kernel(float *xq, const float *x, long nblk32) {
  const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk32) return;
  const float *src = x + b * 32;
  float *dst = xq + b * 32;
  float mx = 0.f;
  for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(src[i]));
  int e = (mx > 0.f) ? (int)ceilf(log2f(mx / 6.f)) : 0;
  if (e < -30) e = -30;
  if (e > 30) e = 30;
  const float s = exp2f((float)e);
  for (int i = 0; i < 32; i++) dst[i] = d_kE2M1[d_to_e2m1(src[i] / s)] * s;
}

/* W4A8 activation round-trip for the decode/small-batch GEMV: quantize f32 -> E4M3 (per-32 dynamic
 * UE8M0 block scale, EXACTLY as pack_act_e4m3_rowmajor) then dequantize back to f32, so the GEMV
 * (f32 act . fp4 weight dot) computes the SAME function as the prefill W4A8 grouped GEMM
 * (E4M3 act x MXFP4 weight). Keeps decode numerics consistent with prefill -- no fp4-vs-E4M3 drift
 * across the prefill/decode boundary, fully source-faithful. */
__global__ static void e4m3_act_roundtrip_kernel(float *xq, const float *x, long nblk32) {
  const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk32) return;
  const float *src = x + b * 32;
  float *dst = xq + b * 32;
  float mx = 0.f;
  for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(src[i]));
  int se = -127; if (mx > 0.f) { int e = (int)floorf(log2f(mx)); se = e - 7; }
  if (se < -127) se = -127; if (se > 127) se = 127;
  const float inv = exp2f((float)-se), s = exp2f((float)se);
  for (int i = 0; i < 32; i++) dst[i] = (float)((cutlass::float_e4m3_t)(src[i] * inv)) * s;
}

/* Persistent activation round-trip buffers, grown on demand and reused across
 * layers/calls -- single GPU-submission thread, same convention as the other
 * static caches. */
static float *g_fp4_gemv_actbuf = nullptr;
static size_t g_fp4_gemv_actbuf_floats = 0;

// Small-batch (n_tokens 2..4) rich-expert FFN over the packed CUTLASS weights.
// down_out gets one pre-weighted FFN result per (token, slot); the caller sums the
// n_expert slices per token (moe_sum_kernel). mid_scratch: [n_tokens*n_expert, mid_dim].
extern "C" int ds4_cutlass_expert_ffn_gemv_small(
    float *down_out, float *mid_scratch, const float *x,
    const int32_t *selected, const float *rweights,
    const uint8_t *gate_w, const uint8_t *up_w, const uint8_t *down_w,
    uint64_t gate_stride, uint64_t gate_data_bytes,
    uint64_t down_stride, uint64_t down_data_bytes,
    float clamp, int n_tokens, int n_expert, unsigned n_total_expert,
    int in_dim, int mid_dim, int out_dim) {
  if (in_dim % 256 || mid_dim % 256 || out_dim % 8) return 1;
  if ((gate_stride & 3u) || (down_stride & 3u)) return 1;   /* uint32 row loads */
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  const size_t xq_floats = (size_t)n_tokens * in_dim;
  const size_t midq_floats = (size_t)n_slots * mid_dim;
  const size_t need = xq_floats + midq_floats;
  if (need > g_fp4_gemv_actbuf_floats) {
    if (g_fp4_gemv_actbuf) cudaFree(g_fp4_gemv_actbuf);
    g_fp4_gemv_actbuf = nullptr;
    if (cudaMalloc(&g_fp4_gemv_actbuf, need * sizeof(float)) != cudaSuccess) {
      g_fp4_gemv_actbuf_floats = 0;
      return 1;
    }
    g_fp4_gemv_actbuf_floats = need;
  }
  float *xq = g_fp4_gemv_actbuf;
  float *midq = xq + xq_floats;
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  {
    const long nb = (long)(xq_floats / 32);
    e4m3_act_roundtrip_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(xq, x, nb);   /* W4A8: E4M3 acts */
  }
  {
    dim3 g((unsigned)((mid_dim + 7) / 8), n_slots);
    expert_gemv_gu_swiglu_kernel<<<g, 256>>>(mid_scratch, xq, selected, rweights,
        gate_w, up_w, gate_stride, gate_data_bytes, sfl_gu, clamp,
        n_expert, n_total_expert, in_dim, mid_dim);
  }
  {
    const long nb = (long)(midq_floats / 32);
    e4m3_act_roundtrip_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(midq, mid_scratch, nb);   /* W4A8: E4M3 acts */
  }
  {
    dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
    expert_gemv_down_kernel<<<g, 256>>>(down_out, midq, selected,
        down_w, down_stride, down_data_bytes, sfl_dn,
        n_total_expert, mid_dim, out_dim);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


/* ---- Single-projection W4A8 GEMV for MIXED type-40 layers at DECODE/small-batch (n<=4). ----
 * Decode is memory-bound; the M=1 CUTLASS tensor-core GEMM wastes launch + pack + TC-underfill
 * overhead. These reuse the lean expert_gemv_* kernels (fp4 weight read directly, dequant via LUT,
 * E4M3-roundtripped f32 activations = same function as the prefill grouped GEMM). One launch over
 * all (token,expert) slots, no per-expert loop, no host sync. `mid`/`down_out` are the pair-layout
 * f32 accumulators the caller composes with the dp4a side. Persistent actbuf grown on demand. */
static int gemv_actbuf_ensure(size_t need_floats) {
  if (need_floats <= g_fp4_gemv_actbuf_floats) return 1;
  if (g_fp4_gemv_actbuf) cudaFree(g_fp4_gemv_actbuf);
  g_fp4_gemv_actbuf = nullptr;
  if (cudaMalloc(&g_fp4_gemv_actbuf, need_floats * sizeof(float)) != cudaSuccess) { g_fp4_gemv_actbuf_floats = 0; return 0; }
  g_fp4_gemv_actbuf_floats = need_floats;
  return 1;
}
/* gate/up W4A8 GEMV -> mid[n_slots,mid_dim] = silu(clamp(gate))*clamp(up)*rw (pair layout). */
extern "C" int ds4_cutlass_gemv_gateup(
    float *mid, const float *x, const int32_t *selected, const float *rweights,
    const uint8_t *gate_w, const uint8_t *up_w, uint64_t gate_stride, uint64_t gate_data_bytes,
    float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim) {
  if (in_dim % 256 || mid_dim % 8 || (gate_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  const size_t xq_floats = (size_t)n_tokens * in_dim;
  if (!gemv_actbuf_ensure(xq_floats)) return 1;
  float *xq = g_fp4_gemv_actbuf;
  auto sfl_gu = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, mid_dim, in_dim, 1));
  { const long nb = (long)(xq_floats / 32);
    e4m3_act_roundtrip_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(xq, x, nb); }
  dim3 g((unsigned)((mid_dim + 7) / 8), n_slots);
  expert_gemv_gu_swiglu_kernel<<<g, 256>>>(mid, xq, selected, rweights, gate_w, up_w,
      gate_stride, gate_data_bytes, sfl_gu, clamp, n_expert, n_total_expert, in_dim, mid_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
/* down W4A8 GEMV -> down_out[n_slots,out_dim] (pair layout, NO routing weight -- applied at gate/up). */
extern "C" int ds4_cutlass_gemv_down(
    float *down_out, const float *mid, const int32_t *selected,
    const uint8_t *down_w, uint64_t down_stride, uint64_t down_data_bytes,
    int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim) {
  if (mid_dim % 256 || out_dim % 8 || (down_stride & 3u)) return 1;
  const unsigned n_slots = (unsigned)(n_tokens * n_expert);
  const size_t midq_floats = (size_t)n_slots * mid_dim;
  if (!gemv_actbuf_ensure(midq_floats)) return 1;
  float *midq = g_fp4_gemv_actbuf;
  auto sfl_dn = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1, out_dim, mid_dim, 1));
  { const long nb = (long)(midq_floats / 32);
    e4m3_act_roundtrip_kernel<<<(unsigned)((nb + 127) / 128), 128>>>(midq, mid, nb); }
  dim3 g((unsigned)((out_dim + 7) / 8), n_slots);
  expert_gemv_down_kernel<<<g, 256>>>(down_out, midq, selected, down_w,
      down_stride, down_data_bytes, sfl_dn, n_total_expert, mid_dim, out_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : 2;
}


// Pack SOURCE-format MXFP4 (separate E2M1 [N,K/2] row-major + E8M0 [N,K/32]) — exactly as the
// DeepSeek-V4-Flash source stores rich experts — into CUTLASS B layout (ColumnMajor packed E2M1
// data + swizzled SFB). Host-side, lossless (copies nibbles+scale verbatim). This is the permanent
// source->CUTLASS packer; nothing consumes ds4's 17-byte format.
extern "C" void ds4_cutlass_pack_source(uint8_t *Bd, ElementSF *Bsf, const uint8_t *e2m1, const uint8_t *e8m0, int N, int K){
  auto lB   = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  auto tB   = make_tensor(recast_ptr<cutlass::float_e2m1_t>(Bd), layB);
  auto tSFB = make_tensor(Bsf, lSFB);
  int nblk=K/32, rowbytes=K/2;
  for(int n=0;n<N;n++) for(int kb=0;kb<nblk;kb++){
    tSFB(n, kb*32, 0) = ElementSF::bitcast(e8m0[(size_t)n*nblk + kb]);   // E8M0 scale
    const uint8_t *row = e2m1 + (size_t)n*rowbytes + kb*16;              // E2M1: 16 bytes = 32 nibbles
    for(int i=0;i<16;i++){ uint8_t byte=row[i]; int k=kb*32+i*2;
      tB(n,k,0)   = cutlass::float_e2m1_t::bitcast(byte & 0xF);
      tB(n,k+1,0) = cutlass::float_e2m1_t::bitcast(byte >> 4); }
  }
}

// Physical element count of the swizzled SF tensor for a weight of shape (N=out, K=in).
extern "C" size_t ds4_cutlass_weight_sf_count(int N, int K){
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  return cute::size(cute::filter_zeros(lSFB));
}

// Packed E2M1 weight-data byte count for a weight of shape (N=out, K=in): 2 nibbles/byte.
extern "C" size_t ds4_cutlass_weight_data_bytes(int N, int K){ return (size_t)N * (size_t)K / 2; }

// ---- Runtime dequant->fp4 weight packer (2-bit prefill path). Quantizes a DEQUANTIZED f32
// weight [N,K] (RowMajor: N rows of K) to MXFP4 on-device (LOSSY) directly into CUTLASS B layout.
// LAYOUT FIX (2026-07-08): the CUTLASS B data section is ROW-MAJOR K-contiguous packed nibbles
// -- logical (n,k) at nibble k + n*K, byte n*(K/2) + k/2 -- verified empirically against
// ds4_cutlass_pack_source, whose data section is an identity copy of the source's row-major
// e2m1 bytes (temp/fp4gemv_test.cu). The previous "(n + k*N)/2 ColumnMajor" math here did NOT
// match pack_source despite the old comment's claim, so the opt-in DS4_MOE_FP4_PREFILL path
// was feeding scrambled weights to the GEMM. One thread per (row, 32-block) owns 16 whole
// output bytes -> no cross-thread nibble RMW race. ----
template<class TSFB>
__global__ void pack_weight_f32_kernel(uint8_t *B_data, TSFB tSFB, const float *W, int N, int K){
  int nblk = K/32;
  long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= (long)N*nblk) return;
  int n = (int)(idx / nblk), kb = (int)(idx % nblk);
  const float *w = W + (size_t)n*K + (size_t)kb*32;
  float mx=0.f;
  for (int i=0;i<32;i++) mx=fmaxf(mx,fabsf(w[i]));
  int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-127)e=-127; if(e>127)e=127;
  float s=exp2f((float)e);
  tSFB(n, kb*32, 0) = ElementSF::bitcast((uint8_t)(e+127));
  uint8_t *ob = B_data + (size_t)n*(K/2) + (size_t)kb*16;
  for (int i=0;i<16;i++){
    uint8_t lo = d_to_e2m1(w[2*i]/s);
    uint8_t hi = d_to_e2m1(w[2*i+1]/s);
    ob[i] = (uint8_t)(lo | (hi<<4));
  }
}

extern "C" void ds4_cutlass_pack_weight_f32(uint8_t *Bd, uint8_t *Bsf, const float *W, int N, int K){
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1));
  auto tSFB = make_tensor(reinterpret_cast<ElementSF*>(Bsf), lSFB);
  int nb = N*(K/32), t=128, b=(nb+t-1)/t;
  pack_weight_f32_kernel<<<b,t>>>(Bd, tSFB, W, N, K);   // legacy default stream
}

#ifdef DS4_MXFP4_REPACK_CLI
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
// Offline converter CLI. Two modes:
//   per-expert: e2m1.bin e8m0.bin N K out_data.bin out_sf.bin
//   stacked   : --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin
//               (in: [ne,N,K/2] + [ne,N,K/32]; out: per expert data||sf, expert-major)
int main(int argc, char **argv){
  if(argc>=8 && strcmp(argv[1],"--stacked")==0){
    int N=atoi(argv[4]), K=atoi(argv[5]), ne=atoi(argv[6]);
    size_t e2each=(size_t)N*(K/2), e8each=(size_t)N*(K/32);
    std::vector<uint8_t> e2(e2each*ne), e8(e8each*ne);
    FILE*f1=fopen(argv[2],"rb"); if(!f1||fread(e2.data(),1,e2.size(),f1)!=e2.size()){ fprintf(stderr,"e2m1 read fail\n"); return 1; } fclose(f1);
    FILE*f2=fopen(argv[3],"rb"); if(!f2||fread(e8.data(),1,e8.size(),f2)!=e8.size()){ fprintf(stderr,"e8m0 read fail\n"); return 1; } fclose(f2);
    size_t sfn=ds4_cutlass_weight_sf_count(N,K);
    FILE*fo=fopen(argv[7],"wb"); if(!fo){ fprintf(stderr,"out open fail\n"); return 1; }
    std::vector<uint8_t> Bd((size_t)N*K/2); std::vector<ElementSF> Bsf(sfn);
    for(int i=0;i<ne;i++){
      std::fill(Bd.begin(),Bd.end(),(uint8_t)0);
      std::fill(Bsf.begin(),Bsf.end(),ElementSF::bitcast(127));
      ds4_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data()+(size_t)i*e2each, e8.data()+(size_t)i*e8each, N, K);
      fwrite(Bd.data(),1,Bd.size(),fo); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fo);
    }
    fclose(fo);
    printf("stacked N=%d K=%d ne=%d -> per-expert data=%zuB sf=%zuB total=%zuB\n",
           N,K,ne, Bd.size(), sfn*sizeof(ElementSF), (size_t)ne*(Bd.size()+sfn*sizeof(ElementSF)));
    return 0;
  }
  if(argc<7){ fprintf(stderr,"usage: %s e2m1.bin e8m0.bin N K out_data.bin out_sf.bin\n       %s --stacked e2m1_stacked.bin e8m0_stacked.bin N K n_expert out_blob.bin\n",argv[0],argv[0]); return 1; }
  int N=atoi(argv[3]), K=atoi(argv[4]);
  size_t e2n=(size_t)N*(K/2), e8n=(size_t)N*(K/32);
  std::vector<uint8_t> e2(e2n), e8(e8n);
  FILE*f1=fopen(argv[1],"rb"); if(!f1||fread(e2.data(),1,e2n,f1)!=e2n){ fprintf(stderr,"e2m1 read fail %s\n",argv[1]); return 1; } fclose(f1);
  FILE*f2=fopen(argv[2],"rb"); if(!f2||fread(e8.data(),1,e8n,f2)!=e8n){ fprintf(stderr,"e8m0 read fail %s\n",argv[2]); return 1; } fclose(f2);
  std::vector<uint8_t> Bd((size_t)N*K/2,0);
  size_t sfn=ds4_cutlass_weight_sf_count(N,K);
  std::vector<ElementSF> Bsf(sfn, ElementSF::bitcast(127));
  ds4_cutlass_pack_source(Bd.data(), Bsf.data(), e2.data(), e8.data(), N, K);
  FILE*fd=fopen(argv[5],"wb"); fwrite(Bd.data(),1,Bd.size(),fd); fclose(fd);
  FILE*fs=fopen(argv[6],"wb"); fwrite(Bsf.data(),sizeof(ElementSF),sfn,fs); fclose(fs);
  printf("packed N=%d K=%d -> data=%zuB sf=%zuB\n",N,K,Bd.size(),sfn*sizeof(ElementSF));
  return 0;
}
#endif

#ifdef DS4_MXFP4_STANDALONE
#include <vector>
#include <random>
#include <cmath>
// host MXFP4 quant (matches device): returns dequantized value array + packs into B ColumnMajor layout.
static const float hE2M1[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f, 0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};
static uint8_t h_nib(float v){ float best=1e30f; uint8_t bn=0; for(uint8_t n=0;n<16;n++){float d=fabsf(v-hE2M1[n]); if(d<best){best=d;bn=n;}} return bn; }
// Pack weight W[out,in] (row-major float) -> B data+SF in CUTLASS ColumnMajor(B) layout for shape (N=out,K=in). Fill dq.
static void host_pack_weight(std::vector<uint8_t>& Bd, std::vector<ElementSF>& Bsf, std::vector<float>& dq,
                             const std::vector<float>& W, int N, int K){
  auto lB  = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N,K,1});
  auto layB = make_layout(make_shape(N,K,1), lB);
  auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(1,N,K,1)); // M unused for SFB shape? use N,K
  Bd.assign((size_t)N*K/2,0); dq.assign((size_t)N*K,0.f);
  Bsf.assign(cute::size(cute::filter_zeros(lSFB)), ElementSF::bitcast(127));
  auto tB  = make_tensor(recast_ptr<cutlass::float_e2m1_t>((uint8_t*)Bd.data()), layB);
  auto tSFB= make_tensor(Bsf.data(), lSFB);
  for(int n=0;n<N;n++) for(int kb=0;kb<K/32;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(W[(size_t)n*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    tSFB(n, kb*32, 0)=ElementSF::bitcast((uint8_t)(e+127));
    for(int i=0;i<32;i++){ int k=kb*32+i; uint8_t nb=h_nib(W[(size_t)n*K+k]/sc); tB(n,k,0)=cutlass::float_e2m1_t::bitcast(nb); dq[(size_t)n*K+k]=hE2M1[nb]*sc; }
  }
}
static void host_quant_act(const std::vector<float>& X, std::vector<float>& dq, int M, int K){
  dq.assign((size_t)M*K,0.f);
  for(int m=0;m<M;m++) for(int kb=0;kb<K/32;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(X[(size_t)m*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    for(int i=0;i<32;i++){ int k=kb*32+i; uint8_t nb=h_nib(X[(size_t)m*K+k]/sc); dq[(size_t)m*K+k]=hE2M1[nb]*sc; }
  }
}
// host: float weight [N,K] -> SOURCE-format arrays (E2M1 [N,K/2] + E8M0 [N,K/32]), same quant as host_pack_weight
static void host_to_source(const std::vector<float>& W, std::vector<uint8_t>& e2, std::vector<uint8_t>& e8, int N, int K){
  int nblk=K/32; e2.assign((size_t)N*(K/2),0); e8.assign((size_t)N*nblk,0);
  for(int n=0;n<N;n++) for(int kb=0;kb<nblk;kb++){
    float mx=0.f; for(int i=0;i<32;i++) mx=fmaxf(mx,fabsf(W[(size_t)n*K+kb*32+i]));
    int e=(mx>0.f)?(int)ceilf(log2f(mx/6.f)):0; if(e<-30)e=-30; if(e>30)e=30; float sc=exp2f((float)e);
    e8[(size_t)n*nblk+kb]=(uint8_t)(e+127);
    uint8_t* row=e2.data()+(size_t)n*(K/2)+kb*16;
    for(int i=0;i<16;i++){ int k=kb*32+i*2; uint8_t lo=h_nib(W[(size_t)n*K+k]/sc), hi=h_nib(W[(size_t)n*K+k+1]/sc); row[i]=(uint8_t)(lo|(hi<<4)); }
  }
}
// Multi-expert grouped-GEMM self-check: several experts, tokens round-robin-routed (n_expert=1),
// run the grouped entry end-to-end (gather to 128-padded rows -> grouped gate/up/down -> scatter)
// and compare per-token output to a double-precision CPU oracle. Catches SF-slice/offset bugs.
static int run_grouped_selfcheck(){
  // E experts total but tokens routed ONLY to a subset -> the rest carry count==0 (M==0 groups),
  // exactly like a real prefill chunk where not every expert is hit. Verifies the grouped kernel
  // tolerates empty groups (else the on-model path would silently fall back to the per-expert loop).
  const int E=8, T=311, in_dim=256, mid_dim=256, out_dim=256; const float clamp=4.0f;
  const int active[4]={1,3,4,6};   // experts 0,2,5,7 stay empty
  std::mt19937 rng(11); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> X((size_t)T*in_dim); for(auto&v:X) v=nd(rng);
  std::vector<std::vector<float>> Wg(E),Wu(E),Wd(E),dqWg(E),dqWu(E),dqWd(E);
  std::vector<std::vector<uint8_t>> Bgd(E),Bud(E),Bdd(E);
  std::vector<std::vector<ElementSF>> Bgs(E),Bus(E),Bds(E);
  for(int e=0;e<E;e++){
    Wg[e].resize((size_t)mid_dim*in_dim); Wu[e].resize((size_t)mid_dim*in_dim); Wd[e].resize((size_t)out_dim*mid_dim);
    for(auto&v:Wg[e])v=nd(rng)*0.3f; for(auto&v:Wu[e])v=nd(rng)*0.3f; for(auto&v:Wd[e])v=nd(rng)*0.3f;
    host_pack_weight(Bgd[e],Bgs[e],dqWg[e],Wg[e],mid_dim,in_dim);
    host_pack_weight(Bud[e],Bus[e],dqWu[e],Wu[e],mid_dim,in_dim);
    host_pack_weight(Bdd[e],Bds[e],dqWd[e],Wd[e],out_dim,mid_dim);
  }
  // routing: round-robin token->expert; per-token routing weight
  std::vector<int> sel(T); std::vector<float> route(T);
  for(int t=0;t<T;t++){ sel[t]=active[t%4]; route[t]=0.5f+std::abs(nd(rng))*0.5f; }
  std::vector<uint32_t> counts(E,0); for(int t=0;t<T;t++) counts[sel[t]]++;
  std::vector<uint32_t> padoff(E); uint32_t run=0; for(int e=0;e<E;e++){ padoff[e]=run; run+=(counts[e]+127u)/128u*128u; }
  const int padded_total=(int)run;
  // gather to padded rows (per-expert running index), record padded row per token for scatter
  std::vector<float> xg((size_t)padded_total*in_dim,0.f), wg((size_t)padded_total,0.f);
  std::vector<int> tok_row(T); std::vector<uint32_t> cur=padoff;
  for(int t=0;t<T;t++){ uint32_t R=cur[sel[t]]++; tok_row[t]=(int)R;
    for(int k=0;k<in_dim;k++) xg[(size_t)R*in_dim+k]=X[(size_t)t*in_dim+k]; wg[R]=route[t]; }
  // oracle (double precision) per token
  std::vector<float> dqX; host_quant_act(X,dqX,T,in_dim);
  std::vector<float> mid((size_t)T*mid_dim), ref((size_t)T*out_dim,0.f), dqMid;
  for(int t=0;t<T;t++){ int e=sel[t];
    for(int j=0;j<mid_dim;j++){ double g=0,u=0; for(int k=0;k<in_dim;k++){ g+=(double)dqX[(size_t)t*in_dim+k]*dqWg[e][(size_t)j*in_dim+k]; u+=(double)dqX[(size_t)t*in_dim+k]*dqWu[e][(size_t)j*in_dim+k]; }
      float gf=(float)g,uf=(float)u; if(clamp>1e-6f){if(gf>clamp)gf=clamp;if(uf>clamp)uf=clamp;if(uf<-clamp)uf=-clamp;} mid[(size_t)t*mid_dim+j]=(gf/(1.f+expf(-gf)))*uf*route[t]; } }
  host_quant_act(mid,dqMid,T,mid_dim);
  for(int t=0;t<T;t++){ int e=sel[t]; for(int o=0;o<out_dim;o++){ double a=0; for(int j=0;j<mid_dim;j++) a+=(double)dqMid[(size_t)t*mid_dim+j]*dqWd[e][(size_t)o*mid_dim+j]; ref[(size_t)t*out_dim+o]=(float)a; } }
  // build device weight blobs [data||sf] per expert, stride = data + sf
  const size_t gdata=(size_t)mid_dim*in_dim/2, gsf=Bgs[0].size()*sizeof(ElementSF);
  const size_t ddata=(size_t)out_dim*mid_dim/2, dsf=Bds[0].size()*sizeof(ElementSF);
  const uint64_t gstride=gdata+gsf, dstride=ddata+dsf;
  std::vector<uint8_t> gate_blob((size_t)E*gstride), up_blob((size_t)E*gstride), down_blob((size_t)E*dstride);
  for(int e=0;e<E;e++){
    memcpy(gate_blob.data()+e*gstride, Bgd[e].data(), gdata); memcpy(gate_blob.data()+e*gstride+gdata, Bgs[e].data(), gsf);
    memcpy(up_blob.data()  +e*gstride, Bud[e].data(), gdata); memcpy(up_blob.data()  +e*gstride+gdata, Bus[e].data(), gsf);
    memcpy(down_blob.data()+e*dstride, Bdd[e].data(), ddata); memcpy(down_blob.data()+e*dstride+ddata, Bds[e].data(), dsf);
  }
  uint8_t *dGate,*dUp,*dDown; cudaMalloc(&dGate,gate_blob.size()); cudaMalloc(&dUp,up_blob.size()); cudaMalloc(&dDown,down_blob.size());
  cudaMemcpy(dGate,gate_blob.data(),gate_blob.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dUp,up_blob.data(),up_blob.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dDown,down_blob.data(),down_blob.size(),cudaMemcpyHostToDevice);
  float *dXg,*dWg2,*dFfn; cudaMalloc(&dXg,xg.size()*4); cudaMalloc(&dWg2,wg.size()*4); cudaMalloc(&dFfn,(size_t)padded_total*out_dim*4);
  cudaMemcpy(dXg,xg.data(),xg.size()*4,cudaMemcpyHostToDevice); cudaMemcpy(dWg2,wg.data(),wg.size()*4,cudaMemcpyHostToDevice);
  uint32_t *dCounts,*dPadoff; cudaMalloc(&dCounts,E*4); cudaMalloc(&dPadoff,E*4);
  cudaMemcpy(dCounts,counts.data(),E*4,cudaMemcpyHostToDevice); cudaMemcpy(dPadoff,padoff.data(),E*4,cudaMemcpyHostToDevice);
  size_t sb=ds4_cutlass_grouped_moe_scratch_bytes(padded_total,E,in_dim,mid_dim,out_dim);
  uint8_t *dScr; cudaMalloc(&dScr,sb);
  int rc=ds4_cutlass_grouped_moe(dFfn,dXg,dWg2,dGate,dUp,dDown,gstride,gdata,dstride,ddata,
        clamp,E,in_dim,mid_dim,out_dim,dCounts,dPadoff,padded_total,dScr,sb);
  cudaDeviceSynchronize();
  std::vector<float> ffn((size_t)padded_total*out_dim); cudaMemcpy(ffn.data(),dFfn,ffn.size()*4,cudaMemcpyDeviceToHost);
  double maxrel=0,maxabs=0; int bad=0;
  for(int t=0;t<T;t++){ const float*o=ffn.data()+(size_t)tok_row[t]*out_dim; const float*r=ref.data()+(size_t)t*out_dim;
    for(int c=0;c<out_dim;c++){ double a=fabs((double)o[c]-r[c]); double rr=a/(fabs(r[c])+1e-3); if(rr>maxrel)maxrel=rr; if(a>maxabs)maxabs=a; if(rr>0.05&&a>0.1)bad++; } }
  printf("grouped(E=%d T=%d padded=%d): rc=%d max_rel=%.5f max_abs=%.4f bad=%d -> %s\n",
         E,T,padded_total,rc,maxrel,maxabs,bad,(rc==0&&bad==0)?"PASS":"FAIL");
  cudaFree(dGate);cudaFree(dUp);cudaFree(dDown);cudaFree(dXg);cudaFree(dWg2);cudaFree(dFfn);cudaFree(dCounts);cudaFree(dPadoff);cudaFree(dScr);
  return (rc==0&&bad==0)?0:1;
}

int main(int argc,char**argv){
  int T=256,in_dim=2048,mid_dim=1408,out_dim=2048;
  if(argc>=5){T=atoi(argv[1]);in_dim=atoi(argv[2]);mid_dim=atoi(argv[3]);out_dim=atoi(argv[4]);}
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,1.f);
  std::vector<float> X((size_t)T*in_dim), Wg((size_t)mid_dim*in_dim), Wu((size_t)mid_dim*in_dim), Wd((size_t)out_dim*mid_dim);
  for(auto&v:X)v=nd(rng); for(auto&v:Wg)v=nd(rng)*0.3f; for(auto&v:Wu)v=nd(rng)*0.3f; for(auto&v:Wd)v=nd(rng)*0.3f;
  // host-pack weights + dequant refs
  std::vector<uint8_t> Wgd,Wud,Wdd; std::vector<ElementSF> Wgsf,Wusf,Wdsf; std::vector<float> dqWg,dqWu,dqWd,dqX;
  host_pack_weight(Wgd,Wgsf,dqWg,Wg,mid_dim,in_dim);
  host_pack_weight(Wud,Wusf,dqWu,Wu,mid_dim,in_dim);
  host_pack_weight(Wdd,Wdsf,dqWd,Wd,out_dim,mid_dim);
  host_quant_act(X,dqX,T,in_dim);
  // validate source packer: float->(E2M1,E8M0)->CUTLASS must equal float->CUTLASS (byte-identical)
  { std::vector<uint8_t> e2,e8; host_to_source(Wg,e2,e8,mid_dim,in_dim);
    std::vector<uint8_t> Bd2(Wgd.size(),0); std::vector<ElementSF> Bsf2(Wgsf.size(),ElementSF::bitcast(127));
    ds4_cutlass_pack_source(Bd2.data(),Bsf2.data(),e2.data(),e8.data(),mid_dim,in_dim);
    int dd=memcmp(Bd2.data(),Wgd.data(),Wgd.size());
    int ds=memcmp(Bsf2.data(),Wgsf.data(),Wgsf.size()*sizeof(ElementSF));
    printf("pack_source(E2M1+E8M0->CUTLASS) check: data=%s sf=%s\n", dd==0?"MATCH":"DIFFER", ds==0?"MATCH":"DIFFER"); }
  // reference FFN using the same quantized values
  std::vector<float> gate((size_t)T*mid_dim),up((size_t)T*mid_dim),mid((size_t)T*mid_dim),dqMid,ref((size_t)T*out_dim,0.f);
  std::vector<float> W_route(T); for(auto&v:W_route) v=0.5f+std::abs(nd(rng))*0.5f;  // per-token routing weight
  float clamp = 4.0f;
  for(int t=0;t<T;t++)for(int j=0;j<mid_dim;j++){ double g=0,u=0; for(int k=0;k<in_dim;k++){ g+=(double)dqX[(size_t)t*in_dim+k]*dqWg[(size_t)j*in_dim+k]; u+=(double)dqX[(size_t)t*in_dim+k]*dqWu[(size_t)j*in_dim+k]; } gate[(size_t)t*mid_dim+j]=(float)g; up[(size_t)t*mid_dim+j]=(float)u; }
  for(int t=0;t<T;t++)for(int j=0;j<mid_dim;j++){ size_t i=(size_t)t*mid_dim+j; float g=gate[i],u=up[i]; if(clamp>1e-6f){if(g>clamp)g=clamp;if(u>clamp)u=clamp;if(u<-clamp)u=-clamp;} float s=g/(1.f+expf(-g)); mid[i]=s*u*W_route[t]; }
  host_quant_act(mid,dqMid,T,mid_dim);
  for(int t=0;t<T;t++)for(int o=0;o<out_dim;o++){ double a=0; for(int j=0;j<mid_dim;j++) a+=(double)dqMid[(size_t)t*mid_dim+j]*dqWd[(size_t)o*mid_dim+j]; ref[(size_t)t*out_dim+o]=(float)a; }
  // device buffers
  float *dX,*dOut,*dWr; cudaMalloc(&dX,X.size()*4); cudaMalloc(&dOut,(size_t)T*out_dim*4); cudaMalloc(&dWr,(size_t)T*4);
  cudaMemcpy(dX,X.data(),X.size()*4,cudaMemcpyHostToDevice);
  cudaMemcpy(dWr,W_route.data(),(size_t)T*4,cudaMemcpyHostToDevice);
  uint8_t *dWg,*dWu,*dWd; ElementSF *dWgs,*dWus,*dWds;
  cudaMalloc(&dWg,Wgd.size()); cudaMalloc(&dWu,Wud.size()); cudaMalloc(&dWd,Wdd.size());
  cudaMalloc(&dWgs,Wgsf.size()*sizeof(ElementSF)); cudaMalloc(&dWus,Wusf.size()*sizeof(ElementSF)); cudaMalloc(&dWds,Wdsf.size()*sizeof(ElementSF));
  cudaMemcpy(dWg,Wgd.data(),Wgd.size(),cudaMemcpyHostToDevice); cudaMemcpy(dWu,Wud.data(),Wud.size(),cudaMemcpyHostToDevice); cudaMemcpy(dWd,Wdd.data(),Wdd.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dWgs,Wgsf.data(),Wgsf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice); cudaMemcpy(dWus,Wusf.data(),Wusf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice); cudaMemcpy(dWds,Wdsf.data(),Wdsf.size()*sizeof(ElementSF),cudaMemcpyHostToDevice);
  int rc=ds4_cutlass_expert_ffn(dOut,dX,dWg,dWgs,dWu,dWus,dWd,dWds,dWr,clamp,T,in_dim,mid_dim,out_dim);
  std::vector<float> got((size_t)T*out_dim); cudaMemcpy(got.data(),dOut,(size_t)T*out_dim*4,cudaMemcpyDeviceToHost);
  double maxrel=0,maxabs=0; int bad=0;
  for(size_t i=0;i<got.size();i++){ double a=fabs((double)got[i]-ref[i]); double r=a/(fabs(ref[i])+1e-3); if(r>maxrel)maxrel=r; if(a>maxabs)maxabs=a; if(r>0.05&&a>0.1)bad++; }
  printf("rc=%d  max_rel=%.5f max_abs=%.4f bad=%d/%zu  -> %s\n", rc,maxrel,maxabs,bad,got.size(), (rc==0&&bad==0)?"PASS":"FAIL");
  int single_fail = (rc==0&&bad==0)?0:1;
  int grouped_fail = run_grouped_selfcheck();
  return (single_fail==0 && grouped_fail==0)?0:1;
}
#endif
