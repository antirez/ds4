// Recursive Flint decode megakernel (fork-owned).
//
// Ported from rec 0.1.1 (Recursive, Apache License 2.0; see
// licenses/Apache-2.0.txt and licenses/rec-NOTICE.txt): the resolved
// `flint_decode_fast_activation` shader of rec_mlx/kernels/megakernel.py with
// the M4 Pro profile's transforms applied (36 workers, 32 expert-team
// workers, 32 head workers, KV teams, own-rank routing, mixed-bits rounding,
// fast activation).  docs/RECURSIVE_FLINT.md describes the port; the trace
// test (tests/flint/trace_compare.py) requires its residual stream to be
// bit-identical to rec's.
//
// Deliberate differences from rec's shader:
//   * one model buffer instead of 13 banks: every descriptor offset is in
//     256-byte units (AT()), which is what lets a 32-bit word address the
//     whole 22.6 GB GGUF (all tensors are 256-byte aligned);
//   * routed experts are addressed through a per-layer table of 256 GPU
//     addresses, one per expert pack (gate/up/down with scales and biases
//     contiguous); the table is the only interface to SSD streaming;
//   * Flint has no router expert bias (config router_expert_bias false), so
//     the zero bias add is gone; the zero add was exact, so nothing changes;
//   * no profiling buffer and no `configuration` input.
//
// This file is compiled on its own with MTLMathModeSafe (rec compiles with
// math_mode "safe"), never into ds4's fast-math library.

#include <metal_stdlib>
using namespace metal;

typedef bfloat bfloat16_t;

// MLX's kernel environment (bf16_math.h) overloads the math functions for
// bfloat so that abs and exp of a bfloat return a bfloat, rounded per
// operation; the activations below rely on exactly that.  The same
// instantiations, for the two functions this kernel applies to bfloat.
#define flint_bf16_math(mfast)                                                   \
    METAL_FUNC bfloat16_t abs(bfloat16_t x) {                                    \
        return static_cast<bfloat16_t>(__metal_fabs(static_cast<float>(x), mfast)); \
    }                                                                            \
    METAL_FUNC bfloat16_t exp(bfloat16_t x) {                                    \
        return static_cast<bfloat16_t>(__metal_exp(static_cast<float>(x), mfast)); \
    }
namespace metal {
flint_bf16_math(__METAL_MAYBE_FAST_MATH__)
namespace fast {
flint_bf16_math(__METAL_FAST_MATH__)
}
namespace precise {
flint_bf16_math(__METAL_PRECISE_MATH__)
}
}
#undef flint_bf16_math

// M4 Pro launch envelope.  Other profiles change these together.
#define FLINT_WORKERS 36u
#define FLINT_GROUPS (FLINT_WORKERS * 8u)         // 288 simdgroups
#define FLINT_THREADS (FLINT_WORKERS * 256u)      // 9216 threads
#define FLINT_MLP_WORKERS 32u
#define FLINT_HEAD_WORKERS 32u
#define FLINT_KV_TEAMS 4u
// Tokens of K and V each lane keeps in flight ahead of the running token in
// the partial attention path (one 8-byte vector of K and one of V per lane
// per token): 1 (rec's), 2 or 4; the host sets it from DS4_FLINT_KV_PREFETCH.
// The loads do not change the arithmetic, so every depth is bit-identical;
// measured (docs/RECURSIVE_FLINT.md, step 3) depths 2 and 4 are no faster,
// the loop is issue-bound, not latency-bound.
#ifndef FLINT_KV_PREFETCH
#define FLINT_KV_PREFETCH 1
#endif
#if FLINT_KV_PREFETCH!=1 && FLINT_KV_PREFETCH!=2 && FLINT_KV_PREFETCH!=4
#error "FLINT_KV_PREFETCH must be 1, 2 or 4"
#endif
#define FLINT_KV_TEAM_SIZE 9u

// One routed expert pack (gguf-tools/flint_int4_to_gguf.py, flint.expert_pack.offsets).
#define FLINT_PACK_GATE_W 0u
#define FLINT_PACK_GATE_S 524288u
#define FLINT_PACK_GATE_B 557056u
#define FLINT_PACK_UP_W 589824u
#define FLINT_PACK_UP_S 1114112u
#define FLINT_PACK_UP_B 1146880u
#define FLINT_PACK_DOWN_W 1179648u
#define FLINT_PACK_DOWN_S 1703936u
#define FLINT_PACK_DOWN_B 1736704u

#define AT(off) (bank + ((size_t)(off) << 8))

struct KVView {
    device bfloat16_t *p0, *p1, *p2, *p3;
    uint offset, cap;
    bool global;
    KVView head(uint index, uint dim) const {
        KVView view=*this; view.offset+=index*cap*128+dim; return view;
    }
    KVView value() const {
        KVView view=*this; view.offset+=cap*512; return view;
    }
    device bfloat16_t* address(uint i) const {
        uint block=global?(i/128)>>16:0;
        uint local=global?i%(65536u*128u):i;
        device bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;
        return p+offset+local;
    }
    bfloat16_t operator[](uint i) const { return *address(i); }
    void store(uint head, uint position, uint dim, bfloat16_t value) const {
        *(this->head(head,dim).address(position*128))=value;
    }
};

inline bool sync_all(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,uint group) {
    (void)group;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*FLINT_WORKERS;
        atomic_fetch_add_explicit(control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        ready[0]=uint(atomic_load_explicit(control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;
}

struct Matrix {
    const device uchar* w;
    const device bfloat16_t* s;
    const device bfloat16_t* b;
    uint n, k;
};
inline Matrix weight_matrix(const device uchar* bank, const device uint* d) {
    return {AT(d[0]), (const device bfloat16_t*)AT(d[1]),
            (const device bfloat16_t*)AT(d[2]),d[3],d[4]};
}
// Exact immutable layer metadata, cooperatively copied once per group.
inline Matrix weight_matrix(const device uchar* bank,const threadgroup uint* d) {
    return {AT(d[0]),(const device bfloat16_t*)AT(d[1]),
            (const device bfloat16_t*)AT(d[2]),d[3],d[4]};
}
// One routed expert's gate (0), up (1) or down (2) matrix inside its pack.
inline Matrix expert_matrix(const device uchar* pack, uint which) {
    if(which==0) return {pack+FLINT_PACK_GATE_W,(const device bfloat16_t*)(pack+FLINT_PACK_GATE_S),
                         (const device bfloat16_t*)(pack+FLINT_PACK_GATE_B),512,2048};
    if(which==1) return {pack+FLINT_PACK_UP_W,(const device bfloat16_t*)(pack+FLINT_PACK_UP_S),
                         (const device bfloat16_t*)(pack+FLINT_PACK_UP_B),512,2048};
    return {pack+FLINT_PACK_DOWN_W,(const device bfloat16_t*)(pack+FLINT_PACK_DOWN_S),
            (const device bfloat16_t*)(pack+FLINT_PACK_DOWN_B),2048,512};
}
inline float load16(const device bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
        v[i]=x[i];v[i+1]=x[i+1]/16.0f;
        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;
    }
    return sum;
}
inline float load16_local(const threadgroup bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        sum+=x[i]+x[i+1]+x[i+2]+x[i+3];
        v[i]=x[i];v[i+1]=x[i+1]/16.0f;
        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;
    }
    return sum;
}
inline float dot16(const device uchar* w,thread float* x,float s,float b,float sum) {
    const device ushort* p=(const device ushort*)w;
    float a=0.0f;
    for(uint j=0;j<4;++j)
        a+=x[4*j]*(p[j]&15u)+x[4*j+1]*(p[j]&240u)
           +x[4*j+2]*(p[j]&3840u)+x[4*j+3]*(p[j]&61440u);
    return s*a+sum*b;
}
inline float dot16_w197_bias23(const device uchar* w,thread float* x,float s,float b,float sum) {
    const device ushort* p=(const device ushort*)w;
    float a=0.0f;
    for(uint j=0;j<4;++j)
        a+=x[4*j]*(as_type<float>(0x4b000000u|(uint(p[j])&15u))-8388608.0f)+x[4*j+1]*(as_type<float>(0x4b000000u|(uint(p[j])&240u))-8388608.0f)
           +x[4*j+2]*(as_type<float>(0x4b000000u|(uint(p[j])&3840u))-8388608.0f)+x[4*j+3]*(as_type<float>(0x4b000000u|(uint(p[j])&61440u))-8388608.0f);
    return s*a+sum*b;
}
template<uint K>
inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
template<uint K>
inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16_local(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
template<uint K>
inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint first) {
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        float v[16];float sum=load16_local(x+scalar,v);
        for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
        }
    }
    for(uint r=0;r<4;++r) {
        float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
    }
}
inline void head_norm(device bfloat16_t* x,const device bfloat16_t* w,uint heads,
                      bool rotary,uint position,float logbase,float eps,uint lane,uint sg) {
    for(uint h=sg;h<heads;h+=FLINT_GROUPS) {
        device bfloat16_t* p=x+h*128;
        float squared=0.0f;for(uint j=0;j<4;++j){float v=p[lane*4+j];squared+=v*v;}
        float inverse=metal::precise::rsqrt(simd_sum(squared)/128.0f+eps);
        if(rotary) {
            for(uint j=0;j<2;++j) {
                uint pair=lane*2+j;
                bfloat16_t a=w[pair]*bfloat16_t(float(p[pair])*inverse);
                bfloat16_t b=w[pair+64]*bfloat16_t(float(p[pair+64])*inverse);
                float theta=float(position)*metal::exp2(-float(pair)/64.0f*logbase);
                float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
                p[pair]=bfloat16_t(float(a)*c-float(b)*s);
                p[pair+64]=bfloat16_t(float(a)*s+float(b)*c);
            }
        } else for(uint j=0;j<4;++j){uint d=lane*4+j;p[d]=w[d]*bfloat16_t(float(p[d])*inverse);}
    }
}
inline uint partial_offset(uint head,uint part,uint dim,uint parts,bool rounded) {
    return rounded?(head*32+dim/4)*parts*4+part*4+dim%4:(head*parts+part)*128+dim;
}

template<bool ROUNDED>
inline void attention_merge_vec4_w2(device bfloat16_t* out,
                      device float* partial,device float* stats,
                      uint parts,uint lane,uint sg) {
    // A SIMD owns 16 output dimensions. Every scalar retains its original
    // per-lane partition order and simd_sum; loads follow native four-wide tiles.
    for(uint task=(sg/72)*64+sg%72;task<(sg/72+1)*64;task+=72) {
        uint h=task/8,dim=(task%8)*16;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);
        float z=0.0f;
        float4 a[4]={float4(0),float4(0),float4(0),float4(0)};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;
            float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint tile=0;tile<4;++tile) {
                float4 value;
                if constexpr(ROUNDED) {
                    uint at=((h*32+dim/4+tile)*parts+p)*4;
                    vec<bfloat16_t,4> raw=*((const device vec<bfloat16_t,4>*)(((const device bfloat16_t*)partial)+at));
                    value=float4(float(raw[0]),float(raw[1]),float(raw[2]),float(raw[3]));
                } else {
                    uint at=(h*parts+p)*128+dim+tile*4;
                    value=*((const device float4*)(partial+at));
                }
                a[tile]+=factor*value;
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint tile=0;tile<4;++tile)for(uint component=0;component<4;++component) {
            uint j=tile*4+component;
            float total=simd_sum(a[tile][component]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<8)for(uint t=0;t<2;++t) {
            uint j=lane*2+t;
            out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);
        }
    }
}

template<uint HEADS,bool ROUNDED>
inline void attention_partial_path_w53(const device bfloat16_t* q,KVView k,
                      KVView v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool runtime_rounded,uint lane,uint sg) {
    (void)out;(void)cap;(void)runtime_rounded;
    constexpr bool rounded=ROUNDED;
    for(uint task=(sg/72)*((8/HEADS)*parts)+sg%72;task<(sg/72+1)*((8/HEADS)*parts);task+=72) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*4);
        KVView value_base=v.head(kv,lane*4);
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            uint at=uint(token)*128;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        // FLINT_KV_PREFETCH tokens in flight per lane: slot i holds token+i*parts
        // (named slots expanded by the preprocessor: an indexed ring, even fully
        // unrolled, went to private memory and ran 2 to 4x slower).
        constexpr uint D=FLINT_KV_PREFETCH;
        bfloat16_t nk0[4],nv0[4],nk1[4],nv1[4],nk2[4],nv2[4],nk3[4],nv3[4];
#define FLINT_KV_LOAD(K,V,T) {uint t_=(T);if(t_<count){ \
            vec<bfloat16_t,4> k_=*(const device vec<bfloat16_t,4>*)key_base.address(t_*128); \
            vec<bfloat16_t,4> v_=*(const device vec<bfloat16_t,4>*)value_base.address(t_*128); \
            for(uint j=0;j<4;++j){K[j]=k_[j];V[j]=v_[j];}}}
#define FLINT_KV_STEP(K,V) if(token<count) { \
            float key[4];bfloat16_t value[4]; \
            for(uint j=0;j<4;++j){key[j]=float(K[j]);value[j]=V[j];} \
            FLINT_KV_LOAD(K,V,token+D*parts); \
            float scores[HEADS]; \
            for(uint h=0;h<HEADS;++h) { \
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j]; \
                scores[h]=simd_sum(score); \
            } \
            float old=0.0f,add=0.0f; \
            if(lane<HEADS) { \
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi)); \
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low; \
                z=z*old+add;hi=next; \
            } \
            for(uint h=0;h<HEADS;++h) { \
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h)); \
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]); \
            } \
            token+=parts; \
        }
        FLINT_KV_LOAD(nk0,nv0,token);
#if FLINT_KV_PREFETCH>=2
        FLINT_KV_LOAD(nk1,nv1,token+parts);
#endif
#if FLINT_KV_PREFETCH>=4
        FLINT_KV_LOAD(nk2,nv2,token+2*parts);FLINT_KV_LOAD(nk3,nv3,token+3*parts);
#endif
        #pragma clang loop unroll_count(1)
        for(;token<count;) {
            FLINT_KV_STEP(nk0,nv0)
#if FLINT_KV_PREFETCH>=2
            FLINT_KV_STEP(nk1,nv1)
#endif
#if FLINT_KV_PREFETCH>=4
            FLINT_KV_STEP(nk2,nv2)FLINT_KV_STEP(nk3,nv3)
#endif
        }
#undef FLINT_KV_LOAD
#undef FLINT_KV_STEP
        if(lane<HEADS) {
            device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;
            stat_out[0]=hi;stat_out[1]=z;
        }
        for(uint h=0;h<HEADS;++h) {
            device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4:(device bfloat16_t*)partial;
            device float* unrounded_out=rounded?partial:partial+((first+h)*parts+part)*128+lane*4;
            if(rounded) {
                for(uint j=0;j<4;j+=4)
                    *((device vec<bfloat16_t,4>*)(rounded_out+j))=vec<bfloat16_t,4>(bfloat16_t(acc[h][j+0]),bfloat16_t(acc[h][j+1]),bfloat16_t(acc[h][j+2]),bfloat16_t(acc[h][j+3]));
            } else {
                for(uint j=0;j<4;j+=4)
                    *((device float4*)(unrounded_out+j))=float4(acc[h][j+0],acc[h][j+1],acc[h][j+2],acc[h][j+3]);
            }
        }
    }
}

inline void route_dot_bf16(const threadgroup bfloat16_t* x,const device bfloat16_t* w,
                      device float* probs,uint lane,uint sg) {
    for(uint row=sg;row<256;row+=FLINT_GROUPS) {
        float a=0.0f;
        for(uint b=0;b<16;++b) {
            uint at=b*128+lane*4;
            const vec<bfloat16_t,4> packed=*((const device vec<bfloat16_t,4>*)(w+row*2048+at));
            const float4 weights=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
            for(uint j=0;j<4;++j)a+=weights[j]*float(x[at+j]);
        }
        for(ushort d=16;d>=1;d>>=1)a+=simd_shuffle_down(a,d);
        a=float(bfloat16_t(a));
        if(lane==0){float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(a)));probs[row]=a<0?tail:1.0f-tail;}
    }
}

inline void rms_256_w4(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {
    float sums[2];
    for(uint leaf_set=0;leaf_set<2;++leaf_set) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=float(x[leaf_set*4+j]);squared+=v*v;}
        sums[leaf_set]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+eps);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=w[i]*bfloat16_t(float(x[j])*inverse);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Independent expert teams: four groups per routed role, eight for shared.
// Routed matrices come from the expert's pack (experts[] holds the GPU
// address of each pack of this layer); the shared expert from the bank.
inline void expert_team_gateup(const device uchar* bank,const threadgroup uint* d,
                      const device ulong* experts,
                      const threadgroup bfloat16_t* x,const device uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,
                      uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4, team=shared?group-24:group%4;
    uint teams=shared?8:4, width=shared?1024:512;
    const device uchar* pack=shared?bank:(const device uchar*)experts[ids[role]];
    Matrix g=shared?weight_matrix(bank,d+45):expert_matrix(pack,0);
    Matrix u=shared?weight_matrix(bank,d+50):expert_matrix(pack,1);
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*2;local<width;local+=teams*8*2) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[role*512+local+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::fast::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[role*512+local+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
inline void expert_team_down(const device uchar* bank,const threadgroup uint* d,
                     const device ulong* experts,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4, team=shared?group-24:group%4;
    uint teams=shared?8:4;
    const device uchar* pack=shared?bank:(const device uchar*)experts[ids[role]];
    Matrix m=shared?weight_matrix(bank,d+55):expert_matrix(pack,2);
    const device bfloat16_t* input=x+role*512;
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*4;local<2048;local+=teams*8*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(v);}
    }
}

inline bool expert_team_handoff(device uint* routing,device atomic_uint* control,
                    uint tid,uint group,uint layer,uint phase,threadgroup uint* ready) {
    uint first=group>=24?24:(group/4)*4, count=group>=24?8:4;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),layer+1,memory_order_relaxed);
        bool complete=true;
        for(uint peer=first;peer<first+count;++peer) {
            uint spins=0;
            while(atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)<layer+1 &&
                  atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
            complete=complete && atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)>=layer+1;
        }
        ready[0]=uint(complete && atomic_load_explicit(control+2,memory_order_relaxed)==0);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    return ready[0]!=0;
}

// Flint's routing has no expert bias: rank on the sigmoid probabilities.
inline void w3_route_own_rank(device float* probs,
                            device float* chosen,device uint* ids,uint lane,uint sg,uint group) {
    if(sg==0 && group<24) {
        uint role=group/4;
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j];}
        for(uint rank=0;rank<=role;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            if(rank==role) {
                float value=simd_shuffle(p[selected/32],selected%32);
                if(lane==0) {
                    ids[role]=selected;
                    if(group%4==0)chosen[role]=value;
                }
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline bool kv_team_sync(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,uint group) {
    device atomic_uint* team_control=control+4+(group/FLINT_KV_TEAM_SIZE)*2;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*FLINT_KV_TEAM_SIZE;
        atomic_fetch_add_explicit(team_control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(team_control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        ready[0]=uint(atomic_load_explicit(team_control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;
}

// Buffers (ds4_metal_flint.m binds them in this order):
//   0 bank         the whole GGUF, descriptor offsets in 256-byte units
//   1 metadata     50 x 64 uint: per-layer weight descriptors, [61] KV offset
//                  (elements), [62] KV capacity; row 49 holds norm/embed/head
//   2..5 kv0..kv3  bf16 KV blocks of 65536 positions (rec's layout)
//   6 parameters   {token, position, initial_position, layer_count}
//   7 silu_table   65536 bf16: silu(x) for every bf16 bit pattern
//   8 control      1024 uint, zero on entry (barrier counters, watchdog)
//   9 experts      49 x 256 ulong: GPU address of each expert pack
//  10 logits       150144 bf16
//  11 trace        50 x 2048 bf16 residual stream
//  12 diagnostic   FLINT_WORKERS uint: 0, or the phase a group timed out in
//  13 workspace    78848 bf16
//  14 partials     2097152 float
//  15 statistics   65536 float
//  16 routing      1280 uint
//  17 requests     49 x 8 + 1 uint, zero on entry: per sparse layer the six
//                  selected expert ids [0..5] and a miss mask [6]; word 392
//                  is 1 + layer when the launch aborted on a cache miss
//
// parameters: {token, position, initial_position, layer_count, layer_start,
// streaming, 0, 0}.  layer_start > 0 resumes a launch from that layer: the
// residual is read back from the trace row the previous layer wrote.  With
// streaming set, a zero expert-table entry for a selected expert aborts the
// launch (every group leaves through the abort word) after publishing the
// layer's ids; the host fetches the packs and relaunches from the layer.
kernel void flint_decode(
        const device uchar* bank [[buffer(0)]],
        const device uint* metadata [[buffer(1)]],
        device bfloat16_t* kv0 [[buffer(2)]],
        device bfloat16_t* kv1 [[buffer(3)]],
        device bfloat16_t* kv2 [[buffer(4)]],
        device bfloat16_t* kv3 [[buffer(5)]],
        const device uint* parameters [[buffer(6)]],
        const device bfloat16_t* silu_table [[buffer(7)]],
        device uint* private_control [[buffer(8)]],
        const device ulong* expert_table [[buffer(9)]],
        device bfloat16_t* logits [[buffer(10)]],
        device bfloat16_t* trace [[buffer(11)]],
        device uint* diagnostic [[buffer(12)]],
        device bfloat16_t* workspace [[buffer(13)]],
        device float* partials [[buffer(14)]],
        device float* statistics [[buffer(15)]],
        device uint* routing [[buffer(16)]],
        device uint* requests [[buffer(17)]],
        uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
        uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
        uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
#define KV_TEAM_SYNC() if(!kv_team_sync(control,tid,kv_phase,ready,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
#define SYNC() if(!sync_all(control,tid,phase,ready,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}

    uint token=parameters[0], position=parameters[1], initial_position=parameters[2], layer_count=parameters[3];
    uint layer_start=parameters[4], streaming=parameters[5];
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,sg=group*8+simdgroup_index_in_threadgroup;
    uint index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];uint phase=0;uint kv_phase=0;uint kv_qkv_base=0;

    threadgroup float reduction[33];
    threadgroup uint layer_descriptor[64];
    // Every resident group consumes scores in its private residual update.
    device uint* ids=routing+group*16;
    if(tid==0)atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    bfloat16_t h[8];
    alignas(8) threadgroup bfloat16_t n[2048];
    device bfloat16_t* q=workspace+4096;
    device bfloat16_t* k=workspace+8192;
    device bfloat16_t* v=workspace+8704;
    device bfloat16_t* a=workspace+9216;
    device bfloat16_t* o=workspace+13312;
    device bfloat16_t* gate=workspace+15360;
    device bfloat16_t* up=workspace+31744;
    device bfloat16_t* act=workspace+48128;
    device bfloat16_t* down=workspace+64512;
    const device uint* final=metadata+49*64;
    if(layer_start==0) {
        Matrix embedding=weight_matrix(bank,final+10);
        for(uint j=0;j<8;++j) {
            uint i=(tid+(j/4)*256)*4+j%4;
            size_t row=uint(token);uint code=embedding.w[row*2048+i];
            uint gi=row*32+i/64;
            h[j]=bfloat16_t(embedding.s[gi]*bfloat16_t(code)+embedding.b[gi]);
            if(group==0)trace[i]=h[j];
        }
    } else {
        // Resume: the residual entering layer_start is the trace row the
        // previous launch wrote at the end of layer_start-1.
        for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=trace[layer_start*2048+i];}
    }
    SYNC();
    for(uint layer=layer_start;layer<uint(layer_count);++layer) {
        if(tid<64)layer_descriptor[tid]=metadata[layer*64+tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup uint* d=layer_descriptor;
        const device ulong* experts=expert_table+layer*256;
        bool global=d[1]!=0,sparse=d[2]!=0;
        {rms_256_w4(h,(const device bfloat16_t*)AT(d[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
        uint cap=d[62],count=global?uint(position)+1:min(uint(position)+1,cap);
        uint insert=global?uint(position):(uint(initial_position)>=cap?(uint(position)-uint(initial_position))%cap:uint(position)%cap);
        KVView keys={kv0,kv1,kv2,kv3,d[61],cap,global};
        KVView vals=keys.value();
        uint kv_team=group/FLINT_KV_TEAM_SIZE,kv_member=group%FLINT_KV_TEAM_SIZE;
        for(;;) {
            uint ticket=0;
            if(lane==0)ticket=atomic_fetch_add_explicit(control+40+kv_team,1u,memory_order_relaxed);
            ticket=simd_broadcast(ticket,0);
            if(ticket>=kv_qkv_base+320u)break;
            uint unit=ticket-kv_qkv_base,member=unit/32u;
            uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
            m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
            q4_local_block<2048>(m,n,(head<32?q:head<36?k:v)+local*128,lane,(unit%32u)*4u);
        }
        kv_qkv_base+=392u;
        KV_TEAM_SYNC();
        for(uint member=kv_member;member<10;member+=FLINT_KV_TEAM_SIZE) {
            uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
            if(simdgroup_index_in_threadgroup==0 && head<36)
                head_norm(out,(const device bfloat16_t*)AT(d[head<32?5:6]),1,!global,uint(position),13.287712379549449f,1e-06f,lane,0);
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {
                KVView target=head<36?keys:vals;
                for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
            }
        }
        KV_TEAM_SYNC();
        bool rounded=global&&count>=1024;
        uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
        if(group<FLINT_WORKERS) {
            if(global && count>1024) {
                if(rounded) attention_partial_path_w53<8,true>(q,keys,vals,(const device bfloat16_t*)AT(d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<8,false>(q,keys,vals,(const device bfloat16_t*)AT(d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
            else {
                if(rounded) attention_partial_path_w53<4,true>(q,keys,vals,(const device bfloat16_t*)AT(d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<4,false>(q,keys,vals,(const device bfloat16_t*)AT(d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
        }
        KV_TEAM_SYNC();
        if(group<FLINT_WORKERS) {if(rounded) attention_merge_vec4_w2<true>(a,partials,statistics,parts,lane,sg); else attention_merge_vec4_w2<false>(a,partials,statistics,parts,lane,sg);}
        SYNC();
        q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,FLINT_GROUPS);
        SYNC();
        {
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=bfloat16_t(float(h[j])+float(o[i]));}
            rms_256_w4(h,(const device bfloat16_t*)AT(d[4]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);
        }
        if(sparse) {
            if(group<FLINT_WORKERS) route_dot_bf16(n,(const device bfloat16_t*)AT(d[60]),statistics,lane,sg);
            SYNC();
            w3_route_own_rank(statistics,(device float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);
            // The local route_top device fence publishes this group's ids and scores.
            if(group<24) {
                // Publish the selection (one group per role) and, when streaming,
                // abort on a pack that is not resident.  The abort word makes
                // every other group leave at its next barrier; the host reads
                // requests[392], fetches, and relaunches from this layer.
                uint role=group/4,e=ids[role];
                bool miss=streaming!=0 && experts[e]==0;
                if(tid==0 && (group%4)==0) {
                    requests[layer*8+role]=e;
                    if(miss)atomic_fetch_or_explicit((device atomic_uint*)(requests+layer*8+6),1u<<role,memory_order_relaxed);
                }
                if(miss) {
                    if(tid==0) {
                        atomic_store_explicit((device atomic_uint*)(requests+392),layer+1u,memory_order_relaxed);
                        atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
                        atomic_store_explicit(control+2,0x40000000u|layer,memory_order_relaxed);
                    }
                    return;
                }
            }
            if(group<FLINT_MLP_WORKERS) {
                expert_team_gateup(bank,d,experts,n,ids,silu_table,act,lane,simdgroup_index_in_threadgroup,group);
                if(!expert_team_handoff(routing,control,tid,group,layer,phase,ready)) {
                    if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);
                    return;
                }
                expert_team_down(bank,d,experts,act,ids,down,lane,simdgroup_index_in_threadgroup,group);
            }
            SYNC();
        } else {
            Matrix gm=weight_matrix(bank,d+30),um=weight_matrix(bank,d+35),dm=weight_matrix(bank,d+40);
            q4_local<2048>(gm,n,gate,lane,sg,FLINT_GROUPS);q4_local<2048>(um,n,up,lane,sg,FLINT_GROUPS);
            SYNC();
            for(uint i=index;i<gm.n;i+=FLINT_THREADS) {
                bfloat16_t g=gate[i],u=up[i];
                auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;
                act[i]=bfloat16_t(bfloat16_t(g*sig)*u);
            }
            SYNC();
            q4<6144>(dm,act,down,lane,sg,FLINT_GROUPS);
            SYNC();
        }
        {
        if(sparse) {
            float w3_chosen=lane<6?((device float*)(routing+8))[lane]:0.0f;
            float w3_sum=0.0f;
            for(uint j=0;j<6;++j)w3_sum=simd_shuffle(w3_chosen,j)+w3_sum;
            w3_sum+=1.0e-20f;
            float w3_weight=lane<6?w3_chosen/w3_sum:0.0f;
            float sums[8];
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t(down[i]*bfloat16_t(simd_shuffle(w3_weight,ushort(0)))))+0.0f;
            }
            for(uint j=0;j<8;++j)for(uint r=1;r<6;++r) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t((float(bfloat16_t(down[r*2048+i]*bfloat16_t(simd_shuffle(w3_weight,ushort(r)))))+0.0f)+sums[j]));
            }
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=bfloat16_t(float(bfloat16_t(sums[j]))+float(down[6*2048+i]));
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        } else {
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=down[i];
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        }
        }
    }
    {rms_256_w4(h,(const device bfloat16_t*)AT(final[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
    if(group<FLINT_HEAD_WORKERS) q4_local<2048>(weight_matrix(bank,final+30),n,logits,lane,sg,FLINT_HEAD_WORKERS*8u);
    if(tid==0)diagnostic[group]=0;
#undef SYNC
#undef KV_TEAM_SYNC
}
