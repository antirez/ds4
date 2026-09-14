// SPDX-License-Identifier: MIT
// Exercise production score dispatch and top-k, with exact legacy score/ID
// parity. Reuse the guarded tensor and independent top-k oracle helpers.
#define DS4_METAL_INDEXER_HEADS_TESTING 1
#define main indexer_topk_fixture_main
#include "test_metal_indexer_topk.m"
#undef main
#include <errno.h>

enum { IH_DIM=128, IH_SAMPLES=20, IH_WARMUP=2 };
typedef struct {uint32_t comp,tokens,heads,pos0,ratio;float scale;} ih_shape;
typedef struct {
    tk_tensor q,weights,keys,scores[4],selected[4];
    uint64_t q_hash,weights_hash,keys_hash;
    ih_shape shape;
    uint32_t k;
} ih_fixture;
static unsigned ih_cases,ih_grouped[3];
static const uint32_t ih_score_poison=0x7fc95327u;
static const float ih_production_scale=0.011048543457f; // 1/sqrt(128*64).

static uint32_t ih_bits(float value) {uint32_t u;memcpy(&u,&value,4);return u;}
static unsigned ih_head(unsigned arm) {return arm==1?2:arm==2?4:arm==3?1:0;}
static uint32_t ih_flag(unsigned arm) {
    return arm==0?DS4_GPU_TEST_INDEXER_LEGACY_HEADS:
           arm==1?DS4_GPU_TEST_INDEXER_HEAD2:
           arm==2?DS4_GPU_TEST_INDEXER_HEAD4:DS4_GPU_TEST_INDEXER_HEAD1;
}
static void ih_poison_scores(tk_tensor *t) {
    tk_reset(t);
    uint32_t *p=ds4_gpu_tensor_contents(t->view);
    for(NSUInteger i=0;i<t->payload/4;++i)p[i]=ih_score_poison;
}
static float ih_random_float(uint32_t *state) {
    const uint32_t random=tk_random(state);
    uint32_t bits=(random&0x807fffffu)|(123u<<23);
    float value;memcpy(&value,&bits,4);return value;
}
static void ih_init(ih_fixture *f,ih_shape s,unsigned pattern,bool bench) {
    *f=(ih_fixture){.shape=s,.k=s.comp<512?s.comp:512};
    // Benchmarks use the same 256-byte-aligned base offsets as scratch and
    // normal graph tensors; correctness cases also exercise scalar offsets.
    f->q=tk_alloc((NSUInteger)s.tokens*s.heads*IH_DIM*4,bench?0:1);
    f->weights=tk_alloc((NSUInteger)s.tokens*s.heads*4,bench?0:3);
    f->keys=tk_alloc((NSUInteger)s.comp*IH_DIM*4,bench?0:5);
    for(unsigned arm=0;arm<4;++arm) {
        f->scores[arm]=tk_alloc((NSUInteger)s.comp*s.tokens*4,bench?0:arm+1);
        f->selected[arm]=tk_alloc((NSUInteger)f->k*s.tokens*4,bench?0:arm+4);
        ih_poison_scores(&f->scores[arm]);
    }
    float *q=ds4_gpu_tensor_contents(f->q.view);
    float *w=ds4_gpu_tensor_contents(f->weights.view);
    float *keys=ds4_gpu_tensor_contents(f->keys.view);
    uint32_t state=9137+s.comp+73*s.tokens+pattern;
    for(uint32_t token=0;token<s.tokens;++token)for(uint32_t head=0;head<s.heads;++head) {
        w[(size_t)token*s.heads+head]=fabsf(ih_random_float(&state))+0.015625f;
        if(pattern==2)w[(size_t)token*s.heads+head]=(head%2?-1.f:1.f)*(head%7?8.f:0.03125f);
        for(uint32_t d=0;d<IH_DIM;++d) {
            float value=ih_random_float(&state);
            if(pattern==1)value=d%2?0.f:-0.f;
            if(pattern==2)value=(d%3?-1.f:1.f)*(0.75f+(head%4)*0.0001220703125f);
            if(pattern==3)value=(d%2?-1.f:1.f)*(1.00048828125f+((int)((d+head)%3)-1)*0x1p-23f);
            q[((size_t)token*s.heads+head)*IH_DIM+d]=value;
        }
    }
    for(uint32_t comp=0;comp<s.comp;++comp)for(uint32_t d=0;d<IH_DIM;++d) {
        float value=ih_random_float(&state);
        if(pattern==1)value=(d%7-3.f)*0.03125f;
        if(pattern==2)value=(d%5?-1.f:1.f)*(0.625f+(comp%7)*0.000244140625f);
        if(pattern==3)value=(d%3?-1.f:1.f)*(0.12506103515625f+((int)((d+comp)%3)-1)*0x1p-26f);
        keys[(size_t)comp*IH_DIM+d]=value;
    }
    f->q_hash=tk_hash(ds4_gpu_tensor_contents(f->q.base),f->q.total);
    f->weights_hash=tk_hash(ds4_gpu_tensor_contents(f->weights.base),f->weights.total);
    f->keys_hash=tk_hash(ds4_gpu_tensor_contents(f->keys.base),f->keys.total);
}
static void ih_free(ih_fixture *f) {
    for(unsigned arm=0;arm<4;++arm) {tk_free(&f->selected[arm]);tk_free(&f->scores[arm]);}
    tk_free(&f->keys);tk_free(&f->weights);tk_free(&f->q);
}
static bool ih_expected_grouped(const ih_fixture *f,unsigned arm) {
    return arm!=0&&f->shape.heads==64&&!g_quality_mode&&
        ds4_gpu_execution_phase_allows_prefill(ds4_gpu_get_execution_phase())&&
        !(ds4_gpu_mpp_available()&&f->shape.tokens>=16);
}
static void ih_encode(ih_fixture *f,unsigned arm,bool topk) {
    const ih_shape s=f->shape;
    tk_require(ds4_gpu_indexer_scores_decode_batch_tensor(f->scores[arm].view,
        f->q.view,f->weights.view,f->keys.view,s.comp,s.tokens,s.pos0,s.heads,IH_DIM,s.ratio,s.scale),
        "production grouped-head score dispatch");
    if(topk)tk_require(ds4_gpu_indexer_topk_tensor(f->selected[arm].view,
        f->scores[arm].view,s.comp,s.tokens,f->k),"production score-to-top-k dispatch");
}
static void ih_counter_check(ih_fixture *f,unsigned arm,uint64_t before1,uint64_t before2,uint64_t before4,unsigned calls) {
    const bool expected=ih_expected_grouped(f,arm);
    tk_require(g_indexer_head2_launches==before2+(expected&&arm==1?calls:0)&&
               g_indexer_head4_launches==before4+(expected&&arm==2?calls:0)&&
               g_indexer_head1_launches==before1+(expected&&arm==3?calls:0),
               "exact forced head-group dispatch counters / legacy or quality fallback");
}
static void ih_dispatch(ih_fixture *f,unsigned arm,bool topk,bool batch) {
    const uint64_t before1=g_indexer_head1_launches,before2=g_indexer_head2_launches,before4=g_indexer_head4_launches;
    ds4_gpu_test_set_flags(ih_flag(arm));
    if(batch)tk_require(ds4_gpu_begin_commands(),"begin score batch");
    ih_encode(f,arm,topk);
    if(batch)tk_require(ds4_gpu_end_commands(),"finish score batch");
    ih_counter_check(f,arm,before1,before2,before4,1);
    if(ih_expected_grouped(f,arm))++ih_grouped[arm-1];
    ds4_gpu_test_set_flags(0);
}
static void ih_inputs(const ih_fixture *f) {
    tk_require(f->q_hash==tk_hash(ds4_gpu_tensor_contents(f->q.base),f->q.total)&&
        f->weights_hash==tk_hash(ds4_gpu_tensor_contents(f->weights.base),f->weights.total)&&
        f->keys_hash==tk_hash(ds4_gpu_tensor_contents(f->keys.base),f->keys.total),
        "query, head weights, compressed keys and their canaries remain immutable");
}
static void ih_compare(ih_fixture *f,unsigned arm,bool topk) {
    const ih_shape s=f->shape;
    const float *reference=ds4_gpu_tensor_contents(f->scores[0].view);
    const float *actual=ds4_gpu_tensor_contents(f->scores[arm].view);
    tk_guards(&f->scores[0]);tk_guards(&f->scores[arm]);
    for(uint32_t token=0;token<s.tokens;++token) {
        const uint32_t visible=(uint32_t)MIN(((uint64_t)s.pos0+token+1)/s.ratio,s.comp);
        for(uint32_t comp=0;comp<s.comp;++comp) {
            const size_t i=(size_t)token*s.comp+comp;
            const uint32_t a=ih_bits(reference[i]),b=ih_bits(actual[i]);
            if(a!=b) {
                fprintf(stderr,"head%u score mismatch ncomp=%u tokens=%u heads=%u pos0=%u token=%u comp=%u legacy=%08x candidate=%08x\n",
                    ih_head(arm),s.comp,s.tokens,s.heads,s.pos0,token,comp,a,b);
                tk_require(false,"score arithmetic must remain bitwise identical");
            }
            tk_require(comp<visible?isfinite(reference[i])&&isfinite(actual[i]):a==0xff800000u&&b==0xff800000u,
                       "every score written: finite visible cells, exact negative infinity causal mask");
        }
    }
    if(topk) {
        tk_guards(&f->selected[0]);tk_guards(&f->selected[arm]);
        tk_require(!memcmp(ds4_gpu_tensor_contents(f->selected[0].view),
            ds4_gpu_tensor_contents(f->selected[arm].view),f->selected[arm].payload),
            "score grouping preserves selected IDs including ties and masked rows");
        tk_oracle(&f->scores[arm],&f->selected[arm],(tk_shape){s.comp,s.tokens,f->k},false);
    }
    ih_inputs(f);
}
static void ih_case(ih_shape s,unsigned pattern) {
    @autoreleasepool {
        ih_fixture f;ih_init(&f,s,pattern,false);
        ih_dispatch(&f,0,true,true);
        for(unsigned arm=1;arm<4;++arm) {
            ih_dispatch(&f,arm,true,(ih_cases/3)%2!=0);
            ih_compare(&f,arm,true);++ih_cases;
        }
        ih_free(&f);
    }
}
static void ih_flag_precedence(void) {
    @autoreleasepool {
        ih_fixture f;ih_init(&f,(ih_shape){513,9,64,2052,4,0.0883883476f},3,false);
        ih_dispatch(&f,0,true,true);
        const uint32_t flags[]={DS4_GPU_TEST_INDEXER_HEAD1|DS4_GPU_TEST_INDEXER_HEAD2,
            DS4_GPU_TEST_INDEXER_HEAD1|DS4_GPU_TEST_INDEXER_HEAD2|DS4_GPU_TEST_INDEXER_HEAD4,
            DS4_GPU_TEST_INDEXER_HEAD1|DS4_GPU_TEST_INDEXER_HEAD2|DS4_GPU_TEST_INDEXER_HEAD4|DS4_GPU_TEST_INDEXER_LEGACY_HEADS};
        const unsigned chosen[]={1,2,0};
        for(unsigned i=0;i<3;++i) {
            const uint64_t before1=g_indexer_head1_launches,before2=g_indexer_head2_launches,before4=g_indexer_head4_launches;
            const unsigned output_arm=chosen[i]?chosen[i]:3;
            ds4_gpu_test_set_flags(flags[i]);
            tk_require(ds4_gpu_begin_commands(),"begin flag precedence test");
            ih_encode(&f,output_arm,true);
            tk_require(ds4_gpu_end_commands(),"finish flag precedence test");
            ih_counter_check(&f,chosen[i],before1,before2,before4,1);
            ih_compare(&f,output_arm,true);++ih_cases;
        }
        ds4_gpu_test_set_flags(0);ih_free(&f);
    }
}
static bool ih_default_case(uint32_t comp,uint32_t tokens,uint32_t heads,
        ds4_gpu_execution_phase phase,bool quality) {
    @autoreleasepool {
        const ds4_gpu_execution_phase oldphase=ds4_gpu_exchange_execution_phase(phase);
        const bool oldquality=g_quality_mode;
        ds4_gpu_set_quality(quality);
        ih_fixture f;
        ih_init(&f,(ih_shape){comp,tokens,heads,comp*4-tokens,4,ih_production_scale},3,false);
        ih_dispatch(&f,0,true,true);
        const uint64_t before1=g_indexer_head1_launches,before2=g_indexer_head2_launches,before4=g_indexer_head4_launches;
        // This oracle deliberately spells out the measured admission bounds;
        // it does not ask the production default-selector for its answer.
        const char *device_name=g_device.name.UTF8String;
        const bool expected=device_name&&strcmp(device_name,"Apple M1 Max")==0&&
            comp>=1024&&comp<=65536&&tokens>=128&&tokens<=512&&heads==64&&
            (phase==DS4_GPU_PHASE_AUTO||phase==DS4_GPU_PHASE_PREFILL)&&!quality&&
            !(ds4_gpu_mpp_available()&&tokens>=16);
        ds4_gpu_test_set_flags(0);
        tk_require(ds4_gpu_begin_commands(),"begin default head selection");
        ih_encode(&f,1,true);
        tk_require(ds4_gpu_end_commands(),"finish default head selection");
        tk_require(g_indexer_head1_launches==before1&&g_indexer_head4_launches==before4&&
            g_indexer_head2_launches==before2+expected,
            "default HG2 device/shape/phase policy must match its documented boundaries");
        ih_compare(&f,1,true);++ih_cases;
        ih_free(&f);
        ds4_gpu_set_quality(oldquality);ds4_gpu_exchange_execution_phase(oldphase);
        return expected;
    }
}
static void ih_default_selection(void) {
    unsigned checked=0,admitted=0;
    const uint32_t boundaries[][2]={{1023,128},{1024,128},{65536,128},{65537,128},
        {1024,127},{1024,512},{1024,513}};
    for(unsigned i=0;i<sizeof(boundaries)/sizeof(boundaries[0]);++i) {
        admitted+=ih_default_case(boundaries[i][0],boundaries[i][1],64,DS4_GPU_PHASE_PREFILL,false);
        ++checked;
    }
    admitted+=ih_default_case(1024,128,64,DS4_GPU_PHASE_AUTO,false);++checked;
    admitted+=ih_default_case(1024,128,64,DS4_GPU_PHASE_DECODE,false);++checked;
    admitted+=ih_default_case(1024,128,64,DS4_GPU_PHASE_PREFILL,true);++checked;
    admitted+=ih_default_case(1024,128,63,DS4_GPU_PHASE_PREFILL,false);++checked;
    printf("PASS: %u default-vs-legacy score/top-k checks, HG2 admitted=%u; measured device boundaries, phase, quality and head-count exclusions.\n",checked,admitted);
}
static tk_time ih_timed(ih_fixture *f,unsigned arm,bool topk,unsigned repeats) {
    @autoreleasepool {
        const uint64_t before1=g_indexer_head1_launches,before2=g_indexer_head2_launches,before4=g_indexer_head4_launches;
        ds4_gpu_test_set_flags(ih_flag(arm));
        const double start=tk_now();
        tk_require(ds4_gpu_begin_commands(),"begin timed score batch");
        id<MTLCommandBuffer> cb=g_batch_cb;
        for(unsigned i=0;i<repeats;++i)ih_encode(f,arm,topk);
        tk_require(ds4_gpu_end_commands(),"finish timed score batch");
        const tk_time result={(tk_now()-start)/repeats,
            (cb.GPUEndTime-cb.GPUStartTime)*1e3/repeats};
        tk_require(result.wall>0&&isfinite(result.wall)&&result.gpu>0&&isfinite(result.gpu),"valid completed GPU timings");
        ih_counter_check(f,arm,before1,before2,before4,repeats);
        ds4_gpu_test_set_flags(0);return result;
    }
}
static double ih_statistics(const tk_time *times,bool gpu,const char *name) {
    double sorted[IH_SAMPLES],sum=0;
    printf("  %s raw_ms=",name);
    for(unsigned i=0;i<IH_SAMPLES;++i) {
        sorted[i]=gpu?times[i].gpu:times[i].wall;sum+=sorted[i];
        printf("%s%.6f",i?",":"",sorted[i]);
    }
    qsort(sorted,IH_SAMPLES,sizeof(*sorted),tk_double_compare);
    const double median=(sorted[IH_SAMPLES/2-1]+sorted[IH_SAMPLES/2])*0.5;
    printf(" mean=%.6f median=%.6f range=%.6f..%.6f\n",sum/IH_SAMPLES,median,sorted[0],sorted[IH_SAMPLES-1]);
    return median;
}
static void ih_bench(ih_shape s,unsigned head_filter) {
    @autoreleasepool {
        ih_fixture f;ih_init(&f,s,0,true);
        tk_require(ih_expected_grouped(&f,1)&&ih_expected_grouped(&f,2)&&ih_expected_grouped(&f,3),
                   "benchmark requires actual candidate kernels, not an inherited fallback");
        const uint64_t cells=(uint64_t)s.comp*s.tokens;
        const unsigned repeats=cells<=524288?8:cells<=2097152?4:cells<=8388608?2:1;
        for(unsigned arm=0;arm<4;++arm)
            if(arm==0||!head_filter||ih_head(arm)==head_filter)ih_dispatch(&f,arm,true,true);
        for(unsigned arm=1;arm<4;++arm)
            if(!head_filter||ih_head(arm)==head_filter)ih_compare(&f,arm,true);
        for(unsigned full=0;full<2;++full)for(unsigned candidate=1;candidate<4;++candidate) {
            if(head_filter&&ih_head(candidate)!=head_filter)continue;
            tk_time a[IH_SAMPLES],b[IH_SAMPLES];
            for(unsigned block=0;block<IH_WARMUP+IH_SAMPLES/2;++block) {
                const tk_time a0=ih_timed(&f,0,full,repeats);
                const tk_time b0=ih_timed(&f,candidate,full,repeats);
                const tk_time b1=ih_timed(&f,candidate,full,repeats);
                const tk_time a1=ih_timed(&f,0,full,repeats);
                if(block>=IH_WARMUP) {
                    const unsigned i=2*(block-IH_WARMUP);
                    a[i]=a0;b[i]=b0;b[i+1]=b1;a[i+1]=a1;
                }
            }
            ih_compare(&f,candidate,full);
            printf("BENCH %s vs head%u ncomp=%u tokens=%u heads=%u K=%u pos0=%u ratio=%u scale=%.10g repeats_per_CB=%u warmup_ABBA=%u samples_per_arm=%u\n",
                full?"scores+topk":"scores",ih_head(candidate),s.comp,s.tokens,s.heads,f.k,s.pos0,s.ratio,s.scale,repeats,IH_WARMUP,IH_SAMPLES);
            const double aw=ih_statistics(a,false,"legacy wall"),bw=ih_statistics(b,false,"candidate wall");
            const double ag=ih_statistics(a,true,"legacy GPU"),bg=ih_statistics(b,true,"candidate GPU");
            printf("  median throughput gain wall=%+.3f%% GPU=%+.3f%%\n",100*(aw/bw-1),100*(ag/bg-1));
            fflush(stdout);
        }
        ih_free(&f);
    }
}
static bool ih_parse_bound(const char *text,uint32_t minimum,uint32_t maximum,uint32_t *out) {
    if(!text||text[0]<'0'||text[0]>'9')return false;
    errno=0;char *end=NULL;
    const unsigned long value=strtoul(text,&end,10);
    if(errno||!end||*end||value>UINT32_MAX||value<minimum||value>maximum)return false;
    *out=(uint32_t)value;return true;
}
int main(int argc,char **argv) {
    uint32_t bench_comp=0,bench_tokens=0,head_filter=0;
    const bool bench=argc==2||argc==4||argc==5;
    if((argc!=1&&!bench)||(bench&&strcmp(argv[1],"--bench"))||
       (argc>=4&&(!ih_parse_bound(argv[2],512,65536,&bench_comp)||
                  !ih_parse_bound(argv[3],1,512,&bench_tokens)||
                  (uint64_t)bench_comp*4<bench_tokens))||
       (argc==5&&(!ih_parse_bound(argv[4],1,4,&head_filter)||head_filter==3))) {
        fprintf(stderr,"Usage: %s [--bench [n_comp n_tokens [head_group]]]\n"
                       "  n_comp: 512..65536; n_tokens: 1..512; head_group: 1, 2 or 4; 4*n_comp >= n_tokens.\n",argv[0]);
        return 2;
    }
    @autoreleasepool {
        tk_require(ds4_gpu_init(),"Metal initialization");
        const ds4_gpu_execution_phase oldphase=ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_PREFILL);
        ds4_gpu_set_quality(false);
        for(unsigned p=0;p<4;++p) {
            const ds4_gpu_execution_phase phases[]={DS4_GPU_PHASE_DECODE,DS4_GPU_PHASE_VERIFY,
                DS4_GPU_PHASE_BATCH_DECODE,DS4_GPU_PHASE_MIXED};
            ds4_gpu_exchange_execution_phase(phases[p]);
            ih_case((ih_shape){513,9,64,2052,4,0.0883883476f},3);
        }
        ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_AUTO);
        ih_case((ih_shape){513,9,64,2052,4,0.0883883476f},3);
        ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_PREFILL);
        ih_flag_precedence();
        printf("Indexer grouped heads device=%s; tensor path=%d\n",g_device.name.UTF8String,ds4_gpu_mpp_available());
        const uint32_t shapes[][2]={{31,1},{32,7},{33,8},{511,9},{512,127},{513,128},
            {4096,512},{8192,128},{4097,7},{8193,9},{33,512},{513,1}};
        for(unsigned shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);++shape)
            for(unsigned pattern=0;pattern<4;++pattern)for(unsigned mask=0;mask<3;++mask) {
                const uint32_t comp=shapes[shape][0],tokens=shapes[shape][1];
                const uint32_t ratio=pattern==3?8:4;
                const uint32_t pos0=mask==0?0:mask==1?ratio*(comp/2):ratio*comp;
                ih_case((ih_shape){comp,tokens,64,pos0,ratio,pattern==2?0.0625f:0.0883883476f},pattern);
            }
        for(uint32_t heads=1;heads<=65;heads=heads==1?3:heads==3?63:heads==63?65:66)
            ih_case((ih_shape){513,9,heads,1024,4,0.0883883476f},3);
        const float scales[]={ih_production_scale,0.f,-0.f,-ih_production_scale,0.1f};
        for(unsigned i=0;i<sizeof(scales)/sizeof(scales[0]);++i)
            for(unsigned pattern=0;pattern<=3;pattern+=3)
                ih_case((ih_shape){513,9,64,1024,4,scales[i]},pattern);
        ds4_gpu_set_quality(true);
        ih_case((ih_shape){513,9,64,2052,4,0.0883883476f},3);
        ds4_gpu_set_quality(false);
        ih_default_selection();
        tk_require(ih_grouped[0]>0&&ih_grouped[1]>0&&ih_grouped[2]>0,"all three candidate kernels must actually execute");
        printf("PASS: %u native score/top-k comparisons, head2=%u head4=%u head1=%u; bitwise scores and selected IDs, causal masks, dispatch counters, canaries and immutable inputs.\n",ih_cases,ih_grouped[0],ih_grouped[1],ih_grouped[2]);
        fflush(stdout);
        if(bench) {
            puts("Warm fixed-input full API ABBA; score-only and score+top-k timings include every score/merge kernel, CPU encoding and command-buffer completion. Allocations and Q/key production are excluded; this is not full model throughput. All head variants use identical inputs and score allocations. pos0 models a long cached context with a causal current chunk.");
            if(argc>=4) {
                ih_bench((ih_shape){bench_comp,bench_tokens,64,bench_comp*4-bench_tokens,4,ih_production_scale},head_filter);
            } else {
                for(uint32_t comp=4096;comp<=65536;comp*=4)for(unsigned batch=0;batch<2;++batch) {
                    const uint32_t tokens=batch?512:128;
                    ih_bench((ih_shape){comp,tokens,64,comp*4-tokens,4,ih_production_scale},0);
                }
            }
        }
        ds4_gpu_test_set_flags(0);ds4_gpu_exchange_execution_phase(oldphase);ds4_gpu_cleanup();
    }
    return 0;
}
