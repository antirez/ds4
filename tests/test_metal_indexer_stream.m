// SPDX-License-Identifier: MIT
// Reuse the guarded tensors and independent CPU top-k oracle from the
// production top-k fixture. No GGUF or model weights are required.
#define DS4_METAL_INDEXER_STREAM_TESTING 1
#define main ds4_indexer_topk_fixture_main
#include "test_metal_indexer_topk.m"
#undef main

typedef struct { uint32_t columns, tokens, pos0, chunk; } ss_shape;
typedef struct {
    ss_shape shape;
    tk_tensor q, weights, keys, scores, reference, actual;
    uint64_t hashes[3];
} ss_case;
static unsigned ss_cases;

static ss_case ss_alloc(ss_shape shape, unsigned pattern) {
    ss_case c = {.shape=shape};
    c.q = tk_alloc((NSUInteger)shape.tokens*64u*128u*4u, 4);
    c.weights = tk_alloc((NSUInteger)shape.tokens*64u*4u, 1);
    c.keys = tk_alloc((NSUInteger)shape.columns*128u*4u, 8);
    c.scores = tk_alloc((NSUInteger)shape.columns*shape.tokens*4u, 3);
    c.reference = tk_alloc((NSUInteger)shape.tokens*512u*4u, 7);
    c.actual = tk_alloc(c.reference.payload, 5);
    tk_tensor *inputs[] = {&c.q, &c.weights, &c.keys};
    uint32_t random = 713u + shape.columns + shape.tokens + pattern;
    for (unsigned tensor = 0; tensor < 3; ++tensor) {
        float *data = ds4_gpu_tensor_contents(inputs[tensor]->view);
        for (NSUInteger j = 0; j < inputs[tensor]->payload/4u; ++j) {
            if (pattern == 1 && tensor == 0) data[j] = j & 1u ? -0.0f : 0.0f;
            else if (pattern == 2 && tensor == 2) data[j] = (float)((int)(j%128u)-64)/128.0f;
            else data[j] = (float)((int)(tk_random(&random)%127u)-63)/128.0f;
        }
        c.hashes[tensor] = tk_hash(ds4_gpu_tensor_contents(inputs[tensor]->base), inputs[tensor]->total);
    }
    return c;
}
static void ss_free(ss_case *c) {
    tk_free(&c->actual); tk_free(&c->reference); tk_free(&c->scores);
    tk_free(&c->keys); tk_free(&c->weights); tk_free(&c->q);
}
static int ss_reference(ss_case *c) {
    const ss_shape s = c->shape;
    if (!ds4_gpu_indexer_scores_decode_batch_tensor(c->scores.view, c->q.view,
            c->weights.view, c->keys.view, s.columns, s.tokens, s.pos0,
            64, 128, 4, 0.125f)) return 0;
    return ds4_gpu_indexer_topk_tensor(c->reference.view, c->scores.view,
                                      s.columns, s.tokens, 512);
}
static int ss_stream(ss_case *c) {
    const ss_shape s = c->shape;
    return ds4_gpu_indexer_scores_topk_stream_tensor(c->actual.view, c->q.view,
            c->weights.view, c->keys.view, s.columns, s.tokens, s.pos0,
            64, 128, 4, 0.125f, 512, s.chunk);
}
static void ss_check(ss_case *c) {
    const tk_tensor *inputs[] = {&c->q, &c->weights, &c->keys};
    for (unsigned i = 0; i < 3; ++i) {
        tk_guards(inputs[i]);
        tk_require(c->hashes[i] == tk_hash(ds4_gpu_tensor_contents(inputs[i]->base), inputs[i]->total),
                   "streaming indexer changed input");
    }
    tk_guards(&c->scores); tk_guards(&c->reference); tk_guards(&c->actual);
    const int32_t *expected = ds4_gpu_tensor_contents(c->reference.view);
    const int32_t *actual = ds4_gpu_tensor_contents(c->actual.view);
    for (NSUInteger i = 0; i < c->actual.payload/4u; ++i) {
        if (expected[i] != actual[i]) {
            fprintf(stderr, "stream n=%u t=%u pos=%u chunk=%u quality=%d row=%lu rank=%lu expected=%d got=%d\n",
                    c->shape.columns, c->shape.tokens, c->shape.pos0, c->shape.chunk,
                    g_quality_mode, (unsigned long)(i/512u), (unsigned long)(i%512u), expected[i], actual[i]);
            tk_require(false, "streaming exact index parity");
        }
    }
    tk_oracle(&c->scores, &c->actual,
              (tk_shape){c->shape.columns,c->shape.tokens,512}, false);
}
static void ss_test(ss_shape shape, unsigned pattern, bool quality) {
    ss_case c = ss_alloc(shape, pattern);
    g_quality_mode = quality;
    const bool batch = ss_cases & 1u;
    if (batch) tk_require(ds4_gpu_begin_commands(), "stream test begin");
    tk_require(ss_reference(&c), "stream reference encode");
    const uint64_t before = g_indexer_stream_launches;
    tk_require(ss_stream(&c) == 1 && g_indexer_stream_launches == before+1u,
               "stream candidate must execute");
    if (batch) tk_require(ds4_gpu_end_commands(), "stream test finish");
    ss_check(&c); ss_free(&c); ++ss_cases;
}
static void ss_invalid(void) {
    ss_case c = ss_alloc((ss_shape){4097,9,0,2048}, 0);
    const ss_shape s = c.shape;
    const uint64_t original = tk_hash(ds4_gpu_tensor_contents(c.actual.base), c.actual.total);
    const uint64_t before = g_indexer_stream_launches;
#define SS_REJECT(out, q, w, k, nc, nt, pos, nh, hd, r, sc, tk, chunk) \
    tk_require(ds4_gpu_indexer_scores_topk_stream_tensor(out,q,w,k,nc,nt,pos,nh,hd,r,sc,tk,chunk)==0, \
               "invalid streaming request admitted")
    SS_REJECT(NULL,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(c.actual.view,NULL,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(c.q.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,UINT32_MAX,0,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,UINT32_MAX,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,0,0.125f,512,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,NAN,512,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,511,s.chunk);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,3000);
    SS_REJECT(c.actual.view,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,8192);
    ds4_gpu_tensor *small = ds4_gpu_tensor_view(c.q.view, 0, 4);
    ds4_gpu_tensor *unaligned = ds4_gpu_tensor_view(c.actual.view, 1, c.actual.payload);
    SS_REJECT(c.actual.view,small,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,s.chunk);
    SS_REJECT(unaligned,c.q.view,c.weights.view,c.keys.view,s.columns,s.tokens,0,64,128,4,0.125f,512,s.chunk);
    ds4_gpu_tensor_free(small); ds4_gpu_tensor_free(unaligned);
    const ds4_gpu_execution_phase old = ds4_gpu_exchange_execution_phase(DS4_GPU_PHASE_VERIFY);
    tk_require(ss_stream(&c)==0, "verification must not take prefill candidate");
    ds4_gpu_exchange_execution_phase(old);
#undef SS_REJECT
    tk_require(!g_batch_cb && g_indexer_stream_launches == before &&
        original == tk_hash(ds4_gpu_tensor_contents(c.actual.base), c.actual.total),
        "decline must not write or encode");
    ss_free(&c);
}
static void ss_growth(void) {
    // Keep both calls in flight while the scratch allocation grows. This is
    // also exercised with commandBufferWithUnretainedReferences.
    ss_case a = ss_alloc((ss_shape){4097,1,16388,2048}, 0);
    ss_case b = ss_alloc((ss_shape){16385,17,65540,8192}, 2);
    g_indexer_stream_buffer = nil; g_indexer_stream_bytes = 0;
    tk_require(ds4_gpu_begin_commands(), "growth begin");
    tk_require(ss_reference(&a) && ss_reference(&b), "growth reference");
    tk_require(ss_stream(&a)==1 && ss_stream(&b)==1, "growth candidate");
    tk_require(ds4_gpu_end_commands(), "growth finish");
    ss_check(&a); ss_check(&b); ss_free(&b); ss_free(&a); ss_cases += 2;
}
static tk_time ss_time(ss_case *c, bool stream, unsigned repeats) {
    const double start = tk_now();
    tk_require(ds4_gpu_begin_commands(), "timing begin");
    id<MTLCommandBuffer> cb = g_batch_cb;
    const uint64_t before = g_indexer_stream_launches;
    for (unsigned i=0; i<repeats; ++i)
        tk_require((stream ? ss_stream(c) : ss_reference(c))==1, "timing encode");
    tk_require(ds4_gpu_end_commands(), "timing finish");
    tk_require(g_indexer_stream_launches == before+(stream ? repeats : 0u), "timing dispatch count");
    return (tk_time){(tk_now()-start)/repeats, 1e3*(cb.GPUEndTime-cb.GPUStartTime)/repeats};
}
static void ss_bench(ss_shape s) {
    ss_case c = ss_alloc(s, 0);
    tk_time old[TK_SAMPLES], candidate[TK_SAMPLES];
    const unsigned repeats = 1u;
    for (unsigned warm=0; warm<2; ++warm) {
        (void)ss_time(&c,false,repeats); (void)ss_time(&c,true,repeats);
        (void)ss_time(&c,true,repeats); (void)ss_time(&c,false,repeats);
    }
    for (unsigned b=0; b<TK_SAMPLES/2; ++b) {
        old[2*b]=ss_time(&c,false,repeats); candidate[2*b]=ss_time(&c,true,repeats);
        candidate[2*b+1]=ss_time(&c,true,repeats); old[2*b+1]=ss_time(&c,false,repeats);
    }
    ss_check(&c);
    printf("STREAM BENCH ncomp=%u ntok=%u pos0=%u chunk=%u samples=%u\n",
           s.columns,s.tokens,s.pos0,s.chunk,TK_SAMPLES);
    const double ow=tk_statistics(old,false,"materialized wall"), nw=tk_statistics(candidate,false,"stream wall");
    const double og=tk_statistics(old,true,"materialized GPU"), ng=tk_statistics(candidate,true,"stream GPU");
    printf("  full scoring+selection gain wall=%+.3f%% GPU=%+.3f%%\n",100*(ow/nw-1),100*(og/ng-1));
    fflush(stdout); ss_free(&c);
}
int main(int argc, char **argv) {
    if (argc>2 || (argc==2 && strcmp(argv[1],"--bench"))) return 2;
    @autoreleasepool {
        tk_require(ds4_gpu_init(), "Metal stream init");
        const ss_shape shapes[] = {
            {2049,1,0,2048}, {2559,7,0,2048}, {2560,8,8192,2048},
            {2561,9,12000,2048}, {3071,15,0,2048}, {3072,16,12288,2048},
            {3073,17,4096,2048}, {4095,3,16380,2048}, {4096,5,0,2048},
            {4097,9,4096,2048}, {8193,17,32772,4096}, {16385,9,65540,8192},
            {65537,1,0,8192}, {4097,128,16388,2048},
        };
        ss_invalid();
        for (unsigned i=0;i<sizeof(shapes)/sizeof(shapes[0]);++i)
            for (unsigned pattern=0;pattern<3;++pattern)
                ss_test(shapes[i],pattern,i%3u==0u);
        g_quality_mode=false;
        ss_growth();
        printf("PASS: %u streaming score/top-k parity cases, 13 rejection cases; original IDs, CPU oracle, guards, input immutability and in-flight scratch growth.\n",ss_cases);
        if (argc==2) {
            puts("Complete score+top-k API, warm balanced ABBA; no model execution, no allocations inside timed calls.");
            const uint32_t columns[]={16384,65536};
            for (unsigned n=0;n<2;++n)
                for (unsigned t=0;t<2;++t)
                    ss_bench((ss_shape){columns[n],t ? 512u : 128u,4u*columns[n],8192});
        }
        ds4_gpu_cleanup();
    }
    return 0;
}
