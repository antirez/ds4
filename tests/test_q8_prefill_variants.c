#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void require(int ok) { if (!ok) { fprintf(stderr, "Q8 prefill test failed\n"); exit(1); } }
static uint32_t seed=123;
static void *maps[32];
static unsigned map_count;
static uint32_t rnd(void) { seed^=seed<<13; seed^=seed>>17; seed^=seed<<5; return seed; }

/* Grow the shared unpack scratch before submitting any of its readers. With
 * unretained command buffers, every old allocation must stay alive until the
 * batch completes, even after a larger matrix replaces the global scratch. */
static void check_scratch_growth(void) {
    enum { D = 64, T = 33, N = 3 };
    const uint32_t rows[N] = {64, 192, 320};
    const uint64_t page = sysconf(_SC_PAGESIZE);
    uint64_t offsets[N], size = 0;
    for (unsigned m = 0; m < N; m++) {
        offsets[m] = size;
        const uint64_t bytes = (uint64_t)D / 32u * 34u * rows[m];
        size += (bytes + page - 1u) / page * page;
    }
    void *map = NULL;
    require(posix_memalign(&map, page, size) == 0);
    maps[map_count++] = map;
    memset(map, 0, size);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(T * D * sizeof(float));
    ds4_gpu_tensor *out[N];
    float *refs[N];
    float input[T * D], actual[T * 320 + 16];
    require(x != NULL);
    for (unsigned i = 0; i < T * D; i++) input[i] = ((int)(rnd() % 257) - 128) / 256.f;
    require(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));
    for (unsigned m = 0; m < N; m++) {
        out[m] = ds4_gpu_tensor_alloc(((uint64_t)T * rows[m] + 16u) * sizeof(float));
        refs[m] = malloc((uint64_t)T * rows[m] * sizeof(float));
        require(out[m] && refs[m]);
        for (uint64_t b = 0; b < (uint64_t)D / 32u * 34u * rows[m]; b += 34u) {
            uint8_t *p = (uint8_t *)map + offsets[m] + b;
            const uint16_t scale = 0x1800 + (rnd() % 16) * 0x100;
            memcpy(p, &scale, sizeof(scale));
            for (unsigned j = 2; j < 34; j++) p[j] = rnd();
        }
    }
    require(ds4_gpu_set_model_map(map, size));
    for (unsigned run = 0; run < 2; run++) {
        require(setenv("DS4_QWEN4_Q8_PREFILL_UNPACK", run ? "1" : "0", 1) == 0);
        for (unsigned m = 0; m < N; m++)
            require(ds4_gpu_tensor_fill_f32(out[m], NAN, (uint64_t)T * rows[m] + 16u));
        require(ds4_gpu_begin_commands());
        for (unsigned m = 0; m < N; m++)
            require(ds4_gpu_qwen4_matmul_q8_0_tensor(out[m], map, size, offsets[m], D, rows[m], x, T));
        require(ds4_gpu_end_commands());
        for (unsigned m = 0; m < N; m++) {
            const uint64_t count = (uint64_t)T * rows[m];
            require(ds4_gpu_tensor_read(out[m], 0, actual, (count + 16u) * sizeof(float)));
            for (uint64_t i = 0; i < count; i++) require(isfinite(actual[i]));
            for (uint64_t i = count; i < count + 16u; i++) require(isnan(actual[i]));
            if (!run) memcpy(refs[m], actual, count * sizeof(float));
            else require(memcmp(refs[m], actual, count * sizeof(float)) == 0);
        }
    }
    for (unsigned m = 0; m < N; m++) { ds4_gpu_tensor_free(out[m]); free(refs[m]); }
    ds4_gpu_tensor_free(x);
    puts("PASS Q8 scratch growth in one batch: exact outputs, guards intact");
}

static void check(uint32_t d, uint32_t o, uint32_t t) {
    const uint64_t bytes=(uint64_t)d/32*34*o, n=(uint64_t)t*o;
    const uint64_t page=sysconf(_SC_PAGESIZE), stride=(bytes+page-1)/page*page;
    void *map=NULL;
    require(posix_memalign(&map,page,2*stride)==0);
    maps[map_count++]=map;
    memset(map,0,2*stride);
    for (uint32_t matrix=0; matrix<2; matrix++) for (uint64_t b=0; b<bytes; b+=34) {
        uint8_t *p=(uint8_t *)map+matrix*stride+b;
        uint16_t scale=0x1800+(rnd()%16)*0x100;
        memcpy(p,&scale,2);
        for (int j=2; j<34; j++) p[j]=rnd();
    }
    require(ds4_gpu_set_model_map(map,2*stride));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((uint64_t)t*d*4);
    ds4_gpu_tensor *a=ds4_gpu_tensor_alloc((n+16)*4), *b=ds4_gpu_tensor_alloc((n+16)*4);
    require(x && a && b);
    float *input=malloc((uint64_t)t*d*4), *actual=malloc((n+16)*4);
    float *refs[2]={malloc(n*4),malloc(n*4)};
    require(input && actual && refs[0] && refs[1]);
    for (uint64_t i=0; i<(uint64_t)t*d; i++) input[i]=((int)(rnd()%257)-128)/256.f;
    require(ds4_gpu_tensor_write(x,0,input,(uint64_t)t*d*4));
    const char *variants[]={"0","1",NULL,"0",NULL};
    for (int run=0; run<5; run++) {
        if (variants[run]) require(setenv("DS4_QWEN4_Q8_PREFILL_UNPACK",variants[run],1)==0);
        else require(unsetenv("DS4_QWEN4_Q8_PREFILL_UNPACK")==0);
        require(ds4_gpu_tensor_fill_f32(a,NAN,n+16) && ds4_gpu_tensor_fill_f32(b,NAN,n+16));
        require(ds4_gpu_begin_commands());
        require(ds4_gpu_qwen4_matmul_q8_0_tensor(a,map,2*stride,0,d,o,x,t));
        require(ds4_gpu_qwen4_matmul_q8_0_tensor(b,map,2*stride,stride,d,o,x,t));
        require(ds4_gpu_end_commands());
        ds4_gpu_tensor *outputs[]={a,b};
        for (int m=0; m<2; m++) {
            require(ds4_gpu_tensor_read(outputs[m],0,actual,(n+16)*4));
            for (uint64_t i=0; i<n; i++) require(isfinite(actual[i]));
            for (uint64_t i=n; i<n+16; i++) require(isnan(actual[i]));
            if (!run) memcpy(refs[m],actual,n*4);
            else if (memcmp(refs[m],actual,n*4)) {
                fprintf(stderr,"Mismatch D=%u O=%u T=%u variant=%s matrix=%d\n",d,o,t,variants[run] ? variants[run] : "default",m);
                exit(1);
            }
        }
    }
    printf("PASS Q8 D=%u O=%u T=%u exact paired outputs, guards intact\n",d,o,t);
    ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(x);
    free(input); free(actual); free(refs[0]); free(refs[1]);
}
int main(int argc, char **argv) {
    const int unretained_child = argc == 2 && strcmp(argv[1], "--unretained-growth") == 0;
    require(argc == 1 || unretained_child);
    if (!unretained_child) {
        /* The runtime samples this environment once. Exec a fresh process
         * before Metal initialization so both ownership modes are exercised. */
        fflush(NULL);
        const pid_t child = fork();
        require(child >= 0);
        if (child == 0) {
            if (setenv("DS4_METAL_UNRETAINED_COMMAND_BUFFERS", "1", 1) != 0) _exit(1);
            execlp(argv[0], argv[0], "--unretained-growth", (char *)NULL);
            _exit(1);
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    require(ds4_gpu_init());
    check_scratch_growth();
    if (unretained_child) goto done;
    const uint32_t shapes[][3]={{32,1,32},{64,48,33},{96,63,63},{128,65,64},
        {2560,48,1},{2560,48,2},{2560,128,8},{2560,128,16},{2560,128,17},
        {2560,128,31},{2560,128,32},{2560,10240,128},{6144,2560,65},
        {2560,32768,32}}; /* decoded weights exceed the 128 MiB cap */
    for (unsigned i=0; i<sizeof(shapes)/sizeof(*shapes); i++) check(shapes[i][0],shapes[i][1],shapes[i][2]);
    ds4_gpu_set_ssd_streaming(true);
    check(64, 65, 8191);
    check(64, 65, 8192);
    ds4_gpu_set_ssd_streaming(false);
done:
    unsetenv("DS4_QWEN4_Q8_PREFILL_UNPACK");
    ds4_gpu_cleanup();
    for (unsigned i=0; i<map_count; i++) free(maps[i]);
    return 0;
}
