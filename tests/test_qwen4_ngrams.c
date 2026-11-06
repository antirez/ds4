#define DS4_NO_GPU
#ifndef __APPLE__
#include <pthread.h>
static int test_pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
#define pthread_create test_pthread_create
#endif
#include "../ds4.c"
#ifndef __APPLE__
#undef pthread_create
static int thread_budget = -1;
static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start)(void *), void *arg) {
    if (!thread_budget) return EAGAIN;
    if (thread_budget > 0) thread_budget--;
    return pthread_create(thread,attr,start,arg);
}
#endif
#include <assert.h>
#include <sys/wait.h>

static void u32(FILE *f, uint32_t n) { assert(fwrite(&n, 4, 1, f) == 1); }
static void u64(FILE *f, uint64_t n) { assert(fwrite(&n, 8, 1, f) == 1); }
static void str(FILE *f, const char *s) { u64(f, strlen(s)); assert(fwrite(s, strlen(s), 1, f) == 1); }

static uint16_t value(size_t row, size_t col) {
    const uint16_t edge[] = {0, 0x8000, 1, 0x8001, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f};
    return col < sizeof(edge)/sizeof(*edge) ? edge[col] : (uint16_t)(0x3000 + (row*17+col) % 4096);
}

static void fixture(const char *path, bool bad_alignment, uint32_t type) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite("GGUF", 4, 1, f) == 1);
    u32(f, 3); u64(f, 2); u64(f, 1);
    str(f, "general.architecture"); u32(f, 8); str(f, "qwen4exp");
    str(f, "token_embd.weight"); u32(f, 2); u64(f, 1); u64(f, 1); u32(f, 0); u64(f, 0);
    str(f, "per_layer_token_embd.weight"); u32(f, 2); u64(f, 160); u64(f, 1000); u32(f, type);
    uint64_t start = ((uint64_t)ftell(f) + 8 + 31) / 32 * 32;
    u64(f, 65536 - start + (bad_alignment ? 1 : 0));
    assert(!fseek(f, (long)start, SEEK_SET));
    u32(f, 0x3f800000);
    assert(!fseek(f, 65536, SEEK_SET));
    for (size_t r = 0; r < 1000; r++) {
        for (size_t c = 0; c < 160; c++) {
            uint16_t v = value(r, c);
            uint8_t bytes[2] = {v & 255, v >> 8};
            assert(fwrite(bytes, 2, 1, f) == 1);
        }
    }
    if (bad_alignment) fputc(0, f);
    assert(!fclose(f));
}

int main(void) {
    char path[] = "/tmp/ds4-qwen-ngrams-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    fixture(path, false, 30);
    ds4_model m;
    model_open(&m, path, false, false);
    assert(m.size == 65536 && m.file_size == 65536 + 320000);
    assert(m.ngram_fd >= 0 && m.ngram_tensor && m.max_tensor_bytes == 4);
    assert(*(const float *)tensor_data(&m, model_find_tensor(&m, "token_embd.weight")) == 1.0f);
    enum { N = 9001 };
    uint32_t *rows = malloc(N * sizeof(*rows));
    float *out = malloc((size_t)N * 160 * sizeof(*out));
    assert(rows && out);
    for (size_t i = 0; i < N; i++) rows[i] = (i * 173u) % 1000;
    rows[1] = rows[0]; rows[N-1] = 999;
    const size_t sizes[] = {0,16,255,256,257,4095,4096,4097,N};
    for (size_t ni = 0; ni < sizeof(sizes)/sizeof(*sizes); ni++) {
        size_t n = sizes[ni];
        assert(qwen4_ngram_read(&m, rows, n, out));
        for (size_t i = 0; i < n; i++) {
            for (size_t c = 0; c < 160; c++) {
                uint32_t bits, expected = (uint32_t)value(rows[i], c) << 16;
                memcpy(&bits, out + i*160+c, 4);
                assert(bits == expected);
            }
        }
    }
#ifndef __APPLE__
    const int budgets[] = {0,1,7};
    for (size_t bi = 0; bi < sizeof(budgets)/sizeof(*budgets); bi++) {
        thread_budget = budgets[bi];
        memset(out,0xff,(size_t)N*160*sizeof(*out));
        assert(qwen4_ngram_read(&m,rows,N,out));
        for (size_t i = 0; i < N; i++) for (size_t c = 0; c < 160; c++) {
            uint32_t bits;
            memcpy(&bits,out+i*160+c,4);
            assert(bits == (uint32_t)value(rows[i],c)<<16);
        }
    }
    thread_budget = -1;
#endif
    uint32_t invalid = 1000;
    assert(!qwen4_ngram_read(&m, &invalid, 1, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, rows, SIZE_MAX, out) && errno == EINVAL);
    assert(!qwen4_ngram_read(&m, NULL, 1, out) && errno == EINVAL);
    fd = open(path, O_WRONLY);
    assert(fd >= 0);
    uint8_t nan[2] = {0xc0, 0x7f};
    assert(pwrite(fd, nan, 2, 65536) == 2);
    uint32_t first = 0;
    assert(!qwen4_ngram_read(&m, &first, 1, out) && errno == EDOM);
    assert(!qwen4_ngram_read(&m, rows, N, out) && errno == EDOM);
    uint16_t zero = 0;
    assert(pwrite(fd,&zero,2,65536) == 2);
    assert(qwen4_ngram_read(&m,rows,N,out));
    assert(!ftruncate(fd, 65536 + 320000 - 1));
    uint32_t last = 999;
    assert(!qwen4_ngram_read(&m, &last, 1, out) && errno == EIO);
    assert(!qwen4_ngram_read(&m,rows,N,out) && errno == EIO);
    close(fd);
    model_close(&m);
    assert(m.ngram_fd == -1 && !m.ngram_tensor);
    for (int bad = 0; bad < 4; bad++) {
        fixture(path, bad == 0, bad == 1 ? 1 : bad == 2 ? 3 : 30);
        if (bad == 3) assert(!truncate(path, 65536 + 320000 - 1));
        pid_t pid = fork();
        assert(pid >= 0);
        if (!pid) { model_open(&m, path, false, false); _exit(0); }
        int status;
        assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) != 0);
    }
    free(rows); free(out); unlink(path);
    puts("Qwen BF16 n-grams: disk-only mapping, exact reads, batches and errors OK");
    return 0;
}
