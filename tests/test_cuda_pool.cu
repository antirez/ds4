#include "ds4_gpu.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#define CUDA(call) assert((call) == cudaSuccess)

int main(void) {
    cudaMemPool_t pool;
    CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t original;
    CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &original));
    for (uint64_t initial : {0ull, 32ull << 20, 128ull << 20}) {
        CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &initial));
        for (int repeat = 0; repeat < 2; repeat++) {
            assert(ds4_gpu_init());
            uint64_t threshold, reserved;
            CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
            assert(threshold == (initial < (64ull << 20) ? (64ull << 20) : initial));
            void *scratch;
            CUDA(cudaMallocAsync(&scratch, 16ull << 20, 0));
            CUDA(cudaFreeAsync(scratch, 0));
            CUDA(cudaDeviceSynchronize());
            CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
            assert(reserved >= (16ull << 20));

            cudaStream_t stream;
            CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            void *result;
            CUDA(cudaMalloc(&result, 256));
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            CUDA(cudaMallocAsync(&scratch, 256, stream));
            CUDA(cudaMemsetAsync(scratch, 0x5a, 256, stream));
            CUDA(cudaMemcpyAsync(result, scratch, 256, cudaMemcpyDeviceToDevice, stream));
            CUDA(cudaFreeAsync(scratch, stream));
            CUDA(cudaStreamEndCapture(stream, &graph));
            CUDA(cudaGraphInstantiate(&exec, graph, 0));
            for (int launch = 0; launch < 3; launch++) {
                CUDA(cudaMemsetAsync(result, 0, 256, stream));
                CUDA(cudaGraphLaunch(exec, stream));
                CUDA(cudaStreamSynchronize(stream));
                unsigned char bytes[256];
                CUDA(cudaMemcpy(bytes, result, sizeof(bytes), cudaMemcpyDeviceToHost));
                for (auto byte : bytes) assert(byte == 0x5a);
            }
            CUDA(cudaGraphExecDestroy(exec));
            CUDA(cudaGraphDestroy(graph));
            CUDA(cudaFree(result));
            CUDA(cudaStreamDestroy(stream));
            ds4_gpu_cleanup();
            CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
            assert(threshold == initial);
        }
    }
    CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &original));
    puts("CUDA scratch retention, graph replay and teardown: PASS");
}
