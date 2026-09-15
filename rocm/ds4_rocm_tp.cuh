/* Tensor-parallel combine kernels for the ROCm backend (Window C, stage 1).
 *
 * The two-rank TP split leaves each rank with a partial block output; the
 * peer's partial arrives in a slot of the NHI receive pool and a combine
 * kernel adds it into the local activation.  World size is fixed at two, so
 * the combine is a single float add per element and is required to be
 * bit-exact against a CPU reference: the TP mode's divergence budget
 * belongs to the reduction ORDER of the split matvecs, never to the
 * combine itself.
 *
 * Memory-model rules follow the transport integration contract validated
 * on the thunderbolt_stream zero-copy campaign (strix-rdma
 * docs/TP_TRANSPORT_CONTRACT.md):
 *   - stamps travel in-band as the LAST word of the slot (rule 5);
 *   - wave-resident polling requires an uncached pool, and the poll /
 *     stamp pair is __hip_atomic_load(ACQUIRE, SYSTEM) /
 *     __hip_atomic_store(RELEASE, SYSTEM) (rules 6-7);
 *   - payload stores complete before the stamp: __syncthreads() between
 *     the payload write and the lane-0 stamp store (rule 7), which is
 *     why the fill and spin kernels run as a single workgroup — the
 *     measured 28 KiB exchange cost with this shape is 35.2 us.
 * Slot geometry mirrors the production ring: 8 frames x 4096 bytes per
 * message, payload at the base, stamp in the final word.
 */

#define DS4_ROCM_TP_FRAME_BYTES 4096u
#define DS4_ROCM_TP_SLOT_FRAMES 64u
#define DS4_ROCM_TP_SLOT_BYTES (DS4_ROCM_TP_FRAME_BYTES * DS4_ROCM_TP_SLOT_FRAMES)
#define DS4_ROCM_TP_STAMP_OFF (DS4_ROCM_TP_SLOT_BYTES - 4u)
#define DS4_ROCM_TP_PAYLOAD_FLOATS_MAX ((DS4_ROCM_TP_STAMP_OFF) / 4u)

/* Plain elementwise combine: acc[i] += remote[i].  Multi-workgroup; used
 * when arrival is guaranteed by a dispatch boundary (host-mediated or
 * event designs), where coarse memory is already coherent. */
__global__ void dsv4_tp_combine_f32_kernel(float *__restrict__ acc,
                                           const float *__restrict__ remote,
                                           uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += remote[i];
}

/* Production engine fill: copy a stable graph partial into the rotating UC
 * TX slot with a normal parallel grid.  A separately ordered one-thread
 * stamp kernel closes every producer workgroup before publication. */
__global__ void dsv4_tp_slot_copy_f32_kernel(unsigned char *__restrict__ slot,
                                              const float *__restrict__ src,
                                              uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ((float *)(void *)slot)[i] = src[i];
}

__global__ void dsv4_tp_stamp_release_kernel(unsigned char *__restrict__ slot,
                                              uint32_t stamp) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t *stamp_ptr = (uint32_t *)(void *)(slot + DS4_ROCM_TP_STAMP_OFF);
        __hip_atomic_store(stamp_ptr, stamp, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

/* Single-workgroup Stage-2 fill: copy the payload into the slot, then publish
 * the stamp.  The __syncthreads() waits every lane's outstanding stores
 * (contract rule 7), after which lane 0 releases the stamp; on an
 * uncached pool that store is also the NHI-visibility release for
 * persistent-sender designs (rule 8). */
__global__ void dsv4_tp_slot_fill_kernel(unsigned char *__restrict__ slot,
                                         const float *__restrict__ src,
                                         uint32_t n,
                                         uint32_t stamp) {
    float *payload = (float *)(void *)slot;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        payload[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t *stamp_ptr = (uint32_t *)(void *)(slot + DS4_ROCM_TP_STAMP_OFF);
        __hip_atomic_store(stamp_ptr, stamp, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

/* Single-workgroup spin-combine: lane 0 polls the slot stamp until it
 * carries the expected sequence value, then every lane adds the slot
 * payload into the accumulator.  The spin guard turns a lost stamp into a
 * detectable failure instead of a hang: after max_spins acquire loads the
 * kernel records the timeout and exits without touching the accumulator. */
/* Production fixed-view gate: wait on the rotating RX slot, then copy it
 * into the graph's stable tp_in tensor.  This kernel is the RX slot's final
 * reader; an event recorded immediately afterward is therefore the safe
 * repost boundary even though later kernels still consume tp_in. */
__global__ void dsv4_tp_spin_copy_f32_kernel(float *__restrict__ dst,
                                             const unsigned char *__restrict__ slot,
                                             uint32_t n,
                                             uint32_t expect_stamp,
                                             unsigned long long max_spins,
                                             uint32_t *__restrict__ failed) {
    __shared__ int arrived;
    const uint32_t *stamp_ptr =
        (const uint32_t *)(const void *)(slot + DS4_ROCM_TP_STAMP_OFF);
    if (threadIdx.x == 0) {
        arrived = 0;
        unsigned long long spins = 0;
        for (;;) {
            if (failed && __hip_atomic_load(failed, __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_SYSTEM) != 0)
                break;
            const uint32_t seen = __hip_atomic_load(stamp_ptr, __ATOMIC_ACQUIRE,
                                                    __HIP_MEMORY_SCOPE_SYSTEM);
            if (seen == expect_stamp) {
                arrived = 1;
                break;
            }
            if (++spins >= max_spins) {
                if (failed) {
                    __hip_atomic_store(failed, 1u, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
                }
                break;
            }
        }
    }
    __syncthreads();
    if (!arrived) return;
    const float *payload = (const float *)(const void *)slot;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = payload[i];
    }
}

__global__ void dsv4_tp_spin_combine_f32_kernel(float *__restrict__ acc,
                                                const unsigned char *__restrict__ slot,
                                                uint32_t n,
                                                uint32_t expect_stamp,
                                                unsigned long long max_spins,
                                                const uint32_t *__restrict__ stamp_override,
                                                int *__restrict__ timeout_flag) {
    __shared__ int arrived;
    const uint32_t *stamp_ptr = stamp_override
        ? stamp_override
        : (const uint32_t *)(const void *)(slot + DS4_ROCM_TP_STAMP_OFF);
    if (threadIdx.x == 0) {
        arrived = 0;
        unsigned long long spins = 0;
        for (;;) {
            const uint32_t seen = __hip_atomic_load(stamp_ptr, __ATOMIC_ACQUIRE,
                                                    __HIP_MEMORY_SCOPE_SYSTEM);
            if (seen == expect_stamp) {
                arrived = 1;
                break;
            }
            if (++spins >= max_spins) {
                if (timeout_flag) {
                    __hip_atomic_store(timeout_flag, 1, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
                }
                break;
            }
        }
    }
    __syncthreads();
    if (!arrived) return;
    const float *payload = (const float *)(const void *)slot;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        acc[i] += payload[i];
    }
}
