/* Real Metal objects with a small synthetic cache: exercise the existing
 * relief/clear paths without exhausting the host's memory. */
#include "../ds4_metal.m"
#include <assert.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static void add_slot(uint32_t expert) {
    const uint64_t part_bytes = 256u * 1024u;
    id<MTLBuffer> gate = nil, up = nil, down = nil;
    NSUInteger gate_inner = 0, up_inner = 0, down_inner = 0;
    assert(ds4_gpu_stream_expert_alloc_slab_slot(part_bytes, part_bytes,
        &gate, &up, &down, &gate_inner, &up_inner, &down_inner));
    uint32_t slot;
    assert(ds4_gpu_stream_expert_slab_slot_for_buffer(gate, gate_inner, &slot));
    memset((char *)gate.contents + gate_inner, expert + 1, 3 * part_bytes);
    assert(ds4_gpu_stream_expert_slab_lock_slot(slot));

    ds4_gpu_stream_expert_cache_entry *entry = &g_stream_expert_cache[0][expert];
    entry->valid = 1;
    entry->slab_backed = 1;
    entry->slab_slot = slot;
    entry->gate_buffer = gate;
    entry->up_buffer = up;
    entry->down_buffer = down;
    entry->gate_inner = gate_inner;
    entry->up_inner = up_inner;
    entry->down_inner = down_inner;
    entry->gate_expert_bytes = part_bytes;
    entry->down_expert_bytes = part_bytes;
    entry->logical_bytes = 3 * part_bytes;
    entry->last_used = expert;
    g_stream_expert_cache_entry_count++;
    g_stream_expert_cache_layer_count[0]++;
    g_stream_expert_cache_bytes += entry->logical_bytes;
}

static void submit_and_drain(void) {
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc fillBuffer:g_stream_expert_cache_slabs[0] range:NSMakeRange(0, 16) value:7];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    assert(cb.status == MTLCommandBufferStatusCompleted);
}

int main(void) {
    if (@available(macOS 15.0, *)) {
        @autoreleasepool {
            assert(ds4_gpu_init());
            unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS");
            setenv("DS4_METAL_STREAMING_SLAB_RESIDENCY", "1", 1);
            setenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB", "16", 1);
            ds4_gpu_set_ssd_streaming(true);
            ds4_gpu_set_streaming_expert_cache_budget(32);
            __weak id old_set = nil;
            __weak id old_buffer = nil;
            @autoreleasepool {
                for (uint32_t i = 0; i < 10; i++) add_slot(i);
                assert(g_stream_slab_residency_set);
                old_set = g_stream_slab_residency_set;
                old_buffer = g_stream_expert_cache_slabs[0];
                submit_and_drain();
                const uint64_t locked = g_stream_expert_cache_mlock_bytes;
                const int32_t protected = 0;
                assert(ds4_gpu_stream_expert_cache_release_mlock_margin(0, &protected, 1) == 1);
                assert(g_stream_expert_cache[0][0].valid);
                assert(g_stream_expert_cache_mlock_bytes == locked - 3 * 256 * 1024);
                assert(g_stream_expert_cache_entry_count == 9);
                assert(!g_stream_slab_residency_set);
                submit_and_drain();
                /* Even another allocation cannot reattach after relief. */
                id<MTLBuffer> extra = ds4_gpu_stream_expert_alloc_slab_buffer(
                    1024 * 1024, @"after relief");
                assert(extra && !g_stream_slab_residency_set);
                assert(ds4_gpu_stream_expert_cache_release_mlock_margin(0, &protected, 1) == 0);
            }
            assert(!old_set); /* The queue no longer retains the set. */
            @autoreleasepool {
                ds4_gpu_set_streaming_expert_cache_budget(32);
            }
            assert(!old_buffer);
            assert(g_stream_expert_cache_mlock_bytes == 0);
            assert(!g_stream_expert_cache_mlock_relief_applied);
            @autoreleasepool {
                add_slot(0);
                assert(g_stream_slab_residency_set);
                submit_and_drain();
                old_set = g_stream_slab_residency_set;
                old_buffer = g_stream_expert_cache_slabs[0];
                ds4_gpu_set_streaming_expert_cache_budget(16);
            }
            assert(!old_set && !old_buffer);
            puts("Metal slab residency: protected-slot relief, no reattach, rebuild and deallocation: PASS");
        }
    } else {
        puts("Metal slab residency: SKIP (requires macOS 15)");
    }
    return 0;
}
