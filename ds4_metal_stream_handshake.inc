/* Handshake for streamed Qwen single-token decode layers.
 *
 * The ordinary streamed MoE drains the GPU at every layer so the CPU can
 * read the router's expert ids, load the missing experts and bind them. A
 * single-token decode layer is instead encoded before its experts are known:
 *
 *   ... router(l) | copy ids | signal arrival || commit
 *   wait release | experts(l) via the layer's expert address table || commit
 *
 * The CPU then blocks on the arrival event, loads the missing experts, writes
 * their addresses into the table and signals release. The GPU idles only for
 * that service, not for a drain plus the encoding of the layer, and the CPU
 * encodes the reduce and the next layer while the experts run.
 *
 * The CPU writes expert payloads and addresses after the command buffer that
 * reads them was committed, so that buffer must not touch the expert cache
 * before its wait: it starts with the wait. (Running the cached experts while
 * the missing ones load, as the ordinary path does, changed the output: the
 * GPU can keep stale cache lines of slots the CPU rewrites while it runs.)
 *
 * Safety:
 * - The expert kernels reach the cache only through the address table and
 *   the slab buffers, which are all made resident at encode time; the
 *   handshake first allocates every slab of the cache budget.
 * - A layer's released experts are marked in flight with their command
 *   buffer's sequence. When the CPU sees layer l's ids, the service checks
 *   that the buffer which signalled them and every buffer committed before
 *   it have completed (none of them can be parked at a release) and retires
 *   all sequences up to that buffer before it loads. No load therefore waits
 *   for in-flight GPU work, which would wait on the buffer parked at the
 *   release. Layer l's own experts are protected by its load and then by
 *   their mark.
 * - Anything the handshake cannot serve, such as slabs it cannot allocate,
 *   slab slots the cache has lost or an expert the loader could only place
 *   in an explicit buffer, switches it off for the process with one message;
 *   the ordinary path then serves every layer.
 * - Failure is reported like the ordinary path's: the call returns 0 with the
 *   batch still open, the cached and shared slots computed, the missing slots
 *   untouched (their address entries stay zero and the kernels skip them),
 *   and the layer dropped from the cache so a retry is exact.
 *
 * Included by ds4_metal.m before ds4_gpu_qwen4_moe_stream_tensor. */

static id<MTLSharedEvent> g_qwen4_hs_arrival;  /* GPU -> CPU: the ids are in memory */
static id<MTLSharedEvent> g_qwen4_hs_release;  /* CPU -> GPU: the experts are bound */
static uint64_t g_qwen4_hs_arrival_value, g_qwen4_hs_release_value;
static id<MTLBuffer> g_qwen4_hs_ids;           /* shared copy of the router's ids */
static const char *g_qwen4_hs_off;             /* why the handshake switched itself off */
static struct { uint64_t layers; double wait_ms, service_ms; } g_qwen4_hs_stats;

static void qwen4_handshake_report(void) {
    const double n = g_qwen4_hs_stats.layers ? (double)g_qwen4_hs_stats.layers : 1.0;
    fprintf(stderr, "ds4: Qwen decode handshake: %llu layers; per layer: CPU wait for GPU %.3f ms, "
            "service %.3f ms\n", (unsigned long long)g_qwen4_hs_stats.layers,
            g_qwen4_hs_stats.wait_ms / n, g_qwen4_hs_stats.service_ms / n);
}

/* On by default; DS4_QWEN4_STREAM_HANDSHAKE=0 keeps the ordinary path. Read
 * on every layer so tests can switch paths. The early-load diagnostic loads
 * nothing, which a handshake layer with a missing expert could not survive. */
static int qwen4_stream_handshake_enabled(void) {
    return ds4_gpu_env_bool("DS4_QWEN4_STREAM_HANDSHAKE") != 0 &&
           getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD") == NULL;
}

static void qwen4_stream_handshake_reset(void) {
    g_qwen4_hs_arrival = nil;
    g_qwen4_hs_release = nil;
    g_qwen4_hs_ids = nil;
    g_qwen4_hs_off = NULL;
}

static void qwen4_handshake_switch_off(const char *why) {
    if (g_qwen4_hs_off) return;
    g_qwen4_hs_off = why;
    fprintf(stderr, "ds4: Qwen decode handshake off: %s; streamed layers take the ordinary path\n", why);
}

/* Allocates the rest of the expert cache's slabs. Kernels encoded before a
 * load reach every slab, so none may appear after encoding; the new slots go
 * to the free list unlocked and are locked when first used, as before. */
static int qwen4_handshake_reserve_slabs(uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    if (budget == 0 || budget > DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES) return 0;
    uint64_t slot_bytes = gate_expert_bytes * 2ull + down_expert_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    if (page) slot_bytes = round_up_u64(slot_bytes, page);
    if (slot_bytes == 0 || (g_stream_expert_cache_slab_slot_bytes &&
                            g_stream_expert_cache_slab_slot_bytes != slot_bytes)) return 0;
    if (g_stream_expert_cache_slab_total_slots >= budget) return 1;
    /* Slots are taken from the free list or the newest slab only, so hand
     * the newest slab's unused slots to the free list first. */
    if (g_stream_expert_cache_slab_count) {
        const uint32_t last = g_stream_expert_cache_slab_count - 1u;
        while (g_stream_expert_cache_slab_slots_used[last] < g_stream_expert_cache_slab_slot_count[last])
            ds4_gpu_stream_expert_slab_push_free_slot(g_stream_expert_cache_slab_start_slot[last] +
                                                      g_stream_expert_cache_slab_slots_used[last]++);
    }
    while (g_stream_expert_cache_slab_total_slots < budget) {
        if (g_stream_expert_cache_slab_count >= DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS) return 0;
        uint64_t slots = ds4_gpu_stream_expert_slab_target_bytes() / slot_bytes;
        if (slots == 0) slots = 1;
        if (slots > budget - g_stream_expert_cache_slab_total_slots)
            slots = budget - g_stream_expert_cache_slab_total_slots;
        id<MTLBuffer> slab = nil;
        for (; slots && !slab; slots /= 2u) {
            if (slots * slot_bytes <= (uint64_t)NSUIntegerMax)
                slab = ds4_gpu_stream_expert_alloc_slab_buffer(slots * slot_bytes, @"ds4_stream_expert_slab");
            if (slab) break;
        }
        if (!slab) return 0;
        const uint32_t i = g_stream_expert_cache_slab_count++;
        g_stream_expert_cache_slabs[i] = slab;
        g_stream_expert_cache_slab_start_slot[i] = g_stream_expert_cache_slab_total_slots;
        g_stream_expert_cache_slab_slot_count[i] = (uint32_t)slots;
        g_stream_expert_cache_slab_slots_used[i] = (uint32_t)slots;
        g_stream_expert_cache_slab_slot_bytes = slot_bytes;
        for (uint32_t k = 0; k < (uint32_t)slots; k++)
            ds4_gpu_stream_expert_slab_push_free_slot(g_stream_expert_cache_slab_total_slots + k);
        g_stream_expert_cache_slab_total_slots += (uint32_t)slots;
    }
    return 1;
}

/* Loads and binds the layer's experts once the GPU has published their ids:
 * one selected load, like the ordinary path. entry[] receives the layer's
 * cache entries. The release is signalled on every exit. n_pending counts
 * the pending command buffers up to and including the router's. */
static int qwen4_handshake_service(const ds4_gpu_stream_expert_table *t, uint32_t n_slots, uint64_t arrival,
                                   uint64_t router_seq, NSUInteger n_pending,
                                   ds4_gpu_stream_expert_cache_entry **entry, uint32_t *n_entries) {
    const double t0 = ds4_gpu_now_ms();
    int ok = [g_qwen4_hs_arrival waitUntilSignaledValue:arrival timeoutMS:60000];
    const double t1 = ds4_gpu_now_ms();
    g_qwen4_hs_stats.wait_ms += t1 - t0;
    if (!ok) fprintf(stderr, "ds4: timeout waiting for the Qwen router's expert ids\n");
    if (ok) {
        /* The in-flight marks up to the router's buffer are retired on those
         * buffers' completion, which Metal reports per buffer; the queue's
         * order alone says nothing about completion. None of these buffers
         * waits at a release, so this cannot block on the parked one. */
        if (n_pending > [g_pending_cbs count]) n_pending = [g_pending_cbs count];
        for (NSUInteger i = 0; i < n_pending; i++) {
            id<MTLCommandBuffer> cb = g_pending_cbs[i];
            if (cb.status != MTLCommandBufferStatusCompleted && cb.status != MTLCommandBufferStatusError)
                [cb waitUntilCompleted];
        }
        if (router_seq > g_stream_expert_cache_done_seq) g_stream_expert_cache_done_seq = router_seq;
    }

    const int32_t *ids = (const int32_t *)[g_qwen4_hs_ids contents];
    int32_t unique[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t n_unique = 0;
    uint32_t frequency[DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT] = {0};
    for (uint32_t i = 0; ok && i < n_slots; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= t->n_total_expert) ok = 0;
        else if (frequency[ids[i]]++ == 0) unique[n_unique++] = ids[i];
    }
    if (ok) {
        ds4_gpu_stream_expert_cache_note_token(t->layer);
        ds4_gpu_stream_expert_cache_note_frequency_hotness(t->layer, frequency, t->n_total_expert);
        ds4_gpu_stream_expert_pending_load_clear();
        /* A failed load returns its reservations, so a retry is exact. */
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(t, unique, n_unique) &&
             ds4_gpu_stream_expert_pending_load_finish(NULL);
    }
    /* The loader counted the hits; read the entries as the ordinary path does. */
    for (uint32_t i = 0; ok && i < n_unique; i++) {
        const uint32_t e = (uint32_t)unique[i];
        entry[i] = &g_stream_expert_cache[t->layer][e];
        if (entry[i]->valid && !entry[i]->slab_backed)
            qwen4_handshake_switch_off("an expert was loaded into an explicit buffer the kernels cannot see");
        ok = entry[i]->valid && entry[i]->slab_backed &&
             ds4_gpu_stream_expert_cache_set_addr_slot_raw(t->layer, e,
                 entry[i]->gate_buffer, entry[i]->gate_inner, entry[i]->up_buffer, entry[i]->up_inner,
                 entry[i]->down_buffer, entry[i]->down_inner);
    }
    *n_entries = ok ? n_unique : 0;
    g_qwen4_hs_release.signaledValue = g_qwen4_hs_release_value;
    if (!ok) {
        /* As after a failed batch read: the released kernels compute the
         * cached and shared slots and skip the rest (their table entries are
         * zero), then the layer's entries are dropped so a retry reloads them. */
        (void)ds4_gpu_synchronize();
        ds4_gpu_stream_expert_cache_clear_layer(t->layer);
    }
    const double service_ms = ds4_gpu_now_ms() - t1;
    g_qwen4_hs_stats.service_ms += service_ms;
    ds4_gpu_stream_expert_timing_note_selected(t1 - t0, 0.0, service_ms);
    return ok;
}

/* Encodes one streamed single-token decode layer with the handshake. Returns
 * 0 when the layer does not qualify (the ordinary path runs), else 1 with *ok
 * set as the ordinary path would set it. */
static int qwen4_stream_handshake_layer(
        int *ok, int had_batch, const ds4_gpu_stream_expert_table *table,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *part, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *selected, uint32_t gate_type, uint32_t down_type,
        uint32_t n_tokens, uint32_t n_slots, uint32_t in_dim, uint32_t ff_dim,
        uint64_t shared_gate_offset, uint64_t shared_up_offset, uint64_t shared_down_offset,
        uint32_t shared_type, uint32_t shared_down_type) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_budget();
    /* Q4_K gate/up rows take the grouped expert tables, not the address table. */
    if (g_qwen4_hs_off || !qwen4_stream_handshake_enabled() || !had_batch || !g_batch_cb || n_tokens != 1u ||
        n_slots > DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED || gate_type == 12u ||
        !ds4_gpu_stream_expert_slab_enabled() ||
        !ds4_gpu_stream_expert_cache_note_expert_size(table->gate_expert_bytes, table->down_expert_bytes) ||
        budget < n_slots ||
        ds4_gpu_stream_expert_cache_effective_cap(table->layer, table->n_total_expert, n_slots) == 0)
        return 0;
    if (g_stream_expert_cache_slab_total_slots < budget &&
        !qwen4_handshake_reserve_slabs(table->gate_expert_bytes, table->down_expert_bytes)) {
        qwen4_handshake_switch_off("the expert cache budget's slabs could not be allocated");
        return 0;
    }
    if (g_stream_expert_cache_slab_count == 0) return 0;
    /* Every load must find a free slot or an entry to reuse: a slot the cache
     * lost would send a load to an explicit buffer the kernels cannot see.
     * Slots are free, unused in the newest slab, entries or a pending load's. */
    {
        const uint32_t last = g_stream_expert_cache_slab_count - 1u;
        const uint64_t slots = (uint64_t)g_stream_expert_cache_free_slot_count +
            g_stream_expert_cache_slab_slot_count[last] - g_stream_expert_cache_slab_slots_used[last] +
            g_stream_expert_cache_entry_count +
            (g_stream_expert_pending_load.active ? g_stream_expert_pending_load.n_loads : 0u);
        if (slots < budget) {
            qwen4_handshake_switch_off("the expert cache lost slab slots");
            return 0;
        }
    }
    if (!g_qwen4_hs_release) {
        g_qwen4_hs_arrival = [g_device newSharedEvent];
        g_qwen4_hs_release = [g_device newSharedEvent];
        g_qwen4_hs_ids = [g_device newBufferWithLength:DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED * sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
        if (!g_qwen4_hs_arrival || !g_qwen4_hs_release || !g_qwen4_hs_ids) {
            qwen4_stream_handshake_reset();
            return 0;
        }
        g_qwen4_hs_arrival_value = g_qwen4_hs_arrival.signaledValue;
        g_qwen4_hs_release_value = g_qwen4_hs_release.signaledValue;
        static int report_registered;
        if (!report_registered && getenv("DS4_QWEN4_STREAM_HANDSHAKE_PROFILE")) {
            report_registered = 1;
            atexit(qwen4_handshake_report);
        }
    }
    if (!ds4_gpu_stream_expert_cache_ensure_addr_buffers(table->layer)) {
        *ok = 0;
        return 1;
    }

    /* ... router | copy ids | signal arrival || commit */
    ds4_gpu_close_batch_encoder();
    id<MTLBlitCommandEncoder> blit = [g_batch_cb blitCommandEncoder];
    [blit copyFromBuffer:ds4_gpu_tensor_buffer(selected) sourceOffset:ds4_gpu_tensor_offset(selected)
                toBuffer:g_qwen4_hs_ids destinationOffset:0 size:(NSUInteger)n_slots * sizeof(int32_t)];
    [blit endEncoding];
    const uint64_t arrival = ++g_qwen4_hs_arrival_value;
    [g_batch_cb encodeSignalEvent:g_qwen4_hs_arrival value:arrival];
    g_batch_has_work = YES;
    const uint64_t router_seq = g_stream_expert_cache_batch_seq;
    ds4_gpu_stream_expert_cache_entry *entries[DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED];
    uint32_t n_entries = 0;
    uint64_t experts_seq = 0;
    NSUInteger n_pending = 0;
    *ok = ds4_gpu_flush_commands();
    if (*ok) {
        n_pending = [g_pending_cbs count];
        /* wait release | experts via the address table || commit */
        [g_batch_cb encodeWaitForEvent:g_qwen4_hs_release value:++g_qwen4_hs_release_value];
        g_batch_has_work = YES;
        g_qwen4_hs_stats.layers++;
        qwen4_stream_weights stream = {
            .table = table,
            .gate = g_stream_expert_cache_gate_addr_buffers[table->layer],
            .up = g_stream_expert_cache_up_addr_buffers[table->layer],
            .down = g_stream_expert_cache_down_addr_buffers[table->layer],
            /* Every slab, including ones the cache replaced since the last layer. */
            .pending = [NSSet setWithObjects:g_stream_expert_cache_slabs count:g_stream_expert_cache_slab_count],
            .addresses = true,
        };
        g_qwen4_stream_weights = &stream;
        *ok = ds4_gpu_qwen4_moe_mid_tensor(mid, x, selected, table->model_map, table->model_size,
                table->gate_offset, table->up_offset, gate_type, table->n_total_expert,
                n_tokens, n_slots, in_dim, ff_dim, shared_gate_offset, shared_up_offset, shared_type) &&
              ds4_gpu_qwen4_moe_down_tensor(part, mid, selected, table->model_map, table->model_size,
                table->down_offset, down_type, table->n_total_expert,
                n_tokens, n_slots, ff_dim, in_dim, shared_down_offset, shared_down_type);
        g_qwen4_stream_weights = NULL;
        /* Commit without opening the successor: opening can fail and wait on
         * this buffer, which parks at the release until the service signals it. */
        experts_seq = g_stream_expert_cache_batch_seq;
        ds4_gpu_parallel_ffn_reset_state(YES);
        ds4_gpu_commit_batch_commands();
    }
    /* The GPU now waits for the release; service whatever happened above. A
     * failed flush encoded no wait and drained, so there is nothing to serve. */
    if (*ok && !qwen4_handshake_service(table, n_slots, arrival, router_seq, n_pending, entries, &n_entries))
        *ok = 0;
    /* Like the ordinary path, return with the caller's batch open. */
    if (!g_batch_cb && !ds4_gpu_begin_commands()) *ok = 0;
    /* The released kernels read these slots until their buffer completes. */
    for (uint32_t i = 0; *ok && i < n_entries; i++) entries[i]->inflight_seq = experts_seq;
#ifdef DS4_TEST_QWEN4_HANDSHAKE
    DS4_TEST_QWEN4_HANDSHAKE(*ok);
#endif
    return 1;
}
