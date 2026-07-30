/* Scheduler/worker (split move-only from generate.c): worker_main and its
 * quantum loop (classic / batched / fused mixed-batch lanes), job->slot
 * routing (choose_slot_for_job) and enqueue, lazy provisioning + the
 * admission/ledger/mem-floor helpers, LRU eviction and the warm-fork
 * make-room path, the proactive spill/restore guard, Tier-2 bank
 * switching, and the frontier/committed-pos readers. */
#include "pulsar_server_internal.h"



/* Tier-2 shared-pool bank switch (defined lower, near provision_slot). Installs
 * `bank`'s device views + host carry on the pool session, saving the outgoing
 * bank first. No-op in classic mode or when `bank` is already live. Called at
 * the top of every per-slot worker op (gen_begin, generate_job_step,
 * worker_evict_one) so all s->sess touches inside that op are live-correct. */
/* Returns false only when the target is a guard-spilled bank whose restore FAILED
 * (KV unrecoverable): the bank is NOT installed and cur_bank is unchanged, so the
 * caller MUST fail the request rather than sample/commit on the wrong bank's KV
 * (review finding 1). True on success (or classic mode / no-op). */
/* plan-33 inc D: evict ONE non-trunk victim (LRU-superseded preferred, else
 * plain LRU) so a warm fork has a free bank — trunk always protected. Defined
 * near worker_evict_one; forward-declared for the routing path above it. */



bool server::enqueue(job *j) {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    s->n_queued++;
    pthread_cond_signal(&s->cv);
    pthread_mutex_unlock(&s->mu);
    return true;
}



/* =========================================================================
 * Increment 3: job→slot binding + round-robin scheduler.
 *
 * Everything below runs on the single GPU worker thread. Client threads only
 * enqueue jobs (enqueue above, under mu) and block on the per-job condvar;
 * they never touch a slot or a pulsar_session. Over-capacity requests stay in
 * the FIFO queue until a slot frees (plan Tier 1 §1.4 "queue them until a
 * slot frees"); the queue is bounded by PULSAR_SERVER_MAX_CLIENTS, since every
 * queued job is one connected client thread.
 * ========================================================================= */

/* Context a request needs from a slot: prompt plus generation budget (plus a
 * small allowance for tool-error recovery injections), capped at the largest
 * (startup) slot so every request can always run on slot 0. */
int server::job_needed_ctx(const job *j) const {
    const auto *s = this; (void)s;
    int64_t need = (int64_t)j->req.prompt.len +
                   (int64_t)(j->req.max_tokens > 0 ? j->req.max_tokens
                                                   : s->default_tokens) +
                   64;
    if (need > s->slots[0].ctx_size) need = s->slots[0].ctx_size;
    if (need < 1) need = 1;
    return (int)need;
}



/* Reconcile a session's admission estimate with the bytes the allocator
 * actually committed (2026-07-13 lockup postmortem: the ledger must track
 * reality, not a formula). Logs the pair, warns loudly on >10% drift — that
 * means gpu_graph_session_bytes (gpu_diag.c) has fallen out of sync with
 * gpu_graph_alloc_raw_cap — and returns the value to commit to the ledger
 * (the actual, unless the engine could not measure one). */
uint64_t server_reconciled_session_cost(int slot_idx, int ctx,
                                        uint64_t est_bytes,
                                        uint64_t actual_bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: slot %d session cost: est=%.2f GiB actual=%.2f GiB (ctx=%d)",
               slot_idx, (double)est_bytes / gib, (double)actual_bytes / gib, ctx);
    if (actual_bytes == 0) return est_bytes;
    if (est_bytes > 0 &&
        (actual_bytes > est_bytes + est_bytes / 10 ||
         est_bytes > actual_bytes + actual_bytes / 10))
    {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: SESSION COST DRIFT >10%%: est=%.2f GiB vs actual=%.2f GiB "
                   "— gpu_graph_session_bytes is out of sync with gpu_graph_alloc_raw_cap "
                   "(or a deliberately unaccounted allocation is enabled: directional-"
                   "steering dirs are in the measured delta but excluded from the "
                   "estimate — see the exclusion list in gpu_diag.c); "
                   "fix the sizing code (gpu_diag.c) before trusting admission control",
                   (double)est_bytes / gib, (double)actual_bytes / gib);
    }
    return actual_bytes;
}



/* MemAvailable from /proc/meminfo, in bytes (0 on parse failure — the caller
 * fails closed and refuses provisioning). One read per slot-provisioning
 * attempt — never on a token/layer hot path. Coarse by design: driver 610's
 * UVM accounting lags MemAvailable, and under UVM pressure MemTotal itself
 * shrinks, so this is a belt-and-suspenders floor check on top of the
 * ledger, not a precise gauge. */
uint64_t server_mem_available_bytes(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char line[256];
    unsigned long long kib = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) break;
    }
    fclose(fp);
    return (uint64_t)kib * 1024ull;
}



/* Lazily provision a fresh slot. In pool mode (production) this is
 * provision_bank: pure host bookkeeping over the ONE pool session. In classic
 * mode (pool_banks == 0) only slot 0 exists — the single session created at
 * startup — so provisioning always refuses and the job queues for slot 0.
 * Returns NULL when the pool is at capacity or MemAvailable would drop below
 * the floor — the job then waits in the queue, exactly like an admission
 * rejection.
 * Since increment 4, an evicted slot leaves a hole (provisioned == false)
 * below n_slots; provisioning reuses the lowest hole before extending the
 * pool, so n_slots stays the high-water published count every reader iterates
 * by (they all skip unprovisioned entries). *refusal reports WHY provisioning
 * failed: the eviction path only acts on refusals eviction can actually
 * relieve (a full pool or a full ledger — never the MemAvailable floor,
 * which freed CUDA memory does not promptly move; see worker_try_bind). */
/* provision_refusal (enum) moved to pulsar_server_internal.h so the server
 * member methods provision_bank/provision_slot/choose_slot_for_job can name it. */

/* ===== Tier-2 shared-pool bank plumbing =================================== */

/* Lazy bank switch on the shared pool session. server_bank_switch(s, b) saves
 * the currently-installed bank's host carry + frontier counters and installs
 * bank b's (device views + carry). Idempotent when b is already live, and a
 * no-op in classic mode. The switch-away save is what keeps every idle bank's
 * carry current, so the per-bank frontier readers (pulsar_session_bank_*) and
 * server_slot_frontier_pos are always correct for non-live banks. */
/* Tier-2 2b: restore a guard-spilled bank's physical + raw KV from disk before it
 * is installed. Defined below (near the guard); forward-declared for the switch. */

bool server::bank_switch(int bank) {
    auto *s = this;
    if (s->pool_banks <= 0) return true;          /* classic mode */
    if (bank < 0 || bank >= s->pool_banks) return true;
    if (bank == s->live_bank && !s->slots[bank].spilled) return true; /* already installed */
    pulsar_session *pool = s->sess;
    /* Snapshot the outgoing bank so its carry stays readable while idle, and
     * record its committed frontier for metrics/routing. live_bank < 0 is the
     * post-batched-quantum sentinel: the pool holds no single bank's clean
     * state (multiseq left a cross-bank superset), so there is nothing safe to
     * save — just install the target (bank_state_restore clears the poison). */
    if (s->live_bank >= 0 && s->live_bank != bank) {
        s->slots[s->live_bank].committed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)s->live_bank);
    }
    /* Tier-2 2b: a guard-spilled target has no physical — reallocate + reload its
     * KV bit-identically from disk before installing (the reload-stall cost). On
     * restore failure the bank is NOT installed and cur_bank is unchanged; return
     * false so the caller fails the request (finding 1). */
    if (s->slots[bank].spilled) {
        if (!s->bank_restore_spilled(bank)) return false;
    }
    /* The state restore can ALSO fail (gpu_graph_bank_repoint rejects a bank
     * whose slabs are missing, e.g. after a partially-failed realloc).  Ignoring
     * it published live_bank over a half-repointed view set with cur_bank
     * unchanged, so the caller ran engine work against mixed banks instead of
     * failing the request.  Fail loud here — this is the one check that catches
     * a bad repoint before any token is generated. */
    if (!pulsar_session_bank_state_restore(pool, (uint32_t)bank)) return false;
    s->live_bank = bank;
    s->slots[bank].committed_pos = pulsar_session_pos(pool);
    return true;
}

int server::slot_frontier_pos(const session_slot *sl) const {
    const auto *s = this;
    if (!sl || !sl->provisioned) return 0;
    if (s->pool_banks > 0) return pulsar_session_bank_pos(s->sess, sl->bank);
    return pulsar_session_pos(s->sess);
}

int server::slot_common_prefix(const session_slot *sl,
                               const pulsar_tokens *prompt) const {
    const auto *s = this;
    if (!sl || !sl->provisioned) return 0;
    if (s->pool_banks > 0)
        return pulsar_session_bank_common_prefix(s->sess, sl->bank, prompt);
    return pulsar_session_common_prefix(s->sess, prompt);
}

/* Provision a bank in the shared pool (Tier-2). No GPU allocation happens here:
 * the whole n-bank pool was allocated and admitted ONCE at startup, so this is
 * pure host bookkeeping — find a free bank slot, install it, reset it to an
 * empty conversation (so gen_begin sees pos 0 / no stale prefix), and charge
 * the even-split per-bank marginal to the ledger. The only runtime pressure is
 * the demand-paged comp/index pages a bank touches as it fills; the belt-and-
 * suspenders MemAvailable floor still guards each provision. Returns NULL with
 * *refusal set on a full pool or a tight box (never on a create/admission
 * failure — there is no runtime create). */
session_slot *server::provision_bank(provision_refusal *refusal) {
    auto *s = this;
    *refusal = PROVISION_OK;
    int idx = -1;
    for (int i = 1; i < s->pool_banks; i++) {     /* bank 0 == slot 0, pinned */
        if (!s->slots[i].provisioned) { idx = i; break; }
    }
    if (idx < 0) { *refusal = PROVISION_REFUSED_POOL_FULL; return NULL; }
    /* Belt-and-suspenders: refuse if the box is physically tight (fail closed on
     * an unreadable gauge), matching provision_slot. The marginal is what the
     * bank may still demand-page as it fills. */
    const uint64_t avail = server_mem_available_bytes();
    if (avail == 0 || !server_mem_floor_admits(avail, s->bank_marginal_bytes)) {
        static bool warned; /* single worker thread */
        if (!warned) {
            warned = true;
            server_log(PULSAR_LOG_WARNING,
                       "pulsar-server: bank provisioning refused: MemAvailable %.2f GiB "
                       "below floor for marginal %.2f GiB (job queued)",
                       (double)avail / (1024.0 * 1024.0 * 1024.0),
                       (double)s->bank_marginal_bytes / (1024.0 * 1024.0 * 1024.0));
        }
        *refusal = PROVISION_REFUSED_MEM_FLOOR;
        return NULL;
    }
    session_slot *sl = &s->slots[idx];
    pulsar_session *pool = s->sess;
    /* Install and reset the bank to an empty conversation with a valid (empty)
     * host carry, so routing/metrics read pos 0 and gen_begin cold-prefills. A
     * free (SLOT_EVICTED) bank is never guard-spilled (spilled banks stay
     * provisioned), so this switch never restores — the result is
     * unconditionally true. */
    (void)s->bank_switch(idx);
    pulsar_session_invalidate(pool);
    pulsar_session_bank_state_save(pool, (uint32_t)idx);
    sl->provisioned = true;
    sl->bank = (uint32_t)idx;
    sl->committed_pos = 0;
    sl->state = SLOT_IDLE;
    sl->ctx_size = s->slots[0].ctx_size;          /* every bank shares pool ctx */
    sl->est_cost_bytes = s->bank_marginal_bytes;
    sl->tokens_emitted = 0;
    sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    sl->continued_last_store_tokens = 0;
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes += s->bank_marginal_bytes;
    if (idx >= s->n_slots) s->n_slots = idx + 1;
    pthread_mutex_unlock(&s->mu);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: provisioned bank %d (pooled) marginal=%.2f GiB "
               "total committed=%.2f GiB / %.2f GiB",
               idx,
               (double)s->bank_marginal_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_committed_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0));
    return sl;
}

session_slot *server::provision_slot(int ctx,
                                    provision_refusal *refusal) {
    auto *s = this;
    if (s->pool_banks > 0) { (void)ctx; return s->provision_bank(refusal); }
    /* Classic mode (pool_banks == 0): the ONE session IS slot 0, created at
     * startup — there is no independent session to create, so a request that
     * needs another slot queues until slot 0 frees. */
    (void)ctx;
    *refusal = PROVISION_REFUSED_POOL_FULL;
    return NULL;
}



/* Context a lazily provisioned slot would be created with for this job: the
 * secondary-slot default, raised to the job's need, capped at slot 0's ctx.
 * Shared by the provisioning path and the eviction could-it-help precheck so
 * they price the same session shape. */
int server::provision_ctx_for_job(const job *j) const {
    const auto *s = this;
    int ctx = PULSAR_SERVER_EXTRA_SLOT_CTX_TOKENS;
    if (ctx > s->slots[0].ctx_size) ctx = s->slots[0].ctx_size;
    const int needed = s->job_needed_ctx(j);
    if (ctx < needed) ctx = needed;
    return ctx;
}



/* Trivial-match classifier for the router's choose-vs-provision decision
 * (unit-tested in cli_main.c). A candidate slot's token-prefix match is
 * TRIVIAL — grounds to prefer provisioning a fresh slot over reusing (and
 * clobbering) the candidate — only when BOTH hold:
 *   - common < trivial_tokens: the match is no deeper than the rendered
 *     template header plus incidental prologue overlap (threshold derived at
 *     startup, cli_main.c / PULSAR_SERVER_SLOT_TRIVIAL_ALLOWANCE_TOKENS), so it
 *     does not indicate the same conversation;
 *   - slot_pos - common >= trivial_tokens: reuse would destroy a meaningful
 *     amount of some conversation's warm KV. When the slot holds less than
 *     that past the match, clobbering costs at worst a sub-threshold
 *     re-prefill (sub-second) — always cheaper than a multi-GiB,
 *     multi-second session create — and this clause also keeps short
 *     same-conversation continuations (whose common covers nearly the whole
 *     slot state) on their warm slot. An empty slot (slot_pos == 0) is
 *     never "clobbered": it is simply free.
 * Deliberate semantic change from v0.2.0 (common == 0 && pos > 0): a slot
 * holding only a sub-threshold warm tail past the match is now reused
 * (clobbered) rather than protected by a fresh provisioning — protecting
 * <threshold tokens of KV is never worth seconds of session create. */
bool server_slot_match_is_trivial(int common, int slot_pos,
                                  int trivial_tokens) {
    return common < trivial_tokens && slot_pos - common >= trivial_tokens;
}



/* Route the job to a slot. Preferences, in order:
 *   1. A live-tool-state continuation binds to the slot that owns its call
 *      ids (waiting for it if busy — running it elsewhere could only 409).
 *      A continuation whose prompt cannot fit its owner slot's context can
 *      never run: it must not run elsewhere (the live tool state exists only
 *      on the owner), and leaving it queued would wedge the FIFO forever
 *      behind an unbindable head — so it is failed explicitly through
 *      *reject_ctx with the same context_length_exceeded client error the
 *      front door sends (http_server.c / request_exceeds_context; the front
 *      door checks against slot 0's ctx and cannot see the owner's smaller
 *      one).
 *   2. A free fitting slot whose live thinking binding byte-matches the
 *      request's visible transcript is that conversation's warm continuation
 *      and wins outright (thinking_live_binds_prompt): for thinking chats
 *      the client replays visible content while the slot's frontier holds
 *      the hidden reasoning, so the token common prefix understates
 *      relatedness and must not out-vote the binding.
 *   3. Among free slots with enough context, the longest common token prefix
 *      wins, keeping a client's follow-ups on their warm KV.
 *   4. A job whose best token match is TRIVIAL — header-deep only, against a
 *      slot holding meaningful warm state past the match
 *      (server_slot_match_is_trivial) — prefers a fresh lazily provisioned
 *      slot over clobbering that conversation (budget permitting); with the
 *      pool exhausted it falls back to the warmest free slot exactly like
 *      the single-session server did. (Through v0.2.0 this gate required
 *      common == 0, which rendered chat traffic can never produce — every
 *      rendered prompt shares the template header, measured 4–9 common
 *      tokens across distinct conversations — so sequential conversations
 *      always clobbered slot 0 and the pool never provisioned; task #24
 *      bounce repro, fixed in task #30.)
 * Returns NULL when the job must wait for a slot to free — except when
 * *reject_ctx is set nonzero (the owner slot's ctx_size), which means the
 * job can never run and must be failed, not left queued. *waiting_owner is
 * set when the NULL means "the continuation's owner slot is busy": eviction
 * cannot help that job, only the owner finishing can. *clobbers is set when
 * the returned slot would overwrite another conversation's warm KV — the
 * caller may prefer evict(LRU)+provision over that (increment 4). *refusal
 * reports why a fresh provisioning was refused (PROVISION_OK when none was
 * attempted or it succeeded) so the eviction path can act only on refusals
 * eviction relieves. */
session_slot *server::choose_slot_for_job(job *j, int *reject_ctx,
                                         bool *waiting_owner, bool *clobbers,
                                         provision_refusal *refusal) {
    auto *s = this;
    *waiting_owner = false;
    *clobbers = false;
    *refusal = PROVISION_OK;
    session_slot *owner = NULL;
    if (j->req.api == API_RESPONSES && j->req.responses_live_call_ids.len > 0) {
        owner = s->responses_live_slot_for_ids(&j->req.responses_live_call_ids);
    } else if (j->req.api == API_ANTHROPIC &&
               j->req.anthropic_live_call_ids.len > 0) {
        owner = s->anthropic_live_slot_for_ids(&j->req.anthropic_live_call_ids);
    }
    if (owner) {
        if (request_exceeds_context(&j->req, owner->ctx_size)) {
            *reject_ctx = owner->ctx_size;
            return NULL;
        }
        *waiting_owner = owner->active_job != NULL;
        return owner->active_job ? NULL : owner;
    }

    const int needed = s->job_needed_ctx(j);
    session_slot *best = NULL;
    int best_common = -1;
    session_slot *bound = NULL;   /* thinking-binding match (preference 2) */
    size_t bound_visible = 0;     /* longest key = most recent frontier */
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        if (sl->active_job || !sl->provisioned) continue;
        if (sl->ctx_size < needed) continue;
        const size_t visible =
            s->thinking_live_binds_prompt(sl, &j->req,
                                       s->slot_frontier_pos(sl));
        if (visible > bound_visible) {
            bound_visible = visible;
            bound = sl;
        }
        const int common = s->slot_common_prefix(sl, &j->req.prompt);
        if (common > best_common) {
            best_common = common;
            best = sl;
        }
    }
    if (bound) return bound;
    const bool best_clobbers_warm_state =
        best && server_slot_match_is_trivial(best_common,
                                             s->slot_frontier_pos(best),
                                             s->slot_trivial_common_tokens);
    if (!best || best_clobbers_warm_state) {
        if (best_clobbers_warm_state) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: slot routing: best match is trivial "
                       "(common=%d pos=%d threshold=%d); preferring a fresh slot",
                       best_common, s->slot_frontier_pos(best),
                       s->slot_trivial_common_tokens);
        }
        session_slot *fresh = s->provision_slot(s->provision_ctx_for_job(j),
                                             refusal);
        if (fresh) return fresh;
    }
    /* plan-33 inc B/D: warm FORK routing. `best`'s committed history shares a
     * token prefix `best_common` with the request (validated bit-for-bit inside
     * the fork before any device write). Instead of continuing IN PLACE (which
     * consumes the trunk — today's behavior), fork the trunk into another bank
     * and continue there, leaving the trunk INTACT so a divergent sibling keeps
     * matching it:
     *   - inc B FULL fork   : best_common == trunk frontier  -> pulsar_session_bank_fork
     *                         (dst resumes at the exact frontier; re-prefill suffix).
     *   - inc D PARTIAL cut : warm_partial_min <= best_common < frontier
     *                         -> pulsar_session_bank_fork_partial (engine aligns down
     *                         to R, byte-stashes the ratio-4 boundary row; dst's
     *                         committed history becomes tokens[0..R), re-prefill [R..).
     * A free bank is used first; else one non-trunk victim is evicted (make_room,
     * LRU-superseded preferred). Any refusal (history moved, evicted src, cut
     * below R, no evictable bank) degrades to today's path — never a client error. */
    const int frontier = best ? s->slot_frontier_pos(best) : 0;
    const bool warm_ok = s->pool_banks > 0 && s->warm_fork_enabled && best &&
                         !best_clobbers_warm_state && !best->active_job &&
                         best_common > 0 && best_common < j->req.prompt.len;
    const bool full    = warm_ok && best_common == frontier;                 /* inc B */
    const bool partial = warm_ok && !full && best_common >= s->warm_partial_min &&
                         best_common < frontier;                             /* inc D */
    /* Always-on routing-decision inputs, so a 0-fork count is never silent (the
     * verbose KVCACHE stream; one line per bind, not per token). Confirmed nuance:
     * re-tokenized generated tail rarely reproduces the trunk's exact frontier, so
     * best_common < frontier (the PARTIAL path) is the common case; full is the
     * rare exact-continuation. */
    if (s->pool_banks > 0 && s->warm_fork_enabled)
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: route: best bank %d common %d frontier %d prompt %d "
                   "partial_min %d -> %s%s%s",
                   best ? (int)best->bank : -1, best_common, frontier,
                   j->req.prompt.len, s->warm_partial_min,
                   full ? "FORK-full" : partial ? "FORK-partial" : "in-place/cold",
                   best_clobbers_warm_state ? " [trivial]" : "",
                   (best && best->active_job) ? " [busy]" : "");
    if (full || partial) {
        provision_refusal fr;
        session_slot *dst = s->provision_slot(s->provision_ctx_for_job(j), &fr);
        if (!dst && fr == PROVISION_REFUSED_POOL_FULL &&
            s->fork_make_room(best)) {
            /* Freed one non-trunk victim; the trunk was protected. Retry once. */
            dst = s->provision_slot(s->provision_ctx_for_job(j), &fr);
        }
        if (dst && dst != best) {
            pulsar_session *pool = s->sess;
            const int rc = full
                ? pulsar_session_bank_fork(pool, best->bank, dst->bank,
                                        j->req.prompt.v, best_common)
                : pulsar_session_bank_fork_partial(pool, best->bank, dst->bank,
                                                j->req.prompt.v, best_common);
            if (rc == 0) {
                /* FULL resumes at best_common; PARTIAL resumes at the engine's
                 * R-aligned cut (read it back rather than recompute the align). */
                dst->committed_pos = full ? best_common
                                          : pulsar_session_bank_pos(pool, dst->bank);
                server_log(PULSAR_LOG_DEFAULT,
                           "pulsar-server: warm-fork-%s: trunk bank %u (frontier %d, "
                           "common %d) -> bank %u (resume %d); trunk preserved",
                           full ? "full" : "partial", best->bank, frontier,
                           best_common, dst->bank, dst->committed_pos);
                *clobbers = false;
                return dst;
            }
            /* Refused (history moved / evicted src / cut below R): dst stays a
             * fresh empty bank — cold on it is safe and beats clobbering best. */
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: warm-fork-%s refused (bank %u); cold on bank %u",
                       full ? "full" : "partial", best->bank, dst->bank);
            *clobbers = false;
            return dst;
        }
        /* No free/evictable bank for the fork: fall through to the divergent
         * guard below (reuses in place only for a linear continuation; otherwise
         * queues rather than clobber). */
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: warm-fork-%s wanted (bank %u common %d) but no free bank",
                   full ? "full" : "partial", best->bank, best_common);
    }
    /* CROSS-WIRE ROOT FIX: in-place continuation is safe ONLY for a linear
     * extension of THIS bank's own conversation (best_common == frontier). When
     * best_common < frontier the bank holds a DIFFERENT conversation past the
     * shared prefix (typically just the system-prompt header); continuing in place
     * REWINDS-and-clobbers it. On the shared pool session that conversation is
     * often still live under concurrency, so the two interleave and a request
     * decodes another's KV (the cross-wire). Provision a FRESH bank instead (the
     * shared prefix is cheap and prefix-cached); queue if the pool is full so a
     * non-active bank can be evicted. Deep divergent matches (best_common >=
     * warm_partial_min) already FORKED above and never reach here. */
    if (best && best_common < frontier) {
        session_slot *fresh = s->provision_slot(s->provision_ctx_for_job(j), refusal);
        if (fresh) {
            server_log(PULSAR_LOG_KVCACHE,
                       "pulsar-server: divergent match bank %u (common %d < frontier %d): "
                       "fresh bank %u, no in-place clobber",
                       best->bank, best_common, frontier, fresh->bank);
            *clobbers = false;
            return fresh;
        }
        /* Provision failed (pool full). Queue to avoid an in-place clobber ONLY
         * when an active job is running: then the queue drains as it finishes AND
         * there is a live conversation on the pool worth protecting. When NOTHING
         * is running, no bank will ever free (idle banks may all be protected), so
         * queuing would LIVE-LOCK the worker — and with no live reader there is
         * nothing to corrupt — so fall through to in-place continuation (progress
         * over warmth). This is the safe half of the cross-wire guard: the
         * corruption only ever happened while a concurrent conversation was live. */
        bool any_active = false;
        for (int i = 0; i < s->n_slots; i++)
            if (s->slots[i].active_job) { any_active = true; break; }
        if (any_active) {
            if (*refusal == PROVISION_OK) *refusal = PROVISION_REFUSED_POOL_FULL;
            return NULL;   /* an active job will free a bank; worker retries */
        }
        /* else: fall through to in-place (no live reader; queue would deadlock) */
    }
    *clobbers = best_clobbers_warm_state;
    return best;
}



/* =========================================================================
 * Increment 4: LRU eviction of idle slots.
 *
 * Sessions were previously never freed at runtime: once the admission budget
 * was consumed by idle-but-warm sessions, new conversations either queued
 * forever (no fitting free slot) or silently clobbered the warmest idle slot
 * chosen by scan order. Now, when the queue head cannot be placed cleanly —
 * no fitting free slot, or only a slot whose warm KV belongs to another
 * conversation — AND provisioning was refused by a constraint eviction can
 * actually relieve (a full pool or a full admission ledger), the worker
 * evicts the least-recently-serviced IDLE slot: snapshot its KV to the disk
 * kv cache (the same LRU store used for shutdown/cold checkpoints — plan
 * Tier 1 §1.3 "spill target"), free the session, release its ACTUAL
 * committed bytes from the ledger, and retry placement (the freed budget +
 * slot entry let provisioning succeed). The clobber fallback remains when
 * eviction cannot help: it is an in-place eviction of that one slot
 * (gen_begin disk-stores the displaced state), so correctness is identical —
 * LRU eviction just picks a better victim and keeps warmer conversations
 * alive.
 *
 * MemAvailable-floor refusals deliberately do NOT trigger eviction. Measured
 * on the GB10 (driver 610, 2026-07-14 smoke): freeing two 2.5 GiB sessions
 * moved MemAvailable only 5.98 -> 6.07 GiB — cudaFree'd memory does not
 * promptly return to the kernel's gauge, so post-eviction provisioning kept
 * refusing on the same floor while two warm conversations were lost for
 * nothing (and the server degraded to one effective slot). A floor refusal
 * means the machine is physically tight; the honest response is to queue the
 * head until a slot frees, not to churn snapshots.
 *
 * Restore path: none of it is new. gen_begin's cache resolution already
 * prefers a disk-text snapshot over cold prefill (kv_cache_try_load), and the
 * "evict" snapshot is keyed exactly like a shutdown checkpoint — by rendered
 * transcript bytes, or by the visible protocol transcript when a live
 * responses/thinking binding covers the frontier. A returning client binds to
 * any free slot and restores from disk there; no extra routing metadata is
 * needed because the kvstore's text-prefix index IS the metadata. Live
 * tool-state continuations of an evicted slot get the protocol's honest 409
 * (their sampled frontier is gone), exactly like a server restart.
 *
 * Slot 0 is PINNED — never evicted: (a) client threads read
 * pulsar_session_ctx(s->sess) lock-free (http_server.c) under the
 * CUDA-state audit's immutable-after-startup exception, so freeing that
 * session would be a data race; (b) slot 0 is the only slot guaranteed to fit
 * any admissible request (job_needed_ctx caps at its ctx), which preserves
 * the scheduler invariant that an all-idle pool always binds the head (the
 * worker sleeps on the condvar when nothing is active) and guarantees the
 * pool never reaches zero slots; (c) as the largest session it is the most
 * expensive one to bring back. When small sessions need room, evicting the
 * idle secondaries (this code) frees the same budget without touching it.
 *
 * Like lazy provisioning, eviction is a deliberate multi-second quantum
 * overshoot on the single GPU worker thread: the snapshot forces a full
 * device sync + a multi-GiB D2H copy + a disk write, and pulsar_session_free
 * tears down the graph. It happens only at a scheduling boundary (bind time),
 * and only when the alternative is queueing forever.
 * ========================================================================= */

uint64_t server_ledger_release(uint64_t committed_total, uint64_t slot_cost) {
    if (slot_cost > committed_total) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: EVICTION LEDGER UNDERFLOW: releasing %.2f GiB from "
                   "%.2f GiB committed — provision/evict pairing is out of sync; "
                   "clamping the ledger to 0 (the MemAvailable floor remains the "
                   "backstop against the resulting over-admission)",
                   (double)slot_cost / (1024.0 * 1024.0 * 1024.0),
                   (double)committed_total / (1024.0 * 1024.0 * 1024.0));
        return 0;
    }
    return committed_total - slot_cost;
}



int server_evict_pick_victim(const session_slot *slots, int n_slots,
                             const bool *protect) {
    int victim = -1;
    for (int i = 1; i < n_slots; i++) { /* slot 0 pinned (see block comment) */
        const session_slot *sl = &slots[i];
        if (!sl->provisioned || sl->active_job) continue;
        if (protect && protect[i]) continue;
        if (victim < 0 ||
            sl->last_serviced_us < slots[victim].last_serviced_us ||
            (sl->last_serviced_us == slots[victim].last_serviced_us &&
             sl->est_cost_bytes < slots[victim].est_cost_bytes))
        {
            victim = i;
        }
    }
    return victim;
}



/* Mark slots some QUEUED live-tool-state continuation still needs: that KV
 * frontier exists only on its owner slot, so evicting it would turn the
 * queued job into a 409 the moment it binds. The queue is snapshotted under
 * mu; the job pointers stay valid afterwards because only this worker pops
 * jobs and each client thread blocks on its job condvar until then. The
 * owner lookups (tool_mu + session pos) run after mu is released — the two
 * locks are never nested. */
void server::worker_protect_queued_owner_slots(bool protect[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    memset(protect, 0, sizeof(protect[0]) * PULSAR_SESSION_POOL_CAP);
    job *queued[PULSAR_SERVER_MAX_CLIENTS];
    int n = 0;
    pthread_mutex_lock(&s->mu);
    for (job *q = s->head; q && n < PULSAR_SERVER_MAX_CLIENTS; q = q->next) {
        queued[n++] = q;
    }
    pthread_mutex_unlock(&s->mu);
    for (int i = 0; i < n; i++) {
        const request *r = &queued[i]->req;
        session_slot *owner = NULL;
        if (r->api == API_RESPONSES && r->responses_live_call_ids.len > 0) {
            owner = s->responses_live_slot_for_ids(&r->responses_live_call_ids);
        } else if (r->api == API_ANTHROPIC &&
                   r->anthropic_live_call_ids.len > 0) {
            owner = s->anthropic_live_slot_for_ids(&r->anthropic_live_call_ids);
        }
        if (owner) protect[owner - s->slots] = true;
    }
}



/* Pointless-eviction guard #2: evicting is only worth its cost if releasing
 * idle sessions can actually admit the provisioning the head job needs.
 * If even reclaiming EVERY unprotected idle slot leaves admission refusing,
 * skip eviction entirely — the head is genuinely waiting for a busy slot to
 * free, and evicting warm idle sessions would only churn snapshots. (Host
 * arithmetic only: pulsar_engine_session_cost_bytes is the same sizing code the
 * allocator uses, no CUDA work; runs only on failed bind attempts. Guard #1
 * is the provisioning-refusal reason check in worker_try_bind.) */
bool server::worker_eviction_could_help(const job *j,
                                       const bool *protect) {
    auto *s = this;
    const uint64_t est =
        pulsar_engine_session_cost_bytes(s->engine, s->provision_ctx_for_job(j));
    if (est == 0) return false;
    /* Model the MemAvailable floor too (2026-07-15 review): a POOL_FULL
     * refusal never consulted it, so on a physically tight box the gate
     * could open, one warm session be evicted, and the re-provisioning then
     * refuse on MEM_FLOOR anyway — the exact churn the refusal-reason gate
     * exists to prevent. Eviction does not promptly raise MemAvailable (see
     * the block comment above), so if the floor refuses NOW it will refuse
     * after the eviction too; skip. Fail closed on an unreadable gauge,
     * matching provision_slot. One /proc read per failed bind attempt. */
    const uint64_t avail = server_mem_available_bytes();
    if (!server_mem_floor_admits(avail, est)) return false;
    uint64_t reclaimable = 0;
    bool any = false;
    for (int i = 1; i < s->n_slots; i++) {
        const session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->active_job || (protect && protect[i])) continue;
        reclaimable += sl->est_cost_bytes;
        any = true;
    }
    if (!any) return false;
    pthread_mutex_lock(&s->mu);
    const uint64_t committed = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    const uint64_t after = committed > reclaimable ? committed - reclaimable : 0;
    return server_kv_admits(s->kv_budget_bytes, after, est);
}



/* Evict one idle slot (LRU victim): snapshot to the disk kv cache when
 * possible, free the session, release the ledger, and leave the slot entry
 * (provisioned == false) for provision_slot to reuse. Failure honesty: a failed or
 * unavailable snapshot only costs the returning client a re-prefill — the
 * eviction itself proceeds, and the response always belongs to the right
 * conversation because the freed KV can never be read again. Returns false
 * when nothing is evictable. Worker thread only. */
bool server::worker_evict_one(bool protect[PULSAR_SESSION_POOL_CAP]) {
    auto *s = this;
    /* plan-33: protect any bank that is a live fork SOURCE mid-clone from disk
     * eviction (belt-and-suspenders — fork and evict are both worker-thread ops
     * and never interleave, but the invariant "a pinned source is never freed"
     * must hold for both eviction paths). */
    if (s->pool_banks > 0 && s->sess) {
        for (int i = 1; i < s->n_slots && i < PULSAR_SESSION_POOL_CAP; i++) {
            if (pulsar_session_bank_fork_pinned(s->sess, s->slots[i].bank)) protect[i] = true;
        }
    }
    const int vi = server_evict_pick_victim(s->slots, s->n_slots, protect);
    if (vi < 0) return false;
    session_slot *sl = &s->slots[vi];

    /* Tier-2: install the victim's bank so the snapshot store below reads its
     * OWN frontier (not the pool's live cursor). server_bank_switch restores a
     * guard-spilled victim first; if that restore fails we cannot snapshot its KV,
     * so skip it (this path is discarding the conversation anyway). */
    const bool switched = s->bank_switch(sl->bank);
    const pulsar_tokens *tokens = switched ? pulsar_session_tokens(s->sess) : NULL;
    const int live_tokens = tokens ? tokens->len : 0;
    bool stored = false;
    if (switched && s->kv.enabled && live_tokens >= s->kv.opt.min_tokens) {
        stored = s->kv_cache_store_current(sl, "evict");
    }
    if (!stored && live_tokens > 0) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: slot %d evicting WITHOUT a disk snapshot (%s); "
                   "a returning client re-prefills (correctness unaffected)",
                   vi,
                   !s->kv.enabled ? "kv disk cache disabled"
                   : live_tokens < s->kv.opt.min_tokens
                       ? "conversation below kv-cache min-tokens"
                       : "snapshot write failed");
    }
    /* The live protocol bindings describe a sampled frontier that is about to
     * lose its GPU state; clear them AFTER the store (snapshot keying reads
     * them). A later continuation of those ids gets the protocol's 409. */
    s->responses_live_clear(sl);
    s->anthropic_live_clear(sl);
    s->thinking_live_clear(sl);

    const uint64_t committed = sl->est_cost_bytes;
    uint64_t freed = 0;
    /* Pooled mode only (classic mode never provisions a slot >= 1, so the
     * picker returns -1 above): the pool session (s->sess, shared by every
     * bank) persists. "Evicting" bank vi means reset it to an empty
     * conversation and drop its host carry so a fresh conversation can reuse
     * it — no pulsar_session_free (that would tear down the whole pool). The
     * demand-paged comp/index pages this bank touched stay resident; the
     * ledger releases only this bank's even-split marginal. */
    pulsar_session_invalidate(s->sess);
    pulsar_session_bank_state_save(s->sess, (uint32_t)sl->bank);
    freed = committed; /* logical release; no allocator delta to verify */
    const int evicted_ctx = sl->ctx_size;
    sl->provisioned = false;
    sl->gen = NULL;
    sl->active_job = NULL;
    sl->state = SLOT_EVICTED;
    sl->ctx_size = 0;
    sl->est_cost_bytes = 0;
    sl->tokens_emitted = 0;
    sl->last_serviced_us = 0;
    sl->continued_last_store_tokens = 0;
    /* Tier-2 2b: a slot being evicted for reuse must not carry a stale guard-spill
     * flag or leave an orphan spill file (invariant: physical freed IFF spilled).
     * server_bank_switch above restored a spilled victim (physical present, flag
     * cleared); belt-and-suspenders in case that restore failed. */
    if (s->pool_banks > 0 && sl->spilled) {
        char spath[600];
        snprintf(spath, sizeof spath, "%s/spill-bank-%u.kv", s->spill_dir, (unsigned)sl->bank);
        remove(spath);
        (void)pulsar_session_bank_alloc_physical(s->sess, sl->bank); /* empty physical for reuse */
        sl->spilled = false;
    }
    pthread_mutex_lock(&s->mu);
    s->kv_committed_bytes = server_ledger_release(s->kv_committed_bytes, committed);
    const uint64_t committed_now = s->kv_committed_bytes;
    pthread_mutex_unlock(&s->mu);
    protect[vi] = true; /* freed hole; never a candidate again this round */
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: evicted slot %d ctx=%d tokens=%d snapshot=%s "
               "released=%.2f GiB (allocator freed %.2f GiB) "
               "committed now %.2f / %.2f GiB, MemAvailable %.2f GiB",
               vi, evicted_ctx, live_tokens, stored ? "disk" : "none",
               (double)committed / (1024.0 * 1024.0 * 1024.0),
               (double)freed / (1024.0 * 1024.0 * 1024.0),
               (double)committed_now / (1024.0 * 1024.0 * 1024.0),
               (double)s->kv_budget_bytes / (1024.0 * 1024.0 * 1024.0),
               (double)server_mem_available_bytes() / (1024.0 * 1024.0 * 1024.0));
    return true;
}



/* plan-33 inc D victim policy: an idle bank is LRU-SUPERSEDED when its whole
 * committed history is a token-prefix of ANOTHER live bank's history (a sibling
 * that already extends past it) — its KV is redundant, so evicting it loses the
 * least. Returns such a slot's index (the least-recently-served among them), or
 * -1. Pure host reads (pulsar_session_bank_tokens / _common_prefix are the same
 * host-carry reads routing already uses on idle banks; no CUDA, no install). */
int server::pick_superseded_idle(const bool *protect) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    if (!pool) return -1;
    int victim = -1;
    for (int i = 1; i < s->n_slots; i++) {
        session_slot *a = &s->slots[i];
        if (!a->provisioned || a->active_job || (protect && protect[i])) continue;
        if (pulsar_session_bank_fork_pinned(pool, a->bank)) continue;
        const pulsar_tokens *at = pulsar_session_bank_tokens(pool, a->bank);
        if (!at || at->len == 0) continue;              /* empty: LRU handles it */
        bool superseded = false;
        for (int k = 1; k < s->n_slots && !superseded; k++) {
            if (k == i) continue;
            session_slot *b = &s->slots[k];
            if (!b->provisioned) continue;
            /* b supersedes a iff a's whole history is b's prefix AND b is strictly
             * longer (b can reconstruct everything a holds). */
            if (s->slot_frontier_pos(b) > at->len &&
                pulsar_session_bank_common_prefix(pool, b->bank, at) >= at->len) {
                superseded = true;
            }
        }
        if (!superseded) continue;
        if (victim < 0 ||
            a->last_serviced_us < s->slots[victim].last_serviced_us) {
            victim = i;
        }
    }
    return victim;
}

/* Evict exactly one NON-trunk victim so a warm fork gets a free bank. Trunk is
 * always protected (a sibling still matches it); LRU-superseded victims go
 * first, else plain LRU (worker_evict_one's picker). Reuses the proven eviction
 * body (snapshot + ledger release + bank reset). Worker thread only; returns
 * true when a bank was freed. */
bool server::fork_make_room(const session_slot *trunk) {
    auto *s = this;
    if (s->pool_banks <= 0 || !trunk) return false;
    bool protect[PULSAR_SESSION_POOL_CAP];
    s->worker_protect_queued_owner_slots(protect);      /* live-tool owners */
    const int ti = (int)(trunk - s->slots);
    if (ti >= 0 && ti < PULSAR_SESSION_POOL_CAP) protect[ti] = true;  /* NEVER the trunk */
    const int sup = s->pick_superseded_idle(protect);
    if (sup >= 0) {
        /* Force worker_evict_one onto the superseded pick by protecting all others. */
        bool only[PULSAR_SESSION_POOL_CAP];
        for (int i = 0; i < PULSAR_SESSION_POOL_CAP; i++) only[i] = (i != sup);
        server_log(PULSAR_LOG_KVCACHE,
                   "pulsar-server: warm-fork make-room: evicting LRU-superseded bank %u "
                   "(trunk bank %u preserved)", s->slots[sup].bank, trunk->bank);
        return s->worker_evict_one(only);
    }
    /* No superseded victim: plain LRU among unprotected idle, trunk still safe. */
    return s->worker_evict_one(protect);
}



/* Bind the head job to a slot if routing allows it. Strict FIFO: when the
 * head must wait (its owner slot is busy, or no fitting slot is free), later
 * jobs wait behind it — simple and starvation-free. Returns true when the
 * head was consumed: bound to a slot, or failed explicitly (a continuation
 * that cannot fit its owner slot — see choose_slot_for_job). When the head
 * cannot be placed cleanly (nothing fits, or only a warm slot it would
 * clobber), it is not waiting on a busy owner, and the provisioning refusal
 * is one eviction can relieve (full pool / full ledger — never the
 * MemAvailable floor, see the increment-4 block above), idle slots are
 * evicted LRU-first until the head binds without clobbering or eviction
 * stops helping — then the clobber fallback binds it exactly like the
 * increment-3 scheduler did. */
bool server::worker_try_bind() {
    auto *s = this;
    pthread_mutex_lock(&s->mu);
    job *j = s->head; /* peek: only the worker pops */
    pthread_mutex_unlock(&s->mu);
    if (!j) return false;

    int reject_ctx = 0;
    bool waiting_owner = false;
    bool clobbers = false;
    provision_refusal refusal = PROVISION_OK;
    session_slot *sl = s->choose_slot_for_job(j, &reject_ctx, &waiting_owner,
                                           &clobbers, &refusal);
    if ((!sl || clobbers) && !waiting_owner && reject_ctx == 0 &&
        (refusal == PROVISION_REFUSED_POOL_FULL ||
         refusal == PROVISION_REFUSED_ADMISSION))
    {
        bool protect[PULSAR_SESSION_POOL_CAP];
        s->worker_protect_queued_owner_slots(protect);
        if (s->worker_eviction_could_help(j, protect)) {
            while ((!sl || clobbers) &&
                   (refusal == PROVISION_REFUSED_POOL_FULL ||
                    refusal == PROVISION_REFUSED_ADMISSION))
            {
                /* Refresh owner protection every iteration (2026-07-15
                 * review): each pass through choose_slot_for_job can stall
                 * for seconds inside pulsar_session_create, and a live-tool
                 * continuation enqueued during that stall would be invisible
                 * to a one-shot snapshot — its owner could then be evicted
                 * into an avoidable 409. Holes are re-derived from
                 * provisioned == false (the refresh clears worker_evict_one's
                 * hole marks). */
                s->worker_protect_queued_owner_slots(protect);
                for (int i = 0; i < s->n_slots; i++) {
                    if (!s->slots[i].provisioned) protect[i] = true;
                }
                if (!s->worker_evict_one(protect)) break;
                sl = s->choose_slot_for_job(j, &reject_ctx, &waiting_owner,
                                         &clobbers, &refusal);
            }
        }
    }
    if (!sl && reject_ctx > 0) {
        /* The job can never run: pop it and send the front door's
         * context_length_exceeded client error (against the owner slot's
         * context), then wake the client thread exactly like
         * worker_finish_slot does for a completed job. */
        pthread_mutex_lock(&s->mu);
        s->head = j->next;
        if (!s->head) s->tail = NULL;
        if (s->n_queued > 0) s->n_queued--;
        pthread_mutex_unlock(&s->mu);
        j->next = NULL;
        http_error_context_length_exceeded(j->fd, s->enable_cors, &j->req,
                                           j->req.prompt.len, reject_ctx);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
        return true;
    }
    if (!sl) return false;

    pthread_mutex_lock(&s->mu);
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    if (s->n_queued > 0) s->n_queued--;
    s->n_generating++;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;

    s->generate_job_begin(sl, j);
    return true;
}



/* Publish the /metrics snapshots — per-slot KV position/context and the
 * engine spec-decode counters — into plain server fields under mu. Client
 * threads must never call into the engine (CUDA-state audit,
 * pulsar_server_internal.h), so the worker exports these at startup (cli_main,
 * before the worker thread runs), after binds, and once per quantum;
 * send_metrics reads only the snapshots. Host-int copies, no GPU work. */
void server::publish_metrics_snapshot() {
    auto *s = this;
    pulsar_spec_metrics m;
    pulsar_engine_spec_metrics(s->engine, &m);
    pthread_mutex_lock(&s->mu);
    for (int i = 0; i < s->n_slots; i++) {
        /* Bank-aware: a non-live bank's pos is its saved carry, never the pool's
         * live cursor (server_slot_frontier_pos). Pure host read, no CUDA. */
        s->m_slot_pos[i] = s->slot_frontier_pos(&s->slots[i]);
        s->m_slot_ctx[i] = s->slots[i].ctx_size;
    }
    s->m_spec = m;
    pthread_mutex_unlock(&s->mu);
}



/* Detach a finished job from its slot and wake its client thread. */
void server::worker_finish_slot(session_slot *sl) {
    auto *s = this;
    job *j = sl->active_job;
    s->generate_job_end(sl);
    pthread_mutex_lock(&s->mu);
    if (s->n_generating > 0) s->n_generating--;
    pthread_mutex_unlock(&s->mu);
    pthread_mutex_lock(&j->mu);
    j->done = true;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
}



/* The single GPU worker (increment 3): a round-robin scheduler over the slot
 * pool. Each pass binds queued jobs to free slots (FIFO), then advances ONE
 * runnable slot by one quantum — a prefill chunk, or up to
 * PULSAR_SERVER_DECODE_QUANTUM_TOKENS decode tokens — and flushes that slot's
 * deferred client bytes. With a single active job this degenerates to the
 * increment-2 loop (quantum after quantum on one slot, with only a queue
 * peek — one mutex op, no GPU work — between quanta), so single-client
 * output is byte-identical. All pulsar_session_* and CUDA work stays on this
 * thread. On shutdown the scheduler keeps stepping bound jobs (the decode
 * loop observes g_stop_requested) and drains the queue, exactly like the
 * increment-2 worker.
 *
 * Quantum overshoot: binding may lazily provision a slot, and
 * provision_slot's pulsar_session_create is a multi-GiB allocation that can
 * take SECONDS — every bound slot stalls for its duration. Binding may also
 * EVICT idle slots first (increment 4): snapshot-to-disk (full device sync +
 * multi-GiB D2H + disk write) plus session teardown, then the provisioning
 * on top. These are the largest quantum overshoots in the system (larger
 * than the DSpark ≤17-token fused burst) and are deliberate single-thread
 * design: the CUDA-state audit (pulsar_server_internal.h) rules out a second
 * GPU thread, and both happen only at a scheduling boundary. */
/* A slot is eligible for the batched (plain multiseq) decode lane when it is in
 * steady-state decode and is NOT a tool-call request. Tool requests keep the
 * classic lane: their structured payload uses temperature-0 gating and the
 * think/tool recovery paths, which are plain-decode-by-contract absent from the
 * batched path. Prefill/init/finish slots are serviced per-slot as usual. */
static bool slot_is_batchable_decode(const session_slot *sl) {
    const gen_state *g = sl->gen;
    return sl->active_job && g && g->phase == GEN_DECODE &&
           !(g->j->req.kind == REQ_CHAT && g->j->req.has_tools);
}

/* Tier-2 §5 batched decode quantum: ONE shared multiseq weight sweep drives up
 * to PULSAR_SERVER_DECODE_QUANTUM_TOKENS steps across every supplied decode slot.
 * Each slot samples its OWN logits row with its OWN sampler/RNG and streams
 * through gen_emit_token (the shared emit path), so per-request
 * output/stop/streaming is byte-identical in shape to the classic lane; only the
 * forward numerics differ (batched vs single-token path). NOTE: this is NOT
 * co-scheduling-neutral for greedy (temp-0) output. The M>=2 co-batched forward's
 * logits differ ~1 ULP from the M=1 single-sequence path (different GEMM tiling /
 * reduction order); at greedy temp-0 that can flip a near-tie argmax, so a
 * request's greedy continuation depends on what else is co-scheduled. This is
 * distribution-preserving and INVISIBLE under sampling (temp>0) — same class as
 * batched inference elsewhere (vLLM et al.) — but it is a real greedy-determinism
 * property, characterized 2026-07-22. The multiseq gate proves N=2-vs-N=3
 * neutrality (bank output invariant to OTHER banks), which is a WEAKER invariant
 * than batched==solo and does NOT cover this ULP. Plain decode
 * only. Slots that stop are set to GEN_FINISH and dropped; the worker finishes
 * them via the per-slot path, which reconciles their host checkpoint against the
 * tokens committed here. Leaves the pool multiseq-poisoned with live_bank = -1
 * so the next server_bank_switch does a real bank restore. */
/* ==== Tier-2 task #55 increment 2b: proactive-eviction guard =================
 *
 * At each decode-quantum boundary, if the live banks growing by one quantum would
 * push total resident KV past the budget, SPILL the LRU-idle smallest-frontier
 * bank: its comp/index KV -> local fast disk (raw, bit-identical), then cudaFree
 * its physical (the only primitive that returns physical on GB10; 2a/2b gates).
 * The conversation stays bound to the slot and is restored (alloc + reload) on its
 * next decode via server_bank_switch. Decisions use the DETERMINISTIC touched
 * accounting + MemAvailable floor backstop, never the coarse cudaMemGetInfo gauge.
 * Worker thread only. */

bool server::bank_restore_spilled(int bank) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    char path[600];
    snprintf(path, sizeof path, "%s/spill-bank-%d.kv", s->spill_dir, bank);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: restore open %s failed: %s",
                   path, strerror(errno));
        return false;
    }
    char err[128];
    const double t0 = server_now_sec();
    const int rc = pulsar_session_bank_kv_load(pool, (uint32_t)bank, fp, err, sizeof err);
    fclose(fp);
    if (rc != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: kv_load bank %d failed: %s", bank, err);
        return false;
    }
    s->slots[bank].spilled = false;
    remove(path);
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: guard RESTORED bank %d from disk (%.1f ms reload stall)",
               bank, (server_now_sec() - t0) * 1e3);
    return true;
}

/* Spill one idle bank: install it, snapshot its KV to disk, save its host carry,
 * repoint AWAY (free_physical refuses the cur bank), then cudaFree its physical. */
bool server::spill_bank(session_slot *victim) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    const uint32_t vb = victim->bank;
    (void)s->bank_switch((int)vb);         /* victim never spilled (pick excludes) → true */
    char path[600];
    snprintf(path, sizeof path, "%s/spill-bank-%u.kv", s->spill_dir, vb);
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: spill open %s failed: %s",
                   path, strerror(errno));
        return false;
    }
    char err[128];
    const double t0 = server_now_sec();
    const int rc = pulsar_session_bank_kv_save(pool, vb, fp, err, sizeof err);
    const int fc = fclose(fp);
    if (rc != 0 || fc != 0) {
        server_log(PULSAR_LOG_WARNING, "pulsar-server: guard: kv_save bank %u failed: %s",
                   vb, rc ? err : "close");
        remove(path);
        return false;
    }
    pulsar_session_bank_state_save(pool, vb);         /* preserve host carry for restore */
    if (!pulsar_session_bank_state_restore(pool, 0)) { remove(path); return false; }
    s->live_bank = 0;
    /* Finding 4: free_physical returns false ONLY on a precondition refusal (nothing
     * freed, bank still live) — abort the spill and drop the unused disk snapshot.
     * A true return means the bank IS evicted (slabs freed), so mark it spilled
     * unconditionally — no half-evicted state (spilled=false over freed slabs). */
    if (!pulsar_session_bank_free_physical(pool, vb)) {
        server_log(PULSAR_LOG_WARNING,
                   "pulsar-server: guard: free_physical bank %u refused (still cur?) — spill aborted", vb);
        remove(path);
        return false;
    }
    victim->spilled = true;
    victim->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
    s->guard_evictions++;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    server_log(PULSAR_LOG_DEFAULT,
               "pulsar-server: guard EVICTED bank %u -> disk (%.1f ms save, physical freed); "
               "touched %.2f GiB / budget %.2f GiB, evictions %llu",
               vb, (server_now_sec() - t0) * 1e3,
               (double)pulsar_session_touched_kv_bytes(pool) / gib,
               (double)s->guard_touched_budget / gib,
               (unsigned long long)s->guard_evictions);
    return true;
}

/* LRU-idle smallest-frontier victim: NOT bank 0 (pinned), NOT in the live decode
 * set, no active job, not already spilled. -1 if none. */
int server::guard_pick_victim(session_slot **dec, int n) {
    auto *s = this;
    pulsar_session *pool = s->sess;
    int best = -1;
    uint64_t best_us = UINT64_MAX, best_touched = UINT64_MAX;
    for (int i = 1; i < s->n_slots; i++) {         /* bank 0 pinned */
        session_slot *sl = &s->slots[i];
        if (!sl->provisioned || sl->spilled || sl->active_job) continue;
        /* plan-33: never free a bank that is a live fork SOURCE mid-clone. */
        if (pulsar_session_bank_fork_pinned(pool, sl->bank)) continue;
        bool live = false;
        for (int k = 0; k < n; k++) if (dec[k] == sl) { live = true; break; }
        if (live) continue;
        const uint64_t t = pulsar_session_bank_touched_kv_bytes(pool, sl->bank);
        if (sl->last_serviced_us < best_us ||
            (sl->last_serviced_us == best_us && t < best_touched)) {
            best = i; best_us = sl->last_serviced_us; best_touched = t;
        }
    }
    return best;
}

void server::guard_maybe_evict(session_slot **dec, int n) {
    auto *s = this;
    if (!s->guard_enabled || s->pool_banks <= 0 || n <= 0) return;
    pulsar_session *pool = s->sess;
    const uint64_t dpb = pulsar_session_quantum_growth_bytes_per_bank(
            pool, (uint32_t)PULSAR_SERVER_DECODE_QUANTUM_TOKENS);
    const uint64_t delta = (uint64_t)n * dpb;      /* all n live banks grow */
    const uint64_t bound = s->guard_touched_budget;
    /* Finding 2: free_physical zeroes a spilled bank's frontier so touched drops
     * after each spill — the loop then re-evaluates and evicts exactly the minimum
     * (usually ONE) per breach, NOT the whole idle set (the cascade bug). The
     * per-quantum count log lets the smoke assert no cascade. */
    int spilled_this_quantum = 0;
    for (;;) {
        const uint64_t projected = pulsar_session_touched_kv_bytes(pool) + delta;
        if (projected <= bound) break;             /* fits */
        const int vi = s->guard_pick_victim(dec, n);
        if (vi < 0) {
            /* No idle victim: back-pressure — proceed and let the live floor guard.
             * (Evicting a LIVE growing bank would thrash; the MemAvailable watchdog
             * is the hard backstop.) */
            static uint64_t last_warn_us;
            const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
            if (now_us - last_warn_us > 5000000ull) {
                last_warn_us = now_us;
                const double gib = 1024.0 * 1024.0 * 1024.0;
                server_log(PULSAR_LOG_WARNING,
                    "pulsar-server: guard: projected touched %.2f GiB > budget %.2f GiB but NO "
                    "idle victim — back-pressure (MemAvailable floor is the backstop)",
                    (double)projected / gib, (double)bound / gib);
            }
            break;
        }
        if (!s->spill_bank(&s->slots[vi])) break;   /* spill failed — stop */
        spilled_this_quantum++;
    }
    if (spilled_this_quantum > 0) {
        server_log(PULSAR_LOG_DEFAULT,
                   "pulsar-server: guard: spilled %d bank(s) this quantum (touched now %.2f GiB / %.2f GiB)",
                   spilled_this_quantum,
                   (double)pulsar_session_touched_kv_bytes(pool) / (1024.0*1024.0*1024.0),
                   (double)bound / (1024.0*1024.0*1024.0));
    }
}

void server::worker_batched_decode_quantum(session_slot **dec, int n) {
    auto *s = this;
    if (n <= 0) return;
    pulsar_session *pool = s->sess;
    const int vocab = pulsar_engine_logits_width(s->engine);

    /* Tier-2 2b: proactive-eviction guard — BEFORE the weight sweep grows the live
     * banks, spill LRU-idle banks if this quantum's projected growth would breach
     * the resident-KV budget. A dec bank that was spilled while idle is restored
     * transparently by server_bank_switch in the ENTRY loop below. */
    s->guard_maybe_evict(dec, n);

    /* ENTRY: a slot not yet in the batch samples its first feed token from its
     * bank's current classic logits while that bank is briefly live, then
     * CAPTURES its per-bank frontier counters (ms_n_comp[bank]) that the
     * multiseq driver's position-true check reads. The explicit save is
     * essential: server_bank_switch only captures the OUTGOING bank, so a bank
     * entering the batch while it is already the live bank (e.g. it just
     * prefilled and was never switched away) would otherwise carry a STALE
     * ms_n_comp and the driver would reject the step ("frontier not
     * position-true"). Redundant when the bank was captured on a prior
     * switch-away — but harmless (same value). */
    for (int i = 0; i < n; i++) {
        session_slot *sl = dec[i];
        gen_state *g = sl->gen;
        if (g->batch_feed_valid) continue;
        /* Finding 1: a failed spill restore must NOT sample this slot's feed token
         * on another bank's KV — fail the slot cleanly and drop it from the batch. */
        if (!s->bank_switch(sl->bank)) {
            snprintf(g->err, sizeof g->err,
                     "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
            g->finish = "error";
            g->batch_feed_valid = false;
            g->phase = GEN_FINISH;
            continue;
        }
        float temp, top_p, min_p; int top_k;
        gen_resolve_sampling(&g->j->req, &temp, &top_k, &top_p, &min_p);
        g->batch_feed_token =
            pulsar_session_sample(pool, temp, top_k, top_p, min_p, &g->rng);
        g->batch_feed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
        g->batch_feed_valid = true;
        g->batch_active = true;
    }

    float *logits = (float *)server_xmalloc((size_t)n * (size_t)vocab * sizeof(float));
    pulsar_multiseq_req reqs[PULSAR_SESSION_POOL_CAP];
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < PULSAR_SERVER_DECODE_QUANTUM_TOKENS; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;            /* already stopped */
            if (g_stop_requested || g->completion >= g->max_tokens) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            if (g->batch_feed_pos >= pulsar_session_ctx(pool)) {
                g->batch_feed_valid = false; g->finish = "length";
                g->phase = GEN_FINISH; continue;
            }
            reqs[m].bank = sl->bank;
            reqs[m].pos = g->batch_feed_pos;
            reqs[m].token = g->batch_feed_token;
            live_idx[m] = i;
            m++;
        }
        if (m == 0) break;

        char err[96];
        /* plan-34 inc 1: route the decode-only lane through the mixed entry
         * (n_rows == n_dec, still exactly 1 row per bank — no prefill rows, no
         * mixing yet). Byte-identical to pulsar_session_decode_multiseq; the only
         * change is the heap descriptor scratch. */
        const int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m,
                                                 logits, (uint32_t)m * vocab,
                                                 NULL, 0u, err, sizeof(err));
        s->live_bank = -1;   /* pool is multiseq-poisoned: no clean live bank */
        if (rc != 0) {
            for (int q = 0; q < m; q++) {
                gen_state *g = dec[live_idx[q]]->gen;
                g->finish = "error";
                snprintf(g->err, sizeof(g->err), "batched decode failed: %s", err);
                g->batch_feed_valid = false;
                g->phase = GEN_FINISH;
            }
            break;
        }

        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            const int committed = g->batch_feed_token; /* committed at batch_feed_pos */
            g->batch_feed_pos++;
            sl->committed_pos = g->batch_feed_pos;
            pulsar_tokens_push(&g->batch_pending, committed);
            if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
            /* Route THIS slot's stream writes through its own deferral queue
             * (the worker-thread-local writer is shared across slots). */
            slot_writer_install(&g->writer);
            if (s->gen_emit_token(sl, committed)) {
                g->batch_feed_valid = false;
                g->phase = GEN_FINISH;
                continue;
            }
            const float *row = logits + (size_t)q * (size_t)vocab;
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling(&g->j->req, &temp, &top_k, &top_p, &min_p);
            g->batch_feed_token =
                pulsar_sample_logits(row, vocab, temp, top_k, top_p, min_p, &g->rng);
        }
    }
    free(logits);
    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
}

/* plan-34 phase-2 inc 5 — find ONE prefilling slot to FOLD into the fused mixed
 * quantum (P=1). Admissible = main-prefill (not cold), already past its FIRST chunk
 * (bank pos>0, so the driver's pos-0 reject is satisfied — the first chunk stays
 * classic), and with more than one fold-chunk of prompt still left (the FINAL tail
 * <= chunk stays classic; it carries the prefill->decode completion bookkeeping).
 * Its bank is necessarily DISTINCT from every decode bank (different phase). NULL
 * when the flag is off, not in pool mode, or nothing qualifies. */
session_slot *server::worker_find_fuse_prefill() {
    auto *s = this;
    if (!s->mixed_batch_enabled || s->pool_banks <= 0) return NULL;
    /* Deep-concurrent guard (see the enable block in cli_main.cpp): fusing a
     * prefill chunk into a decode quantum whose banks already read a deep
     * aggregate KV working set displaces bandwidth-saturated decode. Refuse
     * to fuse while the active decode set's summed committed depth exceeds
     * the threshold; those prefills take the classic (unfused) path instead. */
    if (s->mixed_deep_guard_rows > 0) {
        long deep = 0;
        int n_dec = 0;
        for (int i = 0; i < s->n_slots; i++) {
            const session_slot *dl = &s->slots[i];
            if (dl->provisioned && dl->active_job && dl->state == SLOT_DECODING) {
                deep += dl->committed_pos;
                n_dec++;
            }
        }
        if (n_dec >= 2 && deep > (long)s->mixed_deep_guard_rows) {
            static long last_logged = -1;
            if (deep != last_logged) {
                server_log(PULSAR_LOG_KVCACHE,
                           "pulsar-server: fused lane paused by deep guard (%d decoders, %ld aggregate rows > %d)",
                           n_dec, deep, s->mixed_deep_guard_rows);
                last_logged = deep;
            }
            return NULL;
        }
    }
    pulsar_session *pool = s->sess;
    for (int i = 0; i < s->n_slots; i++) {
        session_slot *sl = &s->slots[i];
        gen_state *g = sl->gen;
        if (!sl->active_job || !g || g->phase != GEN_PREFILL_MAIN || !g->prompt_for_sync ||
            g->no_fuse)
            continue;
        const int P = pulsar_session_bank_pos(pool, sl->bank);
        const int len = g->prompt_for_sync->len;
        if (P <= 0 || P >= len) continue;                       /* first chunk / done: classic */
        return sl;                                              /* P=1: first qualifier */
    }
    return NULL;
}

/* plan-34 phase-2 inc 5 — FUSED mixed-batch quantum. One decode quantum whose EVERY
 * step folds a small (s->mixed_chunk_tokens) prefill run for `pf` into the SAME
 * pulsar_session_decode_mixed sweep as the decode banks (true continuous batching,
 * P=1). Decode banks advance exactly as worker_batched_decode_quantum — the inc-4
 * neutrality gate proves a co-scheduled prefill does not perturb them — so their
 * per-request output is unchanged in shape. The prefill advances up to
 * QUANTUM*chunk tokens, SPREAD uniformly across the steps so no single decode step
 * eats a whole chunk (the p99 lever vs the time-slice's per-interval decode stall).
 * pf's FIRST chunk (pos 0) and FINAL tail (<= chunk) stay CLASSIC: the tail's
 * pulsar_session_sync carries the prefill->decode completion (kv-cache store,
 * gen_stream_begin), so this never reimplements that handoff. Reconciliation of
 * pf's bank is the exact recipe the decode lane uses (bank_state_restore +
 * note_committed_tokens). */
void server::worker_mixed_batch_quantum(session_slot **dec, int n, session_slot *pf) {
    auto *s = this;
    if (n <= 0 || !pf || !pf->gen || !pf->gen->prompt_for_sync) return;
    pulsar_session *pool = s->sess;
    const int vocab = pulsar_engine_logits_width(s->engine);
    gen_state *pg = pf->gen;
    const pulsar_tokens *pp = pg->prompt_for_sync;

    s->guard_maybe_evict(dec, n);

    /* Decode-bank ENTRY — identical to worker_batched_decode_quantum. */
    for (int i = 0; i < n; i++) {
        session_slot *sl = dec[i];
        gen_state *g = sl->gen;
        if (g->batch_feed_valid) continue;
        if (!s->bank_switch(sl->bank)) {
            snprintf(g->err, sizeof g->err,
                     "bank %u state restore failed (evicted KV unrecoverable)", (unsigned)sl->bank);
            g->finish = "error"; g->batch_feed_valid = false; g->phase = GEN_FINISH;
            continue;
        }
        float temp, top_p, min_p; int top_k;
        gen_resolve_sampling(&g->j->req, &temp, &top_k, &top_p, &min_p);
        g->batch_feed_token = pulsar_session_sample(pool, temp, top_k, top_p, min_p, &g->rng);
        g->batch_feed_pos = pulsar_session_pos(pool);
        pulsar_session_bank_state_save(pool, (uint32_t)sl->bank);
        g->batch_feed_valid = true; g->batch_active = true;
    }

    /* Prefill-bank ENTRY: make pf live at its committed frontier P0 and capture its
     * per-bank counters so the mixed driver reads pf's TRUE starting frontier. */
    if (!s->bank_switch(pf->bank)) {
        snprintf(pg->err, sizeof pg->err, "prefill bank %u restore failed", (unsigned)pf->bank);
        pg->finish = "error"; pg->phase = GEN_FINISH; return;
    }
    const int P0 = pulsar_session_pos(pool);
    pulsar_session_bank_state_save(pool, (uint32_t)pf->bank);
    {   /* one-shot observability: confirm the fused lane actually engaged. */
        static int logged = 0;
        if (logged < 3) { logged++;
            server_log(PULSAR_LOG_DEFAULT,
                       "pulsar-server: FUSED mixed quantum engaged (prefill bank %u at pos %d/%d, "
                       "%d decode banks, chunk %d/step)",
                       (unsigned)pf->bank, P0, pp->len, n, s->mixed_chunk_tokens);
        }
    }

    int kstep = s->mixed_chunk_tokens;
    const int cap = pulsar_session_prefill_cap(pool);   /* mixed-step row ceiling */
    if (kstep > cap - PULSAR_SESSION_POOL_CAP) kstep = cap - PULSAR_SESSION_POOL_CAP;
    if (kstep < 1) kstep = 1;
    const int len = pp->len;
    int pf_done = 0;
    bool reached_end = false;                     /* prefill reached len this quantum */
    bool pf_giveup = false;                        /* prefill rejected -> stop folding it */

    const size_t reqcap = (size_t)PULSAR_SESSION_POOL_CAP + (size_t)kstep;
    pulsar_multiseq_req *reqs = (pulsar_multiseq_req *)server_xmalloc(reqcap * sizeof(*reqs));
    float *logits = (float *)server_xmalloc((size_t)(PULSAR_SESSION_POOL_CAP + 1) * (size_t)vocab * sizeof(float));
    /* last-position (len-1) logits captured from the final fused prefill run — the
     * decode seed for the prefill->decode handoff (byte-identical to classic per
     * the inc-4 gate: the fused run's last-of-run logits match classic-resume). */
    float *pf_last_logits = (float *)server_xmalloc((size_t)vocab * sizeof(float));
    int live_idx[PULSAR_SESSION_POOL_CAP];

    for (int step = 0; step < PULSAR_SERVER_DECODE_QUANTUM_TOKENS; step++) {
        int m = 0;
        for (int i = 0; i < n && m < PULSAR_SESSION_POOL_CAP; i++) {
            session_slot *sl = dec[i];
            gen_state *g = sl->gen;
            if (!g->batch_feed_valid) continue;
            if (g_stop_requested || g->completion >= g->max_tokens) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            if (g->batch_feed_pos >= pulsar_session_ctx(pool)) {
                g->batch_feed_valid = false; g->finish = "length"; g->phase = GEN_FINISH; continue;
            }
            reqs[m].bank = sl->bank; reqs[m].pos = g->batch_feed_pos;
            reqs[m].token = g->batch_feed_token; live_idx[m] = i; m++;
        }
        /* Fold this step's ascending prefill sub-chunk from P0+pf_done, all the way
         * to len (the FINAL sub-chunk carries the last-position logits used for the
         * prefill->decode handoff — no non-aligned classic tail resume, so the
         * prefill output stays byte-identical to a cold prefill). */
        const int pos_now = P0 + pf_done;
        int kthis = 0;
        if (!pf_giveup && pos_now < len) {
            kthis = kstep;
            if (pos_now + kthis > len) kthis = len - pos_now;
        }
        for (int j = 0; j < kthis; j++) {
            reqs[m + j].bank = pf->bank;
            reqs[m + j].pos = pos_now + j;
            reqs[m + j].token = pp->v[pos_now + j];
        }
        const int nrows = m + kthis;
        if (nrows == 0) break;

        char err[96];
        uint32_t n_runs = 0;
        /* LEVER 1: on an INTERMEDIATE prefill sub-chunk (prefill does not reach len
         * this step, and there are decode banks to head), emit ONLY the decode banks'
         * logits (max_head_runs = m) — the prefill run's intermediate logits are
         * unused, and the head takes the single-block identity path (no two-block
         * resync, no wasted prefill head). On the FINAL sub-chunk (pos_now+kthis==len)
         * OR a pure-decode step, pass 0 = all runs (the prefill head IS consumed). */
        const uint32_t head_cap =
            (kthis > 0 && m > 0 && pos_now + kthis < len) ? (uint32_t)m : 0u;
        int rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)nrows, logits,
                (int)((size_t)(m + (kthis > 0 ? 1 : 0)) * (size_t)vocab), &n_runs, head_cap, err, sizeof err);
        if (rc == 1 && kthis > 0) {
            /* RECOVERABLE reject (nothing committed) caused by the PREFILL run — e.g.
             * its bank's frontier is not position-true (a cache-warm resume). Do NOT
             * harm the co-scheduled decode banks: stop folding this prefill (route it
             * classic via no_fuse) and retry this step DECODE-ONLY. */
            pf_giveup = true; pg->no_fuse = true;
            if (m > 0) {
                rc = pulsar_session_decode_mixed(pool, reqs, (uint32_t)m, logits,
                        (int)((size_t)m * (size_t)vocab), &n_runs, 0u, err, sizeof err);
            } else { s->live_bank = -1; break; }   /* only prefill this step: just stop */
            kthis = 0;                              /* prefill did not advance */
        }
        s->live_bank = -1;
        if (rc != 0) {
            for (int q = 0; q < m; q++) {
                gen_state *g = dec[live_idx[q]]->gen;
                g->finish = "error";
                snprintf(g->err, sizeof g->err, "fused mixed decode failed: %s", err);
                g->batch_feed_valid = false; g->phase = GEN_FINISH;
            }
            break;   /* real decode failure -> stop */
        }
        pf_done += kthis;
        if (kthis > 0 && pos_now + kthis == len) {
            /* final prefill sub-chunk: capture the len-1 logits (prefill run row =
             * index m, last-of-run) as the decode seed for the handoff. */
            memcpy(pf_last_logits, logits + (size_t)m * (size_t)vocab, (size_t)vocab * sizeof(float));
            reached_end = true;
        }

        for (int q = 0; q < m; q++) {
            session_slot *sl = dec[live_idx[q]];
            gen_state *g = sl->gen;
            const int committed = g->batch_feed_token;
            g->batch_feed_pos++; sl->committed_pos = g->batch_feed_pos;
            pulsar_tokens_push(&g->batch_pending, committed);
            if (g->first_token_t == 0.0) g->first_token_t = server_now_sec();
            slot_writer_install(&g->writer);
            if (s->gen_emit_token(sl, committed)) {
                g->batch_feed_valid = false; g->phase = GEN_FINISH; continue;
            }
            const float *row = logits + (size_t)q * (size_t)vocab;
            float temp, top_p, min_p; int top_k;
            gen_resolve_sampling(&g->j->req, &temp, &top_k, &top_p, &min_p);
            g->batch_feed_token = pulsar_sample_logits(row, vocab, temp, top_k, top_p, min_p, &g->rng);
        }
    }
    free(reqs); free(logits);

    /* Reconcile the prefill bank (same recipe as a decode bank leaving the lane):
     * install its driver-maintained frontier (P0+pf_done) and advance the host
     * checkpoint by exactly the committed prefill tokens, so its next chunk resumes
     * correctly and any store sees the true frontier. */
    if (pf_done > 0 && pg->phase == GEN_PREFILL_MAIN) {
        if (pulsar_session_bank_state_restore(pool, pf->bank)) {
            pulsar_session_note_committed_tokens(pool, &pp->v[P0], pf_done);
            pf->committed_pos = P0 + pf_done;
            s->live_bank = (int)pf->bank;
            if (reached_end) {
                /* Prefill complete: seed the decode with the fused last-position
                 * logits (byte-identical to classic) and run the SAME prefill->
                 * decode handoff the classic path uses (kv-cache store, SSE start,
                 * GEN_DECODE_INIT). No non-aligned classic tail resume => the
                 * request's decode is byte-identical to a cold classic prefill. */
                pulsar_session_set_logits(pool, pf_last_logits, vocab);
                slot_writer_install(&pg->writer);
                s->gen_stream_begin(pf);
                slot_writer_flush(&pg->writer);
            }
        } else {
            snprintf(pg->err, sizeof pg->err, "prefill bank %u reconcile failed", (unsigned)pf->bank);
            pg->finish = "error"; pg->phase = GEN_FINISH;
        }
    }
    free(pf_last_logits);

    const uint64_t now_us = (uint64_t)(server_now_sec() * 1e6);
    for (int i = 0; i < n; i++) {
        dec[i]->last_serviced_us = now_us;
        if (dec[i]->gen) slot_writer_flush(&dec[i]->gen->writer);
    }
    pf->last_serviced_us = now_us;
}

void *worker_main(void *arg) {
    server *s = (server *)arg;
    int rr = 0; /* round-robin cursor: first slot index to consider next */
    for (;;) {
        bool bound = false;
        while (s->worker_try_bind()) bound = true;
        if (bound) s->publish_metrics_snapshot();

        int n_active = 0;
        for (int i = 0; i < s->n_slots; i++) {
            if (s->slots[i].active_job) n_active++;
        }
        if (n_active == 0) {
            /* With every slot free, choose_slot_for_job never returns NULL
             * (slot 0 always fits), so an unbound head cannot reach this
             * wait: sleeping on the condvar until new work or shutdown is
             * safe. */
            pthread_mutex_lock(&s->mu);
            while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
            const bool quit = !s->head && s->stopping;
            pthread_mutex_unlock(&s->mu);
            if (quit) break;
            continue;
        }

        /* Tier-2 three-way lane (§5). Gather steady-state batchable decode slots.
         * n_batched > 0 means a batch is already in flight — those slots stay
         * batched until they finish (no mid-conversation batched->classic switch,
         * which would need stale-logits reconciliation). The batched lane engages
         * when the batchable decode count exceeds spec_max_live, OR whenever a
         * batch is already active. Otherwise the classic per-slot round-robin
         * runs unchanged: n_active==1 -> spec (N=1 byte-identical), n<=spec_max_live
         * -> spec time-slice. Env PULSAR_SERVER_SPEC_MAX_LIVE tunes the crossover. */
        session_slot *dec[PULSAR_SESSION_POOL_CAP];
        int n_dec = 0, n_batched = 0;
        if (s->pool_banks > 0) {
            for (int i = 0; i < s->n_slots; i++) {
                if (slot_is_batchable_decode(&s->slots[i])) {
                    dec[n_dec++] = &s->slots[i];
                    if (s->slots[i].gen->batch_active) n_batched++;
                }
            }
        }
        const bool use_batched =
            s->pool_banks > 0 && n_dec >= 1 &&
            (n_dec > s->spec_max_live || n_batched > 0);

        if (use_batched) {
            /* plan-34 inc 5: when the fused lane is armed and a prefilling slot is
             * admissible (P=1), FOLD its next chunk into the decode sweep instead of
             * the separate classic prefill advance below. Flag OFF (or nothing
             * admissible) => pf_fuse==NULL => today's exact decode-quantum +
             * separate-prefill time-slice, byte-identical. */
            session_slot *pf_fuse = s->worker_find_fuse_prefill();
            if (pf_fuse) s->worker_mixed_batch_quantum(dec, n_dec, pf_fuse);
            else         s->worker_batched_decode_quantum(dec, n_dec);
            /* Finish any slot the batched quantum stopped (per-slot path
             * reconciles its checkpoint). Then also advance ONE non-decode
             * active slot (prefill/init/finish) so prompt ingest never starves
             * behind a long batched decode. */
            for (int i = 0; i < n_dec; i++) {
                session_slot *d = dec[i];
                if (d->gen && d->gen->phase != GEN_DECODE) {
                    s->generate_job_step(d);
                    if (d->gen) slot_writer_flush(&d->gen->writer);
                    d->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
                    if (!d->gen || d->gen->phase == GEN_DONE)
                        s->worker_finish_slot(d);
                }
            }
            session_slot *other = NULL;
            for (int k = 0; k < s->n_slots; k++) {
                session_slot *c = &s->slots[(rr + k) % s->n_slots];
                /* inc 5: skip the slot already advanced in-band by the fused
                 * quantum (pf_fuse) so it does not also run a classic chunk. */
                if (c->active_job && c != pf_fuse && !slot_is_batchable_decode(c) &&
                    !(c->gen && c->gen->batch_active)) {
                    other = c;
                    rr = (int)(c - s->slots) + 1;
                    break;
                }
            }
            if (other && other->gen && other->gen->phase != GEN_DONE) {
                s->generate_job_step(other);
                if (other->gen) slot_writer_flush(&other->gen->writer);
                other->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
                if (!other->gen || other->gen->phase == GEN_DONE)
                    s->worker_finish_slot(other);
            }
            s->publish_metrics_snapshot();
            continue;
        }

        session_slot *sl = NULL;
        for (int k = 0; k < s->n_slots; k++) {
            session_slot *c = &s->slots[(rr + k) % s->n_slots];
            if (c->active_job) {
                sl = c;
                rr = (int)(c - s->slots) + 1;
                break;
            }
        }
        if (!sl) continue; /* unreachable: n_active > 0 */

        if (sl->gen && sl->gen->phase != GEN_DONE) {
            s->generate_job_step(sl);
            if (sl->gen) slot_writer_flush(&sl->gen->writer);
            sl->last_serviced_us = (uint64_t)(server_now_sec() * 1e6);
        }
        if (!sl->gen || sl->gen->phase == GEN_DONE) {
            s->worker_finish_slot(sl);
        }
        s->publish_metrics_snapshot(); /* /metrics: once per quantum */
    }
    return NULL;
}

