#include "pulsar_server_internal.h"



double server_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}



void server_log(pulsar_log_type type, const char *fmt, ...) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char ts[16];
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tm);

    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);

    fprintf(stderr, "%s ", ts);
    if (n < 0) {
        pulsar_log(stderr, type, "%s", fmt);
    } else {
        char *line = (char *)server_xmalloc((size_t)n + 1);
        vsnprintf(line, (size_t)n + 1, fmt, ap);
        pulsar_log(stderr, type, "%s", line);
        free(line);
    }
    va_end(ap);
    fputc('\n', stderr);
}



bool id_list_contains(const stop_list *ids, const char *id);


void id_list_push_unique(stop_list *ids, const char *id);


int tool_memory_max_entries(const tool_memory *m) {
    return m && m->max_entries > 0 ? m->max_entries : PULSAR_TOOL_MEMORY_DEFAULT_MAX_IDS;
}



static size_t tool_memory_max_bytes(const tool_memory *m) {
    return m && m->max_bytes > 0 ? m->max_bytes : PULSAR_TOOL_MEMORY_MAX_BYTES;
}



static void tool_memory_init_locked(tool_memory *m) {
    if (m->by_id && m->by_block) return;
    m->by_id = raxNew();
    m->by_block = raxNew();
    if (!m->by_id || !m->by_block) die("out of memory");
}



static void tool_memory_link_head(tool_memory *m, tool_memory_entry *e) {
    e->prev = NULL;
    e->next = m->head;
    if (m->head) m->head->prev = e;
    else m->tail = e;
    m->head = e;
}



static void tool_memory_unlink(tool_memory *m, tool_memory_entry *e) {
    if (e->prev) e->prev->next = e->next;
    else m->head = e->next;
    if (e->next) e->next->prev = e->prev;
    else m->tail = e->prev;
    e->prev = e->next = NULL;
}



static void tool_memory_touch(tool_memory *m, tool_memory_entry *e) {
    e->stamp = ++m->clock;
    if (m->head == e) return;
    tool_memory_unlink(m, e);
    tool_memory_link_head(m, e);
}



static void tool_block_unlink_entry(tool_memory_block *b, tool_memory_entry *e) {
    tool_memory_entry **p = &b->entries;
    while (*p) {
        if (*p == e) {
            *p = e->block_next;
            e->block_next = NULL;
            return;
        }
        p = &(*p)->block_next;
    }
}



tool_memory_block *tool_memory_find_block_locked(tool_memory *m,
                                                        const char *dsml,
                                                        size_t len) {
    if (!m->by_block || !dsml || len == 0) return NULL;
    void *v = raxFind(m->by_block, (unsigned char *)dsml, len);
    return v == raxNotFound ? NULL : (tool_memory_block *)v;
}



static tool_memory_block *tool_memory_get_block_locked(tool_memory *m,
                                                       const char *dsml,
                                                       size_t len) {
    tool_memory_block *b = tool_memory_find_block_locked(m, dsml, len);
    if (b) return b;

    b = (tool_memory_block *)server_xmalloc(sizeof(*b));
    memset(b, 0, sizeof(*b));
    b->dsml = xstrndup(dsml, len);
    b->len = len;
    b->bytes = len + 1 + sizeof(*b);
    if (!raxInsert(m->by_block, (unsigned char *)b->dsml, b->len, b, NULL)) {
        free(b->dsml);
        free(b);
        die("out of memory");
    }
    m->bytes += b->bytes;
    return b;
}



static void tool_memory_release_block_locked(tool_memory *m, tool_memory_block *b) {
    if (!b) return;
    if (--b->refs > 0) return;
    if (m->by_block) {
        void *old = NULL;
        (void)raxRemove(m->by_block, (unsigned char *)b->dsml, b->len, &old);
    }
    if (m->bytes >= b->bytes) m->bytes -= b->bytes;
    else m->bytes = 0;
    free(b->dsml);
    free(b);
}



static void tool_memory_remove_entry_locked(tool_memory *m, tool_memory_entry *e) {
    if (!e) return;
    if (m->by_id && e->id) {
        void *old = NULL;
        (void)raxRemove(m->by_id, (unsigned char *)e->id, strlen(e->id), &old);
    }
    tool_memory_unlink(m, e);
    if (e->block) tool_block_unlink_entry(e->block, e);
    if (m->bytes >= e->bytes) m->bytes -= e->bytes;
    else m->bytes = 0;
    if (m->entries > 0) m->entries--;
    free(e->id);
    tool_memory_release_block_locked(m, e->block);
    free(e);
}



static void tool_memory_prune_locked(tool_memory *m) {
    while ((m->entries > tool_memory_max_entries(m) ||
            m->bytes > tool_memory_max_bytes(m)) && m->tail)
    {
        tool_memory_remove_entry_locked(m, m->tail);
    }
}



static tool_memory_entry *tool_memory_find_entry_locked(tool_memory *m,
                                                        const char *id) {
    if (!m->by_id || !id || !id[0]) return NULL;
    void *v = raxFind(m->by_id, (unsigned char *)id, strlen(id));
    return v == raxNotFound ? NULL : (tool_memory_entry *)v;
}



static void tool_memory_put_locked(tool_memory *m, const char *id,
                                   const char *dsml, tool_memory_source source) {
    if (!id || !id[0] || !dsml || !dsml[0]) return;
    tool_memory_init_locked(m);

    size_t dsml_len = strlen(dsml);
    tool_memory_entry *old = tool_memory_find_entry_locked(m, id);
    if (old && old->block && old->block->len == dsml_len &&
        !memcmp(old->block->dsml, dsml, dsml_len))
    {
        if (source == TOOL_MEMORY_RAM) old->source = TOOL_MEMORY_RAM;
        tool_memory_touch(m, old);
        tool_memory_prune_locked(m);
        return;
    }
    if (old) tool_memory_remove_entry_locked(m, old);

    tool_memory_block *b = tool_memory_get_block_locked(m, dsml, dsml_len);
    tool_memory_entry *e = (tool_memory_entry *)server_xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->id = xstrdup(id);
    e->block = b;
    e->bytes = strlen(id) + 1 + sizeof(*e);
    e->stamp = ++m->clock;
    e->source = source;
    e->block_next = b->entries;
    b->entries = e;
    b->refs++;

    if (!raxInsert(m->by_id, (unsigned char *)e->id, strlen(e->id), e, NULL)) {
        tool_block_unlink_entry(b, e);
        free(e->id);
        free(e);
        tool_memory_release_block_locked(m, b);
        die("out of memory");
    }
    tool_memory_link_head(m, e);
    m->entries++;
    m->bytes += e->bytes;
    tool_memory_prune_locked(m);
}



void tool_memory_free(tool_memory *m) {
    while (m->tail) tool_memory_remove_entry_locked(m, m->tail);
    if (m->by_id) raxFree(m->by_id);
    if (m->by_block) raxFree(m->by_block);
    memset(m, 0, sizeof(*m));
}



/* Per-slot live protocol-tool state.
 *
 * This is not an implementation of durable remote conversation storage.  It is
 * only an in-memory binding from protocol tool-call IDs to one slot's current
 * sampled KV frontier.  If it does not match, DS4 falls back to the same prefix
 * and disk-cache machinery used by chat/completions, or returns a clear error
 * for tool-result-only requests that have no replayable prefix.  The states
 * live on the session_slot (they describe that session, and move with it);
 * server.tool_mu guards them all because request parsing reads call-ids on
 * client threads before a job is bound to a slot. */
static void live_tool_state_clear_locked(live_tool_state *st) {
    if (!st) return;
    stop_list_clear(&st->call_ids);
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->valid = false;
    st->live_tokens = 0;
}



void live_tool_state_free(live_tool_state *st) {
    if (!st) return;
    live_tool_state_clear_locked(st);
    free(st->call_ids.v);
    memset(st, 0, sizeof(*st));
}



static void visible_live_clear_locked(visible_live_state *st) {
    if (!st) return;
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->live_tokens = 0;
    st->valid = false;
}



void visible_live_free(visible_live_state *st) {
    if (!st) return;
    visible_live_clear_locked(st);
    memset(st, 0, sizeof(*st));
}



void thinking_live_clear(server *s, session_slot *sl) {
    if (!s || !sl) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&sl->thinking_live);
    pthread_mutex_unlock(&s->tool_mu);
}



void thinking_live_remember(server *s, session_slot *sl, const char *visible_text) {
    if (!s || !sl || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&sl->thinking_live);
    sl->thinking_live.visible_text = xstrdup(visible_text);
    sl->thinking_live.visible_len = strlen(visible_text);
    sl->thinking_live.live_tokens = server_slot_frontier_pos(s, sl);
    sl->thinking_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}



void responses_live_remember(server *s, session_slot *sl, const char *visible_text,
                                    const tool_calls *calls) {
    if (!s || !sl || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&sl->responses_live);
    sl->responses_live.visible_text = xstrdup(visible_text);
    sl->responses_live.visible_len = strlen(visible_text);
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            id_list_push_unique(&sl->responses_live.call_ids, calls->v[i].id);
        }
    }
    sl->responses_live.live_tokens = server_slot_frontier_pos(s, sl);
    sl->responses_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}



void anthropic_live_remember(server *s, session_slot *sl, const tool_calls *calls) {
    if (!s || !sl || !calls || calls->len == 0) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&sl->anthropic_live);
    for (int i = 0; i < calls->len; i++) {
        id_list_push_unique(&sl->anthropic_live.call_ids, calls->v[i].id);
    }
    sl->anthropic_live.live_tokens = server_slot_frontier_pos(s, sl);
    sl->anthropic_live.valid = sl->anthropic_live.call_ids.len > 0;
    pthread_mutex_unlock(&s->tool_mu);
}



void responses_live_clear(server *s, session_slot *sl) {
    if (!s || !sl) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&sl->responses_live);
    pthread_mutex_unlock(&s->tool_mu);
}



void anthropic_live_clear(server *s, session_slot *sl) {
    if (!s || !sl) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&sl->anthropic_live);
    pthread_mutex_unlock(&s->tool_mu);
}



/* Parse-time id lookups run on client threads before the request is bound to
 * a slot, so they scan every provisioned slot's live binding. n_slots is
 * published under mu (its owning lock) — take mu for the snapshot rather than
 * asserting cross-lock visibility. A momentarily stale snapshot would only
 * miss a slot provisioned this instant, whose bindings are still empty. */
static int server_n_slots_snapshot(server *s) {
    pthread_mutex_lock(&s->mu);
    const int n = s->n_slots;
    pthread_mutex_unlock(&s->mu);
    return n;
}



bool responses_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    const int n = server_n_slots_snapshot(s);
    pthread_mutex_lock(&s->tool_mu);
    bool found = false;
    for (int i = 0; i < n && !found; i++) {
        const live_tool_state *st = &s->slots[i].responses_live;
        found = st->valid && id_list_contains(&st->call_ids, id);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}



bool anthropic_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    const int n = server_n_slots_snapshot(s);
    pthread_mutex_lock(&s->tool_mu);
    bool found = false;
    for (int i = 0; i < n && !found; i++) {
        const live_tool_state *st = &s->slots[i].anthropic_live;
        found = st->valid && id_list_contains(&st->call_ids, id);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}



static bool live_state_matches_ids_locked(const live_tool_state *st,
                                          const stop_list *ids,
                                          int live_tokens) {
    bool ok = st->valid &&
              st->live_tokens == live_tokens &&
              st->call_ids.len == ids->len;
    for (int i = 0; ok && i < ids->len; i++) {
        ok = id_list_contains(&st->call_ids, ids->v[i]);
    }
    return ok;
}



bool responses_live_matches_request(server *s, const session_slot *sl,
                                           const stop_list *ids,
                                           int live_tokens) {
    if (!s || !sl || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = live_state_matches_ids_locked(&sl->responses_live, ids, live_tokens);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}



bool anthropic_live_matches_request(server *s, const session_slot *sl,
                                           const stop_list *ids,
                                           int live_tokens) {
    if (!s || !sl || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = live_state_matches_ids_locked(&sl->anthropic_live, ids, live_tokens);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}



/* Scheduler routing (worker thread): find the slot whose live binding holds
 * ALL of the request's continuation ids at that slot's current frontier, so
 * the job can be bound to the session that owns its conversation. */
static session_slot *live_slot_for_ids(server *s, const stop_list *ids,
                                       bool anthropic) {
    if (!s || !ids || ids->len == 0) return NULL;
    session_slot *found = NULL;
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < s->n_slots && !found; i++) {
        session_slot *sl = &s->slots[i];
        const live_tool_state *st = anthropic ? &sl->anthropic_live
                                              : &sl->responses_live;
        /* Bank-aware: match against THIS slot's bank frontier, not the pool's
         * live cursor (Tier-2 shared pool). -1 for an unprovisioned slot. */
        const int pos = sl->provisioned ? server_slot_frontier_pos(s, sl) : -1;
        if (live_state_matches_ids_locked(st, ids, pos)) found = sl;
    }
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}



session_slot *responses_live_slot_for_ids(server *s, const stop_list *ids) {
    return live_slot_for_ids(s, ids, false);
}



session_slot *anthropic_live_slot_for_ids(server *s, const stop_list *ids) {
    return live_slot_for_ids(s, ids, true);
}



bool tool_memory_has_id(server *s, const char *id) {
    if (!s || s->disable_exact_dsml_tool_replay || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = tool_memory_find_entry_locked(&s->tool_mem, id) != NULL;
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}



static const char *tool_memory_lookup_locked(tool_memory *m, const char *id,
                                             tool_memory_source *source,
                                             tool_memory_block **block) {
    tool_memory_entry *e = tool_memory_find_entry_locked(m, id);
    if (!e || !e->block) return NULL;
    tool_memory_touch(m, e);
    if (source) *source = e->source;
    if (block) *block = e->block;
    return e->block->dsml;
}



void tool_memory_remember(server *s, const tool_calls *calls) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !calls || !calls->raw_dsml || !calls->raw_dsml[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < calls->len; i++) {
        tool_memory_put_locked(&s->tool_mem, calls->v[i].id, calls->raw_dsml,
                               TOOL_MEMORY_RAM);
    }
    pthread_mutex_unlock(&s->tool_mu);
}



void tool_memory_put_source(server *s, const char *id, const char *dsml,
                                   tool_memory_source source) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !id || !id[0] || !dsml || !dsml[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    tool_memory_put_locked(&s->tool_mem, id, dsml, source);
    pthread_mutex_unlock(&s->tool_mu);
}



#ifdef PULSAR_SERVER_TEST

void tool_memory_put(server *s, const char *id, const char *dsml) {
    tool_memory_put_source(s, id, dsml, TOOL_MEMORY_RAM);
}


#endif


void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats) {
    if (!msgs) return;
    if (!s || s->disable_exact_dsml_tool_replay) {
        if (stats) {
            for (int i = 0; i < msgs->len; i++) {
                tool_calls *calls = &msgs->v[i].calls;
                if (calls->len == 0 || calls->raw_dsml) continue;
                stats->canonical++;
                stats->missing_ids += calls->len;
            }
        }
        return;
    }
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < msgs->len; i++) {
        tool_calls *calls = &msgs->v[i].calls;
        if (calls->len == 0 || calls->raw_dsml) continue;
        tool_memory_block *matched = NULL;
        tool_memory_source matched_source = TOOL_MEMORY_DISK;
        bool exact = true;
        int missing = 0;
        for (int j = 0; j < calls->len; j++) {
            tool_memory_source source = TOOL_MEMORY_DISK;
            tool_memory_block *block = NULL;
            const char *dsml =
                tool_memory_lookup_locked(&s->tool_mem, calls->v[j].id,
                                          &source, &block);
            if (!dsml) {
                exact = false;
                missing++;
                continue;
            }
            if (!matched) {
                matched = block;
                matched_source = source;
            } else if (matched != block) {
                exact = false;
            }
            if (source == TOOL_MEMORY_RAM) matched_source = TOOL_MEMORY_RAM;
        }
        if (exact && matched) {
            calls->raw_dsml = xstrdup(matched->dsml);
            if (stats) {
                if (matched_source == TOOL_MEMORY_RAM) stats->mem++;
                else stats->disk++;
            }
        } else if (stats) {
            stats->canonical++;
            stats->missing_ids += missing;
        }
    }
    pthread_mutex_unlock(&s->tool_mu);
}



static bool tool_calls_contains_id(const tool_calls *calls, const char *id, int upto) {
    if (!calls || !id || !id[0]) return false;
    if (upto > calls->len) upto = calls->len;
    for (int i = 0; i < upto; i++) {
        if (calls->v[i].id && !strcmp(calls->v[i].id, id)) return true;
    }
    return false;
}



void assign_tool_call_ids(server *s, tool_calls *calls, api_style api) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), api);
            if (!tool_calls_contains_id(calls, id, i) && !tool_memory_has_id(s, id)) break;
        }
        calls->v[i].id = xstrdup(id);
    }
}



void apply_openai_stream_tool_ids(tool_calls *calls,
                                         const openai_stream *st) {
    if (!calls || !st) return;
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}



void apply_anthropic_stream_tool_ids(tool_calls *calls,
                                            const anthropic_stream *st) {
    if (!calls || !st) return;
    /* The SSE stream may have exposed tool ids before final DSML parsing.  The
     * parsed calls must inherit those ids before assign_tool_call_ids() and
     * tool_memory_remember(), otherwise the client returns a tool_result for an
     * id that the continuation fast path does not know. */
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}

