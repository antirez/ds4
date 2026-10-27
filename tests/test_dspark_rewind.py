#!/usr/bin/env python3
"""Exercise actual DSpark rewind helpers against checked, bounded host mocks.

Requires only Python's standard library and a C compiler when run. --prepare-only
extracts and validates the fixture without invoking a compiler or executable.
No model, accelerator, real TP transport or GPU numerical/state proof is involved.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


FUNCTIONS = ('dspark_rewind_begin', 'dspark_rewind_keep',
             'ds4_session_restore_speculative_prefix', 'dspark_rewind_recent',
             'ds4_session_rewind_speculative')


def extract(source, name):
    matches = list(re.finditer(r'^(?:static (?:inline )?)?(?:int|bool|void) ' +
                              re.escape(name) + r'\s*\(', source, re.M))
    if len(matches) != 1:
        raise ValueError(f'require exactly one production helper: {name}')
    start = matches[0].start()
    opening = source.index('{', start)
    token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match.group() == '{':
            depth += 1
        elif match.group() == '}':
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError('unterminated production helper: ' + name)


PRELUDE = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#define DS4_DSPARK_MAX_BLOCK_SIZE 16
#define DS4_N_VOCAB 8
#define CAPACITY 64
typedef struct { int len; int *v; } ds4_tokens;
typedef struct {
    int target_pos, draft_pos;
    uint32_t target[3], draft[3];
} mock_state;
typedef struct { mock_state state; unsigned epoch; } ds4_spec_frontier;
typedef struct { struct { bool active; void *ctx; } tp; } ds4_engine;
typedef struct {
    ds4_engine *engine;
    ds4_tokens checkpoint;
    bool checkpoint_valid, mtp_draft_valid, dspark_stochastic_draft;
    bool dspark_draft_valid, dspark_tp_proposal;
    uint32_t dspark_draft_len;
    ds4_spec_frontier dspark_rewind_frontier;
    float *dspark_rewind_logits, logits[DS4_N_VOCAB];
    int dspark_rewind_start, dspark_rewind_end;
    uint32_t dspark_rewind_prefixes;
    uint64_t tp_session_id;
    int rank, storage[CAPACITY];
    bool leader, capture_valid;
    unsigned epoch;
    mock_state graph, prefixes[DS4_SPEC_PREFIX_SLOTS];
} ds4_session;
static ds4_session sessions[2];
static ds4_engine engines[2];
static int history[CAPACITY];
static int assertions, failures, cases, mock_errors, allocations;
static const char *label;
#define CHECK(x) do { assertions++; if(!(x)) { failures++; if(failures<=32) \
    fprintf(stderr,"FAIL %s line %d: %s\n",label,__LINE__,#x); } } while(0)
#define MOCK(x) do { if(!(x)) { mock_errors++; if(mock_errors<=8) \
    fprintf(stderr,"MOCK %s line %d: %s\n",label,__LINE__,#x); } } while(0)
static struct {
    bool failed, in_recent, send_fail, wait_fail, local_unavailable;
    bool worker_exited, invalidate_fail, eval_clears_checkpoint;
    bool fail_restore[2], fail_prefix[2];
    int send_calls, wait_calls, mark_failed_calls, invalidates, invalidate_deliveries, eval_calls, eval_fail_at;
    int restored[2], committed[2], capture_invalidations[2];
    int sent_base, worker_status, original_start, expected_base;
    uint64_t ack_session;
    unsigned ack_reserved;
    bool ack_ok;
} wire;
int ds4_session_restore_speculative_prefix(ds4_session *,int);

/* State values depend on the whole bounded token prefix, position and rank,
   not just a length counter. Prefix snapshots deliberately exclude drafter state. */
static void reference_state(uint32_t out[3],int pos,int rank,int domain) {
    out[0]=17u+(unsigned)rank; out[1]=31u+(unsigned)domain; out[2]=47u;
    for(int i=0;i<pos;i++) {
        out[0]=out[0]*33u+(unsigned)history[i]+1u;
        out[1]^=(out[0]+(unsigned)i)*65599u;
        out[2]=out[2]*17u+out[1]+(unsigned)domain;
    }
}
static void set_target(ds4_session *s,int pos) {
    s->graph.target_pos=pos; reference_state(s->graph.target,pos,s->rank,1);
}
static void set_draft(ds4_session *s,int pos) {
    s->graph.draft_pos=pos; reference_state(s->graph.draft,pos,s->rank,2);
}
static bool state_matches(ds4_session *s,int target,int draft) {
    uint32_t a[3],b[3]; reference_state(a,target,s->rank,1); reference_state(b,draft,s->rank,2);
    return s->graph.target_pos==target && s->graph.draft_pos==draft &&
           !memcmp(a,s->graph.target,sizeof(a)) && !memcmp(b,s->graph.draft,sizeof(b));
}
static void set_logits(ds4_session *s,int pos) {
    for(int i=0;i<DS4_N_VOCAB;i++) s->logits[i]=(float)(10000+pos*101+s->rank*1000+i);
}
static bool logits_match(ds4_session *s,int pos) {
    for(int i=0;i<DS4_N_VOCAB;i++)
        if(s->logits[i]!=(float)(10000+pos*101+s->rank*1000+i)) return false;
    return true;
}
static void *xmalloc(size_t n) {
    MOCK(n==DS4_N_VOCAB*sizeof(float)); allocations++;
    void *p=malloc(n); if(!p) abort(); return p;
}
static bool ds4_session_tp_leader(ds4_session *s) { return s->engine->tp.active && s->leader; }
static void ds4_session_dspark_capture_invalidate(ds4_session *s) {
    wire.capture_invalidations[s->rank]++; s->capture_valid=false;
    s->dspark_rewind_end=0;
    s->dspark_draft_valid=s->dspark_tp_proposal=false; s->dspark_draft_len=0;
}
static bool spec_frontier_restore(ds4_spec_frontier *f,ds4_session *s) {
    wire.restored[s->rank]++;
    MOCK(f==&s->dspark_rewind_frontier && f->epoch==s->epoch);
    MOCK(!s->capture_valid);
    if(wire.in_recent && ds4_session_tp_leader(s)) MOCK(wire.send_calls==1 && wire.wait_calls==0);
    if(wire.fail_restore[s->rank]) { s->graph.target[0]^=1u; return false; }
    s->graph=f->state;
    return true;
}
static bool spec_frontier_commit_prefix(ds4_session *s,uint32_t n) {
    wire.committed[s->rank]++;
    MOCK(n>0 && n<=DS4_SPEC_PREFIX_SLOTS && n<=s->dspark_rewind_prefixes);
    MOCK(wire.restored[s->rank]==1 && s->graph.target_pos==s->dspark_rewind_start);
    if(n==0 || n>DS4_SPEC_PREFIX_SLOTS) return false;
    if(wire.fail_prefix[s->rank]) { s->graph.target[1]^=1u; return false; }
    s->graph.target_pos=s->prefixes[n-1].target_pos;
    memcpy(s->graph.target,s->prefixes[n-1].target,sizeof(s->graph.target));
    return true;
}
static int ds4_tp_send_spec_restore(void *ctx,uint64_t id,int base) {
    wire.send_calls++;
    MOCK(ctx==&wire && id==42 && wire.in_recent && !wire.eval_calls && !wire.wait_calls);
    MOCK(wire.restored[0]==0 && base==wire.expected_base);
    if(wire.send_fail) { wire.failed=true; return 0; }
    wire.sent_base=base;
    /* The worker executes the actual extracted restore helper, then queues its
       status. A queued response alone never authorizes coordinator replay. */
    wire.worker_status=ds4_session_restore_speculative_prefix(&sessions[1],base);
    wire.worker_exited=wire.worker_status<0;
    if(wire.local_unavailable) sessions[0].checkpoint_valid=false;
    return 1;
}
static int ds4_tp_wait_command_status(void *ctx,uint64_t id,int *status,
        const char *operation,char *err,size_t errlen) {
    (void)err; (void)errlen; wire.wait_calls++;
    MOCK(ctx==&wire && id==42 && wire.send_calls==1 && !wire.eval_calls);
    MOCK(!strcmp(operation,"speculative restore"));
    MOCK(wire.restored[0]==1 || wire.local_unavailable);
    /* Model the public status API's failure contract, not its frame parser. */
    if(wire.wait_fail || wire.ack_session!=id || wire.ack_reserved) {
        wire.failed=true; return 0;
    }
    *status=wire.worker_status; wire.ack_ok=(*status==0); return 1;
}
static bool ds4_tp_failed(void *ctx) { MOCK(ctx==&wire); return wire.failed; }
static void ds4_tp_mark_failed(void *ctx) {
    MOCK(ctx==&wire && wire.in_recent && wire.wait_calls==1);
    MOCK(!wire.invalidates && !wire.eval_calls);
    MOCK(wire.worker_status<0 || wire.wait_fail || wire.ack_session!=42 || wire.ack_reserved);
    wire.mark_failed_calls++; wire.failed=true;
}
static int ds4_tp_send_invalidate(void *ctx,uint64_t id) {
    MOCK(ctx==&wire && id==42 && wire.wait_calls==1 && !wire.failed && !wire.eval_calls);
    wire.invalidates++;
    if(wire.worker_exited || wire.invalidate_fail) { wire.failed=true; return 0; }
    wire.invalidate_deliveries++;
    sessions[1].checkpoint_valid=false; sessions[1].checkpoint.len=0;
    ds4_session_dspark_capture_invalidate(&sessions[1]);
    return 1;
}
static int ds4_session_eval_probe_tp(ds4_session *s,int token,bool prepare,char *err,size_t errlen) {
    (void)err; (void)errlen; wire.eval_calls++;
    MOCK(s==&sessions[0] && !prepare && s->checkpoint_valid && !wire.failed);
    MOCK(wire.restored[0]==1 && s->checkpoint.len==s->graph.target_pos);
    int pos=s->checkpoint.len;
    MOCK(pos>=0 && pos<CAPACITY && token==history[pos]);
    if(ds4_session_tp_leader(s)) {
        MOCK(wire.wait_calls==1 && wire.ack_ok && wire.worker_status==0);
        MOCK(!wire.worker_exited);
        MOCK(sessions[1].checkpoint_valid && sessions[1].checkpoint.len==pos);
        MOCK(sessions[1].graph.target_pos==pos);
    }
    if(wire.eval_fail_at==wire.eval_calls) {
        wire.failed=ds4_session_tp_leader(s);
        if(wire.failed) sessions[1].checkpoint_valid=false;
        if(wire.eval_clears_checkpoint) {
            s->checkpoint.len=0; s->checkpoint_valid=false;
            ds4_session_dspark_capture_invalidate(s);
        }
        return -1;
    }
    int ranks=ds4_session_tp_leader(s) ? 2 : 1;
    for(int rank=0;rank<ranks;rank++) {
        ds4_session *peer=&sessions[rank];
        MOCK(state_matches(peer,pos,wire.eval_calls==1 ? (wire.original_start>0 ? wire.original_start-1 : 0) : pos));
        peer->checkpoint.v[peer->checkpoint.len++]=token;
        set_target(peer,pos+1); set_draft(peer,pos+1); set_logits(peer,pos+1);
        /* Ordinary evaluation retires the saved verifier range. */
        peer->dspark_rewind_end=0;
    }
    return 0;
}
'''


TESTS = r'''
static int minimum(int a,int b) { return a<b ? a : b; }
static void setup(int start,int rows,int kept,bool mirror) {
    for(int rank=0;rank<2;rank++) free(sessions[rank].dspark_rewind_logits);
    memset(sessions,0,sizeof(sessions)); memset(engines,0,sizeof(engines));
    memset(&wire,0,sizeof(wire)); mock_errors=allocations=0;
    wire.original_start=start; wire.ack_session=42;
    for(int i=0;i<CAPACITY;i++) history[i]=(i*3+5)%DS4_N_VOCAB;
    for(int rank=0;rank<2;rank++) {
        ds4_session *s=&sessions[rank]; s->rank=rank; s->engine=&engines[rank];
        s->engine->tp.active=mirror; s->engine->tp.ctx=&wire; s->leader=(rank==0);
        s->tp_session_id=42; s->epoch=100u+(unsigned)rank;
        s->checkpoint.v=s->storage; memcpy(s->storage,history,sizeof(history));
        s->checkpoint_valid=true; s->checkpoint.len=start;
        s->mtp_draft_valid=s->dspark_stochastic_draft=s->capture_valid=true;
        s->dspark_draft_valid=s->dspark_tp_proposal=true; s->dspark_draft_len=3;
        set_target(s,start); set_draft(s,start>0 ? start-1 : 0); set_logits(s,start);
        ds4_spec_frontier snapshot={s->graph,s->epoch};
        dspark_rewind_begin(s,&snapshot,start,rows);
        for(int n=1;n<=DS4_SPEC_PREFIX_SLOTS;n++) {
            set_target(s,start+n); s->prefixes[n-1]=s->graph;
        }
        set_target(s,start+kept); set_draft(s,start+kept); set_logits(s,start+kept);
        s->checkpoint.len=start+kept; dspark_rewind_keep(s);
    }
}
static bool invoke(int pos) {
    ds4_session *s=&sessions[0]; int start=s->dspark_rewind_start;
    wire.expected_base=pos==start ? start : minimum(pos-1,start+(int)s->dspark_rewind_prefixes);
    wire.in_recent=true;
    bool ok=dspark_rewind_recent(s,pos);
    wire.in_recent=false;
    return ok;
}
static void check_no_activity(void) {
    CHECK(!wire.send_calls && !wire.wait_calls && !wire.mark_failed_calls &&
          !wire.invalidates && !wire.eval_calls);
    for(int rank=0;rank<2;rank++)
        CHECK(!wire.restored[rank] && !wire.committed[rank] && !wire.capture_invalidations[rank]);
    CHECK(mock_errors==0);
}
static void test_capture(void) {
    label="snapshot range, owned start logits and captured-prefix clamp";
    for(int rows=1;rows<=DS4_DSPARK_MAX_BLOCK_SIZE;rows++) {
        setup(3,rows,rows,false); ds4_session *s=&sessions[0];
        CHECK(s->dspark_rewind_prefixes==(uint32_t)minimum(rows-1,DS4_SPEC_PREFIX_SLOTS));
        CHECK(s->dspark_rewind_start==3 && s->dspark_rewind_end==3+rows);
        CHECK(s->dspark_rewind_logits!=s->logits && allocations==2);
        float *owned=s->dspark_rewind_logits;
        s->checkpoint.len=2; set_target(s,2); set_draft(s,1); set_logits(s,2);
        ds4_spec_frontier frontier={s->graph,s->epoch};
        dspark_rewind_begin(s,&frontier,2,rows);
        CHECK(s->dspark_rewind_end==0 && s->dspark_rewind_logits==owned && allocations==2);
        CHECK(!memcmp(owned,s->logits,sizeof(s->logits)));
        frontier.state.target[0]^=1u; s->logits[0]=-999.0f;
        CHECK(s->dspark_rewind_frontier.state.target[0]!=frontier.state.target[0]);
        CHECK(owned[0]==10202.0f);
        CHECK(!invoke(2)); check_no_activity(); cases++;
    }
}
static void test_restore_prefixes(void) {
    label="actual restore, every available prefix and state-only interior logits";
    CHECK(ds4_session_restore_speculative_prefix(NULL,0)==1);
    for(int start=0;start<=7;start+=7) for(int rows=1;rows<=DS4_DSPARK_MAX_BLOCK_SIZE;rows++)
    for(int prefix=0;prefix<=minimum(rows-1,DS4_SPEC_PREFIX_SLOTS);prefix++) {
        setup(start,rows,rows,false); ds4_session *s=&sessions[0];
        CHECK(ds4_session_restore_speculative_prefix(s,start+prefix)==0);
        CHECK(s->checkpoint_valid && s->checkpoint.len==start+prefix);
        CHECK(state_matches(s,start+prefix,start>0 ? start-1 : 0));
        CHECK(logits_match(s,prefix ? start+rows : start));
        CHECK(!s->mtp_draft_valid && !s->dspark_stochastic_draft && !s->capture_valid);
        CHECK(!s->dspark_draft_valid && !s->dspark_tp_proposal && !s->dspark_draft_len);
        CHECK(s->dspark_rewind_end==0);
        CHECK(wire.restored[0]==1 && wire.committed[0]==(prefix!=0));
        CHECK(!memcmp(s->storage,history,sizeof(history)) && mock_errors==0);
        /* A second restore cannot reuse an end tied to the previous length. */
        CHECK(ds4_session_restore_speculative_prefix(s,start)==1);
        CHECK(wire.restored[0]==1); cases++;
    }
}
static void test_recent_prefixes(void) {
    label="bounded replay, partial retained blocks, both ranks and start zero";
    for(int mirror=0;mirror<=1;mirror++) for(int start=0;start<=7;start+=7)
    for(int rows=1;rows<=DS4_DSPARK_MAX_BLOCK_SIZE;rows++) for(int kept=1;kept<=rows;kept++)
    for(int prefix=0;prefix<kept;prefix++) {
        setup(start,rows,kept,mirror!=0);
        int base=prefix ? minimum(prefix-1,minimum(rows-1,DS4_SPEC_PREFIX_SLOTS)) : 0;
        int replay=prefix-base, pos=start+prefix;
        CHECK(invoke(pos));
        CHECK(wire.eval_calls==replay && (prefix==0 ? replay==0 : replay>=1));
        CHECK(replay<=DS4_DSPARK_MAX_BLOCK_SIZE);
        for(int rank=0;rank<(mirror ? 2 : 1);rank++) {
            ds4_session *peer=&sessions[rank];
            CHECK(peer->checkpoint_valid && peer->checkpoint.len==pos);
            CHECK(state_matches(peer,pos,replay ? pos : (start>0 ? start-1 : 0)));
            CHECK(logits_match(peer,pos));
            CHECK(!peer->mtp_draft_valid && !peer->dspark_stochastic_draft);
            CHECK(wire.restored[rank]==1 && wire.committed[rank]==(base!=0));
            CHECK(!memcmp(peer->storage,history,sizeof(history)));
        }
        CHECK(wire.send_calls==mirror && wire.wait_calls==mirror && !wire.invalidates);
        if(mirror) CHECK(wire.sent_base==start+base && wire.ack_ok);
        CHECK(mock_errors==0);
        /* Even a zero-replay restore makes the old end/length pair stale. */
        CHECK(!invoke(pos)); CHECK(wire.eval_calls==replay); cases++;
    }
    label="replay clamps to actual captured prefix count, including zero";
    for(int captured=0;captured<=DS4_SPEC_PREFIX_SLOTS;captured++) {
        setup(2,16,16,true);
        for(int rank=0;rank<2;rank++) sessions[rank].dspark_rewind_prefixes=(uint32_t)captured;
        CHECK(invoke(17)); CHECK(wire.sent_base==2+captured);
        CHECK(wire.eval_calls==15-captured && logits_match(&sessions[0],17));
        CHECK(state_matches(&sessions[0],17,17) && mock_errors==0); cases++;
    }
}
static void test_guards(void) {
    label="invalid, stale and out-of-range snapshots do not execute";
    for(int kind=0;kind<10;kind++) for(int recent=0;recent<=1;recent++) {
        setup(4,8,8,false); ds4_session *s=&sessions[0]; int pos=6;
        float *owned=s->dspark_rewind_logits;
        switch(kind) {
            case 0: s->checkpoint_valid=false; break;
            case 1: s->dspark_rewind_logits=NULL; break;
            case 2: s->dspark_rewind_end=s->dspark_rewind_start; break;
            case 3: s->dspark_rewind_end=s->dspark_rewind_start-1; break;
            case 4: s->checkpoint.len++; break;
            case 5: s->checkpoint.len--; break;
            case 6: pos=3; break;
            case 7: pos=12; break;
            case 8: pos=13; break;
            case 9: pos=-1; break;
        }
        unsigned char before[sizeof(*s)]; memcpy(before,s,sizeof(*s));
        if(recent) CHECK(!invoke(pos));
        else CHECK(ds4_session_restore_speculative_prefix(s,pos)==1);
        CHECK(!memcmp(before,s,sizeof(*s))); check_no_activity();
        s->dspark_rewind_logits=owned; cases++;
    }
    setup(3,16,16,false);
    CHECK(ds4_session_restore_speculative_prefix(&sessions[0],4+DS4_SPEC_PREFIX_SLOTS)==1);
    check_no_activity(); cases++;
    setup(3,16,16,false); sessions[0].checkpoint.len=20; sessions[0].dspark_rewind_end=20;
    CHECK(!invoke(5)); check_no_activity(); cases++;
    setup(0,6,6,true); sessions[0].leader=false;
    CHECK(!invoke(1)); check_no_activity(); cases++;
    setup(0,6,6,false); sessions[0].dspark_rewind_frontier.epoch++;
    sessions[0].dspark_rewind_end=0;
    CHECK(!invoke(1)); check_no_activity(); cases++;
}
static void test_failures(void) {
    label="local restore or target-prefix failure invalidates checkpoint";
    for(int prefix=0;prefix<=2;prefix+=2) for(int commit=0;commit<=1;commit++) {
        if(commit && !prefix) continue;
        setup(2,8,8,false);
        wire.fail_restore[0]=!commit; wire.fail_prefix[0]=commit!=0;
        CHECK(ds4_session_restore_speculative_prefix(&sessions[0],2+prefix)==-1);
        CHECK(!sessions[0].checkpoint_valid && wire.eval_calls==0 && mock_errors==0); cases++;
    }
    label="send failure never executes either restore or replay";
    setup(2,8,8,true); wire.send_fail=true;
    CHECK(!invoke(5)); CHECK(!sessions[0].checkpoint_valid);
    CHECK(wire.send_calls==1 && !wire.wait_calls && !wire.restored[0] && !wire.restored[1]);
    CHECK(!wire.eval_calls && !wire.invalidates && !wire.mark_failed_calls && mock_errors==0); cases++;

    /* Nonzero statuses are independent: consume the reply even after local
       failure. A fatal worker exits; mark the link failed before any fallback. */
    label="both-rank ACK failures never authorize replay";
    for(int local=0;local<3;local++) for(int worker=0;worker<3;worker++) {
        if(!local && !worker) continue;
        setup(2,8,8,true);
        wire.local_unavailable=local==1; wire.fail_restore[0]=local==2;
        if(worker==1) sessions[1].checkpoint_valid=false;
        wire.fail_restore[1]=worker==2;
        CHECK(!invoke(5)); CHECK(wire.send_calls==1 && wire.wait_calls==1);
        CHECK(wire.worker_status==(worker==2 ? -1 : worker));
        CHECK(wire.invalidates==(worker==2 ? 0 : 1) && !wire.eval_calls);
        CHECK(wire.invalidate_deliveries==(worker==2 ? 0 : 1));
        CHECK(wire.mark_failed_calls==(worker==2 ? 1 : 0));
        CHECK(wire.failed==(worker==2));
        CHECK(!sessions[0].checkpoint_valid && !sessions[1].checkpoint_valid);
        CHECK(sessions[0].checkpoint.len==5 && sessions[0].dspark_rewind_end==0);
        CHECK(!memcmp(sessions[0].storage,history,sizeof(history)) && mock_errors==0); cases++;
    }
    label="prefix-copy failure ACK permits invalidation only with live worker";
    for(int rank=0;rank<2;rank++) {
        setup(2,8,8,true); wire.fail_prefix[rank]=true;
        CHECK(!invoke(5)); CHECK(wire.committed[rank]==1 && wire.wait_calls==1);
        CHECK(wire.invalidates==(rank==1 ? 0 : 1) && !wire.eval_calls && !sessions[0].checkpoint_valid);
        CHECK(wire.invalidate_deliveries==(rank==1 ? 0 : 1));
        CHECK(wire.mark_failed_calls==(rank==1 ? 1 : 0) && wire.failed==(rank==1));
        CHECK(!sessions[1].checkpoint_valid && mock_errors==0); cases++;
    }
    label="worker newer checkpoint cannot authorize local replay";
    setup(2,8,8,true); sessions[1].checkpoint.len++;
    CHECK(!invoke(5)); CHECK(wire.worker_status==1 && wire.wait_calls==1);
    CHECK(!wire.eval_calls && wire.invalidate_deliveries==1 && !wire.mark_failed_calls && !wire.failed);
    CHECK(!sessions[0].checkpoint_valid && !sessions[1].checkpoint_valid && mock_errors==0); cases++;

    label="invalidation transport failure leaves local state unusable";
    setup(2,8,8,true); wire.fail_restore[0]=true; wire.invalidate_fail=true;
    CHECK(!invoke(5)); CHECK(wire.invalidates==1 && !wire.invalidate_deliveries && wire.failed);
    CHECK(!wire.eval_calls && !sessions[0].checkpoint_valid);
    CHECK(sessions[0].checkpoint.len==5 && sessions[0].dspark_rewind_end==0 && mock_errors==0); cases++;

    label="missing, wrong-session or reserved-bit ACK never permits replay";
    for(int error=0;error<3;error++) {
        setup(2,8,8,true);
        if(error==0) wire.wait_fail=true;
        if(error==1) wire.ack_session=43;
        if(error==2) wire.ack_reserved=1;
        CHECK(!invoke(5)); CHECK(wire.wait_calls==1 && !wire.eval_calls);
        CHECK(!wire.invalidates && wire.failed && !sessions[0].checkpoint_valid);
        CHECK(wire.mark_failed_calls==1);
        CHECK(sessions[0].checkpoint.len==5 && sessions[0].dspark_rewind_end==0);
        CHECK(mock_errors==0); cases++;
    }
    label="replay failure stops immediately and retains requested transcript length";
    for(int mirror=0;mirror<=1;mirror++) for(int fail=1;fail<=3;fail++) for(int cleared=0;cleared<=1;cleared++) {
        setup(0,8,8,mirror!=0); wire.eval_fail_at=fail; wire.eval_clears_checkpoint=cleared!=0;
        sessions[0].dspark_rewind_prefixes=sessions[1].dspark_rewind_prefixes=0;
        CHECK(!invoke(3)); CHECK(wire.eval_calls==fail && !sessions[0].checkpoint_valid);
        CHECK(sessions[0].checkpoint.len==3 && !wire.invalidates && mock_errors==0); cases++;
        CHECK(!memcmp(sessions[0].storage,history,sizeof(history)));
    }
}
static void test_public_wrapper(void) {
    label="public wrapper rejects null, missing engine and negative positions";
    setup(0,6,6,false);
    CHECK(!ds4_session_rewind_speculative(NULL,0)); check_no_activity(); cases++;
    const int negative[]={INT_MIN,-1};
    for(unsigned i=0;i<sizeof(negative)/sizeof(negative[0]);i++) {
        ds4_session before=sessions[0];
        CHECK(!ds4_session_rewind_speculative(&sessions[0],negative[i]));
        CHECK(!memcmp(&before,&sessions[0],sizeof(before))); check_no_activity(); cases++;
    }
    sessions[0].engine=NULL;
    ds4_session before=sessions[0];
    CHECK(!ds4_session_rewind_speculative(&sessions[0],0));
    CHECK(!memcmp(&before,&sessions[0],sizeof(before))); check_no_activity(); cases++;
    label="public wrapper preserves requested position and valid logits";
    for(int pos=0;pos<=3;pos+=3) {
        setup(0,6,6,false);
        CHECK(ds4_session_rewind_speculative(&sessions[0],pos));
        CHECK(sessions[0].checkpoint_valid && sessions[0].checkpoint.len==pos);
        CHECK(state_matches(&sessions[0],pos,pos) && logits_match(&sessions[0],pos));
        CHECK(wire.eval_calls==(pos ? 1 : 0) && mock_errors==0); cases++;
    }
    label="public wrapper reports unavailable without fallback execution";
    setup(0,6,6,false); sessions[0].dspark_rewind_end=0; before=sessions[0];
    CHECK(!ds4_session_rewind_speculative(&sessions[0],3));
    CHECK(!memcmp(&before,&sessions[0],sizeof(before))); check_no_activity(); cases++;
}
int main(void) {
    test_capture(); test_restore_prefixes(); test_recent_prefixes(); test_guards(); test_failures();
    test_public_wrapper();
    for(int rank=0;rank<2;rank++) free(sessions[rank].dspark_rewind_logits);
    printf("{\"prefix_slots\":%d,\"cases\":%d,\"assertions\":%d,\"failures\":%d}\n",
           DS4_SPEC_PREFIX_SLOTS,cases,assertions,failures);
    return failures ? 1 : 0;
}
'''


NO_GPU_PRELUDE = r'''
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
typedef struct { unsigned char untouched[64]; } ds4_session;
'''
NO_GPU_TESTS = r'''
int main(void) {
    ds4_session s,before; memset(&s,0xa5,sizeof(s)); before=s;
    const int positions[]={INT_MIN,-1,0,1,16,INT_MAX};
    int failures=0;
    for(unsigned i=0;i<sizeof(positions)/sizeof(positions[0]);i++) {
        if(ds4_session_restore_speculative_prefix(&s,positions[i])!=1 ||
           memcmp(&s,&before,sizeof(s))) failures++;
    }
    if(ds4_session_restore_speculative_prefix(NULL,0)!=1) failures++;
    for(unsigned i=0;i<sizeof(positions)/sizeof(positions[0]);i++) {
        if(ds4_session_rewind_speculative(&s,positions[i]) ||
           memcmp(&s,&before,sizeof(s))) failures++;
    }
    if(ds4_session_rewind_speculative(NULL,0)) failures++;
    printf("{\"no_gpu_stub\":true,\"cases\":14,\"failures\":%d}\n",failures);
    return failures ? 1 : 0;
}
'''


def fixtures(source):
    if not re.search(r'^#define DS4_DSPARK_MAX_BLOCK_SIZE 16$', source, re.M):
        raise ValueError('production block capacity changed; update the bounded fixture explicitly')
    slots = {int(n) for n in re.findall(r'^#define DS4_SPEC_PREFIX_SLOTS (\d+)$', source, re.M)}
    if slots != {4, 5}:
        raise ValueError('production prefix capacities changed; update the checked variants explicitly')
    helpers = {name: extract(source, name) for name in FUNCTIONS}
    body = '\n'.join([PRELUDE, *helpers.values(), TESTS])
    variants = {f'prefix_slots_{n}': f'#define DS4_SPEC_PREFIX_SLOTS {n}\n' + body for n in sorted(slots)}
    variants['no_gpu'] = '\n'.join(['#define DS4_NO_GPU 1', NO_GPU_PRELUDE,
        helpers['ds4_session_restore_speculative_prefix'],
        helpers['ds4_session_rewind_speculative'], NO_GPU_TESTS])
    return helpers, variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'ds4.c')
    parser.add_argument('--output', type=Path, help='exclusive-new evidence directory')
    parser.add_argument('--prepare-only', action='store_true', help='extract only; no compiler or test executable')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    helpers, variants = fixtures(raw.decode('utf-8'))
    sha = lambda data: hashlib.sha256(data).hexdigest()
    report = {'scope': 'actual extracted rewind/begin/keep functions; bounded target/drafter state and checked TP mocks',
              'coverage_limits': 'No model, backend arithmetic, real GPU/KV buffers, actual worker execution/ACK parsing, or public rewind fallback proof. A worker-fatal status is terminal, not successful two-rank recovery.',
              'source': str(args.source.resolve()), 'source_sha256': sha(raw),
              'helper_sha256': {name: sha(body.encode()) for name, body in helpers.items()},
              'test_sha256': sha(Path(__file__).read_bytes()),
              'prepared_only': args.prepare_only,
              'variants': {name: {'emitted_c_sha256': sha(code.encode())} for name, code in variants.items()}}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.prepare_only:
        if args.output:
            for name, code in variants.items():
                (args.output / (name + '.c')).write_text(code)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return 0
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    if not compiler:
        raise ValueError('empty CC')
    report['compiler_version'] = subprocess.run(compiler + ['--version'], capture_output=True,
                                               text=True, timeout=10).stdout
    report['returncode'] = 0
    env = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
    with tempfile.TemporaryDirectory(prefix='ds4-rewind-contract-') as temporary:
        root = args.output or Path(temporary)
        for name, code in variants.items():
            c, binary = root / (name + '.c'), root / name
            c.write_text(code)
            command = compiler + ['-std=c11', '-O0', '-fno-fast-math', '-Wall', '-Wextra', '-Werror',
                                  str(c), '-o', str(binary)]
            build = subprocess.run(command, capture_output=True, text=True, timeout=60)
            result = report['variants'][name]
            result.update(build_command=command, build_returncode=build.returncode,
                          build_stdout=build.stdout, build_stderr=build.stderr, returncode=build.returncode)
            if build.returncode == 0:
                run = subprocess.run([str(binary.resolve())], capture_output=True, text=True,
                                     env=env, timeout=30)
                result.update(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
                              binary_sha256=sha(binary.read_bytes()),
                              results=[json.loads(line) for line in run.stdout.splitlines()])
            if result['returncode']:
                report['returncode'] = 1
        if args.output:
            (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return report['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
