#!/usr/bin/env python3
"""Check cancellation ownership at the actual speculative rebuild call site.

The mock sync replaces compressed history and checks that request rollback has
already been retired. Fast rewinds retain rollback. Both successful and failed
rebuilds must prevent a later cancellation from restoring stale history.
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

def extract(source, name):
    matches=list(re.finditer(r'static (?:int|bool) '+re.escape(name)+r'\s*\(',source))
    if len(matches)!=1:
        raise ValueError('require exactly one production helper: '+name)
    start=matches[0].start();opening=source.index('{',start)
    token=re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',re.S)
    depth=0
    for match in token.finditer(source,opening):
        if match.group()=='{': depth+=1
        elif match.group()=='}':
            depth-=1
            if depth==0: return source[start:match.end()]
    raise ValueError('unterminated production helper: '+name)

PRELUDE = r'''
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
typedef struct { int len; } ds4_tokens;
typedef struct { int identity; } ds4_vision_span;
typedef struct { ds4_tokens tokens; bool valid; } ds4_session;
typedef struct { int compressed_history; } ds4_cancel_checkpoint;
typedef enum { DS4_SESSION_REWRITE_OK, DS4_SESSION_REWRITE_REBUILD_NEEDED } ds4_session_rewrite_result;
typedef struct { int inference_mu; } server;
typedef struct { ds4_session *session; } server_slot;
typedef struct { const ds4_vision_span *images; size_t image_count; } request;
typedef struct { request req; } job;
static int locked, allocations, rewinds, syncs, frees, tests;
static bool needs_rebuild, sync_fails;
static ds4_cancel_checkpoint **observed_checkpoint;
static bool *observed_preservable;
static char *observed_error;
static const ds4_vision_span *expected_images;
static size_t expected_image_count;
static void pthread_mutex_lock(int *mu) { (void)mu; CHECK(!locked); locked=1; }
static void pthread_mutex_unlock(int *mu) { (void)mu; CHECK(locked); locked=0; }
static void ds4_session_cancel_checkpoint_free(ds4_cancel_checkpoint *checkpoint) {
    CHECK(!locked);
    if (checkpoint) { CHECK(checkpoint==*observed_checkpoint); frees++; }
}
static void ds4_session_rewind(ds4_session *s,int pos) {
    CHECK(locked && pos<s->tokens.len); rewinds++;
    s->tokens.len=pos; s->valid=!needs_rebuild;
}
static const ds4_tokens *ds4_session_tokens(ds4_session *s) { CHECK(locked); return &s->tokens; }
static void ds4_tokens_copy(ds4_tokens *dst,const ds4_tokens *src) { CHECK(locked); *dst=*src; allocations++; }
static void ds4_tokens_free(ds4_tokens *p) { (void)p; CHECK(!locked && allocations==1); allocations--; }
static int ds4_session_common_prefix(ds4_session *s,const ds4_tokens *p) { CHECK(locked); return s->valid?p->len:0; }
static int ds4_session_pos(ds4_session *s) { CHECK(locked); return s->tokens.len; }
static bool ds4_session_checkpoint_valid(ds4_session *s) { CHECK(locked); return s->valid; }
static bool ds4_session_vision_state_matches(ds4_session *s,const ds4_vision_span *images,size_t n) {
    (void)s; CHECK(locked && images==expected_images && n==expected_image_count); return true;
}
static void ds4_session_invalidate(ds4_session *s) { CHECK(locked); s->valid=false; }
static int server_session_sync_multimodal(server *s,server_slot *slot,const ds4_tokens *p,
        const ds4_vision_span *images,size_t n,char *err,size_t errlen) {
    (void)s; CHECK(!locked && images==expected_images && n==expected_image_count);
    /* This is the destructive boundary: cancellation can no longer trust the
     * old append-only compressed rows, even if this sync subsequently fails. */
    CHECK(observed_checkpoint && *observed_checkpoint==NULL);
    CHECK(observed_preservable && !*observed_preservable);
    CHECK(strstr(observed_error,"destructive checkpoint rebuild"));
    syncs++;
    if (sync_fails) { snprintf(err,errlen,"sync interrupted after replacing history"); return 1; }
    slot->session->tokens=*p; slot->session->valid=true; return 0;
}
'''

WRAPPER = r'''
static int boundary(server *s,server_slot *slot,job *j,int pos,
        ds4_cancel_checkpoint **saved,bool *preservable,char *saved_error) {
    ds4_cancel_checkpoint *cancel_checkpoint=*saved;
    bool cancel_prompt_frontier_preservable=*preservable;
    char cancel_checkpoint_err[160]={0},err[160]={0};
    observed_checkpoint=&cancel_checkpoint;
    observed_preservable=&cancel_prompt_frontier_preservable;
    observed_error=cancel_checkpoint_err;
    int rc=PRODUCTION_CALL;
    *saved=cancel_checkpoint; *preservable=cancel_prompt_frontier_preservable;
    strcpy(saved_error,cancel_checkpoint_err);
    return rc;
}
'''

TESTS = r'''
int main(void) {
    ds4_vision_span image={47};
    for (int with_image=0;with_image<2;with_image++)
    for (int with_checkpoint=0;with_checkpoint<2;with_checkpoint++)
    for (int preservable=0;preservable<2;preservable++)
    for (int rebuild=0;rebuild<2;rebuild++)
    for (int failure=0;failure<2;failure++) {
        CHECK(!locked && !allocations);
        rewinds=syncs=frees=0; needs_rebuild=rebuild; sync_fails=failure;
        expected_images=with_image?&image:NULL; expected_image_count=with_image;
        ds4_cancel_checkpoint checkpoint={1},*saved=with_checkpoint?&checkpoint:NULL;
        ds4_session session={{32},true}; server s={0}; server_slot slot={&session};
        job j={{expected_images,expected_image_count}};
        bool can_preserve=preservable; char retired_error[160]={0};
        int rc=boundary(&s,&slot,&j,17,&saved,&can_preserve,retired_error);
        CHECK(rc==(rebuild&&failure));
        CHECK(rewinds==1 && syncs==rebuild && frees==(rebuild&&with_checkpoint));
        CHECK(!locked && !allocations);
        if (rebuild) {
            CHECK(saved==NULL && !can_preserve && retired_error[0]);
            /* A cancellation after either sync outcome has no stale snapshot
             * and cannot mistake a rebuilt prompt for an untouched prompt. */
        } else {
            CHECK(saved==(with_checkpoint?&checkpoint:NULL));
            CHECK(can_preserve==(bool)preservable && !retired_error[0]);
        }
        tests++;
    }
    printf("PASS cancellation/rebuild handoff: %d cases, actual helper and call site\n",tests);
    return 0;
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'ds4_server.c')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    raw=args.source.read_bytes(); source=raw.decode()
    retirement=extract(source,'begin_destructive_checkpoint_rebuild')
    rewind=extract(source,'server_generation_rewind')
    start=source.index('if (server_generation_rewind(',source.index('decode_again:'))+4
    end=source.index(' != 0',start)
    call=source[start:end]
    code=PRELUDE+retirement+'\n'+rewind+WRAPPER.replace('PRODUCTION_CALL',call)+TESTS
    if args.output: args.output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='ds4-cancel-rebuild-') as temporary:
        root=args.output or Path(temporary); c=root/'contract.c'; binary=root/'contract'
        c.write_text(code)
        command=shlex.split(os.environ.get('CC','cc'))+['-std=c11','-O0','-Wall','-Wextra','-Werror',
            '-Wno-unused-function',str(c),'-o',str(binary)]
        build=subprocess.run(command,capture_output=True,text=True,timeout=60)
        report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),
            retirement_sha256=hashlib.sha256(retirement.encode()).hexdigest(),
            rewind_sha256=hashlib.sha256(rewind.encode()).hexdigest(),
            call=call,command=command,build_exit=build.returncode,build_stderr=build.stderr)
        rc=build.returncode
        if not rc:
            run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
            rc=run.returncode;report.update(run_exit=rc,stdout=run.stdout,stderr=run.stderr)
        if args.output: (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report));return rc


if __name__=='__main__':
    raise SystemExit(main())
