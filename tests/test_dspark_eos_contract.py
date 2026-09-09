#!/usr/bin/env python3
"""Execute the actual DSpark greedy host function against checked target-state mocks.

Run with Python's standard library and a C compiler; no model or GPU is needed.
This checks commit/frontier ownership, not backend math or public-call dispatch.
An optional source path admits a pre-fix negative control without editing it.
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


FUNCTION = 'ds4_session_eval_dspark_speculative_argmax'


def extract(source, name=FUNCTION):
    matches = list(re.finditer(r'static int ' + re.escape(name) + r'\s*\(', source))
    if len(matches) != 1:
        raise ValueError('require exactly one production greedy helper')
    start = matches[0].start()
    opening = source.index('{', start)
    # Braces inside comments and strings must not delimit the actual function.
    token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match.group() == '{':
            depth += 1
        elif match.group() == '}':
            depth -= 1
            if depth == 0:
                return source[start:match.end()]
    raise ValueError('unterminated production helper')


PRELUDE = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DS4_DSPARK_MAX_BLOCK_SIZE 16
#define DS4_SPEC_PREFIX_SLOTS 4
#define DS4_N_VOCAB 16
#define DS4_TP_VERIFY_COMMIT_FULL 1
#define DS4_TP_VERIFY_COMMIT_PREFIX 2
#define DS4_TP_VERIFY_ROLLBACK_REPLAY 3
enum { DS4_BACKEND_METAL, DS4_BACKEND_CUDA, DS4_BACKEND_CPU };
typedef int ds4_think_mode;
typedef struct { int len, data[64]; } ds4_tokens;
typedef struct { int frontier; } ds4_gpu_graph;
typedef struct { int model, weights, backend; struct { void *ctx; bool vocab_split; } tp; } ds4_engine;
typedef struct {
    double upload_ms, layer_ms, head_ms, read_ms;
    bool fused_head;
} ds4_verify_suffix_timing;
typedef struct {
    ds4_engine *engine; ds4_tokens checkpoint; ds4_gpu_graph graph;
    bool dspark_draft_valid, checkpoint_valid; uint32_t dspark_draft_len;
    int dspark_draft_tokens[16], ctx_size, tp_session_id;
    float logits[DS4_N_VOCAB], *spec_row_logits;
    double dspark_last_propose_ms;
    struct { STATS_FIELDS } dspark_stats;
} ds4_session;
typedef struct { int frontier; } ds4_spec_frontier;
static int verified_width, verified_start, verified_tokens[16], target_next[16];
static int selected_row, prefix_committed, replayed, capture_notes, mock_errors;
static int tp_width, tp_tokens[16], tp_commit_kind, tp_commit_count, capture_prefix_calls;
static bool tp_leader, seed_batch, fail_prefix, fail_read, fail_verify, prefixes_captured;
static int extra_stop = -1;
static bool ds4_dspark_stats_enabled(void) { return true; }
static bool ds4_dspark_scheduler_enabled(ds4_session *s) { (void)s; return false; }
static bool ds4_dspark_scheduler_timing_enabled(void) { return false; }
static double now_sec(void) { return 1.0; }
static void ds4_dspark_stats_note_len(uint64_t *hist, uint32_t n) {
    if (n >= 17) { mock_errors++; return; } hist[n]++;
}
static void ds4_session_dspark_scheduler_note(ds4_session *s,uint32_t n,bool miss,double ms) {
    (void)s; (void)n; (void)miss; (void)ms;
}
static bool ds4_token_is_stop_for_think_mode(ds4_engine *e,int token,ds4_think_mode think) {
    (void)e; (void)think; return token == 1 || token == extra_stop;
}
static int sample_argmax(const float *p,int n) {
    int best=0; for(int i=1;i<n;i++) if(p[i]>p[best]) best=i; return best;
}
static void token_vec_push(ds4_tokens *t,int token) {
    if(t->len<0 || t->len>=64) { mock_errors++; return; } t->data[t->len++]=token;
}
static bool spec_frontier_snapshot(ds4_spec_frontier *f,ds4_session *s) {
    f->frontier=s->graph.frontier;
    if(f->frontier != s->checkpoint.len) mock_errors++;
    return true;
}
static void spec_frontier_free(ds4_spec_frontier *f) { (void)f; }
static bool spec_frontier_restore(ds4_spec_frontier *f,ds4_session *s) {
    s->graph.frontier=f->frontier; return true;
}
static bool ds4_session_tp_leader(ds4_session *s) { (void)s; return tp_leader; }
static int ds4_tp_send_verify(void *ctx,int id,const int *drafts,uint32_t n) {
    (void)ctx; (void)id;
    if(n==0 || n>16) { mock_errors++; return 0; }
    memcpy(tp_tokens,drafts,(size_t)n*sizeof(*drafts)); tp_width=(int)n; return 1;
}
static int ds4_tp_send_verify_commit(void *ctx,int kind,int count) {
    (void)ctx; tp_commit_kind=kind; tp_commit_count=count; return 1;
}
static int ds4_tp_recv_logits_half(void *ctx,float *p,uint32_t n) {
    (void)ctx; (void)p; (void)n; mock_errors++; return 0;
}
static bool metal_graph_verify_suffix_tops(ds4_gpu_graph *g,int *m,int *w,ds4_tokens *tokens,
        uint32_t start,uint32_t n,bool capture,bool keep,int *tops,void *unused,
        ds4_verify_suffix_timing *timing) {
    (void)m; (void)w; (void)keep; (void)unused;
    if(n==0 || n>16 || tokens->len!=(int)(start+n) || g->frontier!=(int)start) mock_errors++;
    prefixes_captured=capture;
    verified_width=(int)n; verified_start=(int)start;
    for(uint32_t i=0;i<n;i++) {
        verified_tokens[i]=tokens->data[start+i];
        if(tops) tops[i]=target_next[i];
    }
    g->frontier=(int)(start+n);
    if(timing) memset(timing,0,sizeof(*timing));
    return !fail_verify;
}
static void row_values(float *p,int frontier) {
    for(int i=0;i<DS4_N_VOCAB;i++) p[i]=(float)(frontier*100+i);
}
static bool metal_graph_read_spec_logits_row(ds4_gpu_graph *g,uint32_t row,float *p) {
    (void)g;
    if(row>=(uint32_t)verified_width) { mock_errors++; return false; }
    selected_row=(int)row;
    row_values(p,verified_start+(int)row+1);
    return !fail_read;
}
static bool ds4_session_dspark_seed_batch_enabled(ds4_session *s) { (void)s; return seed_batch; }
#define ds4_session_dspark_seed_batch_short_fallback(s) \
    ((s)->engine->backend == DS4_BACKEND_METAL)
static void ds4_session_dspark_capture_invalidate(ds4_session *s) { (void)s; }
static bool spec_frontier_commit_prefix(ds4_session *s,uint32_t n) {
    if(n==0 || n>4 || n>=(uint32_t)verified_width) { mock_errors++; return false; }
    if(fail_prefix || !prefixes_captured) return false;
    prefix_committed=(int)n; s->graph.frontier=verified_start+(int)n; return true;
}
static bool metal_graph_dspark_capture_commit_prefix(ds4_gpu_graph *g,uint32_t n) {
    if(g->frontier!=verified_start+(int)n) mock_errors++;
    capture_prefix_calls++; return true;
}
static void ds4_session_dspark_capture_note_checkpoint(ds4_session *s) {
    capture_notes++;
    if(s->checkpoint.len!=s->graph.frontier) mock_errors++;
}
static bool metal_graph_eval_token_raw_swa(ds4_gpu_graph *g,int *m,int *w,int token,uint32_t pos,float *p) {
    (void)m; (void)w;
    if(g->frontier!=(int)pos || token<0 || token>=DS4_N_VOCAB) { mock_errors++; return false; }
    g->frontier++; replayed++; row_values(p,g->frontier); return true;
}
static const char *ds4_backend_name(int backend) { (void)backend; return "mock"; }
'''


TESTS = r'''
static int failures, assertions, cases, observed_negative, unaffected_failures;
static const char *label;
#define CHECK(x) do { assertions++; if(!(x)) { failures++; if(failures<=24) \
    fprintf(stderr,"FAIL %s line %d: %s\n",label,__LINE__,#x); } } while(0)
static ds4_engine engine;
static ds4_session session;
static float output_logits[DS4_N_VOCAB];
static int accepted[20], start, already;
static void setup(const int *draft,int n,int seed,int reject) {
    memset(&engine,0,sizeof(engine)); memset(&session,0,sizeof(session));
    engine.backend=DS4_BACKEND_CUDA; /* CUDA and ROCm share this public backend. */
    session.engine=&engine; session.ctx_size=64; session.checkpoint_valid=true;
    session.checkpoint.data[0]=8; session.checkpoint.data[1]=9; session.checkpoint.len=2;
    for(int i=0;i<20;i++) accepted[i]=-99;
    already=seed;
    if(seed) { session.checkpoint.data[2]=2; session.checkpoint.len++; accepted[0]=2; }
    start=session.checkpoint.len; session.graph.frontier=start;
    session.dspark_draft_valid=true; session.dspark_draft_len=(uint32_t)n;
    memcpy(session.dspark_draft_tokens,draft,(size_t)n*sizeof(int));
    session.spec_row_logits=output_logits;
    for(int i=0;i<DS4_N_VOCAB;i++) session.logits[i]=-100;
    session.logits[(draft[0]>=0 && draft[0]<DS4_N_VOCAB) ? draft[0] : 0]=100;
    for(int i=0;i<n;i++) target_next[i]=(i+1<n) ? draft[i+1] : 5;
    if(reject==0) { for(int i=0;i<DS4_N_VOCAB;i++) session.logits[i]=-100; session.logits[15]=100; }
    if(reject>0) target_next[reject-1]=15;
    verified_width=verified_start=prefix_committed=replayed=capture_notes=mock_errors=0;
    tp_width=tp_commit_kind=tp_commit_count=capture_prefix_calls=0;
    selected_row=-1; seed_batch=(seed==0); tp_leader=false;
    fail_prefix=fail_read=fail_verify=prefixes_captured=false; extra_stop=-1;
}
static int invoke(int max_tokens,int cap,bool ignore) {
    char err[256]={0};
    return ds4_session_eval_dspark_speculative_argmax(&session,already,max_tokens,1,ignore,0,accepted,cap,err,sizeof(err));
}
static void check_frontier(int returned,int expected,int width,const int *draft,int cap) {
    CHECK(returned==already+expected);
    CHECK(session.checkpoint.len==start+expected);
    CHECK(session.graph.frontier==start+expected);
    CHECK(verified_width==width);
    CHECK(mock_errors==0);
    CHECK(session.checkpoint_valid);
    CHECK(!session.dspark_draft_valid && session.dspark_draft_len==0);
    for(int i=0;i<expected;i++) {
        CHECK(accepted[already+i]==draft[i]);
        CHECK(session.checkpoint.data[start+i]==draft[i]);
    }
    CHECK(accepted[cap]==-99 && accepted[19]==-99);
    if(expected) {
        for(int i=0;i<DS4_N_VOCAB;i++) CHECK(isfinite(session.logits[i]) && session.logits[i]==(float)((start+expected)*100+i));
        CHECK(capture_notes==1);
    }
    if(tp_leader) {
        CHECK(tp_width==width);
        for(int i=0;i<width;i++) CHECK(tp_tokens[i]==draft[i]);
        if(expected==width && width) CHECK(tp_commit_kind==DS4_TP_VERIFY_COMMIT_FULL);
        if(expected<width && expected && prefix_committed) {
            CHECK(tp_commit_kind==DS4_TP_VERIFY_COMMIT_PREFIX && tp_commit_count==expected);
        }
        if(expected<width && expected && !prefix_committed) {
            CHECK(tp_commit_kind==DS4_TP_VERIFY_ROLLBACK_REPLAY && tp_commit_count==expected);
        }
    }
    cases++;
}
int main(void) {
    /* Old source retains three evaluated rows but exposes only two tokens. */
    const int regression[]={2,1,1}; label="interior-EOS full commit";
    setup(regression,3,0,-1);
    int returned=invoke(8,8,false);
    printf("{\"probe\":\"interior_eos\",\"returned\":%d,\"advanced\":%d,\"verified\":%d}\n",returned,session.checkpoint.len-start,verified_width);
    observed_negative=(returned==2 && session.checkpoint.len-start==3);
    check_frontier(returned,2,2,regression,8);

    /* Distinct target rows exercise full/partial acceptance, including a
       rejection only AFTER EOS. The old partial branch can hide extra state
       behind a shortened checkpoint; graph frontier/logits reveal that. */
    for(int seed=0;seed<=1;seed++) for(int n=1;n<=16;n++) for(int eos=-1;eos<n;eos++)
    for(int ignore=0;ignore<=1;ignore++) for(int reject=-1;reject<n;reject++) {
        int draft[16]; for(int i=0;i<n;i++) draft[i]=3+(i%12);
        if(eos>=0) draft[eos]=1;
        setup(draft,n,seed,reject);
        tp_leader=(seed==1 && (reject%2)==0);
        int bounded=eos<0 ? n : eos+(ignore ? 0 : 1);
        int expected=(reject>=0 && reject<bounded) ? reject : bounded;
        int width=(bounded==0 || reject==0) ? 0 : bounded;
        label="EOS positions, seed/ordinary, ignore, partial and TP";
        int previous_failures=failures;
        returned=invoke(18,18,ignore!=0);
        check_frontier(returned,expected,width,draft,18);
        if(expected==width && width) CHECK(selected_row==expected-1);
        if(eos<0 || ignore) unaffected_failures+=failures-previous_failures;
    }
    for(int seed=0;seed<=1;seed++) for(int first=0;first<15;first++) {
        int draft[16]; for(int i=0;i<16;i++) draft[i]=3+(i%12);
        draft[first]=1; draft[first+1]=1;
        setup(draft,16,seed,-1); label="multiple EOS";
        returned=invoke(18,18,false); check_frontier(returned,first+1,first+1,draft,18);
    }
    const int bounded_draft[]={3,4,1,6,7,8};
    for(int seed=0;seed<=1;seed++) for(int slots=0;slots<=6;slots++) for(int limit=0;limit<3;limit++) {
        setup(bounded_draft,6,seed,-1);
        int max=8,cap=8;
        if(limit==0) max=already+slots;
        if(limit==1) cap=already+slots;
        if(limit==2) session.ctx_size=start+slots+1;
        int expected=slots<3 ? slots : 3;
        label="generation, output capacity and context room budgets";
        returned=invoke(max,cap,false); check_frontier(returned,expected,expected,bounded_draft,cap);
    }
    const int invalid_after_eos[]={3,1,99};
    setup(invalid_after_eos,3,0,-1); label="invalid suffix still rejected before EOS clipping";
    returned=invoke(8,8,false); check_frontier(returned,0,0,invalid_after_eos,8);
    CHECK(session.dspark_stats.invalid_draft==1);
    setup(invalid_after_eos,3,0,-1); label="out of budget invalid token remains uninspected";
    returned=invoke(2,8,false); check_frontier(returned,2,2,invalid_after_eos,8);

    const int stop_draft[]={3,7,4};
    setup(stop_draft,3,0,-1); extra_stop=7; label="ignore EOS still excludes other think-mode stop tokens";
    returned=invoke(8,8,true); check_frontier(returned,1,1,stop_draft,8);
    setup(stop_draft,3,0,-1); extra_stop=7; label="normal EOS handling does not truncate other tokens";
    returned=invoke(8,8,false); check_frontier(returned,3,3,stop_draft,8);

    setup(bounded_draft,6,0,1); fail_prefix=true; label="partial commit failure restores and replays bounded prefix";
    returned=invoke(8,8,false); check_frontier(returned,1,3,bounded_draft,8); CHECK(replayed==1);
    setup(bounded_draft,6,0,-1); fail_read=true; label="full logits read failure restores and replays through EOS";
    returned=invoke(8,8,false); check_frontier(returned,3,3,bounded_draft,8); CHECK(replayed==3);
    setup(bounded_draft,6,0,-1); fail_verify=true; label="verifier failure restores original state";
    returned=invoke(8,8,false); check_frontier(returned,0,3,bounded_draft,8);
    const int extended[]={3,4,5,6,7,8};
    setup(extended,6,0,5); label="four-prefix plus ordinary fifth row remains aligned";
    returned=invoke(8,8,false); check_frontier(returned,5,6,extended,8); CHECK(prefix_committed==4 && replayed==1);

    printf("{\"cases\":%d,\"assertions\":%d,\"failures\":%d,\"no_eos_or_ignore_eos_failures\":%d,\"observed_old_advance3_return2\":%s}\n",
        cases,assertions,failures,unaffected_failures,observed_negative?"true":"false");
    return failures ? 1 : 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'ds4.c')
    parser.add_argument('--output', type=Path, help='exclusive-new report directory with emitted C, build and run evidence')
    args = parser.parse_args()
    raw = args.source.read_bytes()
    function = extract(raw.decode('utf-8'))
    if not re.search(r'^#define DS4_DSPARK_MAX_BLOCK_SIZE 16$', raw.decode('utf-8'), re.M):
        raise ValueError('production block capacity changed; update this checked host fixture explicitly')
    fields = sorted(set(re.findall(r's->dspark_stats\.([a-z_]+)', function)))
    declarations = '\n'.join(('uint64_t ' + f + '[17];') if f.endswith('_hist') else
                             ('double ' if f.endswith('_ms') else 'uint64_t ') + f + ';' for f in fields)
    code = PRELUDE.replace('STATS_FIELDS', declarations) + '\n' + function + '\n' + TESTS
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    if not compiler:
        raise ValueError('empty CC')
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='ds4-eos-contract-') as temporary:
        root = args.output or Path(temporary)
        c, binary = root / 'contract.c', root / 'ds4-host-eos-contract'
        c.write_text(code)
        command = compiler + ['-std=c11', '-O0', '-fno-fast-math', '-Wall', '-Wextra', '-Werror', str(c), '-lm', '-o', str(binary)]
        build = subprocess.run(command, capture_output=True, text=True, timeout=60)
        report = {'scope': 'actual extracted greedy verifier function; checked host target/TP mocks, no model or GPU arithmetic',
                  'tested_draft_widths': [1, 16], 'production_max_block_size': 16,
                  'coverage_limits': 'Public caller selection, proposal kernels, GPU arithmetic, physical KV state, actual TP worker execution and stochastic helper execution are not mocked as proven.',
                  'source': str(args.source.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                  'helper_sha256': hashlib.sha256(function.encode()).hexdigest(),
                  'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'emitted_c_sha256': hashlib.sha256(code.encode()).hexdigest(),
                  'build_command': command, 'build_returncode': build.returncode,
                  'build_stdout': build.stdout, 'build_stderr': build.stderr,
                  'compiler_version': subprocess.run(compiler + ['--version'], capture_output=True, text=True, timeout=10).stdout,
                  'returncode': build.returncode}
        if build.returncode == 0:
            # Diagnostic environment changes must not affect this deterministic host fixture.
            env = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
            run = subprocess.run([str(binary.resolve())], capture_output=True, text=True, env=env, timeout=30)
            report.update(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
                          binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                          results=[json.loads(line) for line in run.stdout.splitlines()])
        if args.output:
            (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return report['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
