from pathlib import Path
import subprocess, sys

tree=Path(sys.argv[1]).resolve()
source=(tree/'ds4_bench.c').read_text()
helper=r'''
/* Campaign-only capture wrapper. All inference uses the public session API.
 * Prefix snapshots and logits are written outside the throughput baseline.
 * A loaded prefix is never reported as measured prefill work. */
static int campaign_load(ds4_session *session, int frontier, char *err, size_t errlen) {
    const char *dir=getenv("DS4_BENCH_LOAD_PREFIX");
    if (!dir) return 1;
    char path[4096];
    snprintf(path,sizeof(path),"%s/%d.bin",dir,frontier);
    FILE *f=fopen(path,"rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f,0,SEEK_END)) { fclose(f); return -1; }
    long bytes=ftell(f);
    if (bytes<=0 || fseek(f,0,SEEK_SET)) { fclose(f); return -1; }
    ds4_session_snapshot snap={0};
    snap.ptr=malloc((size_t)bytes); snap.len=snap.cap=(uint64_t)bytes;
    if (!snap.ptr || fread(snap.ptr,1,(size_t)bytes,f)!=(size_t)bytes) {
        free(snap.ptr); fclose(f); return -1;
    }
    fclose(f);
    int rc=ds4_session_load_snapshot(session,&snap,err,errlen);
    ds4_session_snapshot_free(&snap);
    if (rc) return -1;
    fprintf(stderr,"campaign: loaded prefix %d (not prefill timing)\n",frontier);
    return 0;
}
static int campaign_save(ds4_session *session, int frontier, char *err, size_t errlen) {
    const char *dir=getenv("DS4_BENCH_SAVE_PREFIX");
    if (!dir) return 0;
    ds4_session_snapshot snap={0};
    if (ds4_session_save_snapshot(session,&snap,err,errlen)) return -1;
    char path[4096];snprintf(path,sizeof(path),"%s/%d.bin",dir,frontier);
    FILE *f=fopen(path,"wb");
    int rc=(!f || fwrite(snap.ptr,1,snap.len,f)!=snap.len) ? -1:0;
    if (f && fclose(f)) rc=-1;
    fprintf(stderr,"campaign: saved prefix %d (%llu bytes)\n",frontier,(unsigned long long)snap.len);
    ds4_session_snapshot_free(&snap);
    return rc;
}
static int campaign_trace(ds4_engine *engine, ds4_session *session) {
    const char *path=getenv("DS4_BENCH_TRACE_LOGITS");
    if (!path) return 0;
    static FILE *f;
    static float *logits;
    const int vocab=ds4_engine_vocab_size(engine);
    if (!f) { f=fopen(path,"wb"); logits=malloc((size_t)vocab*sizeof(float)); }
    if (!f || !logits || ds4_session_copy_logits(session,logits,vocab)!=vocab) return -1;
    uint32_t header[2]={(uint32_t)ds4_session_pos(session),(uint32_t)vocab};
    if (fwrite(header,sizeof(header),1,f)!=1 || fwrite(logits,sizeof(float),vocab,f)!=(size_t)vocab || fflush(f)) return -1;
    return 0;
}
'''
pos=source.index('static double bench_now_sec(')
source=source[:pos]+helper+'\n'+source[pos:]
old='''        const int prefill_rc =
            ds4_session_sync(session, &prefix, err, sizeof(err));'''
new='''        const int loaded = campaign_load(session, frontier, err, sizeof(err));
        const int prefill_rc = loaded < 0 ? -1 : loaded == 0 ? 0 :
            ds4_session_sync(session, &prefix, err, sizeof(err));'''
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('const double prefill_sec = prefill_t1 - prefill_t0;', 'const double prefill_sec = loaded == 0 ? 0.0 : prefill_t1 - prefill_t0;')
old='''        const bool need_restore_after_generation ='''
new='''        if (campaign_save(session, frontier, err, sizeof(err)) || campaign_trace(engine, session)) {
            fprintf(stderr,"campaign: prefix capture failed: %s\\n",err); rc=1; break;
        }
        const bool need_restore_after_generation ='''
source=source.replace(old,new,1)
old='''            const double token_t1 = bench_now_sec();'''
new='''            if (campaign_trace(engine, session)) { rc=1; break; }
            const double token_t1 = bench_now_sec();'''
assert source.count(old)==1
source=source.replace(old,new)
capture=tree/'campaign_bench.c';capture.write_text(source)
objects=['ds4_help.o','ds4_gpu_args.o','ds4.o','ds4_image.o','ds4_distributed.o','ds4_tp.o','ds4_ssd.o','ds4_metal.o','ds4_layer_pack.o']
subprocess.run(['cc','-O3','-ffast-math','-g','-mcpu=native','-Wall','-Wextra','-std=c99','-I.',str(capture),*objects,'-o','campaign-bench','-lm','-pthread','-framework','Foundation','-framework','Metal'],cwd=tree,check=True)
print('built capture wrapper',tree)
