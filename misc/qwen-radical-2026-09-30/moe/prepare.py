from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
q=(root/'metal/qwen4.metal').read_text()
(out/'baseline.metal').write_text(q)
# Reuse only the frozen shared quantization helpers needed by the standalone compiler.
(out/'baseline-moe.metal').write_bytes((root/'misc/qwen-gain5-2026-09-30/prefill/baseline-moe.metal').read_bytes())
old='''    const uint2 block = qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = qwen4_moe_mm_expert(args, tgpig.y);'''
new='''    // Experimental schedule only. The appended descriptor is (tile << 9)|expert.
    // It names one existing tile; no dot product or epilogue changes.
    const bool compact = args.expert_major == 2u;
    const uint desc = compact ? (uint)lists[(uint64_t)args.n_expert * args.list_cap + tgpig.y] : 0u;
    const uint2 block = compact ? uint2(tgpig.x, desc >> 9u) : qwen4_moe_mm_block(args, tgpig);
    const uint rb = block.x, e = compact ? (desc & 511u) : qwen4_moe_mm_expert(args, tgpig.y);'''
assert q.count(old)==2,q.count(old)
(out/'compact.metal').write_text(q.replace(old,new))
s=(root/'misc/qwen-gain5-2026-09-30/prefill/barrier-oracle.m').read_text()
s=s.replace('#include <string.h>','#include <string.h>\n#include <time.h>')
s=s.replace('static NSString *routing_path;','''static NSString *routing_path;
static NSString *schedule_mode = @"compact";
static bool encoding_candidate;
static double worklist_ms;
static uint32_t compact_work_count;''')
# Production's active array is ascending expert ID, not the previous oracle's
# deliberately reversed ID order used only as a remapping correctness stress.
s=s.replace('''    /* Active-only dispatch, descending IDs to exercise remapping in both arms. */
    m.n_active_expert=d.n_active_expert=0;
    for(unsigned ei=experts;ei>0;ei--)if(counts[ei-1]) {
        m.active_expert[m.n_active_expert++]=ei-1;
        d.active_expert[d.n_active_expert++]=ei-1;
    }''','''    /* Match production SSD: ascending original expert IDs. */
    m.n_active_expert=d.n_active_expert=0;
    for(unsigned e=0;e<experts;e++)if(counts[e]) {
        m.active_expert[m.n_active_expert++]=e;
        d.active_expert[d.n_active_expert++]=e;
    }''')
s=s.replace('f.lists = buffer(device, (size_t)experts * tokens * sizeof(int32_t));', 'f.lists = buffer(device, ((size_t)experts * tokens + (size_t)experts * ((tokens+31u)/32u)) * sizeof(int32_t));')
anchor='''    f.midArgs=m;f.downArgs=d;
    f.x = buffer'''
replacement='''    f.midArgs=m;f.downArgs=d;
    // Frequencies are already present on CPU in the SSD path. Include measured
    // schedule construction cost in the paired screen instead of hiding it.
    struct timespec wl_start,wl_end;clock_gettime(CLOCK_MONOTONIC,&wl_start);
    uint32_t order[512];unsigned ne=0;
    for(unsigned e=0;e<experts;e++)if(counts[e])order[ne++]=e;
    if([schedule_mode isEqualToString:@"compact-heavy"]) {
        for(unsigned i=1;i<ne;i++) { uint32_t e=order[i];unsigned j=i;
            while(j && counts[order[j-1]]<counts[e]) {order[j]=order[j-1];j--;}
            order[j]=e;
        }
    }
    uint32_t *work=(uint32_t *)(lists+(size_t)experts*tokens);compact_work_count=0;
    for(unsigned i=0;i<ne;i++)for(unsigned tile=0;tile<((unsigned)counts[order[i]]+31u)/32u;tile++)
        work[compact_work_count++]=(tile<<9u)|order[i];
    clock_gettime(CLOCK_MONOTONIC,&wl_end);
    worklist_ms=(wl_end.tv_sec-wl_start.tv_sec)*1000.+(wl_end.tv_nsec-wl_start.tv_nsec)*1e-6;
    f.x = buffer'''
assert anchor in s
s=s.replace(anchor,replacement)
# All runtimes require only existing buffers; compact descriptors are an appended
# read-only tail of the lists allocation, with unchanged public row layout.
oldenc='''    [en setComputePipelineState:p]; [en setBytes:&args length:sizeof(args) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; i++) bind(en, buffers[i], (unsigned)i + 1);'''
newenc='''    if(encoding_candidate && [schedule_mode isEqualToString:@"major"]) args.expert_major=1u;
    const bool compact=encoding_candidate && [schedule_mode hasPrefix:@"compact"];
    if(compact) {args.expert_major=2u;args.tiles_per_launch=args.n_tokens;args.tail_base=0u;}
    [en setComputePipelineState:p]; [en setBytes:&args length:sizeof(args) atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; i++) bind(en, buffers[i], (unsigned)i + 1);'''
assert oldenc in s
s=s.replace(oldenc,newenc)
s=s.replace('''    [en dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake([candidate_mid_pipelines containsObject:p]?candidate_mid_threads:128,1,1)];''','''    if(compact) grid=MTLSizeMake(rb,compact_work_count,1);
    if(!grid.width || !grid.height || !grid.depth) return;
    [en dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake([candidate_mid_pipelines containsObject:p]?candidate_mid_threads:128,1,1)];''')
s=s.replace('''    id<MTLBuffer> mid = reference ? f.refMid : f.gotMid, part = reference ? f.refPart : f.gotPart;''','''    encoding_candidate=!reference;
    id<MTLBuffer> mid = reference ? f.refMid : f.gotMid, part = reference ? f.refPart : f.gotPart;''')
s=s.replace('''    return elapsed;
}''','''    return elapsed + (!reference && [schedule_mode hasPrefix:@"compact"] && !timing_stage ? worklist_ms : 0.);
}''')
s=s.replace('''        else if(!strcmp(argv[i],"--stages"))stages=true;''','''        else if(!strcmp(argv[i],"--stages"))stages=true;
        else if(!strcmp(argv[i],"--schedule")&&i+1<argc)schedule_mode=[NSString stringWithUTF8String:argv[++i]];''')
s=s.replace('''    if(!basePath||!candidatePath||reps<3''','''    if(![@[@"baseline",@"major",@"compact",@"compact-heavy"] containsObject:schedule_mode])fail(@"invalid schedule");
    if(!basePath||!candidatePath||reps<3''')
s=s.replace('''@"routing_file":routing_path?:@"synthetic",''','''@"schedule":schedule_mode,@"worklist_cpu_ms":@(worklist_ms),@"compact_work_count":@(compact_work_count),@"active_order":@"ascending expert IDs, matching production SSD",@"routing_file":routing_path?:@"synthetic",''')
# Oracle still stresses both expert-major and standard reference, while timed
# replay uses false = production M1 default standard launch.
(out/'schedule-oracle.m').write_text(s)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'baseline.metal',out/'compact.metal',out/'baseline-moe.metal',out/'schedule-oracle.m']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
