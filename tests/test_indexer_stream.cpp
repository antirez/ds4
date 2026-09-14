// SPDX-License-Identifier: MIT
#include "ds4_indexer_stream.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
using uint=unsigned;
struct uint3 {uint x,y,z;};
struct ushort3 {uint16_t x,y,z;};
using std::min;
// PRODUCTION_SOURCE

using pair_t=ds4_metal_indexer_stream_pair;
static constexpr size_t guard=17;
static unsigned geometry_cases,tree_cases,merge_cases;
static void check(bool ok,const char *message){
    if(!ok){std::fprintf(stderr,"Indexer stream FAIL: %s\n",message);std::exit(1);}
}
static uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
static pair_t poison(){uint32_t b=0x7fc51234;float v;std::memcpy(&v,&b,4);return {v,-771733};}
static bool same(pair_t a,pair_t b){return bits(a.score)==bits(b.score)&&a.index==b.index;}
static void guards(const std::vector<pair_t>&x){
    for(size_t i=0;i<guard;++i)check(same(x[i],poison())&&same(x[x.size()-1-i],poison()),"pair guards");
}
static std::vector<int32_t> full_merge(const std::vector<float>&scores,
        const std::vector<int32_t>&a,const std::vector<int32_t>&b){
    size_t i=0,j=0;std::vector<int32_t> out;
    while(i<a.size()||j<b.size()){
        if(j==b.size()||(i<a.size()&&scores[a[i]]>=scores[b[j]]))out.push_back(a[i++]);
        else out.push_back(b[j++]);
    }
    return out;
}
static std::vector<int32_t> reference(const std::vector<float>&scores,uint32_t nth,uint32_t k){
    std::vector<std::vector<int32_t>> runs;
    for(uint32_t base=0;base<scores.size();base+=nth){
        auto ids=actual_leaf(scores.data(),uint32_t(scores.size()),base,nth);
        ids.resize(std::min<size_t>(k,ids.size()));runs.push_back(std::move(ids));
    }
    while(runs.size()>1){
        std::vector<std::vector<int32_t>> next;
        for(size_t i=0;i<runs.size();i+=2)
            next.push_back(i+1<runs.size()?full_merge(scores,runs[i],runs[i+1]):runs[i]);
        runs=std::move(next);
    }
    runs[0].resize(k);return runs[0];
}
template<bool final> static void merge_run(const ds4_metal_args_indexer_stream_merge &a,
        const pair_t *input,char *output,uint32_t threads){
    const uint64_t groups=(uint64_t(a.input_width)+2ull*a.run_length-1)/(2ull*a.run_length);
    for(uint32_t group=uint32_t(groups)*a.n_rows;group-->0;)
        for(uint32_t lane=threads;lane-->0;)
            kernel_indexer_stream_merge<final>(a,input,output,{group,0,0},
                {uint16_t(lane),0,0},{uint16_t(threads),1,1});
    ++merge_cases;
}
static void one(uint32_t n,uint32_t k,uint32_t nth,uint32_t chunk,uint32_t pattern){
    constexpr uint32_t rows=3;
    ds4_indexer_stream_geometry geometry{};
    check(n>chunk&&ds4_indexer_stream_initial(n,rows,k,nth,chunk,&geometry),"stream shape admitted");
    const uint32_t candidate_width=geometry.candidate_width;
    std::vector<std::vector<float>> scores(rows,std::vector<float>(n));
    std::vector<std::vector<int32_t>> expected;
    for(uint32_t row=0;row<rows;++row){
        uint32_t state=17+row+n;
        for(uint32_t i=0;i<n;++i){
            state=state*1664525u+1013904223u;
            float value=float(int(state%17)-8);
            if(pattern==1)value=(i&1)?-0.f:0.f;
            if(pattern==2)value=i%13==0?INFINITY:(i%3==0?-INFINITY:value);
            if(pattern==3)value=float(i);
            if(pattern==4)value=-float(i);
            scores[row][i]=value;
        }
        expected.push_back(reference(scores[row],nth,k));
    }
    const auto scores_before=scores;
    const uint32_t threads=pattern==0?1:pattern==1?7:pattern==2?32:pattern==3?127:512;
    std::vector<pair_t> candidates(2*guard+size_t(rows)*candidate_width,poison());
    for(uint32_t c=0;c<geometry.chunks;++c){
        const uint32_t base=c*chunk,count=std::min(chunk,n-base);
        ds4_indexer_topk_geometry shape{};
        check(ds4_indexer_topk_initial(count,rows,std::min(k,count),nth,&shape),"tail geometry");
        const uint32_t leaf_length=geometry.leaf_length;
        std::vector<float> local_scores(2*guard+size_t(rows)*count,poison().score);
        for(uint32_t row=0;row<rows;++row)
            std::copy(scores[row].begin()+base,scores[row].begin()+base+count,local_scores.begin()+guard+size_t(row)*count);
        const auto local_scores_before=local_scores;
        const bool direct=shape.leaves==1;
        std::vector<pair_t> local(2*guard+size_t(rows)*shape.width,poison());
        ds4_metal_args_indexer_stream_leaf a={count,rows,direct?candidate_width:shape.width,leaf_length,base,0,0,0};
        for(uint32_t group=shape.leaves*rows;group-->0;){
            std::vector<int32_t> ids(nth,-1);std::vector<float> values(nth,poison().score);
            kernel_indexer_stream_leaf_f32_i32(a,local_scores.data()+guard,
                direct?candidates.data()+guard+size_t(c)*k:local.data()+guard,
                ids.data(),values.data(),{group,0,0},{0,0,0},{uint16_t(nth),1,1});
        }
        guards(local);guards(candidates);
        check(!std::memcmp(local_scores.data(),local_scores_before.data(),local_scores.size()*4),"leaf input and guard immutable");
        uint32_t width=shape.width,length=leaf_length;
        for(uint32_t depth=0;!direct&&length<width;++depth){
            check(depth<32,"local tree terminates");ds4_indexer_topk_stage next{};
            check(ds4_indexer_topk_next(width,length,k,&next),"local next stage");
            const bool last=next.groups==1;
            const auto before=local;
            std::vector<pair_t> output(2*guard+size_t(rows)*next.width,poison());
            ds4_metal_args_indexer_stream_merge m={0,rows,width,length,last?candidate_width:next.width,next.run_length,0};
            merge_run<false>(m,local.data()+guard,reinterpret_cast<char*>(last?
                candidates.data()+guard+size_t(c)*k:output.data()+guard),threads);
            check(!std::memcmp(local.data(),before.data(),local.size()*sizeof(pair_t)),"local merge input immutable");
            guards(output);guards(candidates);local=std::move(output);width=next.width;length=next.run_length;
        }
        // Verify every record of this completed chunk, including global row
        // stride and the per-chunk offset, before its scratch is reused.
        for(uint32_t row=0;row<rows;++row){
            std::vector<float> slice(scores[row].begin()+base,scores[row].begin()+base+count);
            const auto ids=reference(slice,nth,std::min(k,count));
            for(size_t i=0;i<ids.size();++i){
                const auto p=candidates[guard+size_t(row)*candidate_width+size_t(c)*k+i];
                check(p.index==int32_t(base)+ids[i]&&bits(p.score)==bits(slice[ids[i]]),"chunk record score bits / global ID / row stride");
            }
        }
    }
    uint32_t width=candidate_width,length=k;
    std::vector<int32_t> selected(2*guard+size_t(rows)*k,-771733);
    for(uint32_t depth=0;length<width;++depth){
        check(depth<32,"global tree terminates");ds4_indexer_topk_stage next{};
        check(ds4_indexer_topk_next(width,length,k,&next),"global next stage");
        const bool last=next.groups==1;const auto before=candidates;
        std::vector<pair_t> output(2*guard+size_t(rows)*next.width,poison());
        ds4_metal_args_indexer_stream_merge m={0,rows,width,length,next.width,next.run_length,0};
        if(last)merge_run<true>(m,candidates.data()+guard,reinterpret_cast<char*>(selected.data()+guard),threads);
        else merge_run<false>(m,candidates.data()+guard,reinterpret_cast<char*>(output.data()+guard),threads);
        check(!std::memcmp(candidates.data(),before.data(),candidates.size()*sizeof(pair_t)),"global merge input immutable");
        guards(output);candidates=std::move(output);width=next.width;length=next.run_length;
    }
    for(size_t i=0;i<guard;++i)check(selected[i]==-771733&&selected[selected.size()-1-i]==-771733,"selected guards");
    for(uint32_t row=0;row<rows;++row){
        check(!std::memcmp(selected.data()+guard+size_t(row)*k,expected[row].data(),k*4),"stream exact legacy full-tree indices");
        check(!std::memcmp(scores[row].data(),scores_before[row].data(),n*4),"full scores immutable");
    }
    ++tree_cases;
}
static void bounds(){
    static_assert(sizeof(ds4_indexer_stream_pair)==8&&sizeof(pair_t)==8,"pair ABI");
    static_assert(sizeof(ds4_indexer_stream_leaf_args)==32&&sizeof(ds4_metal_args_indexer_stream_leaf)==32,"leaf ABI");
    static_assert(sizeof(ds4_metal_args_indexer_stream_merge)==32,"merge ABI");
    for(uint32_t n:{0u,1u,31u,512u,1024u,2049u,65537u,uint32_t(INT32_MAX),UINT32_MAX})
        for(uint32_t rows:{0u,1u,3u,uint32_t(INT32_MAX),UINT32_MAX})
            for(uint32_t k:{0u,1u,31u,512u,777u,n})
                for(uint32_t nth:{0u,1u,32u,512u,1024u,1025u})
                    for(uint32_t chunk:{0u,16u,31u,32u,1024u,2048u,4096u,uint32_t(1u<<30),UINT32_MAX}){
                        ds4_indexer_stream_geometry g{};ds4_indexer_topk_geometry global{};
                        const bool initial=ds4_indexer_topk_initial(n,rows,k,nth,&global);
                        const bool valid=initial&&chunk>=32&&chunk<=uint32_t(INT32_MAX)/2&&!(chunk&(chunk-1))&&chunk>=nth&&chunk>=k;
                        check(ds4_indexer_stream_initial(n,rows,k,nth,chunk,&g)==valid,"stream geometry boundary admission");
                        if(valid){
                            const uint64_t full=n/chunk,tail=n%chunk;
                            check(g.chunks==full+(tail!=0)&&g.candidate_width==full*k+std::min<uint64_t>(tail,k)&&
                                  g.candidate_width<=global.width&&g.leaf_length==global.leaf_length,"stream count independent quotient/tail oracle");
                        }
                        ++geometry_cases;
                    }
    ds4_indexer_stream_geometry g{};
    check(!ds4_indexer_stream_initial(8192,1,512,1024,4096,nullptr),"null geometry");
    check(ds4_indexer_stream_initial(8193,3,512,1024,4096,&g)&&g.candidate_width==1025,"one-score final chunk preserved");
}
int main(){
    bounds();
    for(uint32_t nth:{512u,1024u})for(uint32_t chunk:{2048u,4096u})
        for(uint32_t n:{chunk+1,chunk+31,chunk+511,chunk+512,chunk+513,chunk+1023,chunk+1025,3*chunk+7})
            for(uint32_t pattern=0;pattern<5;++pattern)one(n,512,nth,chunk,pattern);
    for(uint32_t k:{1u,31u,777u,1536u})for(uint32_t pattern=0;pattern<5;++pattern)
        one(8193,k,1024,4096,pattern);
    std::printf("PASS: %u stream geometry checks, %u trees with 3 rows, %u pair-merge stages; actual legacy/stream leaves, exact ties/IDs, score bits, tails<K, strides and guards.\n",geometry_cases,tree_cases,merge_cases);
    std::puts("Host-only: scoring math, Metal compilation, synchronization and speed require native tests.");
}
