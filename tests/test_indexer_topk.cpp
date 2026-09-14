// SPDX-License-Identifier: MIT
// Independent sequential merges check the extracted production Metal body.
#include "ds4_indexer_topk.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using uint = unsigned;
struct uint3 { uint x, y, z; };
struct ushort3 { unsigned short x, y, z; };
using std::min;
#include "indexer_topk_actual.h"

static unsigned cases, stages, causal_leaves;
static uint64_t geometry_checks;
static constexpr int32_t poison = -715813;
static constexpr size_t guard = 19;
static void require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "indexer top-k host FAIL: %s\n", message); std::abort(); }
}
static std::vector<int32_t> stable_merge(const std::vector<int32_t> &a,
        const std::vector<int32_t> &b, const std::vector<float> &scores, size_t k) {
    std::vector<int32_t> out;
    size_t i = 0, j = 0;
    while (out.size() < k && (i < a.size() || j < b.size())) {
        const bool left = j == b.size() ||
            (i < a.size() && scores[a[i]] >= scores[b[j]]);
        out.push_back(left ? a[i++] : b[j++]);
    }
    return out;
}
static void one(uint32_t n, uint32_t k, uint32_t threads, uint32_t rows, unsigned pattern) {
    ds4_indexer_topk_geometry geometry{};
    require(ds4_indexer_topk_initial(n, rows, k, threads, &geometry), "valid initial geometry");
    std::mt19937 random(n + 31*k + pattern);
    std::vector<std::vector<std::vector<int32_t>>> runs(rows);
    // A score-row stride includes padding; IDs remain local to each row.
    const size_t stride = n + 7;
    const unsigned values = pattern % 4;
    std::vector<float> scores(guard + stride*rows + guard, -123.5f);
    for (uint32_t row = 0; row < rows; ++row) {
        float *p = scores.data() + guard + stride*row;
        for (uint32_t i = 0; i < n; ++i) {
            if (values == 0) p[i] = static_cast<float>(i);
            if (values == 1) p[i] = static_cast<float>(random()%7);
            if (values == 2) p[i] = i%17==0 ? std::numeric_limits<float>::infinity() :
                (i < row*13 + k/2 ? float(i%17) : -std::numeric_limits<float>::infinity());
            if (values == 3) p[i] = i%2 ? 0.f : -0.f;
        }
        for (uint32_t first = 0; first < n; first += threads) {
            std::vector<int32_t> leaf(std::min(threads, n-first));
            std::iota(leaf.begin(), leaf.end(), first);
            // Tie order deliberately differs from ID order, as can the actual
            // bitonic leaf. Only score order is required at the merge boundary.
            if (pattern >= 4) leaf = actual_leaf(p,n,first,threads);
            else {
                std::shuffle(leaf.begin(), leaf.end(), random);
                std::stable_sort(leaf.begin(), leaf.end(), [p](int32_t a, int32_t b) {return p[a] > p[b];});
            }
            if (leaf.size() > k) leaf.resize(k);
            runs[row].push_back(std::move(leaf));
        }
    }
    const auto before_scores = scores;
    uint32_t expected_width = 0;
    for (const auto &leaf : runs[0]) expected_width += static_cast<uint32_t>(leaf.size());
    require(geometry.leaves == runs[0].size() && geometry.width == expected_width &&
            geometry.leaf_length == runs[0][0].size(), "independent initial counts");
    std::vector<std::vector<int32_t>> oracle(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        const std::vector<float> score_row(scores.begin()+guard+row*stride,
                                         scores.begin()+guard+row*stride+n);
        for (const auto &leaf : runs[row])
            oracle[row] = stable_merge(oracle[row], leaf, score_row, n);
        oracle[row].resize(k);
    }
    uint32_t width = geometry.width, length = geometry.leaf_length;
    unsigned stage_count = 0;
    bool useful = false;
    while (runs[0].size() > 1) {
        require(++stage_count <= 32, "tree must terminate");
        ds4_indexer_topk_stage next{};
        require(ds4_indexer_topk_next(width, length, k, &next), "valid stage geometry");
        std::vector<int32_t> input(guard + size_t(width)*rows + guard, poison);
        for (uint32_t row = 0; row < rows; ++row) {
            auto dst = input.begin() + guard + size_t(row)*width;
            for (const auto &run : runs[row]) dst = std::copy(run.begin(), run.end(), dst);
        }
        const auto before_input = input;
        std::vector<std::vector<std::vector<int32_t>>> expected(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            const std::vector<float> score_row(scores.begin()+guard+row*stride,
                                             scores.begin()+guard+row*stride+n);
            for (size_t i = 0; i < runs[row].size(); i += 2) {
                const std::vector<int32_t> empty;
                expected[row].push_back(stable_merge(runs[row][i],
                    i+1 < runs[row].size() ? runs[row][i+1] : empty, score_row, k));
            }
        }
        uint32_t next_width = 0;
        for (const auto &run : expected[0]) next_width += static_cast<uint32_t>(run.size());
        require(next.groups == expected[0].size() && next.width == next_width &&
                next.run_length == expected[0][0].size(), "independent stage counts");
        useful |= next.groups > 1 && next.width < width;
        ds4_indexer_topk_merge_args args{stride*sizeof(float), rows, width, length,
                                        next.width, next.run_length, 0};
        // Cover one thread, partial waves, and excess threads with zero work.
        for (const unsigned nthreads : {1u, 7u, 32u, 128u, 512u}) {
            std::vector<int32_t> output(guard + size_t(next.width)*rows + guard, poison);
            for (uint32_t group = next.groups*rows; group-->0;)
                for (unsigned tid = nthreads; tid-->0;)
                    actual_compact_merge(args, reinterpret_cast<const char *>(scores.data()+guard),
                        input.data()+guard, output.data()+guard, {group,0,0},
                        {static_cast<unsigned short>(tid),0,0},
                        {static_cast<unsigned short>(nthreads),1,1});
            for (size_t i = 0; i < guard; ++i)
                require(output[i] == poison && output[output.size()-1-i] == poison,
                        "actual kernel output guard");
            for (uint32_t row = 0; row < rows; ++row) {
                size_t offset = guard + size_t(row)*next.width;
                for (const auto &run : expected[row]) {
                    require(std::equal(run.begin(), run.end(), output.begin()+offset),
                            "actual kernel vs independent stable merge");
                    offset += run.size();
                }
            }
            require(input == before_input &&
                    !std::memcmp(scores.data(), before_scores.data(), scores.size()*sizeof(float)),
                    "kernel mutated IDs, scores or guards");
            ++stages;
        }
        runs = std::move(expected);
        width = next.width; length = next.run_length;
    }
    require(width == k && length == k, "final K output shape");
    for (uint32_t row = 0; row < rows; ++row)
        require(runs[row][0] == oracle[row], "hierarchical pruning preserves global stable top-k");
    require(ds4_indexer_topk_compaction_useful(geometry, k) == useful,
            "dispatch useful iff an intermediate merge prunes");
    ++cases;
}
static void bounds() {
    ds4_indexer_topk_geometry g{};
    ds4_indexer_topk_stage s{};
    require(sizeof(ds4_indexer_topk_merge_args) == 32 &&
            offsetof(ds4_indexer_topk_merge_args, pad) == 28 &&
            sizeof(ds4_metal_args_argsort_compact_merge) == sizeof(ds4_indexer_topk_merge_args) &&
            offsetof(ds4_metal_args_argsort_compact_merge, pad) == offsetof(ds4_indexer_topk_merge_args, pad),
            "actual Metal argument ABI");
    const std::array<std::array<uint32_t,4>,12> bad{{
        {{0,1,1,32}}, {{1,0,1,32}}, {{1,1,0,32}}, {{1,1,2,32}},
        {{1,1,1,0}}, {{1,1,1,3}}, {{1,1,1,2048}}, {{UINT32_MAX,1,1,32}},
        {{1,UINT32_MAX,1,32}}, {{INT32_MAX,1,INT32_MAX,1024}},
        {{65536,UINT32_MAX/2,512,1024}}, {{INT32_MAX,3,512,1024}}
    }};
    for (const auto &a : bad)
        require(!ds4_indexer_topk_initial(a[0], a[1], a[2], a[3], &g), "reject unsafe initial bounds");
    require(!ds4_indexer_topk_initial(1,1,1,1,nullptr), "null geometry");
    require(!ds4_indexer_topk_next(0,1,1,&s) && !ds4_indexer_topk_next(1,0,1,&s) &&
            !ds4_indexer_topk_next(1,1,0,&s) && !ds4_indexer_topk_next(1,2,2,&s) &&
            !ds4_indexer_topk_next(8,4,2,&s) && !ds4_indexer_topk_next(UINT32_MAX,1,1,&s) &&
            !ds4_indexer_topk_next(1,1,1,nullptr), "reject unsafe stage bounds");
    require(ds4_indexer_topk_initial(2147482624u,1,1,1024,&g), "large sparse geometry accepted");
    require(g.leaves == 2097151u && g.width == 2097151u, "large geometry exact arithmetic");
}
static void geometry_one(uint32_t columns,uint32_t rows,uint32_t k,uint32_t threads) {
    ds4_indexer_topk_geometry initial{};
    const bool valid=columns&&rows&&k&&k<=columns&&columns<=INT32_MAX&&rows<=INT32_MAX&&
        threads&&threads<=1024&&!(threads&(threads-1));
    uint64_t expected_width=0;
    if(valid)expected_width=uint64_t(columns/threads)*std::min(k,threads)+std::min(k,columns%threads);
    const bool admitted=valid&&expected_width<=uint64_t(INT32_MAX)/2&&expected_width*rows<=INT32_MAX;
    require(ds4_indexer_topk_initial(columns,rows,k,threads,&initial)==admitted,"64-bit admission oracle");
    ++geometry_checks;
    if(!admitted)return;
    require(initial.width==expected_width&&initial.leaf_length==std::min(k,threads)&&
            initial.leaves==uint64_t(columns)/threads+(columns%threads!=0),"quotient/tail initial oracle");
    uint32_t width=initial.width,length=initial.leaf_length;
    bool useful=false;
    for(uint32_t depth=0;length<width;++depth){
        require(depth<32,"geometry schedule terminates");
        ds4_indexer_topk_stage next{};
        require(ds4_indexer_topk_next(width,length,k,&next),"reachable stage accepted");
        const uint64_t complete=uint64_t(width)/(2ull*length),tail=uint64_t(width)%(2ull*length);
        const uint64_t kept=std::min<uint64_t>(k,2ull*length);
        require(next.width==complete*kept+std::min(kept,tail)&&next.run_length==kept&&
                next.groups==complete+(tail!=0),"quotient/tail stage oracle");
        require(next.width<=width&&(next.width<width||next.run_length>length),"strict schedule progress");
        useful|=next.groups>1&&next.width<width;
        width=next.width;length=next.run_length;++geometry_checks;
    }
    require(width==k&&ds4_indexer_topk_compaction_useful(initial,k)==useful,"terminal K/useful geometry");
}
static void geometry_sweep(){
    for(uint32_t n:{0u,1u,2u,3u,31u,512u,1023u,1024u,1025u,8193u,65535u,
        uint32_t(INT32_MAX/2),uint32_t(INT32_MAX/2+1),uint32_t(INT32_MAX-1),uint32_t(INT32_MAX),
        uint32_t(INT32_MAX)+1u,UINT32_MAX})
        for(uint32_t rows:{0u,1u,2u,17u,4096u,uint32_t(INT32_MAX),UINT32_MAX})
            for(uint32_t k:{0u,1u,3u,512u,777u,2048u,n,UINT32_MAX})
                for(uint32_t threads:{0u,1u,2u,3u,32u,256u,512u,1024u,1025u,UINT32_MAX})
                    geometry_one(n,rows,k,threads);
    uint32_t state=0x75a021u;
    for(uint32_t i=0;i<10000;++i){
        state=state*1664525u+1013904223u;const uint32_t n=state&INT32_MAX;
        state=state*1664525u+1013904223u;const uint32_t k=n?1u+state%n:0u;
        state=state*1664525u+1013904223u;
        geometry_one(n,1u+(state&7u),k,1u<<((state>>3)%11));
    }
}
static void causal_leaf_sweep() {
    // A causal row must have the legacy permutation of its visible prefix,
    // even when future scores would win. Shuffle changes transport only.
    for (uint32_t n : {1024u,1025u,2049u,4097u,8193u})
    for (uint32_t nth : {256u,512u,1024u})
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        std::vector<float> scores(n);
        for (uint32_t i = 0; i < n; ++i) {
            scores[i] = pattern == 0 ? float((i*73u)%19u) : pattern == 1 ? float(i) :
                pattern == 2 ? ((i & 1u) ? -0.0f : 0.0f) :
                (i%17u == 0 ? std::numeric_limits<float>::infinity() :
                             -std::numeric_limits<float>::infinity());
        }
        const auto original = scores;
        for (uint32_t ratio : {1u,2u})
        for (uint32_t initial : {1023u,1024u,2048u})
        for (uint32_t row : {0u,1u,2u,7u}) {
            const uint32_t start = initial*ratio;
            const uint32_t visible = std::min(n,initial+(row+1u)/ratio);
            for (uint32_t begin = 0; begin < n; begin += nth) {
                const auto expected = actual_leaf(scores.data(),visible,begin,nth);
                require(actual_leaf(scores.data(),n,begin,nth,false,start,ratio,row) == expected,
                        "causal leaf matches independent clipped prefix");
                require(actual_leaf(scores.data(),n,begin,nth,true,start,ratio,row) == expected,
                        "causal shuffle preserves scalar permutation");
                ++causal_leaves;
            }
        }
        require(!std::memcmp(scores.data(),original.data(),n*sizeof(float)),
                "causal leaves preserve score input");
    }
}
int main() {
    bounds();
    geometry_sweep();
    causal_leaf_sweep();
    for (uint32_t n : {1u,2u,7u,31u,33u,255u,257u,511u,513u,1025u,2049u,4097u})
        for (uint32_t threads : {1u,32u,256u,1024u})
            for (uint32_t k : {1u,6u,31u,512u,1025u,n})
                if (k <= n)
                    for (unsigned pattern = 0; pattern < 4; ++pattern)
                        one(n,k,threads,3,pattern);
    for (uint32_t n : {16385u,65536u}) one(n,512,1024,2,1);
    for(uint32_t threads:{32u,256u,1024u})
        for(uint32_t n:{threads+1,3*threads+7,9*threads+13})
            for(uint32_t k:{1u,3u,31u,127u,512u,777u,2048u,n})
                if(k<=n)for(unsigned pattern=4;pattern<8;++pattern)one(n,k,threads,3,pattern);
    std::printf("PASS: %llu geometry/stage checks, %u tree cases, %u actual-source kernel stage/thread-grid cases; actual legacy leaves and shuffled tied runs, infinities, signed zero, full-merge exact IDs, guards, immutable inputs.\n",
        static_cast<unsigned long long>(geometry_checks),cases,stages);
    std::printf("PASS: %u causal leaf cases; masked future rows, tails, scalar/shuffle exact IDs.\n",causal_leaves);
    std::puts("Host-only: Metal compilation, synchronization and GPU performance require native validation.");
}
