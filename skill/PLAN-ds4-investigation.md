# DS4 DSpark Exactness Investigation — Global Plan

## Commit SHA

```text
b0309611041655f4e45671cfd9c986aff161406 (upstream main)
```

所有后续计划只记录 commit SHA + 符号名，不记录硬编码行号。通过 `grep -n 'symbol' ds4.c` 动态定位。

## 目标

在 DSpark support GGUF（~5.6 GiB）就绪后，完成 DSpark batched verifier numerical drift 的完整调查与修复流程：
1. 找到第一个分歧点（first divergence）
2. 设计并实现 exact-row Metal kernel
3. 验证修复效果并通过评测

## 阶段总览

P0 — Skill / handoff bundle（已完成）
P1 — Static code audit（通用 verifier vs exact N=2）
P2 — Runtime first-divergence probe
P3 — Minimal exact fix
P4 — Correctness + performance evaluation

---

## P0: Skill / Handoff Bundle（✅ 完成）

将 handoff bundle（10 个文件）转化为可检索的 skill 文档，附录所有原始文件原文。

输出：
- SKILL-ds4-dspark-exactness.md — Master skill 文档
- MEMORY.md 索引行已添加

---

## P1: Static Code Audit

**目标**: 完成 ds4.c 中通用 verifier vs exact N=2 verifier 的逐阶段对比审计。不依赖 GGUF，纯静态分析。

### 步骤（按 TASK_BEFORE_GGUF.md 的 7 项）

**Step 1: Symbol Location**
```bash
grep -n 'symbol_name' ds4.c   # 对以下每个符号运行：
ds4_session_eval_dspark_speculative_argmax
metal_graph_verify_suffix_tops_impl / metal_graph_verify_suffix_tops
metal_graph_verify_decode2_exact
metal_graph_eval_token_raw_swa
spec_frontier_snapshot / spec_frontier_restore / spec_frontier_commit_prefix1
ds4_gpu_matmul_f16_pair_compressor_store_tensor / ds4_gpu_matmul_f16_pair_tensor
ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor / ds4_gpu_matmul_q8_0_decode_rows_exact_tensor
ds4_gpu_matmul_f16_router_rows_exact_tensor
metal_graph_encode_shared_rows_exact
```

**Step 2: Exact N=2 Verifier Deep Read**
- 定位 `metal_graph_verify_decode2_exact()`（commit b030961）
- 逐行阅读，标注每操作是 rowwise 还是 batched
- 分析 row 0 → row 1 之间 mutable state advance 了什么
- **交付物**: one-page note

**Step 3: Stage Comparison Table**
- Trace `metal_graph_verify_suffix_tops_impl()` → `metal_graph_encode_layer_batch()` 对比 exact N=2 verifier
- **交付物**: 填写完整的 stage comparison table（13 行）
- **注意**: 任何算子不能因为"理论上 stateless"就消失在第一分歧审计中

**Step 4: Existing Exact Row Primitives Audit**
审查 `*_decode_rows_exact_tensor` 系列 Metal 实现：对每个 row launch one kernel / single dispatch + row dimension / explicit reduction order / alias batch kernel？
- **交付物**: 各 exact primitive 的实现方式总结

**Step 5: Canonical Compressor Path Location**
定位 `ds4_gpu_matmul_f16_pair_compressor_store_tensor()`（canonical）和 `ds4_gpu_matmul_f16_pair_tensor()`（batch counterpart），分析两个调用位置和差异。
- **交付物**: compressor path mapping

**Step 6: API Proposal（仅设计不实现）**
- 写 proposed exact compressor rows API signature
- 精确标识 input/output state/tensor
- **交付物**: API proposal 文档

**Step 7: Probe Integration**
- `ds4_exactness_probe.[ch]` 集成到 ds4 build
- 注意：`ds4_gpu_tensor_read()` 在 ds4_gpu.h:55 有定义，不需要 adapter
- **交付物**: 编译通过 + behavior-off by default

### P1 Gate 条件（满足全部才能进入 P2）
- [x] 13 行 stage table 填写完整
- [x] one-page exact N=2 verifier note
- [x] existing exact primitives audit summary
- [x] proposed probe insertion points
- [x] proposed exact compressor rows API signature
- [x] probe helper 编译通过 + CPU comparator 单元测试通过 (39/39 pass)
- [x] source commit SHA（b030961）记录在案

---

## P2: Runtime First-Divergence Probe

**前提条件**: P1 gate 全部满足 + DSpark support GGUF (~5.6 GiB) 就绪。

**目标**: 从同一个 pre-spec speculation snapshot 对比 A（batched verifier）vs B（ordinary sequential decode），找到第一个分歧点。核心问题：**哪个 operator 第一个偏离 canonical trajectory?**

### 步骤

1. Probe insertion：在以下每个 checkpoint 位置插入 `ds4_probe_compare_f32_arrays`
   - Compressor projected KV → score → mutable KV frontier → mutable score frontier → indexer compressed rows → emitted compressed row → layer output hidden state
2. 从同一 snapshot 分别跑 A（通用 verifier）和 B（ordinary decode）
3. 分析 probe 输出，找到第一个 `exact=0` 的 checkpoint

**交付物**: first divergence tensor/operator + layer/row 信息

---

## P3: Minimal Exact Fix

**前提条件**: P2 确认 compressor pair projection/frontier 为第一分歧点。否则调整修复策略。

**目标**: 实现 exact-row Metal kernel，使得 batched verifier 的 compressor 操作与 ordinary single-token decode bit-for-bit 一致。

### 设计原则（NOT design principles — hypotheses from P2 audit）

- Potentially batch-safe: operations proven bit-identical by audit
- Must remain canonical: any operation whose row-wise/batched impl alters committed or downstream-visible state
- Do NOT classify HC / router / MoE / output / compressor before audit complete

### Exact-Row Kernel Contract（严格）

```text
candidate(rows=N) == N repeated canonical single-row calls
```

**byte-for-byte for N=1..5**。不接受 tolerance-based equality。

---

## P4: Correctness + Performance Evaluation

**前提条件**: P3 kernel 通过 exact-row contract 测试。

### A. Correctness Test
- Bit-exact match with plain decode on regression tests
- `make dspark-acceptance` / `make dspark-verify-depth`（来自 QA_BEFORE_RELEASES.md）
- `./ds4_test --metal-kernels` + `--logprob-vectors`

### B. Performance Test
- Plain baseline vs DSpark with exact-row verifier
- Metrics: tok/s, acceptance rate, replay time, net speedup
- `./ds4-bench` + `speed-bench/plot_speed.py`

### C. Success Criteria
- Correctness: bit-exact match with plain decode
- Performance: net speedup > 0% (replay cost < speculative saving)
- Regression: no regression on `make test`
- Exact-row contract: byte-for-byte for N=1..5

---

## Risk & Mitigation

第一分歧点不是 compressor pair projection（Medium）— P2 probe 在所有 checkpoint 都会覆盖
Exact-row kernel too slow（High）— 只 exactify minimum necessary operators
GGUF 未就绪/模型不符 — Probe 设计是 model-independent
Metal encoder overhead from row splitting（Medium）— 使用 inline compute copies (PR #590)

---

## Baseline Performance (P0/P1 context)

M4 Max Mac Studio, 128GB RAM, Metal API:
```
prefill:   41.71 t/s
generation: 31.96 t/s (n=256 tokens)
nspec=128 max_chunk=6 draft_depth=5 worst_argmax_gap=0.000 at=-1
memory: KV 0.61 GiB + buffers 0.25 GiB + resident model 90.88 GiB = 91.74 GiB planned

drift-patch flags: hc_stable=on norm_unify=on kv_raw_f32=off rope_exp2_log2=off math_safe=off tensor_matmul=off
```

## File Index

### Skill & Planning
- `SKILL-ds4-dspark-exactness.md` — Master skill document
- `PLAN-ds4-investigation.md` — This file
- `PROGRESS-P0.md` — P0 action log (ds4/ root)

### Handoff Bundle (original files archived in SKILL doc appendix)
全部 10 个文件原文已附录在 SKILL doc 中。

### ds4 Repository
- `/Users/sijiaguo/ds4/ds4.c` — Primary source file（commit SHA: b030961）
- `/Users/sijiaguo/ds4/ds4_gpu.h` — GPU API definitions（ds4_gpu_tensor_read at line 55）
- `/Users/sijiaguo/ds4/Makefile` — Build system
