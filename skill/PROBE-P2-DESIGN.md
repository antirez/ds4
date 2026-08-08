# P2: Runtime First-Divergence Probe — Integration Design

## Goal

Find the **first layer + stage** where the batched verifier (generic) first produces a non-canonical result compared to ordinary single-token decode. All comparisons from the same pre-speculation snapshot.

## Architecture

The probe consists of two files:
- `ds4_exactness_probe.h/c` — Core comparison helpers (already built & tested, 39/39 CPU test)
- `probe_p2.c` — P2 checkpoint functions (new)

Probe is **disabled by default** (env var `DS4_DSPARK_EXACTNESS_PROBE=1`).
Zero-cost when disabled: only `getenv()` call per checkpoint.

## 5 Checkpoints (ordered by pipeline position)

### CP1: Q/KV Projection Output — **EARLIEST SUSPECT**
**Location in ds4.c**: Inside `metal_graph_encode_layer_batch` → `metal_graph_encode_layer_attention_batch`, after the matmul that produces Q/KV for each row.

**How to hook**: After line 29610 (`metal_graph_encode_layer_attention_batch`) returns, read back row 0 and row 1 Q/KV tensors from `g->batch_cur_hc` (offset by row dimension). Compare against per-row canonical results.

**What we compare**: 
- batched: Q/KV for each token as computed in the single batch dispatch
- canonical: run same tokens one-at-a-time through `ds4_gpu_matmul_q8_0_tensor`, read back results
- dimension: varies by model (e.g. DS4_Q_DIM, DS4_KV_DIM)

**Why this matters**: Both verifiers use different kernels for Q/KV proj. Exactifier calls the kernel once per row; generic calls it once with all rows batched. **Different reduction order in the kernel's internal tiling can cause first divergence.**

### CP2: Raw KV Store
**Location**: Inside `metal_graph_decode_kv_store` called within `metal_graph_encode_layer_attention_batch`.

**What we compare**: The KV tensor positions written by each verifier for this layer. Batched verifier writes all n_tokens positions at once; exactifier writes position 0, processes attention on it, then writes position 1.

### CP3: Compressor Projected KV + Score — **SECOND SUSPECT**
**Location**: After compressor projection in `metal_graph_encode_layer_attention_batch`, before compressor update.

**What we compare**: 
- CP3-KV: projected KV values for each token (output of matmul)
- CP3-Score: projected score values for each token

### CP4: Mutable Score Frontier (after compressor update)
**Location**: After `ds4_gpu_compressor_update_tensor` in the layer pipeline.

**What we compare**: The compressed score tensor after update. This is a critical mutable state that propagates downstream.

### CP5: Layer Output Hidden State — **LATEST SUSPECT**
**Location**: After FFN, before HC swap (between layers). In `metal_graph_encode_layer_batch` line 29623-29625, read `metal_graph_batch_cur_hc(g)` before it's swapped.

**What we compare**: Final hidden state of the layer for each row. If all previous checkpoints are exact, this is the only place drift can show up (accumulated from upstream).

## Probe Integration Steps

### Step 1: Include probe in build
```makefile
# In Makefile (Darwin):
PROBE_OBJS := ds4_exactness_probe.o probe_p2.o
CORE_OBJS += $(PROBE_OBJS)
```

### Step 2: Add probe hooks in `metal_graph_encode_layer_batch`
After the attention batch call (line ~29610):
```c
/* [PROBE] CP5 — read HC before any downstream ops */
if (ds4_exactness_probe_enabled()) {
    float *batched_hc = ...; /* from metal_graph_batch_cur_hc(g) */
    ds4_probe_cp5(batched_hc, canonical_hc, dim, il);
}
```

For the intermediate checkpoints (CP1-CP4), we need to hook INSIDE `metal_graph_encode_layer_attention_batch` where each sub-stage's output tensor is accessible. The cleanest approach: add a probe-aware wrapper around each matmul call.

### Step 3: Run comparison mode
For P2 runtime test:
```bash
DS4_DSPARK_EXACTNESS_PROBE=1 DS4_TEST_MODEL=ds4flash.gguf \
DS4_TEST_DSPARK=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
./ds4_test --dspark-probe-p2
```

## Expected Output

```
cp=1 layer=0 row=0 exact=1 mismatches=0 max_abs=0.000000 max_rel=0.000000 max_ulp=0
cp=1 layer=0 row=1 exact=1 mismatches=0 max_abs=0.000000 max_rel=0.000000 max_ulp=0
cp=3 layer=0 exact=0 mismatches=4096 max_abs=0.000244 max_rel=0.001876 max_ulp=5
ds4: *** FIRST DIVERGENCE DETECTED cp=3 layer=0 ***
...
=== DSpark Exactness P2 Probe Summary ===
Total comparisons: 288
Exact matches:     268
Divergent:         20
First divergence at CP3 layer=0
=== End P2 Summary ===
```

## Risk

- **Probe overhead**: Reading tensors from GPU on every layer adds significant latency. Mitigation: only do CPU comparison for the first speculation step, then disable.
- **Tensor lifetime**: The tensors we're reading may be reused by subsequent kernels. Need to ensure `ds4_gpu_end_commands()` is called before reads, or use event-based synchronization.

## DSpark Support GGUF Verification Commands

**GGUF location**: `/Users/sijiaguo/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` (5.6GB)

### 1. GGUF metadata check (no GPU required)
```bash
# Check GGUF header/tensor count — ds4 prints this on load:
# Expected: stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=0 metadata_errors=0
./ds4 -m gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf 2>&1 | grep 'stages='
# (will fail with "required metadata key is missing: deepseek4.block_count" because support GGUF lacks main model keys)
```

### 2. Full DSpark runtime test (GPU required)
```bash
# With ds4flash.gguf and dspark support GGUF:
DS4_TEST_MODEL="$(pwd)/ds4flash.gguf" \
DS4_TEST_DSPARK="$(pwd)/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
make dspark-verify-depth
```

### 3. Standalone support model test (GPU required, no main model)
```bash
# Test the support GGUF loads and functions correctly:
DS4_TEST_DSPARK="$(pwd)/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
./ds4_test --dspark-verify-depth
```

### 4. Direct DSpark decode (GPU required)
```bash
# Actual speculative decoding with support model:
./ds4 -m ds4flash.gguf --dspark -p "Test prompt here" -n 256
```

### Expected outputs when support GGUF is valid
- **Metadata**: `stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=0 metadata_errors=0`
- **dspark-verify-depth**: passes (no ERR output)
- **--dspark decode**: produces speculative tokens (acceptance rate > 0%)

### Notes
- Support GGUF cannot be loaded alone with `./ds4 -m` — it will fail with missing main model metadata keys (e.g., `deepseek4.block_count`). This is expected behavior.
- Requires GPU memory for any runtime test (Metal warmup needs ~9GB VRAM per the earlier observation)

## Next Step (P2 gate from P1)

All P1 gates passed → proceed with probe integration + runtime test on real DSpark model.
