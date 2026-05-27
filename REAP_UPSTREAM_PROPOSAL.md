# REAP-Compact GGUF Support: Upstream Proposal

## Summary

Proposal to add REAP-compact GGUF support to mainline ds4 (github.com/antirez/ds4).

## What is REAP?

REAP (Router-weighted Expert Activation Pruning) is a published technique from Cerebras that prunes low-utility experts from MoE models. REAP25 removes 25% of experts (256→192) while maintaining quality.

## Changes Required

The implementation is minimal and backward-compatible:

### Core Changes (ds4.c)

1. **Add per-layer expert count tracking** (~5 lines)
   - Global array: `static uint32_t g_reap_layer_expert_count[DS4_N_LAYER];`
   - Helper: `static uint32_t reap_layer_expert_count(uint32_t il);`

2. **Add REAP metadata reading** (~70 lines)
   - Function: `reap_read_metadata(const ds4_model *m)`
   - Detects `reap.enabled=true`
   - Infers per-layer expert counts from tensor dimensions
   - Called during engine initialization

3. **Update validation** (~10 lines)
   - Replace `DS4_N_EXPERT` with `reap_layer_expert_count(il)` in validation

4. **Update routing functions** (~50 lines)
   - Add `uint32_t il` parameter to 4 routing functions
   - Update ~10 call sites

### Makefile

Add `DS4_EXPERT_COUNT` variable (optional, defaults to 256):
```makefile
DS4_EXPERT_COUNT ?= 256
CFLAGS += -DDS4_EXPERT_COUNT=$(DS4_EXPERT_COUNT)
```

## Testing

Model tested: `DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf`
- Source: https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4
- 63.87 GiB (vs 80.76 GiB stock) - ~17GB savings
- **Status**: ✅ Loads, validates, runs successfully

## Backward Compatibility

- ✅ Stock models (256 experts) work unchanged
- ✅ No API changes
- ✅ No performance impact on non-REAP models

## Benefits

1. **Reduced memory usage**: ~17GB savings at ctx=512
2. **Enables larger contexts**: 96GB machines can run longer contexts
3. **Future-proof**: Architecture supports other REAP variants (REAP50, etc.)

## Known Limitations

1. **router_masked metadata**: Not currently read; derived from hash_preserved
2. **MTP support**: Uses layer 0 expert count (likely correct but untested)
3. **CPU backend**: Untested (Metal-only on macOS)

## Next Steps for Upstream

1. Create clean feature branch from upstream/main
2. Port only ds4.c and Makefile changes
3. Add test case with REAP-compact model
4. Submit PR with:
   - Description of REAP compaction
   - Link to Cerebras REAP paper
   - Performance/memory numbers
   - Test results

## Alternative: Draft PR

If you'd like, I can create a draft PR against the upstream repository with these changes isolated as a clean patch.
