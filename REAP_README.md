# REAP-Compact GGUF Support for DS4

## Overview

This branch adds **REAP-compact GGUF support** to the DS4 inference engine, enabling DeepSeek V4 Flash models with 25% expert pruning (256→192 experts) to run on 96GB RAM machines with ~17GB memory savings.

## What is REAP?

**REAP** (Router-weighted Expert Activation Pruning) is a technique from Cerebras Research that removes low-utility experts from Mixture-of-Experts (MoE) models while maintaining quality. REAP25 prunes 25% of experts, reducing memory requirements significantly.

### Key Papers

- **Cerebras Research**: [REAP: Router-based Expert Activation Pruning](https://www.cerebras.net/blog/reap-router-based-expert-activation-pruning)
- **GitHub**: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)

## REAP25-LCB50 Model

The **DeepSeek-V4-Flash-REAP25-LCB50-DS4** model applies REAP pruning with 50-sample LiveCodeBench calibration:

- **Source**: [eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4 on HuggingFace](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4)
- **Calibration**: 50 samples from LiveCodeBench (balanced by difficulty)
- **Pruning**: 25% of experts removed (256 → 192 per layer)
- **Layers 0-2**: 256 experts (hash-preserved, unchanged)
- **Layers 3-42**: 192 experts (REAP-pruned)
- **Memory savings**: ~17GB at ctx=512

### Memory Comparison

| Configuration | Source Model | REAP25 Model | Savings |
|---------------|--------------|--------------|---------|
| Mapped @ ctx=512 | 82,697 MiB | 65,397 MiB | ~17,300 MiB (~16.9 GB) |
| File size | 80.76 GB | 63.87 GB | ~16.9 GB |

## Running REAP-Compact Models

### Quick Start

```bash
cd /path/to/ds4
./ds4 -m DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf \
       --ctx 32768 --nothink --temp 0 -n 64 \
       -p 'write a python function to merge two sorted lists'
```

### Server Mode

```bash
./ds4-server \
  -m DeepSeek-V4-Flash-REAP25-LCB50-DS4-compact-IQ2XXS.gguf \
  --ctx 32768 --tokens 1024 \
  --host 127.0.0.1 --port 8000
```

## Implementation

This branch implements REAP-compact support with the following changes:

### Core Changes (ds4.c)

1. **Per-layer expert count tracking** - `g_reap_layer_expert_count[DS4_MAX_LAYER]` array
2. **REAP metadata reading** - `reap_read_metadata()` detects REAP and infers per-layer counts from tensor dimensions
3. **Updated validation** - Uses `reap_layer_expert_count(il)` for tensor validation
4. **Updated routing** - Routing functions use per-layer expert counts

### Key Insight

The GGUF `reap.layer.expert_count` metadata contains the **original** expert count (256), not the actual compacted count. The implementation infers the actual per-layer expert counts by examining tensor dimensions directly.

### Backward Compatibility

✅ **Fully backward compatible** - Stock DeepSeek V4 Flash models (256 experts per layer) work unchanged.

## Acknowledgments

### Original Projects

- **DS4**: [github.com/antirez/ds4](https://github.com/antirez/ds4) by antirez - The original DeepSeek V4 Flash inference engine
- **REAP Research**: [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) - Router-weighted Expert Activation Pruning methodology
- **llama.cpp**: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF format, quantization, and kernel implementations
- **DeepSeek V4 Flash**: [DeepSeek-AI/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) - Base model

### REAP-Compact Model

- **REAP25-LCB50 DS4**: [eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4](https://huggingface.co/eouya2/DeepSeek-V4-Flash-REAP25-LCB50-DS4) on HuggingFace
  - REAP-pruned GGUF with DS4-compact layout
  - Includes bundled DS4 runtime (pre-compiled binaries)
  - 50-sample LiveCodeBench calibration

### Implementation Contributors

This REAP-compact support implementation was developed with assistance from:

- **Anthropic Claude** (claude.ai/code) - AI assistant for code development
- **Z.ai GLM** - AI assistant for code development

The implementation focused on:
1. Reverse-engineering the REAP-compact GGUF format
2. Adapting the DS4 engine to support per-layer expert counts
3. Maintaining backward compatibility with stock models
4. Testing on 96GB RAM M2 Max MacBook Pro

## Performance Notes

### Test System
- **Hardware**: 2023 MacBook Pro M2 Max, 96GB RAM
- **OS**: macOS
- **Backend**: Metal (GPU graph)

### Benchmarks (Preliminary)

| Metric | Stock DS4 | REAP25 DS4 | Notes |
|--------|-----------|------------|-------|
| Load time | Similar | Similar | GGUF is smaller |
| Prefill (short) | ~32 t/s | ~32 t/s | No performance loss |
| Generation | ~11.7k t/s | ~11.7k t/s | No performance loss |
| Memory @ ctx=512 | ~82GB | ~65GB | ~17GB savings |
| Memory @ ctx=32k | ~106GB | ~88GB | ~18GB savings |

## Future Work

- [ ] Extended quality testing (LongCodeBench, HumanEval, etc.)
- [ ] Long-context testing (32K, 128K token contexts)
- [ ] Tool calling verification
- [ ] Thinking mode verification
- [ ] Support for other REAP variants (REAP50, custom calibrations)

## Branch Information

- **Branch**: `reap-compact-support`
- **Based on**: `upstream/main` (github.com/antirez/ds4)
- **Fork**: github.com/ljubomirj/l26f
- **Commit**: REAP-compact GGUF support implementation

## License

This implementation maintains compatibility with the original DS4 license and acknowledgments. The REAP research is published as open source by Cerebras.
