# Distributed Benchmark Methodology

## Overview

Measure the overhead of JACCL expert parallelism by comparing single-node and multi-node ds4-bench runs. The expected overhead per token is minimal: 16KB all_sum per layer x 43 layers = 688KB/token. At 11.7 GB/s measured RDMA bandwidth, this adds ~0.06ms per token.

## Prerequisites

- ds4 built with JACCL: `make clean && make JACCL=1`
- GGUF model accessible on all nodes (same path, lazy mmap handles partial loading)
- RDMA verified: `asmi links` shows active links between nodes
- PD health checked: no prior PD exhaustion (reboot nodes if uncertain)

## Single-Node Baseline

Run on the coordinator node (hub):

```bash
# Generation benchmark (decode)
./ds4-bench --metal --model gguf/ds4flash.gguf --ctx 32768 --batch 1 --repeat 3

# Prefill benchmark
./ds4-bench --metal --model gguf/ds4flash.gguf --ctx 32768 --batch 512 --repeat 3
```

Record: tok/s generation, tok/s prefill, peak memory.

## 2-Node Distributed

Use the launch script with ds4-bench instead of ds4-server:

```bash
# On rank 0 (hub):
export JACCL_RANK=0
export JACCL_WORLD_SIZE=2
export JACCL_COORDINATOR=$(tailscale ip -4)
export JACCL_IBV_DEVICES='[[null, "rdma_enX"], ["rdma_enY", null]]'
./ds4-bench --distributed --metal --model gguf/ds4flash.gguf --ctx 32768 --batch 1 --repeat 3

# On rank 1 (m3u4) — same env vars but JACCL_RANK=1:
export JACCL_RANK=1
export JACCL_WORLD_SIZE=2
export JACCL_COORDINATOR=<rank0_tailscale_ip>
export JACCL_IBV_DEVICES='[[null, "rdma_enX"], ["rdma_enY", null]]'
./ds4-bench --distributed --metal --model gguf/ds4flash.gguf --ctx 32768 --batch 1 --repeat 3
```

Use `asmi links` on each node to fill in the actual RDMA interface names.

Record: tok/s generation, tok/s prefill, overhead %.

## Expected Overhead Calculation

### Per-Token Communication Cost

```
all_sum payload per layer:  DS4_N_EMBD * sizeof(float) = 4096 * 4 = 16,384 bytes (16 KB)
Number of MoE layers:       43
Total per-token RDMA:       16 KB * 43 = 688 KB

Measured RDMA bandwidth:    11.7 GB/s (from JACCL 4-node baseline benchmark)
Transfer time per token:    688 KB / 11.7 GB/s = 0.057 ms

Single-node generation:     ~15 tok/s = 66.7 ms/tok
Expected overhead:          0.057 / 66.7 = 0.09%
```

### Prefill (Batch) Communication Cost

```
Batch all_sum payload:      n_tok * 16 KB * 43 layers
For n_tok=512:              512 * 688 KB = 344 MB per prefill pass
Transfer time:              344 MB / 11.7 GB/s = 29 ms
Single-node 512-tok prefill: ~200 ms (estimated)
Expected overhead:          29 / 200 = 14.5%
```

Prefill overhead is higher because the batch all_sum is proportional to token count. For short prompts (<64 tokens), overhead is negligible.

## 4-Node Distributed

Same procedure with JACCL_WORLD_SIZE=4. Expected compute savings: each node runs 64/256 experts instead of 256/256. The GPU still dispatches all 6 expert projections per token (weight masking, not early exit), so GPU time savings are limited. CPU path sees full 4x savings on expert compute.

## Recording Results

After each run, record in this table:

| Config | Nodes | Generation tok/s | Prefill tok/s | Overhead % |
|--------|-------|-------------------|---------------|------------|
| Baseline | 1 (hub) | | | - |
| 2-node | hub + m3u4 | | | |
| 4-node | hub + m3u1 + m3u3 + m3u4 | | | |

## Profiling

For per-layer timing breakdown:

```bash
DS4_DECODE_PROFILE_DETAIL=1 ./ds4 --distributed --metal -c 4096 -n 10 -p "test"
DS4_PREFILL_PROFILE_DETAIL=1 ./ds4 --distributed --metal -c 4096 -p "long prompt here..."
```

This prints per-layer timing for HC, norm, routed MoE, shared FFN, and post-processing. Compare the routed MoE time between single-node and distributed to isolate all_sum overhead from compute savings.

## Known Limitations

- **GPU waste on masked experts:** The fused Metal kernel still dispatches all 6 expert projections even when weights are zeroed. Shader-level early-exit on weight=0 is future work.
- **PD exhaustion risk:** One JACCL Group per process lifetime. Never restart the benchmark without a clean process exit. If PD exhaustion occurs, power-cycle all nodes (full shutdown, not reboot).
- **Prefill batch size:** Large batch all_sum (>100MB) may saturate RDMA bandwidth and show higher variance. Use --repeat 5 for large batch benchmarks.
