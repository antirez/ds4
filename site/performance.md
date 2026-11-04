# Performance

## Platform Overview

<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-title">M5 Max — Flash Q2 Prefill</div>
    <div class="metric-value">~200</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">2048-token chunks, 64K context</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:65%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">DGX Spark — V4.1 Prefill</div>
    <div class="metric-value">384</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">32K prefill, 64 GiB cache budget</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:100%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">DGX Spark — DSpark</div>
    <div class="metric-value">31.4</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">256-token decode, greedy, C code</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:28%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">M5 Max — Flash Q2 Decode</div>
    <div class="metric-value">18</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">Typical generation throughput</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:20%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">8× L40S — Flash Q4</div>
    <div class="metric-value">126</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">16 decode rows aggregate</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:55%"></div></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Two Sparks — V4.1</div>
    <div class="metric-value">21.9</div>
    <div class="metric-unit">t/s</div>
    <div class="metric-detail">1K prompt, 64K context, RoCE</div>
    <div class="metric-bar"><div class="metric-bar-fill" style="width:22%"></div></div>
  </div>
</div>

## Latency Percentiles

<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-title">TTFT</div>
    <div class="metric-detail">M5 Max, Flash Q2, 2K prefill</div>
    <table class="metric-table">
      <tr><td>p50</td><td class="metric-val">0.8 s</td></tr>
      <tr><td>p90</td><td class="metric-val">1.2 s</td></tr>
      <tr><td>p99</td><td class="metric-val">2.1 s</td></tr>
    </table>
  </div>
  <div class="metric-card">
    <div class="metric-title">ITL (Decode)</div>
    <div class="metric-detail">M5 Max, Flash Q2</div>
    <table class="metric-table">
      <tr><td>p50</td><td class="metric-val">52 ms</td></tr>
      <tr><td>p90</td><td class="metric-val">68 ms</td></tr>
      <tr><td>p99</td><td class="metric-val">95 ms</td></tr>
    </table>
  </div>
  <div class="metric-card">
    <div class="metric-title">ITL w/ DSpark</div>
    <div class="metric-detail">DGX Spark, Flash DSpark</div>
    <table class="metric-table">
      <tr><td>p50</td><td class="metric-val">31 ms</td></tr>
      <tr><td>p90</td><td class="metric-val">42 ms</td></tr>
      <tr><td>p99</td><td class="metric-val">64 ms</td></tr>
    </table>
  </div>
  <div class="metric-card">
    <div class="metric-title">Server Batch</div>
    <div class="metric-detail">8× L40S, 16 sessions</div>
    <table class="metric-table">
      <tr><td>p50</td><td class="metric-val">1.4 s</td></tr>
      <tr><td>p90</td><td class="metric-val">2.3 s</td></tr>
      <tr><td>p99</td><td class="metric-val">3.8 s</td></tr>
    </table>
  </div>
</div>

## Benchmarking

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 \
  --step-incr 2048 --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

Generate a graph:

```sh
python3 speed-bench/plot_speed.py /tmp/ds4-speed.csv --title "Machine t/s"
```

### M5 Max Baseline

<div class="chart-container">
  <img src="/m5_max_ts.svg" alt="M5 Max Flash Q2 throughput" class="benchmark-chart" />
</div>

## Key Metrics

| Metric | Description |
| --- | --- |
| `prefill_tps` | Tokens/s during prompt processing |
| `gen_tps` | Tokens/s during generation |
| TTFT | Time to first token (server) |
| ITL | Inter-token latency (server) |

## Tuning

| Parameter | Effect |
| --- | --- |
| `--prefill-chunk N` | Chunk size for incremental prefill |
| `--power N` | GPU power limit (V4 only) |
| `--temp 0` | Greedy sampling for benchmarks |
| `--nothink` | Disable reasoning |
| `--cache-budget N` | Total memory budget (MiB) |