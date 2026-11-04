# Models & Vision

DwarfStar is not a general GGUF runner. Use the download targets below.

Run `./download_model.sh --help` for all available targets.

## Memory Requirements

<div v-pre class="model-chart">
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-width:760px">
  <!-- Grid lines -->
  <g stroke="#334155" stroke-width="1" stroke-dasharray="3,3" opacity="0.5">
    <line x1="135" y1="20" x2="135" y2="245" />
    <line x1="285" y1="20" x2="285" y2="245" />
    <line x1="435" y1="20" x2="435" y2="245" />
    <line x1="585" y1="20" x2="585" y2="245" />
  </g>
  <!-- Axis labels -->
  <g fill="currentColor" opacity="0.6" font-size="11" font-family="system-ui">
    <text x="135" y="268" text-anchor="middle">0</text>
    <text x="285" y="268" text-anchor="middle">100 GB</text>
    <text x="435" y="268" text-anchor="middle">200 GB</text>
    <text x="585" y="268" text-anchor="middle">300 GB</text>
  </g>
  <!-- Row 1: Qwen3.8 Q2 - 42 GB -->
  <text x="125" y="44" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">Qwen3.8 Q2</text>
  <rect x="135" y="30" width="63" height="20" fill="#60a5fa" rx="3" />
  <text x="143" y="44" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">42 GB</text>
  <!-- Row 2: Flash Q2 - 81 GB -->
  <text x="125" y="80" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">Flash Q2</text>
  <rect x="135" y="66" width="121" height="20" fill="#3b82f6" rx="3" />
  <text x="143" y="80" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">81 GB</text>
  <!-- Row 3: GLM 5.3 Q2 - 90 GB -->
  <text x="125" y="116" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">GLM 5.3 Q2</text>
  <rect x="135" y="102" width="135" height="20" fill="#60a5fa" rx="3" />
  <text x="143" y="116" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">90 GB</text>
  <!-- Row 4: Flash MXFP4 - 120 GB -->
  <text x="125" y="152" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">Flash MXFP4</text>
  <rect x="135" y="138" width="179" height="20" fill="#3b82f6" rx="3" />
  <text x="143" y="152" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">~120 GB</text>
  <!-- Row 5: PRO Q2 - 250 GB -->
  <text x="125" y="188" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">PRO Q2</text>
  <rect x="135" y="174" width="374" height="20" fill="#93c5fd" rx="3" />
  <text x="143" y="188" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">~250 GB</text>
  <!-- Row 6: V4.1 Q2 - 341 GB -->
  <text x="125" y="224" fill="currentColor" font-size="12" font-family="system-ui" text-anchor="end">V4.1 Flash Q2</text>
  <rect x="135" y="210" width="510" height="20" fill="#1d4ed8" rx="3" />
  <text x="143" y="224" fill="#fff" font-size="11" font-weight="600" font-family="system-ui">341 GB incl. Engram</text>
</svg>
</div>

## Download Targets

| Target | Model | Size | Use Case |
| --- | --- | --- | --- |
| `qwen38-q2` | Qwen3.8 Flash Next Q2 | 42 GB | 64 GB systems |
| `ds4f-q2` | DeepSeek V4 Flash Q2 | ~81 GiB | 96/128 GB systems |
| `glm53-q2` | GLM 5.3 Flash Q2 | ~90 GiB | 128 GB systems |
| `ds4f-q2-q4` | Flash Q2 w/ Q4 layers | ~100 GiB | Larger memory |
| `ds4f-mxfp4` | Flash MXFP4 | ~120 GiB | Blackwell native |
| `ds4f-q4` | Flash Q4 | ~150 GiB | Distributed |
| `pro-q2-imatrix` | PRO Q2 | ~250 GiB | 512 GB / SSD |
| `ds41f-q2` | V4.1 Flash Q2 | 341 GiB | SSD streaming |
| `pro-q4-split` | PRO Q4 (split) | ~400 GiB | Pipeline |

## DeepSeek V4 Flash

```sh
./download_model.sh ds4f-q2
./ds4
```

Vision experimental:

```sh
./download_model.sh ds4f-vision-q2
./ds4 --vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf
```

## DeepSeek V4.1 Flash

| Target | Description |
| --- | --- |
| `ds41f-q2` | V4.1 Flash Q2, 341 GiB incl. Engram tables |
| `ds41f-q2-q4` | Mostly Q2 w/ higher-precision routed layers |
| `ds41f-q4` | Full Q4; needs 512 GB Mac or Spark |

On Metal: text + vision. On CUDA: text only.

### Thinking Levels

```sh
./ds4 --think-level 25
```

1-100 reasoning effort. 0 disables. 75 is default.

## GLM 5.x

| Target | Description |
| --- | --- |
| `glm53-q2` | GLM 5.3 Flash Q2, ~90 GiB |
| `glm53-q4` | GLM 5.3 Flash Q4 |
| `glm52-q2` | Full GLM 5.2 Q2, SSD streaming |

## Qwen3.8 Flash Next

| Target | Description |
| --- | --- |
| `qwen38-q2` | 42 GB weights + 95 GB n-grams on disk |
| `qwen38-q4k` | 70 GB weights + 95 GB n-grams |
| `qwen38-vision` | Optional vision encoder |

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

## Vision Support

| Model | Encoder | Platforms |
| --- | --- | --- |
| DeepSeek V4 Flash Vision | Separate GGUF | Metal, CUDA, ROCm |
| DeepSeek V4.1 Flash | Built-in | Metal only |
| GLM 5.3 Flash | Separate GGUF | Metal, ROCm |
| Qwen3.8 Flash Next | Optional GGUF | Metal, CUDA |