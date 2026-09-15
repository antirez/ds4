# AMD Strix Halo

[README](../README.md) | [Getting started](../README.md#start-here)

The reference system is a 128 GB Strix Halo with Radeon 8060S (`gfx1151`),
such as the Framework Desktop. The ROCm build uses the standard binary names
and selects the ROCm backend by default.

## Prerequisites

For a container setup, see the maintained
[ROCm toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox/blob/main/toolboxes/Dockerfile.rocm-10.0).
It can also be managed with
[AI Toolbox Cockpit](https://github.com/kyuz0/ai-toolbox-cockpit).

For a native Ubuntu build you need HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA,
and hipCUB development files. The Ubuntu 26.04 setup used these packages:

```sh
sudo apt-get update
sudo apt-get install -y hipcc rocminfo rocm-smi \
  libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
sudo usermod -aG render,video "$USER"
```

Log out and back in after changing groups. `rocminfo` must report `gfx1151`
and be able to open `/dev/kfd` before DwarfStar can run.

Some packaged rocWMMA headers omit `rocwmma/internal/`. If compilation fails
there, install the complete headers matching your ROCm installation, or use
the container. Do not mix header versions as a general workaround.

## GPU-visible memory

Check the GPU-visible memory pool reported by `rocminfo`. Some 128 GB systems expose only about 62 GiB to the GPU. The tested 128 GB Fedora Linux Strix Halo system, running a recent kernel and ROCm 10.0, used these boot parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

The GTT/TTM settings expose about 124 GiB to the GPU. An SSD expert-cache request such as `92GB` is fitted to that GPU-visible limit as well as available system RAM; a stock ~62 GiB pool can therefore yield a much smaller cache. `amd_iommu=off` was part of the tested setup, but is not required for GTT sizing and disables DMA isolation. Keep RAM available for the OS. See the [host configuration guide](https://strix-halo-toolboxes.com/#config) for Fedora, Ubuntu/Debian, and systemd-boot instructions.

## Build and run Flash

```sh
make strix-halo
./download_model.sh ds4f-q2
./ds4 --rocm
```

`make rocm` is an alias. Use the current 0731 Q2 download for a first run;
larger mixed and Q4 models have substantially higher memory requirements.
Flash's ROCm resident and pipeline paths should not be confused with the GLM
SSD-streaming path.

## DeepSeek V4.1 Flash

- ROCm 10.0 supports calibrated V4.1 Flash Q2 text/vision, resident experts, SSD streaming and [two-machine TCP/USB4STREAM/RoCE](CLUSTERING_ROCM.md). Engram remains disk-backed in every mode.
- Tested SSD configuration: 128 GB Framework Desktop, 16-core Strix Halo engineering sample `100-000001243-50_Y`, Radeon `gfx1151`; Kingston FURY Renegade 2 TB (`SFYRD2000G`, PCIe 4.0 ×4, btrfs) holds the model.
- Linux `7.2.5-100.fc43.x86_64`, ROCm SDK `10.0.0-4` / HIP `7.15.26333`; TuneD **`accelerator-performance`**, fans at maximum speed. Existing boot flags: the [GTT/TTM settings above](#gpu-visible-memory), plus `pci=realloc pcie_aspm=off`; their individual effects were not isolated.

### SSD performance

Native `ds4-bench`, full fresh text prefix, greedy decoding, no DSpark or images; 92 GiB expert/staging cache. One run per row, startup and a separate GPU readiness warmup excluded; **tokens/s**:

| Prompt tokens | Allocated context | Generated tokens | Prefill | Decode |
|---:|---:|---:|---:|---:|
| 16,384 | 69,632 | 128 | 302.12 | 8.68 |
| 65,536 | 69,632 | 128 | 350.56 | 8.49 |

- All 129,280 frontier logits and complete printed continuations match the corresponding resident runs. Minimum usable RAM: 15.1 GiB; no OOM. Host zram swap-out pages in table order: 0, 0. No cold-cache claim; other qualification runs recorded nonzero host swap.
- The tuned Engram matrix path requires hipBLASLt 100401, revision `8d1ae90e`; other library versions retain the existing fallback and may have different prefill performance.
- Actual prompts reach 65,536 tokens; populated 256K was not tested. Cache admission depends on available RAM, context and sessions; images may need a smaller cache. The GPU-visible limit shares system RAM and is not a cache budget.
- Six resident image/state cases and two focused SSD cases (photo and screenshot) pass on this source. Official probability results are mixed; see [quality and limitations](../QA_BEFORE_RELEASES.md#deepseek-v41-flash-rocmgfx1151). No image-conditioned prefill timing is included.

### Run text or vision

```bash
make strix-halo ROCM_ARCH=gfx1151
./download_model.sh ds41f-q2
./download_model.sh ds41f-vision
MODEL=gguf/DeepSeek-V4.1-Flash-Q2.gguf
VISION=gguf/DeepSeek-V4.1-Flash-Vision.gguf

# CLI, text
./ds4 --rocm -m "$MODEL" --ssd-streaming \
  --ssd-streaming-cache-experts 92GB --ctx 69632

# HTTP server, text and images; --vision takes the matching sidecar.
./ds4-server --rocm -m "$MODEL" --vision "$VISION" \
  --ssd-streaming --ssd-streaming-cache-experts 92GB --ctx 69632 \
  --batched-session 1 --host 127.0.0.1 --port 8080
```

For a machine with sufficient RAM for resident experts, omit both SSD options. Keep `--vision` for image requests and set `--ctx` to the required allocation. See [image request examples](MODELS.md#vision).

### Reproduce SSD measurements

Run one configuration per process; preserve the CSV, full frontier files and printed output. The timing input is the repository's `speed-bench/promessi_sposi.txt`.

```bash
tuned-adm active    # Expect accelerator-performance during the workload
tuned-adm verify
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
DEPTH=16384
ALLOC=69632
GEN=128
# Other row: DEPTH=65536 ALLOC=69632 GEN=128

DS4_METAL_CB_TIMES=1 ./ds4-bench --backend rocm -m "$MODEL" \
  --ssd-streaming --ssd-streaming-cache-experts 92GB \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start "$DEPTH" --ctx-max "$DEPTH" --ctx-alloc "$ALLOC" \
  --gen-tokens "$GEN" --show-output --csv "ssd-$DEPTH-$GEN.csv" \
  --dump-frontier-logits-dir "ssd-frontiers-$DEPTH-$GEN"
```

- `DS4_METAL_CB_TIMES` is scoped to this command and prints the measured prefill time window on ROCm too. No tuning override is needed.
- Check the active power profile during the measurement; save revision/build flags, model filename/size and existing provenance, cache/KV configuration, actual prompt/output counts, and memory/swap/OOM counters. Do not substitute HTTP timings for this native table.

## GLM 5.3 Flash

The reference Q2 setup uses SSD streaming to leave room for its graph and KV
state. Begin with automatic cache sizing and a small context:

```sh
./download_model.sh glm53-q2
./ds4 --rocm -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```

GLM 5.2 also supports ROCm streaming. Full-model GLM 5.2 inference requires it;
distributed layer slices can be resident. See [SSD streaming](SSD_STREAMING.md)
before adjusting the cache budget.

Both GLM 5.3 Flash and DeepSeek Flash Vision Experimental support images on
ROCm. Add the matching encoder with `--vision FILE`, as described in
[models and vision](MODELS.md#vision).

For a model-free routed-kernel check, use `make test-mxfp4-rocm`.
Full-model validation is described in [testing](TESTING.md).
