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

Check the memory pool reported by `rocminfo`. Some 128 GB configurations expose
only about 62 GB to the GPU, which is insufficient for resident Flash Q2 plus
runtime buffers. Firmware and kernel GTT/TTM settings control this limit.

The native reference setup used these memory parameters:

```text
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

They are a system-specific starting point, not an allocation budget for
DwarfStar. Preserve existing boot options and consult your kernel's settings
before changing them. Keep RAM available for the OS even when the GPU can
address most of it. Do not disable the IOMMU merely to copy another host's
configuration; doing so changes device isolation.

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

Resident single-token DeepSeek V4 decode on ROCm fuses Q/KV weighted RMS
normalization and KV RoPE into one launch when the shape has at most 256 RoPE
pairs. This attention optimization is independent of the routed-expert
quantization and applies to both Q4_K and MXFP4 models. Larger shapes retain the
two-launch path. Set `DS4_ROCM_DISABLE_QKV_KV_ROPE_FUSION=1` before process
startup for a diagnostic rollback. Run the model-independent, bit-exact
reference test with:

```sh
make HIPCC=/opt/rocm-therock/bin/hipcc test-rocm-qkv-fusion
```

It covers dense and YaRN parameters, inverse and multi-row/head layouts,
zero rotation, the 256-pair boundary, the 257-pair fallback, and rejected
shapes, requiring bit-exact Q and KV output.

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
