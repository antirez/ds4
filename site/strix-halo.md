# Strix Halo (ROCm)

## Prerequisites

For container setup, see the [ROCm toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox).

Native Ubuntu:

```sh
sudo apt-get install -y hipcc rocminfo rocm-smi \
  libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
sudo usermod -aG render,video "$USER"
```

## GPU Memory

Some 128 GB Strix Halo configs only expose ~62 GB to the GPU. Kernel parameters may help:

```
amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

## Build and Run Flash

```sh
make strix-halo
./download_model.sh ds4f-q2
./ds4 --rocm
```

## GLM 5.3 Flash

```sh
./download_model.sh glm53-q2
./ds4 --rocm -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```