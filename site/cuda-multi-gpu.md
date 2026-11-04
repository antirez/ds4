# CUDA Multi-GPU

Supports Ada Lovelace through Blackwell, including 48 GB L40S cards.

## Build

```sh
make cuda-generic
# Or for Ada:
make cuda CUDA_ARCH=sm_89
```

## Layer Placement

```sh
./download_model.sh ds4f-q2
./ds4 --cuda --gpu-devices 0,1,2,3 --gpu-vram auto --ctx 32768
```

## Tensor Parallelism

Pairs GPUs and divides routed-expert work:

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-devices 0,2,4,6,1,3,5,7 --gpu-vram auto \
  --ctx 100000 --batched-session 16 --host 0.0.0.0
```

Eight L40S with 16 sessions: ~126 aggregate tokens/s.