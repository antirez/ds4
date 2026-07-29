# Kimi K3 on Metal

This branch supports the DwarfStar mixed-Q2 Kimi K3 GGUF and the original
SafeTensors directory. Both use Metal. K3 vision, CUDA, ROCm, speculative
decoding, steering, and reduced-power execution are not supported.

```sh
make
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 4096 -p "Explain Redis streams."
./ds4-agent -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
./ds4-server -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
```

The GGUF is 799.783 GiB. It streams from SSD automatically when its local
weights cannot fit safely in memory. The router still selects all 16 experts;
streaming changes where the weights live, not the computation.

Routed gate/up weights use IQ2_XXS and routed down weights use Q2_K.
Shared experts remain MXFP4, latent up projections use Q8_0, and attention,
embeddings, the output head, routers, and norms retain higher precision.

K3 uses its native XTML chat and tool format. Its sampling defaults are
temperature 1, top-p 0.95, and min-p 0. Saved sessions rebuild their inference
state by replaying the saved tokens.

## Two 512 GiB Macs

Put the same GGUF on both machines. This is a contiguous-layer pipeline, not
tensor parallelism. The split below assigns about 401.16 GiB to the coordinator
and 398.62 GiB to the worker, before context and runtime buffers.

If needed, raise the Metal wired-memory limit on each **512 GiB** machine:

```sh
sudo sysctl iogpu.wired_limit_mb=480000
```

Do not use that limit on a smaller Mac.

```sh
# Machine A
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768 \
  --role coordinator --layers 0:46 --listen 0.0.0.0 9000

# Machine B
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768 \
  --role worker --layers 47:output --coordinator MACHINE_A 9000
```

The runtime pins each local slice only when it fits the live memory budget.
The pipeline transfers the complete FP32 AttnRes state and negotiates the
prefill chunk size between nodes. `K3_GGUF_RESIDENT=0` disables residency
for diagnostic comparisons.

## Conversion

With the original checkpoint in `models/Kimi-K3`:

```sh
make -C gguf-tools k3-quantize
./gguf-tools/k3-quantize --model models/Kimi-K3 \
  --out models/Kimi-K3-IQ2_XXS-Q2_K.gguf --threads 16
```

Use `--resume` after an interrupted conversion. The converter checks the
partial file and its progress journal before continuing.

`make test-k3-gguf` checks the completed GGUF and its Metal layer path.
`make test-k3-gguf-source` additionally compares retained tensors with the
original checkpoint.
