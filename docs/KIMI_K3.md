# Kimi K3 on Metal

Kimi K3.1 Pro text inference runs on Metal with Moonshot’s original
SafeTensors. Point `-m` at the directory containing the shards, config and
tokenizer. The full release has 96 shards and uses 1.56 TB of storage.

```sh
hf download moonshotai/Kimi-K3.1-Pro --revision 6d0a0202956796543364d666ab19c6ad1a494ee1 --local-dir models/Kimi-K3.1-Pro
make
./ds4 -m models/Kimi-K3.1-Pro --ctx 4096 -p "Explain Redis streams."
```

During SafeTensors prefill, Metal starts computing ready experts while the
SSD reads the remaining selections. Large batches alternate between halves
of the existing expert buffers, so the reader can start the next group while
the GPU finishes the current one.

K3.1 uses a learned index in 24 attention layers. Each query attends to all
available positions through 2,048 tokens and to its selected 2,048 positions
thereafter. The original K3 SafeTensors and its existing mixed-Q2 GGUF also
run on this branch. K3 vision, CUDA, ROCm, speculative decoding, steering, and reduced-power execution are not supported.

```sh
./ds4 -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 4096 -p "Explain Redis streams."
./ds4-agent -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
./ds4-server -m models/Kimi-K3-IQ2_XXS-Q2_K.gguf --ctx 32768
```

The original K3 GGUF is 799.783 GiB. It streams from SSD automatically when its local
weights cannot fit safely in memory. The router still selects all 16 experts;
streaming changes where the weights live, not the computation.

On a 128 GiB Mac, the runtime keeps the non-routed weights resident when they
fit, then uses the remaining safe budget for an expert cache. With less free
memory it keeps only some layers' non-routed weights resident. This is automatic;
no cache or residency flags are needed.

During single-token GGUF decoding, the runtime uses the next layer’s router
and the current hidden state to predict four likely experts. While Metal
runs the current layer, macOS reads ahead from the GGUF for guesses that
are missing from the expert cache. The next layer’s actual router then
selects all 16 experts. A wrong guess only costs I/O. The runtime skips
prediction across an AttnRes block boundary or when the next layer’s static
weights are on disk. Set `K3_ROUTE_PREDICT=0` to benchmark without lookahead.

After the actual router selects its 16 experts, Metal can start work on experts
already in its cache or whose reads have finished. The other reads continue
while the GPU computes. If the first experts in router order are ready, Metal
can also start their output sum early and resume it in the original order.

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

With the K3.1 Pro checkpoint in `models/Kimi-K3.1-Pro`:

```sh
make -C gguf-tools k3-quantize
./gguf-tools/k3-quantize --model models/Kimi-K3.1-Pro \
  --out models/Kimi-K3.1-Pro-IQ2_XXS-Q2_K.gguf --threads 16
```

Use `--resume` after an interrupted conversion. The converter checks the
partial file and its progress journal before continuing.

`make test-k3-gguf` checks the completed GGUF and its Metal layer path.
`make test-k3-gguf-source` compares retained tensors with the matching source checkpoint. For a K3.1 conversion set both `K3_MODEL` and `K3_GGUF` to its source and output paths.
