# Inference Across Machines

[README](../README.md)

There are two modes:

| Mode | Split | Main use |
| --- | --- | --- |
| Tensor parallelism | Routed experts and per-layer work across two Macs | Resident inference with lower per-token work on each GPU |
| Pipeline parallelism | Complete layer ranges across several machines | Fit larger models and overlap long prefills |

These are separate from [tensor parallelism across CUDA cards](CUDA_MULTI_GPU.md).
Network protocols have no authentication or encryption. Use trusted machines
and a trusted network; run the same commit on every peer. Model paths and
artifacts must agree. Update all TP peers together when changing versions.

## Tensor parallelism between two Macs

This is a 50/50 split with exactly one worker. Do not pass `--layers`.
Routed experts are sharded; dense weights remain replicated. Both GPUs work
on the same token and exchange partial results. This can reduce generation
latency, but the gain depends on the model, link, and comparison setup.

Two 128 GB Macs are useful for Flash Q4/MXFP4 or GLM 5.3 Flash Q4.
GLM 5.2 IQ2_XXS is another tested capacity setup. A larger quant may need
larger machines even though its tensor layout is supported.

### Link setup

Use a Thunderbolt cable. RDMA requires an active verbs device with an
IPv4-mapped GID; a working ping alone does not establish that.

```sh
rdma_ctl status
ibv_devinfo -v
```

Addresses must be on the cabled member interfaces, not only the Thunderbolt
bridge. For example, after checking which interfaces are active:

```sh
# Machine A, example member interface en1.
sudo ifconfig en1 inet 10.99.0.2/30 alias
# Machine B, example member interface en6.
sudo ifconfig en6 inet 10.99.0.1/30 alias
```

For the large tested shards on otherwise idle 128 GB Macs, the setup raised
the per-boot GPU wired-memory limit on both machines:

```sh
sudo sysctl iogpu.wired_limit_mb=120000
```

This is specific to that memory configuration. It grants a larger GPU budget;
it does not create more RAM. Check model, context, and system headroom before
raising a limit on your machine.

### Start the pair

Download the same model on both machines. For GLM 5.3 Flash Q4:

```sh
./download_model.sh glm53-q4
```

Start the worker first; it retries while the coordinator loads:

```sh
# Machine B.
./ds4 --tensor-parallel --role worker \
  --coordinator 10.99.0.2 9911 --transport rdma --ctx 8192

# Machine A.
./ds4 --tensor-parallel --role coordinator \
  --listen 10.99.0.2 9911 --transport rdma --ctx 8192
```

The verbs device and GID are selected automatically. If ambiguous, specify
`--rdma-device` and `--rdma-gid-index` from `ibv_devinfo`. Use `--transport tcp`
on both peers when RDMA is unavailable. Do not probe the waiting coordinator
with `curl` or `nc`: it may treat the connection as a worker handshake.

Keep workers running in a terminal or managed session and retain both logs.
Do not treat repeated handshake or RDMA timeouts as successful QA merely
because a retry works.

The coordinator can be `ds4`, `ds4-agent`, `ds4-server`, or `ds4-bench`;
workers run `ds4`. Pass the same `--vision FILE` to both for image input.
For GLM MTP, enable `--mtp` on both. For DeepSeek DSpark, both need the
matching support model and DSpark options.

TP disk-cache restore currently rebuilds the exact saved token prefix on both
ranks rather than restoring the coordinator alone. Expect prefill on restore.
See [speculation](SPECULATIVE_DECODING.md) and [serving](SERVER.md).

## Pipeline parallelism

Each process maps only its assigned layers, retaining that slice of the KV
state. Layer ranges are inclusive. `N:output` includes the final layer and
output head. Activations travel from one stage to the next over TCP.

For Flash Q4 on two machines, download `ds4f-q4` on both, then start each side.
Replace the example address with your coordinator's reachable address:

```sh
# Machine A.
./ds4 --role coordinator --layers 0:19 --listen 10.99.0.2 9911

# Machine B.
./ds4 --role worker --layers 20:output --coordinator 10.99.0.2 9911
```

Normally give the output head to the final worker. With several workers,
choose non-overlapping ranges covering the entire model. Workers register
their ranges with the coordinator; intermediate workers forward activations
directly to the next stage.

Long prefill chunks can occupy different stages simultaneously. A single
generation stream cannot use that overlap: each token must finish the route
before the next one is sampled. Use pipeline mode primarily for capacity and
long-prefill throughput, not as a guaranteed decode speedup.

### Full PRO Q4

For two 512 GB Mac Studios, use the split artifacts:

```sh
# Machine A.
./download_model.sh pro-q4-layers00-30
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers00-30.gguf \
  --role coordinator --layers 0:30 --listen 10.99.0.2 9911

# Machine B.
./download_model.sh pro-q4-layers31-output
./ds4 -m gguf/DeepSeek-V4-Pro-Q4K-Layers-31-output.gguf \
  --role worker --layers 31:output --coordinator 10.99.0.2 9911
```

These downloads do not change `ds4flash.gguf`. Startup is expensive because
each side must make its model slice resident.

### Tuning and recovery

Keep default chunk sizes first. `--dist-prefill-window N` controls the number
of chunks in flight; `--dist-prefill-chunk N` overrides the session-derived
chunk size. `--debug` shows route and per-hop timings.

Activations use 32-bit transport by default. `--dist-activation-bits 16` halves
the payload; `8` is more aggressive. These change numerical precision on the
wire, not weights or KV storage. Validate output when changing them.

A disconnected worker invalidates the route. In-flight work can fail; later
requests need a complete route before proceeding, and the coordinator can
replay the saved token prefix to rebuild worker state. Pipeline snapshots
serialize all layer slices into one payload and redistribute them when loaded.

For protocol details, see [ds4_distributed.c](../ds4_distributed.c)
and [ds4_tp.c](../ds4_tp.c).
