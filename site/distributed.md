# Distributed Inference

## Two Macs (Tensor Parallelism)

Thunderbolt RDMA recommended.

**Coordinator:**

```sh
./ds4 --role coordinator --tp-partner hostname-or-ip:port --tp-link rdma
```

**Worker:**

```sh
./ds4 --role worker --tp-partner coordinator-hostname-or-ip:port --tp-link rdma
```

Both need the same GGUF at the same path.

## Two DGX Sparks

```sh
./ds4 --cuda --role coordinator --tp-partner spark2-ip:port
./ds4 --cuda --role worker --tp-partner spark1-ip:port
```

Both Sparks need the GGUF on local SSD.

## Pipeline Parallelism

Chain multiple machines:

```sh
./ds4 --role pipeline --pipeline-index 0 --pipeline-count 3
./ds4 --role pipeline --pipeline-index 1 --pipeline-count 3
./ds4 --role pipeline --pipeline-index 2 --pipeline-count 3
```

## Link Options

| Link | Latency | Notes |
| --- | --- | --- |
| Thunderbolt RDMA | Low | Recommended for two Macs |
| TCP | Higher | Any network |
| RoCE | Low | Two Sparks |

## Performance

Two Sparks RoCE: 21.9 t/s per session, ~28 t/s aggregate (8 sessions).