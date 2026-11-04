# SSD Streaming

Run models larger than available RAM.

## Automatic Cache Sizing

```sh
./ds4 --ssd-streaming
```

## Manual Configuration

```sh
./ds4 --ssd-streaming --expert-cache-size 4096
```

With a total memory budget:

```sh
./ds4 --ssd-streaming --cache-budget 64000
```

This reserves 64 GiB for the entire DwarfStar process.

## Performance Tips

- Use a fast local SSD (NVMe recommended)
- Engram tables remain on disk in every mode
- Repetitive prompts benefit from larger cache
- Don't oversize the cache — leave memory for the OS

## V4.1 SSD Performance

On a 128 GB Spark with automatic cache sizing:

| Scenario | Speed |
| --- | --- |
| 3,241-token prefill | 93-96 t/s |
| 32K prefill, 64 GiB budget | 384 t/s |
| 3.2K append | 88 t/s |
| 256-token decode | ~5% slower than resident |