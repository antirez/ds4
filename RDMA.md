# Hardware-assisted DS4 transport over Thunderbolt/USB4

## Goal

Reduce the latency and CPU overhead of moving DS4 boundary tensors between a
paired set of Strix Halo systems. The target is hardware-assisted data movement,
not a software RDMA implementation layered over the existing Thunderbolt TCP
link.

## Conclusion

Soft-RoCE (`rxe`) or software iWARP (`siw`) is not the useful path here. Both
would retain the network stack and execute the RDMA protocol in software, so
they are unlikely to improve on well-tuned TCP over `thunderbolt-net` for this
latency-sensitive workload.

The experimental path explored here is a custom zero-copy DS4 transport built
on the USB4/Thunderbolt Native Host Interface (NHI) DMA rings. The NHI already
performs the physical
transfer from a local DMA address into pre-posted memory on the other host. A
kernel driver can expose fixed, mmap-able TX/RX buffer pools to userspace and
submit those buffers directly to the NHI rings. If ROCm can access the same
pages, boundary tensors can move:

```text
sending GPU -> shared DMA page -> NHI -> cable -> NHI -> shared DMA page -> receiving GPU
```

This would be hardware-DMA message passing with send/receive semantics. It is
not full InfiniBand-style RDMA: the Strix Halo NHI has no public support for
remote virtual addresses, rkeys, queue-pair state, RDMA reads/writes, atomics,
or an RDMA reliability engine. A kernel patch cannot add those missing hardware
features. True one-sided RDMA would require an external RDMA NIC or FPGA.

## Implementation status (2026-08-06)

The initial one-device-to-one-device DS4 path is now implemented in software.
TCP remains the control channel. Protocol v3 negotiates either
descriptor-framed TCP or NHI, uses an ACK/READY activation barrier, and carries
an exact 64-byte bulk descriptor with generation, directional sequence,
identity, size, width, and frame count. The Linux NHI backend persistently owns
the zero-copy mmap and one mixed completion dispatcher.

The production selection is `--dist-transport auto` with no
`--dist-nhi-device`. That keeps descriptor-framed protocol v3 and selects TCP
when no NHI candidate is supplied. Explicit `--dist-transport tcp` selects
legacy protocol v2. Any configuration that supplies an NHI device is lab-only
and remains an explicit single-link experiment because it has not improved
full-model token throughput.

Within the experimental NHI backend, the normal path stages payloads through
the mmap pools with CPU copies. Contiguous 32-bit payloads can instead be handed
off through mapped TX/RX leases by setting `DS4_DIST_NHI_MAPPED=1` on both
peers. In that explicit qualification mode, graph tensor copies go directly to
or from the registered driver slot and a HIP synchronization fences every NHI
ownership transfer; there is no intermediate application heap buffer.
Ring-wrapping or reduced-precision payloads retain the CPU-copy NHI path.
Single-token ROCm decode can additionally rebind graph I/O to the mapped alias
with `DS4_DIST_NHI_DIRECT_SLOTS=rx|tx|both` (default `off`). The mapped pages are
system/GTT memory, not peer VRAM and not GPUDirect RDMA.

`DS4_DIST_NHI_UNSAFE_FAST=1` enables a benchmark-only handoff that skips the
redundant userspace HIP synchronization when the caller has already completed
GPU work, and omits canonical copyback for terminal outputs proven dead before
reuse. The transport still validates generation/sequence descriptors and the
kernel still owns NHI DMA submission and RX reposting. The conservative path
remains the default.

Every connection gets a new coordinator-issued generation. TCP descriptors and
NHI envelopes repeat it and the exact next directional sequence. Malformed,
stale, partially written, or ambiguous traffic abandons the generation. The
implementation currently rejects v3 routes with more than one remote worker.
Semantic WORK rejection consumes and reposts an already validated mapped RX
lease before returning the error, so ordinary request errors do not poison a
healthy generation.

The current applications build and pass the ROCm, transport, and protocol
gates in isolated directories on both Strix hosts. Fixed-size raw zero-copy,
ring-wrap, mapped-pool, orderly-close, GPU-alias, asymmetric DS4-shape, and
full-model equivalence checks pass. The earlier full-model delivery failure was
traced below DS4 to a lost MSI-X notification and RX priming order; the stream
driver repairs now survive ring wrap and fresh-open qualification. Local TX
completion is still not proof of remote delivery, and pre-activation fallback
still does not make an already active or ambiguous generation safe to replay.

The host safety prerequisites are now in place: translated/default IOMMU
operation is enabled, ROCm and the NPU remain healthy, the stream device is
`0660 root:tbstream`, and the allocator/follower lifecycle publishes
`/run/ds4-tbstream/device`. The repaired pair completes repeated full-model NHI
cohorts without timeout or replay. A matched 800-token A/B on the current
direct-slot build measured 11.2557 tok/s on staged NHI and 11.2407 tok/s with
direct TX plus the fast handoff, a noise-scale -0.13% change. Per-token
profiling attributes about 33.5 ms to each synchronized model-evaluation span
in a 68.1 ms distributed decode step; all four mapped-lease synchronizations
total about 0.015 ms. The spans include setup and staging inside the evaluation
calls, so the remaining target is evaluation internals and scheduling, not
ownership-fence removal.

The opt-in ROCm event profiler now resolves that evaluation span without adding
a synchronization point. Set `DS4_ROCM_EVENT_PROFILE=1`; reports contain
`DS4_ROCM_EVENT_PROFILE_INTERVAL` samples (default `32`). It is limited to
ordinary resident, non-streaming, non-speculative single-token layer-slice
decode on one tier/device, with no placement plan or synchronized stage
profiler. Fourteen timing-enabled HIP events form 13 adjacent stream-0 segments
while rotating the sampled layer across tokens. Collection occurs only after
the existing `ds4_gpu_end_commands()` completion. Timing failure disables
profiling and leaves inference running.

Two balanced 800-token TCP arms measured pooled request service time of
`7.15150 s` with profiling and `7.14820 s` without it: `+0.04617%` latency, or
`-0.04614%` throughput, with zero dropped samples or coverage differences. The
profiled full-model means were `33.98704 ms` stream-0 versus `34.09416 ms` host
on the coordinator (`0.10712 ms` residual), and `33.69936 ms` versus
`33.82140 ms` on the worker (`0.12204 ms` residual). The worker output head was
`2.51466 ms`. The approximately extrapolated sampled-layer body divided into
routed MoE `24.55%`, attention output `24.24%`, QKV `20.71%`, shared/post
`10.05%`, compressor/indexer `9.80%`, FFN/router `7.33%`, and attention core
`3.13%`. These percentages are directional because one rotating layer carries
the event overhead. They show that stream-0 GPU work accounts for nearly all
of each evaluation span and that no one repeated layer stage dominates it.

The event `copyback` segment counts only queued direct/canonical D2D copies;
ordinary staged GPU-to-host reads occur after collection and appear in
`host_minus_stream0`. In a same-nonce NHI comparison, worker output-head time
was `2.51525 ms` staged and `2.58050 ms` with direct logits (`+0.06525 ms`,
`+2.594%`), while combined coordinator-plus-worker stream-0 time was
effectively unchanged (`67.84875 ms` staged versus `67.85050 ms` direct).
Direct-slot coverage was 124/128 worker samples at mask `0x4` and 127/128
coordinator samples at mask `0x2`. This is consistent with mapped GTT output
slightly slowing the head rather than exposing hidden transport savings.

## Why the current TCP path costs more

The current DS4 layer-slice path reads a GPU tensor into host memory, sends it
through the distributed TCP transport, and writes the received host buffer into
the peer GPU tensor. Below that, `thunderbolt-net` copies TX payload into its own
4 KiB DMA buffers, while TCP/socket handling adds further copies, protocol work,
and wakeups.

The resulting data path is approximately:

```text
GPU -> DS4 host buffer -> socket/TCP -> thunderbolt-net DMA pages
    -> NHI/cable/NHI -> network/TCP -> DS4 host buffer -> GPU
```

The relevant DS4 calls are currently in `ds4_session_eval_layer_slice()` and
use `ds4_gpu_tensor_read()`, the distributed socket transport, and
`ds4_gpu_tensor_write()`.

Linux's [`thunderbolt-net` TX path](https://github.com/torvalds/linux/blob/master/drivers/net/thunderbolt/main.c#L1040-L1151)
supports scatter/gather and TCP segmentation offload, but it still copies SKB
payload into NHI ring pages.

## Upstream USB4STREAM support

Upstream Linux has a much better starting point than `thunderbolt-net`:

- [`USB4STREAM` support](https://github.com/torvalds/linux/commit/6db21d817b43f8ce5654ccc7aff80d40e4dba4ac)
  adds the `thunderbolt-stream` driver and `/dev/tbstreamX` character devices.
- It transfers data directly over a Thunderbolt/USB4 tunnel without traversing
  the network stack.
- Multiple bidirectional streams can coexist, including alongside
  `thunderbolt-net`.
- It supports configurable ring sizes and interrupt throttling.
- The ABI entry in the upstream change identifies Linux 7.2. The test systems
  were running 7.1.5 when this was investigated, so using it requires a 7.2
  kernel or a backport.

The [upstream kernel documentation](https://docs.kernel.org/admin-guide/thunderbolt.html#streaming-data-directly-over-thunderbolt-cable)
describes ConfigFS setup and the `/dev/tbstreamX` interface.

Stock USB4STREAM is not yet zero-copy. It allocates and DMA-maps its own pages,
then uses [`copy_page_from_iter()` on TX](https://github.com/torvalds/linux/blob/master/drivers/thunderbolt/stream.c#L467-L505)
and [`copy_page_to_iter()` on RX](https://github.com/torvalds/linux/blob/master/drivers/thunderbolt/stream.c#L599-L745).
It should remove TCP overhead and is worth measuring as a baseline, but it will
not provide the desired GPU-to-NHI direct path without further changes.

## What the NHI can offload

An NHI ring descriptor contains a local physical/DMA address, a length, framing
flags, and completion/interrupt flags. See the upstream
[`nhi_ring_desc`](https://github.com/torvalds/linux/blob/master/drivers/thunderbolt/nhi_regs.h#L20-L38).
The public ring interface currently describes frames of at most 4096 bytes.

That is enough for an efficient pre-posted message transport:

1. The receiver posts DMA-mapped slots to its RX ring.
2. The sender fills a registered TX slot and submits descriptors that reference
   it.
3. NHI hardware moves the frames over the cable into the peer's posted slots.
4. The receiver gets one completion for the logical tensor and hands ownership
   to the GPU.

Large tensors will span many 4 KiB NHI frames. The driver should request a
completion interrupt only for the last descriptor in a logical message, rather
than waking the CPU for every frame. The current ring code may need a small core
change because descriptor interrupt behavior is not fully controlled by the
USB4STREAM client.

## DS4 NHI transport

TCP is kept for discovery, negotiation, health checks, and small control
messages. A dedicated NHI stream carries only bulk boundary tensors and
successful batched logits.

This section describes the implemented experimental path, not the current
production configuration. Production uses v3 TCP via `auto` without an NHI
device.

### Kernel side

Start from Linux 7.2 USB4STREAM or backport its driver, then add a zero-copy UAPI
with the following concepts:

- Fixed TX and RX slot pools allocated and DMA-mapped once.
- `mmap()` access to those pools from the DS4 process.
- Explicit `POST_RX`, `SUBMIT_TX`, and `REAP_COMPLETIONS` operations, either as
  ioctls or `io_uring` commands.
- Producer/consumer indices and generation numbers so stale completions cannot
  reuse a slot incorrectly.
- Credit-based flow control so the sender never overruns peer RX slots.
- Scatter/gather descriptors for messages larger than one page.
- Per-message rather than per-frame completion interrupts.
- Event notification through polling, `eventfd`, or `io_uring`, with a busy-poll
  option for the lowest-latency benchmark.
- Correct disconnect cancellation and cleanup of every pinned/mapped page.

Use multiple slots (at least double buffering, probably 4-8 per direction) so
the next tensor can be produced while the previous one is on the link.

### ROCm integration

The first proof of concept should mmap driver-owned pages into the DS4 process
and test whether `hipHostRegister()`/mapped host memory lets the Strix Halo GPU
read and write those pages at acceptable latency. The integrated GPU and shared
system memory make this plausible, but it must be demonstrated; it should not
be assumed.

If registering driver-mapped pages is unsupported or slow, investigate exporting
the slot pool as DMA-BUF and importing it through the supported ROCm/KFD memory
path. Explicit ownership fencing is required in either case:

```text
TX: GPU completion -> DMA sync/fence -> NHI owns slot -> TX completion -> reusable
RX: NHI completion -> DMA sync/fence -> GPU owns slot -> GPU completion -> repost
```

The driver must use the Linux DMA API correctly for the NHI device. Keep the
IOMMU enabled: it provides DMA isolation and does not prevent this design.
Disabling it would enlarge the blast radius of a driver or protocol bug and is
not a prerequisite for direct DMA.

### DS4 integration

The implemented transport abstraction preserves TCP rather than replacing it.
The initial NHI backend:

- negotiates protocol capabilities and local geometry without exposing local
  slot IDs, because peer ring cursors are independent;
- validates a generation/sequence-tagged TCP descriptor before consuming NHI;
- moves 32-bit graph tensor data directly between HIP and a registered TX/RX
  slot where the payload is contiguous;
- retains a persistent CPU-copy NHI path for wrapping/reduced-width data;
- carries only control metadata on TCP for an out-of-band tensor;
- supports hidden-state and successful logit payloads; and
- preserves descriptor-framed TCP and legacy v2 reconnect as pre-activation
  fallbacks.

## Suggested implementation order

1. Record current TCP latency, bandwidth, CPU time, copies, message sizes, and
   token throughput at batch sizes 1, 2, 4, and 8.
2. Boot Linux 7.2 (or backport USB4STREAM) on both hosts and benchmark unmodified
   `/dev/tbstreamX` with DS4-sized messages. This isolates the benefit of
   bypassing TCP even though copies remain.
3. Add mmap-able fixed buffer pools and a userspace ping-pong test. Validate
   correctness, NHI DMA, interrupt count, and QD1 latency before involving ROCm.
4. Prove GPU access to the mapped pool and measure GPU-to-GPU latency. Fall back
   to DMA-BUF work only if the simpler mapped-memory path fails.
5. Add the optional NHI backend to DS4 and pipeline tensor production, transfer,
   and consumption.
6. Tune ring depth, slot count, CPU affinity, interrupt affinity, interrupt
   throttling, frame batching, and spin-versus-sleep completion behavior.
7. Run long soak tests, disconnect/reconnect tests, IOMMU fault tests, and output
   equivalence checks before using it as the default transport.

## Benchmark matrix

Compare all three transports with identical tensors and model settings:

| Transport | Network stack | User/kernel payload copies | Intended role |
|---|---:|---:|---|
| TCP over `thunderbolt-net` | Yes | Yes | Production baseline |
| Stock USB4STREAM | No | Yes | Low-risk intermediate baseline |
| NHI stream | No | CPU-copy or mapped, by mode | Experimental; no measured model gain |

For each, measure:

- One-way and round-trip latency at the actual hidden-state and logit sizes.
- p50, p95, p99, and maximum latency, not just aggregate bandwidth.
- Queue depth 1 and pipelined transfers.
- Batch sizes 1, 2, 4, and 8.
- Prefill and decode separately.
- CPU utilization, context switches, interrupts, and bytes copied.
- End-to-end tokens/second and time/token.

The end-to-end gain is bounded by the share of token time currently spent on
transport. A large reduction in microbenchmark latency may produce a smaller
token-rate improvement if GPU compute dominates. Batched inference and logit
movement are likely to benefit more because transferred messages are larger and
there is more opportunity to pipeline work.

## Decision gates

- If stock USB4STREAM does not materially beat TCP at DS4's message sizes,
  profile the GPU staging copies and scheduling before writing a large kernel
  patch.
- If mmap zero-copy materially beats stock USB4STREAM, proceed with GPU mapping.
- If the GPU cannot access the NHI pool efficiently, estimate DMA-BUF work versus
  the measured maximum benefit before continuing.
- If true one-sided RDMA semantics become a requirement, use external hardware;
  do not try to synthesize an HCA in the NHI driver.

## Rough effort

A focused proof of concept is approximately one week once both hosts have a
USB4STREAM-capable kernel: bring-up and baseline testing, mmap-able ring pools,
a userspace ping-pong tool, and a first ROCm registration experiment. Production
hardening and upstream-quality interfaces would take longer, especially around
disconnect handling, synchronization, security, and compatibility.
