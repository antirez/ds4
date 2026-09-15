# DeepSeek V4.1: two-machine ROCm cluster

- Two ROCm/gfx1151 machines; tested with 128 GB RAM each.
- Same engine revision and `DeepSeek-V4.1-Flash-Q2.gguf` on both. Each keeps the full GGUF; each machine loads approximately 80.6 GiB of weights into RAM. Engram stays on disk.
- Exactly two machines: one coordinator and one worker. They share attention computation and split the experts equally.
- Cluster mode requires the assigned experts to fit in RAM. SSD expert streaming, DSpark and splitting by `--layers` are not supported.
- All three transports require a reachable TCP control address. Use a trusted network: peer traffic has no authentication or encryption.
- Build both peers with `make strix-halo ROCM_ARCH=gfx1151` after installing any required RoCE headers.
- Run from the engine build directory. Set these variables in **both** terminals; `MODEL` may differ between machines:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1       # Coordinator's address on the selected link
CTX=16384
```

## TCP over Ethernet or USB4

- Working Ethernet/IP connection; coordinator TCP port 9911 reachable from the worker. USB4 Ethernet (`thunderbolt_net`) works with the same commands: set `COORD` to the coordinator's USB IP address.
- No verbs packages or USB stream device required.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport tcp --batched-session 1 --host 127.0.0.1 --port 8080

# Worker, in its own terminal
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport tcp
```

## USB4STREAM

- USB4/Thunderbolt host-to-host cable and kernel with `CONFIG_USB4_STREAM` and `CONFIG_USB4_CONFIGFS` (tested: Linux 7.2.5).
- Keep IP connectivity for control; USB Ethernet over the same cable is sufficient. Set `COORD` to its coordinator address.
- If the cable is the only IP path, assign unused addresses to its USB Ethernet interface first; do not replace a management/default route:

```bash
USB_IF=thunderbolt0            # Use the USB Ethernet name printed by ip -br link
# Coordinator only
sudo ip link set "$USB_IF" up
sudo ip address add 10.99.0.1/30 dev "$USB_IF"
# Worker only
sudo ip link set "$USB_IF" up
sudo ip address add 10.99.0.2/30 dev "$USB_IF"
# Both terminals: COORD=10.99.0.1
```

- Create **one bidirectional stream** on each host. The stream name must match; device indexes can differ.
- Setup runs on the hosts as root. Inference runs as your ordinary user.

```bash
# Worker first, then coordinator after the worker HopID allocation below
sudo modprobe thunderbolt_net
sudo modprobe thunderbolt_stream
mountpoint -q /sys/kernel/config || sudo mount -t configfs none /sys/kernel/config
ip -br address                  # Identify the USB Ethernet interface/address

# On the host being configured: identify the connected peer's stream service; never guess its number.
for key in /sys/bus/thunderbolt/devices/*/key; do
  [ "$(cat "$key")" = stream ] && dirname "$key"
done
SERVICE=1-2.0                  # Replace with that host's printed service name
STREAM=/sys/kernel/config/thunderbolt/stream/$SERVICE/ds4data
sudo mkdir -p "$(dirname "$STREAM")"
sudo mkdir "$STREAM"           # Refuses to overwrite an existing stream
```

Run the preceding setup on the **worker first**, then allocate its HopIDs:

```bash
# Worker only; complete this before creating the coordinator's ds4data directory.
printf '%s\n' -1 | sudo tee "$STREAM/in_hopid" "$STREAM/out_hopid"
```

Now run the setup block on the coordinator. Its same-named stream adopts the worker's advertised HopIDs. On **both** hosts:

```bash
cat "$STREAM/in_hopid" "$STREAM/out_hopid"   # Both >= 8; coordinator IN = worker OUT and vice versa
USB_DEV=/dev/tbstream$(cat "$STREAM/index")
sudo udevadm settle
sudo chown "$(id -u):$(id -g)" "$USB_DEV"
sudo chmod 600 "$USB_DEV"
test -c "$USB_DEV" && test -r "$USB_DEV" && test -w "$USB_DEV"
```

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport usb4stream --usb4stream-device "$USB_DEV" \
  --batched-session 1 --host 127.0.0.1 --port 8080

# Worker
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport usb4stream --usb4stream-device "$USB_DEV"
```

- Startup must report `transport=usb4stream` and the intended device.
- Configuration and permissions are temporary; recreate after reboot/reconnection if lost. Stop inference before removing your stream with `sudo rmdir "$STREAM"`.
- Containers need the configured character device, GPU devices and TCP connectivity. Kernel/module setup belongs on the host.

### Tested Strix Halo interrupt fix

- Stock 7.2.5 stalled during sustained stream traffic on the tested AMD controller.
- Applied one change in `ring_clear_msix()` in `drivers/thunderbolt/nhi.c`: read back the interrupt register after its posted clear write.
- Based on Jonathan Yates's [MSI-X clear patch](https://github.com/jyatesdotdev/strix-rdma/blob/d19af99ce91abda691a2bd0f21eb114e0a64bacd/kernel/zerocopy/0011-thunderbolt-Flush-posted-MSI-X-interrupt-clears.patch). The separate RX-prime patch was **not** applied.
- Tested by hot-loading a matching `thunderbolt.ko`; no reboot or persistent boot/module installation. Replacing it interrupts every Thunderbolt user, including USB Ethernet.
- Build against the running distribution kernel's matching source, configuration and headers. Do not load the test binary into another kernel. Secure Boot may require module signing.
- Patch, build, temporary load and rollback: [USB4STREAM kernel fix](USB4STREAM_KERNEL.md). TCP and RoCE do not require this patch.

## RoCE

- Both hosts need RoCE-capable Ethernet adapters, a working driver and an active Ethernet verbs port. Ordinary Ethernet alone is insufficient.
- Install the runtime/provider packages where inference runs; development headers are needed when building the engine. Package names: [Fedora](https://packages.fedoraproject.org/pkgs/rdma-core/), [Ubuntu](https://packages.ubuntu.com/source/jammy/rdma-core).

```bash
# Fedora, both inference/build environments
sudo dnf install libibverbs libibverbs-utils rdma-core-devel iproute

# Ubuntu/Debian alternative
sudo apt install rdma-core libibverbs1 ibverbs-providers ibverbs-utils libibverbs-dev iproute2
```

```bash
# Both hosts/environments: inspect local device, port and GIDs
ibv_devices
ibv_devinfo -v
rdma link
ulimit -l                      # Locked-memory allowance; at least 16 MiB for RoCE staging
DEV=rocep194s0                 # Replace with this host's verbs device
PORT=1
for file in /sys/class/infiniband/"$DEV"/ports/"$PORT"/gids/*; do
  idx=${file##*/}
  printf '%s  %s  %s  %s\n' "$idx" "$(cat "$file")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/types/"$idx")" \
    "$(cat /sys/class/infiniband/"$DEV"/ports/"$PORT"/gid_attrs/ndevs/"$idx")"
done
GID=1                         # Choose this host's nonzero RoCE v2 GID for the cabled NIC/IP
```

- `rdma link` comes from [Fedora iproute](https://packages.fedoraproject.org/pkgs/iproute/iproute/fedora-rawhide.html) or [Ubuntu iproute2](https://packages.ubuntu.com/jammy/all/iproute2/filelist).
- Device names and GID indexes may differ between hosts. `ibv_devinfo` must show an active port with Ethernet link layer.
- The selected device's `/dev/infiniband/uverbs*` must be accessible. Containers also need its device access, userspace provider and adequate memlock allowance; packages alone do not configure the host NIC.
- Set `COORD` to the coordinator's address on the RoCE Ethernet link.

```bash
# Coordinator
./ds4-server --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role coordinator --listen "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID" \
  --batched-session 1 --host 127.0.0.1 --port 8080

# Worker
./ds4 --rocm -m "$MODEL" --ctx "$CTX" \
  --tensor-parallel --role worker --coordinator "$COORD" 9911 \
  --transport rdma --rdma-device "$DEV" --rdma-port "$PORT" --rdma-gid-index "$GID"
```

- RoCE transfers use buffers in system RAM. GPU-direct transfers are not implemented; RCCL is not required.
- Explicit `tcp`, `usb4stream` or `rdma` fails if unavailable. `auto` negotiates configured RoCE, then configured USB4STREAM, then TCP at connection setup; no mid-generation fallback.

## Vision and first request

- Add `--vision /absolute/path/DeepSeek-V4.1-Flash-Vision.gguf` to **both** commands. Keep `--ctx` equal on both.
- The HTTP API runs only on the coordinator. Test port 8080 after startup; do not send HTTP to peer port 9911.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Say hello."}],"temperature":0,"max_tokens":64,"thinking":false}'
```


## Measured performance

- Two Framework Desktop systems, 128 GB each, 16-core Strix Halo / `gfx1151`; coordinator Ryzen AI Max+ 395, worker engineering sample `100-000001243-50_Y`.
- Model drives: coordinator SK hynix PC711 1 TB (PCIe 3.0 ×4, ext4); worker Kingston FURY Renegade 2 TB (`SFYRD2000G`, PCIe 4.0 ×4, btrfs). Engram remains disk-backed.
- TCP/RoCE: Intel E810-C QSFP NICs, 100 Gb/s link, MTU 9000. Coordinator NIC negotiated PCIe 3.0 ×4; worker PCIe 4.0 ×4. USB4STREAM: one 40 Gb/s cable link with the interrupt-readback patch above.
- Existing boot settings include `pci=realloc pcie_aspm=off`, in addition to the [GPU-visible memory settings](STRIX_HALO.md#gpu-visible-memory). Their individual performance effect was not isolated.
- Linux `7.2.5-100.fc43.x86_64`, ROCm 10.0 SDK (`10.0.0-4`, HIP `7.15.26333`). TuneD `accelerator-performance`, fans at maximum speed on both machines.
- Same Q2 file, 69,632 allocated context, fresh full prefix, 128 fixed greedy outputs (127 steady), no DSpark or images. Native `ds4-bench`; one run per cell; startup and a 256-token/128-output warmup excluded. Values are **prefill / decode tokens/s**.

| Prompt tokens | TCP, 100 GbE | USB4STREAM, 40 Gb/s | RoCE RC, 100 GbE |
|---:|---:|---:|---:|
| 8,192 | 375.02 / 14.35 | 340.56 / 14.34 | 372.02 / 14.64 |
| 16,384 | 412.62 / 13.95 | 374.76 / 14.20 | 410.44 / 14.40 |
| 65,536 | 434.16 / 13.70 | 396.84 / 13.76 | 431.55 / 14.10 |

- Longer continuation, 16,384 prompt / 512 generated tokens: TCP **411.53 / 14.01**, USB4STREAM **375.56 / 14.29**, RoCE **405.17 / 14.41** prefill/decode tokens/s. Reproduce with `--gen-tokens 512`.
- All 129,280 frontier logits and printed continuations match across transports at each depth, including 512 outputs. No OOM; minimum usable RAM across the final transport checks: 32.9 GiB. Host zram swap-out was nonzero; these are not zero-swap or cold-cache measurements.
- RoCE device logs and hardware send counters confirm RDMA payloads on both peers. Its TCP control connection is intentional; similar decode rates do not indicate TCP fallback.
- V4.1 CED uses about 8B active parameters/token in prefill and 16B in decode. Full prefixes exercise the decoder-suffix optimization; short appends can follow a different schedule. [Architecture](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/README.md?code=true).
- Results apply to these drives, NIC attachment, profile and USB patch. Other network adapters and USB controllers have not been tested. Long-running production use has not been tested.

### Appending to an existing prompt

Same hardware, allocation and warmup; one live session, no generation between frontiers. Values time only the newly appended tokens, in tokens/s.

| Existing → final tokens | Added tokens | TCP, 100 GbE | USB4STREAM | RoCE |
|---:|---:|---:|---:|---:|
| 4,096 → 8,192 | 4,096 | 219.61 | 201.04 | 218.78 |
| 8,192 → 16,384 | 8,192 | 359.70 | 329.44 | 355.86 |
| 57,344 → 65,536 | 8,192 | 314.92 | 292.18 | 313.65 |

### TCP over the same USB4 cable

| Transport | 16K prefill | Decode, 128 outputs |
|---|---:|---:|
| TCP over USB4 Ethernet | 371.48 | 12.84 |
| USB4STREAM | 374.76 | 14.20 |

- USB4STREAM is optional. Plain TCP over USB4 works with the TCP commands above and the USB IP address. The USB4STREAM setup and stream device are unnecessary for TCP.
- This single comparison observed less than 1% prefill difference and 10.6% faster decode with USB4STREAM. Identical inputs, allocation, binaries, warmup and outputs; both peers' USB routes and byte counters checked. The Ethernet NIC carried no model payload.
- TCP recorded 20 USB receive errors on the coordinator. Both runs used the same patched controller; this comparison does not establish stock-kernel behavior or repeatability of the speed difference.

### Reproduce the table

Build the engine on both machines. On the coordinator, build the included TP-only warmup adapter; it links the existing engine objects and leaves `ds4-bench` untouched:

```bash
make strix-halo ROCM_ARCH=gfx1151
bash speed-bench/build-rocm-v41-warmup.sh  # Coordinator only

tuned-adm active                        # Expect accelerator-performance during the workload
tuned-adm verify                        # Verify the applied profile
```

Set this in both terminals after the relevant device setup above:

```bash
MODEL=/absolute/path/DeepSeek-V4.1-Flash-Q2.gguf
COORD=10.99.0.1                         # Coordinator address on the selected link
TRANSPORT=rdma                         # tcp, usb4stream, or rdma
USBDEV=/dev/tbstream1
DEV=rocep194s0                          # This host's active verbs device
GID=1                                  # This host's matching RoCE v2 GID
LINK=(--tensor-parallel --transport "$TRANSPORT")
case "$TRANSPORT" in
  usb4stream) LINK+=(--usb4stream-device "$USBDEV") ;;
  rdma) LINK+=(--rdma-device "$DEV" --rdma-port 1 --rdma-gid-index "$GID") ;;
esac
```

```bash
# Coordinator: repeat separately with DEPTH=8192, 16384, 65536.
DEPTH=16384
./ds4-bench-warm --backend rocm -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start "$DEPTH" --ctx-max "$DEPTH" --ctx-alloc 69632 \
  --gen-tokens 128 --show-output --csv "tp-$TRANSPORT-$DEPTH.csv" \
  --dump-frontier-logits-dir "frontiers-$TRANSPORT-$DEPTH" \
  --role coordinator --listen "$COORD" 19475 "${LINK[@]}"

# Worker: start for each coordinator run.
./ds4 --rocm -m "$MODEL" --ctx 69632 \
  --role worker --coordinator "$COORD" 19475 "${LINK[@]}"
```

- The adapter warms 256 prefix tokens and 128 decode steps, then creates a fresh session before the unchanged native measured loop. A separate short process is not the same warmup procedure.
- Preserve the CSV, full frontier files, printed continuation, revision/build flags, model identity, active power profile and swap/OOM counters. Verify the active power profile before comparing timings; no image-conditioned prefill timings.

For the append table, keep the worker command and replace the coordinator command with:

```bash
# 4K → 8K → 16K, no generated tokens between appends
./ds4-bench-warm --backend rocm -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 4096 --ctx-max 16384 --step-mul 2 --ctx-alloc 69632 \
  --gen-tokens 0 --csv "append-$TRANSPORT.csv" \
  --dump-frontier-logits-dir "append-frontiers-$TRANSPORT" \
  --role coordinator --listen "$COORD" 19475 "${LINK[@]}"
# For 56K → 64K: --ctx-start 57344 --ctx-max 65536 --step-mul 1 --step-incr 8192
```

For the USB TCP comparison, run the fresh 16K command twice with `COORD` set to the USB IP address on both machines: first `TRANSPORT=tcp`, then `TRANSPORT=usb4stream`. Recreate `LINK` using the case block each time. Check `ip route get "$COORD"` on the worker and `ip -s link show thunderbolt0` on both peers; use your actual USB interface name.
