# DS4 on AMD GPUs other than Strix Halo

[STRIXHALO.md](STRIXHALO.md) covers Ubuntu, GRUB and `gfx1151` with 128 GB of
RAM. This file covers what changes on an Arch-family distribution with
systemd-boot and a smaller RDNA3 part. The reference machine is a Radeon 780M
(`gfx1103`, Ryzen 7 8845HS) with 58 GiB of usable system RAM, ROCm 7.2.4 and
kernel 7.2.0.

What was verified here: `make rocm` builds, `make test-mxfp4-rocm` passes,
`make cpu` still builds, and DeepSeek V4 Flash IQ2XXS answers prompts through
`--ssd-streaming` (section 7). What was not: `make test`, which needs a model
file. The build carries one pre-existing warning in `ds4_server.c`, unrelated
to anything described here.

## 1. ROCm packages

On Arch and derivatives the SDK is one group, and rocWMMA is a separate
package:

```sh
sudo pacman -S --needed rocm-hip-sdk rocwmma rocminfo
```

Add yourself to the `render` and `video` groups as in STRIXHALO.md.

## 2. hipcc does not find its own headers

On this setup `hipcc` invokes `clang++` without adding the ROCm include
directory, so any HIP source fails at the first include:

```
fatal error: 'hip/hip_runtime.h' file not found
```

`/opt/rocm/include/hip/hip_runtime.h` is present; the driver simply does not
pass it. `--rocm-path=/opt/rocm` fixes it. Section 4 shows where to put it.

## 3. rocWMMA does not know gfx1103

`rocwmma/internal/config.hpp` enumerates the architectures it supports and
fails the build on anything else:

```
config.hpp:79:15: error: static assertion failed: Unsupported architecture
```

In rocWMMA 2.2.0, shipped with ROCm 7.2.4, that list is `gfx908`, `gfx90a`,
`gfx942`, `gfx950`, `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`,
`gfx1200`, `gfx1201`. `gfx1103` is absent although it implements the same
RDNA3 WMMA instructions as `gfx1102`. Reproduce with:

```sh
printf '#include <rocwmma/rocwmma.hpp>\nint main(){return 0;}\n' > /tmp/w.hip
hipcc --offload-arch=gfx1103 --rocm-path=/opt/rocm -c -o /tmp/w.o /tmp/w.hip
```

Two ways out. The packaged one is the AUR build `rocwmma-gfx1103`. The manual
one is a local header tree with the alias added, which keeps the system package
untouched:

```sh
git clone --depth 1 --branch rocm-7.2.0 https://github.com/ROCm/rocWMMA.git ~/rocWMMA
```

A cloned tree has no `rocwmma/rocwmma-version.hpp`; CMake generates it from
`internal/rocwmma-version.hpp.in`, so substitute the version by hand. Then, in
`rocwmma/internal/config.hpp`, immediately before the `__gfx1200__` branch:

```c
#elif defined(__gfx1103__) && ROCWMMA_DEVICE_COMPILE
#define ROCWMMA_ARCH_GFX1102 1
```

The alias is what makes the existing `gfx11` paths at `config.hpp:137` apply.

## 4. Build

Override `HIPCC`, not `ROCM_CFLAGS`. `ROCM_CFLAGS` carries
`--offload-arch=$(ROCM_ARCH)` (`Makefile:58`), so replacing it wholesale
silently makes `ROCM_ARCH` inert and pins the architecture to whatever string
you copied:

```sh
make rocm -j"$(nproc)" ROCM_ARCH=gfx1103 \
  HIPCC="hipcc --rocm-path=/opt/rocm -I$HOME/rocWMMA/library/include"
```

Drop the `-I` if you installed the AUR package. Confirm the device code really
targets the part, rather than trusting the command line:

```sh
roc-obj-ls ds4
```

Then run the oracle, which needs no model file:

```sh
make test-mxfp4-rocm
```

It ends with `MXFP4 ROCm routed MoE: PASS` after reporting `failures=0/...`
for `mid` and `out` at 1, 3, 32, 128 and 512 tokens.

## 5. GPU-visible memory under systemd-boot

STRIXHALO.md edits `/etc/default/grub`. With systemd-boot the kernel command
line comes from `LINUX_OPTIONS` in `/etc/sdboot-manage.conf`. `amdgpu.gttsize`
is in MiB and `ttm.pages_limit` is in 4 KiB pages, so the two must describe the
same size: N GiB is `N * 1024` and `N * 262144`. For 46 GiB on this machine the
line reads:

```text
LINUX_OPTIONS="zswap.enabled=0 nowatchdog splash amd_iommu=on iommu=pt amdgpu.cwsr_enable=0 amdgpu.gttsize=47104 ttm.pages_limit=12058624"
```

Keep a copy of the file first, then regenerate the boot entries:

```sh
sudo cp /etc/sdboot-manage.conf /etc/sdboot-manage.conf.bak
sudoedit /etc/sdboot-manage.conf
sudo sdboot-manage gen
```

Check the generated entry before rebooting:

```sh
sudo sh -c 'grep -H options /boot/loader/entries/*.conf'
```

The `sh -c` matters. The ESP is usually mounted `root`-only, so an unprivileged
shell cannot expand the glob: zsh refuses to run the command and bash passes
the pattern through literally, which makes `grep` complain about a file name
containing `*`. Neither failure tells you anything about the entry.

If the machine does not come back, systemd-boot can edit the entry for one
boot: press `e` at the menu and remove the two parameters. Restore with
`sudo cp /etc/sdboot-manage.conf.bak /etc/sdboot-manage.conf && sudo
sdboot-manage gen`.

After rebooting:

```sh
cat /sys/class/drm/card*/device/mem_info_gtt_total
```

46 GiB reads as `49392123904`.

Two cautions. GTT is pinned system RAM, so a `ttm.pages_limit` close to total
RAM lets the GPU starve the rest of the machine; leave real headroom. And the
BIOS UMA frame buffer is a different setting that does not need to change: it
permanently removes memory from the system, while GTT is an aperture over
ordinary RAM that is pinned on demand.

This machine keeps `amd_iommu=on iommu=pt`, unlike STRIXHALO.md, which sets
`amd_iommu=off`. Passthrough mode was enough here. `ttm.page_pool_size` was
also left at its default. Neither was needed for the oracle to pass; a full
model run may say otherwise.

## 6. What this part measures

Not from any in-tree benchmark: `tests/bench_mxfp4_rocm.c` has no make target
and reports GB/s, so these come from a standalone HIP kernel that streams a
buffer large enough to sit in GTT rather than the 4 GiB VRAM carve-out.

```c
__global__ void sum_kernel(const float4 *__restrict__ src, size_t n4, float *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < n4; i += stride) {
        float4 v = src[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5f) out[0] = acc;   /* never true: keeps the loop live */
}
```

Launched at 4096 blocks x 256 threads over an 8 GiB buffer, four repetitions:

```text
8 GiB buffer (GTT):  76.8 - 79.3 GiB/s
2 GiB buffer (VRAM): 76.3 - 77.6 GiB/s
```

77 GiB/s is 82.7 GB/s, against 89.6 GB/s theoretical for dual-channel
DDR5-5600, so about 92 percent. Reads out of GTT and out of the VRAM carve-out
are within noise of each other, so the larger aperture costs no bandwidth.

Separately, `test_mxfp4_rocm` reports its own load step: a 3.19 GiB synthetic
model reaches the device cache in 0.392 s, about 8.1 GiB/s. That is a
host-to-device staging path, not the device-side read above, and the two are
not comparable.

## 7. DeepSeek V4 Flash through SSD streaming

README.md documents SSD streaming on Metal and for GLM 5.2 on ROCm, and says
the two-host Strix Halo split "does not add ROCm SSD streaming support for
Flash". Nothing in the code refuses it:
`ds4_backend_supports_ssd_streaming` returns true for any ROCm build without
looking at the model family. On this machine Flash does run, so I read those
lines as a statement about what has been tested rather than a limit.

The credit for the idea goes to jhohertz, who reported it on the pull request
that carries this file. What I add is that the `gfx1100` substitution his
recipe uses is not required to get a run.

```text
Radeon 780M (gfx1103), Ryzen 7 8845HS, 58 GiB RAM, 4 GiB VRAM, 46 GiB GTT
CachyOS, kernel 7.2.0, ROCm 7.2.4
DeepSeek-V4-Flash-IQ2XXS-...-imatrix-0731.gguf, 80.76 GiB, on the NVMe drive
```

```sh
DS4_ROCM_STREAM_FREE_RESERVE_GB=2 ./ds4 --ssd-streaming -m ./ds4flash.gguf \
  -p "What is the capital of France? Answer with just the city name."
```

The startup accounting is what decides whether it fits:

```text
non-routed weights: 8.20 GiB
routed expert size: 6.75 MiB
total expert budget 28.60 GiB = 3.38 GiB prefill headroom
                              + 25.23 GiB dynamic cache (3827 experts)
memory: KV 0.78 + buffers 0.25 + resident model 0.99
      + expert cache 25.23 + prefill reserve 3.38 = 30.62 GiB planned
```

The 8.20 GiB of non-routed weights is the number to look at first, because that
part stays resident by construction. It leaves room on a 58 GiB machine.

`DS4_ROCM_STREAM_FREE_RESERVE_GB` defaults to 16 GiB and is clamped to 2..64.
On a 46 GiB aperture the default withholds a third of the memory from the
expert cache, so it is worth lowering.

### Native gfx1103 against a gfx1100 build

I built the same tree twice, kept the first binary aside, and alternated the
two. Decode rate on the prompt above, in t/s, with the execution order inside
each pair noted:

| | gfx1103, no override | gfx1100 + `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
|---|---|---|
| gfx1103 first | 2.19 · 2.50 · 4.48 | 3.00 · 4.18 · (run lost) |
| gfx1100 first | 2.48 · 2.41 · 2.63 | 2.84 · 3.04 · 2.72 |

The `gfx1100` build wins in both orders, so the gap is not an artifact of
position: roughly 2.9 against 2.45 t/s. I cannot explain the two 4.x readings,
one per architecture. Prefill was 0.33 t/s in every run.

The native `gfx1103` build needs no `HSA_OVERRIDE_GFX_VERSION`. It is slower,
not broken.

### Long generations collapse

"What is the capital of France?" is answered correctly by both builds, with
sensible reasoning. "Please write me a single-page HTML flappy bird game" is
not. The `gfx1100` build emits valid HTML, CSS and JavaScript and then repeats
one block of variable declarations until the context runs out. The `gfx1103`
build degrades further, through broken grammar into single letters.

That is one run per build, so I would not read the difference in severity as
real. Nothing here tells a 2-bit quantization effect apart from a sampling one,
and I have no second AMD part to check against.

## 8. Limits

Memory is the binding constraint, not compute. STRIXHALO.md sizes the DeepSeek
V4 Flash IQ2XXS GGUF at 80.76 GiB and README.md sizes GLM 5.2 IQ2_XXS at
188 GiB, so a 64 GB machine holds neither resident. Streaming makes Flash run
at 2.5 to 3 t/s, which serves short answers and not code generation. GLM 5.2
was not tried.
