# DS4 on AMD GPUs other than Strix Halo

[STRIXHALO.md](STRIXHALO.md) covers Ubuntu, GRUB and `gfx1151` with 128 GB of
RAM. This file covers what changes on an Arch-family distribution with
systemd-boot and a smaller RDNA3 part. The reference machine is a Radeon 780M
(`gfx1103`, Ryzen 7 8845HS) with 58 GiB of usable system RAM, ROCm 7.2.4 and
kernel 7.2.0.

What was verified here: `make rocm` builds, `make test-mxfp4-rocm` passes, and
`make cpu` still builds. What was not: `make test`, which needs a model file.
No supported model has been run end to end on this machine, because none of the
published GGUFs fit in 58 GiB resident. The build carries one pre-existing
warning in `ds4_server.c`, unrelated to anything described here.

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

## 7. Limits

Memory is the binding constraint, not compute. STRIXHALO.md sizes the DeepSeek
V4 Flash IQ2XXS GGUF at 80.76 GiB and README.md sizes GLM 5.2 IQ2_XXS at
188 GiB, so a 64 GB machine holds neither resident. SSD streaming is the only
route, and README.md states it is not supported for Flash on ROCm.
