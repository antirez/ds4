# DS4 on Strix Halo

This is the minimal setup for DS4 ROCm inference on a
Strix Halo machine with 128 GB RAM and Radeon 8060S (`gfx1151`).

## 1. Install ROCm

On Ubuntu 26.04 LTS, install the ROCm compiler/runtime and libraries used by the Strix Halo backend:

```sh
sudo apt-get update
sudo apt-get install -y \
  hipcc rocminfo rocm-smi \
  libamdhip64-dev \
  libhipblas-dev libhipblaslt-dev \
  librocblas-dev \
  librocwmma-dev \
  libhipcub-dev
```

The backend uses rocWMMA. On this Ubuntu 26.04 setup, `librocwmma-dev`
installs the top-level rocWMMA headers but misses `rocwmma/internal/`.
No Ubuntu package currently provides those internal headers. Install a complete
matching rocWMMA header tree:

```sh
git clone --depth 1 --branch rocm-7.1.0 https://github.com/ROCm/rocWMMA.git /tmp/rocWMMA-rocm-7.1.0
sudo mkdir -p /usr/local/include
sudo cp -a /tmp/rocWMMA-rocm-7.1.0/library/include/rocwmma /usr/local/include/
```

If ROCm is installed under `/usr` but tooling expects `/opt/rocm`, add these
compatibility links:

```sh
sudo mkdir -p /opt/rocm/bin
sudo ln -sf /usr/bin/hipcc /opt/rocm/bin/hipcc
sudo ln -sfn /usr/lib/x86_64-linux-gnu /opt/rocm/lib
sudo ln -sfn /usr/include /opt/rocm/include
```

## 2. Enable ROCm access

The user running DS4 must be able to open `/dev/kfd` and the DRM render node:

```sh
sudo usermod -aG render,video "$USER"
```

Log out and back in, or reboot. Verify:

```sh
rocminfo | grep -A80 'Name:                    gfx1151'
```

If DS4 says `no ROCm-capable device is detected`, check that `rocminfo` can open
`/dev/kfd` and that `groups` includes `render`.

## 3. Increase GPU-visible memory

A 128 GB Strix Halo system may initially expose only about 62 GB of GPU-visible
memory. DS4 needs the larger GTT aperture for the 80.76 GiB model plus runtime
buffers.

Use these kernel parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856
```

On Ubuntu with GRUB:

```sh
sudo cp /etc/default/grub /etc/default/grub.bak
sudoedit /etc/default/grub
```

Set:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 ttm.page_pool_size=32505856"
```

Then:

```sh
sudo update-grub
sudo reboot
```

After reboot, verify:

```sh
cat /proc/cmdline
sudo dmesg | grep -Ei 'GTT|gttsize|TTM|VRAM'
rocminfo | grep -A80 'Name:                    gfx1151'
```

Expected signs:

```text
amdgpu:  126976M of GTT memory ready
rocminfo gfx1151 pool: 130023424 KB
```

## 4. Build DS4

Use the normal Strix Halo target. It builds the standard binary names:

```sh
make strix-halo -j"$(nproc)"
```

`make rocm` is an alias for `make strix-halo`.

## 5. Use the right GGUF

Use the standard IQ2XXS/Q2K/Q8 imatrix GGUF:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Avoid the mixed IQ2/IQ4 or IQ2/Q4 GGUFs on this machine for now. They put much
more memory pressure on the ROCm path and can trigger system OOM instead of a
clean DS4 failure.

## 6. Run DS4

Run it normally:

```sh
./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

The ROCm build uses the Strix Halo backend automatically.
