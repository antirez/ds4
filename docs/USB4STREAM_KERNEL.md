# USB4STREAM: tested interrupt fix

- Needed on the tested Strix Halo pair with Linux `7.2.5-100.fc43.x86_64`: stock native streams stalled under sustained traffic. TCP and RoCE are unaffected by this prerequisite.
- [Patch](patches/thunderbolt-msix-readback-v7.2.5.patch): read back the MSI-X status register after clearing the interrupt, flushing the posted write. Adapted from [Jonathan Yates's patch11](https://github.com/jyatesdotdev/strix-rdma/blob/d19af99ce91abda691a2bd0f21eb114e0a64bacd/kernel/zerocopy/0011-thunderbolt-Flush-posted-MSI-X-interrupt-clears.patch); no RX-prime patch.
- This is a tested workaround, not an upstream fix or a guarantee for every USB4 controller. The original rebuilt module matched the stock driver's executable sections before testing the patched module.
- Hot-loading requires Thunderbolt built as a module; a built-in driver needs a kernel containing the fix.
- Use the running distribution kernel's matching source/configuration/headers and compiler. Never reuse another kernel's `.ko`. Secure Boot may require signing.

## Build a separate module

Run in a disposable copy of the matching kernel source; `DS4_SRC` is this engine checkout. The build does not install anything.

```bash
DS4_SRC=/absolute/path/to/ds4
KERNEL_SRC=/absolute/path/to/matching-kernel-source
KVER=$(uname -r)
cd "$KERNEL_SRC"
patch --dry-run -p1 < "$DS4_SRC/docs/patches/thunderbolt-msix-readback-v7.2.5.patch"
patch -p1 < "$DS4_SRC/docs/patches/thunderbolt-msix-readback-v7.2.5.patch"
make -C "/lib/modules/$KVER/build" M="$KERNEL_SRC/drivers/thunderbolt" modules
PATCHED_KO="$KERNEL_SRC/drivers/thunderbolt/thunderbolt.ko"
modinfo -F vermagic "$PATCHED_KO"    # Must match the running kernel
sha256sum "$PATCHED_KO"
```

## Temporary load, without reboot

- Stop both inference peers and close every `/dev/tbstream*` user. Remove only your own ConfigFS stream directories, then their empty service directory.
- Use local access or management Ethernet independent of USB4: unloading Thunderbolt disconnects its networking and devices.
- Save the loaded modules and their parameters before unloading. The tested dependency order was `ucsi_acpi`, `typec_thunderbolt` (if loaded), `typec_ucsi`, `typec`, `thunderbolt_net`, `thunderbolt_stream`, `thunderbolt`. Other hardware can have different holders; never force an unload.

```bash
# Keep this terminal open through the trial and rollback.
STATE=$(mktemp -d)
ORDER=(ucsi_acpi typec_thunderbolt typec_ucsi typec thunderbolt_net thunderbolt_stream thunderbolt)
LOADED=()
for mod in "${ORDER[@]}"; do
  [ -d "/sys/module/$mod" ] || continue
  LOADED+=("$mod")
  : > "$STATE/$mod.args"
  for file in /sys/module/"$mod"/parameters/*; do
    [ -f "$file" ] && printf '%s=%s\0' "${file##*/}" "$(cat "$file")" >> "$STATE/$mod.args"
  done
done
# Inspect holders first. Stop if a holder is outside the recorded set.
for mod in "${LOADED[@]}"; do ls "/sys/module/$mod/holders"; done

unload_trial_modules() {
  for mod in "${LOADED[@]}"; do sudo rmmod "$mod" || return; done
}
reload_dependents() {
  local mod
  local -a args
  for ((i=${#LOADED[@]}-1; i>=0; i--)); do
    mod=${LOADED[i]}
    [ "$mod" = thunderbolt ] && continue
    mapfile -d '' -t args < "$STATE/$mod.args"
    sudo modprobe "$mod" "${args[@]}" || return
  done
}
mapfile -d '' -t TB_PARAMS < "$STATE/thunderbolt.args"
unload_trial_modules && sudo insmod "$PATCHED_KO" "${TB_PARAMS[@]}" && reload_dependents
od -An -tx1 -v /sys/module/thunderbolt/notes/.note.gnu.build-id
```

- If any unload/load fails, stop; do not force it. Use the rollback block to reload missing modules, then inspect the reported holder or signature error.
- Recreate the streams using [the setup guide](CLUSTERING_ROCM.md#usb4stream) after **both** hosts reload. Rediscover `key=stream`; service numbers can change.
- The tested temporary module had build ID `1e7c7ac37802197aa0c2e3dbd8e769cf9511a6b5`; a different build need not have that ID. Verify yours against its ELF build ID.
- No `modules_install`, `depmod`, boot entry or reboot is needed for this trial. The stock on-disk module remains available.

## Roll back

- Stop the peers, remove their streams and safely unload the same dependency set.
- Replace `insmod` with the following, preserving the saved parameters; reload the previous dependents and recreate streams:

```bash
# If the candidate is still loaded, first run unload_trial_modules successfully.
sudo modprobe thunderbolt "${TB_PARAMS[@]}" && reload_dependents
```

- A reboot also discards the temporary replacement. On affected stock drivers, use TCP or RoCE until the native-stream stall is resolved.
