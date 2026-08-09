#!/usr/bin/env bash
# tests/test_rocm_make.sh — smoke tests for ROCm Makefile targets and
# architecture auto-detection.  Runs from the repo root.
set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0
LOG=$(mktemp)

ok()   { PASS=$((PASS+1)); echo "ok $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1"; }

# -------------------------------------------------------------------
# 1: make help lists rocm, rx9070, and strix-halo targets.
# -------------------------------------------------------------------
make help 2>&1 | tee "$LOG" > /dev/null
for target in rocm rx9070 strix-halo; do
    if grep -q "make $target" "$LOG"; then
        ok "help lists 'make $target'"
    else
        fail "help missing 'make $target'"
    fi
done

# -------------------------------------------------------------------
# 2: make help rocm description mentions auto-detect / RDNA.
# -------------------------------------------------------------------
if grep -q "Auto-detect\|auto-detect\|RDNA" "$LOG"; then
    ok "help rocm description mentions auto-detect / RDNA"
else
    fail "help rocm description should mention auto-detect / RDNA"
fi

# -------------------------------------------------------------------
# 3: rocminfo detection — if rocminfo is available, verify it
#    produces a gfxNNN pattern.
# -------------------------------------------------------------------
if command -v rocminfo >/dev/null 2>&1; then
    detected=$(rocminfo 2>/dev/null | grep -oP 'gfx\d{4}' | head -1)
    if [ -n "$detected" ]; then
        ok "rocminfo detects $detected"
    else
        fail "rocminfo found but no gfxNNN detected"
    fi
else
    ok "rocminfo not installed — skipping detection test"
fi

# -------------------------------------------------------------------
# 4: make rocm builds successfully (smoke test).
#    We do a full clean + build to verify the entire ROCm pipeline.
# -------------------------------------------------------------------
make clean >/dev/null 2>&1
if make rocm > "$LOG" 2>&1; then
    ok "make rocm builds successfully"
else
    fail "make rocm failed to build"
    head -20 "$LOG" | sed 's/^/    /'
fi

# -------------------------------------------------------------------
# 5: all expected ROCm binaries exist and are executable.
# -------------------------------------------------------------------
for bin in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
    if [ -x "./$bin" ]; then
        ok "$bin is built and executable"
    else
        fail "$bin not found or not executable"
    fi
done

# -------------------------------------------------------------------
# 6: built binaries print help without crashing.
# -------------------------------------------------------------------
for bin in ds4 ds4-server ds4-bench ds4-agent; do
    if [ ! -x "./$bin" ]; then
        fail "$bin not available for help test"
        continue
    fi
    if ./"$bin" --help >/dev/null 2>&1; then
        ok "$bin --help exits cleanly"
    else
        fail "$bin --help crashed"
    fi
done

# -------------------------------------------------------------------
# 7: built binaries mention ROCm / HIP in their help output.
# -------------------------------------------------------------------
if [ -x ./ds4 ]; then
    ./ds4 --help 2>&1 | tee "$LOG" > /dev/null
    if grep -qiE "roc|hip|amd|gpu" "$LOG"; then
        ok "ds4 --help mentions ROCm/HIP/AMD/GPU"
    else
        fail "ds4 --help should mention ROCm/HIP/AMD/GPU"
    fi
fi

# -------------------------------------------------------------------
# 8: verify that ROCm binaries are linked against hipblas (not cublas).
# -------------------------------------------------------------------
if [ -x ./ds4 ]; then
    libs=$(ldd ./ds4 2>/dev/null | grep -o 'libhipblas' || true)
    if [ -n "$libs" ]; then
        ok "ds4 is linked against libhipblas (ROCm build)"
    else
        # Check for hipblaslt as well
        libs=$(ldd ./ds4 2>/dev/null | grep -o 'hipblas' || true)
        if [ -n "$libs" ]; then
            ok "ds4 is linked against hipblas (ROCm build)"
        else
            fail "ds4 should be linked against hipblas for ROCm build"
        fi
    fi
fi

# -------------------------------------------------------------------
# 9: make rocm sets ROCM_ARCH correctly when rocminfo is available.
#    We verify by running make rocm with ROCM_ARCH override and
#    checking the build uses it.
# -------------------------------------------------------------------
if command -v rocminfo >/dev/null 2>&1; then
    expected_arch=$(rocminfo 2>/dev/null | grep -oP 'gfx\d{4}' | head -1)
    if [ -n "$expected_arch" ]; then
        make clean >/dev/null 2>&1
        if make rocm 2>&1 | grep -q "offload-arch=$expected_arch"; then
            ok "make rocm uses auto-detected arch $expected_arch"
        else
            # Also check the log for the arch
            make rocm > "$LOG" 2>&1
            if grep -q "offload-arch=$expected_arch" "$LOG"; then
                ok "make rocm uses auto-detected arch $expected_arch"
            else
                fail "make rocm should use auto-detected arch $expected_arch"
            fi
        fi
    fi
else
    ok "rocminfo not available — skipping arch detection verification"
fi

# -------------------------------------------------------------------
# 10: make rx9070 builds with explicit gfx1201.
# -------------------------------------------------------------------
make clean >/dev/null 2>&1
if make rx9070 > "$LOG" 2>&1; then
    ok "make rx9070 builds successfully"
else
    fail "make rx9070 failed to build"
    head -20 "$LOG" | sed 's/^/    /'
fi

# Verify gfx1201 was used
if grep -q "offload-arch=gfx1201" "$LOG"; then
    ok "make rx9070 uses --offload-arch=gfx1201"
else
    fail "make rx9070 should use --offload-arch=gfx1201"
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
rm -f "$LOG"

echo ""
echo "test_rocm_make: PASS=$PASS FAIL=$FAIL"
if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
