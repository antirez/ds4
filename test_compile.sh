#!/bin/bash
set -e
cd /c/Users/wren/git/ds4

FLAGS="-O0 -std=gnu99 -D_GNU_SOURCE -DDS4_NO_GPU -DDS4_ROCM_BUILD -I./compat/win32"
PASS=0
FAIL=0

for f in ds4.c ds4_distributed.c ds4_server.c ds4_cli.c ds4_bench.c ds4_eval.c ds4_ssd.c ds4_kvstore.c ds4_help.c ds4_web.c rax.c linenoise.c; do
  if gcc $FLAGS -c "$f" -o /tmp/${f%.c}.o 2>&1; then
    echo "✓ $f"
    ((PASS++))
  else
    echo "✗ $f"
    ((FAIL++))
  fi
done

echo ""
echo "RESULTS: $PASS passed, $FAIL failed"
exit $FAIL
