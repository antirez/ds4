#!/bin/sh
# Session self-test in default mode (variable unset) and EXACT mode.  usage: selftest.sh [GGUF]   (default $IK_MODEL)
# Never run the default case as DS4_INKLING2_EXACT=0: unset, "", 0, false, off and no all mean the same (default kernels).
cd "$(dirname "$0")/../.." || exit 1
M=${1:-${IK_MODEL:?give a GGUF path or set IK_MODEL}}
for ex in unset 1; do
  echo "== EXACT=$ex"
  if [ "$ex" = unset ]; then unset DS4_INKLING2_EXACT; else export DS4_INKLING2_EXACT=$ex; fi
  DS4_LOCK_FILE=${DS4_LOCK_FILE:-/tmp/ds4-inkling.lock} DS4_INKLING2_SESSION_TEST=1 DS4_INKLING2_FT_TOKENS=16,11,220,17,11,220,18,11,220,19,11,220,20,11 \
    ./ds4 -m "$M" --metal --first-token-test --raw-prompt -p x -t 8 2>&1 | grep -E "FAIL|checks|known"
done
