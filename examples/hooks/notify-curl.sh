#!/bin/sh
set -eu
: "${DS4_HOOK_URL:?set DS4_HOOK_URL}"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary @- \
  "$DS4_HOOK_URL"
