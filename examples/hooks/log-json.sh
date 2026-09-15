#!/bin/sh
set -eu
log=${DS4_HOOK_LOG:-./ds4-agent-hooks.jsonl}
cat >> "$log"
