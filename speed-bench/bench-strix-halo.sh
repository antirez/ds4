#!/usr/bin/env bash
# bench-strix-halo.sh — reproducible DwarfStar (ds4) benchmark for AMD Strix Halo / gfx1151 (ROCm).
#
# Produces the STANDARD context sweep in speed-bench's own CSV format
# (columns: ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,
#           gen_steady_tokens,gen_steady_tps,kvcache_bytes)
# plus a .meta.json capturing the full environment, and a ds4-eval capability run.
#
# RUN ONLY WHEN THE BOX IS OTHERWISE IDLE. A co-resident process (another model server,
# ComfyUI, ollama, or even a large browser) changes the load/memory numbers this exists to
# record. The script asserts idleness and refuses to run otherwise (override: --allow-busy).
#
# It never touches a production service: it only runs ds4-bench / ds4-eval from THIS checkout.
#
# Usage:
#   speed-bench/bench-strix-halo.sh [--out BASE] [--model GGUF] [--ctx-max N]
#                                   [--gen-tokens N] [--eval-suite NAME] [--skip-eval]
#                                   [--allow-busy] [--dry-run]
#
# The eval defaults to the FAST 'hard-smoke' suite. 'core' is a 92-case capability run
# that can take HOURS (16k-token generation budget per case; measured >5.7 h on a Strix
# Halo for 66/92 cases) — ask for it explicitly with --eval-suite core.
#
# Example (the delivery command for a Strix Halo test night):
#   ./bench-strix-halo.sh --model /home/tim/ds4/gguf/ds4flash.gguf
set -euo pipefail

OUT_BASE="speed-bench/strix_halo"
MODEL="ds4flash.gguf"
CTX_START=2048
CTX_MAX=65536
STEP_INCR=2048
GEN_TOKENS=128
ALLOW_BUSY=0
SKIP_EVAL=0
EVAL_SUITE="hard-smoke"
DRY_RUN=0

usage() {
  sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT_BASE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --ctx-max) CTX_MAX="$2"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="$2"; shift 2 ;;
    --eval-suite) EVAL_SUITE="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --allow-busy) ALLOW_BUSY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "bench-strix-halo: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUT_BASE")"

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die()  { log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------- prerequisites
[ -d speed-bench ] || die "run from the repository root (speed-bench/ not found)."
[ -f speed-bench/promessi_sposi.txt ] || die "speed-bench/promessi_sposi.txt missing."
if [ "$DRY_RUN" -eq 0 ]; then
  [ -x ./ds4-bench ] || die "ds4-bench not built here. Run 'make strix-halo' first."
  [ -x ./ds4-eval ]  || die "ds4-eval not built here."
  if [ "$MODEL" = "ds4flash.gguf" ] && [ ! -e ./ds4flash.gguf ]; then
    die "no ./ds4flash.gguf — pass --model /path/to/DeepSeek-V4-Flash-...-0731.gguf"
  fi
fi

# ---------------------------------------------------------------- idle assertion
check_idle() {
  local busy=0
  if command -v rocm-smi >/dev/null 2>&1; then
    local pids
    pids="$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+\t/ {print $1"\t"$2}')"
    if [ -n "$pids" ]; then
      log "  ! GPU processes are running:"; printf '      %s\n' "$pids" >&2; busy=1
    fi
  fi
  # Co-tenants that would taint a benchmark. Match process NAMES exactly for the known servers:
  # cmdline matching false-positives on e.g. `less ds4-server.log` (its argv contains the string).
  for name in ds4-server ds4 llama-server ollama vllm; do
    if pgrep -x -- "$name" >/dev/null 2>&1; then
      log "  ! process '$name' is running"; busy=1
    fi
  done
  # Python-based tenants: match the cmdline, but ignore pagers/editors reading a log.
  local extra
  extra="$(pgrep -af -- 'ComfyUI|comfyui' 2>/dev/null \
    | grep -viE '(^|[[:space:]])(less|more|tail|head|cat|grep|rg|vim|nano|emacs|code|bat|journalctl)([[:space:]]|$)' || true)"
  if [ -n "$extra" ]; then
    log "  ! python tenant running:"; printf '      %s\n' "$extra" >&2; busy=1
  fi
  local avail_mb
  avail_mb="$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)"
  if [ "${avail_mb:-0}" -lt 40000 ]; then
    log "  ! MemAvailable is only ${avail_mb} MiB (< 40 GiB idle expectation)"; busy=1
  fi
  if [ "$busy" -eq 1 ]; then
    [ "$ALLOW_BUSY" -eq 1 ] && { log "WARNING: box is not idle (--allow-busy given; numbers will be tainted)"; return 0; }
    die "box is not idle — stop all GPU tenants and re-run (or pass --allow-busy to override)."
  fi
  log "idle check passed"
}

# ---------------------------------------------------------------- env capture
gtt_field() {  # $1 = rocm-smi label; prints the byte count (last numeric field), or nothing
  command -v rocm-smi >/dev/null 2>&1 || { printf ''; return 0; }
  rocm-smi --showmeminfo gtt 2>/dev/null \
    | awk -v k="$1" 'index($0,k){ for (i=NF; i>=1; i--) if ($i ~ /^[0-9]+$/) { print $i; exit } }' || true
}
gtt_used_b()   { gtt_field "GTT Total Used"; }
gtt_total_b()  { gtt_field "GTT Total Memory"; }

write_meta() {
  local git_head git_branch git_dirty bin_sha
  git_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  git_dirty="$(git status --porcelain 2>/dev/null | head -1 | grep -q . && echo dirty || echo clean)"
  bin_sha="$(sha256sum ./ds4-bench 2>/dev/null | awk '{print $1}' || true)"

  META_OUT="$OUT_BASE.meta.json" \
  CAP_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  CAP_HOST="$(hostname)" \
  CAP_VENDOR="$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null)" \
  CAP_PRODUCT="$(cat /sys/class/dmi/id/product_name 2>/dev/null)" \
  CAP_BOARD="$(cat /sys/class/dmi/id/board_name 2>/dev/null)" \
  CAP_CPU="$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)" \
  CAP_OS="$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")" \
  CAP_KERNEL="$(uname -r)" \
  CAP_CMDLINE="$(cat /proc/cmdline 2>/dev/null)" \
  CAP_ROCM="$(cat /opt/rocm/.info/version 2>/dev/null || dpkg-query -W -f='${Version}' rocm-core 2>/dev/null || echo unknown)" \
  CAP_GFX="$( { rocminfo 2>/dev/null || true; } | grep -oE 'gfx[0-9]+' | sort -u | paste -sd, - || true)" \
  CAP_GTT_TOTAL="$(gtt_total_b)" \
  CAP_GTT_USED="$(gtt_used_b)" \
  CAP_MEM="$(free -m | awk '/^Mem:/{print $2}')" \
  CAP_HEAD="$git_head" CAP_BRANCH="$git_branch" CAP_DIRTY="$git_dirty" CAP_BIN_SHA="$bin_sha" \
  CAP_MODEL="$MODEL" CAP_CTX_MAX="$CTX_MAX" CAP_GEN_TOKENS="$GEN_TOKENS" \
  python3 - <<'PY'
import json, os
keys = {
  "captured_at":"CAP_TS","hostname":"CAP_HOST","sys_vendor":"CAP_VENDOR","product":"CAP_PRODUCT",
  "board":"CAP_BOARD","cpu":"CAP_CPU","os":"CAP_OS","kernel":"CAP_KERNEL","boot_cmdline":"CAP_CMDLINE",
  "rocm_version":"CAP_ROCM","gfx":"CAP_GFX","gtt_total_bytes":"CAP_GTT_TOTAL","gtt_used_bytes":"CAP_GTT_USED",
  "mem_total_mb":"CAP_MEM","git_head":"CAP_HEAD","git_branch":"CAP_BRANCH","git_tree":"CAP_DIRTY",
  "ds4_bench_sha256":"CAP_BIN_SHA","model":"CAP_MODEL","ctx_max":"CAP_CTX_MAX",
  "gen_tokens":"CAP_GEN_TOKENS",
}
out = {v: os.environ.get(k, "") for v, k in keys.items()}
for k in ("ctx_max", "gen_tokens"):
    try:
        out[k] = int(out[k])
    except (TypeError, ValueError):
        out[k] = 0
print(json.dumps(out, indent=2))
PY
  # (json.dumps to stdout; caller redirects)
}

# ---------------------------------------------------------------- run
log "ds4 Strix Halo benchmark — repo=$REPO_ROOT commit=$(git rev-parse --short HEAD 2>/dev/null)"
check_idle
write_meta > "$OUT_BASE.meta.json"
log "environment captured -> $OUT_BASE.meta.json"

if [ "$DRY_RUN" -eq 1 ]; then
  log "--dry-run: skipping ds4-bench / ds4-eval"
  log "meta written; re-run without --dry-run on test night."
  exit 0
fi

log "running ds4-bench sweep (ctx $CTX_START..$CTX_MAX, step $STEP_INCR, gen $GEN_TOKENS)…"
BENCH_LOG="$OUT_BASE.bench.log"
./ds4-bench -m "$MODEL" --rocm \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start "$CTX_START" --ctx-max "$CTX_MAX" --step-incr "$STEP_INCR" \
  --gen-tokens "$GEN_TOKENS" 2>&1 | tee "$BENCH_LOG" >/dev/null

# Extract the CSV block (header line + rows) into the canonical file.
awk '/^ctx_tokens,/ {print; grab=1; next} grab && /^[0-9]/ {print}' "$BENCH_LOG" > "$OUT_BASE.csv"
[ -s "$OUT_BASE.csv" ] || die "no CSV block found in $BENCH_LOG — check the log"
log "sweep CSV -> $OUT_BASE.csv ($(wc -l < "$OUT_BASE.csv") lines)"

if [ "$SKIP_EVAL" -eq 0 ]; then
  log "running ds4-eval (suite $EVAL_SUITE)…"
  ./ds4-eval -m "$MODEL" --rocm --plain --suite "$EVAL_SUITE" > "$OUT_BASE.eval.txt" 2>&1 || true
  # Tally per-case outcomes so the run reports itself instead of needing a manual grep.
  awk '/^[[:space:]]*[0-9]+ (PASSED|FAILED|INCOMPLETE)/{c[$2]++; t++}
       END{ if (t) { printf "    eval outcomes: ";
                      for (k in c) printf "%s=%d ", k, c[k];
                      printf "total=%d\n", t } }' "$OUT_BASE.eval.txt" >&2
  log "eval -> $OUT_BASE.eval.txt"
fi

log "done. Summary:"
awk -F, 'NR==1{next} {printf "    ctx=%-6s prefill=%-8s t/s  gen=%-6s t/s\n", $1,$3,$5}' "$OUT_BASE.csv" >&2
