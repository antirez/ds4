#!/usr/bin/env bash
# test_ornith15_bench.sh — correctness + smoke harness for Ornith-1.5-35B-A3B GGUF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${DS4_TEST_MODEL:-}"
LOG="/tmp/ornith15_test_$$.log"
LLAMA_LOG="/tmp/ornith15_llama_$$.log"
PROMPT="The capital of France is"
GEN_TOKENS=8

fail(){ echo "FAIL: $*" >&2; exit 1; }
ok(){ echo "ok: $*"; }

trap 'rm -f "$LOG" "$LLAMA_LOG"' EXIT

# 1. Discover model path or default candidates
if [[ -z "$MODEL" ]]; then
  for candidate in \
    "$ROOT/ornith35.gguf" \
    "$ROOT/gguf/Ornith-1.5-35B-Q4_K_M.gguf" \
    "$ROOT/Ornith-1.5-35B-Q4_K_M.gguf" \
    "$ROOT/ornith.gguf"; do
    if [[ -f "$candidate" ]]; then
      MODEL="$candidate"
      break
    fi
  done
fi

# Fallback: check if ds4flash.gguf is Ornith
if [[ -z "$MODEL" && -f "$ROOT/ds4flash.gguf" ]]; then
  if python3 -c '
import sys, struct
with open(sys.argv[1], "rb") as f:
    if f.read(4) != b"GGUF": sys.exit(1)
    _ = f.read(4) # ver
    _ = f.read(8) # nt
    _ = struct.unpack("<Q", f.read(8))[0] # kvc
    data = f.read(16384)
    if b"qwen35moe" in data or b"Ornith" in data:
        sys.exit(0)
    sys.exit(1)
' "$ROOT/ds4flash.gguf" 2>/dev/null; then
    MODEL="$ROOT/ds4flash.gguf"
  fi
fi

# 2. Clean skip if model is absent
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "skip: Ornith 1.5 35B model not found." >&2
  echo "      Run './download_model.sh ornith-q4' to download (~20.2 GiB) or set DS4_TEST_MODEL=PATH" >&2
  if [[ -x "$ROOT/ds4-bench" ]]; then
    "$ROOT/ds4-bench" --help | grep -q "model" || fail "ds4-bench help check failed"
    ok "ds4-bench binary help ok"
  fi
  if [[ -x "$ROOT/ds4" ]]; then
    "$ROOT/ds4" --help | grep -q "model" || fail "ds4 help check failed"
    ok "ds4 binary help ok"
  fi
  ok "Ornith 1.5 harness skipped cleanly (model absent)"
  exit 0
fi

ok "Found Ornith model: $MODEL"

# 3. Verify GGUF header & metadata strictly via Python
python3 - "$MODEL" << 'PY'
import sys, struct

path = sys.argv[1]
with open(path, "rb") as f:
    magic = f.read(4)
    assert magic == b'GGUF', f"Invalid magic: {magic}"
    ver = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    kv_count = struct.unpack('<Q', f.read(8))[0]
    assert ver >= 2, f"Unexpected GGUF version {ver}"
    assert tensor_count > 0, "No tensors found in GGUF"
    assert kv_count > 0, "No metadata KV pairs found in GGUF"

    # Peek metadata block for qwen35moe architecture
    data = f.read(65536)
    assert b'qwen35moe' in data, "Architecture mismatch: expected qwen35moe"
    print(f"ok GGUF container verified: ver={ver}, tensors={tensor_count}, metadata_kvs={kv_count}, arch=qwen35moe")
PY
ok "GGUF container and qwen35moe architecture strictly verified"

# 4. Ensure ds4 binary is built (always execute in repo root)
cd "$ROOT"
if [[ ! -x "$ROOT/ds4" ]]; then
  make -C "$ROOT" ds4 -j4 >/dev/null 2>&1 || make -C "$ROOT" ds4 >/dev/null || fail "Failed to build ds4"
fi

# 5. Strict ds4 --inspect test
echo "Running ds4 --inspect on $MODEL ..."
"$ROOT/ds4" -m "$MODEL" --inspect > "$LOG" 2>&1 || {
  cat "$LOG" >&2
  fail "ds4 --inspect failed on $MODEL"
}
cat "$LOG"
ok "ds4 --inspect succeeded"

# 6. Deterministic CLI smoke test (short generation suitable for 20GB model)
echo "Running deterministic CLI smoke on $MODEL ..."
"$ROOT/ds4" -m "$MODEL" -p "$PROMPT" -n "$GEN_TOKENS" --temp 0.0 > "$LOG" 2>&1 || {
  cat "$LOG" >&2
  fail "ds4 Metal CLI execution failed on $MODEL"
}
cat "$LOG"
ok "ds4 Metal CLI smoke generation succeeded"

# 7. Deterministic observable contract: exact match against reference llama-cli when present
if command -v llama-cli >/dev/null 2>&1; then
  echo "Comparing deterministic output against reference llama-cli ..."
  python3 - "$ROOT" "$MODEL" "$LOG" "$LLAMA_LOG" "$PROMPT" "$GEN_TOKENS" << 'PY'
import sys, os, subprocess, re

root, model, ds4_log_path, llama_log_path, prompt, gen_tokens_str = sys.argv[1:7]
gen_tokens = int(gen_tokens_str)

def clean_text(text):
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    # Strip ds4: diagnostic banners and inline prefill footers
    text = re.sub(r'ds4:.*', '', text)
    lines = []
    for l in text.splitlines():
        l = l.strip().lstrip('>').lstrip('|').strip()
        if l and not l.startswith("objc[") and not l.startswith("!!") and not l.startswith("available commands:") and not l.startswith("/"):
            lines.append(l)
    return "\n".join(lines).strip()

with open(ds4_log_path, "r", errors="replace") as f:
    ds4_raw = f.read()
ds4_clean = clean_text(ds4_raw)

llama_cmd = [
    "llama-cli", "-m", model, "-p", prompt, "-n", str(gen_tokens),
    "--temp", "0.0", "--top-k", "1", "--top-p", "1.0", "-ngl", "999", "--no-warmup"
]
try:
    p = subprocess.Popen(llama_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    llama_out, _ = p.communicate(input="\n/exit\n", timeout=30)
    with open(llama_log_path, "w") as f:
        f.write(llama_out)
except Exception as e:
    print(f"FAIL: llama-cli execution failed: {e}", file=sys.stderr)
    sys.exit(1)

llama_clean = ""
if prompt in llama_out:
    after_prompt = llama_out.split(prompt, 1)[1]
    if "[ Prompt:" in after_prompt:
        after_prompt = after_prompt.split("[ Prompt:")[0]
    elif "///exit" in after_prompt:
        after_prompt = after_prompt.split("///exit")[0]
    llama_clean = clean_text(after_prompt)
else:
    llama_clean = clean_text(llama_out)

print(f"  ds4 output:       {repr(ds4_clean)}")
print(f"  llama-cli output: {repr(llama_clean)}")

if ds4_clean != llama_clean:
    print(f"FAIL: generated text mismatch between ds4 and reference llama-cli!", file=sys.stderr)
    print(f"  ds4:   {ds4_clean}", file=sys.stderr)
    print(f"  llama: {llama_clean}", file=sys.stderr)
    sys.exit(1)

print("ok deterministic output exact match verified against reference llama-cli")
PY
  ok "llama-cli deterministic contract verified"
else
  echo "skip: llama-cli not found on PATH; skipping reference comparison"
fi

ok "Ornith 1.5 test harness complete"
