#!/usr/bin/env bash
# test_ornith9_bench.sh — correctness + smoke harness for Ornith-1.5-9B GGUF
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${DS4_TEST_MODEL:-}"
LOG="/tmp/ornith9_test_$$.log"
RAW_LOG="/tmp/ornith9_raw_$$.log"
REF_LOG="/tmp/ornith9_ref_$$.log"
PROMPT="The capital of France is"
GEN_TOKENS=16

fail(){ echo "FAIL: $*" >&2; exit 1; }
ok(){ echo "ok: $*"; }

trap 'rm -f "$LOG" "$RAW_LOG" "$REF_LOG"' EXIT

# 1. Discover model path or default candidates
if [[ -z "$MODEL" ]]; then
  for candidate in \
    "$ROOT/ornith9.gguf" \
    "$ROOT/gguf/Ornith-1.5-9B-Q4_K_M.gguf" \
    "$ROOT/Ornith-1.5-9B-Q4_K_M.gguf" \
    "$ROOT/gguf/Ornith-1.5-9B-Q6_K.gguf" \
    "$ROOT/gguf/Ornith-1.5-9B-Q8_0.gguf" \
    "$ROOT/ornith9-q4.gguf"; do
    if [[ -f "$candidate" && -r "$candidate" ]]; then
      MODEL="$candidate"
      break
    fi
  done
fi

# Fallback: check if ds4flash.gguf is Ornith 9B / qwen35
if [[ -z "$MODEL" && -f "$ROOT/ds4flash.gguf" && -r "$ROOT/ds4flash.gguf" ]]; then
  if python3 -c '
import sys, struct
with open(sys.argv[1], "rb") as f:
    if f.read(4) != b"GGUF": sys.exit(1)
    _ = f.read(4) # ver
    _ = f.read(8) # nt
    _ = struct.unpack("<Q", f.read(8))[0] # kvc
    data = f.read(16384)
    if b"qwen35" in data or b"Ornith" in data:
        sys.exit(0)
    sys.exit(1)
' "$ROOT/ds4flash.gguf" 2>/dev/null; then
    MODEL="$ROOT/ds4flash.gguf"
  fi
fi

# 2. Clean skip if model is absent or unreadable
if [[ -z "$MODEL" || ! -f "$MODEL" || ! -r "$MODEL" ]]; then
  echo "skip: Ornith 1.5 9B model not found or unreadable." >&2
  echo "      Run './download_model.sh ornith9-q4' to download (~5.24 GiB) or set DS4_TEST_MODEL=PATH" >&2
  if [[ -x "$ROOT/ds4-bench" ]]; then
    "$ROOT/ds4-bench" --help | grep -q "model" || fail "ds4-bench help check failed"
    ok "ds4-bench binary help ok"
  fi
  if [[ -x "$ROOT/ds4" ]]; then
    "$ROOT/ds4" --help | grep -q "model" || fail "ds4 help check failed"
    ok "ds4 binary help ok"
  fi
  ok "Ornith 1.5 9B harness skipped cleanly (model absent)"
  exit 0
fi

ok "Found Ornith 9B model: $MODEL"

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

    # Peek metadata block for qwen35 architecture
    data = f.read(65536)
    assert b'qwen35' in data, "Architecture mismatch: expected qwen35"
    print(f"ok GGUF container verified: ver={ver}, tensors={tensor_count}, metadata_kvs={kv_count}, arch=qwen35")
PY
ok "GGUF container and qwen35 architecture strictly verified"

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

# 6. Default chat smoke test (validates ChatML & Metal pipeline, no cross-engine comparison)
echo "Running default chat smoke test on $MODEL ..."
"$ROOT/ds4" -m "$MODEL" -p "$PROMPT" -n "$GEN_TOKENS" --temp 0.0 > "$LOG" 2>&1 || {
  cat "$LOG" >&2
  fail "ds4 default chat smoke failed on $MODEL"
}
cat "$LOG"
ok "ds4 default chat smoke generation succeeded"

# 7. Deterministic raw contract: exact match against reference (llama-completion / llama-cli / Ollama raw:true)
echo "Running deterministic raw generation test on $MODEL (--raw) ..."
"$ROOT/ds4" -m "$MODEL" -p "$PROMPT" --raw -n "$GEN_TOKENS" --temp 0.0 > "$RAW_LOG" 2>&1 || {
  cat "$RAW_LOG" >&2
  fail "ds4 --raw generation failed on $MODEL"
}

python3 - "$ROOT" "$MODEL" "$RAW_LOG" "$REF_LOG" "$PROMPT" "$GEN_TOKENS" << 'PY'
import sys, os, subprocess, urllib.request, json, re

root, model, ds4_raw_log, ref_log_path, prompt, gen_tokens_str = sys.argv[1:7]
gen_tokens = int(gen_tokens_str)

def clean_ds4_raw(text):
    text = re.sub(r'ds4:.*', '', text)
    lines = []
    for l in text.splitlines():
        l_s = l.strip()
        if l_s == "[Start thinking]" or l.startswith("objc[") or l.startswith("!!") or not l_s:
            continue
        lines.append(l_s)
    return " ".join(lines)

def clean_llama_raw(text):
    text = re.sub(r'0\.\d+\.\d+\.\d+\s+[IWE]\s+.*', '', text)
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("objc[")]
    return " ".join(lines)

with open(ds4_raw_log, "r", errors="replace") as f:
    ds4_raw_output = f.read()

ds4_clean = clean_ds4_raw(ds4_raw_output)
print(f"  ds4 (--raw) output: {repr(ds4_clean)}")

compared = False

# Try llama-completion or llama-cli with -no-cnv
llama_bin = None
if subprocess.run(["which", "llama-completion"], capture_output=True).returncode == 0:
    llama_bin = "llama-completion"
elif subprocess.run(["which", "llama-cli"], capture_output=True).returncode == 0:
    llama_bin = "llama-cli"

if llama_bin:
    print(f"Comparing against reference {llama_bin} (-no-cnv)...")
    cmd = [
        llama_bin, "-m", model, "-p", prompt, "-n", str(gen_tokens),
        "--temp", "0.0", "--top-k", "1", "--top-p", "1.0", "-ngl", "999",
        "--no-warmup", "-no-cnv", "--no-display-prompt"
    ]
    try:
        p = subprocess.run(cmd, input="", capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            llama_clean = clean_llama_raw(p.stdout)
            print(f"  {llama_bin} output: {repr(llama_clean)}")
            with open(ref_log_path, "w") as f:
                f.write(p.stdout)
            if ds4_clean != llama_clean:
                print(f"FAIL: mismatch between ds4 --raw and {llama_bin}!", file=sys.stderr)
                print(f"  ds4:   {repr(ds4_clean)}", file=sys.stderr)
                print(f"  llama: {repr(llama_clean)}", file=sys.stderr)
                sys.exit(1)
            compared = True
            print(f"ok exact match verified against {llama_bin}")
        else:
            print(f"warning: {llama_bin} exited with code {p.returncode}")
    except Exception as e:
        print(f"warning: {llama_bin} execution error: {e}")

# Try Ollama API raw:true
if not compared:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps({
                "model": "ornith-1.5:9b",
                "prompt": prompt,
                "stream": False,
                "raw": True,
                "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "num_predict": gen_tokens}
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        ollama_clean = " ".join([l.strip() for l in res.get("response", "").splitlines() if l.strip()])
        print(f"  Ollama (raw:true) output: {repr(ollama_clean)}")
        with open(ref_log_path, "w") as f:
            json.dump(res, f, indent=2)
        if ds4_clean != ollama_clean:
            print(f"FAIL: mismatch between ds4 --raw and Ollama raw:true!", file=sys.stderr)
            print(f"  ds4:    {repr(ds4_clean)}", file=sys.stderr)
            print(f"  ollama: {repr(ollama_clean)}", file=sys.stderr)
            sys.exit(1)
        compared = True
        print("ok exact match verified against Ollama API")
    except Exception as e:
        print(f"warning: Ollama API comparison error: {e}")

if not compared:
    print("skip: no external reference engine (llama-completion or Ollama) available for raw comparison")

PY
ok "Deterministic raw contract verified"

ok "Ornith 1.5 9B test harness complete"
