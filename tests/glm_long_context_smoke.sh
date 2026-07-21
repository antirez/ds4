#!/bin/sh
set -eu

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'USAGE'
Usage: tests/glm_long_context_smoke.sh [MODEL]

Optional long-context GLM-5.2 smoke gate. It builds a long C-code prompt that
ends with an incomplete include line and checks that greedy generation continues
with ">" instead of corrupting the token stream.

Environment:
  DS4_BIN=./ds4
  DS4_GLM_MODEL=models/GLM-5.2-UD-Q4_K_XL.gguf
  DS4_GLM_BACKEND=metal
  DS4_GLM_EXTRA_ARGS="--gpu-vram auto --gpu-devices 0,1"
  DS4_GLM_LONG_CONTEXT_CTX=100000
  DS4_GLM_LONG_CONTEXT_REPEATS=130
  DS4_GLM_LONG_CONTEXT_GEN=32
USAGE
    exit 0
fi

bin=${DS4_BIN:-./ds4}
model=${1:-${DS4_GLM_MODEL:-models/GLM-5.2-UD-Q4_K_XL.gguf}}
ctx=${DS4_GLM_LONG_CONTEXT_CTX:-100000}
repeats=${DS4_GLM_LONG_CONTEXT_REPEATS:-130}
gen=${DS4_GLM_LONG_CONTEXT_GEN:-32}
backend=${DS4_GLM_BACKEND:-metal}
case "$backend" in
    metal|cuda|cpu) ;;
    *) echo "invalid DS4_GLM_BACKEND: $backend" >&2; exit 1 ;;
esac

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-glm-long-context.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT HUP TERM

make_prompt() {
    count=$1
    out=$2
    {
        cat <<'PROMPT'
You are completing a C source file. Return only source text, no prose.

The following audit notes are context padding. Preserve exact code identifiers,
headers, punctuation, and spelling when you later continue the source prefix.
PROMPT
        i=0
        while [ "$i" -lt "$count" ]; do
            printf '/* audit %04d: ggrep searches mmap-backed files with a Boyer-Moore-Horspool skip table. */\n' "$i"
            printf 'static unsigned ggrep_shift_%04d(unsigned char c) { return c == (unsigned char)0 ? 1u : 7u; }\n' "$i"
            printf 'static int ggrep_line_%04d(const char *p, const char *e) { return p < e && *p != '\\''\\n'\\''; }\n' "$i"
            printf '/* note %04d: include spellings must remain stdio.h, stdint.h, string.h, errno.h, fcntl.h, unistd.h. */\n' "$i"
            i=$((i + 1))
        done
        cat <<'PROMPT'

Now continue exactly after the final character of this C source prefix.
The first output character must be ">".

/*
 * ggrep - a small, fast grep.
 *
 * Speed techniques:
 *   - mmap whole regular files and search the page cache directly
 *   - Boyer-Moore-Horspool with a 256-entry bad-character shift table
 *   - line-spanning reporting: on a hit, expand to the enclosing newlines
 *   - one growable output buffer flushed with a single write()
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h
PROMPT
    } > "$out"
}

for repeat in $repeats; do
    prompt="$tmpdir/prompt-$repeat.txt"
    stdout="$tmpdir/stdout-$repeat.txt"
    generated="$tmpdir/generated-$repeat.txt"
    stderr="$tmpdir/stderr-$repeat.txt"
    make_prompt "$repeat" "$prompt"

    echo "glm-long-context-smoke: repeat=$repeat ctx=$ctx gen=$gen"
    # DS4_GLM_EXTRA_ARGS is intentionally shell-split into backend options.
    if ! "$bin" -m "$model" "--$backend" ${DS4_GLM_EXTRA_ARGS:-} \
        --ctx "$ctx" --nothink --temp 0 -n "$gen" \
        --prompt-file "$prompt" >"$stdout" 2>"$stderr"; then
        cat "$stderr" >&2
        exit 1
    fi

    tokens=$(sed -n 's/.*processing \([0-9][0-9]*\) input tokens.*/\1/p' "$stderr" | tail -n 1)
    # CUDA multi-GPU selection reports its resolved layout on stdout before
    # generation. Keep that diagnostic in the raw evidence but exclude it from
    # checks over model text.
    sed '/^ds4: GPU config: .*$/d' "$stdout" > "$generated"
    first=$(LC_ALL=C dd if="$generated" bs=1 count=1 2>/dev/null || true)
    if [ "$first" != ">" ]; then
        echo "glm-long-context-smoke: FAIL repeat=$repeat tokens=${tokens:-unknown}: first byte was not '>'" >&2
        echo "stdout preview:" >&2
        LC_ALL=C head -c 240 "$generated" >&2 || true
        echo >&2
        echo "stderr tail:" >&2
        tail -n 40 "$stderr" >&2 || true
        exit 1
    fi

    if LC_ALL=C grep -aE 'unistdFlush|stdiint|stdintFlush|stdioFlush|errnoFlush' "$generated" >/dev/null; then
        echo "glm-long-context-smoke: FAIL repeat=$repeat tokens=${tokens:-unknown}: known corruption marker found" >&2
        LC_ALL=C grep -aE 'unistdFlush|stdiint|stdintFlush|stdioFlush|errnoFlush' "$generated" >&2 || true
        exit 1
    fi

    perf=$(sed -n 's/^ds4: prefill: //p' "$stderr" | tail -n 1)
    echo "glm-long-context-smoke: PASS repeat=$repeat tokens=${tokens:-unknown} ${perf:-}"
done
