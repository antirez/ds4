#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE=${TMPDIR:-/tmp}
RUN_DIR=$(mktemp -d "${BASE%/}/ds4-agent-context-loop.XXXXXX")
HOME_DIR="$RUN_DIR/home"
WORK_DIR="$RUN_DIR/work"
OUT="$RUN_DIR/output.txt"
LEDGER="$WORK_DIR/ds4-generated-loop.md"
PROMPT_FILE="$RUN_DIR/prompt.md"
REPORT_DIR="$ROOT/tests/generated"
REPORT="$REPORT_DIR/ds4_agent_context_loop_report.md"
REPORT_PROMPT="$REPORT_DIR/ds4_agent_context_loop_prompt.md"
REPORT_OUTPUT="$REPORT_DIR/ds4_agent_context_loop_output.txt"
REPORT_LEDGER="$REPORT_DIR/ds4_agent_context_loop_ledger.md"

print_report_file() {
    if [ -f "$1" ]; then
        sed 's/```/` ` `/g' "$1"
    else
        printf '(missing: %s)\n' "$1"
    fi
}

write_report() {
    mkdir -p "$REPORT_DIR"
    if [ -f "$PROMPT_FILE" ]; then
        cp "$PROMPT_FILE" "$REPORT_PROMPT"
    fi
    if [ -f "$OUT" ]; then
        cp "$OUT" "$REPORT_OUTPUT"
    fi
    if [ -f "$LEDGER" ]; then
        cp "$LEDGER" "$REPORT_LEDGER"
    fi
    {
        printf '# DS4 Agent Context Loop Report\n\n'
        printf 'prompt_file: `%s`\n\n' "$REPORT_PROMPT"
        printf 'response_file: `%s`\n\n' "$REPORT_OUTPUT"
        printf 'ledger_file: `%s`\n\n' "$REPORT_LEDGER"
        printf 'run_dir: `%s`\n\n' "$RUN_DIR"
        printf '## Prompt\n\n```text\n'
        print_report_file "$PROMPT_FILE"
        printf '\n```\n\n## DS4 Response\n\n```text\n'
        print_report_file "$OUT"
        printf '\n```\n\n## Ledger\n\n```text\n'
        print_report_file "$LEDGER"
        printf '\n```\n'
    } >"$REPORT"
}

cleanup() {
    write_report >/dev/null 2>&1 || true
    if [ "${DS4_KEEP_LOOP_TEST_DIR:-0}" != "1" ]; then
        rm -rf "$RUN_DIR"
    else
        printf 'kept test directory: %s\n' "$RUN_DIR" >&2
    fi
}
trap cleanup EXIT

mkdir -p "$HOME_DIR" "$WORK_DIR"

escape_sed() {
    printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

ROOT_ESC=$(escape_sed "$ROOT")
LEDGER_ESC=$(escape_sed "$LEDGER")
PROMPT=$(sed \
    -e "s|__ROOT__|$ROOT_ESC|g" \
    -e "s|__LEDGER__|$LEDGER_ESC|g" \
    "$ROOT/tests/ds4_agent_context_loop_prompt.md")
printf '%s\n' "$PROMPT" >"$PROMPT_FILE"

if ! HOME="$HOME_DIR" "$ROOT/ds4-agent" \
    --model "$ROOT/ds4flash.gguf" \
    --non-interactive \
    --chdir "$ROOT" \
    --ctx "${DS4_AGENT_LOOP_CTX:-8192}" \
    --tokens "${DS4_AGENT_LOOP_TOKENS:-2500}" \
    --temp 0 \
    --seed 1 \
    --nothink \
    --prompt "$PROMPT" >"$OUT" 2>&1
then
    cat "$OUT" >&2
    echo "ds4-agent loop run failed" >&2
    exit 1
fi

if [ ! -f "$LEDGER" ]; then
    cat "$OUT" >&2
    echo "missing generated ledger: $LEDGER" >&2
    exit 1
fi

grep -q "loop_limit=2" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing loop_limit=2" >&2
    exit 1
}

grep -Fq "ds4_prompt=validate DS4's own agent context loop capability" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing the compact DS4 prompt" >&2
    exit 1
}

grep -q "ds4_response=LOOP_DONE" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing the final DS4 response" >&2
    exit 1
}

grep -q "attempt=1 status=pass" "$LEDGER" || {
    cat "$LEDGER" >&2
    cat "$OUT" >&2
    echo "DS4 did not mark the measured attempt as pass" >&2
    exit 1
}

grep -q "attempt=1 metric=ds4_agent_context_test passed" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing the expected metric" >&2
    exit 1
}

grep -q "LOOP_DONE" "$OUT" || {
    cat "$OUT" >&2
    echo "DS4 did not finish the loop" >&2
    exit 1
}

CONTEXT_DIR="$HOME_DIR/.ds4/kvcache/context"
if [ ! -d "$CONTEXT_DIR" ]; then
    cat "$OUT" >&2
    echo "missing context checkpoint directory: $CONTEXT_DIR" >&2
    exit 1
fi

if ! grep -R "ds4-generated-loop-after-pass" "$CONTEXT_DIR" >/dev/null 2>&1; then
    find "$CONTEXT_DIR" -maxdepth 1 -type f -print >&2
    cat "$OUT" >&2
    echo "missing DS4-generated context checkpoint label" >&2
    exit 1
fi

write_report

grep -q "^## Prompt" "$REPORT" || {
    cat "$REPORT" >&2
    echo "report is missing the prompt section" >&2
    exit 1
}

grep -q "^## DS4 Response" "$REPORT" || {
    cat "$REPORT" >&2
    echo "report is missing the DS4 response section" >&2
    exit 1
}

grep -q "^## Ledger" "$REPORT" || {
    cat "$REPORT" >&2
    echo "report is missing the ledger section" >&2
    exit 1
}

grep -q "LOOP_DONE" "$REPORT" || {
    cat "$REPORT" >&2
    echo "report is missing the final DS4 response" >&2
    exit 1
}

printf 'ds4 agent context loop e2e: ok\n'
