#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE=${TMPDIR:-/tmp}
RUN_DIR=$(mktemp -d "${BASE%/}/ds4-agent-compact-canary.XXXXXX")
HOME_DIR="$RUN_DIR/home"
WORK_DIR="$RUN_DIR/work"
OUT="$RUN_DIR/output.txt"
TRACE="$RUN_DIR/trace.txt"
LEDGER="$WORK_DIR/ds4-compact-canary-ledger.md"
PROMPT_FILE="$RUN_DIR/prompt.md"
PROMPT_TMP="$RUN_DIR/prompt.tmp"
PADDING_FILE="$RUN_DIR/padding.txt"
REPORT_DIR="$ROOT/tests/generated"
REPORT="$REPORT_DIR/ds4_agent_context_compact_canary_report.md"
REPORT_PROMPT="$REPORT_DIR/ds4_agent_context_compact_canary_prompt.md"
REPORT_OUTPUT="$REPORT_DIR/ds4_agent_context_compact_canary_output.txt"
REPORT_TRACE="$REPORT_DIR/ds4_agent_context_compact_canary_trace.txt"
REPORT_LEDGER="$REPORT_DIR/ds4_agent_context_compact_canary_ledger.md"

print_report_file() {
    if [ -f "$1" ]; then
        sed 's/```/` ` `/g' "$1"
    else
        printf '(missing: %s)\n' "$1"
    fi
}

write_report() {
    mkdir -p "$REPORT_DIR"
    [ -f "$PROMPT_FILE" ] && cp "$PROMPT_FILE" "$REPORT_PROMPT"
    [ -f "$OUT" ] && cp "$OUT" "$REPORT_OUTPUT"
    [ -f "$TRACE" ] && cp "$TRACE" "$REPORT_TRACE"
    [ -f "$LEDGER" ] && cp "$LEDGER" "$REPORT_LEDGER"
    {
        printf '# DS4 Agent Context Compact Canary Report\n\n'
        printf 'prompt_file: `%s`\n\n' "$REPORT_PROMPT"
        printf 'response_file: `%s`\n\n' "$REPORT_OUTPUT"
        printf 'trace_file: `%s`\n\n' "$REPORT_TRACE"
        printf 'ledger_file: `%s`\n\n' "$REPORT_LEDGER"
        printf 'run_dir: `%s`\n\n' "$RUN_DIR"
        printf '## Prompt\n\n```text\n'
        print_report_file "$PROMPT_FILE"
        printf '\n```\n\n## DS4 Response\n\n```text\n'
        print_report_file "$OUT"
        printf '\n```\n\n## Trace\n\n```text\n'
        print_report_file "$TRACE"
        printf '\n```\n\n## Ledger\n\n```text\n'
        print_report_file "$LEDGER"
        printf '\n```\n'
    } >"$REPORT"
}

cleanup() {
    write_report >/dev/null 2>&1 || true
    if [ "${DS4_KEEP_COMPACT_CANARY_TEST_DIR:-0}" != "1" ]; then
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

PADDING_LINES=${DS4_AGENT_COMPACT_CANARY_PADDING_LINES:-180}
i=1
while [ "$i" -le "$PADDING_LINES" ]; do
    printf 'Padding line %03d: irrelevant build-note-%03d contains no canary values and should not be retained.\n' "$i" "$i" >>"$PADDING_FILE"
    i=$((i + 1))
done

ROOT_ESC=$(escape_sed "$ROOT")
LEDGER_ESC=$(escape_sed "$LEDGER")
sed \
    -e "s|__ROOT__|$ROOT_ESC|g" \
    -e "s|__LEDGER__|$LEDGER_ESC|g" \
    "$ROOT/tests/ds4_agent_context_compact_canary_prompt.md" >"$PROMPT_TMP"

while IFS= read -r line; do
    if [ "$line" = "__PADDING__" ]; then
        cat "$PADDING_FILE"
    else
        printf '%s\n' "$line"
    fi
done <"$PROMPT_TMP" >"$PROMPT_FILE"

if ! HOME="$HOME_DIR" "$ROOT/ds4-agent" \
    --model "$ROOT/ds4flash.gguf" \
    --non-interactive \
    --chdir "$ROOT" \
    --ctx "${DS4_AGENT_COMPACT_CANARY_CTX:-8192}" \
    --tokens "${DS4_AGENT_COMPACT_CANARY_TOKENS:-3500}" \
    --temp 0 \
    --seed 7 \
    --nothink \
    --trace "$TRACE" \
    --prompt "$(cat "$PROMPT_FILE")" >"$OUT" 2>&1
then
    cat "$OUT" >&2
    echo "ds4-agent compact canary run failed" >&2
    exit 1
fi

if [ ! -f "$LEDGER" ]; then
    cat "$OUT" >&2
    echo "missing generated compact canary ledger: $LEDGER" >&2
    exit 1
fi

COMPACT_LINE=$(grep 'compacted reason="canary-retention-test"' "$TRACE" | tail -n 1 || true)
if [ -z "$COMPACT_LINE" ]; then
    cat "$TRACE" >&2
    echo "trace does not prove context compaction happened" >&2
    exit 1
fi

COMPACT_OLD=$(printf '%s\n' "$COMPACT_LINE" | sed -n 's/.* old=\([0-9][0-9]*\) .*/\1/p')
COMPACT_NEW=$(printf '%s\n' "$COMPACT_LINE" | sed -n 's/.* new=\([0-9][0-9]*\) .*/\1/p')
COMPACT_TAIL_START=$(printf '%s\n' "$COMPACT_LINE" | sed -n 's/.* tail_start=\([0-9][0-9]*\) .*/\1/p')

if [ -z "$COMPACT_OLD" ] || [ -z "$COMPACT_NEW" ] || [ -z "$COMPACT_TAIL_START" ]; then
    cat "$TRACE" >&2
    echo "trace compact line is missing old/new/tail_start metrics" >&2
    exit 1
fi

if [ "$COMPACT_NEW" -ge "$COMPACT_OLD" ]; then
    cat "$TRACE" >&2
    echo "context compaction did not reduce token count" >&2
    exit 1
fi

if [ "$COMPACT_TAIL_START" -lt "${DS4_AGENT_COMPACT_CANARY_MIN_TAIL_START:-3000}" ]; then
    cat "$TRACE" >&2
    echo "recent tail started too early to make this a useful retention canary" >&2
    exit 1
fi

grep -q "compaction=done" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing compaction=done" >&2
    exit 1
}

grep -q "canary_alpha=ORCHID-47" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger lost canary_alpha" >&2
    exit 1
}

grep -q "canary_beta=FJORD-932" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger lost canary_beta" >&2
    exit 1
}

grep -q "canary_gamma=LEMMA-18" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger lost canary_gamma" >&2
    exit 1
}

grep -q "canary_delta=RUNE-604" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger lost canary_delta" >&2
    exit 1
}

grep -q "canary_epsilon=VECTOR-251" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger lost canary_epsilon" >&2
    exit 1
}

grep -q "final=COMPACT_CANARY_DONE" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing final marker" >&2
    exit 1
}

grep -q "COMPACT_CANARY_DONE" "$OUT" || {
    cat "$OUT" >&2
    echo "DS4 did not finish compact canary scenario" >&2
    exit 1
}

write_report

printf 'ds4 agent context compact canary e2e: ok\n'
