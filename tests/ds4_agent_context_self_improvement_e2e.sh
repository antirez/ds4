#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE=${TMPDIR:-/tmp}
RUN_DIR=$(mktemp -d "${BASE%/}/ds4-agent-context-self-improvement.XXXXXX")
HOME_DIR="$RUN_DIR/home"
REPO="$RUN_DIR/repo"
OUT="$RUN_DIR/output.txt"
TRACE="$RUN_DIR/trace.txt"
LEDGER="$RUN_DIR/ds4-context-self-improvement-ledger.md"
PROMPT_FILE="$RUN_DIR/prompt.md"
REPORT_DIR="$ROOT/tests/generated"
REPORT="$REPORT_DIR/ds4_agent_context_self_improvement_report.md"
REPORT_PROMPT="$REPORT_DIR/ds4_agent_context_self_improvement_prompt.md"
REPORT_OUTPUT="$REPORT_DIR/ds4_agent_context_self_improvement_output.txt"
REPORT_TRACE="$REPORT_DIR/ds4_agent_context_self_improvement_trace.txt"
REPORT_LEDGER="$REPORT_DIR/ds4_agent_context_self_improvement_ledger.md"

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
        printf '# DS4 Agent Context Self Improvement Report\n\n'
        printf 'prompt_file: `%s`\n\n' "$REPORT_PROMPT"
        printf 'response_file: `%s`\n\n' "$REPORT_OUTPUT"
        printf 'trace_file: `%s`\n\n' "$REPORT_TRACE"
        printf 'ledger_file: `%s`\n\n' "$REPORT_LEDGER"
        printf 'run_dir: `%s`\n\n' "$RUN_DIR"
        printf 'repo: `%s`\n\n' "$REPO"
        printf '## Prompt\n\n```text\n'
        print_report_file "$PROMPT_FILE"
        printf '\n```\n\n## DS4 Response\n\n```text\n'
        print_report_file "$OUT"
        printf '\n```\n\n## Trace\n\n```text\n'
        print_report_file "$TRACE"
        printf '\n```\n\n## Ledger\n\n```text\n'
        print_report_file "$LEDGER"
        printf '\n```\n\n## Final toy_math.py\n\n```python\n'
        print_report_file "$REPO/toy_math.py"
        printf '\n```\n'
    } >"$REPORT"
}

cleanup() {
    write_report >/dev/null 2>&1 || true
    if [ "${DS4_KEEP_CONTEXT_SELF_IMPROVEMENT_TEST_DIR:-0}" != "1" ]; then
        rm -rf "$RUN_DIR"
    else
        printf 'kept test directory: %s\n' "$RUN_DIR" >&2
    fi
}
trap cleanup EXIT

mkdir -p "$HOME_DIR" "$REPO"

cat >"$REPO/toy_math.py" <<'PY'
def normalize_score(value, maximum):
    """Return value as a score in the inclusive range 0.0..1.0."""
    if maximum == 0:
        return 0.0
    return value / maximum
PY

cat >"$REPO/test_toy_math.py" <<'PY'
from toy_math import normalize_score


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"{name}: got {got!r}, expected {expected!r}")


check("normal", normalize_score(3, 6), 0.5)
check("zero maximum", normalize_score(3, 0), 0.0)
check("negative maximum", normalize_score(3, -1), 0.0)
check("lower clamp", normalize_score(-2, 10), 0.0)
check("upper clamp", normalize_score(12, 10), 1.0)
print("toy_math tests passed")
PY

git -C "$REPO" init -q
git -C "$REPO" config user.email ds4-agent-test@example.invalid
git -C "$REPO" config user.name "DS4 Agent Test"
git -C "$REPO" add toy_math.py test_toy_math.py
git -C "$REPO" commit -q -m "initial broken toy math"

escape_sed() {
    printf '%s' "$1" | sed 's/[&|]/\\&/g'
}

REPO_ESC=$(escape_sed "$REPO")
LEDGER_ESC=$(escape_sed "$LEDGER")
sed \
    -e "s|__REPO__|$REPO_ESC|g" \
    -e "s|__LEDGER__|$LEDGER_ESC|g" \
    "$ROOT/tests/ds4_agent_context_self_improvement_prompt.md" >"$PROMPT_FILE"

if ! HOME="$HOME_DIR" "$ROOT/ds4-agent" \
    --model "$ROOT/ds4flash.gguf" \
    --non-interactive \
    --chdir "$ROOT" \
    --ctx "${DS4_AGENT_CONTEXT_SELF_IMPROVEMENT_CTX:-8192}" \
    --tokens "${DS4_AGENT_CONTEXT_SELF_IMPROVEMENT_TOKENS:-4500}" \
    --temp 0 \
    --seed 11 \
    --nothink \
    --trace "$TRACE" \
    --prompt "$(cat "$PROMPT_FILE")" >"$OUT" 2>&1
then
    cat "$OUT" >&2
    echo "ds4-agent context self-improvement run failed" >&2
    exit 1
fi

if [ ! -f "$LEDGER" ]; then
    cat "$OUT" >&2
    echo "missing generated context self-improvement ledger: $LEDGER" >&2
    exit 1
fi

python3 "$REPO/test_toy_math.py" >/dev/null

grep -q "git_status_checked=yes" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report git status check" >&2
    exit 1
}

grep -Eq "git_status_mode=(native|bash)" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report a valid git status mode" >&2
    exit 1
}

grep -q "git_diff_checked=yes" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report git diff check" >&2
    exit 1
}

grep -Eq "git_diff_mode=(native|bash)" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report a valid git diff mode" >&2
    exit 1
}

grep -q "context_checkpoint_before=yes" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report pre-fix checkpoint" >&2
    exit 1
}

grep -q "context_checkpoint_after=yes" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report post-fix checkpoint" >&2
    exit 1
}

grep -q "context_restore_used=yes" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report restore usage" >&2
    exit 1
}

grep -q "tests_before_restore=pass" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report pre-restore passing tests" >&2
    exit 1
}

grep -q "tests_after_restore=pass" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger does not report post-restore passing tests" >&2
    exit 1
}

grep -q "final=CONTEXT_SELF_IMPROVEMENT_DONE" "$LEDGER" || {
    cat "$LEDGER" >&2
    echo "ledger is missing final marker" >&2
    exit 1
}

grep -q "CONTEXT_SELF_IMPROVEMENT_DONE" "$OUT" || {
    cat "$OUT" >&2
    echo "DS4 did not finish context self-improvement scenario" >&2
    exit 1
}

if grep -Eq "git[[:space:]]+action=status" "$OUT"; then
    :
elif grep -Fq "git status --short" "$OUT"; then
    :
else
    cat "$OUT" >&2
    echo "output does not prove git status was checked" >&2
    exit 1
fi

if grep -Eq "git[[:space:]]+action=diff" "$OUT"; then
    :
elif grep -Fq "git diff -- toy_math.py" "$OUT"; then
    :
else
    cat "$OUT" >&2
    echo "output does not prove git diff was checked" >&2
    exit 1
fi

grep -Eq "context[[:space:]]+action=checkpoint" "$OUT" || {
    cat "$OUT" >&2
    echo "output does not prove context checkpoint was used" >&2
    exit 1
}

grep -Eq "context[[:space:]]+action=restore|Context restored from checkpoint" "$OUT" || {
    cat "$OUT" >&2
    echo "output does not prove context restore was used" >&2
    exit 1
}

if git -C "$REPO" diff --quiet -- toy_math.py; then
    git -C "$REPO" diff -- toy_math.py >&2
    echo "final patch did not modify toy_math.py" >&2
    exit 1
fi

write_report

printf 'ds4 agent context self-improvement e2e: ok\n'
