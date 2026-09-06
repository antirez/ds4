#!/usr/bin/env python3
"""Generate and verify the complete environment-variable reference."""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "ENVIRONMENT_VARIABLES.md"
METADATA = ROOT / "scripts" / "environment_variables.tsv"
START_MARKER = "<!-- BEGIN GENERATED ENVIRONMENT VARIABLE INVENTORY -->"
END_MARKER = "<!-- END GENERATED ENVIRONMENT VARIABLE INVENTORY -->"

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".inc", ".m", ".mm"}
PRODUCTION_EXCLUDED_DIRS = {
    ".git",
    "dir-steering",
    "tests",
}

# These uppercase DS4 strings are protocol markers, compile-time identifiers,
# or display names rather than environment-variable names.  Treat every other
# DS4_* string literal in production C-family sources as an environment input.
NON_ENV_DS4_STRINGS = {
    "DS4_CUDA",
    "DS4_IMATRIX_PROMPT",
    "DS4_MAX_GPUS",
    "DS4_METAL_HAS_TENSOR",
    "DS4_SORT_ORDER_ASC",
    "DS4_SORT_ORDER_DESC",
}
TEST_NON_ENV_DS4_STRINGS = {"DS4_N_LAYER"}

GROUPS = (
    ("Metal", "DS4_METAL_"),
    ("CUDA", "DS4_CUDA_"),
    ("ROCm", "DS4_ROCM_"),
    ("GLM shared", "DS4_GLM_"),
    ("Distributed", "DS4_DIST_"),
    ("DSpark shared", "DS4_DSPARK_"),
)

DS4_TOKEN_RE = re.compile(r"\bDS4_[A-Z][A-Z0-9_]*\b")
C_STRING_RE = re.compile(r'(?:u8|u|U|L)?"(?:\\.|[^"\\])*"')
DIRECT_GETENV_RE = re.compile(
    r'(?:std::)?getenv\s*\(\s*"([A-Z][A-Z0-9_]*)"\s*\)', re.MULTILINE
)
PYTHON_ENV_RE = re.compile(
    r'os\.(?:environ\.get|getenv)\s*\(\s*["\']([A-Z][A-Z0-9_]*)["\']'
)
PYTHON_API_KEY_RE = re.compile(r'["\']([A-Z][A-Z0-9_]*_API_KEY)["\']')
SHELL_DEFAULT_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*):-([^}\n]*)\}")
SHELL_ENV_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::?[-+?=])")
SHELL_DEFAULT_OVERRIDES = {
    "DS4_DSPARK_MODEL": "${DS4_TEST_MODEL:-./ds4flash.gguf}",
}
PLACEHOLDER_METADATA_PHRASES = (
    "ambiguous:",
    "call-site specific",
    "configure or diagnose ",
    "configure or require test fixture",
    "configure test script",
    "configure the managed nvidia",
    "test-script flag/value",
    "string/path test input",
    "integer test parameter",
    "test/script input string or numeric value",
    "semantics owned by os/linenoise/vendor helper",
)


@dataclass(frozen=True)
class Occurrence:
    path: Path
    line: int
    line_text: str
    context: str


@dataclass(frozen=True)
class VariableDoc:
    scope: str
    name: str
    value_default: str
    purpose: str
    source: str


def relative(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def source_files() -> List[Path]:
    files: List[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        directory_path = Path(directory)
        rel_dir = relative(directory_path)
        if directory_path == ROOT:
            dirnames[:] = [name for name in dirnames if name not in PRODUCTION_EXCLUDED_DIRS]
        elif rel_dir.parts[:2] == ("cuda", "mmq"):
            dirnames[:] = [name for name in dirnames if name != "test"]
        for filename in filenames:
            path = directory_path / filename
            if path.suffix.lower() in SOURCE_SUFFIXES:
                files.append(path)
    return sorted(files, key=lambda item: str(relative(item)))


def test_files() -> List[Path]:
    files = [
        path
        for path in (ROOT / "tests").rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES | {".py", ".sh"}
    ]
    mmq_tests = ROOT / "cuda" / "mmq" / "test"
    if mmq_tests.exists():
        files.extend(
            path
            for path in mmq_tests.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES | {".py", ".sh"}
        )
    return sorted(set(files), key=lambda item: str(relative(item)))


def add_line_occurrence(
    result: MutableMapping[str, List[Occurrence]],
    name: str,
    path: Path,
    lines: Sequence[str],
    line_index: int,
) -> None:
    first = max(0, line_index - 5)
    last = min(len(lines), line_index + 6)
    result.setdefault(name, []).append(
        Occurrence(
            relative(path),
            line_index + 1,
            lines[line_index].strip(),
            "\n".join(lines[first:last]),
        )
    )


def scan_ds4_string_literals(files: Iterable[Path]) -> Dict[str, List[Occurrence]]:
    result: Dict[str, List[Occurrence]] = {}
    for path in files:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_index, line in enumerate(lines):
            for literal in C_STRING_RE.finditer(line):
                for token in DS4_TOKEN_RE.finditer(literal.group(0)):
                    add_line_occurrence(result, token.group(0), path, lines, line_index)
    return result


def scan_direct_getenv(files: Iterable[Path]) -> Dict[str, List[Occurrence]]:
    result: Dict[str, List[Occurrence]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        line_starts = [0]
        line_starts.extend(match.end() for match in re.finditer("\n", text))

        matches = list(DIRECT_GETENV_RE.finditer(text))
        if path.suffix == ".py":
            matches.extend(PYTHON_ENV_RE.finditer(text))
            matches.extend(PYTHON_API_KEY_RE.finditer(text))
        for match in sorted(matches, key=lambda item: item.start()):
            line_index = bisect.bisect_right(line_starts, match.start()) - 1
            add_line_occurrence(result, match.group(1), path, lines, line_index)
    return result


def scan_shell_inputs(files: Iterable[Path]) -> Tuple[Dict[str, List[Occurrence]], Dict[str, str]]:
    result: Dict[str, List[Occurrence]] = {}
    defaults: Dict[str, str] = {}
    for path in files:
        if path.suffix != ".sh":
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_index, line in enumerate(lines):
            for match in SHELL_ENV_RE.finditer(line):
                add_line_occurrence(result, match.group(1), path, lines, line_index)
            for match in SHELL_DEFAULT_RE.finditer(line):
                defaults.setdefault(match.group(1), match.group(2).strip())
    for name, default in SHELL_DEFAULT_OVERRIDES.items():
        if name in result:
            defaults[name] = default
    return result, defaults


def load_metadata() -> List[VariableDoc]:
    try:
        handle = METADATA.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError(f"cannot read {METADATA.relative_to(ROOT)}: {error}") from error
    with handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = list(reader)
    expected_header = ["SCOPE", "NAME", "VALUE_DEFAULT_SEMANTICS", "PURPOSE", "SOURCE"]
    if not rows or rows[0] != expected_header:
        raise ValueError(f"{METADATA.relative_to(ROOT)} has an invalid header")

    result: List[VariableDoc] = []
    seen: Set[Tuple[str, str]] = set()
    for line, row in enumerate(rows[1:], 2):
        if len(row) != 5 or any(not field.strip() for field in row):
            raise ValueError(
                f"{METADATA.relative_to(ROOT)}:{line}: expected five nonempty TSV fields"
            )
        item = VariableDoc(*(field.strip() for field in row))
        searchable = f"{item.value_default}\n{item.purpose}".lower()
        placeholder = next(
            (phrase for phrase in PLACEHOLDER_METADATA_PHRASES if phrase in searchable),
            None,
        )
        if placeholder:
            raise ValueError(
                f"{METADATA.relative_to(ROOT)}:{line}: placeholder metadata "
                f"{placeholder!r} remains for {item.name}"
            )
        key = (item.scope, item.name)
        if key in seen:
            raise ValueError(
                f"{METADATA.relative_to(ROOT)}:{line}: duplicate scope/name {key!r}"
            )
        seen.add(key)
        result.append(item)

    expected_order = sorted(result, key=lambda item: (item.scope, item.name))
    if result != expected_order:
        raise ValueError(
            f"{METADATA.relative_to(ROOT)} must be sorted by SCOPE and NAME"
        )

    source_cache: Dict[Path, List[str]] = {}
    for item in result:
        found_name = item.name.startswith("<")
        for reference in (part.strip() for part in item.source.split(";")):
            match = re.fullmatch(r"(.+):(\d+)", reference)
            if not match:
                raise ValueError(
                    f"{METADATA.relative_to(ROOT)}: invalid source reference {reference!r}"
                )
            rel_path, line_text = match.groups()
            path = ROOT / rel_path
            if path not in source_cache:
                try:
                    source_cache[path] = path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                except OSError as error:
                    raise ValueError(f"cannot read metadata source {rel_path}: {error}") from error
            lines = source_cache[path]
            line = int(line_text)
            if line < 1 or line > len(lines):
                raise ValueError(f"metadata source is outside {rel_path}: {line}")
            first = max(0, line - 6)
            last = min(len(lines), line + 5)
            if item.name in "\n".join(lines[first:last]):
                found_name = True
        if not found_name:
            raise ValueError(
                f"metadata source for {item.scope}/{item.name} does not mention the name"
            )
    return result


def metadata_source_links(source: str) -> str:
    links: List[str] = []
    for value in (part.strip() for part in source.split(";")):
        match = re.fullmatch(r"(.+):(\d+)", value)
        if not match:
            links.append(f"`{value}`")
            continue
        path, line = match.groups()
        links.append(f"[{path}:{line}]({path}#L{line})")
    return "; ".join(links)


def metadata_table(entries: Sequence[VariableDoc]) -> List[str]:
    lines = [
        "| Variable | Accepted value and default | Effect | Source |",
        "| --- | --- | --- | --- |",
    ]
    for item in sorted(entries, key=lambda entry: entry.name):
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                item.name,
                escape_cell(item.value_default),
                escape_cell(item.purpose),
                metadata_source_links(item.source),
            )
        )
    return lines


def group_for(name: str) -> str:
    for title, prefix in GROUPS:
        if name.startswith(prefix):
            return title
    return "General and shared"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def merge_occurrences(
    *mappings: Mapping[str, Sequence[Occurrence]],
) -> Dict[str, List[Occurrence]]:
    result: Dict[str, List[Occurrence]] = {}
    for mapping in mappings:
        for name, occurrences in mapping.items():
            result.setdefault(name, []).extend(occurrences)
    return result


def auxiliary_inputs(
    runtime_names: Set[str], external_runtime_names: Set[str]
) -> Tuple[Dict[str, List[Occurrence]], Dict[str, str], Dict[str, List[Occurrence]], Dict[str, str]]:
    tests = test_files()
    test_direct = scan_direct_getenv(tests)
    test_literals = scan_ds4_string_literals(
        path for path in tests if path.suffix.lower() in SOURCE_SUFFIXES
    )
    test_shell, test_defaults = scan_shell_inputs(tests)
    test_all = merge_occurrences(test_direct, test_literals, test_shell)
    test_only = {
        name: occurrences
        for name, occurrences in test_all.items()
        if name not in runtime_names and name not in external_runtime_names
        and name not in NON_ENV_DS4_STRINGS
        and name not in TEST_NON_ENV_DS4_STRINGS
        and (name.startswith("DS4_") or name in {"DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "PROTO_Q8_DEBUG", "TMPDIR"})
    }

    tool_files: List[Path] = []
    for directory, dirnames, filenames in os.walk(ROOT):
        directory_path = Path(directory)
        rel_dir = relative(directory_path)
        dirnames[:] = [
            name
            for name in dirnames
            if name not in {".git", "__pycache__", "out", "dataset", "tests"}
            and not name.endswith(".dSYM")
        ]
        if rel_dir.parts[:2] == ("cuda", "mmq"):
            dirnames[:] = [name for name in dirnames if name != "test"]
        for filename in filenames:
            path = directory_path / filename
            if path.suffix not in {".py", ".sh"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            tool_files.append(path)
    tool_direct = scan_direct_getenv(tool_files)
    tool_shell, tool_defaults = scan_shell_inputs(tool_files)
    tool_all = {
        name: occurrences
        for name, occurrences in merge_occurrences(tool_direct, tool_shell).items()
        if name.startswith("DS4_")
        or name in {"DEEPSEEK_API_KEY", "FLATTEN_DOWNLOADS", "FORCE_HF_DOWNLOAD", "HF_TOKEN", "OPENROUTER_API_KEY"}
    }
    return test_only, test_defaults, tool_all, tool_defaults


def generated_block() -> Tuple[str, int, int, int, int]:
    production = source_files()
    literal_occurrences = scan_ds4_string_literals(production)
    unexpected_allowlist = NON_ENV_DS4_STRINGS - set(literal_occurrences)
    if unexpected_allowlist:
        names = ", ".join(sorted(unexpected_allowlist))
        raise ValueError(f"stale NON_ENV_DS4_STRINGS entries: {names}")

    runtime_names = set(literal_occurrences) - NON_ENV_DS4_STRINGS
    direct = scan_direct_getenv(production)
    external_names = {name for name in direct if not name.startswith("DS4_")}

    metadata = load_metadata()
    runtime_docs: Dict[str, VariableDoc] = {}
    external_docs: Dict[str, VariableDoc] = {}
    for item in metadata:
        target: Dict[str, VariableDoc]
        if item.scope.startswith("runtime/"):
            target = runtime_docs
        elif item.scope == "external/system":
            target = external_docs
        else:
            continue
        if item.name in target:
            raise ValueError(f"duplicate documented runtime name: {item.name}")
        target[item.name] = item

    missing_runtime = runtime_names - set(runtime_docs)
    stale_runtime = set(runtime_docs) - runtime_names
    missing_external = external_names - set(external_docs)
    stale_external = set(external_docs) - external_names
    if missing_runtime or stale_runtime or missing_external or stale_external:
        details: List[str] = []
        if missing_runtime:
            details.append("undocumented runtime: " + ", ".join(sorted(missing_runtime)))
        if stale_runtime:
            details.append("stale runtime metadata: " + ", ".join(sorted(stale_runtime)))
        if missing_external:
            details.append("undocumented external runtime: " + ", ".join(sorted(missing_external)))
        if stale_external:
            details.append("stale external metadata: " + ", ".join(sorted(stale_external)))
        raise ValueError("; ".join(details))

    test_binary = [item for item in metadata if item.scope == "test-only"]
    test_scripts = [item for item in metadata if item.scope == "test-script"]
    tools = [item for item in metadata if item.scope.startswith("script/")]

    scanned_tests, _test_defaults, scanned_tools, _tool_defaults = auxiliary_inputs(
        runtime_names, external_names
    )
    documented_test_names = {
        item.name for item in test_binary + test_scripts
    } - runtime_names - external_names
    if set(scanned_tests) != documented_test_names:
        missing = set(scanned_tests) - documented_test_names
        stale = documented_test_names - set(scanned_tests)
        details = []
        if missing:
            details.append("undocumented test input: " + ", ".join(sorted(missing)))
        if stale:
            details.append("stale test metadata: " + ", ".join(sorted(stale)))
        raise ValueError("; ".join(details))

    documented_tool_names = {item.name for item in tools}
    allowed_dynamic_tool_names = {"<NAME_FROM_--api-key-env>", "HOME"}
    if set(scanned_tools) != documented_tool_names - allowed_dynamic_tool_names:
        missing = set(scanned_tools) - documented_tool_names
        stale = documented_tool_names - allowed_dynamic_tool_names - set(scanned_tools)
        details = []
        if missing:
            details.append("undocumented tool input: " + ", ".join(sorted(missing)))
        if stale:
            details.append("stale tool metadata: " + ", ".join(sorted(stale)))
        raise ValueError("; ".join(details))

    lines = [
        START_MARKER,
        "## Complete implementation inventory",
        "",
        "This section is generated by `scripts/generate_environment_variables.py`; do not edit it by hand.",
        "Human-reviewed value/default and purpose metadata lives in",
        "`scripts/environment_variables.tsv`; the generator verifies it against the source tree.",
        "It lists every `DS4_*` string consumed by production C/C++/Objective-C/CUDA/ROCm",
        "sources, including names passed indirectly through helper functions, macros, and",
        "source-specification arrays. Unless a variable appears in the user-facing reference",
        "above, it is an unstable internal diagnostic or tuning interface. The linked source",
        "remains normative for exact eligibility",
        "gates, bounds, and architecture-specific defaults.",
        "",
        f"Inventory totals: **{len(runtime_names)} `DS4_*` runtime variables** and",
        f"**{len(external_names)} external runtime variables**.",
        f"The auxiliary inventories contain **{len(test_binary) + len(test_scripts)} test/test-fixture entries**",
        f"and **{len(tools)} tool/wrapper entries**.",
        "",
    ]

    ordered_groups = [title for title, _prefix in GROUPS] + ["General and shared"]
    for title in ordered_groups:
        entries = [item for name, item in runtime_docs.items() if group_for(name) == title]
        if not entries:
            continue
        lines.extend(
            [
                "<details>",
                f"<summary><strong>{title} ({len(entries)})</strong></summary>",
                "",
                *metadata_table(entries),
                "",
                "</details>",
                "",
            ]
        )

    if external_names:
        lines.extend(
            [
                "### External runtime environment",
                "",
                "These names are not owned by the `DS4_*` namespace but are read directly by",
                "the binaries or vendored runtime code.",
                "",
                *metadata_table(list(external_docs.values())),
                "",
            ]
        )

    lines.extend(
        [
            "## Test and fixture environment inputs",
            "",
            "These entries are consumed by repository test binaries or fixture scripts. Some",
            "production runtime controls are repeated here because a maintained fixture exposes",
            "them as part of its own test contract.",
            "",
            "### Test binaries and cleanup hooks",
            "",
            *metadata_table(test_binary),
            "",
            "### Test fixture scripts",
            "",
            *metadata_table(test_scripts),
            "",
            "## Tool and wrapper environment inputs",
            "",
            "These variables configure maintained download, service-wrapper, and offline tooling.",
            "A tool that accepts a variable name dynamically (for example `--api-key-env`) may read",
            "the caller-selected name in addition to the literal defaults listed here.",
            "",
            *metadata_table(tools),
            "",
            END_MARKER,
        ]
    )
    return (
        "\n".join(lines),
        len(runtime_names),
        len(external_names),
        len(test_binary) + len(test_scripts),
        len(tools),
    )


def replace_generated_block(document: str, block: str) -> str:
    start = document.find(START_MARKER)
    end = document.find(END_MARKER)
    if start < 0 or end < 0 or end < start:
        raise ValueError("environment inventory markers are missing from ENVIRONMENT_VARIABLES.md")
    end += len(END_MARKER)
    return document[:start] + block + document[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the checked-in reference")
    mode.add_argument("--emit", action="store_true", help="print the generated inventory block")
    args = parser.parse_args()

    try:
        block, runtime_count, external_count, test_count, tool_count = generated_block()
        if args.emit:
            print(block)
            return 0
        document = DOCUMENT.read_text(encoding="utf-8")
        expected = replace_generated_block(document, block)
    except (OSError, ValueError) as error:
        print(f"environment-variable documentation: {error}", file=sys.stderr)
        return 1

    summary = (
        f"{DOCUMENT.relative_to(ROOT)}: {runtime_count} DS4 runtime, "
        f"{external_count} external runtime, {test_count} test/test-fixture entries, "
        f"{tool_count} tool/wrapper entries"
    )
    if args.check:
        if document != expected:
            print(f"{summary}; generated inventory is stale", file=sys.stderr)
            print("run: python3 scripts/generate_environment_variables.py", file=sys.stderr)
            return 1
        print(f"{summary}; generated inventory is current")
        return 0

    DOCUMENT.write_text(expected, encoding="utf-8")
    print(f"updated {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
