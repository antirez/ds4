#!/usr/bin/env python3
"""Build a local calibration corpus for DeepSeek V4 Flash imatrix collection.

The quantizer needs activation statistics from prompts that look like the real
workload.  This script creates deterministic DS4-rendered chat prompts from the
repo itself, agent/tool conversations, multilingual translation tasks,
programming prompts, and long-context code reviews.  The output is
intentionally plain JSONL/text so the imatrix collector can consume it without
depending on this script.

The chat/tool rendering MUST match ``ds4_server.c``.  The low-bit routed MoE
experts are sensitive to prompt shape: if agent/tool traffic is
underrepresented, the imatrix can be less representative of real DS4 usage even
when the resulting GGUF is otherwise valid.  Keep this corpus realistic and
provider-neutral rather than tuned around a single client brand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"

DEFAULT_SYSTEM = (
    "You are DeepSeek V4 Flash running locally. Answer accurately, preserve "
    "technical details, and use tools only when the prompt asks for tool use."
)

TOOLS_PROMPT_INTRO = (
    "## Tools\n\n"
    "You have access to a set of tools to help answer the user question. "
    "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "...\n"
    "</｜DSML｜invoke>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n\n"
    "String parameters should be specified as raw text and set `string=\"true\"`. "
    "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal "
    "string characters with XML or HTML entity escapes. Only if a string value "
    "itself contains the exact closing parameter tag `</｜DSML｜parameter>`, "
    "write that tag as `&lt;/｜DSML｜parameter>` inside the value. "
    "For all other types (numbers, booleans, arrays, objects), pass the value "
    "in JSON format and set `string=\"false\"`.\n\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your "
    "complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n\n"
    "### Available Tool Schemas\n\n"
)

TOOLS_PROMPT_OUTRO = (
    "\n\nYou MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls. Use the exact parameter names from the schemas."
)


def render_tool_schema_lines(tools: list[dict]) -> str:
    """Match ``openai_function_schema_from_tool`` + ``append_raw_json_line``.

    The server unwraps ``{"type":"function","function":{...}}`` envelopes and
    joins one bare function object per line with ``\n``, with no ``[]`` array
    wrapper.  Real clients send the JSON compact, so we use compact dumps so
    the calibration sees the same token shape as the runtime emits.
    """

    lines = []
    for t in tools:
        inner = t.get("function", t) if isinstance(t, dict) else t
        lines.append(json.dumps(inner, separators=(",", ":"), ensure_ascii=False))
    return "\n".join(lines)


def tools_prompt_text(tools: list[dict]) -> str:
    return TOOLS_PROMPT_INTRO + render_tool_schema_lines(tools) + TOOLS_PROMPT_OUTRO


@dataclass
class Record:
    rid: str
    category: str
    mode: str
    source: str
    messages: list[dict]
    rendered: str


def escape_tool_result(text: str) -> str:
    """Match ``append_tool_result_text`` for tool result bodies."""

    return text.replace("</tool_result>", "&lt;/tool_result>")


def escape_dsml_parameter(text: str) -> str:
    """Match ``append_dsml_parameter_text`` (escapes the closing sentinel)."""

    end = "</｜DSML｜parameter>"
    out = []
    i = 0
    while i < len(text):
        if text.startswith(end, i):
            out.append("&lt;")
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def dsml_param(key: str, value: str, *, is_string: bool) -> str:
    flag = "true" if is_string else "false"
    body = escape_dsml_parameter(value) if is_string else value
    return f'<｜DSML｜parameter name="{key}" string="{flag}">{body}</｜DSML｜parameter>\n'


def dsml_invoke(name: str, params: list[tuple[str, str, bool]]) -> str:
    body = "".join(dsml_param(k, v, is_string=s) for k, v, s in params)
    return f'<｜DSML｜invoke name="{name}">\n{body}</｜DSML｜invoke>\n'


def dsml_tool_calls(calls: list[tuple[str, list[tuple[str, str, bool]]]]) -> str:
    """Match ``append_dsml_tool_calls_text`` exactly: two leading newlines, one
    newline inside, no trailing newline after the closing tag."""

    inner = "".join(dsml_invoke(name, params) for name, params in calls)
    return f"\n\n<｜DSML｜tool_calls>\n{inner}</｜DSML｜tool_calls>"


def history_uses_tool_context(messages: list[dict], tools: bool) -> bool:
    """Match ``chat_history_uses_tool_context``."""

    if tools:
        return True
    for m in messages:
        role = m.get("role", "")
        if role in ("tool", "function"):
            return True
        if role == "assistant" and m.get("dsml"):
            return True
    return False


def render(messages: list[dict], mode: str, tools_schema: list[dict] | None = None) -> str:
    """Mirror ``render_chat_prompt_text`` in ``ds4_server.c``.

    ``tools_schema`` is the OpenAI-style tool list; if non-empty it is rendered
    via :func:`tools_prompt_text` and appended to the joined system block, and
    every historic assistant turn is wrapped in ``<think></think>`` rather than
    a bare ``</think>``.
    """

    tools = bool(tools_schema)
    tool_context = history_uses_tool_context(messages, tools)
    think = mode == "think"

    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    if tools:
        system_parts.append(tools_prompt_text(tools_schema))
    system = "\n\n".join(p for p in system_parts if p)

    last_user_idx = max(
        (i for i, m in enumerate(messages)
         if m.get("role") in ("user", "tool", "function")),
        default=-1,
    )

    out = [BOS, system]
    pending_assistant = False
    pending_tool_result = False
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        if role == "user":
            out.extend([USER, content])
            pending_assistant = True
            pending_tool_result = False
        elif role in ("tool", "function"):
            if not pending_tool_result:
                out.append(USER)
            out.extend(["<tool_result>", escape_tool_result(content), "</tool_result>"])
            pending_assistant = True
            pending_tool_result = True
        elif role == "assistant":
            if pending_assistant:
                out.append(ASSISTANT)
                if think:
                    if tool_context or i > last_user_idx:
                        out.extend(["<think>", msg.get("reasoning", "") or "", "</think>"])
                    else:
                        out.append("</think>")
                else:
                    out.append("</think>")
            out.append(content)
            if msg.get("dsml"):
                out.append(msg["dsml"])
            out.append(EOS)
            pending_assistant = False
            pending_tool_result = False

    if pending_assistant:
        out.extend([ASSISTANT, "<think>" if think else "</think>"])

    return "".join(out)


def stable_id(category: str, source: str, mode: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(category.encode())
    h.update(b"\0")
    h.update(source.encode())
    h.update(b"\0")
    h.update(mode.encode())
    h.update(b"\0")
    h.update(text[:4096].encode("utf-8", "ignore"))
    return f"{category}-{h.hexdigest()[:12]}"


def add_record(records: list[Record], category: str, source: str,
               messages: list[dict], *, tools_schema: list[dict] | None = None,
               modes: Iterable[str] = ("nothink", "think")) -> None:
    for mode in modes:
        rendered = render(messages, mode, tools_schema=tools_schema)
        rid = stable_id(category, source, mode, rendered)
        records.append(Record(rid, category, mode, source, messages, rendered))


PROVIDER_REFERENCE_PATTERNS = ("clau" "de",)


def has_provider_reference(text: str) -> bool:
    """Keep the imatrix corpus independent from one branded client."""

    lower = text.lower()
    return any(pattern in lower for pattern in PROVIDER_REFERENCE_PATTERNS)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            nl = text.rfind("\n", i + size // 2, j)
            if nl > i:
                j = nl
        part = text[i:j].strip()
        if part:
            chunks.append(part)
        if j == len(text):
            break
        i = max(j - overlap, i + 1)
    return chunks


def code_prompt(path: str, chunk: str, task: str) -> list[dict]:
    lang = "metal" if path.endswith(".metal") else "c"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": f"{task}\n\nFile: {path}\n\n```{lang}\n{chunk}\n```"},
    ]


def doc_prompt(path: str, chunk: str, task: str) -> list[dict]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": f"{task}\n\nDocument: {path}\n\n{chunk}"},
    ]


def make_source_records(root: Path, records: list[Record]) -> None:
    files = [
        "ds4.c", "ds4_server.c", "ds4_cli.c", "ds4_metal.m", "ds4.h", "ds4_gpu.h",
        "README.md", "AGENT.md", "gguf-tools/README.md",
        "gguf-tools/imatrix/README.md", "gguf-tools/imatrix/dataset/README.md",
        "gguf-tools/quality-testing/README.md",
    ]
    files += [str(p.relative_to(root)) for p in sorted((root / "metal").glob("*.metal"))]
    tasks = [
        "Review this excerpt for correctness risks, memory lifetime issues, and performance bottlenecks.",
        "Explain what this code does and identify the inference stage it belongs to.",
        "Suggest a minimal correctness-preserving optimization for this excerpt.",
        "Trova bug sottili e spiega quali invarianti dell'inferenza possono rompersi.",
    ]

    for name in files:
        text = read_text(root / name)
        if not text:
            continue
        is_code = name.endswith((".c", ".h", ".m", ".metal"))
        size = 2600 if is_code else 3200
        chunks = chunk_text(text, size=size, overlap=180)
        for idx, chunk in enumerate(chunks):
            task = tasks[(idx + len(name)) % len(tasks)]
            msgs = code_prompt(name, chunk, task) if is_code else doc_prompt(name, chunk, task)
            add_record(records, "source", f"{name}:{idx}", msgs)


def make_general_records(records: list[Record]) -> None:
    english = [
        "Explain how a B-tree insertion works, including splits and the root special case.",
        "Write a concise design for a TCP echo server that handles slow clients without blocking other clients.",
        "Compare mmap-backed model loading with copying all weights into private buffers on macOS.",
        "Derive why RMSNorm can be implemented with one sum of squares and one scale pass.",
        "Explain indexed sparse attention in a mixture-of-experts language model.",
        "Create a careful migration plan from a C++ graph executor to a pure C model-specific executor.",
        "Find the bug in a ring buffer where speculative writes may overwrite still-visible rows.",
        "Explain why quantization imatrices should be collected from realistic prompts instead of random text.",
        "Summarize the difference between prefill and decode in transformer inference.",
        "Write a troubleshooting guide for an OpenAI-compatible local chat server that hangs during tool calls.",
    ]
    italian = [
        "Spiega come funziona l'inserimento in un B-tree, inclusi split e caso della radice.",
        "Scrivi un progetto conciso per un server TCP echo che gestisce client lenti senza bloccare gli altri.",
        "Confronta il caricamento mmap dei pesi con la copia completa in buffer privati su macOS.",
        "Deriva perche RMSNorm richiede una somma dei quadrati e un passaggio di scala.",
        "Spiega l'attenzione sparsa indicizzata in un modello linguistico mixture-of-experts.",
        "Crea un piano di migrazione da un executor a grafo C++ a un executor specifico in C puro.",
        "Trova il bug in un ring buffer dove scritture speculative possono sovrascrivere righe ancora visibili.",
        "Spiega perche le imatrix di quantizzazione vanno raccolte da prompt realistici e non testo casuale.",
        "Riassumi la differenza tra prefill e decode nell'inferenza transformer.",
        "Scrivi una guida di debug per un server chat locale compatibile OpenAI che si blocca nei tool call.",
    ]
    for idx, prompt in enumerate(english + italian):
        add_record(records, "general", f"general:{idx}", [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ])


def make_programming_records(records: list[Record]) -> None:
    """Add compact coding prompts across common languages and shells.

    These are not training targets; they give the imatrix a broader mix of
    syntax, identifiers, punctuation, and task wording that users normally send
    to DS4 outside full agent/tool sessions.
    """

    prompts = [
        ("c", "Find the lifetime bug in this C function and give a minimal patch:\n\n"
              "char *join(const char *a,const char *b){char tmp[256];snprintf(tmp,sizeof tmp,\"%s/%s\",a,b);return tmp;}"),
        ("c", "Explain why this C ring buffer can overwrite unread data, then rewrite the push condition."),
        ("cpp", "Convert this C++ RAII wrapper to avoid double-close after move assignment."),
        ("cpp", "Review a C++17 template that stores string_view keys and explain the dangling-reference risk."),
        ("python", "Rewrite this Python log parser so it streams lines instead of loading the whole file."),
        ("python", "Given a failing pytest fixture that mutates global state, explain the bug and propose a fix."),
        ("rust", "Explain the borrow-checker error in a Rust iterator adapter that returns references into a temporary Vec."),
        ("rust", "Implement a Rust LRU cache skeleton using HashMap plus a linked list index arena."),
        ("go", "Find the data race in this Go worker pool and show where the mutex or channel ownership belongs."),
        ("go", "Write a small Go HTTP handler that validates JSON input and returns structured errors."),
        ("javascript", "Debug this JavaScript async loop that accidentally runs requests sequentially."),
        ("typescript", "Design TypeScript types for a tool-call JSON schema with discriminated union variants."),
        ("sql", "Optimize this SQL query with a covering index and explain why the current plan scans too many rows."),
        ("sql", "Translate a denormalized event table query into a window-function query with row_number()."),
        ("bash", "Write a Bash script that resumes a curl download, validates SHA256, and updates a symlink atomically."),
        ("bash", "Review this Bash snippet for quoting bugs: for f in $FILES; do cp $f $DEST; done"),
        ("java", "Explain how to make this Java cache thread-safe without synchronizing the whole hot path."),
        ("java", "Refactor a Java method that catches Exception and hides interrupted status."),
        ("swift", "Write a Swift function that parses command-line flags into a small struct with defaults."),
        ("zig", "Explain how Zig error unions differ from nullable pointers in a file-reading helper."),
        ("lua", "Write a Lua function that memoizes pure function calls with string keys."),
        ("php", "Find the SQL injection risk in a PHP PDO snippet and rewrite it with bound parameters."),
        ("ruby", "Convert a Ruby script from slurping a file to enumerating lines lazily."),
        ("shell", "Explain when POSIX sh is preferable to Bash, and rewrite [[ -f \"$x\" ]] for POSIX sh."),
    ]
    for idx, (lang, prompt) in enumerate(prompts):
        add_record(records, "programming", f"{lang}:{idx}", [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ])


def make_algorithm_records(records: list[Record]) -> None:
    prompts = [
        "Recall Dijkstra's algorithm, state its invariant, and write pseudocode for a binary heap implementation.",
        "Explain why BFS gives shortest paths in an unweighted graph and show the queue invariant.",
        "Implement union-find with path compression and union by size; explain amortized complexity.",
        "Compare merge sort and quicksort for linked lists, including cache behavior and stability.",
        "Derive the dynamic programming recurrence for edit distance and show how to reduce memory to two rows.",
        "Explain KMP prefix-function construction using the string 'ababaca' as an example.",
        "Describe topological sorting with cycle detection and produce clear pseudocode.",
        "Explain a Fenwick tree: update, prefix sum, and why i += i & -i works.",
        "Implement an LRU cache with O(1) get and put, naming every pointer update edge case.",
        "Explain reservoir sampling and prove every item has probability k/n of being retained.",
        "Recall A* search and explain the difference between admissible and consistent heuristics.",
        "Show how to detect an integer overflow in binary search midpoint calculation.",
        "Explain suffix arrays at a high level and when they are preferable to suffix trees.",
        "Write pseudocode for Tarjan strongly connected components and explain lowlink.",
        "Explain consistent hashing and what happens when one node is added or removed.",
        "Derive binary exponentiation and extend it to modular matrix exponentiation.",
    ]
    for idx, prompt in enumerate(prompts):
        add_record(records, "algorithms", f"algorithm:{idx}", [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ])


def make_translation_records(records: list[Record]) -> None:
    languages = [
        ("Italian", "italiano"),
        ("French", "français"),
        ("Spanish", "español"),
        ("German", "Deutsch"),
        ("Portuguese", "português"),
        ("Dutch", "Nederlands"),
        ("Polish", "polski"),
        ("Romanian", "română"),
        ("Swedish", "svenska"),
        ("Chinese", "简体中文"),
    ]
    source_sentences = [
        "The server stores a checkpoint only after the prompt prefix is stable.",
        "Please keep the answer concise, but do not omit the important edge cases.",
        "This model is running locally and does not send the prompt to a remote API.",
        "The script should resume the download and verify the checksum before replacing the file.",
        "Translate the user-facing error without changing file names, flags, or code identifiers.",
        "The algorithm uses a heap, a visited set, and a predecessor map to reconstruct the path.",
        "If the cache is cold, prefill is slower; once the prefix is reused, generation can start sooner.",
        "The database migration must preserve existing rows and remain safe under concurrent writes.",
    ]

    idx = 0
    for english_name, native_name in languages:
        for text in source_sentences:
            add_record(records, "translation", f"en-to-{english_name.lower()}:{idx}", [
                {"role": "system", "content": (
                    "You are a precise technical translator. Preserve code, file "
                    "paths, command-line flags, and product names exactly."
                )},
                {"role": "user", "content": (
                    f"Translate this English text to {english_name} ({native_name}):\n\n{text}"
                )},
            ])
            idx += 1

    reverse_prompts = [
        ("Italian", "Il server salva il checkpoint solo quando il prefisso del prompt è stabile."),
        ("French", "Le script doit reprendre le téléchargement et vérifier la somme de contrôle."),
        ("Spanish", "La migración de la base de datos debe conservar las filas existentes."),
        ("German", "Die Antwort soll kurz bleiben, aber wichtige Randfälle nicht auslassen."),
        ("Portuguese", "O modelo roda localmente e não envia o prompt para uma API remota."),
        ("Dutch", "Vertaal de foutmelding zonder bestandsnamen of opties te wijzigen."),
        ("Polish", "Algorytm używa kopca, zbioru odwiedzonych wierzchołków i mapy poprzedników."),
        ("Romanian", "Dacă memoria cache este rece, prefill-ul este mai lent."),
        ("Swedish", "Behåll kodidentifierare och kommandoradsflaggor oförändrade."),
        ("Chinese", "如果缓存是冷的，预填充会更慢；复用前缀后，生成可以更早开始。"),
    ]
    for idx, (language, text) in enumerate(reverse_prompts):
        add_record(records, "translation", f"{language.lower()}-to-en:{idx}", [
            {"role": "system", "content": (
                "You are a precise technical translator. Preserve code, file "
                "paths, command-line flags, and product names exactly."
            )},
            {"role": "user", "content": f"Translate this {language} text to English:\n\n{text}"},
        ])


def make_long_context_records(root: Path, records: list[Record]) -> None:
    sources = [
        ("README.md", read_text(root / "README.md")),
        ("AGENT.md", read_text(root / "AGENT.md")),
        ("METAL.md", read_text(root / "METAL.md")),
        ("ds4_server.c", read_text(root / "ds4_server.c")),
        ("ds4_metal.m", read_text(root / "ds4_metal.m")),
        ("metal/dsv4_hc.metal", read_text(root / "metal/dsv4_hc.metal")),
        ("metal/moe.metal", read_text(root / "metal/moe.metal")),
    ]
    blocks = []
    for name, text in sources:
        for i, chunk in enumerate(chunk_text(text, size=5200, overlap=120)[:6]):
            blocks.append(f"### {name} chunk {i}\n{chunk}")
    random.Random(7).shuffle(blocks)
    for i in range(0, len(blocks), 4):
        group = blocks[i:i + 4]
        if len(group) < 2:
            continue
        body = "\n\n".join(group)
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": (
                "Read the following repository excerpts as one long context. "
                "Identify the three most important invariants for correctness and "
                "the two most likely performance bottlenecks.\n\n" + body
            )},
        ]
        add_record(records, "long_context", f"long:{i//4}", msgs)


# --- Agent / tool-call calibration --------------------------------------------
#
# These are the records that actually shape how the routed MoE experts handle
# tool-attached chat.  Keep them realistic in shape and size, not just count.

SWIVAL_SHORT_SYSTEM = (
    "You are a coding agent operating in a local repository. Be terse. "
    "Use tools for filesystem inspection and edits; answer plain questions "
    "directly. Never narrate what you are about to do."
)

SWIVAL_LONG_SYSTEM = (
    "You are Swival, a coding agent with a friendly, human-like personality. "
    "You solve tasks autonomously using the tools provided, taking the optimal "
    "decisions at every step. Keep going until the task is fully complete. "
    "Do not call tools for simple math, greetings, or unclear standalone "
    "questions. For minor ambiguity, pick the most likely intent and briefly "
    "state your choice.\n\n"
    "## Workflow\n\n"
    "- Use tools only when needed; answer plain questions directly.\n"
    "- Explore before editing: outline, list_files, grep, then read_file.\n"
    "- Use think before multi-step coding tasks, debugging, choosing "
    "alternatives, and editing code.\n"
    "- Use todo for multi-step work and mark items done as you finish.\n"
    "- One logical change at a time. Do not re-read files after editing.\n"
    "- On tool errors, use think to diagnose before retrying.\n\n"
    "## Editing\n\n"
    "- Copy old_string verbatim from read_file.\n"
    "- For multiple matches, pass line_number. Use replace_all only when every "
    "occurrence should change.\n\n"
    "## Safety\n\n"
    "- Never print secrets to the console.\n"
    "- Do not take any action that could degrade the workspace."
)

GENERIC_CLI_SYSTEM = (
    "You are an interactive CLI tool that helps users with software "
    "engineering tasks. Use the instructions below and the tools available "
    "to you to assist the user.\n\n"
    "IMPORTANT: Refuse to write malicious code.\n"
    "IMPORTANT: Before you begin work, think about what the code you are "
    "editing is supposed to do based on the filenames and directory structure."
)

OPENCODE_SYSTEM = (
    "You are opencode, a terminal-based coding assistant. Resolve user "
    "tasks by inspecting the repository, planning a minimal change, and "
    "applying it. Use the bash, read, edit, and grep tools as needed."
)

ITALIAN_SYSTEM = (
    "Sei un assistente di programmazione che opera in un repository locale. "
    "Usa gli strumenti per ispezionare i file e applicare modifiche. "
    "Rispondi in italiano. Sii conciso."
)


# Tool schema fragments, in OpenAI tool format (the runtime strips the envelope).
TOOL_BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to execute."},
                "description": {"type": "string", "description": "Short description of intent."},
                "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
            },
            "required": ["command"],
        },
    },
}

TOOL_READ = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a region of a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "default": 1},
                "lines": {"type": "integer", "default": 200},
            },
            "required": ["path"],
        },
    },
}

TOOL_GREP = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search files for a regex pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string", "description": "Filename glob filter."},
                "ignore_case": {"type": "boolean", "default": False},
            },
            "required": ["pattern"],
        },
    },
}

TOOL_LIST = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files matching a glob pattern, depth-limited.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "max_depth": {"type": "integer", "default": 3},
            },
            "required": ["pattern"],
        },
    },
}

TOOL_EDIT = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Apply a small old/new text edit to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "line_number": {"type": "integer", "description": "Disambiguate when old_string occurs more than once."},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

TOOL_TODO = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": "Maintain a todo list that survives context compaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "done": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

TOOL_THINK = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Record private reasoning. The text is not shown to the user.",
        "parameters": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
}

TOOL_FETCH = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch a URL and return the body as text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
                "timeout": {"type": "integer", "default": 20},
                "headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
        },
    },
}

TOOLSETS = {
    "shell-only": [TOOL_BASH],
    "read-only": [TOOL_READ, TOOL_GREP, TOOL_LIST],
    "edit": [TOOL_READ, TOOL_GREP, TOOL_LIST, TOOL_EDIT],
    "edit+shell": [TOOL_BASH, TOOL_READ, TOOL_GREP, TOOL_LIST, TOOL_EDIT],
    "agent-full": [TOOL_BASH, TOOL_READ, TOOL_GREP, TOOL_LIST, TOOL_EDIT, TOOL_TODO, TOOL_THINK],
    "agent-net": [TOOL_BASH, TOOL_READ, TOOL_GREP, TOOL_LIST, TOOL_EDIT, TOOL_FETCH],
}


SYSTEMS = {
    "swival-short": SWIVAL_SHORT_SYSTEM,
    "swival-long": SWIVAL_LONG_SYSTEM,
    "generic-cli": GENERIC_CLI_SYSTEM,
    "opencode": OPENCODE_SYSTEM,
    "italian": ITALIAN_SYSTEM,
}


# Small library of realistic tool_result bodies.  Some carry characters that
# exercise the &/</> escape path; some are long enough to stress prefill in
# the tool-result region; one is a fake error string.
TOOL_RESULTS = {
    "ls": (
        ".git\nREADME.md\nds4.c\nds4_server.c\nds4_cli.c\nds4.h\nds4_metal.m\n"
        "metal/\nmisc/\ntests/\ngguf-tools/\nspeed-bench/\nMakefile\n"
    ),
    "ls-long": "\n".join(
        f"src/{group}/file_{i:03d}.c" for group in ("core", "metal", "io", "test") for i in range(64)
    ),
    "grep-hit": (
        "ds4_server.c:1646:static void append_tools_prompt_text(buf *b, const char *tool_schemas) {\n"
        "ds4_server.c:1922:        append_tools_prompt_text(&system, tool_schemas);\n"
    ),
    "grep-empty": "",
    "grep-error": "grep: bad regex: unbalanced parenthesis\n",
    "read-code": (
        "static int sample_argmax(const float *logits, int n) {\n"
        "    int best = 0;\n    float best_v = logits[0];\n"
        "    for (int i = 1; i < n; i++) {\n        if (logits[i] > best_v) {\n"
        "            best_v = logits[i];\n            best = i;\n        }\n    }\n"
        "    return best;\n}\n"
    ),
    "read-html": (
        "<html><body>\n"
        "<h1>Server status</h1>\n"
        "<pre>load=0.32 mem=78% temp=64C</pre>\n"
        "<a href=\"/\">home</a> &amp; <a href=\"/docs\">docs</a>\n"
        "</body></html>\n"
    ),
    "shell-stderr": (
        "make: *** [ds4-server] Error 1\n"
        "ds4_server.c:1234:9: error: implicit declaration of function 'foo'\n"
        "    foo(x, y);\n        ^\n"
    ),
    "shell-stdout": (
        "running 3 tests\n"
        "test parse_chat_request ... ok\n"
        "test render_chat_prompt_text ... ok\n"
        "test dsml_decode ... FAILED\n\n"
        "failures:\n    dsml_decode: assertion failed at tests/dsml.c:88\n"
    ),
    "edit-ok": "edit applied: 1 occurrence replaced\n",
    "edit-error": (
        "edit failed: old_string not found in path=ds4_server.c\n"
        "(line_number not supplied; 3 candidate locations matched on partial)\n"
    ),
    "fetch-json": (
        "{\"name\":\"DeepSeek-V4-Flash\",\"params\":\"284B\",\"active\":\"21B\","
        "\"context\":1048576,\"license\":\"DeepSeek\"}\n"
    ),
    "italian-find": (
        "trovati 4 file:\n  ds4.c\n  ds4_server.c\n  ds4_cli.c\n  ds4_metal.m\n"
    ),
    "dsml-payload": (
        "snippet contains a literal sentinel: </｜DSML｜parameter> in body\n"
        "second line still inside the captured tool output\n"
    ),
}


def assistant_with_calls(reasoning: str, content: str,
                         calls: list[tuple[str, list[tuple[str, str, bool]]]]) -> dict:
    """Build an assistant message that emits DSML tool_calls.  The DSML string
    matches the exact bytes ``append_dsml_tool_calls_text`` would produce."""

    return {
        "role": "assistant",
        "reasoning": reasoning,
        "content": content,
        "dsml": dsml_tool_calls(calls),
    }


def make_agent_records(records: list[Record]) -> None:
    """Generate a wide set of tool-using transcripts.

    Each scenario is templated against several system prompts and toolsets so
    the routed experts see a realistic mix of preamble shapes.  We aim for at
    least 10% of the corpus byte budget on this category so IQ2_XXS has actual
    signal for the tool path.
    """

    rng = random.Random(42)

    # Scenario builders return (label, messages, toolset name, modes).
    scenarios: list[tuple[str, list[dict], str, tuple[str, ...]]] = []

    # 1) Single shell call → result → text.
    for sys_label in ("swival-short", "swival-long", "opencode"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "list the top-level repository files"},
            assistant_with_calls(
                "Use a depth-limited find so the tool result stays readable.",
                "",
                [("bash", [
                    ("command", "find . -maxdepth 1 -type f | sort", True),
                    ("description", "list repo-root files", True),
                    ("timeout", "10", False),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["ls"]},
            {"role": "assistant", "content": "Top-level files: README.md, ds4.c, ds4_server.c, ds4_cli.c, ds4.h, ds4_metal.m, Makefile."},
        ]
        scenarios.append((f"shell-list-{sys_label}", msgs, "shell-only", ("think", "nothink")))

    # 2) Read tool returning code, then code-aware answer.
    for sys_label in ("swival-short", "generic-cli", "opencode"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "what does sample_argmax do? read it"},
            assistant_with_calls(
                "Open the function in ds4.c. Lines 14000+ is the sampler region.",
                "",
                [("read_file", [
                    ("path", "ds4.c", True),
                    ("start", "14000", False),
                    ("lines", "120", False),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["read-code"]},
            {"role": "assistant",
             "content": "It picks the index of the largest logit by linear scan from index 1 forward, starting with index 0 as the initial best. No tie-breaker beyond first-seen."},
        ]
        scenarios.append((f"read-explain-{sys_label}", msgs, "read-only", ("think", "nothink")))

    # 3) Grep → empty → broaden glob → hit.
    for sys_label in ("swival-long", "opencode"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "where do we render the tool-call preamble?"},
            assistant_with_calls(
                "Try a tight pattern first; broaden if empty.",
                "",
                [("grep", [
                    ("pattern", "tool_calls_preamble", True),
                    ("path", ".", True),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["grep-empty"]},
            assistant_with_calls(
                "No hits. The actual symbol is append_tools_prompt_text.",
                "",
                [("grep", [
                    ("pattern", "append_tools_prompt_text", True),
                    ("path", ".", True),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["grep-hit"]},
            {"role": "assistant",
             "content": "It is in ds4_server.c at line 1646, called from render_chat_prompt_text at 1922."},
        ]
        scenarios.append((f"grep-recover-{sys_label}", msgs, "edit", ("think", "nothink")))

    # 4) Edit attempt → error → retry with line_number.
    for sys_label in ("swival-long", "generic-cli"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "change the prefill chunk size from 2048 to 4096"},
            assistant_with_calls(
                "The constant likely appears more than once. Try a direct edit, fall back to line_number on failure.",
                "",
                [("edit", [
                    ("path", "ds4_server.c", True),
                    ("old_string", "prefill_chunk=2048", True),
                    ("new_string", "prefill_chunk=4096", True),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["edit-error"]},
            assistant_with_calls(
                "Multiple matches. Use read_file then line_number-scoped edit.",
                "",
                [("read_file", [
                    ("path", "ds4_server.c", True),
                    ("start", "1", False),
                    ("lines", "40", False),
                ])],
            ),
            {"role": "tool", "content": "1: #include \"ds4.h\"\n2: #include \"rax.h\"\n..."},
            {"role": "assistant",
             "content": "I need the actual line. Will re-scan with grep -n before editing."},
        ]
        scenarios.append((f"edit-retry-{sys_label}", msgs, "edit", ("think", "nothink")))

    # 5) Two-call block (parallel tools in a single DSML envelope).
    for sys_label in ("swival-short", "opencode"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "tell me about ds4.c and ds4_server.c"},
            assistant_with_calls(
                "Both reads in one DSML block; the runtime handles ordered tool_calls.",
                "",
                [
                    ("read_file", [("path", "ds4.c", True), ("start", "1", False), ("lines", "60", False)]),
                    ("read_file", [("path", "ds4_server.c", True), ("start", "1", False), ("lines", "60", False)]),
                ],
            ),
            {"role": "tool", "content": "ds4.c first 60 lines elided ..."},
            {"role": "tool", "content": "ds4_server.c first 60 lines elided ..."},
            {"role": "assistant",
             "content": "ds4.c is the model graph runtime; ds4_server.c is the HTTP/chat layer that drives it."},
        ]
        scenarios.append((f"parallel-read-{sys_label}", msgs, "edit", ("think", "nothink")))

    # 6) Output containing the DSML closing sentinel — exercises escape path.
    for sys_label in ("swival-long",):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "show what a problematic tool body looks like"},
            assistant_with_calls(
                "Demonstrate the escape rule: a literal closing tag inside string content.",
                "",
                [("bash", [
                    ("command", "cat tmp/dsml-sentinel.txt", True),
                    ("description", "print a file that contains the DSML sentinel", True),
                ])],
            ),
            {"role": "tool", "content": TOOL_RESULTS["dsml-payload"]},
            {"role": "assistant",
             "content": "The body contains a literal </｜DSML｜parameter> inside the tool output. The renderer escapes the leading character to &lt;/｜DSML｜parameter> so the model never sees a stray closing tag."},
        ]
        scenarios.append((f"dsml-escape-{sys_label}", msgs, "edit+shell", ("think", "nothink")))

    # 7) HTML output (escaped chars in tool result, common with web_fetch).
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "fetch http://localhost/status and summarize"},
        assistant_with_calls(
            "Plain GET; no body needed.",
            "",
            [("web_fetch", [
                ("url", "http://localhost/status", True),
                ("timeout", "20", False),
            ])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["read-html"]},
        {"role": "assistant",
         "content": "Server reports load 0.32, memory 78%, temperature 64C. The body is plain HTML with two links."},
    ]
    scenarios.append(("web-fetch-html", msgs, "agent-net", ("think", "nothink")))

    # 8) JSON response from a fetch, no further calls.
    msgs = [
        {"role": "system", "content": OPENCODE_SYSTEM},
        {"role": "user", "content": "what context window does this model report?"},
        assistant_with_calls(
            "Hit the local metadata endpoint.",
            "",
            [("web_fetch", [
                ("url", "http://127.0.0.1:8000/v1/models/default", True),
            ])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["fetch-json"]},
        {"role": "assistant",
         "content": "It advertises a 1,048,576-token context window. Active params 21B, total 284B, DeepSeek license."},
    ]
    scenarios.append(("fetch-json", msgs, "agent-net", ("think", "nothink")))

    # 9) Italian agent loop.
    msgs = [
        {"role": "system", "content": ITALIAN_SYSTEM},
        {"role": "user", "content": "elenca i file C principali"},
        assistant_with_calls(
            "Uso un find limitato in profondità per restare conciso.",
            "",
            [("bash", [
                ("command", "find . -maxdepth 1 -name '*.c' | sort", True),
                ("description", "elenca i sorgenti C nella radice", True),
            ])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["italian-find"]},
        {"role": "assistant", "content": "I quattro file C principali sono ds4.c, ds4_server.c, ds4_cli.c e ds4_metal.m."},
    ]
    scenarios.append(("italian-shell", msgs, "shell-only", ("think", "nothink")))

    # 10) Long ls output → grep into it.
    msgs = [
        {"role": "system", "content": SWIVAL_SHORT_SYSTEM},
        {"role": "user", "content": "find files in src/metal"},
        assistant_with_calls(
            "Use a glob list_files call so the tool output is structured.",
            "",
            [("list_files", [("pattern", "src/metal/**", True), ("max_depth", "5", False)])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["ls-long"]},
        {"role": "assistant", "content": "256 files across src/core, src/metal, src/io, src/test. Want me to focus on src/metal?"},
    ]
    scenarios.append(("long-ls", msgs, "read-only", ("think", "nothink")))

    # 11) Shell test runner with mixed stdout/stderr.
    msgs = [
        {"role": "system", "content": GENERIC_CLI_SYSTEM},
        {"role": "user", "content": "run the test suite"},
        assistant_with_calls(
            "Run with low timeout so a hang does not stall the loop.",
            "",
            [("bash", [
                ("command", "make -s test", True),
                ("description", "run the test suite", True),
                ("timeout", "180", False),
            ])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["shell-stdout"]},
        {"role": "assistant",
         "content": "Two passes, one failure in dsml_decode at tests/dsml.c:88. Open the file?"},
    ]
    scenarios.append(("test-run", msgs, "edit+shell", ("think", "nothink")))

    # 12) Build failure → diagnose.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "build the server"},
        assistant_with_calls(
            "Run make and surface the first error.",
            "",
            [("bash", [
                ("command", "make ds4-server 2>&1 | head -40", True),
                ("description", "build ds4-server", True),
                ("timeout", "600", False),
            ])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["shell-stderr"]},
        {"role": "assistant",
         "content": "Build broke at ds4_server.c:1234 on an implicit declaration of foo. Likely a missing include after a refactor."},
    ]
    scenarios.append(("build-fail", msgs, "edit+shell", ("think", "nothink")))

    # 13) Todo planning at the start of a multi-step task.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "add a /healthz endpoint to the server"},
        assistant_with_calls(
            "Plan first, then implement. Three steps.",
            "",
            [("todo", [
                ("items", "[\"locate route table\",\"add handler\",\"wire test\"]", False),
            ])],
        ),
        {"role": "tool", "content": "todo updated: 3 open, 0 done\n"},
        assistant_with_calls(
            "Step 1: find the route table.",
            "",
            [("grep", [("pattern", "/v1/chat/completions", True), ("path", "ds4_server.c", True)])],
        ),
        {"role": "tool", "content": "ds4_server.c:7600:    if (!strcmp(path, \"/v1/chat/completions\")) {\n"},
        {"role": "assistant",
         "content": "Routes dispatch off the path string in client_main near line 7600. I will add the /healthz arm there."},
    ]
    scenarios.append(("todo-plan", msgs, "agent-full", ("think", "nothink")))

    # 14) think-tool noted reasoning, then act.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "should we cache the imatrix file on disk?"},
        assistant_with_calls(
            "Record the considerations privately before recommending.",
            "",
            [("think", [
                ("note", "the .dat is read once per quant run; disk cache buys nothing the OS page cache does not", True),
            ])],
        ),
        {"role": "tool", "content": "noted\n"},
        {"role": "assistant",
         "content": "No. The file is read once per quant run and the OS page cache already keeps it hot if you re-run shortly after."},
    ]
    scenarios.append(("think-then-answer", msgs, "agent-full", ("think", "nothink")))

    # 15) Plain greeting that does NOT call tools (must teach the model to skip
    # tools when none are needed; this is the most common swival entry point).
    for sys_label in ("swival-short", "swival-long", "generic-cli", "opencode", "italian"):
        msgs = [
            {"role": "system", "content": SYSTEMS[sys_label]},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hello. What would you like to work on?"},
        ]
        scenarios.append((f"greet-{sys_label}", msgs, "agent-full", ("think", "nothink")))

    # 16) Direct factual answer with tools attached but unused.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "what is the difference between prefill and decode?"},
        {"role": "assistant",
         "content": "Prefill processes the prompt tokens in batched matmuls to fill the KV cache; decode then runs one token at a time using the cached state. Prefill is bandwidth-bound on large prompts; decode is dominated by per-token attention scan plus MoE routing."},
    ]
    scenarios.append(("factual-no-tools", msgs, "agent-full", ("think", "nothink")))

    # 17) Italian factual answer with tools attached but unused.
    msgs = [
        {"role": "system", "content": ITALIAN_SYSTEM},
        {"role": "user", "content": "qual e la differenza tra prefill e decode?"},
        {"role": "assistant",
         "content": "Il prefill elabora i token del prompt in matmul batch per riempire la KV cache; il decode procede un token alla volta usando lo stato cachato. Il prefill e limitato dalla banda di memoria sui prompt lunghi; il decode dal costo per token dell'attenzione e del routing MoE."},
    ]
    scenarios.append(("factual-it-no-tools", msgs, "agent-full", ("think", "nothink")))

    # 18) Tool-only turn (assistant emits a call as its first action, no text).
    msgs = [
        {"role": "system", "content": OPENCODE_SYSTEM},
        {"role": "user", "content": "where is the Makefile target for tests defined?"},
        assistant_with_calls(
            "Open the Makefile rather than guessing.",
            "",
            [("read_file", [("path", "Makefile", True), ("start", "1", False), ("lines", "120", False)])],
        ),
        {"role": "tool", "content": "all: ds4 ds4-server ds4-bench\n\nds4-server: ds4_server.o rax.o $(CORE_OBJS)\n..."},
        {"role": "assistant",
         "content": "There is no 'test' phony target. Tests are built ad hoc per Makefile. We can add one if you want."},
    ]
    scenarios.append(("first-action-tool", msgs, "read-only", ("think", "nothink")))

    # 19) Iterated agent loop: 4 tool turns before the final text answer.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "investigate why ds4-server logs 'shutdown requested' mid-request"},
        assistant_with_calls(
            "Find the only place that emits this string.",
            "",
            [("grep", [("pattern", "shutdown requested", True), ("path", "ds4_server.c", True)])],
        ),
        {"role": "tool",
         "content": "ds4_server.c:7297:        snprintf(err, sizeof(err), \"shutdown requested\");\nds4_server.c:8199:    server_log(DS4_LOG_DEFAULT, \"ds4-server: shutdown requested, draining requests\");\n"},
        assistant_with_calls(
            "Open the conditional that sets the error.",
            "",
            [("read_file", [("path", "ds4_server.c", True), ("start", "7290", False), ("lines", "20", False)])],
        ),
        {"role": "tool",
         "content": "    if (g_stop_requested && strcmp(finish, \"error\") != 0) {\n        finish = \"error\";\n        snprintf(err, sizeof(err), \"shutdown requested\");\n    }\n"},
        assistant_with_calls(
            "Find the writers of g_stop_requested.",
            "",
            [("grep", [("pattern", "g_stop_requested\\s*=", True), ("path", "ds4_server.c", True)])],
        ),
        {"role": "tool",
         "content": "ds4_server.c:46:    g_stop_requested = 1;\n"},
        assistant_with_calls(
            "Confirm only the signal handler writes it.",
            "",
            [("read_file", [("path", "ds4_server.c", True), ("start", "43", False), ("lines", "10", False)])],
        ),
        {"role": "tool",
         "content": "static void stop_signal_handler(int sig) {\n    (void)sig;\n    if (g_stop_requested) _exit(130);\n    g_stop_requested = 1;\n    ...\n}\n"},
        {"role": "assistant",
         "content": "g_stop_requested is set only by SIGINT/SIGTERM, so the mid-request 'shutdown requested' means the process received that signal. It is not raised by an internal timeout or by a client disconnect."},
    ]
    scenarios.append(("long-agent-loop", msgs, "agent-full", ("think", "nothink")))

    # 20) Italian multi-turn agent.
    msgs = [
        {"role": "system", "content": ITALIAN_SYSTEM},
        {"role": "user", "content": "trova dove viene parsato reasoning_effort"},
        assistant_with_calls(
            "Cerco il nome del campo nel server.",
            "",
            [("grep", [("pattern", "reasoning_effort", True), ("path", "ds4_server.c", True)])],
        ),
        {"role": "tool",
         "content": "ds4_server.c:710:static bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {\nds4_server.c:725:static bool parse_reasoning_effort_value(const char **p, ds4_think_mode *out) {\n"},
        assistant_with_calls(
            "Apro la funzione di parsing per vedere i valori accettati.",
            "",
            [("read_file", [("path", "ds4_server.c", True), ("start", "710", False), ("lines", "20", False)])],
        ),
        {"role": "tool",
         "content": "static bool parse_reasoning_effort_name(const char *s, ds4_think_mode *out) {\n    if (!s) return false;\n    if (!strcmp(s, \"max\")) ... 'xhigh' 'high' 'medium' 'low'\n}\n"},
        {"role": "assistant",
         "content": "Il parser accetta solo max, xhigh, high, medium, low. Il valore 'none' viene rifiutato come JSON invalido."},
    ]
    scenarios.append(("italian-agent-loop", msgs, "agent-full", ("think", "nothink")))

    # 21) Recovery from a grep that returns garbage stderr.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "find functions named *_kv_*"},
        assistant_with_calls(
            "First attempt with a regex that has an unbalanced group.",
            "",
            [("grep", [("pattern", "(_kv_", True), ("path", ".", True)])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["grep-error"]},
        assistant_with_calls(
            "Fix the regex.",
            "",
            [("grep", [("pattern", "_kv_", True), ("path", ".", True)])],
        ),
        {"role": "tool", "content": "ds4.c:88:static void ds4_kv_reset(...)\nds4.c:412:static int  ds4_kv_size(...)\n"},
        {"role": "assistant",
         "content": "Two candidates: ds4_kv_reset and ds4_kv_size, both in ds4.c. Want me to open either?"},
    ]
    scenarios.append(("grep-regex-error", msgs, "edit+shell", ("think", "nothink")))

    # 22) Editing with replace_all then verifying.
    msgs = [
        {"role": "system", "content": OPENCODE_SYSTEM},
        {"role": "user", "content": "rename the local variable tmpBuf to scratch_buf everywhere"},
        assistant_with_calls(
            "replace_all is safe here since the identifier is local-only.",
            "",
            [("edit", [
                ("path", "ds4_metal.m", True),
                ("old_string", "tmpBuf", True),
                ("new_string", "scratch_buf", True),
                ("replace_all", "true", False),
            ])],
        ),
        {"role": "tool", "content": "edit applied: 6 occurrences replaced\n"},
        {"role": "assistant", "content": "Renamed 6 occurrences in ds4_metal.m. Anything else?"},
    ]
    scenarios.append(("edit-replace-all", msgs, "edit", ("think", "nothink")))

    # 23) Tool call that returns an HTML-escape-heavy body, then text.
    msgs = [
        {"role": "system", "content": SWIVAL_LONG_SYSTEM},
        {"role": "user", "content": "fetch the local docs page"},
        assistant_with_calls(
            "Plain fetch; the body has HTML which the renderer will escape on the way in.",
            "",
            [("web_fetch", [("url", "http://localhost/docs", True)])],
        ),
        {"role": "tool", "content": TOOL_RESULTS["read-html"]},
        {"role": "assistant",
         "content": "Two pages linked from the index, plus a small load/mem/temp block. The body comes in escaped: &amp; for &, &lt; for <, &gt; for >."},
    ]
    scenarios.append(("html-escape-loop", msgs, "agent-net", ("think", "nothink")))

    # Emit every scenario across the requested modes.
    for label, msgs, toolset, modes in scenarios:
        tools = TOOLSETS[toolset]
        add_record(records, "agent", f"{toolset}:{label}", msgs,
                   tools_schema=tools, modes=modes)

    # Compound the corpus by replaying every scenario under additional
    # toolsets so the model sees diverse preambles around the same shapes.
    extra_toolsets = ["shell-only", "read-only", "edit", "edit+shell", "agent-full", "agent-net"]
    for label, msgs, original_toolset, modes in scenarios:
        for extra in extra_toolsets:
            if extra == original_toolset:
                continue
            tools = TOOLSETS[extra]
            add_record(records, "agent", f"{extra}:replay:{label}", msgs,
                       tools_schema=tools, modes=modes)

    # Add many short single-turn tool calls; these are cheap and dominate
    # decode-time activations on the assistant tool_call prefix.
    short_user_tool_pairs = [
        ("show me the current branch", "bash", [
            ("command", "git rev-parse --abbrev-ref HEAD", True),
            ("description", "current git branch", True),
        ], "main\n"),
        ("count lines in ds4.c", "bash", [
            ("command", "wc -l ds4.c", True),
            ("description", "line count of ds4.c", True),
        ], " 23456 ds4.c\n"),
        ("find TODO comments", "grep", [
            ("pattern", "TODO", True),
            ("path", ".", True),
        ], "ds4_server.c:842:    // TODO: revisit when streaming usage is finalized\n"),
        ("list .metal kernels", "list_files", [
            ("pattern", "metal/*.metal", True),
            ("max_depth", "2", False),
        ], "metal/dsv4_hc.metal\nmetal/moe.metal\nmetal/sampling.metal\n"),
        ("open Makefile", "read_file", [
            ("path", "Makefile", True),
            ("start", "1", False),
            ("lines", "40", False),
        ], "CC ?= cc\nCFLAGS ?= -O3 -ffast-math ...\n"),
        ("fix typo in README", "edit", [
            ("path", "README.md", True),
            ("old_string", "DrawfStar", True),
            ("new_string", "DwarfStar", True),
        ], "edit applied: 1 occurrence replaced\n"),
        ("note that streaming is on", "todo", [
            ("items", "[\"validate streaming\",\"add backpressure test\"]", False),
        ], "todo updated: 2 open, 0 done\n"),
        ("record decision about quant width", "think", [
            ("note", "stay at IQ2_XXS for routed gate/up; bump scratch buffers to fp16", True),
        ], "noted\n"),
        ("fetch model card", "web_fetch", [
            ("url", "http://127.0.0.1:8000/v1/models/default", True),
        ], TOOL_RESULTS["fetch-json"]),
    ]
    for idx, (q, tool, params, result) in enumerate(short_user_tool_pairs):
        for sys_label in ("swival-short", "swival-long", "generic-cli", "opencode", "italian"):
            base = [
                {"role": "system", "content": SYSTEMS[sys_label]},
                {"role": "user", "content": q},
                assistant_with_calls("Direct call; the answer is in the tool output.", "",
                                     [(tool, params)]),
                {"role": "tool", "content": result},
                {"role": "assistant", "content": "Done. Output above."},
            ]
            for ts in TOOLSETS:
                add_record(records, "agent", f"{ts}:short:{sys_label}:{idx}", base,
                           tools_schema=TOOLSETS[ts], modes=("think", "nothink"))

    # Light random walks: pick a system, a toolset, glue two short transcripts
    # together to vary turn counts.  Cap to avoid blowing past 30% agent share.
    glue_count = 80
    short_msgs_pool: list[list[dict]] = []
    for _, msgs, _, _ in scenarios:
        short_msgs_pool.append(msgs)
    for n in range(glue_count):
        a = rng.choice(short_msgs_pool)
        b = rng.choice(short_msgs_pool)
        merged: list[dict] = [a[0]]  # first system
        for msg in a[1:]:
            merged.append(msg)
        # Treat the second scenario as a follow-up: drop its system, keep
        # the rest.  This produces 6-12 turn transcripts that look like a
        # real session.
        for msg in b[1:]:
            merged.append(msg)
        ts = rng.choice(list(TOOLSETS.keys()))
        add_record(records, "agent", f"{ts}:glue:{n}", merged,
                   tools_schema=TOOLSETS[ts], modes=("think", "nothink"))


def write_outputs(outdir: Path, records: list[Record]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: (r.category, r.source, r.mode, r.rid))
    unique_records: list[Record] = []
    seen_rendered: set[str] = set()
    for r in records:
        if has_provider_reference(r.source) or has_provider_reference(r.rendered):
            continue
        if r.rendered in seen_rendered:
            continue
        seen_rendered.add(r.rendered)
        unique_records.append(r)
    records = unique_records

    jsonl = outdir / "prompts.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r.rid,
                "category": r.category,
                "mode": r.mode,
                "source": r.source,
                "messages": r.messages,
                "rendered": r.rendered,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write_rendered(path: Path, rows: Iterable[Record]) -> int:
        count = 0
        with path.open("w", encoding="utf-8") as f:
            for count, r in enumerate(rows, start=1):
                f.write(f"\n\n===== DS4_IMATRIX_PROMPT {r.rid} {r.category} {r.mode} {r.source} =====\n")
                f.write(r.rendered)
        return count

    write_rendered(outdir / "rendered_prompts.txt", records)
    write_rendered(outdir / "rendered_prompts_nothink.txt", [r for r in records if r.mode == "nothink"])
    write_rendered(outdir / "rendered_prompts_think.txt", [r for r in records if r.mode == "think"])

    categories: dict[str, int] = {}
    modes: dict[str, int] = {}
    bytes_by_category: dict[str, int] = {}
    for r in records:
        categories[r.category] = categories.get(r.category, 0) + 1
        modes[r.mode] = modes.get(r.mode, 0) + 1
        bytes_by_category[r.category] = bytes_by_category.get(r.category, 0) + len(r.rendered.encode("utf-8"))

    rendered_bytes = sum(len(r.rendered.encode("utf-8")) for r in records)
    manifest = {
        "version": 3,
        "purpose": "DeepSeek V4 Flash imatrix calibration prompts",
        "record_count": len(records),
        "rendered_utf8_bytes": rendered_bytes,
        "rough_token_estimate_bytes_div_4": rendered_bytes // 4,
        "categories": categories,
        "modes": modes,
        "bytes_by_category": bytes_by_category,
        "files": {
            "jsonl": "prompts.jsonl",
            "all_rendered": "rendered_prompts.txt",
            "nothink_rendered": "rendered_prompts_nothink.txt",
            "think_rendered": "rendered_prompts_think.txt",
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output directory. Defaults to this script's directory.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = find_repo_root(script_dir)
    outdir = Path(args.out).resolve() if args.out else script_dir

    records: list[Record] = []
    make_source_records(root, records)
    make_agent_records(records)
    make_general_records(records)
    make_programming_records(records)
    make_algorithm_records(records)
    make_translation_records(records)
    make_long_context_records(root, records)
    write_outputs(outdir, records)

    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    print(f"wrote {manifest['record_count']} prompts to {outdir}")
    print(f"rendered bytes: {manifest['rendered_utf8_bytes']}")
    print(f"rough tokens: {manifest['rough_token_estimate_bytes_div_4']}")
    for cat, count in sorted(manifest['categories'].items()):
        share = manifest['bytes_by_category'][cat] / manifest['rendered_utf8_bytes']
        print(f"  {cat}: {count} records, {share*100:.1f}% bytes")


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "ds4.c").exists() and (path / "metal").is_dir():
            return path
    raise RuntimeError(f"could not find ds4.c repository root from {start}")


if __name__ == "__main__":
    main()
