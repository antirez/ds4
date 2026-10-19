#!/usr/bin/env python3
"""Three-way template A/B, model-free.

Renders the same synthetic bodies with ds4's C renderer (through
`./ds4_test --qwen-render`) and with one or more Jinja chat templates, then
reports where the framing differs.  Both renderers copy message bytes verbatim,
so any difference here is template framing rather than content.  Nothing here
reads a model, a transcript, or a private corpus: scenarios are synthetic and
templates are supplied locally.

  python3 tests/template_parity_ab.py --template shipped=/tmp/shipped.jinja \
      --template fixed=/tmp/froggeric.jinja [--verbose] [--strict]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

IS = "<|im_start|>"
IE = "<|im_end|>"
TO = "<think>"
TC = "</think>"
ESC_SEQ = re.compile(r"\\u[0-9a-fA-F]{4}")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]},
    },
}]


# Some clients spell non-ASCII in tool schemas as \u2014 on the wire.  The
# templates render the schema through tojson, which decodes it and writes the
# character raw, so the escape spelling must not survive into the prompt.
ESCAPE_TOOLS_TEXT = (
    '[{"type":"function","function":{"name":"search",'
    '"description":"directory to search \\u2014 one path or a list",'
    '"parameters":{"type":"object","properties":{"path":{"type":"string",'
    '"description":"glob \\u2014 emoji \\ud83d\\ude00 and \\u0007 a bell"},'
    '"limit":{"type":"integer"}},"required":["path"]}}}]')
ESCAPE_TOOLS = json.loads(ESCAPE_TOOLS_TEXT)


def msg(role, content, reasoning=None, tool_call_id=None):
    out = {"role": role, "content": content}
    if reasoning is not None:
        out["reasoning_content"] = reasoning
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id
    return out


def call(name, arguments):
    return {"id": "c1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def two_turn_history():
    return [
        msg("user", "Hello"),
        msg("assistant", "Hi.", "stale chain"),
        msg("user", "Do the task"),
        msg("assistant", "Working.", "live chain"),
    ]


SCENARIOS = [
    # differs: ds4 canonicalizes the tool schema JSON compactly where the
    # shipped template emits spaced separators (pre-existing, not framing)
    {"name": "baseline-tool-loop", "expect": "differs", "effort": "high",
     "preserve": True, "tools": True,
     "messages": [
         msg("system", "You are terse."),
         msg("user", "List files"),
         {"role": "assistant", "content": "", "reasoning_content": "run ls",
          "tool_calls": [call("bash", {"command": "ls"})]},
         msg("tool", "a.txt\nb.txt", tool_call_id="c1"),
         msg("assistant", "Two files.", "counted them"),
         msg("user", "Now count lines"),
     ]},
    # differs: shipped prepends an empty pair before the client's own block, so
    # even a canonically formatted client block renders as two bodies
    {"name": "inlined-block-canonical", "expect": "differs", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [msg("user", "Explain"),
                  msg("assistant", TO + "\nhidden chain\n" + TC + "\n\nVisible")]},
    {"name": "inlined-block-client-spacing", "expect": "differs", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [msg("user", "Explain"),
                  msg("assistant", TO + "hidden chain" + TC + " Visible")]},
    {"name": "nonthinking-echo", "expect": "differs", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [msg("user", "Next"), msg("assistant", TC + "\n\nAnswer only")]},
    {"name": "duplicate-reasoning-channels", "expect": "differs", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [msg("user", "Again"),
                  msg("assistant", TO + "duplicate chain" + TC + " Visible",
                      "real chain")]},
    {"name": "quoted-block-mid-answer", "expect": "match", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [msg("user", "Quote it"),
                  msg("assistant", "Before " + TO + "inner" + TC + " after",
                      "quoted chain")]},
    {"name": "retention-archive", "expect": "match", "effort": "high",
     "preserve": True, "tools": False, "messages": two_turn_history()},
    # match: the shipped template windows on the last query too, once asked
    {"name": "retention-window", "expect": "match", "effort": "high",
     "preserve": False, "tools": False, "messages": two_turn_history()},
    {"name": "notice-only-user-turn", "expect": "match", "effort": "high",
     "preserve": True, "tools": False,
     "messages": [
         msg("user", "<system-notice>\nContinue the unfinished work.\n</system-notice>"),
         msg("assistant", "Resuming.", "notice is not a new task"),
         msg("user", "Actually stop"),
     ]},
    # shipped injects an extra reminder line at low effort; ds4 renders the
    # template's own low instruction only, so this one is a known divergence
    {"name": "effort-low", "expect": "differs", "effort": "low", "preserve": True,
     "tools": False, "messages": [msg("user", "Be brief")]},
    {"name": "effort-none", "expect": "match", "effort": "none", "preserve": True,
     "tools": False, "messages": [msg("user", "Be brief")]},
    # schema JSON spacing is a recorded divergence, so this scenario pins the
    # escape policy through the invariant rather than the whole-text match
    {"name": "tool-schema-escapes", "expect": "differs", "effort": "high",
     "preserve": True, "tool_text": ESCAPE_TOOLS_TEXT,
     "messages": [
         msg("user", "Find the archives"),
         msg("assistant", "", "Listing the archive paths first."),
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "c1", "type": "function",
              "function": {"name": "search", "arguments": json.dumps({"path": "src"})}}]},
         msg("tool", "12 archives", tool_call_id="c1"),
         msg("user", "Total segments?"),
     ]},
]


class _AddedTokens(dict):
    """A shipped template looks special token ids up on the tokenizer object HF
    passes to apply_chat_template.  Text scenarios never render them, so a
    missing key resolving to any id keeps the comparison about framing."""

    def __missing__(self, key):
        return 0


class _Tokenizer:
    class ggml:
        added_tokens = _AddedTokens()


TOKENIZER = _Tokenizer()

# The chat_template embedded in our shipped GGUF ends with this stray literal
# text (a broken special-token lookup that survived into the file), so the
# reference render grows by its length on every prompt.  Stripping it keeps the
# A/B about framing; the artifact itself is reported per scenario.
TEMPLATE_TAIL_ARTIFACT = "MISSING tokenizer.ggml.added_tokens\n"


def jinja_env():
    import jinja2

    env = jinja2.Environment(loader=jinja2.DictLoader({"t": ""}),
                             keep_trailing_newline=True)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
    env.globals["tojson"] = env.filters["tojson"]

    def raise_exception(text):
        raise RuntimeError("template raised: " + text)

    env.globals["raise_exception"] = raise_exception
    return env


def args_to_mapping(messages):
    """The shipped Qwen 3.8 template applies an items filter to tool-call
    arguments, so it renders only when arguments are a mapping.  Real OpenAI
    clients send a JSON string; convert so the reference can render at all."""
    out = json.loads(json.dumps(messages))
    converted = 0
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    converted += 1
                except ValueError:
                    fn["arguments"] = {"__raw__": fn["arguments"]}
    return out, converted


def ds4_render(binary, scenario):
    handles = []
    try:
        msgs_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        handles.append(msgs_file.name)
        json.dump(scenario["messages"], msgs_file)
        msgs_file.close()
        tools_path = "-"
        if scenario.get("tool_text"):
            tools_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                                     encoding="utf-8")
            handles.append(tools_file.name)
            tools_file.write(scenario["tool_text"])
            tools_file.close()
            tools_path = tools_file.name
        elif scenario.get("tools"):
            tools_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            handles.append(tools_file.name)
            json.dump(TOOLS, tools_file)
            tools_file.close()
            tools_path = tools_file.name
        proc = subprocess.run(
            [binary, "--qwen-render", scenario["effort"],
             "true" if scenario["preserve"] else "false", msgs_file.name, tools_path],
            capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit("ds4_test failed rc=%d: %s"
                             % (proc.returncode, proc.stderr[:200]))
        return proc.stdout
    finally:
        for path in handles:
            os.unlink(path)


def mask(text):
    for lit in (IS, IE, TO, TC):
        text = text.replace(lit, "[" + lit.strip("<>").replace("|", "") + "]")
    return text.replace("\n", "\\n")


def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", action="append", default=[], metavar="NAME=PATH",
                    help="reference Jinja template to A/B against")
    ap.add_argument("--binary", default="./ds4_test")
    ap.add_argument("--pythonpath", default="",
                    help="directory containing jinja2, e.g. a vendored MLX wheel tree")
    ap.add_argument("--effort", default="xhigh",
                    help="reasoning_effort passed to the templates")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a scenario differs from a reference")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.pythonpath:
        sys.path.insert(0, args.pythonpath)
    env = jinja_env()
    refs = []
    for spec in args.template:
        name, _, path = spec.partition("=")
        refs.append((name or path, env.from_string(open(path, encoding="utf-8").read())))
    if not refs:
        print("no --template given: reporting ds4 framing only", file=sys.stderr)

    differing = 0
    unexpected = 0
    for sc in SCENARIOS:
        out = ds4_render(args.binary, sc)
        cells = ["%-31s ds4=%7dB" % (sc["name"], len(out))]
        diffs_here = 0
        esc_bad = 0
        shipped_differs = None
        for name, tmpl in refs:
            msgs = sc["messages"]
            converted = 0
            if "shipped" in name:
                msgs, converted = args_to_mapping(msgs)
            tmpl_tools = (ESCAPE_TOOLS if sc.get("tool_text")
                          else TOOLS if sc.get("tools") else [])
            kwargs = {"messages": msgs, "tools": tmpl_tools,
                      "add_generation_prompt": sc["messages"][-1]["role"] != "assistant",
                      "enable_thinking": sc["effort"] != "none",
                      "reasoning_effort": args.effort if sc["effort"] != "none" else "none",
                      "preserve_thinking": sc["preserve"],
                      "tokenizer": TOKENIZER}
            try:
                ref = tmpl.render(**kwargs)
            except Exception as exc:
                cells.append("%s=ERROR(%s)" % (name, type(exc).__name__))
                diffs_here += 1
                continue
            cmp_ref = ref
            note = ""
            if ref.endswith(TEMPLATE_TAIL_ARTIFACT):
                cmp_ref = ref[:-len(TEMPLATE_TAIL_ARTIFACT)]
                note = " stripped-artifact:%d" % len(TEMPLATE_TAIL_ARTIFACT)
            same = cmp_ref == out
            diffs_here += 0 if same else 1
            ds4_esc, ref_esc = len(ESC_SEQ.findall(out)), len(ESC_SEQ.findall(cmp_ref))
            cells.append("%s=esc%d/%d%s" % (name, ds4_esc, ref_esc,
                                            "" if ds4_esc == ref_esc else " ESC-MISMATCH"))
            if ds4_esc != ref_esc:
                esc_bad += 1
            if "shipped" in name:
                shipped_differs = not same
            cells.append("%s=%7dB %s%s%s" % (name, len(ref), "==" if same else "!=",
                                             note,
                                             " args->mapping:%d" % converted if converted else ""))
            if not same and args.verbose:
                i = first_diff(out, cmp_ref)
                cells.append("\n      first diff at %d (%d%%):\n"
                             "        ds4:  %s\n        %s: %s"
                             % (i, 100 * i // max(len(out), 1),
                                mask(out[max(0, i - 60):i + 90]), name,
                                mask(cmp_ref[max(0, i - 60):i + 90])))
        opens, closes = out.count(TO), out.count(TC)
        cells.append("think=%d/%d%s" % (opens, closes,
                                        " UNBALANCED" if opens < closes else ""))
        print(" ".join(cells))
        differing += diffs_here
        if esc_bad:
            unexpected += 1
            print("      note: %s keeps a different number of \\u escape sequences than "
                  "the template, which should render none of them" % sc["name"])
        if shipped_differs is None:
            continue
        if sc["expect"] == "match" and shipped_differs:
            unexpected += 1
            print("      note: %s should render identically to the shipped template"
                  % sc["name"])
        elif sc["expect"] == "differs" and not shipped_differs:
            unexpected += 1
            print("      note: %s should diverge from the shipped framing" % sc["name"])

    print("\nscenarios=%d references=%d differing=%d unexpected=%d"
          % (len(SCENARIOS), len(refs), differing, unexpected))
    return 1 if args.strict and unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
