#!/usr/bin/env python3
"""Add rec's exact prompt token ids to tests/flint/golden.json.

collect_golden.py records what `rec serve` returns, which does not include
the prompt ids.  This renders every case's messages (and tools) with rec's
own tokenizer and chat template (rec_mlx.chat.load_tokenizer, the template
shipped inside the runtime) and stores them as `prompt_ids`, checking the
count against the `prompt_tokens` rec reported.  Runs inside rec's runtime
but loads no model, so it is safe next to anything else.

    tests/flint/golden_prompt_ids.py [--golden tests/flint/golden.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RUNTIMES = Path.home() / ".local/share/rec/runtimes"
MODELS = Path.home() / ".local/share/rec/models"


def newest(path: Path, marker: str) -> Path | None:
    cands = [d for d in path.iterdir() if (d / marker).exists()] if path.is_dir() else []
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path(__file__).resolve().parent / "golden.json")
    ap.add_argument("--runtime", type=Path, default=None)
    ap.add_argument("--src", type=Path, default=None)
    args = ap.parse_args()
    runtime = args.runtime or newest(RUNTIMES, "python/bin/python3")
    if runtime is None:
        sys.exit("no rec runtime extracted yet; run `rec serve` once")
    py = runtime / "python/bin/python3"
    if Path(sys.executable).resolve() != py.resolve():
        os.execv(str(py), [str(py), "-I", os.path.abspath(__file__), *sys.argv[1:]])

    from rec_mlx.chat import load_tokenizer, normalize_messages

    src = args.src or newest(MODELS, "flint-int4.json")
    tok = load_tokenizer(src)
    g = json.loads(args.golden.read_text())
    for case in g["cases"]:
        ids = tok.apply_chat_template(normalize_messages(case["messages"]), tools=case.get("tools"), tokenize=True,
                                      add_generation_prompt=True, return_dict=False)
        ids = list(map(int, ids))
        if len(ids) != case["prompt_tokens"]:
            print("%s: rendered %d tokens, rec serve reported %d" % (case["id"], len(ids), case["prompt_tokens"]), file=sys.stderr)
        case["prompt_ids"] = ids
        print("%-20s %6d tokens" % (case["id"], len(ids)), file=sys.stderr)
    args.golden.write_text(json.dumps(g, ensure_ascii=False) + "\n")
    print("updated %s" % args.golden, file=sys.stderr)


if __name__ == "__main__":
    main()
