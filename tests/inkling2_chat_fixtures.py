#!/usr/bin/env python3
"""Build tests/test-vectors/inkling2-chat/*.json from a llama.cpp capture.

The capture directory (required) has ref/NNN_name.json files written by
capture.py against llama.cpp b12802 (e8d7128) serving Inkling-2-Mini-NVFP4.gguf: the OpenAI request, the prompt from
POST /apply-template, token ids from POST /tokenize (parse_special), and for tool cases the raw generation with
special tokens and the message llama.cpp parsed from exactly that text.

Each output file keeps what ds4's tests need:
  name, request, prompt, tokens, llama_error, ds4, [prompt_ds4], raw_generation, expect{content, reasoning_content, calls[]}
"ds4" says what ds4 must do with the request: "match" (default), "error" (refuse like llama.cpp), "skip" (a documented
difference, see docs/INKLING2.md).

usage: tests/inkling2_chat_fixtures.py CAPTURE_DIR [OUT_DIR]   (OUT_DIR default: test-vectors/inkling2-chat next to this script)
"""
import glob, json, os, sys

ERROR_LIKE_LLAMA = {"034", "035", "037", "061", "062"}   # unknown effort strings, bad tool-call arguments
SKIP = {
    "047": "llama.cpp rejects a tool without the {type:function,function:{..}} wrapper; ds4 declares it as given",
    "050": "llama.cpp refuses to continue a final assistant message with tool calls; ds4 renders the history as is",
}
# llama.cpp prints the integer 12345678901234567890 wrapped to int64 (a bug); the HF template prints it as written
WRAPPED = {"054": ("-6101065172474983726", "12345678901234567890")}


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-vectors", "inkling2-chat")
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "ref", "*.json"))):
        d = json.load(open(f))
        num = d["name"][:3]
        v = {"name": d["name"], "request": d["request"], "prompt": d["prompt"], "tokens": d["tokens"],
             "llama_error": d["error"] is not None}
        if num in ERROR_LIKE_LLAMA:
            v["ds4"] = "error"
        elif num in SKIP:
            v["ds4"] = "skip"
            v["ds4_note"] = SKIP[num]
        if num in WRAPPED and d["prompt"]:
            v["prompt_ds4"] = d["prompt"].replace(*WRAPPED[num])
        c = d.get("chat")
        if c and c.get("raw_generation") is not None:
            m = c["message"]
            v["raw_generation"] = c["raw_generation"]
            v["expect"] = {
                "content": m.get("content") or "",
                "reasoning_content": m.get("reasoning_content") or "",
                "calls": [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
                          for t in m.get("tool_calls") or []],
            }
        json.dump(v, open(os.path.join(out, d["name"] + ".json"), "w"), ensure_ascii=False, indent=1)
    print("wrote", len(glob.glob(os.path.join(out, "*.json"))), "vectors to", out)


if __name__ == "__main__":
    main()
