#!/usr/bin/env python3
"""Token-for-token check of the Inkling 2 chat prompts against llama.cpp.

1. `./ds4_test --server` renders every request of tests/test-vectors/inkling2-chat through ds4's server code and
   writes the prompt text to a directory (DS4_INKLING2_DUMP_DIR).
2. `./ds4 --cpu --first-token-test` with DS4_INKLING2_TOK_FILE tokenizes those texts with ds4's tokenizer
   (special tokens parsed, like llama.cpp's /tokenize parse_special) using the model's own vocabulary.
3. The ids are compared with the ids llama.cpp produced for its prompt (field "tokens" of each vector).

Case 054 holds a 20-digit integer that llama.cpp prints wrapped to int64 (a llama.cpp bug).  ds4 prints the digits
as written (checked as text by ds4_test); here llama.cpp's own text of that case is tokenized instead, so the
tokenizer is still compared with the reference ids.

usage: tests/inkling2_chat_tokens.py --model MODEL.gguf [--ds4 ./ds4] [--ds4-test ./ds4_test]
"""
import argparse, glob, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="an Inkling 2 GGUF (only its vocabulary is used)")
    ap.add_argument("--ds4", default="./ds4")
    ap.add_argument("--ds4-test", default="./ds4_test")
    ap.add_argument("--vectors", default=os.path.join(HERE, "test-vectors", "inkling2-chat"))
    ap.add_argument("--lock", default=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"))
    ap.add_argument("--cli", metavar="BIN", help="also run tests/test_inkling2_chat_cli (make tests/test_inkling2_chat_cli)")
    a = ap.parse_args()

    vecs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.vectors, "*.json")))]
    env = dict(os.environ, DS4_LOCK_FILE=a.lock)
    with tempfile.TemporaryDirectory() as tmp:
        env_t = dict(env, DS4_INKLING2_DUMP_DIR=tmp, DS4_INKLING2_CHATVEC=a.vectors)
        subprocess.run([a.ds4_test, "--server"], env=env_t, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cases = []
        for v in vecs:
            path = os.path.join(tmp, v["name"] + ".txt")
            if os.path.exists(path):
                text = open(path, encoding="utf-8", newline="").read()
                if v.get("prompt_ds4") is not None:
                    assert text == v["prompt_ds4"], v["name"]
                    text = v["prompt"]   # llama.cpp's text: the reference ids belong to it
                cases.append((v, text))
        tok_file = os.path.join(tmp, "tok.txt")
        with open(tok_file, "w", encoding="utf-8") as f:
            for _, text in cases:
                f.write("1" + json.dumps(text, ensure_ascii=True) + "\n")
        out = subprocess.run([a.ds4, "-m", a.model, "--cpu", "--first-token-test", "--raw-prompt", "-p", "x"],
                             env=dict(env, DS4_INKLING2_TOK_FILE=tok_file), check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
    got = [json.loads(l) for l in out.splitlines() if l.startswith("[")]
    if len(got) != len(cases):
        sys.exit("expected %d token lines, got %d" % (len(cases), len(got)))
    ok = bad = skipped = 0
    for (v, text), ids in zip(cases, got):
        ref = v.get("tokens")
        if ref is None:
            skipped += 1
            print("skip  %s (llama.cpp refused the request)" % v["name"])
            continue
        if ids == ref:
            ok += 1
            continue
        bad += 1
        i = next((k for k in range(min(len(ids), len(ref))) if ids[k] != ref[k]), min(len(ids), len(ref)))
        print("FAIL  %s: first difference at token %d (ds4 %s, llama.cpp %s), lengths %d vs %d" % (
            v["name"], i, ids[i] if i < len(ids) else None, ref[i] if i < len(ref) else None, len(ids), len(ref)))
    print("tokens identical: %d/%d (skipped %d, failed %d)" % (ok, ok + bad, skipped, bad))
    if a.cli:
        by_name = {v["name"]: v for v in vecs}
        ids = [json.loads(l) for l in subprocess.run([a.cli, a.model], env=env, check=True, stdout=subprocess.PIPE,
                                                     stderr=subprocess.DEVNULL, text=True).stdout.splitlines()]
        names = ["002_system_user", "020_effort_none", "005_multi_turn"]
        for n, got in zip(names, ids):
            same = got == by_name[n]["tokens"]
            print("CLI %-18s %s" % (n, "identical" if same else "DIFFERENT"))
            bad += 0 if same else 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
