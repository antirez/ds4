#!/usr/bin/env python3
"""Rendered prompts of the Anthropic and Responses requests (tests/test-vectors/inkling2-api): ds4 against llama.cpp.

llama.cpp serves /v1/messages and /v1/responses too.  It converts both to an OpenAI chat body, which a verbose
server (-v) writes to its log ("converted request: {...}").  That body goes to llama.cpp's /apply-template (the
prompt text it renders) and /tokenize with special tokens parsed (the ids); the number of ids must also equal the
`task.n_tokens` the same server logged for the real request.

  capture (needs a llama-server started with -v, log file given):
      tests/inkling2_api_tokens.py capture --llama http://127.0.0.1:8080 --log LLAMA.log --out refdir
  compare (ds4's own rendering of the same requests, tokenized with ds4's tokenizer on the model's vocabulary):
      tests/inkling2_api_tokens.py compare --refs refdir --model MODEL.gguf [--ds4 ./ds4] [--ds4-test ./ds4_test]
"""
import argparse, glob, json, os, re, subprocess, sys, tempfile, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(HERE, "test-vectors", "inkling2-api")


def post(base, path, body):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


def capture(a):
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.vectors, "*.json"))):
        doc = json.load(open(f))
        for api, path, patch in (("anthropic", "/v1/messages", {"max_tokens": 1}), ("responses", "/v1/responses", {"max_output_tokens": 1})):
            size = os.path.getsize(a.log)
            try:
                post(a.llama, path, dict(doc[api], **patch))
            except urllib.error.HTTPError as e:
                print("llama.cpp refuses %s %s: HTTP %d %s" % (doc["name"], api, e.code, e.read().decode()[:160]))
                continue
            time.sleep(0.3)
            with open(a.log, "rb") as lf:
                lf.seek(size)
                log = lf.read().decode("utf-8", "replace")
            m = re.search(r"converted request: (\{.*\})\s*$", log, re.M)
            n = re.search(r"task\.n_tokens = (\d+)", log)
            if not m or not n:
                print("FAIL %s %s: no converted request / n_tokens in the log" % (doc["name"], api))
                continue
            conv = json.loads(m.group(1))
            prompt = post(a.llama, "/apply-template", conv)["prompt"]
            ids = post(a.llama, "/tokenize", {"content": prompt, "parse_special": True, "add_special": False})["tokens"]
            out = dict(name=doc["name"], api=api, prompt=prompt, tokens=ids, n_tokens=int(n.group(1)), converted=conv)
            json.dump(out, open(os.path.join(a.out, "%s.%s.json" % (doc["name"], api)), "w"))
            print("%-40s %-9s %d ids, llama.cpp counted %s" % (doc["name"], api, len(ids), n.group(1)))


def compare(a):
    refs = {}
    for f in sorted(glob.glob(os.path.join(a.refs, "*.json"))):
        r = json.load(open(f))
        refs[(r["name"], r["api"])] = r
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, DS4_LOCK_FILE=a.lock, DS4_INKLING2_DUMP_DIR=tmp, DS4_INKLING2_APIVEC=a.vectors)
        subprocess.run([a.ds4_test, "--server"], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        keys, texts = [], []
        for (name, api), r in sorted(refs.items()):
            p = os.path.join(tmp, "api_%s.%s.txt" % (name, api))
            if not os.path.exists(p):
                print("MISSING ds4 prompt for %s %s" % (name, api))
                continue
            keys.append((name, api))
            texts.append(open(p, encoding="utf-8", newline="").read())
        tok_file = os.path.join(tmp, "tok.txt")
        with open(tok_file, "w", encoding="utf-8") as f:
            for t in texts:
                f.write("1" + json.dumps(t, ensure_ascii=True) + "\n")
        out = subprocess.run([a.ds4, "-m", a.model, "--cpu", "--first-token-test", "--raw-prompt", "-p", "x"],
                             env=dict(os.environ, DS4_LOCK_FILE=a.lock, DS4_INKLING2_TOK_FILE=tok_file), check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout
    got = [json.loads(l) for l in out.splitlines() if l.startswith("[")]
    if len(got) != len(keys):
        sys.exit("expected %d token lines, got %d" % (len(keys), len(got)))
    text_ok = ids_ok = count_ok = bad = 0
    for k, text, ids in zip(keys, texts, got):
        r = refs[k]
        t_same = text == r["prompt"]
        i_same = ids == r["tokens"]
        c_same = len(r["tokens"]) == r["n_tokens"]
        text_ok += t_same
        ids_ok += i_same
        count_ok += c_same
        if not (t_same and i_same and c_same):
            bad += 1
            at = next((i for i in range(min(len(text), len(r["prompt"]))) if text[i] != r["prompt"][i]), None)
            print("DIFF %s %s: text %s (first difference %s), ids %s, llama.cpp /tokenize count %s its own count %d" % (
                k[0], k[1], "same" if t_same else "different", at, "same" if i_same else "different", c_same, r["n_tokens"]))
    print("compared %d requests (%d anthropic + %d responses): rendered text identical %d, token ids identical %d, "
          "llama.cpp ids == its own prompt token count %d" % (
              len(keys), sum(1 for k in keys if k[1] == "anthropic"), sum(1 for k in keys if k[1] == "responses"),
              text_ok, ids_ok, count_ok))
    sys.exit(1 if bad else 0)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("capture")
c.add_argument("--llama", default="http://127.0.0.1:8080")
c.add_argument("--log", required=True)
c.add_argument("--out", required=True)
c.add_argument("--vectors", default=VEC)
d = sub.add_parser("compare")
d.add_argument("--refs", required=True)
d.add_argument("--model", required=True)
d.add_argument("--ds4", default="./ds4")
d.add_argument("--ds4-test", default="./ds4_test")
d.add_argument("--vectors", default=VEC)
d.add_argument("--lock", default=os.environ.get("DS4_LOCK_FILE", "/tmp/ds4-inkling.lock"))
a = ap.parse_args()
{"capture": capture, "compare": compare}[a.cmd](a)
