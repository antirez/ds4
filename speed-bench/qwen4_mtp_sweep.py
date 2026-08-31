#!/usr/bin/env python3
"""Qwen3.8-Flash-Next Q4_0-routed MTP sweep driver (session twenty).

Runs interleaved plain-vs-MTP decode A/Bs against ds4-server instances on
the standard Q4_0-routed pack, parsing the server's own decode-throughput
and Qwen MTP timing lines. All env vars are passed explicitly per server
(no shell variable indirection), and every config's engagement is verified
through observable log lines before any number is recorded.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "/Users/ifioravanti/github/ds4"
PACK = "/Users/ifioravanti/models/qwen3.8-flash-next-q40routed-v3dense-exp"
BASE = f"{PACK}/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf"
PLE = f"{PACK}/Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
MTP = f"{PACK}/qwen3.8-flash-next-q4-mtp.gguf"
CORPUS = f"{REPO}/speed-bench/promessi_sposi.txt"
OUT_DIR = "/tmp/ds4_mtp_round"

DEC_RE = re.compile(
    r"ds4-server: (\S+) ctx=(\d+)\.\.(\d+):(\d+) gen=(\d+).*? "
    r"decoding chunk=[0-9.]+ t/s avg=([0-9.]+) t/s ([0-9.]+)s")
PREFILL_RE = re.compile(
    r"ds4-server: completion ctx=(\d+)\.\.(\d+):(\d+) "
    r"prompt done ([0-9.]+)s")
MTP_TIMING_RE = re.compile(
    r"ds4: Qwen MTP timing drafted=(\d+) accepted=(\d+) "
    r"target_tokens=(\d+) cycle=([0-9.]+) ms(?: verifier=(\S+))?")
MTP_STAGES_RE = re.compile(
    r"ds4: Qwen MTP stages snapshot=([0-9.]+) ms draft=([0-9.]+) ms "
    r"verify=([0-9.]+) ms capture=(\S+) restore-select=([0-9.]+) ms "
    r"history=([0-9.]+) ms commit=([0-9.]+) ms path=(\S+) "
    r"counts\(full=(\d+) partial=(\d+) replay=(\d+)\)")
HIST_RE = re.compile(r"ds4: Qwen MTP accepted-length histogram((?: \d+=\d+)+)")
BYPASS_RE = re.compile(
    r"ds4: Qwen MTP scheduler switching to target decode "
    r"cycles=(\d+) target_tokens=(\d+) actual=([0-9.]+)ms baseline=([0-9.]+)ms")
GRAPH_RE = re.compile(r"ds4: Qwen graph allocated:.*")
SIDECAR_RE = re.compile(r"ds4: Qwen matching v\d+ .*MTP sidecar loaded:.*")
CAPTURE_RE = re.compile(r"ds4: Qwen MTP state capture: depth=(\d+) slots=(\d+).*")


def corpus_slice(char_start, char_len):
    with open(CORPUS, "rb") as f:
        f.seek(char_start)
        data = f.read(char_len)
    text = data.decode("utf-8", errors="replace")
    # trim a partial leading/trailing word for cleanliness
    sp = text.find(" ")
    if sp > 0:
        text = text[sp + 1:]
    return text


class Server:
    def __init__(self, name, port, mtp_depth=None, ctx=2048,
                 extra_env=None, timing=True):
        self.name = name
        self.port = port
        self.log_path = f"{OUT_DIR}/{name}.log"
        self.log = open(self.log_path, "wb")
        args = [f"{REPO}/ds4-server",
                "--model", BASE,
                "--ple", PLE,
                "--ctx", str(ctx),
                "--host", "127.0.0.1",
                "--port", str(port)]
        if mtp_depth is not None:
            args += ["--mtp-model", MTP,
                     "--mtp-draft", str(mtp_depth)]
            if timing:
                args += ["--mtp-timing"]
        env = dict(os.environ)
        env["DS4_LOCK_FILE"] = f"{OUT_DIR}/{name}.lock"
        for k, v in (extra_env or {}).items():
            env[k] = v
        self.env = env
        self.proc = subprocess.Popen(args, cwd=REPO, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.model_id = None
        self.offset = 0

    def wait_ready(self, timeout=420):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name}: server exited early, see {self.log_path}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as r:
                    data = json.load(r)
                self.model_id = data["data"][0]["id"]
                return True
            except (urllib.error.URLError, OSError, KeyError,
                    json.JSONDecodeError):
                time.sleep(1.0)
        raise RuntimeError(f"{self.name}: not ready after {timeout}s")

    def read_new(self):
        size = os.path.getsize(self.log_path)
        with open(self.log_path, "rb") as f:
            f.seek(self.offset)
            data = f.read(size - self.offset)
        self.offset = size
        return data.decode("utf-8", errors="replace")

    def request(self, prompt, max_tokens):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 1,
            "ignore_eos": True,
            "stream": False,
        }
        body = json.dumps(payload).encode()
        t0 = time.monotonic()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            resp = json.load(r)
        wall = time.monotonic() - t0
        text = self.read_new()
        rec = {"server": self.name, "wall_s": round(wall, 3)}
        pre = list(PREFILL_RE.finditer(text))
        if pre:
            rec["prompt_tokens"] = int(pre[-1].group(3))
            rec["prefill_s"] = float(pre[-1].group(4))
        dec = [m for m in DEC_RE.finditer(text)
               if int(m.group(5)) == max_tokens]
        if dec:
            rec["decode_tps"] = float(dec[-1].group(6))
            rec["decode_s"] = float(dec[-1].group(7))
        cyc = list(MTP_TIMING_RE.finditer(text))
        if cyc:
            cycles = [{
                "drafted": int(m.group(1)), "accepted": int(m.group(2)),
                "target_tokens": int(m.group(3)),
                "cycle_ms": float(m.group(4)), "verifier": m.group(5)}
                for m in cyc]
            tt = sum(c["target_tokens"] for c in cycles)
            total_cyc_ms = sum(c["cycle_ms"] for c in cycles)
            rec["mtp"] = {
                "cycles": len(cycles),
                "target_tokens_sum": tt,
                "block_cycles": sum(1 for c in cycles if c["drafted"] > 0),
                "bypass_cycles": sum(1 for c in cycles if c["drafted"] == 0),
                "mean_cycle_ms": round(total_cyc_ms / len(cycles), 3),
                "sum_cycle_ms": round(total_cyc_ms, 1),
                "implied_tps": round(tt / (total_cyc_ms / 1000.0), 2),
                "mean_accepted": round(
                    sum(c["accepted"] for c in cycles if c["drafted"] > 0) /
                    max(1, sum(1 for c in cycles if c["drafted"] > 0)), 3),
            }
        st = list(MTP_STAGES_RE.finditer(text))
        if st:
            n = len(st)
            def mean(i):
                return round(sum(float(m.group(i)) for m in st) / n, 3)
            last = st[-1]
            rec["stages"] = {
                "n": n,
                "snapshot_ms": mean(1), "draft_ms": mean(2),
                "verify_ms": mean(3), "restore_ms": mean(5),
                "history_ms": mean(6), "commit_ms": mean(7),
                "final_counts": {"full": int(last.group(9)),
                                 "partial": int(last.group(10)),
                                 "replay": int(last.group(11))},
            }
        hist = list(HIST_RE.finditer(text))
        if hist:
            counts = {}
            for tok in hist[-1].group(1).split():
                k, v = tok.split("=")
                counts[int(k)] = int(v)
            rec["accepted_hist"] = counts
        byp = list(BYPASS_RE.finditer(text))
        if byp:
            rec["scheduler_bypass"] = [{
                "cycles": int(m.group(1)), "target_tokens": int(m.group(2)),
                "actual_ms": float(m.group(3)),
                "baseline_ms": float(m.group(4))} for m in byp]
        return rec, resp

    def startup_evidence(self):
        with open(self.log_path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        ev = {"graph": bool(GRAPH_RE.search(text)),
              "sidecar": bool(SIDECAR_RE.search(text))}
        cap = CAPTURE_RE.search(text)
        if cap:
            ev["capture_depth"] = int(cap.group(1))
        g = GRAPH_RE.search(text)
        if g:
            line = g.group(0)
            ev["moe_down_q40"] = "Q4_0-expert-split" in line
        return ev

    def stop(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(records, label):
    print(f"\n=== {label} ===")
    by_cfg = {}
    for r in records:
        by_cfg.setdefault(r["server"], []).append(r)
    for cfg, rs in by_cfg.items():
        tps = [r["decode_tps"] for r in rs if "decode_tps" in r]
        line = f"{cfg:12s} n={len(rs)} decode_tps="
        line += " ".join(f"{t:.2f}" for t in tps)
        if tps:
            line += f" | mean {mean(tps):.2f}"
        mtps = [r for r in rs if "mtp" in r]
        if mtps:
            m = [r["mtp"] for r in mtps]
            line += (f" | cycles {sum(x['cycles'] for x in m)}"
                     f" bypass {sum(x['bypass_cycles'] for x in m)}"
                     f" tgt {sum(x['target_tokens_sum'] for x in m)}")
            eng = [x for x in m if x["block_cycles"] > 0]
            if eng:
                line += f" | cycle_ms {mean([x['mean_cycle_ms'] for x in eng]):.1f}"
                line += f" | acc {mean([x['mean_accepted'] for x in eng]):.2f}"
            st = [r["stages"] for r in mtps if "stages" in r]
            if st:
                line += (f" | verify {mean([s['verify_ms'] for s in st]):.1f}"
                         f" draft {mean([s['draft_ms'] for s in st]):.1f}"
                         f" hist {mean([s['history_ms'] for s in st]):.1f}"
                         f" commit {mean([s['commit_ms'] for s in st]):.1f}")
            nb = sum(len(r.get("scheduler_bypass", [])) for r in mtps)
            if nb:
                line += f" | SCHEDULER-BYPASS-EVENTS {nb}"
        print(line)
    sys.stdout.flush()
