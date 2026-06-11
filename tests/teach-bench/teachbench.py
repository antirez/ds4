#!/usr/bin/env python3
"""teach-bench: benchmark the teaching asides of ds4-agent.

ds4-agent in teach mode emits <teach> asides that the terminal renderer
shows as "📚 ..." paragraphs.  This tool drives the agent over a small corpus
of fast, self-contained coding tasks, records the asides, has an OpenAI
model judge them against the agent's own teaching contract, and lets a human
rate the same asides for comparison.

Built to sit inside an automated improvement loop (e.g. Claude Code editing
the teaching prompt, then re-benchmarking): every command is non-interactive
except rate, runs are immutable and timestamped, each run records the
teaching prompt it executed with (recovered from the agent's trace), and
history can emit JSON.

Subcommands:

  bench     run + eval + report in one shot (the loop entry point)
  list      show the corpus
  run       drive ds4-agent over the corpus, record asides into results/
  eval      score recorded asides with an OpenAI judge (needs OPENAI_API_KEY)
  rate      score them yourself, interactively
  report    aggregate judge + human scores for one run
  history   one line per run, with teaching-prompt version hashes (--json)
  selftest  validate the corpus without touching the agent

Standard library only.  See README.md for details.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_CORPUS = os.path.join(HERE, "prompts.json")
DEFAULT_RESULTS = os.path.join(HERE, "results")
DEFAULT_BIN = os.path.join(REPO, "ds4-agent")
DEFAULT_MODEL = os.path.join(REPO, "ds4flash.gguf")
DEFAULT_EVAL_MODEL = os.environ.get("TEACHBENCH_EVAL_MODEL", "gpt-5.2")
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TEACH_MARKER = "📚"
DIMENSIONS = ("insight", "calibration", "engagement", "grounding", "economy")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def die(msg):
    print("teachbench: " + msg, file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------------

def load_corpus(path):
    try:
        with open(path) as f:
            corpus = json.load(f)
    except (OSError, ValueError) as e:
        die("cannot load corpus %s: %s" % (path, e))
    seen = set()
    for p in corpus.get("prompts", []):
        for key in ("id", "title", "prompt"):
            if key not in p:
                die("corpus prompt missing %r: %s" % (key, json.dumps(p)[:120]))
        if p["id"] in seen:
            die("duplicate prompt id %r in corpus" % p["id"])
        seen.add(p["id"])
        persona = p.get("persona")
        if persona and persona not in corpus.get("personas", {}):
            die("prompt %r references unknown persona %r" % (p["id"], persona))
    if not corpus.get("prompts"):
        die("corpus has no prompts")
    return corpus


def select_prompts(corpus, ids_csv):
    prompts = corpus["prompts"]
    if not ids_csv:
        return prompts
    wanted = [s.strip() for s in ids_csv.split(",") if s.strip()]
    by_id = {p["id"]: p for p in prompts}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        die("unknown prompt ids: %s (try: %s list)" %
            (", ".join(missing), os.path.basename(sys.argv[0])))
    return [by_id[w] for w in wanted]


# ----------------------------------------------------------------------------
# Aside parsing
#
# Primary source: the agent's --trace log, which records every generated
# token verbatim, so the literal <teach>...</teach> spans can be extracted
# exactly (a multi-paragraph aside is unambiguous there).  Prompt-token dumps
# inside the trace are announced by a "tokens label=... start=S len=L"
# header followed by exactly L-S token lines; skip those so the system
# prompt's own <teach> example is never picked up.
#
# Fallback: rendered stdout, where each aside is a paragraph starting with
# "📚 ".  A blank line ends it there, so a multi-paragraph aside is truncated
# to its first paragraph - acceptable for a fallback.
# ----------------------------------------------------------------------------

TRACE_TOKEN_RE = re.compile(r' token index=-?\d+ id=-?\d+ bytes=\d+ '
                            r'text="(.*)" hex=[0-9a-fA-F]*\s*$')
TRACE_DUMP_RE = re.compile(r' tokens label=(\S*) start=(-?\d+) len=(-?\d+)')
TEACH_RE = re.compile(r'<teach>(.*?)</teach>', re.S)


def trace_unescape(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1]
        simple = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
        elif nxt == "x" and i + 3 < n:
            try:
                out.append(chr(int(s[i + 2:i + 4], 16)))
                i += 4
            except ValueError:
                out.append(c)
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_asides_from_trace(path):
    """Returns the asides, or None when the trace is missing or has no teach
    spans (caller falls back to stdout parsing)."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    pieces, skip = [], 0
    for line in lines:
        m = TRACE_DUMP_RE.search(line)
        if m:
            skip = max(0, int(m.group(3)) - int(m.group(2)))
            continue
        m = TRACE_TOKEN_RE.search(line)
        if not m:
            continue
        if skip:
            skip -= 1
            continue
        pieces.append(trace_unescape(m.group(1)))
    stream = "".join(pieces)
    if "<teach>" not in stream:
        return None
    return [t.strip() for t in TEACH_RE.findall(stream) if t.strip()]


def parse_trace_dump(path, label):
    """Concatenated text of the first prompt-token dump with this label."""
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    pieces, want = [], 0
    for line in lines:
        m = TRACE_DUMP_RE.search(line)
        if m:
            if pieces:
                break
            want = (max(0, int(m.group(3)) - int(m.group(2)))
                    if m.group(1) == label else 0)
            continue
        if want:
            m = TRACE_TOKEN_RE.search(line)
            if m:
                pieces.append(trace_unescape(m.group(1)))
                want -= 1
    return "".join(pieces) or None


def teach_prompt_from_trace(trace_path):
    """The '# Teaching' section of the system prompt the agent actually ran
    with, recovered from the trace.  This is what a prompt-tuning loop wants
    to diff between runs."""
    sysprompt = parse_trace_dump(trace_path, "initial_system_prompt")
    if not sysprompt:
        return None
    m = re.search(r"\n# Teaching\n.*?(?=\n# |\Z)", sysprompt, re.S)
    return m.group(0).strip() if m else None


def parse_asides(text):
    text = ANSI_RE.sub("", text)
    asides, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if current is not None:
            if line:
                current.append(line)
                continue
            asides.append(" ".join(current))
            current = None
        if line.startswith(TEACH_MARKER):
            body = line[len(TEACH_MARKER):].strip()
            current = [body] if body else []
    if current is not None:
        asides.append(" ".join(current))
    return [a for a in asides if a]


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def write_workspace_files(workdir, files):
    for f in files:
        path = os.path.join(workdir, f["path"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if "base64" in f:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(f["base64"]))
        else:
            with open(path, "w") as fh:
                fh.write(f["text"])


def metal_env(agent_bin):
    """ds4's Metal backend loads metal/*.metal relative to the cwd, but the
    benchmark runs the agent inside per-task workspaces.  Every source honors
    a DS4_METAL_<NAME>_SOURCE override whose name is derived from the
    filename, so point them all at the metal/ directory next to the binary."""
    env = {}
    mdir = os.path.join(os.path.dirname(agent_bin), "metal")
    if os.path.isdir(mdir):
        for f in os.listdir(mdir):
            if f.endswith(".metal"):
                name = os.path.splitext(f)[0].upper()
                env["DS4_METAL_%s_SOURCE" % name] = os.path.join(mdir, f)
    return env


def kill_tree(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    proc.wait()


def run_agent(cmd, cwd, env, stdout_path, stderr_path, timeout, progress=None):
    """Run ds4-agent in its own process group so a timeout (or ^C) can kill
    the whole tree (the agent may have bash children).  progress, when given,
    is called with the elapsed seconds about twice a second while the agent
    runs."""
    started = time.time()
    timed_out = False
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while True:
                elapsed = time.time() - started
                if elapsed >= timeout:
                    timed_out = True
                    kill_tree(proc)
                    break
                try:
                    proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if progress:
                        progress(elapsed)
        except KeyboardInterrupt:
            kill_tree(proc)
            raise
    return proc.returncode, time.time() - started, timed_out


def save_run(run_dir, run):
    tmp = os.path.join(run_dir, "run.json.tmp")
    with open(tmp, "w") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, os.path.join(run_dir, "run.json"))


class AgentLocked(Exception):
    """Another ds4 instance holds the global lock; nothing was recorded."""


def agent_problem(args):
    """Why the agent cannot be run, or None."""
    if not os.access(os.path.abspath(args.bin), os.X_OK):
        return "agent binary not executable: %s" % os.path.abspath(args.bin)
    if not os.path.exists(os.path.abspath(args.model)):
        return "model not found: %s" % os.path.abspath(args.model)
    return None


def create_run(args, run_id):
    """Make the run directory and skeleton.  Returns (run_dir, run), or
    (None, None) when a run with that id already exists."""
    run_dir = os.path.join(args.results, run_id)
    if os.path.exists(os.path.join(run_dir, "run.json")):
        return None, None
    os.makedirs(run_dir, exist_ok=True)
    run = {
        "run_id": run_id,
        "created": now_iso(),
        "agent_bin": os.path.abspath(args.bin),
        "agent_model": os.path.abspath(args.model),
        "corpus": os.path.abspath(args.corpus),
        "argv": sys.argv[1:],
        "results": [],
    }
    return run_dir, run


def run_single(run, run_dir, pdef, personas, trial, args, progress=None):
    """Run one prompt through the agent, append the result to run["results"],
    save run.json, and return the result.  Raises AgentLocked when another
    ds4 instance holds the global lock."""
    rid = pdef["id"] + (".t%d" % trial if args.trials > 1 else "")
    base = os.path.join(run_dir, rid)
    workdir = os.path.join(base, "work")
    home = os.path.join(base, "home")
    os.makedirs(workdir)
    os.makedirs(os.path.join(home, ".ds4"))
    write_workspace_files(workdir, pdef.get("files", []))

    persona = pdef.get("persona")
    persona_text = personas.get(persona, "") if persona else ""
    if persona_text:
        with open(os.path.join(home, ".ds4", "learner.md"), "w") as f:
            f.write(persona_text.strip() + "\n")

    agent_bin = os.path.abspath(args.bin)
    level = pdef.get("teach_level", "medium")
    cmd = [agent_bin, "--non-interactive", "--teach", level,
           "-m", os.path.abspath(args.model), "-p", pdef["prompt"]]
    if not args.think:
        cmd.append("--nothink")
    if args.seed is not None:
        cmd += ["--seed", str(args.seed + trial - 1)]

    env = dict(os.environ, HOME=home, **metal_env(agent_bin))
    stdout_path = os.path.join(base, "stdout.txt")
    stderr_path = os.path.join(base, "stderr.txt")
    trace_path = os.path.join(base, "trace.log")
    cmd += ["--trace", trace_path]

    exit_code, duration, timed_out = run_agent(
        cmd, workdir, env, stdout_path, stderr_path, args.timeout, progress)

    if exit_code and not timed_out:
        with open(stderr_path, errors="replace") as f:
            lock_msg = re.search(
                r"another ds4 process is already running[^\n]*", f.read())
        if lock_msg:
            raise AgentLocked(lock_msg.group(0))

    asides = parse_asides_from_trace(trace_path)
    aside_source = "trace"
    if asides is None:
        aside_source = "stdout"
        with open(stdout_path, errors="replace") as f:
            asides = parse_asides(f.read())

    # Record the teaching prompt this run actually executed with, once per
    # run, so prompt-tuning loops can tie scores to prompt versions.
    if "teach_prompt" not in run:
        teach_prompt = teach_prompt_from_trace(trace_path)
        if teach_prompt:
            run["teach_prompt"] = teach_prompt
            run["teach_prompt_sha"] = hashlib.sha256(
                teach_prompt.encode()).hexdigest()[:12]

    check = pdef.get("check")
    check_passed = None
    if check and not timed_out:
        try:
            rc = subprocess.run(["/bin/sh", "-c", check], cwd=workdir,
                                env=env, timeout=60,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode
            check_passed = rc == 0
        except subprocess.TimeoutExpired:
            check_passed = False

    result = {
        "prompt_id": pdef["id"],
        "trial": trial,
        "dir": rid,
        "persona": persona,
        "teach_level": level,
        "duration_s": round(duration, 1),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "check_passed": check_passed,
        "asides": asides,
        "aside_source": aside_source,
    }
    run["results"].append(result)
    save_run(run_dir, run)
    return result


def result_status(r):
    return "TIMEOUT" if r["timed_out"] else (
        "exit %d" % r["exit_code"] if r["exit_code"] else "ok")


def cmd_run(args):
    corpus = load_corpus(args.corpus)
    prompts = select_prompts(corpus, args.prompts)
    personas = corpus.get("personas", {})
    problem = agent_problem(args)
    if problem:
        die(problem)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, run = create_run(args, run_id)
    if run is None:
        die("run %s already exists" % run_id)

    total = len(prompts) * args.trials
    n = 0
    print("run %s: %d prompt(s) x %d trial(s) -> %s" %
          (run_id, len(prompts), args.trials, run_dir))
    for pdef in prompts:
        for trial in range(1, args.trials + 1):
            n += 1
            rid = pdef["id"] + (".t%d" % trial if args.trials > 1 else "")
            print("[%d/%d] %-18s " % (n, total, rid), end="", flush=True)
            try:
                r = run_single(run, run_dir, pdef, personas, trial, args)
            except AgentLocked as e:
                print("LOCKED")
                die("ds4-agent refused to start: %s\n"
                    "Wait for that instance to finish, then rerun "
                    "(results so far are saved in %s)." %
                    (e, os.path.join(run_dir, "run.json")))
            check_str = {True: "PASS", False: "FAIL", None: "-"}[r["check_passed"]]
            print("%5.0fs  %-7s  check %-4s  %d aside(s)" %
                  (r["duration_s"], result_status(r), check_str,
                   len(r["asides"])))

    emitted = sum(1 for r in run["results"] if r["asides"])
    print("done: %d/%d runs emitted at least one aside; results in %s" %
          (emitted, len(run["results"]), os.path.join(run_dir, "run.json")))
    if getattr(args, "eval", False):
        args.run = run_id
        args.force = False
        cmd_eval(args)
    return run_id


def cmd_bench(args):
    """run + eval + report in one shot."""
    args.eval = False  # chained explicitly below
    run_id = cmd_run(args)
    args.run = run_id
    args.force = False
    print()
    if os.environ.get("OPENAI_API_KEY"):
        cmd_eval(args)
    else:
        print("skipping eval: OPENAI_API_KEY is not set "
              "(judge it later with: %s eval --run %s)" %
              (os.path.basename(sys.argv[0]), run_id))
    print()
    cmd_report(args)
    print("\nrate the asides yourself: %s rate --run %s" %
          (os.path.basename(sys.argv[0]), run_id))


# ----------------------------------------------------------------------------
# Runs on disk
# ----------------------------------------------------------------------------

def resolve_run(args):
    if not os.path.isdir(args.results):
        die("no results directory at %s (run the 'run' subcommand first)" %
            args.results)
    run_id = getattr(args, "run", None)
    if not run_id or run_id == "latest":
        runs = [d for d in os.listdir(args.results)
                if os.path.exists(os.path.join(args.results, d, "run.json"))]
        if not runs:
            die("no runs found under %s" % args.results)
        run_id = max(runs, key=lambda d: os.path.getmtime(
            os.path.join(args.results, d, "run.json")))
    run_dir = os.path.join(args.results, run_id)
    path = os.path.join(run_dir, "run.json")
    if not os.path.exists(path):
        die("no such run: %s" % run_id)
    with open(path) as f:
        return run_dir, json.load(f)


def result_label(r):
    return "%s (persona: %s, level: %s, check: %s)" % (
        r["dir"], r["persona"] or "none", r["teach_level"],
        {True: "PASS", False: "FAIL", None: "n/a"}[r["check_passed"]])


# ----------------------------------------------------------------------------
# eval: OpenAI judge
# ----------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an exacting judge of teaching quality for teach-bench, a benchmark of
ds4-agent: a terminal coding agent that doubles as a programming mentor.
While the agent works on a real task it emits short teaching asides (shown to
you as numbered [1], [2], ...).  You score those asides.

The agent's own design contract for its teaching, which is the standard you
hold it to:
- Persona: Khan Academy patience plus slightly unhinged hacker glee.  Feral
  enthusiasm for the craft, dry wit, opinions earned from scars ("I have been
  burned by strtok before").  No corporate cheer, no "Great question!", no
  exclamation-point confetti.
- Nothing is magic: explain from first principles, anchored to something the
  developer already knows.
- Make them lean in: open loops, foreshadowing, prediction questions,
  failures treated as the interesting part.
- Calibrate to the learner profile: skip what this developer already knows,
  pitch one level above where they are, zero condescension.
- Teach decisions, not keystrokes: trade-offs, invariants, debugging
  strategy, idioms, how to verify.  One concept per aside, 1-4 sentences,
  tied to the code at hand.  Never generic lectures, never narrate trivia.
- Asides belong at the moment of relevance, not bundled at the end.

Score every aside 1-5 on each dimension.  3 is competent-average, 5 is rare.
- insight: 1 = narrates trivia or restates the obvious; 3 = correct, useful,
  unsurprising; 5 = a genuine "I'll remember this" point that a developer at
  this profile's level likely did not know.
- calibration: 1 = wrong level (lectures a senior on basics, buries a junior
  in jargon); 3 = plausible for the level; 5 = pitched exactly one level
  above the profile and anchored to what they know.
- engagement: 1 = beige, corporate, or condescending; 3 = competent but
  flat; 5 = makes you lean in: stakes, an open loop, a strong earned opinion,
  vivid without being cute.
- grounding: 1 = could be pasted into any session unchanged (generic
  lecture); 3 = loosely references the task; 5 = inseparable from this exact
  code, bug, or decision.
- economy: 1 = rambling, buried reveal, multiple concepts; 3 = fine; 5 = one
  concept, tight, plain words, lands the reveal.

Also weigh the transcript: placement (asides right after the step that
taught them), prediction questions posed in prose, whether the persona shows
up in the narration too.  Penalize factual errors hard wherever you can
verify them from the transcript.

Reply with STRICT JSON only, no markdown fences, in exactly this shape:
{
  "aside_scores": [
    {"aside": 1, "insight": N, "calibration": N, "engagement": N,
     "grounding": N, "economy": N, "comment": "one or two blunt sentences"}
  ],
  "overall": N,            // 0-100 holistic teaching quality for this run
  "summary": "two or three blunt sentences on the run as a whole",
  "flags": []              // any of: "generic-lecture", "condescending",
                           // "corporate-cheer", "wrong-level",
                           // "factual-error", "buried-reveal",
                           // "narrates-trivia", "overlong", "bundled-at-end"
}
"""


def clip(text, head, tail):
    if len(text) <= head + tail + 40:
        return text
    return text[:head] + "\n[... %d chars truncated ...]\n" % (
        len(text) - head - tail) + text[-tail:]


def build_judge_user_message(run, result, pdef, persona_text, transcript):
    parts = []
    parts.append("## Task given to the coding agent\n%s" % pdef["prompt"])
    files = pdef.get("files", [])
    if files:
        chunks = []
        for f in files:
            if "base64" in f:
                chunks.append("### %s\n(binary file, %d bytes)" %
                              (f["path"], len(base64.b64decode(f["base64"]))))
            else:
                chunks.append("### %s\n```\n%s```" %
                              (f["path"], clip(f["text"], 1500, 0)))
        parts.append("## Workspace files (before the agent ran)\n" +
                     "\n\n".join(chunks))
    parts.append("## Learner profile the agent sees for this developer\n%s" %
                 (persona_text.strip() if persona_text else
                  "(empty - first session, the agent knows nothing about "
                  "this developer yet)"))
    parts.append("## Teaching level\n%s" % result["teach_level"])
    if pdef.get("notes"):
        parts.append("## What a good mentor would plausibly teach here "
                     "(orientation for you, not requirements)\n%s" % pdef["notes"])
    outcome = {True: "the task check PASSED", False: "the task check FAILED",
               None: "no automated check"}[result["check_passed"]]
    parts.append("## Task outcome\n%s; agent ran %.0fs%s" %
                 (outcome, result["duration_s"],
                  ", TIMED OUT" if result["timed_out"] else ""))
    asides = "\n\n".join("[%d] %s" % (i + 1, a)
                         for i, a in enumerate(result["asides"]))
    parts.append("## Teaching asides to score\n%s" % asides)
    parts.append("## Session transcript (context; agent stdout, may be "
                 "truncated)\n```\n%s\n```" % clip(transcript, 4000, 3000))
    return "\n\n".join(parts)


def openai_chat(messages, model, api_key, base_url):
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }).encode()
    url = base_url.rstrip("/") + "/chat/completions"
    last_err = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        })
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.load(resp)
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:1000]
            last_err = "HTTP %d from %s: %s" % (e.code, url, body)
            if e.code not in (429, 500, 502, 503):
                break
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            last_err = "%s: %s" % (type(e).__name__, e)
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(last_err)


def parse_judge_json(content):
    try:
        return json.loads(content)
    except ValueError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def judge_composite(ev):
    """Mean of the five dimensions across asides, scaled to 0-100."""
    scores = ev.get("aside_scores") or []
    means = []
    for s in scores:
        dims = [s[d] for d in DIMENSIONS if isinstance(s.get(d), (int, float))]
        if dims:
            means.append(sum(dims) / len(dims))
    if not means:
        return None
    return round(sum(means) / len(means) * 20, 1)


def judge_result(run, run_dir, r, pdef, personas, model, api_key, base_url):
    """Judge one result, store r["eval"], and save run.json.  Returns None
    on success or an error string."""
    persona_text = personas.get(r["persona"], "") if r["persona"] else ""
    stdout_path = os.path.join(run_dir, r["dir"], "stdout.txt")
    try:
        with open(stdout_path, errors="replace") as f:
            transcript = ANSI_RE.sub("", f.read())
    except OSError:
        transcript = "(transcript missing)"
    user = build_judge_user_message(run, r, pdef, persona_text, transcript)
    try:
        content = openai_chat(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            model, api_key, base_url)
        ev = parse_judge_json(content)
    except (RuntimeError, ValueError) as e:
        return str(e)
    ev["judge_model"] = model
    ev["judged_at"] = now_iso()
    ev["composite"] = judge_composite(ev)
    r["eval"] = ev
    save_run(run_dir, run)
    return None


def cmd_eval(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        die("OPENAI_API_KEY is not set")
    run_dir, run = resolve_run(args)
    corpus = load_corpus(run.get("corpus") or args.corpus)
    by_id = {p["id"]: p for p in corpus["prompts"]}
    personas = corpus.get("personas", {})

    todo = [r for r in run["results"]
            if r["asides"] and (args.force or "eval" not in r)]
    skipped = sum(1 for r in run["results"] if not r["asides"])
    if skipped:
        print("note: %d run(s) emitted no asides and cannot be judged" % skipped)
    if not todo:
        print("nothing to evaluate (use --force to re-judge)")
        return
    print("judging %d result(s) from run %s with %s" %
          (len(todo), run["run_id"], args.eval_model))

    for r in todo:
        pdef = by_id.get(r["prompt_id"])
        if not pdef:
            print("  %-18s skipped: prompt no longer in corpus" % r["dir"])
            continue
        err = judge_result(run, run_dir, r, pdef, personas,
                           args.eval_model, api_key, args.base_url)
        if err:
            print("  %-18s judge error: %s" % (r["dir"], err))
            continue
        ev = r["eval"]
        print("  %-18s composite %-5s overall %-3s %s" %
              (r["dir"], ev["composite"], ev.get("overall", "?"),
               ("flags: " + ",".join(ev["flags"])) if ev.get("flags") else ""))
    print("saved to %s" % os.path.join(run_dir, "run.json"))


# ----------------------------------------------------------------------------
# rate: human feedback
# ----------------------------------------------------------------------------

def ask(promptline):
    try:
        return input(promptline)
    except EOFError:
        return None


def rate_run(run_dir, run, corpus, redo=False):
    """Interactive rating loop over a run's results.  Used by both the rate
    subcommand and the TUI."""
    ratable = [r for r in run["results"] if r["asides"]]
    if not ratable:
        print("run %s has no asides to rate" % run["run_id"])
        return
    todo = [r for r in ratable if redo or "human" not in r]
    if not todo:
        print("all %d result(s) already rated (use --redo to re-rate)" %
              len(ratable))
        return
    print("rating %d of %d result(s) from run %s" %
          (len(todo), len(ratable), run["run_id"]))
    print("score each aside 1-5 (5 = I'd want this mentor every day), "
          "s = skip aside, q = save and quit\n")
    prompts_by_id = {p["id"]: p for p in corpus["prompts"]}

    quit_now = False
    for i, r in enumerate(todo):
        if quit_now:
            break
        print("=" * 72)
        print("[%d/%d] %s" % (i + 1, len(todo), result_label(r)))
        pdef = prompts_by_id.get(r["prompt_id"])
        if pdef:
            print("task: %s" % pdef["prompt"])
        ratings = []
        for j, aside in enumerate(r["asides"]):
            print("\n  aside %d/%d:\n  📚 %s\n" % (j + 1, len(r["asides"]), aside))
            score = None
            while True:
                ans = ask("  score 1-5 / s / q: ")
                if ans is None or ans.strip().lower() == "q":
                    quit_now = True
                    break
                ans = ans.strip().lower()
                if ans == "s":
                    break
                if ans in ("1", "2", "3", "4", "5"):
                    score = int(ans)
                    break
                print("  (enter 1-5, s, or q)")
            if quit_now:
                break
            if score is None:
                continue
            comment = ask("  comment (enter to skip): ")
            ratings.append({"aside": j + 1, "score": score,
                            "comment": (comment or "").strip()})
            ev = r.get("eval", {})
            for s in ev.get("aside_scores", []):
                if s.get("aside") == j + 1:
                    print("  judge said: " + ", ".join(
                        "%s %s" % (d, s.get(d, "?")) for d in DIMENSIONS) +
                        (" - %s" % s["comment"] if s.get("comment") else ""))
        if ratings:
            r["human"] = {
                "ratings": ratings,
                "mean": round(sum(x["score"] for x in ratings) / len(ratings), 2),
                "rated_at": now_iso(),
            }
            save_run(run_dir, run)
    print("\nsaved to %s" % os.path.join(run_dir, "run.json"))


def cmd_rate(args):
    run_dir, run = resolve_run(args)
    corpus = load_corpus(run.get("corpus") or args.corpus)
    rate_run(run_dir, run, corpus, redo=args.redo)


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


def print_table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    def fmt(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip()
    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for r in rows:
        print(fmt(r))


def cmd_report(args):
    run_dir, run = resolve_run(args)
    results = run["results"]
    if not results:
        die("run %s has no results" % run["run_id"])

    rows = []
    for r in results:
        ev = r.get("eval", {})
        human = r.get("human", {})
        rows.append([
            r["dir"], r["persona"] or "-", r["teach_level"],
            "%.0fs" % r["duration_s"],
            "TIMEOUT" if r["timed_out"] else
            {True: "PASS", False: "FAIL", None: "-"}[r["check_passed"]],
            len(r["asides"]),
            ev.get("composite", "-") if ev else "-",
            ev.get("overall", "-") if ev else "-",
            human.get("mean", "-") if human else "-",
        ])
    print("run %s  (created %s%s)" %
          (run["run_id"], run["created"],
           ", teach prompt %s" % run["teach_prompt_sha"]
           if run.get("teach_prompt_sha") else ""))
    if getattr(args, "teach_prompt", False):
        print()
        print(run.get("teach_prompt") or
              "(no teaching prompt captured for this run)")
    print()
    print_table(rows, ["prompt", "persona", "level", "time", "check",
                       "asides", "judge", "overall", "human"])

    n = len(results)
    emitted = [r for r in results if r["asides"]]
    checked = [r for r in results if r["check_passed"] is not None]
    passed = [r for r in checked if r["check_passed"]]
    print()
    print("reliability: %d/%d runs emitted >=1 aside; %d asides total" %
          (len(emitted), n, sum(len(r["asides"]) for r in results)))
    if checked:
        print("task success: %d/%d checks passed" % (len(passed), len(checked)))
    print("mean duration: %.0fs" %
          (sum(r["duration_s"] for r in results) / n))

    judged = [r for r in results if r.get("eval", {}).get("aside_scores")]
    if judged:
        dim_sums = {d: [] for d in DIMENSIONS}
        for r in judged:
            for s in r["eval"]["aside_scores"]:
                for d in DIMENSIONS:
                    if isinstance(s.get(d), (int, float)):
                        dim_sums[d].append(s[d])
        comps = [r["eval"]["composite"] for r in judged
                 if r["eval"].get("composite") is not None]
        if comps:
            print("judge composite (0-100): mean %.1f  min %.1f  max %.1f" %
                  (sum(comps) / len(comps), min(comps), max(comps)))
        print("judge dimension means (1-5): " + "  ".join(
            "%s %.2f" % (d, sum(v) / len(v))
            for d, v in dim_sums.items() if v))
        flags = {}
        for r in judged:
            for fl in r["eval"].get("flags", []):
                flags[fl] = flags.get(fl, 0) + 1
        if flags:
            print("judge flags: " + "  ".join(
                "%s x%d" % (k, v) for k, v in sorted(flags.items())))

    pairs = []
    for r in results:
        ev_scores = {s.get("aside"): s for s in
                     r.get("eval", {}).get("aside_scores", [])}
        for h in r.get("human", {}).get("ratings", []):
            s = ev_scores.get(h["aside"])
            if not s:
                continue
            dims = [s[d] for d in DIMENSIONS
                    if isinstance(s.get(d), (int, float))]
            if dims:
                pairs.append((sum(dims) / len(dims), h["score"]))
    if pairs:
        hx = [p[1] for p in pairs]
        print("human mean (1-5): %.2f over %d rated aside(s)" %
              (sum(hx) / len(hx), len(hx)))
        r_val = pearson([p[0] for p in pairs], hx)
        if r_val is not None:
            print("judge vs human agreement: pearson r = %.2f (n=%d)" %
                  (r_val, len(pairs)))


# ----------------------------------------------------------------------------
# history: one line per run, across all runs
# ----------------------------------------------------------------------------

def run_aggregates(run):
    rs = run["results"]
    agg = {
        "tests": len(rs),
        "asides": sum(len(r["asides"]) for r in rs),
        "emitted": sum(1 for r in rs if r["asides"]),
        "checked": sum(1 for r in rs if r["check_passed"] is not None),
        "passed": sum(1 for r in rs if r["check_passed"]),
        "judge": None,
        "human": None,
    }
    comps = [r["eval"]["composite"] for r in rs
             if r.get("eval", {}).get("composite") is not None]
    if comps:
        agg["judge"] = sum(comps) / len(comps)
    hums = [r["human"]["mean"] for r in rs
            if r.get("human", {}).get("mean") is not None]
    if hums:
        agg["human"] = sum(hums) / len(hums)
    return agg


def list_runs(results_dir):
    """Load every run on disk, oldest first."""
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for d in os.listdir(results_dir):
        path = os.path.join(results_dir, d, "run.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                runs.append(json.load(f))
        except (OSError, ValueError):
            continue
    runs.sort(key=lambda r: r.get("created", ""))
    return runs


HISTORY_HEADER = ["run", "created (utc)", "teach", "tests", "pass", "emit",
                  "asides", "judge", "human"]


def history_rows(runs):
    rows = []
    for run in runs:
        a = run_aggregates(run)
        rows.append([
            run["run_id"],
            run.get("created", "?").replace("T", " ").rstrip("Z"),
            run.get("teach_prompt_sha", "-"),
            a["tests"],
            "%d/%d" % (a["passed"], a["checked"]) if a["checked"] else "-",
            "%d/%d" % (a["emitted"], a["tests"]),
            a["asides"],
            "%.1f" % a["judge"] if a["judge"] is not None else "-",
            "%.2f" % a["human"] if a["human"] is not None else "-",
        ])
    return rows


def cmd_history(args):
    runs = list_runs(args.results)
    if not runs:
        die("no runs found under %s" % args.results)
    if args.json:
        out = []
        for run in runs:
            a = run_aggregates(run)
            a["run_id"] = run["run_id"]
            a["created"] = run.get("created")
            a["teach_prompt_sha"] = run.get("teach_prompt_sha")
            out.append(a)
        print(json.dumps(out, indent=2))
        return
    print_table(history_rows(runs), HISTORY_HEADER)
    print("\nteach = teaching-prompt version hash; pass = task checks passed; "
          "emit = runs with >=1 aside;\njudge = composite 0-100; "
          "human = your mean 1-5")


# ----------------------------------------------------------------------------
# selftest: validate the corpus without running the agent
#
# Every prompt's check must FAIL on the seeded (broken) files and PASS after
# the canonical fixes from its "fixes" field are applied.  This keeps the
# benchmark honest: a check that passes before the agent does anything, or
# that cannot pass at all, would silently corrupt the task-success metric.
# ----------------------------------------------------------------------------

def run_check(check, workdir):
    try:
        return subprocess.run(["/bin/sh", "-c", check], cwd=workdir,
                              timeout=120, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode
    except subprocess.TimeoutExpired:
        return -1


def cmd_selftest(args):
    corpus = load_corpus(args.corpus)
    prompts = select_prompts(corpus, args.prompts)
    failures = 0
    for p in prompts:
        if not p.get("check"):
            print("%-18s SKIP (no check)" % p["id"])
            continue
        msgs = []

        d = tempfile.mkdtemp(prefix="teachbench-self-")
        write_workspace_files(d, p.get("files", []))
        if run_check(p["check"], d) == 0:
            msgs.append("check PASSES on the seeded broken files")
        shutil.rmtree(d)

        if p.get("fixes"):
            d = tempfile.mkdtemp(prefix="teachbench-self-")
            write_workspace_files(d, p.get("files", []))
            for fx in p["fixes"]:
                path = os.path.join(d, fx["path"])
                with open(path) as f:
                    content = f.read()
                if fx["old"] not in content:
                    msgs.append("fix pattern not found in %s" % fx["path"])
                    continue
                with open(path, "w") as f:
                    f.write(content.replace(fx["old"], fx["new"]))
            if run_check(p["check"], d) != 0:
                msgs.append("check FAILS after canonical fixes")
            shutil.rmtree(d)
        else:
            msgs.append("no fixes field (fixed-state direction untested)")

        if msgs:
            failures += 1
            print("%-18s FAIL: %s" % (p["id"], "; ".join(msgs)))
        else:
            print("%-18s ok" % p["id"])
    if failures:
        die("%d corpus problem(s)" % failures)
    print("corpus ok: %d prompt(s) validated both ways" % len(prompts))


# ----------------------------------------------------------------------------
# list
# ----------------------------------------------------------------------------

def cmd_list(args):
    corpus = load_corpus(args.corpus)
    rows = [[p["id"], p.get("persona") or "-", p.get("teach_level", "medium"),
             "yes" if p.get("check") else "-", p["title"]]
            for p in corpus["prompts"]]
    print_table(rows, ["id", "persona", "level", "check", "title"])
    print("\npersonas: " + ", ".join(corpus.get("personas", {})))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        prog="teachbench",
        description="Benchmark ds4-agent's teaching asides.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="prompt corpus (default: %(default)s)")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="results directory (default: %(default)s)")
    sub = ap.add_subparsers(dest="command", required=False)

    sub.add_parser("list", help="show the corpus")

    def add_run_opts(p):
        p.add_argument("--prompts",
                       help="comma-separated prompt ids (default: all)")
        p.add_argument("--trials", type=int, default=1,
                       help="trials per prompt (default: 1)")
        p.add_argument("--bin", default=DEFAULT_BIN,
                       help="ds4-agent binary (default: %(default)s)")
        p.add_argument("--model", default=DEFAULT_MODEL,
                       help="gguf model path (default: %(default)s)")
        p.add_argument("--timeout", type=int, default=420,
                       help="seconds per agent run (default: 420)")
        p.add_argument("--seed", type=int,
                       help="sampling seed (trial index is added); "
                            "default random")
        p.add_argument("--think", action="store_true",
                       help="leave thinking on (default passes --nothink)")
        p.add_argument("--run-id", help="name this run (default: timestamp)")
        p.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL,
                       help="judge model (default: %(default)s, or "
                            "$TEACHBENCH_EVAL_MODEL)")
        p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                       help="OpenAI-compatible API base "
                            "(default: %(default)s)")

    bp = sub.add_parser("bench", help="run + eval + report in one shot")
    add_run_opts(bp)

    rp = sub.add_parser("run", help="drive ds4-agent over the corpus")
    add_run_opts(rp)
    rp.add_argument("--eval", action="store_true",
                    help="judge the run with OpenAI immediately after")

    ep = sub.add_parser("eval", help="judge a run's asides with OpenAI")
    ep.add_argument("--run", default="latest", help="run id (default: latest)")
    ep.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL,
                    help="judge model (default: %(default)s, or "
                         "$TEACHBENCH_EVAL_MODEL)")
    ep.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="OpenAI-compatible API base (default: %(default)s)")
    ep.add_argument("--force", action="store_true", help="re-judge everything")

    hp = sub.add_parser("rate", help="rate a run's asides yourself")
    hp.add_argument("--run", default="latest", help="run id (default: latest)")
    hp.add_argument("--redo", action="store_true",
                    help="re-rate results that already have human ratings")

    pp = sub.add_parser("report", help="summarize a run")
    pp.add_argument("--run", default="latest", help="run id (default: latest)")
    pp.add_argument("--teach-prompt", action="store_true",
                    help="also print the teaching prompt this run executed with")

    yp = sub.add_parser("history", help="one-line summary of every run")
    yp.add_argument("--json", action="store_true",
                    help="machine-readable output (for automation loops)")

    sp = sub.add_parser("selftest",
                        help="validate corpus checks without the agent")
    sp.add_argument("--prompts", help="comma-separated prompt ids (default: all)")

    args = ap.parse_args()
    if args.command is None:
        ap.print_help()
        return
    try:
        {"list": cmd_list, "bench": cmd_bench, "run": cmd_run,
         "eval": cmd_eval, "rate": cmd_rate, "report": cmd_report,
         "history": cmd_history, "selftest": cmd_selftest}[args.command](args)
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
