from pathlib import Path
import argparse
import hashlib, json, os, subprocess, time, struct, math, re
lane = Path(__file__).resolve().parent
repo = lane.parents[1]
parser = argparse.ArgumentParser(description="Repeat the isolated dense prefill ablation on a local Q4 GGUF.")
parser.add_argument("--snapshot", default="head", help="Snapshot directory, relative to this archive or absolute")
parser.add_argument("--output", default="dense-ab", help="Fresh output directory, relative to this archive or absolute")
args = parser.parse_args()
head = (lane / args.snapshot).resolve()
out = (lane / args.output).resolve()
out.mkdir(exist_ok=False)
env = {k:v for k,v in os.environ.items() if not k.startswith("DS4_")}
env["DS4_DIAG_TRACE_PREFILL_DENSE"] = "1"
arms = {"head": {}, "base_dense": {"DS4_DIAG_BASE_PREFILL_DENSE":"1"}, "base_split": {"DS4_DIAG_BASE_PREFILL_SPLIT_ONLY":"1"}}
report = {"scope":"same committed HEAD snapshot with diagnostic dense policies; M1 Max Q4 SSD; not M4 Q2 replication", "runs":[], "comparisons":[]}
source_paths = [head / "ds4", head / "ds4.c", head / "ds4_metal.m", *sorted((head / "metal").glob("*.metal"))]
report["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
references = {}
schedule = [(name,arm) for name in ("rome29","book575") for arm in ("head","base_dense","base_split","head_repeat")] + [("book5942",arm) for arm in ("head","base_split")]
for name,arm in schedule:
    run_env = env | arms["head" if arm == "head_repeat" else arm]
    target = out / f"{name}-{arm}.json"
    cmd = [str(head / "ds4"), "-m", str(repo / "gguf/Qwen3.8-Flash-Next-Q4.gguf"), "--metal", "--ssd-streaming", "--ctx", "16384", "--prefill-chunk", "2048", "--nothink", "--temp", "0", "--prompt-file", str(lane / "fixtures" / (name + ".txt")), "--dump-logits", str(target)]
    print("START", name, arm, flush=True)
    started = time.monotonic()
    with (out / f"{name}-{arm}.log").open("wb") as log:
        rc = subprocess.run(cmd, cwd=head, env=run_env, stdout=log, stderr=subprocess.STDOUT).returncode
    assert rc == 0 and target.is_file(), (name,arm,rc)
    data = json.loads(target.read_text())
    assert data["vocab"] == len(data["logits"]) == 248320
    assert data["prompt_tokens"] == int(re.search(r"(\d+)$",name).group(1))
    assert all(x is not None and math.isfinite(x) for x in data["logits"])
    bits = b"".join(struct.pack("<f",x) for x in data["logits"])
    row = {"name":name,"arm":arm,"seconds":time.monotonic()-started,"command":cmd,"env":{k:v for k,v in run_env.items() if k.startswith("DS4_")},"prompt_tokens":data["prompt_tokens"],"argmax":data["argmax_token"],"argmax_logit":data["argmax_logit"],"float32_sha256":hashlib.sha256(bits).hexdigest()}
    report["runs"].append(row)
    if arm == "head": references[name] = data
    else:
        ref = references[name]; delta=[abs(a-b) for a,b in zip(ref["logits"],data["logits"])]
        comparison = {"name":name,"arm":arm,"max_abs":max(delta),"max_abs_token":delta.index(max(delta)),"mean_abs":sum(delta)/len(delta),"different_float32":sum(struct.pack("<f",a)!=struct.pack("<f",b) for a,b in zip(ref["logits"],data["logits"])),"same_argmax":ref["argmax_token"]==data["argmax_token"],"json_identical":(out / f"{name}-head.json").read_bytes()==target.read_bytes()}
        report["comparisons"].append(comparison)
        if arm == "head_repeat": assert comparison["json_identical"], "control not repeatable"
        print("COMPARE",json.dumps(comparison),flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==report["source_sha256"][str(p)] for p in source_paths)
    (out / "results.json").write_text(json.dumps(report,indent=2)+"\n")
    print("DONE",name,arm,round(row["seconds"],2),flush=True)
report["status"]="COMPLETE"
(out / "results.json").write_text(json.dumps(report,indent=2)+"\n")
