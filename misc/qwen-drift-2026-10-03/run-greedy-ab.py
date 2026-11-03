from pathlib import Path
import argparse
import hashlib,json,os,subprocess,time
lane=Path(__file__).resolve().parent
repo=lane.parents[1]
parser = argparse.ArgumentParser(description="Repeat 64-step greedy comparisons after the prefill ablation.")
parser.add_argument("--snapshot", default="head", help="Snapshot directory, relative to this archive or absolute")
parser.add_argument("--output", default="greedy-ab", help="Fresh output directory, relative to this archive or absolute")
args = parser.parse_args()
head = (lane / args.snapshot).resolve()
out = (lane / args.output).resolve()
out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith("DS4_")}
report={"scope":"controlled HEAD dense policy ablation, M1 Max Q4 SSD, MTP off", "runs":[],"comparisons":[]}
ref={}
for name in ("rome29","book575"):
    for arm in ("head","base_dense"):
        run_env=env|({"DS4_DIAG_BASE_PREFILL_DENSE":"1"} if arm=="base_dense" else {})
        target=out/f"{name}-{arm}.json"
        cmd=[str(head/"ds4"),"-m",str(repo/"gguf/Qwen3.8-Flash-Next-Q4.gguf"),"--metal","--ssd-streaming","--ctx","16384","--prefill-chunk","2048","--nothink","--temp","0","-n","64","--prompt-file",str(lane/"fixtures"/(name+".txt")),"--dump-logprobs",str(target)]
        print("START",name,arm,flush=True)
        begin=time.monotonic()
        with (out/f"{name}-{arm}.log").open("wb") as log:
            rc=subprocess.run(cmd,cwd=head,env=run_env,stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0
        data=json.loads(target.read_text())
        ids=[step["selected"]["id"] for step in data["steps"]]
        report["runs"].append({"name":name,"arm":arm,"seconds":time.monotonic()-begin,"command":cmd,"env":{k:v for k,v in run_env.items() if k.startswith("DS4_")},"steps":len(ids),"ids":ids,"json_sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
        if arm=="head":ref[name]=data
        else:
            original=ref[name];a=[x["selected"]["id"] for x in original["steps"]]
            divergence=next((i for i,(x,y) in enumerate(zip(a,ids)) if x!=y),None)
            if divergence is None and len(a)!=len(ids):divergence=min(len(a),len(ids))
            c={"name":name,"same_tokens":a==ids,"first_different_step_zero_based":divergence,"steps_head":len(a),"steps_base_dense":len(ids)}
            if divergence is not None:
                c["selected_head"]=original["steps"][divergence]["selected"]
                c["selected_base_dense"]=data["steps"][divergence]["selected"]
            report["comparisons"].append(c)
            print("COMPARE",json.dumps(c),flush=True)
        (out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
        print("DONE",name,arm,round(time.monotonic()-begin,2),flush=True)
report["status"]="COMPLETE"
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
