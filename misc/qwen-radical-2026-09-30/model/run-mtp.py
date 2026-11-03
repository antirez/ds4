from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--name',default='mtp-abba');p.add_argument('--order',default='0110');p.add_argument('--prompts',default='rome,code');p.add_argument('--tokens',type=int,default=128);p.add_argument('--user-cli',action='store_true');a=p.parse_args()
r=Path.cwd();d=r/'misc/qwen-radical-2026-09-30/model';out=d/a.name;out.mkdir(exist_ok=False)
binary=r/'ds4'
env={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
env.update(DS4_CLI_FORCE_SESSION='1',DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY='1',DS4_DIAG_QWEN4_GAIN5='0',DS4_DIAG_QWEN4_PREFILL_BUNDLE='0')
if a.user_cli:env.pop('DS4_CLI_FORCE_SESSION')
files=[r/'ds4.c',r/'ds4_metal.m',r/'metal/qwen4.metal',r/'metal/dense.metal',binary]
report={'kind':'embedded MTP versus plain, '+('ordinary user CLI paths' if a.user_cli else 'same session CLI path')+', same cache capacity','order':a.order,'env':{k:v for k,v in env.items() if k.startswith('DS4_')},'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'runs':[]}
refs={}
for case in a.prompts.split(','):
 prompt=d/'prompt-rome.txt' if case=='rome' else d/'prompt-code.txt'
 for idx,variant in enumerate(map(int,a.order),1):
  names=subprocess.check_output(['/bin/ps','-axo','comm='],text=True).splitlines()
  conflict=[x for x in names if Path(x.strip()).name in {'ds4','ds4-trial','ds4-server','ds4-bench','clang','make','qwen-radical-schedule-oracle'}]
  if conflict:raise SystemExit('Concurrent workload '+repr(conflict))
  tag=f'{case}-{idx}-'+('mtp' if variant else 'plain')
  cmd=[str(binary),'-m',str(r/'gguf/Qwen3.8-Flash-Next-Q4.gguf'),'--metal','--ssd-streaming','--ctx','32768','--prefill-chunk','8192','--nothink','--temp','0','-n',str(a.tokens),'--prompt-file',str(prompt)]
  if variant:cmd+=['--mtp-timing']
  start=time.monotonic()
  with (out/(tag+'.out')).open('wb') as stdout,(out/(tag+'.log')).open('wb') as stderr:
   rc=subprocess.run(['/usr/bin/time','-l',*cmd],env=env,stdout=stdout,stderr=stderr).returncode
  log=(out/(tag+'.log')).read_text();raw=(out/(tag+'.out')).read_bytes();refs.setdefault(case,raw)
  rate=re.search(r'prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s',log)
  cache=re.search(r'cache budget=(\d+) experts.*?hits=(\d+) misses=(\d+).*?miss_pread=([0-9.]+) GiB',log)
  mtp=re.search(r'Qwen3.8 mtp: (\d+) verify cycles, (\d+) first drafts accepted \(([0-9.]+)%\)',log)
  peak=re.search(r'(\d+)\s+peak memory footprint',log)
  row={'tag':tag,'case':case,'variant':variant,'command':cmd,'exit_code':rc,'wall_s':time.monotonic()-start,'stdout_sha256':hashlib.sha256(raw).hexdigest(),'same_output':raw==refs[case],'prefill_tps':float(rate[1]) if rate else None,'decode_tps':float(rate[2]) if rate else None,'cache':dict(capacity=int(cache[1]),hits=int(cache[2]),misses=int(cache[3]),pread_gib_rounded=float(cache[4])) if cache else None,'mtp':dict(cycles=int(mtp[1]),first_accepted=int(mtp[2]),acceptance_pct=float(mtp[3])) if mtp else None,'peak_footprint_bytes':int(peak[1]) if peak else None}
  report['runs'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(row),flush=True)
  if rc or not rate or not cache or not row['same_output']:raise SystemExit('Invalid/mismatched run; stop timing comparison')
  if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h for f,h in report['source_sha256'].items()):raise SystemExit('Source changed')
report['summary']={}
for case in a.prompts.split(','):
 stats={}
 for phase in ['prefill','decode']:
  xs={v:[x[phase+'_tps'] for x in report['runs'] if x['case']==case and x['variant']==v] for v in [0,1]}
  if not all(xs.values()):continue
  speed={v:len(z)/sum(1/y for y in z) for v,z in xs.items()}
  stats[phase]={'harmonic_tps':speed,'gain_pct':100*(speed[1]/speed[0]-1),'samples':xs}
 report['summary'][case]=stats
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']),flush=True)
