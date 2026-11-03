from pathlib import Path
import json,statistics,hashlib
p=Path(__file__).resolve().parent
rows=[]
for name in ['major-layer9','compact-layer9','heavy-layer9','heavy-layer47']:
 x=json.loads((p/(name+'.json')).read_text());b=x['benchmarks'][0]
 a=b['baseline_gpu_ms'];c=b['candidate_gpu_ms'];cpu=x['worklist_cpu_ms'] if x['schedule'].startswith('compact') else 0
 rows.append({'name':name,'variant':x['schedule'],'layer':x['routing_layer'],
 'median_baseline_pair_ms':statistics.median(a),'median_candidate_pair_plus_cpu_ms':statistics.median(c),
 'median_gain_percent':100*(statistics.median(a)/statistics.median(c)-1),
 'mean_baseline_ms':statistics.mean(a),'mean_candidate_pair_plus_cpu_ms':statistics.mean(c),
 'mean_gain_percent':100*(statistics.mean(a)/statistics.mean(c)-1),
 'cpu_worklist_ms':cpu,'cpu_schedule_timing_samples':1,
 'baseline_gpu_samples_ms':a,'candidate_gpu_samples_ms':[v-cpu for v in c],
 'candidate_pair_plus_cpu_samples_ms':c,
 'block_mean_gains_percent':[100*(statistics.mean(a[i:i+2])/statistics.mean(c[i:i+2])-1) for i in range(0,len(a),2)],
 'descriptor_used_bytes':4*x['compact_work_count'],'descriptor_capacity_in_fixture_bytes':512*256*4,
 'descriptor_capacity_in_fixture_shared_by_both_arms':True,
 'comparisons_fp32':x['bit_exact_float_comparisons'],'all_samples_included':True,
 'conversion_median_ms':{'baseline':b['stages'][0]['baseline_median_ms'],'candidate':b['stages'][0]['candidate_median_ms']},
 'mid_median_ms':{'baseline':b['stages'][1]['baseline_median_ms'],'candidate':b['stages'][1]['candidate_median_ms']},
 'down_median_ms':{'baseline':b['stages'][2]['baseline_median_ms'],'candidate':b['stages'][2]['candidate_median_ms']}})
result={'status':'PASS','scope':'resident paired MoE microbenchmarks, synthetic weights and captured real GPU routing, no whole-model evidence',
 'baseline':'current production M1 SSD K32 NT4, ordinary tile-major0 grid, ascending active IDs',
 'reference_sha256':hashlib.sha256((p/'baseline.metal').read_bytes()).hexdigest(),
 'small_fast_fp32_comparisons':json.loads((p/'compact-fast.json').read_text())['bit_exact_float_comparisons'],
 'small_safe_fp32_comparisons':json.loads((p/'compact-safe.json').read_text())['bit_exact_float_comparisons'],
 'note_raw_field':'Original per-run candidate_gpu_ms historically named field includes measured CPU worklist only in whole-pair compact runs. Here candidate_gpu_samples_ms subtracts that exact recorded overhead; stages were GPU-only.',
 'results':rows}
(p/'results-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for x in rows:print(x['name'],round(x['median_gain_percent'],3),round(x['mean_gain_percent'],3),'CPU ms',x['cpu_worklist_ms'],'block gains',x['block_mean_gains_percent'])
