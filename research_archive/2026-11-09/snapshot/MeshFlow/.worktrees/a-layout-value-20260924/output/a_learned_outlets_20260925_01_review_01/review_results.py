from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path.cwd().resolve();run=ROOT/'output/a_learned_outlets_20260925_01';out=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments.a_radial.io_utils import array_hash

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):assert np.allclose(a,b,atol=1e-12,rtol=1e-12),(a,b)
p=read(run/'RUN_PROTOCOL.json');reg=read(run/'MODEL_REGISTRY.json');result=read(run/'RESULTS.json');summary=read(run/'SUMMARY.json');rows=result['rows']
assert result['protocol_sha256']==reg['protocol_sha256']==sha(run/'RUN_PROTOCOL.json')
assert result['model_registry_sha256']==sha(run/'MODEL_REGISTRY.json')
assert p['raw_only'] and not p['Fixed_fallback'] and p['primary_CPU_threads']==1
expected={(s,f'{i:05d}',a) for s in [10,11] for i in range(64) for a in ['longfit','balanced']}
assert len(rows)==256 and {(r['seed'],r['sample_id'],r['arm']) for r in rows}==expected
base={(r['seed'],r['sample_id'],r['arm']):r for r in read(run/'REFERENCE_METRICS.json')['rows']};assert len(base)==384
for name,m in reg['models'].items():assert sha(m['path'])==m['file_sha256']
calc=[];failed=[]
for r in rows:
 seed,sid,arm=r['seed'],r['sample_id'],r['arm'];fixed=base[(seed,sid,'fixed')]
 assert r['raw_only'] and not r['fallback_to_Fixed'] and r['primary_CPU_threads']==1 and r['generation_seed_equals_A_seed']
 assert r['model_identity']==reg['models'][f'{arm}_{seed}']
 assert r==read(run/'attempts'/f'seed{seed}'/sid/arm/'RESULT.json')
 if r['metric_status']!='ok':failed.append([seed,sid,arm]);assert not r['paired_available'];continue
 mesh=run/r['mesh_file'];assert sha(mesh)==r['mesh_npz_sha256']
 with np.load(mesh,allow_pickle=False) as z:v=z['vertices'];f=z['faces']
 assert array_hash(v,f)==r['stored_array_hash'] and array_hash(v.astype(np.float64),f.astype(np.int64))==r['canonical_geometry_hash']
 assert r['source_nodes_sha256']==fixed['source_nodes_sha256'] and r['reference_surface_hash']==fixed['reference_surface_hash']
 assert r['final_accepted_radius_requery_exact']
 assert [m['seed'] for m in r['surface_metrics']]==[993501+(seed-10)*64+int(sid),1093501+(seed-10)*64+int(sid)]
 assert all(m['points_per_direction']==8192 for m in r['surface_metrics'])
 q=r['quality'];fq=fixed['quality'];assert q['face_count']==len(f)
 margins=[q['min_triangle_angle_deg']-fq['min_triangle_angle_deg'],q['triangle_min_angle_p05_deg']-fq['triangle_min_angle_p05_deg']]
 qualified=bool(len(f)==512 and r['audit']['valid'] and q['degenerate_count']==0 and all(x>=-1e-6 for x in margins))
 assert r['qualified']==qualified and r['gate_available'];same(margins,[r['min_angle_margin_deg'],r['p05_angle_margin_deg']])
 gains=100*(1-np.array([m['rms_bbox_percent'] for m in r['surface_metrics']])/np.array([m['rms_bbox_percent'] for m in fixed['surface_metrics']]))
 same(gains,r['rms_gain_percent_two_measurements']);assert r['qualified_ge2pct_both']==bool(qualified and np.all(gains>=2))
 assert r['improved_both']==bool(np.all(gains>0))
 orig=base[(seed,sid,'a')];a_gains=100*(1-np.array([m['rms_bbox_percent'] for m in r['surface_metrics']])/np.array([m['rms_bbox_percent'] for m in orig['surface_metrics']]))
 same(a_gains,r['comparison_to_original_A']['rms_gain_percent_two_measurements'])
 calc.append({'seed':seed,'sample_id':sid,'arm':arm,'qualified':qualified,'valid':bool(r['audit']['valid']),'faces':len(f),'gains':gains,'margins':margins,'relative_original_A_gains':a_gains})
assert failed==summary['failed_slots']
groups=[]
for seed in ['pooled',10,11]:
 for arm in ['longfit','balanced']:
  rr=[r for r in calc if r['arm']==arm and (seed=='pooled' or r['seed']==seed)];g=next(x for x in summary['groups'] if x['seed']==seed and x['arm']==arm)
  n=128 if seed=='pooled' else 64;assert g['planned_samples']==g['attempts']==n and g['metric_success_count']==len(rr) and g['metric_failed_count']==n-len(rr)
  qualified=sum(x['qualified'] for x in rr);valid=sum(x['valid'] for x in rr);q2=sum(x['qualified'] and np.all(x['gains']>=2) for x in rr);both=sum(np.all(x['gains']>0) for x in rr)
  assert g['qualified_count']==qualified and g['exact_valid_count']==valid and g['qualified_ge2pct_both_count']==q2 and g['improved_both_count']==both
  gains=np.array([x['gains'] for x in rr]);margins=np.array([x['margins'] for x in rr]);ag=np.array([x['relative_original_A_gains'] for x in rr])
  if len(rr):
   for k in [0,1]:
    gd=g['paired_gain_percent_distribution_two_measurements'][k];same([gd['mean'],gd['median'],gd['min'],gd['max']],[gains[:,k].mean(),np.median(gains[:,k]),gains[:,k].min(),gains[:,k].max()])
   group={'seed':seed,'arm':arm,'attempts':n,'calculation_success':len(rr),'qualified':qualified,'exact_valid':valid,'qualified_ge2both':int(q2),'both_improve':int(both),'paired_gain_median':np.median(gains,axis=0).tolist(),'paired_gain_mean':gains.mean(0).tolist(),'paired_gain_min':gains.min(0).tolist(),'quality_margin_median_deg':np.median(margins,axis=0).tolist(),'gain_vs_original_A_median':np.median(ag,axis=0).tolist(),'gain_vs_original_A_mean':ag.mean(0).tolist()}
  else:group={'seed':seed,'arm':arm,'attempts':n,'calculation_success':0,'qualified':0,'exact_valid':0,'qualified_ge2both':0,'both_improve':0}
  groups.append(group)
review={'status':'READ_ONLY_IDENTITY_DENOMINATOR_PAIRED_STATISTICS_VERIFIED','source_sha256':sha(__file__),'protocol_sha256':sha(run/'RUN_PROTOCOL.json'),'registry_sha256':sha(run/'MODEL_REGISTRY.json'),'results_sha256':sha(run/'RESULTS.json'),'summary_sha256':sha(run/'SUMMARY.json'),'groups':groups,'failed_slots':failed,'raw_only':True,'no_new_model_forward_or_geometry_sampling':True,'reference':'RadialSupport.mesh(saved generated nodes), same field as original Fixed/A/QEM; not raw original-object mesh','two_MC_meaning':'two surface samplings of the same reference geometry, not two different ground-truth surfaces','scope':'closed16 learned-outlet exploration; originalAE3200/FM8000 unchanged; no external victory or class generalization'}
(out/'REVIEW.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n');print(json.dumps(review,indent=2,ensure_ascii=False))
