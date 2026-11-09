"""Independent cached-result arithmetic and identity audit. No models or rescoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv, json, math, hashlib, collections, datetime
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/e/MeshFlow');OUT=ROOT/'experiments/r2_bench_backend_shape_preservation_v1'
def j(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def c(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def num(v):
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None
def eq(a,b):
    a,b=num(a),num(b)
    if a is None or b is None:return a is None and b is None
    return math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-12)
def avg(vals):
    vals=[num(v) for v in vals];vals=[v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None
def groups(rows,keys):
    d=collections.defaultdict(list)
    for r in rows:d[tuple(r[k] for k in keys)].append(r)
    return d
def agg(rows,keys,fields):
    result=[]
    for identity,rs in sorted(groups(rows,keys).items()):
        out=dict(zip(keys,identity));out['rows']=len(rs)
        for f in fields:
            out[f]=avg(r.get(f) for r in rs);out[f+'__valid_n']=sum(num(r.get(f)) is not None for r in rs)
        result.append(out)
    return result
def compare(rows,path,keys,fields):
    other={tuple(r[k] for k in keys):r for r in c(path)}
    assert len(rows)==len(other),(path,len(rows),len(other))
    for r in rows:
        q=other[tuple(r[k] for k in keys)]
        for f in fields:assert eq(r[f],q[f]),(path,tuple(r[k] for k in keys),f,r[f],q[f])

manifest=j(OUT/'data/manifest.json');receipt=j(OUT/'scores/COMPLETE.json')
tasks=j(ROOT/'experiments/r2_bench_geo_only_acceptance_v1/manifest.json')['tasks']
metrics=receipt['metric_fields'];assert len(metrics)==177
allrows=c(OUT/'scores/per_output.csv');new=c(OUT/'scores/new_per_output.csv');old=c(OUT/'scores/original_reference_scores.csv')
assert len(allrows)==270 and len(new)==60 and len(old)==210
assert len({(r['label'],r['stage'],r['request_id']) for r in allrows})==270
coverage=[]
expected={t['request_id'] for t in tasks}
for (label,stage),rows in groups(allrows,['label','stage']).items():
    assert len(rows)==30 and {r['request_id'] for r in rows}==expected
    assert len({r['parent_id'] for r in rows})==15
    assert all(len(rs)==2 and {int(r['noise_index']) for r in rs}=={0,1} for rs in groups(rows,['parent_id']).values())
    coverage.append(dict(label=label,stage=stage,parents=15,requests=30))
conditions=agg(allrows,['label','stage','eval_group','parent_id','condition_id'],metrics)
parents=agg(conditions,['label','stage','eval_group','parent_id'],metrics)
macros=agg(parents,['label','stage','eval_group'],metrics)
allcheck=metrics+[m+'__valid_n' for m in metrics]+['rows']
compare(conditions,OUT/'scores/per_condition.csv',['label','stage','eval_group','parent_id','condition_id'],allcheck)
compare(parents,OUT/'scores/per_parent.csv',['label','stage','eval_group','parent_id'],allcheck)
compare(macros,OUT/'scores/parent_macro_summary.csv',['label','stage','eval_group'],allcheck)

# Check underlying newly scored rows instead of trusting the combined table only.
by={(r['request_id'],r['stage']):r for r in new}
for t in tasks:
    for row in j(OUT/'cpu/SHAPE_D0'/t['request_id']/'metric_rows.json'):
        tab=by[(t['request_id'],row['stage'])]
        for f in metrics:assert eq(row[f],tab[f]),(t['request_id'],row['stage'],f)
hash_count=0
for table in ('files','reference_hashes','source_hashes'):
    for p,sha in receipt[table].items():
        assert h(p)==sha,(table,p);hash_count+=1

# Recompute coordinate preservation from cached arrays, including original source winding.
c_checks=[]
for t in tasks:
    rid=t['request_id'];k=int(t['K']);raw=arr(t['new_raw_path']);inp=arr(t['request']['inference_npz'])
    expectedC=inp['C_vertices'][inp['C_faces_original']]
    pre=arr(OUT/'geometry'/rid/'pre_geo.npz')['output']
    proposal=arr(OUT/'evaluation/CANDIDATE_RAW'/rid/'proposal.npz')['official_unconstrained_proposal']
    assert np.array_equal(pre[:k],raw['raw'][:k]) and np.array_equal(pre[k:],proposal[k:])
    for stage,p in [('BASE_POST',OUT/'geometry'/rid/'post.npz'),('CLEAN_FINAL_POST',OUT/'cpu/SHAPE_D0'/rid/'CLEAN_FINAL_POST/post.npz')]:
        post=arr(p);assert np.array_equal(post['output'][:k],expectedC),(rid,stage)
        assert np.array_equal(post['vertices'][post['faces'][:k]],expectedC),(rid,stage,'indexed_C')
        c_checks.append(dict(request_id=rid,stage=stage,C_coordinates_original_order_exact=True,C_indexed_faces_coordinates_exact=True))

# Independently compute held-out masked coordinate MSE and source-area retention.
SCALE=.95*.3762;L=math.sqrt(3)*2/.3762
def areas(x):
    z=np.asarray(x,dtype=float)
    return np.linalg.norm(np.cross(z[:,1]-z[:,0],z[:,2]-z[:,0]),axis=1)/2
def geometry(tri,k,source,ids):
    tri=np.asarray(tri);sa=areas(source);ta=areas(tri)
    lowest=float(source[:k,:,1].min())
    proxy=np.array([i>=k and x[:,1].min()<lowest-.02*L and x[:,1].mean()<lowest for i,x in enumerate(source)])
    den=sa[proxy].sum();picked=proxy[np.asarray(ids,dtype=int)]
    return dict(free_faces=len(tri)-k,free_area=float(ta[k:].sum()),below_C_proxy_source_faces=int(proxy.sum()),
      below_C_proxy_area=float(ta[picked].sum()),below_C_proxy_source_area=float(den),
      below_C_proxy_area_ratio=float(ta[picked].sum()/den) if den else None,
      free_y_span=float(tri[k:,:,1].max()-tri[k:,:,1].min()) if len(tri)>k else 0.)
instance=[];stages=[]
for row in manifest['validation_rows']:
    aa=arr(row['path']);src=aa['target_model'];x=aa['x_model'];mask=aa['loss_mask'].astype(bool);k=int(aa['known_faces'])
    for label in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
        proposal=arr(OUT/'evaluation'/label/row['row_id']/'proposal.npz')['official_unconstrained_proposal']
        # Flatten XYZ so the MSE denominator is explicit and independent.
        mse=float(np.mean((proposal[mask].astype(np.float64)-src[mask].astype(np.float64)).reshape(-1)**2))
        inp=float(np.mean((x[mask].astype(np.float64)-src[mask].astype(np.float64)).reshape(-1)**2))
        restored=proposal.copy();restored[:k]=src[:k]
        item=dict(row_id=row['row_id'],parent_id=row['parent_id'],kind=row['kind'],label=label,masked_corners=int(mask.sum()),
          masked_model_MSE=mse,masked_denoiser_MSE=mse*SCALE**2,input_masked_model_MSE=inp,**geometry(restored,k,src,np.arange(len(src))))
        instance.append(item)
        if row['kind']=='IDENTITY':
            post=arr(OUT/'source_analysis'/row['parent_id']/label/'post.npz')
            for name,tri,ids in [('SOURCE',src,np.arange(len(src))),('PROPOSAL',proposal,np.arange(len(src))),('RESTORED_C',restored,np.arange(len(src))),('GEO',post['output'],post['source_face_ids'])]:
                stages.append(dict(row_id=row['row_id'],parent_id=row['parent_id'],label=label,stage=name,**geometry(tri,k,src,ids)))
f=['masked_corners','masked_model_MSE','masked_denoiser_MSE','input_masked_model_MSE','free_faces','free_area','below_C_proxy_area_ratio','free_y_span']
compare(instance,OUT/'source_analysis/per_instance.csv',['row_id','label'],f)
compare(stages,OUT/'source_analysis/per_stage.csv',['row_id','label','stage'],f[4:])
source_summary=agg(instance,['kind','label'],['masked_model_MSE','input_masked_model_MSE','below_C_proxy_area_ratio'])
compare(source_summary,OUT/'analysis/source_summary.csv',['kind','label'],['masked_model_MSE','input_masked_model_MSE','below_C_proxy_area_ratio'])

# Recompute the parent gate from original per-output rows, not its CSV.
pby={(r['label'],r['stage'],r['parent_id']):r for r in parents};gate=[]
for r in parents:
    if r['label']!='SHAPE_D0' or r['stage']!='CLEAN_FINAL_POST':continue
    oldr=pby[('COVERAGE_FINAL','CLEAN_FINAL_POST',r['parent_id'])]
    item=dict(parent_id=r['parent_id'],eval_group=r['eval_group'])
    for short,name in [('FF','FF_illegal_affected_free_fraction'),('CF','CF_illegal_affected_free_fraction'),('I','ConflictCov_C')]:item[short+'_delta_pp']=100*(r[name]-oldr[name])
    item['alert']=item['FF_delta_pp']>=10 or item['CF_delta_pp']>=5 or item['I_delta_pp']>=5;gate.append(item)
compare(gate,OUT/'analysis/geometry_gate_per_parent.csv',['parent_id'],['FF_delta_pp','CF_delta_pp','I_delta_pp'])
oldalerts=[r for r in gate if r['alert'] and r['eval_group'] in ('CURRENT_TRAIN','DEV')]
assert len(oldalerts)==j(OUT/'analysis/COMPLETE.json')['geometry_alert_old_parents']
assert (len(oldalerts)<2)==j(OUT/'analysis/COMPLETE.json')['geometry_gate_pass']

losses=[json.loads(s) for s in (OUT/'training/losses.jsonl').read_text().splitlines()]
assert [r['update'] for r in losses]==list(range(1,501))
exposures=collections.Counter();kinds=collections.Counter()
for update,r in enumerate(losses):
    assert [v['row_index'] for v in r['samples']]==manifest['stream'][update]
    assert r['cumulative_forwards']==4*(update+1) and r['cumulative_backwards']==4*(update+1)
    assert eq(r['mse_denoiser'],avg(s['mse_denoiser'] for s in r['samples']))
    for s in r['samples']:
        source=manifest['train_rows'][s['row_index']]
        assert s['parent']==source['parent'] and s['kind']==source['kind'] and eq(s['sigma'],source['sigma'])
        exposures[s['parent']]+=1;kinds[s['kind']]+=1
assert len(exposures)==16 and set(exposures.values())=={125}
assert not set(exposures)&{r['parent_id'] for r in manifest['validation_rows']}

keys=['CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_minimum_angle_lt5_fraction','actual_free_faces','deleted_free_faces','nonmanifold_edges','C_F_valid_edge_count','opposite_winding_reached_free_area_fraction']
keymacros=[{k:r[k] for k in ['label','stage','eval_group','rows']+keys} for r in macros if (r['label'],r['stage']) in [('COVERAGE_FINAL','CLEAN_FINAL_POST'),('SHAPE_D0','CLEAN_FINAL_POST'),('COVERAGE_FINAL','GEO_ONLY')]]
result=dict(status='PASS_NO_ARITHMETIC_OR_IDENTITY_MISMATCH',reviewer='backend_data',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  method='Independent NumPy/CSV arithmetic from cached per-output rows and saved proposals/NPZ; no production aggregate or source_analysis imported.',
  scope=dict(model_calls=0,GPU_calls=0,rescoring_calls=0,new_GEO_calls=0,source_MSE_instances=60,source_stage_rows=120,score_fields=177,new_scores=60,original_scores=210),
  score_coverage=coverage,hash_checks=hash_count,parent_macro_summary_key_fields=keymacros,
  source_summary=source_summary,geometry_gate_per_parent=gate,geometry_alert_old_parents=len(oldalerts),geometry_gate_pass=len(oldalerts)<2,
  C_preservation=dict(paired_requests=30,post_stages_checked=60,all_exact=True,details=c_checks),
  training=dict(updates=500,forwards=2000,backwards=2000,parent_exposures=dict(exposures),kinds=dict(kinds),validation_parent_disjoint=True),
  limitations=['Source below-C area is a geometric proxy, not semantic support success.','C coordinate/indexed-face checks do not certify all generated topology.','The metric computations themselves were inherited and were not rerun.','No claim that adaptation held-out parents are unseen by official pretraining.'])
dest=OUT/'audits/results_review.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=result['status'],hash_checks=hash_count,source_summary=source_summary,geometry_alert_old_parents=len(oldalerts),alerts=oldalerts,path=str(dest),sha256=h(dest)),ensure_ascii=False))
