"""Independent CPU validation; does not instantiate RNG or a learned model."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow/experiments/r2_anchor_aware_denoiser_v1')
OUT=ROOT/'stream_independent_audit'; STREAM=ROOT/'stream'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def safe(p):
 p=Path(p).absolute()
 assert not any(q.is_symlink() for q in (p,*p.parents))
 p=p.resolve();assert p.is_relative_to(OUT.resolve()) and p!=OUT.resolve();return p
def write(p,v):
 p=safe(p);tmp=safe(p.with_suffix(p.suffix+'.tmp'))
 with tmp.open('w',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
protect={}
def locked(p,expected=None):
 h=sha(p)
 if expected is not None:assert h==expected,(str(p),'hash')
 protect[str(p)]=h
 return h
locked(__file__)
for p in [STREAM/'plan.json',STREAM/'manifest.json',STREAM/'synthetic/manifest.json',ROOT/'data_audit/training_whitelist.json',ROOT/'data_audit/dev_conditions.json']:locked(p)
plan=read(STREAM/'plan.json');manifest=read(STREAM/'manifest.json');synthetic=read(STREAM/'synthetic/manifest.json')
white=read(ROOT/'data_audit/training_whitelist.json');dev=read(ROOT/'data_audit/dev_conditions.json')
sources={r['condition_id']:r for r in white['rows']+dev['rows']}; cache={}
def source(cid):
 r=sources[cid]
 if cid not in cache:
  locked(r['npz'],r['sha256'])
  with np.load(r['npz'],allow_pickle=False) as z:cache[cid]=(z['C'].copy(),z['Y'].copy())
 return cache[cid]
assert len(plan['rows'])==8000 and len(manifest['batches'])==1000
assert manifest['plan_sha256']==sha(STREAM/'plan.json')
exposure={p:0 for p in white['parent_ids']}; checks=[]
for index,row in enumerate(manifest['batches']):
 step=index+1;pr=plan['rows'][index*8:(index+1)*8]
 assert row['step']==step and row['rows']==pr and len(pr)==8
 assert len({r['uid'] for r in pr})==8 and sum(r['clean'] for r in pr)==4
 for slot,r in enumerate(pr):
  assert r['step']==step and r['slot']==slot and r['sample_index']==index*8+slot
  assert r['condition_id'] in sources and sources[r['condition_id']]['uid']==r['uid']
  assert r['alpha']==1. if r['clean'] else .9-1e-7<=r['alpha']<=1.1+1e-7
  exposure[r['uid']]+=1
 expected_offsets=[0]+list(np.cumsum([r['N'] for r in pr]));den=9*sum(r['N']-r['K'] for r in pr)
 assert row['offsets']==expected_offsets and row['free_coordinate_denominator']==den
 assert Path(row['path']).resolve().is_relative_to((STREAM/'batches').resolve())
 locked(row['path'],row['sha256'])
 receipt_path=Path(row['path']).with_suffix('.json');locked(receipt_path)
 assert read(receipt_path)==row
 checks.append({'step':step,'distinct_parents':8,'clean':4,'width':4,'denominator':den,'hash':'PASS'})
assert exposure==plan['actual_parent_exposure']
selected=sorted({1,500,1000,min(plan['rows'],key=lambda r:r['K'])['step'],max(plan['rows'],key=lambda r:r['N'])['step']})
arraychecks=[]
for step in selected:
 receipt=manifest['batches'][step-1]
 with np.load(receipt['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 assert int(a['step'])==step and int(a['free_coordinate_denominator'])==receipt['free_coordinate_denominator']
 assert 9*int((~a['known']).sum())==receipt['free_coordinate_denominator']
 for name,meta in [('x_d','input_array_sha256'),('target_d','target_array_sha256'),('known','known_array_sha256'),('epsilon','epsilon_array_sha256')]:assert hashlib.sha256(a[name].tobytes()).hexdigest()==receipt[meta]
 for j,r in enumerate(receipt['rows']):
  lo,hi=map(int,a['offsets'][j:j+2]);n=r['N'];k=r['K']
  c,y=source(r['condition_id']);target=np.concatenate([c,y]);target[:,:,0]*=np.float32(r['alpha']);target_d=target*np.float32(.95*.3762)
  face=a['face_permutations'][lo:hi];corner=a['corner_permutations'][lo:hi]
  assert np.array_equal(np.sort(face),np.arange(n)) and np.array_equal(np.sort(corner,axis=1),np.tile(np.arange(3),(n,1)))
  assert np.array_equal(face[a['inverse_face_permutations'][lo:hi]],np.arange(n))
  assert np.array_equal(np.take_along_axis(corner,a['inverse_corner_permutations'][lo:hi],axis=1),np.tile(np.arange(3),(n,1)))
  restored={key:np.empty((n,3,3),np.float32) for key in ['x_d','target_d','target_model','epsilon']}
  for fi in range(n):
   for co in range(3):
    for key in restored:restored[key][face[fi],corner[fi,co]]=a[key][lo+fi,co]
  assert same(restored['target_model'],target) and same(restored['target_d'],target_d)
  assert same(restored['x_d'][:k],target_d[:k]) and not np.any(restored['epsilon'][:k])
  expected=target_d.copy();expected[k:]+=np.float32(.02)*restored['epsilon'][k:]
  assert same(restored['x_d'],expected)
  assert np.array_equal(a['known'][lo:hi],face<k)
  arraychecks.append(dict(step=step,slot=j,condition_id=r['condition_id'],N=n,K=k,source_coordinate_and_inverse_permutation='PASS',known_noise_zero=True,free_additive_noise='BITWISE_MATCH_SAVED_EPSILON'))
synchecks=[]
assert len(synthetic['rows'])==18 and sum(r['clean'] for r in synthetic['rows'])==6
for r in synthetic['rows']:
 locked(r['path'],r['sha256']);locked(r['source_interface_json'],r['source_interface_sha256'])
 with np.load(r['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 c,y=source(r['condition_id']);target=np.concatenate([c,y]);k=len(c)
 assert same(a['C'],c) and same(a['target_model'],target) and same(a['target_d'],target*np.float32(.95*.3762))
 assert same(a['x_d'][:k],a['target_d'][:k]) and not np.any(a['epsilon'][:k])
 expected=a['target_d'].copy();expected[k:]+=np.float32(.02)*a['epsilon'][k:];assert same(a['x_d'],expected)
 if r['clean']:assert same(a['x_d'],a['target_d']) and not np.any(a['epsilon'])
 interface=read(r['source_interface_json'])
 for key in ['supported_Y_local_corner_indices','supported_Y_flat_corner_indices','supported_joint_C_then_Y_flat_corner_indices']:assert np.array_equal(a[key],np.asarray(interface[key],np.int64))
 ci=a['supported_Y_local_corner_indices'];assert np.array_equal(a['supported_joint_C_then_Y_flat_corner_indices'],(ci[:,0]+k)*3+ci[:,1])
 synchecks.append(dict(synthetic_id=r['synthetic_id'],clean=r['clean'],supported_free_corners=len(ci),result='PASS'))
for p,h in protect.items():assert sha(p)==h,('CHANGED',p)
summary=dict(status='PASS',all_batch_metadata_and_hashes=len(checks),training_exposures=sum(exposure.values()),train_parent_count=len(exposure),selected_array_batches=selected,selected_array_microbatches=len(arraychecks),synthetic_inputs=len(synchecks),source_code_sha256=sha(__file__),protected_files=len(protect),all_before_after_equal=True,new_RNG_instantiations=0,new_noise_draws=0,model_calls=0,GPU_calls=0,notes='Independent direct scatter inverse, source reconstruction, saved epsilon arithmetic; no re-generation of Gaussian arrays or model calls.')
write(OUT/'REPORT.json',summary);write(OUT/'batch_metadata.json',checks);write(OUT/'selected_arrays.json',arraychecks);write(OUT/'synthetic.json',synchecks);write(OUT/'protected.json',protect)
print(json.dumps(summary))
