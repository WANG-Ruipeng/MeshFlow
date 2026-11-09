"""CPU-only target geometry statistics and frozen relative-deformation caches.

This stage reuses the previous 2,000 registered instances; it never samples noise.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_relative_deformation_v1'
OLD=ROOT/'experiments/r2_bench_backend_shape_preservation_v1'
SCALE=.95*.3762
QUANTILES=[0,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,.999,1]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def rows_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def arrays(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def quantiles(v):return {str(q):float(x) for q,x in zip(QUANTILES,np.quantile(np.asarray(v,float),QUANTILES))}
def weighted_quantiles(values,weights):
 values=np.asarray(values,float);weights=np.asarray(weights,float);order=np.argsort(values,kind='stable')
 values=values[order];weights=weights[order];cdf=np.cumsum(weights)/weights.sum()
 return {str(q):float(values[min(int(np.searchsorted(cdf,q,side='left')),len(values)-1)]) for q in QUANTILES}

def stats():
 dest=OUT/'data/target_stats';receipt=dest/'COMPLETE.json'
 if receipt.exists():
  result=read(receipt)
  for p,h in result['files'].items():assert sha(p)==h
  return result
 if dest.exists():raise RuntimeError('Partial statistics exist; no silent replacement: '+str(dest))
 tick=time.perf_counter();manifest_path=OLD/'data/manifest.json';m=read(manifest_path)
 by={}
 for row in m['train_rows']:by.setdefault(row['parent'],row)
 assert len(by)==16 and len(m['train_rows'])==2000 and len(m['stream'])==500
 perface=[];perparent=[];source_hashes={str(manifest_path):sha(manifest_path)}
 for parent,row in sorted(by.items()):
  assert sha(row['path'])==row['sha256'];source_hashes[row['path']]=row['sha256']
  a=arrays(row['path']);y=a['target_model'].astype(np.float64)*SCALE;k=int(a['known_faces']);mask=a['loss_mask'].astype(bool)
  assert not mask[:k].any();valid=mask.all(1);valid[:k]=False
  E=np.stack((y[:,1]-y[:,0],y[:,2]-y[:,0]),axis=-1)
  sv=np.linalg.svd(E,compute_uv=False);area=np.linalg.norm(np.cross(E[:,:,0],E[:,:,1]),axis=1)/2
  cond=sv[:,0]/np.maximum(sv[:,1],np.finfo(float).tiny)
  ids=np.flatnonzero(valid)
  assert len(ids)>0 and np.isfinite(cond[ids]).all() and (sv[ids,1]>0).all()
  for i in ids:
   perface.append(dict(parent=parent,face_index=int(i),source_face_id=int(a['source_face_ids'][i]),N=len(y),K=k,
       sigma_max=float(sv[i,0]),sigma_min=float(sv[i,1]),condition_number=float(cond[i]),area_denoiser=float(area[i]),
       parent_equal_weight=1/(16*len(ids))))
  perparent.append(dict(parent=parent,N=len(y),K=k,free_faces=len(y)-k,trusted_free_faces=len(ids),
      partially_masked_free_faces=int(((mask.any(1))&~mask.all(1))[k:].sum()),
      target_sha256=hashlib.sha256(a['target_model'].tobytes()).hexdigest(),input_path=row['path'],
      sigma_max=quantiles(sv[ids,0]),sigma_min=quantiles(sv[ids,1]),condition_number=quantiles(cond[ids]),area_denoiser=quantiles(area[ids])))
 keys=['sigma_max','sigma_min','condition_number','area_denoiser']
 pooled={key:quantiles([r[key] for r in perface]) for key in keys}
 balanced={key:weighted_quantiles([r[key] for r in perface],[r['parent_equal_weight'] for r in perface]) for key in keys}
 floors=[]
 for floor in (1e-5,1e-4,5e-4,.001,.002,.005,.01):
  low=[r for r in perface if r['sigma_min']<floor]
  floors.append(dict(sigma_floor_denoiser=floor,faces_below=len(low),face_fraction=len(low)/len(perface),parent_equal_fraction=sum(r['parent_equal_weight'] for r in low)))
 result=dict(status='COMPLETE_STATS_ONLY_NO_PARAMETERS_FROZEN',parents=16,unique_trusted_free_faces=len(perface),training_rows=2000,stream_updates=500,
     SCALE=SCALE,E_definition='3x2 columns [target_corner1-target_corner0, target_corner2-target_corner0] in denoiser units; preserved source corner order',
     coordinate_dtype='float64 offline statistics from float32 saved target',
     weighted_quantile_definition='Left inverse empirical CDF; each parent total mass1/16, each eligible face uniform within parent',
     all_face_quantiles=pooled,parent_equal_quantiles=balanced,parent_N=quantiles([r['N'] for r in perparent]),
     parent_trusted_faces=quantiles([r['trusted_free_faces'] for r in perparent]),per_parent=perparent,absolute_floor_impact=floors,
     noise_draws=0,model_calls=0,GPU_calls=0,source_hashes=source_hashes,seconds=time.perf_counter()-tick)
 dest.mkdir(parents=True);rows_csv(dest/'per_face.csv',perface);write(dest/'statistics.json',result)
 result['files']={str(dest/'per_face.csv'):sha(dest/'per_face.csv'),str(dest/'statistics.json'):sha(dest/'statistics.json')}
 write(receipt,result);return result

def prepare():
 from .relative_loss import prepare_target,validate_config
 dest=OUT/'data/relative_cache';mp=OUT/'data/relative_cache_manifest.json'
 if mp.exists():
  r=read(mp)
  for p,h in r['protected_files'].items():assert sha(p)==h,p
  for row in r['parents']:assert sha(row['path'])==row['sha256'],row['path']
  return r
 if dest.exists():raise RuntimeError('Partial relative cache retained; no automatic replacement')
 config_path=OUT/'loss_config.json';config=validate_config(read(config_path))
 original=OLD/'data/manifest.json';m=read(original);tick=time.perf_counter()
 assert len(m['train_rows'])==2000 and len(m['validation_rows'])==30
 dest.mkdir(parents=True);parents=[];indices={};output_rows={};protected={str(original):sha(original),str(config_path):sha(config_path)}
 for group,key in [('train','train_rows'),('validation','validation_rows')]:
  out=[]
  for i,row in enumerate(m[key]):
   assert sha(row['path'])==row['sha256'],row['path'];protected[row['path']]=row['sha256']
   a=arrays(row['path']);parent=row.get('parent',row['parent_id']);cid=group+'__'+parent
   fingerprint=hashlib.sha256(a['target_model'].tobytes()+a['loss_mask'].tobytes()+a['known_faces'].tobytes()).hexdigest()
   if cid not in indices:
    pr=prepare_target(a['target_model'],a['loss_mask'],SCALE,config)
    path=dest/(cid+'.npz')
    with path.open('xb') as f:np.savez_compressed(f,**{k:pr[k] for k in ('face_ids','target_edges','inverse')})
    item=dict(cache_id=cid,split=group,parent=parent,parent_id=parent,path=str(path),sha256=sha(path),
         N=int(len(a['target_model'])),K=int(a['known_faces']),target_mask_fingerprint=fingerprint,
         representative_input_path=row['path'],representative_input_sha256=row['sha256'],
         **{k:v for k,v in pr.items() if k not in ('face_ids','target_edges','inverse')})
    indices[cid]=item;parents.append(item)
   item=indices[cid]
   assert item['target_mask_fingerprint']==fingerprint,(cid,'Target/mask changed across reused instances')
   out.append(dict(row_index=i,row_id=row.get('row_id',row.get('id')),parent=parent,parent_id=parent,
       cache_id=cid,cache_path=item['path'],cache_sha256=item['sha256'],input_path=row['path'],input_sha256=row['sha256']))
  output_rows[key]=out
 assert sum(p['split']=='train' for p in parents)==16 and sum(p['split']=='validation' for p in parents)==15
 assert not {p['parent'] for p in parents if p['split']=='train'}&{p['parent'] for p in parents if p['split']=='validation'}
 train=[p for p in parents if p['split']=='train']
 result=dict(status='READY',schema='target_relative_deformation_cache_v1',source_manifest_path=str(original),source_manifest_sha256=sha(original),
     config_path=str(config_path),config_sha256=sha(config_path),config=config,SCALE=SCALE,parents=parents,**output_rows,
     train_summary=dict(parents=16,trusted_faces=sum(p['trusted_faces'] for p in train),eligible_faces=sum(p['eligible_faces'] for p in train),
         clipped_faces=sum(p['clipped_faces'] for p in train),excluded_rank_faces=sum(p['excluded_rank_faces'] for p in train),
         maximum_inverse_norm=max(p['maximum_inverse_norm'] for p in train)),
     protected_files=protected,model_calls=0,GPU_calls=0,noise_draws=0,copied_source_arrays=0,seconds=time.perf_counter()-tick,
     limitations=['Singular floor changes normalization but does not remove eligible faces.',
       'Corner order follows the saved paired target; clipped normalization is not claimed invariant to arbitrary corner permutation.',
       'Validation geometry did not select the fixed floor, rank threshold, Huber threshold or coefficient.'])
 write(mp,result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['stats','prepare']);a=p.parse_args()
 r=stats() if a.stage=='stats' else prepare()
 print(json.dumps({k:v for k,v in r.items() if k not in ('per_parent','source_hashes','files','parents','train_rows','validation_rows','protected_files')},ensure_ascii=False))
