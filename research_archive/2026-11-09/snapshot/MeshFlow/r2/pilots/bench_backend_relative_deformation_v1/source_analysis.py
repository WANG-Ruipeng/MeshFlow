"""Three-way held-out source recovery, target-relative error and source shape.

All input examples and official/previous proposals are cached and read-only.
Only the current candidate's fifteen identity proposals receive a new GEO pass.
No model, torch, prediction relabeling or output-dependent target normalization.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import csv,html,json,sys,time
import numpy as np
from .common import *
from r2.pilots.bench_backend_counterfactual_v1 import core as B

LABELS=('OFFICIAL_SOURCE','PREVIOUS_SOURCE','CANDIDATE_SOURCE')
def records(p):
 with resolve(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def area(tri):
 tri=np.asarray(tri,dtype=np.float64)
 return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
def geomstats(tri,k,source,sourceids=None):
 aa=area(tri);sa=area(source);ids=np.arange(len(tri)) if sourceids is None else sourceids
 proxy=(source[:,:,1].min(1)<source[:k,:,1].min()-.02*L)&(source.mean(1)[:,1]<source[:k,:,1].min())
 proxy[:k]=False;pick=proxy[ids];den=float(sa[proxy].sum())
 return dict(free_faces=len(tri)-k,free_area=float(aa[k:].sum()),below_C_proxy_source_faces=int(proxy.sum()),
  below_C_proxy_area=float(aa[pick].sum()),below_C_proxy_source_area=den,
  below_C_proxy_area_ratio=float(aa[pick].sum()/den) if den else None,
  free_y_span=float(np.ptp(tri[k:,:,1])) if len(tri)>k else 0.)
def prop_path(label,rid):
 if label=='OFFICIAL_SOURCE':return PREV/'evaluation/OFFICIAL_SOURCE'/rid/'proposal.npz'
 if label=='PREVIOUS_SOURCE':return PREV/'evaluation/CANDIDATE_SOURCE'/rid/'proposal.npz'
 return OUT/'evaluation/CANDIDATE_SOURCE'/rid/'proposal.npz'
def post_path(label,parent):
 if label=='OFFICIAL_SOURCE':return PREV/'source_analysis'/parent/'OFFICIAL_SOURCE/post.npz'
 if label=='PREVIOUS_SOURCE':return PREV/'source_analysis'/parent/'CANDIDATE_SOURCE/post.npz'
 return OUT/'source_analysis'/parent/'CANDIDATE_SOURCE/post.npz'
def relative_values(proposal,cache,delta):
 # Same FP32 edge/residual/radial norm as the train term, using fixed target cache.
 x=np.asarray(proposal,np.float32)*np.float32(SCALE);ids=cache['face_ids'];p=x[ids]
 e=np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=-1)
 d=np.matmul(e-cache['target_edges'],cache['inverse'])
 r=np.linalg.norm(d.reshape(-1,6),axis=1)/np.float32(np.sqrt(6.))
 loss=np.where(r<=delta,.5*r*r,delta*(r-.5*delta))
 return r,loss
def avg(v):
 v=[float(x) for x in v if x is not None and x!='']
 return float(np.mean(v)) if v else None

def run():
 done=OUT/'source_analysis/COMPLETE.json'
 if done.exists():r=read(done);verify(r['files']);verify(r['protected_inputs']);return r
 cm=read(OUT/'data/relative_cache_manifest.json');config=read(OUT/'loss_config.json')
 assert config==cm['config'] and sha(OUT/'loss_config.json')==cm['config_sha256']
 m=read(PREV/'data/manifest.json');old=read(OLD/'manifest.json');byparent={t['parent_id']:t for t in old['tasks']}
 assert len(m['validation_rows'])==30 and len({r['parent_id'] for r in m['validation_rows']})==15
 oldreceipt=read(PREV/'source_analysis/COMPLETE.json');verify(oldreceipt['files'])
 # All proposal receipts must be complete before this new CPU source stage starts.
 protected={str(PREV/'data/manifest.json'):sha(PREV/'data/manifest.json'),str(OUT/'data/relative_cache_manifest.json'):sha(OUT/'data/relative_cache_manifest.json'),
             str(OUT/'loss_config.json'):sha(OUT/'loss_config.json'),str(PREV/'source_analysis/COMPLETE.json'):sha(PREV/'source_analysis/COMPLETE.json')}
 for row in m['validation_rows']:
  assert sha(row['path'])==row['sha256'];protected[row['path']]=row['sha256']
  for label in LABELS:
   path=prop_path(label,row['row_id']);receipt=read(path.parent/'receipt.json');verify(receipt['files'])
   protected[str(path)]=sha(path);protected[str(path.parent/'receipt.json')]=sha(path.parent/'receipt.json')
 oldstage=records(PREV/'source_analysis/per_stage.csv');stages=[]
 for r in oldstage:
  r=dict(r);r['label']='PREVIOUS_SOURCE' if r['label']=='CANDIDATE_SOURCE' else r['label'];r['result_origin']='READ_ONLY_PREVIOUS_SOURCE_STAGE';stages.append(r)
 source_old={(r['row_id'],r['label']):r for r in records(PREV/'source_analysis/per_instance.csv')}
 cache_rows={r['row_id']:r for r in cm['validation_rows']};parent_meta={r['cache_id']:r for r in cm['parents']}
 scores=[];faces=[];plots=[];geo_count=0;tick=time.perf_counter()
 P,geo,io,_=B.dependencies()
 for row in m['validation_rows']:
  a=arrays(row['path']);src=a['target_model'];x=a['x_model'];mask=a['loss_mask'].astype(bool);k=int(a['known_faces']);rid=row['row_id']
  cr=cache_rows[rid];assert sha(cr['cache_path'])==cr['cache_sha256'];cache=arrays(cr['cache_path']);meta=parent_meta[cr['cache_id']]
  input_r,input_loss=relative_values(x,cache,float(config['huber_delta']))
  for label in LABELS:
   proposal=arrays(prop_path(label,rid))['official_unconstrained_proposal'];restored=proposal.copy();restored[:k]=src[:k]
   delta=proposal[mask].astype(float)-src[mask].astype(float);input_delta=x[mask].astype(float)-src[mask].astype(float)
   rv,lv=relative_values(proposal,cache,float(config['huber_delta']))
   scores.append(dict(row_id=rid,parent_id=row['parent_id'],kind=row['kind'],label=label,masked_corners=int(mask.sum()),
      masked_model_MSE=float(np.mean(delta**2)),masked_denoiser_MSE=float(np.mean(delta**2)*SCALE**2),input_masked_model_MSE=float(np.mean(input_delta**2)),
      C_restored_exact=bool(np.array_equal(restored[:k],src[:k])),free_proposal_restoration_exact=bool(np.array_equal(restored[k:],proposal[k:])),
      relative_r_mean=float(rv.mean()) if len(rv) else None,relative_r_median=float(np.median(rv)) if len(rv) else None,
      relative_r_p95=float(np.quantile(rv,.95)) if len(rv) else None,relative_huber_mean=float(lv.mean()) if len(lv) else None,
      input_relative_r_mean=float(input_r.mean()) if len(input_r) else None,input_relative_huber_mean=float(input_loss.mean()) if len(input_loss) else None,
      relative_eligible_faces=meta['eligible_faces'],relative_excluded_rank_faces=meta['excluded_rank_faces'],relative_clipped_faces=meta['clipped_faces'],
      result_origin='NEW_CANDIDATE_SOURCE' if label=='CANDIDATE_SOURCE' else 'READ_ONLY_PROPOSAL_RECOMPUTED_DIAGNOSTIC',**geomstats(restored,k,src)))
   if label!='CANDIDATE_SOURCE':
    previous=source_old[(rid,'CANDIDATE_SOURCE' if label=='PREVIOUS_SOURCE' else label)]
    assert np.isclose(scores[-1]['masked_model_MSE'],float(previous['masked_model_MSE']),rtol=1e-12,atol=1e-15)
   for fid,r,l in zip(cache['face_ids'],rv,lv):
    faces.append(dict(row_id=rid,parent_id=row['parent_id'],kind=row['kind'],label=label,face_index=int(fid),source_face_id=int(a['source_face_ids'][fid]),relative_r=float(r),relative_huber=float(l)))
   if row['kind']!='IDENTITY' or label!='CANDIDATE_SOURCE':continue
   t=byparent[row['parent_id']];raw=arrays(t['original']['raw_path']);source=arrays(t['source_condition_path'])
   assert int(raw['known_faces'])==k and int(raw['input_faces'])==len(src)
   assert not raw['raw_seam_slot_mask'].any() and (raw['raw_anchor_ids'][k:]<0).all()
   assert np.array_equal(src[:k],raw['C_vertices'][raw['C_faces_model']])
   raw['raw']=src.copy();raw['output']=src.copy();dest=OUT/'source_analysis'/row['parent_id']/label;complete=dest/'complete.json'
   identity=dict(proposal_sha256=sha(prop_path(label,rid)),source_sha256=sha(row['path']),raw_sha256=sha(t['original']['raw_path']),source_condition_sha256=sha(t['source_condition_path']))
   if complete.exists():
    receipt=read(complete);assert receipt['identity']==identity;verify(receipt['files']);post=arrays(dest/'post.npz')
   else:
    if dest.exists():raise RuntimeError('Partial source GEO retained; no automatic replay: '+str(dest))
    dest.mkdir(parents=True);write(dest/'attempt.json',dict(identity=identity,model_calls=0,GEO_calls=1))
    pre,canonical,post,g,ori,anchors=P.geometry_from_proposal(raw,proposal,source);geo_count+=1
    assert np.array_equal(pre[:k],src[:k]) and np.array_equal(pre[k:],proposal[k:])
    npz(dest/'post.npz',post);npz(dest/'post_canonical.npz',canonical);npz(dest/'pre_geo.npz',dict(output=pre))
    write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']));write(dest/'anchors.json',anchors)
    write(complete,dict(status='COMPLETE',identity=identity,GEO_calls=1,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
   for stage,tri,ids in [('SOURCE',src,None),('PROPOSAL',proposal,None),('RESTORED_C',restored,None),('GEO',post['output'],post['source_face_ids'])]:
    stages.append(dict(parent_id=row['parent_id'],row_id=rid,label=label,stage=stage,result_origin='NEW_CANDIDATE_SOURCE_STAGE',**geomstats(tri,k,src,ids)))
  if row['kind']=='IDENTITY':plots.append(dict(row=row,task=byparent[row['parent_id']]))
 assert len(scores)==90 and len(stages)==180 and len(plots)==15
 csvwrite(OUT/'source_analysis/per_instance.csv',scores);csvwrite(OUT/'source_analysis/per_face_relative.csv',faces);csvwrite(OUT/'source_analysis/per_stage.csv',stages)
 fields=['masked_model_MSE','input_masked_model_MSE','below_C_proxy_area_ratio','relative_r_mean','relative_r_median','relative_r_p95','relative_huber_mean','input_relative_r_mean','input_relative_huber_mean']
 summary=[]
 for kind in ('IDENTITY','CORNER_NOISE'):
  for label in LABELS:
   rs=[r for r in scores if r['kind']==kind and r['label']==label]
   summary.append(dict(kind=kind,label=label,parents=len(rs),**{f:avg(r[f] for r in rs) for f in fields}))
 csvwrite(OUT/'source_analysis/parent_macro_summary.csv',summary)
 render(plots)
 if 'torch' in sys.modules:raise RuntimeError('Source CPU geometry unexpectedly imported torch')
 verify(protected)
 result=dict(status='COMPLETE',source_rows=90,identity_parents=15,new_GEO_calls=geo_count,total_candidate_GEO=15,reused_old_stage_rows=120,source_stage_rows=180,
   additional_model_calls=0,noise_draws=0,relative_definition=config,summary=summary,seconds=time.perf_counter()-tick,protected_inputs=protected,
   limitation='Below-C area is an unlabelled geometric support proxy; target-relative diagnostics apply to paired source recovery, not unique-parent style matching for free generation.',
   files={str(p):sha(p) for p in (OUT/'source_analysis').rglob('*') if p.is_file()})
 write(done,result);return result

def render(plots):
 from r2.pilots.chair_multipart_seam_v1 import gallery as G
 H,D=G.H,G.D;plt=H._mpl();body='<meta charset="utf-8"><h1>Held-out source recovery: original D0 / previous adapted D0 / relative-deformation D0</h1>'
 for e in plots:
  row,t=e['row'],e['task'];p=OUT/'source_analysis'/row['parent_id'];a=arrays(row['path']);src=a['target_model'];k=int(a['known_faces']);rid=row['row_id'];q=t['request']
  specs=[('INPUT',src)]
  for label in LABELS:specs.extend([(label+'_PROPOSAL',arrays(prop_path(label,rid))['official_unconstrained_proposal']),(label+'_GEO',arrays(post_path(label,row['parent_id']))['output'])])
  fig=plt.figure(figsize=(32,12),dpi=120);grid=fig.add_gridspec(4,7,wspace=.15,hspace=.25)
  for col,(name,tri) in enumerate(specs):
   role=np.asarray(['C']*k+['BULK']*(len(tri)-k));g=dict(C=tri[:k],F=tri[k:],C_original=src[:k],tri=tri)
   for r,az in enumerate((-60,120)):
    raster,_=G.role_raster(tri,role,k,D.basis(az,23),np.asarray(q['gallery_depth_bounds']['center']),q['gallery_depth_bounds']['span'])
    ax=fig.add_subplot(grid[r,col]);ax.imshow(raster);ax.axis('off');ax.set_title(name,fontsize=8)
   for r,v in ((2,'front'),(3,'side')):
    ax=fig.add_subplot(grid[r,col]);H.draw_section(ax,g,[],q['gallery_bounds'],L,v,1);ax.set_title(name+' '+v,fontsize=7)
  fig.suptitle(row['parent_id']+' | fixed same-instance source recovery; unconstrained C only shown diagnostically')
  image=p/'identity.png';fig.savefig(image);plt.close(fig)
  body+='<h2>'+html.escape(row['parent_id'])+'</h2><a href="'+row['parent_id']+'/identity.png"><img width="100%" src="'+row['parent_id']+'/identity.png"></a>'
 with (OUT/'source_analysis/index.html').open('x',encoding='utf-8') as f:f.write(body)

if __name__=='__main__':
 r=run();print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected_inputs','summary')}))
