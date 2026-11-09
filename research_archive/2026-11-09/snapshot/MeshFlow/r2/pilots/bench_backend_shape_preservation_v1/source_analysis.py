"""Holdout same-instance recovery and source structure retention, CPU only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys, time
import numpy as np
from .common import *
sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_backend_counterfactual_v1 import core as B

def area(a):return .5*np.linalg.norm(np.cross(a[:,1].astype(float)-a[:,0],a[:,2].astype(float)-a[:,0]),axis=-1)
def geomstats(tri,k,source,sourceids=None):
 aa=area(tri); origarea=area(source)
 ids=np.arange(len(tri)) if sourceids is None else sourceids
 proxy=(source[:, :,1].min(1)<source[:k,:,1].min()-.02*L)&(source.mean(1)[:,1]<source[:k,:,1].min())
 proxy[:k]=False
 pick=proxy[ids];free=np.arange(len(tri))>=k
 den=float(origarea[proxy].sum())
 return dict(free_faces=int(free.sum()),free_area=float(aa[free].sum()),below_C_proxy_source_faces=int(proxy.sum()),below_C_proxy_area=float(aa[pick].sum()),below_C_proxy_source_area=den,below_C_proxy_area_ratio=float(aa[pick].sum()/den) if den else None,free_y_span=float(np.ptp(tri[k:,:,1])) if len(tri)>k else 0.)

def run():
 done=OUT/'source_analysis/COMPLETE.json'
 if done.exists():r=read(done);verify(r['files']);return r
 m=read(OUT/'data/manifest.json');old=read(OLD/'manifest.json');byparent={t['parent_id']:t for t in old['tasks']}
 P,geo,io,_=B.dependencies();scores=[];stages=[];plots=[];geo_count=0
 for row in m['validation_rows']:
  a=arrays(row['path']);src=a['target_model'];x=a['x_model'];mask=a['loss_mask'].astype(bool);k=int(a['known_faces']);rid=row['row_id']
  identity=np.array_equal(src,x)
  for label in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
   p=OUT/'evaluation'/label/rid/'proposal.npz';yp=arrays(p)['official_unconstrained_proposal']
   restored=yp.copy();restored[:k]=src[:k]
   err=np.sum((yp.astype(float)-src)**2,axis=-1)
   before=np.sum((x.astype(float)-src)**2,axis=-1)
   scores.append(dict(row_id=rid,parent_id=row['parent_id'],kind=row['kind'],label=label,masked_corners=int(mask.sum()),masked_model_MSE=float(err[mask].mean()/3),masked_denoiser_MSE=float(err[mask].mean()/3*SCALE**2),input_masked_model_MSE=float(before[mask].mean()/3),C_restored_exact=np.array_equal(restored[:k],src[:k]),free_proposal_restoration_exact=np.array_equal(restored[k:],yp[k:]),**geomstats(restored,k,src)))
   if not identity:continue
   t=byparent[row['parent_id']];raw=arrays(t['original']['raw_path'])
   assert int(raw['known_faces'])==k and int(raw['input_faces'])==len(src)
   assert not raw['raw_seam_slot_mask'].any() and (raw['raw_anchor_ids'][k:]<0).all()
   raw['raw']=src.copy();raw['output']=src.copy()
   if not np.array_equal(src[:k],raw['C_vertices'][raw['C_faces_model']]):raise ValueError('Holdout sourceC differs')
   source=arrays(t['source_condition_path']);dest=OUT/'source_analysis'/row['parent_id']/label
   receipt=dest/'complete.json'
   if receipt.exists():
    rr=read(receipt);verify(rr['files']);post=arrays(dest/'post.npz')
   else:
    if dest.exists():raise RuntimeError('Unresolved source geometry attempt '+str(dest))
    dest.mkdir(parents=True);write(dest/'attempt.json',dict(proposal_sha256=sha(p),model_calls=0,GEO_calls=1))
    pre,can,post,g,ori,anchors=P.geometry_from_proposal(raw,yp,source);geo_count+=1
    npz(dest/'post.npz',post);npz(dest/'post_canonical.npz',can);npz(dest/'pre_geo.npz',dict(output=pre))
    write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']));write(dest/'anchors.json',anchors)
    write(receipt,dict(status='COMPLETE',GEO_calls=1,files={str(p):sha(p) for p in dest.iterdir() if p.is_file()}))
   for stage,tri,ids in [('SOURCE',src,None),('PROPOSAL',yp,None),('RESTORED_C',restored,None),('GEO',post['output'],post['source_face_ids'])]:
    stages.append(dict(parent_id=row['parent_id'],row_id=rid,label=label,stage=stage,**geomstats(tri,k,src,ids)))
  if identity:plots.append(dict(row=row,task=byparent[row['parent_id']]))
 csvwrite(OUT/'source_analysis/per_instance.csv',scores);csvwrite(OUT/'source_analysis/per_stage.csv',stages)
 render(plots)
 result=dict(status='COMPLETE',source_rows=len(scores),identity_parents=len(plots),new_GEO_calls=geo_count,total_GEO=2*len(plots),additional_model_calls=0,limitation='Below-C proxy is geometric, not a semantic support annotation; triangle areas are summed. Known source defects masked in MSE, source diagnostic not open-generation style matching.',files={str(p):sha(p) for p in (OUT/'source_analysis').rglob('*') if p.is_file()})
 write(done,result);return result

def render(plots):
 from r2.pilots.chair_multipart_seam_v1 import gallery as G
 import html
 H,D=G.H,G.D;plt=H._mpl();body='<meta charset="utf-8"><h1>Held-out source identity: fixed input versus D0 and adapted D0</h1>'
 for e in plots:
  row,t=e['row'],e['task'];p=OUT/'source_analysis'/row['parent_id'];a=arrays(row['path']);src=a['target_model'];k=int(a['known_faces']);rid=row['row_id'];q=t['request']
  specs=[('INPUT',src)]
  for lab in ('OFFICIAL_SOURCE','CANDIDATE_SOURCE'):
   specs.extend([(lab+'_PROPOSAL',arrays(OUT/'evaluation'/lab/rid/'proposal.npz')['official_unconstrained_proposal']),(lab+'_GEO',arrays(p/lab/'post.npz')['output'])])
  fig=plt.figure(figsize=(24,12),dpi=120);grid=fig.add_gridspec(4,5,wspace=.15,hspace=.25)
  for col,(name,tri) in enumerate(specs):
   role=np.asarray(['C']*k+['BULK']*(len(tri)-k));g=dict(C=tri[:k],F=tri[k:],C_original=src[:k],tri=tri)
   for r,az in enumerate((-60,120)):
    raster,_=G.role_raster(tri,role,k,D.basis(az,23),np.asarray(q['gallery_depth_bounds']['center']),q['gallery_depth_bounds']['span']);ax=fig.add_subplot(grid[r,col]);ax.imshow(raster);ax.axis('off');ax.set_title(name,fontsize=9)
   for r,v in ((2,'front'),(3,'side')):
    ax=fig.add_subplot(grid[r,col]);H.draw_section(ax,g,[],q['gallery_bounds'],L,v,1);ax.set_title(name+' '+v,fontsize=8)
  fig.suptitle(row['parent_id']+' | source same-instance identity; proposal C allowed to move only for diagnostic')
  image=p/'identity.png';fig.savefig(image);plt.close(fig)
  body+='<h2>'+html.escape(row['parent_id'])+'</h2><a href="'+row['parent_id']+'/identity.png"><img width="100%" src="'+row['parent_id']+'/identity.png"></a>'
 (OUT/'source_analysis/index.html').write_text(body,encoding='utf-8')

if __name__=='__main__':print(run()['status'])
