"""Resumable fixed-panel CPU entry. No training or model-forward path exists."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, sys, time
import numpy as np
from . import common as C

def geometry(reg):
 from .blend import coordinate_restore
 free_rows=[];anchor_rows=[];receipts=[]
 for r in reg['records']:
  dest=C.OUT/'pre_geo/BLEND'/r['request_id'];out=dest/'denoised.npz';rp=dest/'receipt.json'
  ident=dict(request_id=r['request_id'],method='BLEND',raw_sha256=r['raw_sha256'],registration_sha256=C.sha(C.OUT/'registration.json'))
  if rp.exists():
   q=C.read(rp);assert q['identity']==ident and q['status']=='PASS';C.verify(q['files'])
  else:
   if dest.exists():raise RuntimeError('Partial BLEND output; no automatic replay')
   raw=C.payload(r['raw_path']);source=C.payload(r['condition_source']);d0=C.payload(r['D0_denoised_path'])
   c=source['C'];proposal=d0['official_unconstrained_proposal'];assert C.same(raw['C'],c)
   tick=time.perf_counter();C.event('BLEND_STARTED',**ident)
   output,diag,details=coordinate_restore(c,proposal,source['C_source_vertex_ids'],L=reg['fixed_L'])
   C.save_npz(out,output=output,C=c,official_unconstrained_proposal=proposal)
   C.save_npz(dest/'details.npz',**details);C.write(dest/'diagnostics.json',diag)
   q=dict(status='PASS',identity=ident,seconds=time.perf_counter()-tick,output_sha256=C.sha(out),
    files={str(p):C.sha(p) for p in (out,dest/'details.npz',dest/'diagnostics.json')})
   C.write(rp,q);C.event('BLEND_COMPLETE',**ident,seconds=q['seconds'])
  receipts.append(q);d=C.payload(dest/'details.npz');m={k:r[k] for k in ('request_id','parent_id','region_type','candidate')}
  for i,x in enumerate(d['free_actual_displacement_FP32']):
   free_rows.append(dict(**m,free_face=i//3,corner=i%3,support_count=int(d['free_support_count'][i]),
    support_sum=d['free_support_sum'][i],alpha_sum=d['free_alpha_sum'][i],participating_b_max=d['free_participating_b_max_norm'][i],
    bound=d['free_actual_bound'][i],unclipped_norm=np.linalg.norm(d['free_displacement_unclipped'][i]),
    actual_norm=np.linalg.norm(x),dx=x[0],dy=x[1],dz=x[2],safety_cap_active=bool(d['free_safety_cap_active'][i]),
    rounding_adjusted=bool(d['free_FP32_cap_rounding_adjusted'][i])))
  for i,vid in enumerate(d['anchor_source_vertex_ids']):
   b=d['anchor_displacement_target'][i]
   anchor_rows.append(dict(**m,C_face=i//3,corner=i%3,source_vertex_id=int(vid),weight=d['anchor_weight'][i],
    bx=b[0],by=b[1],bz=b[2],b_norm=np.linalg.norm(b)))
 C.csvwrite(C.OUT/'blend_per_free_corner.csv',free_rows);C.csvwrite(C.OUT/'blend_per_anchor_corner.csv',anchor_rows)
 C.write(C.OUT/'geometry_completion.json',dict(status='PASS',outputs=12,receipts=receipts,model_calls=0))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','orientation','blend','score','gallery','all'],default='all');a=p.parse_args()
 for name in ('orientation_checks','blend_checks'):
  receipt=C.OUT/name/'completion.json'
  if not receipt.exists() or C.read(receipt)['status']!='PASS':raise RuntimeError('Required CPU checks missing: '+name)
 reg=C.prepare()
 from . import evaluation as E
 tick=time.perf_counter()
 if a.stage in ('all','orientation'):E.orient_baselines(reg)
 if a.stage in ('all','blend'):geometry(reg)
 if a.stage in ('all','score'):
  rows=E.postprocess(reg);E.score(reg,rows);E.aggregate(rows)
 if a.stage in ('all','gallery'):E.render(C.read(C.OUT/'post_index.json'))
 if a.stage=='all':
  C.verify(reg['protected']);C.verify(reg['runner_hashes'])
  assert 'torch' not in sys.modules
  for kind,expected in (('ORIENTATION',36),('GEO',12),('QUALITY',12)):
   for suffix in ('started','complete'):
    assert len(list((C.OUT/'events').glob('*/*/'+kind+'_'+suffix+'.json')))==expected
  C.write(C.OUT/'execution_completion.json',dict(status='PASS',phase_seconds=time.perf_counter()-tick,
   new_model_forwards=0,training_updates=0,new_generator_outputs=0,new_noise=0,
   new_blend_outputs=12,new_GEO_C=12,new_POST_quality=12,orientation_outputs=36,
   original_files_unchanged=len(reg['protected']),frozen_sources_unchanged=True,torch_imported=False))
 C.event('STAGE_COMPLETE',stage=a.stage,seconds=time.perf_counter()-tick)
 print('STAGE_COMPLETE',a.stage,flush=True)

if __name__=='__main__':main()
