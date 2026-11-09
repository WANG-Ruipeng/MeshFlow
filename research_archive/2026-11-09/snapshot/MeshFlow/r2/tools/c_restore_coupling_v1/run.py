"""Resumable pilot entry. No successful model call is replayed."""
import argparse,subprocess,sys,time
import common as C

def geometry():
 import numpy as np
 from geometry import coordinate_restore
 reg=C.prepare();regh=C.sha(C.OUT/'registration.json');rows=[]
 for r in reg['records']:
  rid=r['request_id'];dest=C.OUT/'pre_geo/RBF'/rid;out=dest/'denoised.npz';receipt=dest/'receipt.json'
  identity=dict(request_id=rid,method='RBF',raw_sha256=r['raw_sha256'],registration_sha256=regh)
  if receipt.exists():
   q=C.read(receipt)
   if q['status']!='PASS' or any(q[k]!=v for k,v in identity.items()) or q['output_sha256']!=C.sha(out):raise ValueError('RBF resume mismatch')
   rows.append(q);continue
  if dest.exists():raise RuntimeError('Partial RBF output exists; no silent replay')
  with np.load(r['raw_path'],allow_pickle=False) as a:c=a['C'].copy()
  with np.load(r['condition_source'],allow_pickle=False) as a:ids=a['C_source_vertex_ids'].copy()
  with np.load(r['D0_denoised_path'],allow_pickle=False) as a:proposal=a['official_unconstrained_proposal'].copy()
  tick=time.perf_counter();C.event('RBF_STARTED',**identity)
  output,diagnostics,details=coordinate_restore(c,proposal,ids,L=reg['fixed_L'])
  if output[:len(c)].tobytes()!=c.tobytes():raise ValueError('RBF C changed')
  C.save_npz(out,output=output,C=c,official_unconstrained_proposal=proposal)
  C.write(dest/'diagnostics.json',diagnostics)
  for name,value in details.items():
   if isinstance(value,list):C.csvwrite(dest/(name+'.csv'),value)
   elif isinstance(value,dict):C.write(dest/(name+'.json'),value)
   else:C.save_npz(dest/(name+'.npz'),data=np.asarray(value))
  proof=dict(status='PASS',**identity,output_sha256=C.sha(out),seconds=time.perf_counter()-tick,C_bitwise=True,diagnostics=diagnostics)
  C.write(receipt,proof);rows.append(proof);C.event('RBF_COMPLETE',**identity,seconds=proof['seconds'])
 C.write(C.OUT/'geometry_completion.json',dict(status='PASS',outputs=len(rows),new_D0_forwards=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','geometry','denoise','score','gallery','all'],default='all');a=p.parse_args()
 C.prepare()
 if a.stage in ('all','geometry'):
  if not (C.OUT/'geometry_checks/completion.json').exists():subprocess.run([sys.executable,'-B',str(C.CODE/'geometry_checks.py')],check=True)
  if C.read(C.OUT/'geometry_checks/completion.json')['status']!='PASS':raise ValueError('Geometry checks failed')
  geometry()
 if a.stage in ('all','denoise'):subprocess.run([sys.executable,'-B',str(C.CODE/'denoise.py')],check=True)
 if a.stage in ('all','score','gallery'):
  import evaluation
  if a.stage in ('all','score'):evaluation.score()
  if a.stage in ('all','gallery'):evaluation.render()
 print('STAGE_COMPLETE',a.stage)
