"""One fixed five-model120-output matrix, original unchanged clamped sampler."""
from .common import *
from .checkpoint import load_endpoint
from .evaluation import load_bank
from ..loss_screen.sampling import _ledger,_noises,_sample_one as original_sample_one
from ..loss_screen.training import source_guard
import traceback

def sample_one(r,out,*args,reference=None):
 c=_ledger(r)
 if reference is None and c['formal_output_attempts']>=120:raise RuntimeError('Pilot120 formal cap')
 if reference is not None and c['replay_output_attempts']>=3:raise RuntimeError('Pilot3 replay cap')
 return original_sample_one(r,out,*args,reference=reference)

def sample_all(r,out):
 source_guard(r)
 if r.get('sampling_phases') or r['generated']:raise RuntimeError('No automatic repeated generation')
 if any(r['arms'][a]['status']!='PASS' or r['arms'][a]['completed_updates']!=500 for a in ARMS):raise RuntimeError('Both independently scheduled branches must finish before the fixed matrix')
 seeds=tuple(r['protocol']['evaluation']['seeds'])
 if tuple(r['protocol']['evaluation']['models'])!=MODELS or seeds!=SEEDS:raise RuntimeError('Fixed evaluation changed')
 banks=load_bank(r['assets']['evaluation_inputs']['path'],r['assets']['evaluation_inputs']['sha256'])
 noises=_noises(r,Path(out),'pilot',seeds,create=True);tick=time.perf_counter()
 block=dict(status='RUNNING',models=list(MODELS),seeds=list(seeds),expected_outputs=120,completed_outputs=0,models_audit={})
 r['sampling_phases']={'pilot':block};write(r,out)
 try:
  for name in MODELS:
   model=None;entry=dict(status='STARTING',completed_outputs=0);block['models_audit'][name]=entry
   try:
    model,entry['load']=load_endpoint(r,name)
    for patch in (0,1,2):
     for parent in (0,1):
      C=banks[patch]['cases_by_key'][parent,12]['constraints']
      for seed in seeds:
       sample_one(r,out,model,name,parent,patch,seed,'pilot',noises[seed],C)
       entry['completed_outputs']+=1;block['completed_outputs']+=1
    entry['state_after']=state_sha256(model)
    if entry['state_after']!=entry['load']['state_sha256']:raise RuntimeError('Frozen model changed in sampling')
    rows=[v for v in r['generated'] if v['model']==name]
    expected={(p,j,s) for p in (0,1) for j in (0,1,2) for s in seeds}
    if len(rows)!=24 or {(v['parent'],v['patch'],v['seed']) for v in rows}!=expected:raise RuntimeError('Sampling matrix incomplete')
    entry['status']='PASS'
   except Exception as exc:
    entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False)
   finally:
    if model is not None and 'state_after' not in entry:entry['state_after']=state_sha256(model)
    model=None;release();write(r,out)
  block['status']='PASS' if all(e['status']=='PASS' for e in block['models_audit'].values()) else 'FAILED'
  if block['status']!='PASS':raise RuntimeError('Fixed matrix contains numerical/identity failure, retained without retry')
 finally:
  block['seconds']=time.perf_counter()-tick;write(r,out)

def replay_once(r,out):
 if 'sampling_replay' in r or r['replays']:raise RuntimeError('One registered replay already attempted')
 seed=SEEDS[0];name='JEdge5';phase='pilot'
 matches=[x for x in r['generated'] if (x['model'],x['parent'],x['patch'],x['seed'])==(name,0,0,seed)]
 if len(matches)!=1:raise RuntimeError('Fixed JEdge5 replay reference missing')
 banks=load_bank(r['assets']['evaluation_inputs']['path'],r['assets']['evaluation_inputs']['sha256'])
 noise=_noises(r,Path(out),phase,SEEDS,create=False)[seed]
 entry=dict(status='STARTING',model=name,parent=0,patch=0,seed=seed,selection_by_result=False);r['sampling_replay']=entry
 tick=time.perf_counter();model=None
 try:
  model,entry['load']=load_endpoint(r,name)
  row=sample_one(r,out,model,name,0,0,seed,phase,noise,banks[0]['cases_by_key'][0,12]['constraints'],reference=matches[0])
  after=state_sha256(model)
  if after!=entry['load']['state_sha256']:raise RuntimeError('Reload replay changed model')
  entry.update(status='PASS',all51_bitwise_equal=True,state_after=after,row=row)
 except Exception as exc:
  entry.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  entry['seconds']=time.perf_counter()-tick;model=None;release();write(r,out)
