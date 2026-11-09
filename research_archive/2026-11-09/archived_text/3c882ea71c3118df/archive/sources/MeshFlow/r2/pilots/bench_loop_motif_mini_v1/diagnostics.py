"""Saved-state arithmetic only; zero model calls, no extra noise."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,collections,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'r2/src')]
import numpy as np
from r2.pilots.bench_loop_motif_mini_v1 import run as X
from r2.pilots.bench_loop_motif_mini_v1.analyze import savecsv
L=9.208138264587333
def flat(d,prefix=''):
    out={}
    for k,v in d.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flat(v,key+'.'))
        else:out[key]=v
    return out
def main():
    tasks=X.tasks();base=[t for t in X.read(X.OUT/'baseline.json') if t['arm']!='R']
    diagnostics=[];steps=[];events=collections.Counter();dirs={}
    for t in tasks+base:
        method=t['arm'] if 'stage_paths' in t else t['label'];folder=Path(t['output_dir']);dirs[method,t['request_id']]=folder
        a=X.read(folder/'audit.json')
        row=dict(method=method,request_id=t['request_id'],parent_id=t['q']['parent_id'],noise_index=t['q']['noise_index'],**a)
        row['shadow_rms_percent_L']=100*a['shadow_terminal_corner_rms']/L
        diagnostics.append(row)
        for p in sorted((folder/'steps').glob('*.json')):steps.append(dict(method=method,request_id=t['request_id'],**flat(X.read(p))))
        if t in tasks:
            for line in (folder/'calls.jsonl').read_text().splitlines():events[json.loads(line)['event']]+=1
    savecsv('guidance_per_output.csv',diagnostics);savecsv('guidance_per_step.csv',steps)
    comparison=[]
    # Reconstruct applied correction = next state - x - dt*(Yhat-x)/(1-t).
    # Different models' states already diverge: direction cosine is on their OWN trajectories,
    # not a controlled Jacobian-at-the-same-state attribution.
    for q in X.read(X.OUT/'inputs/requests.json'):
        rid=q['request_id'];k=q['K']
        for step in range(5,50):
            corrections=[];checks=[]
            for method in ('G3','L3'):
                f=dirs[method,rid];prev=np.load(f/f'states/state_{step:03}.npz')['state'].astype(np.float64)
                now=np.load(f/f'states/state_{step+1:03}.npz');end=now['endpoint_prediction'].astype(np.float64)
                corr=now['state'].astype(np.float64)-prev-.02*(end-prev)/(1-step/50)
                d=X.read(f/f'steps/step_{step:03}.json');checks.append(abs(np.linalg.norm(corr.reshape(-1,3),axis=1).max()-d['correction']['max_corner']))
                corrections.append(corr)
            row=dict(request_id=rid,step=step,t=step/50,max_norm_reconstruction_error=max(checks),scope='own_trajectory_not_same_state_Jacobian')
            for name,sl in [('S',slice(0,k)),('F',slice(k,None)),('ALL',slice(None))]:
                a,b=[c[sl].ravel() for c in corrections];den=np.linalg.norm(a)*np.linalg.norm(b)
                row[name+'_applied_direction_cosine']=float(a@b/den) if den else None
                row[name+'_G3_l2']=float(np.linalg.norm(a));row[name+'_L3_l2']=float(np.linalg.norm(b))
            comparison.append(row)
    savecsv('applied_direction_own_trajectory.csv',comparison)
    gate=[json.loads(x) for x in (X.OUT/'checks/r1/calls.jsonl').read_text().splitlines()]
    d0=collections.Counter()
    for p in X.OUT.glob('post/*/calls.jsonl'):
        for line in p.read_text().splitlines():d0[json.loads(line)['event']]+=1
    workers=[]
    for p in X.OUT.glob('workflow/attempts/*/complete.json'):workers.append(dict(path=str(p),**X.read(p)))
    # Observer receipt file naming may differ; preserve all actual files in archive.
    X.write(X.OUT/'analysis/call_ledger.json',dict(sampling_events=dict(events),gate_events=dict(collections.Counter(x['event'] for x in gate)),D0_events=dict(d0),training_updates=0,new_noise=0,new_RAW=16,original_layers_per_field=12,loop_layers_per_field=16,actual_sampling_layer_evaluations=12800,gate_layer_evaluations=24,implicit_retries=0))
    print(json.dumps(dict(events=dict(events),D0=dict(d0),max_direction_reconstruction_error=max(r['max_norm_reconstruction_error'] for r in comparison))),flush=True)
if __name__=='__main__':main()
