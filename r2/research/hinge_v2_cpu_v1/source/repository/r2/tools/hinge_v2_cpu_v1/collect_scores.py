"""Full inherited metrics plus explicit BRIDGE/original-parent accounting."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import concurrent.futures
import multiprocessing
import resource
import time
from pathlib import Path
import numpy as np
from . import run as S,scoring as M
C,B,OUT=S.C,S.B,S.OUT/'scores'
STAGES=('BASELINE',*S.ARMS)

def work(q):
    results=[];fields=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    for arm in STAGES:
        dest=OUT/arm/q['request_id'];done=dest/'complete.json'
        if done.exists():rr=C.read(done);B.verify(rr['files']);results.append(rr['row']);continue
        src=S.basepath(q) if arm=='BASELINE' else S.OUT/'cells'/arm/q['request_id']
        mesh=C.arrays(src/'post.npz');bridge=mesh.get('face_role',np.asarray([]))=='BRIDGE'
        existing=None if np.any(bridge) else C.read(src/'pairs.json')
        tick=time.perf_counter();r=M.score(q,mesh,dest,arm,existing_pairs=existing)
        if arm=='BASELINE':
            old=C.read(S.basepath(q)/'complete.json')['row'];checks=[]
            for key in fields:
                x,y=old[key],r['row'][key]
                equal=(x is None and y is None) or (x is not None and y is not None and np.isclose(x,y,atol=1e-12,rtol=1e-12,equal_nan=True))
                checks.append(dict(metric=key,old=x,new=y,pass_check=bool(equal)))
            C.write(OUT/'equivalence'/f"{q['request_id']}.json",dict(status='PASS' if all(x['pass_check'] for x in checks) else 'FAIL',checks=checks))
            assert all(x['pass_check'] for x in checks),[x for x in checks if not x['pass_check']]
        C.write(OUT/'timings'/arm/f"{q['request_id']}.json",dict(seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            pair_source=r['score']['pair_origin']))
        results.append(r['row'])
    print(q['request_id']+' SCORED 5',flush=True);return results

def run():
    reg=S.freeze();done=OUT/'completion.json'
    if done.exists():r=C.read(done);B.verify(r['files']);B.verify(r['code']);return r
    code={str(Path(z.__file__)):C.sha(z.__file__) for z in (M,)};code[str(Path(__file__))]=C.sha(__file__);code.update(S.Q.definition_hashes())
    protocol=OUT/'protocol.json'
    if protocol.exists():B.verify(C.read(protocol)['code'])
    else:C.write(protocol,dict(utc=C.utc(),status='FROZEN',stages=STAGES,rows=140,code=code,
        no_bridge='Reuse current complete pair cache, independently reevaluate other metrics',bridge='Full original finite-face scan; independently compare incremental trial cache',
        fixed_parent='Original HINGE_OPP015 free parent union, no new face in denominator',original_metrics=135))
    C.resources(OUT/'resources_before.json');S.event('SCORING_START');tick=time.perf_counter();first=work(reg['requests'][0]);snap=C.resources(OUT/'resources_after_first.json')
    mem=int(next(x.split()[1] for x in snap['memory'].splitlines() if x.startswith('MemAvailable:')))*1024
    peaks=[C.read(OUT/'timings'/arm/f"{reg['requests'][0]['request_id']}.json")['peak_RSS_KiB']*1024 for arm in STAGES]
    peak=max(peaks);workers=max(1,min(4,int(max(0,mem-(4<<30))/(peak*1.2)),int(max(1,snap['cpu_logical']-snap['loadavg'][0]-3))))
    C.write(OUT/'dispatch.json',dict(workers=workers,threads_per_worker=1,observed_first_request_peak_RSS_bytes=peak,available_RAM_bytes=mem,GPU_workers=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for rows in pool.map(work,reg['requests'][1:]):pass
    allrows=[];files={}
    for q in reg['requests']:
        eq=OUT/'equivalence'/f"{q['request_id']}.json";assert C.read(eq)['status']=='PASS';files[str(eq)]=C.sha(eq)
        for arm in STAGES:
            p=OUT/arm/q['request_id']/'complete.json';rec=C.read(p);B.verify(rec['files']);files.update(rec['files']);files[str(p)]=C.sha(p);allrows.append(rec['row'])
    C.csvwrite(OUT/'per_output.csv',allrows);files[str(OUT/'per_output.csv')]=C.sha(OUT/'per_output.csv')
    B.verify(code);B.verify(reg['protected']);C.resources(OUT/'resources_after.json')
    rec=dict(status='COMPLETE',rows=len(allrows),baseline_equivalence_checks=28*135,seconds=time.perf_counter()-tick,workers=workers,model_calls=0,new_queries=0,code=code,files=files)
    C.write(done,rec);S.event('SCORING_COMPLETE',rows=len(allrows),seconds=rec['seconds']);print('SCORE_COMPLETE',len(allrows),flush=True);return rec

if __name__=='__main__':run()
