"""Registered finite CPU HINGE V2 test; no model, GPU, D0, RNG or training."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import concurrent.futures
import json
import multiprocessing
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from r2.tools.hinge_opposite_v1 import run as O

C,B,Q=O.C,O.B,O.Q
OUT=C.ROOT/'experiments/r2_hinge_v2_cpu_v1'
ARMS=('MIDPOINT_REPLAY','ALPHA','HOLE','ALPHA_HOLE')

def event(kind,**kw):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(C.clean(dict(utc=C.utc(),event=kind,**kw)),ensure_ascii=False)+'\n')

def basepath(q):return O.OUT/'cells/HINGE_OPP015'/q['request_id']

def freeze():
    from . import alpha,holes
    p=OUT/'protocol.json'
    if p.exists():
        reg=C.read(p);B.verify(reg['protected']);B.verify(reg['code']);return reg
    qq=O.OLD.requests();assert len(qq)==28 and sum(q['split']=='TRAIN' for q in qq)==16
    protected={}
    for q in qq:
        done=basepath(q)/'complete.json';rec=C.read(done);B.verify(rec['files']);protected.update(rec['files']);protected[str(done)]=C.sha(done)
        pp=O.OLD.paths('PLAIN_S2000',q)
        for key in ('pre','raw'):protected[str(pp[key])]=C.sha(pp[key])
        for key in ('inference_npz','metadata_npz'):protected[q[key]]=C.sha(q[key])
        de=q['query_descriptor'];protected[de['path']]=de['sha256']
    for p0 in (O.OUT/'protocol.json',O.OUT/'gallery/definition.json',C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json',C.ROOT/'experiments/r2_hinge_subsegment_v1/RUN_COMPLETE.json'):
        protected[str(p0)]=C.sha(p0)
    source=Path('/mnt/c/Users/warpwang/.codex/attachments/1d76f5d8-2351-489e-bc5e-9b3cd75af199/Pasted text.txt')
    dest=OUT/'user_attachment.txt';OUT.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(source,dest)
    assert C.sha(source)==C.sha(dest);protected[str(source)]=C.sha(source)
    code={str(Path(x.__file__)):C.sha(x.__file__) for x in (alpha,holes,O.H,O,B,O.OLD.PE)}
    code[str(Path(__file__))]=C.sha(__file__);code.update(Q.definition_hashes())
    reg=dict(status='FROZEN',utc=C.utc(),version='HINGE_V2_finite_alpha_and_exact_cycle_v1',requests=qq,
        baseline='Saved PLAIN_S2000 + HINGE_OPP015',arms=ARMS,cells=112,delta_fraction=.015,
        alpha_targets='Original midpoint first, then alpha 0,.25,.5,.75,1; FP32-deduplicated. C-containing cluster only original C target.',
        midpoint_control='One extra original-only pass, same guards. Separates rescan effect from extra target choices.',
        candidate_policy='Existing shared-index open-edge hinge pairs, endpoint gap <=1.5%L; static deterministic queue per arm, no normal offset or angle-gate change.',
        holes='Exact indexed 3-edge open cycles only, all incident faces free and >=1 SEAM. Opposite all three edges, min angle >=5, q<=10, no duplicate or new illegal/unknown pair. Append BRIDGE with no RAW/source-parent claim.',
        arms_definition={'MIDPOINT_REPLAY':'baseline -> one original midpoint pass','ALPHA':'baseline -> finite alpha pass','HOLE':'baseline -> exact triangle cycles','ALPHA_HOLE':'registered ALPHA result -> exact triangle cycles'},
        common_guards='C/seam anchors bitwise fixed; cumulative original PRE_GEO displacement <=1.5%L; zero new illegal CF/FF, unresolved pair, nonmanifold edge, same-winding two-face edge or deleted face; preserve all old legal edges; old shape guards.',
        metric_policy='Original 135 retained, plus original-parent fixed denominators and BRIDGE-only counts/area. Original bulk area excludes bridges, graph may traverse bridges. Full per-output/per-parent paired values.',
        visualization='Same frozen exterior/section cameras as previous HINGE; AI_PROXY actual review of changed outputs and DEV counterexamples.',
        closest_prior='R05 protected weld, R10 partner closure, retained HINGE opposite; subsegment V1 only one already-positive TRAIN repair. New: finite merge location choices and new true 3-cycle surface, not radius increase or seam split.',
        bounded_interpretation='593 subsegment pairs are not old HINGE candidates; holes are not inferred from failed angle tests. No physical upper bound or universal impossibility conclusion.',
        stop='Complete this fixed CPU matrix, full scoring, review and report; no training/default change/commit/push.',
        model_calls=0,GPU_calls=0,new_noise=0,new_queries=0,new_D0=0,protected=protected,code=code)
    C.write(p,reg);event('FROZEN',cells=112,arms=ARMS,protocol_sha256=C.sha(p));return reg

def save_cell(q,arm,out,pairs,records,summary,tick):
    p=OUT/'cells'/arm/q['request_id']
    B.save_npz(p/'post.npz',**out);C.write(p/'pairs.json',pairs);C.write(p/'records.json',records);C.write(p/'summary.json',summary)
    C.csvwrite(p/'candidates.csv',records)
    rr=dict(status='COMPLETE',request_id=q['request_id'],arm=arm,summary=summary,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CPU_only='torch' not in sys.modules,model_calls=0,
        files={str(z):C.sha(z) for z in p.iterdir() if z.is_file()})
    assert rr['CPU_only'];C.write(p/'complete.json',rr)
    print(json.dumps(dict(request=q['request_id'],arm=arm,accepted=summary.get('accepted'),seconds=rr['seconds'])),flush=True);return rr

def worker(q):
    from . import alpha,holes
    bp=basepath(q);base=C.arrays(bp/'post.npz');pairs=C.read(bp/'pairs.json');pp=O.OLD.paths('PLAIN_S2000',q)
    pre=C.arrays(pp['pre'])['output'];raw=C.arrays(pp['raw'])['output'];results=[]
    for arm in ARMS:
        p=OUT/'cells'/arm/q['request_id'];done=p/'complete.json'
        if done.exists():rr=C.read(done);B.verify(rr['files']);results.append(rr);continue
        if p.exists():raise RuntimeError('Partial attempt preserved; explicit engineering recovery required: '+str(p))
        tick=time.perf_counter();C.write(p/'started.json',dict(utc=C.utc(),request_id=q['request_id'],arm=arm,pid=os.getpid()))
        try:
            if arm in ('MIDPOINT_REPLAY','ALPHA'):
                out,records,summary,_,_,npairs=alpha.decode(base,pre,raw,pairs,.015,alphas=arm=='ALPHA')
            else:
                if arm=='HOLE':inp,ipairs=base,pairs
                else:
                    ap=OUT/'cells/ALPHA'/q['request_id'];inp,ipairs=C.arrays(ap/'post.npz'),C.read(ap/'pairs.json')
                out,npairs,_,records,summary=holes.decode(inp,ipairs)
            rr=save_cell(q,arm,out,npairs,records,summary,tick);results.append(rr)
        except BaseException as e:
            C.write(p/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    return results

def run(max_workers=4):
    reg=freeze();done=OUT/'geometry_complete.json'
    if done.exists():r=C.read(done);B.verify(r['files']);return r
    C.resources(OUT/'resources/before_geometry.json');event('GEOMETRY_START');tick=time.perf_counter()
    first=worker(reg['requests'][0]);snap=C.resources(OUT/'resources/after_first_request.json')
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in snap['memory'].splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    peak=max(x['peak_RSS_KiB'] for x in first)*1024;available=mem['MemAvailable'];cpu=snap['cpu_logical'];load=snap['loadavg'][0]
    workers=max(1,min(max_workers,int(max(0,available-(4<<30))/(1.2*peak)),int(max(1,cpu-load-3))))
    C.write(OUT/'dispatch.json',dict(workers=workers,threads_per_worker=1,first_request_peak_RSS_bytes=peak,available_RAM_bytes=available,
        cpu_logical=cpu,cpu_load=load,GPU_workers=0,reason='28 independent saved requests; 4-arm dependencies sequential within request. Start one real request then measure; max4 leaves room for independent audit.'))
    event('CPU_DISPATCH',workers=workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        fs=[pool.submit(worker,q) for q in reg['requests'][1:]]
        for f in concurrent.futures.as_completed(fs):f.result()
    rows=[];files={}
    for q in reg['requests']:
        for arm in ARMS:
            p=OUT/'cells'/arm/q['request_id']/'complete.json';r=C.read(p);B.verify(r['files']);files[str(p)]=C.sha(p);files.update(r['files'])
            rows.append(dict(request_id=q['request_id'],split=q['split'],arm=arm,seconds=r['seconds'],**r['summary']))
    C.csvwrite(OUT/'geometry_summary.csv',rows);files[str(OUT/'geometry_summary.csv')]=C.sha(OUT/'geometry_summary.csv')
    B.verify(reg['protected']);B.verify(reg['code']);C.resources(OUT/'resources/after_geometry.json')
    result=dict(status='COMPLETE',cells=len(rows),workers=workers,seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0,files=files)
    C.write(done,result);event('GEOMETRY_COMPLETE',cells=len(rows),seconds=result['seconds']);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run']);p.add_argument('--workers',type=int,choices=[1,2,3,4],default=4);a=p.parse_args()
    r=freeze() if a.stage=='freeze' else run(a.workers);print(json.dumps({k:v for k,v in r.items() if k not in ('protected','code','requests','files')}),flush=True)
