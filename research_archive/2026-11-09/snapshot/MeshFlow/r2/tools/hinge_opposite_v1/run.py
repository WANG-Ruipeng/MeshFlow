"""Finite, resumable CPU replay of 28 S2000 requests at two fixed HINGE caps.

No training, model imports, new noise, D0 calls, or modifications to old assets.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_hinge_quality_v1 import hinge as OLD
from r2.pilots.seam_hinge_quality_v1 import topology as T
from . import engine as H

C,B,Q=OLD.C,OLD.B,OLD.Q
OUT=C.ROOT/'experiments/r2_hinge_opposite_v1'
DELTAS=(.015,.025)
VERSION='S2000_HINGE_all_induced_edges_opposite_v2'

def name(delta):return 'HINGE_OPP'+str(round(delta*1000)).zfill(3)

def event(kind,**kwargs):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ledger.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(C.clean(dict(utc=C.utc(),event=kind,**kwargs)),ensure_ascii=False)+'\n')

def freeze():
    p=OUT/'protocol.json'
    if p.exists():
        reg=C.read(p);B.verify(reg['protected']);B.verify(reg['code']);return reg
    qq=OLD.requests()
    assert len(qq)==28 and sum(q['split']=='TRAIN' for q in qq)==16
    protected={}
    for q in qq:
        for path in OLD.paths('PLAIN_S2000',q).values():protected[str(path)]=C.sha(path)
        for key in ('inference_npz','metadata_npz'):protected[q[key]]=C.sha(q[key])
        desc=q['query_descriptor'];protected[desc['path']]=desc['sha256']
        for d in DELTAS:
            old=OLD.OUT/'cells'/OLD.label('PLAIN_S2000',d)/q['request_id']
            rec=C.read(old/'complete.json');B.verify(rec['files'])
            protected.update(rec['files']);protected[str(old/'complete.json')]=C.sha(old/'complete.json')
    for path in (OLD.OLD/'protocol.json',OLD.NEW/'results/final/per_output.csv',OLD.NEW/'REPORT.md'):
        protected[str(path)]=C.sha(path)
    code={str(Path(x.__file__)):C.sha(x.__file__) for x in (H,OLD,B,T,OLD.PE)}
    code.update(Q.definition_hashes());code[str(Path(__file__))]=C.sha(__file__)
    reg=dict(status='FROZEN',utc=C.utc(),version=VERSION,requests=qq,deltas=DELTAS,cells=56,
             generator='Frozen PLAIN_S2000 saved POST; old RAW and single D0 reused',
             comparison='Baseline POST, historical HINGE015/025, new opposite-guard HINGE015/025',
             scientific_change='Reject whole-cluster trial whenever any newly induced two-face edge is same-directed, including collateral CF/FF edges; preserve old same-directed edges; do not reorient',
             require_opposite_winding=True,other_guards='All previous HINGE guards, ordering, candidates, shape thresholds and cumulative PRE_GEO caps unchanged',
             metrics='Original124 plus original11 full-mesh topology diagnostics; I/q costs remain reported, no added quality veto',
             main_delta=.015,stress_delta=.025,model_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
             closest_prior='Original HINGE audit: nominated hinge opposite already, but 3/4 collateral same-directed edges passed. R05/R10/BOUND guard costs retained.',
             positive_criterion='Bounded improvement in actual connectivity and FF with original legal edges preserved; not universal all-metric dominance',
             protected=protected,code=code)
    C.write(p,reg);event('FROZEN',cells=56,protected=len(protected),protocol_sha256=C.sha(p));return reg

def worker(q):
    p=OLD.paths('PLAIN_S2000',q)
    base=C.arrays(p['post']);pre=C.arrays(p['pre'])['output'];raw=C.arrays(p['raw'])['output'];bp=C.read(p['pairs'])
    OLD.validate_arrays(base,pre,q);results=[]
    for d in DELTAS:
        label=name(d);dest=OUT/'cells'/label/q['request_id'];done=dest/'complete.json'
        if done.exists():
            rr=C.read(done);B.verify(rr['files']);results.append(rr);continue
        if dest.exists():raise RuntimeError('Incomplete attempt preserved, no silent retry: '+str(dest))
        tick=time.perf_counter();C.write(dest/'started.json',dict(utc=C.utc(),request_id=q['request_id'],delta=d,pid=os.getpid()))
        try:
            out,records,summ,bl,nl,_=H.decode(base,pre,raw,bp,d,require_opposite_winding=True)
            B.save_npz(dest/'post.npz',**out)
            C.write(dest/'hinge.json',summ);C.write(dest/'candidate_records.json',records);C.csvwrite(dest/'candidates.csv',records)
            score,row,pairs,shape,edges,vectors,face_rows,roles=B.score_output(out,q,label)
            C.write(dest/'score.json',score);C.write(dest/'pairs.json',pairs);C.write(dest/'shape_per_face.json',shape)
            C.csvwrite(dest/'valid_edges.csv',edges);C.csvwrite(dest/'seam_bulk_faces.csv',face_rows);C.csvwrite(dest/'illegal_pairs_by_role.csv',roles);B.save_npz(dest/'vectors.npz',**vectors)
            nl=Q.connectivity_ledger(out['vertices'],out['faces'],int(q['K']),pairs,out['source_face_ids'],out['source_corner_ids'])
            paired=Q.compare_ledgers(bl,nl,bp,pairs,summ['affected_source_raw_slots'])
            C.write(dest/'connectivity.json',nl);C.write(dest/'paired_ledger.json',paired)
            for key in ('baseline_edge_fates','new_edges','illegal_pair_fates'):C.csvwrite(dest/(key+'.csv'),paired[key])
            for key in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                assert paired['summary'][key]==0,(label,q['request_id'],key)
            diag,faces=T.measure(out,int(q['K']),edges,C.L)
            C.csvwrite(dest/'per_face_connectivity.csv',faces)
            row.update(diag,label='PLAIN_S2000',stage=label,mesh_path=str(dest/'post.npz'),topology_definition_version=T.VERSION)
            rr=dict(status='COMPLETE',version=VERSION,row=row,summary=summ,pair_summary=paired['summary'],seconds=time.perf_counter()-tick,
                    peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,GPU_calls=0,
                    CPU_only='torch' not in sys.modules,files={str(z):C.sha(z) for z in dest.iterdir() if z.is_file()})
            assert rr['CPU_only'],'Unexpected torch import'
            C.write(done,rr);results.append(rr)
            print(json.dumps(dict(request=q['request_id'],delta=d,accepted=summ['accepted'],seconds=rr['seconds'])),flush=True)
        except BaseException as e:
            C.write(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    return results

def aggregate():
    reg=freeze();rows=[];summary=[];funnels=[]
    for q in reg['requests']:
        for d in DELTAS:
            p=OUT/'cells'/name(d)/q['request_id'];r=C.read(p/'complete.json');B.verify(r['files'])
            rows.append(r['row']);summary.append(dict(label=name(d),request_id=q['request_id'],split=q['split'],seconds=r['seconds'],**r['pair_summary']))
            funnels.append(dict(label=name(d),request_id=q['request_id'],split=q['split'],**r['summary']))
    C.csvwrite(OUT/'new_per_output.csv',rows);C.csvwrite(OUT/'connection_ledger.csv',summary);C.csvwrite(OUT/'rejection_funnel.csv',funnels)
    return rows

def run(max_workers):
    reg=freeze();done=OUT/'execution_complete.json'
    if done.exists():
        r=C.read(done);B.verify(r['files']);return r
    tick=time.perf_counter();first=worker(reg['requests'][0]);snap=C.resources(OUT/'resources/after_first.json')
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in (snap.get('memory') or '').splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    peak=max(r['peak_RSS_KiB'] for r in first)*1024
    available=mem.get('MemAvailable',0);cpu=os.cpu_count() or 1;load=(snap.get('loadavg') or [cpu])[0]
    workers=min(max_workers,max(1,int((available-(4<<30))/max(1,1.2*peak))),max(1,int(cpu-load-2)))
    workers=max(1,workers)
    C.write(OUT/'dispatch.json',dict(workers=workers,threads_per_worker=1,first_authorized_peak_RSS_bytes=peak,available_RAM_bytes=available,cpu_logical=cpu,load=load,GPU_workers=0,
                                  reason='Independent fixed request CPU geometry/scoring; ample measured RAM/cores, max4 avoids audit oversubscription'))
    event('DISPATCH',workers=workers,first_authorized_cells=2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,q) for q in reg['requests'][1:]]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    rows=aggregate();B.verify(reg['protected']);B.verify(reg['code'])
    C.resources(OUT/'resources/after.json')
    result=dict(status='COMPLETE',cells=len(rows),workers=workers,seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0,
                new_endpoint_quality_scores=56,files={str(OUT/k):C.sha(OUT/k) for k in ('new_per_output.csv','connection_ledger.csv','rejection_funnel.csv')})
    C.write(done,result);event('CPU_MATRIX_COMPLETE',**{k:v for k,v in result.items() if k!='files'});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','aggregate'));p.add_argument('--workers',type=int,choices=(1,2,3,4),default=4);a=p.parse_args()
    r=freeze() if a.stage=='freeze' else run(a.workers) if a.stage=='run' else aggregate()
    print(json.dumps({k:v for k,v in r.items() if k not in ('protected','code','requests','files')} if isinstance(r,dict) else dict(rows=len(r))),flush=True)

if __name__=='__main__':main()
