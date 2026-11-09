"""Spawn-safe CPU dispatch for the frozen, unchanged multipart scorer core."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time
from . import evaluate as E

V,C,OUT,STAGES=E.V,E.C,E.OUT,E.STAGES


def configure():
    # This runs inside every spawned task, not only in its parent process.
    E.configure()


def score_one(job):
    configure()
    return V.score_one(job)


def interface_probe(job):
    """Actual spawned interpreter path resolution; zero geometry scoring."""
    configure();root,q,stage=job
    assert 'torch' not in sys.modules
    return dict(label=V.LABEL,root=str(V.ROOT),stages=list(V.STAGES),
                mesh_path=str(V.mesh_path(root,q,stage)),score_path=str(V.score_dir(root,q,stage)),
                request_count=len(V.requests(root)),torch_imported=False,model_forwards=0,geometry_scans=0)


def registration():
    configure();path=OUT/'evaluation/dispatcher_registration.json'
    score_reg=OUT/'evaluation/score_registration.json'
    core=E.prepare(OUT)
    sources=dict(core['source_hashes'])
    sources[str(Path(E.__file__))]=C.sha(E.__file__)
    sources[str(Path(V.__file__))]=C.sha(V.__file__)
    value=dict(status='FROZEN',schema='multipart_spawn_dispatch_v1',
               dispatcher_path=str(Path(__file__)),dispatcher_sha256=C.sha(__file__),
               score_registration_path=str(score_reg),score_registration_sha256=C.sha(score_reg),
               frozen_core_hashes=sources,change='Each spawned task configures experiment label/root before invoking unchanged V.score_one',
               scientific_changes=False,model_calls=0)
    if path.exists():
        if C.read(path)!=value:raise ValueError('Dispatcher or frozen score core identity changed')
    else:C.write(path,value)
    return value,core


def check():
    reg,core=registration();q=core['requests'][0]
    with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        result=pool.submit(interface_probe,(str(OUT),q,'RAW')).result()
    expected=OUT/'evaluation'/E.LABEL/q['request_id']/'raw/raw.npz'
    assert Path(result['mesh_path'])==expected and result['label']==E.LABEL and result['request_count']==44
    out=OUT/'evaluation/dispatcher_cpu_check.json'
    value=dict(status='PASS',production_worker_initialization_used=True,spawned_process=result,
               dispatcher_registration_sha256=C.sha(OUT/'evaluation/dispatcher_registration.json'),
               geometry_scans=0,model_calls=0)
    if out.exists():
        if C.read(out)!=value:raise ValueError('Existing dispatcher CPU check differs')
    else:C.write(out,value)
    return value


def run(workers=2):
    dispatch,reg=registration();qq=reg['requests'];root=OUT
    for q in qq:
        for stage in STAGES:V.artifact(root,q,stage)
    done=root/'evaluation/completions'/('__'.join(STAGES)+'.json')
    if done.exists():
        saved=C.read(done);V.E.verify(saved['files']);return saved
    identity=C.sha(root/'evaluation/score_registration.json')
    jobs=[(str(root),q,s,identity) for q in qq for s in STAGES]
    before=root/'evaluation/score_dispatch'/('__'.join(STAGES)+'_resources_before.json')
    if not before.exists():C.write(before,C.resources())
    tick=time.perf_counter();first=score_one(jobs[0]);snapshot=C.resources()
    count=V.worker_count(snapshot,first,workers)
    runpath=root/'evaluation/score_dispatch'/('__'.join(STAGES)+'.json')
    C.write(runpath,dict(workers=count,threads_per_worker=1,resources=snapshot,first_authorized_score=first['identity'],
                        model_calls=0,dispatcher_registration_sha256=C.sha(root/'evaluation/dispatcher_registration.json')),
            replace=runpath.exists())
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*list(pool.map(score_one,jobs[1:]))]
    V.E.verify(reg['protected']);V.E.verify(reg['source_hashes']);V.E.verify(dispatch['frozen_core_hashes'])
    if C.sha(__file__)!=dispatch['dispatcher_sha256']:raise ValueError('Dispatcher changed during scoring')
    result=dict(status='COMPLETE',stages=list(STAGES),rows=len(receipts),workers=count,seconds=time.perf_counter()-tick,
        files={str(V.score_dir(root,q,s)/'audit_receipt.json'):C.sha(V.score_dir(root,q,s)/'audit_receipt.json') for q in qq for s in STAGES},
        dispatcher_registration_sha256=C.sha(root/'evaluation/dispatcher_registration.json'),model_calls=0)
    C.write(done,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-cpu',action='store_true');parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args=parser.parse_args();result=check() if args.check_cpu else run(args.workers)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
