"""Fixed original metric definitions, full179-field rows, CPU-only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,multiprocessing,resource,sys,time
from pathlib import Path
import numpy as np
from .eval_prepare import OUT,OLD,LABELS,HISTORICAL,P,requests,verify
from r2.pilots.seam_abc_v1 import eval_common as C

def base_root(label):return OLD if label in HISTORICAL else OUT

def artifact(label,q,stage):
    if label not in (*HISTORICAL,*LABELS) or stage not in ('RAW','BASE_POST'):raise ValueError('Unregistered score')
    kind='raw' if stage=='RAW' else 'post'
    path=base_root(label)/'evaluation'/label/q['request_id']/kind/(kind+'.npz')
    receipt=path.parent/'receipt.json';r=P.read(receipt);verify(r.get('files',{}))
    assert r['status']=='COMPLETE' and r['request_id']==q['request_id']
    assert P.sha(path)==r[kind+'_sha256']
    assert r.get('generator_forwards')==50 if stage=='RAW' else (r['D0_forwards']==1 and r['GEO_calls']==1)
    return path,receipt

def prepare():
    from . import post_cpu
    qq=requests();dest=OUT/'evaluation/score_registration.json'
    sources={str(Path(__file__)):P.sha(__file__),str(Path(post_cpu.__file__)):P.sha(post_cpu.__file__)}
    from r2.pilots.chair_back_transfer_v1.evaluate import source_hashes
    sources.update(source_hashes())
    value=dict(status='FROZEN',version='bench_clean_original179_cpu_v1',requests=qq,
        stages=['RAW','BASE_POST'],new_labels=list(LABELS),historical_RAW_labels=list(HISTORICAL),
        source_hashes=sources,eval_manifest_sha256=P.sha(OUT/'eval_manifest.json'),
        metric_policy='Unchanged original finite-face detectors, original query arrays; closedC seam fields NA; Native seam-specific role fields NA; all135 original fields plusbridge/fixedparent179',
        source_style_success=False,model_calls=0,new_query_arrays=0)
    if dest.exists():
        assert P.read(dest)==value;verify(sources)
    else:P.immutable(dest,value)
    return value

def one(job):
    label,q,stage,regsha=job
    from .post_cpu import score_current
    dest=OUT/'evaluation/scores'/label/q['request_id']/stage
    mesh,producer=artifact(label,q,stage)
    identity=dict(label=label,request_id=q['request_id'],stage=stage,registration_sha256=regsha,
        mesh_path=str(mesh),mesh_sha256=P.sha(mesh),producer_sha256=P.sha(producer))
    done=dest/'audit_receipt.json'
    if done.exists():
        r=P.read(done);assert r['identity']==identity;verify(r['files']);return r
    if dest.exists():raise RuntimeError('Unresolved score attempt, preserved '+str(dest))
    tick=time.perf_counter();a=C.arrays(mesh)
    if 'index_domain' not in a:a['index_domain']=np.asarray('ACTUAL_RAW_C_SEAM_SHARED_INDEX')
    existing=None;source_pairs=None
    if label in HISTORICAL:
        source_pairs=OLD/'evaluation/scores'/label/q['request_id']/('RAW' if stage=='RAW' else 'POST')/'pairs.json'
        score=P.read(source_pairs.parent/'score.json');verify(score['files'])
        assert score['identity']['mesh_sha256']==P.sha(mesh)
        existing=P.read(source_pairs)
    score_current(q,a,dest,stage,label!='NATIVE_S21000',existing_pairs=existing)
    r=dict(status='PASS',identity=identity,new_queries=0,model_calls=0,
        full_pair_scan=existing is None,source_pairs=str(source_pairs) if source_pairs else None,
        seconds=time.perf_counter()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):P.sha(p) for p in dest.iterdir() if p.is_file()})
    assert 'torch' not in sys.modules
    P.immutable(done,r);print('SCORE',label,stage,q['request_id'],flush=True);return r

def run(labels,stages,max_workers=2):
    from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
    reg=prepare();qq=reg['requests']
    if any(label not in (*LABELS,*HISTORICAL) for label in labels):raise ValueError('Bad label')
    if any(label in HISTORICAL for label in labels) and stages!=['RAW']:raise ValueError('Historical BASE owned by post_cpu')
    key='__'.join(labels)+'__'+'__'.join(stages)
    done=OUT/'evaluation/completions'/(key+'.json')
    if done.exists():
        r=P.read(done);verify(r['files']);return r
    jobs=[(label,q,stage,P.sha(OUT/'evaluation/score_registration.json')) for label in labels for q in qq for stage in stages]
    for label,q,stage,_ in jobs:artifact(label,q,stage)
    tick=time.perf_counter();snapshot=C.resources();first=one(jobs[0]);workers=worker_count(C.resources(),first,max_workers)
    dispatch=OUT/'evaluation/score_dispatch'/(key+'.json')
    P.immutable(dispatch,dict(workers=workers,threads_per_worker=1,resources_before=snapshot,resources_after_first=C.resources(),
        measured_first_peak_RSS_KiB=first['peak_RSS_KiB']))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*list(pool.map(one,jobs[1:]))]
    verify(reg['source_hashes'])
    result=dict(status='COMPLETE',labels=labels,stages=stages,cells=len(receipts),workers=workers,
        seconds=time.perf_counter()-tick,model_calls=0,new_queries=0,
        files={str(OUT/'evaluation/scores'/label/q['request_id']/stage/'audit_receipt.json'):P.sha(OUT/'evaluation/scores'/label/q['request_id']/stage/'audit_receipt.json') for label,q,stage,_ in jobs})
    P.immutable(done,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','score'))
    p.add_argument('--labels',nargs='+',choices=(*HISTORICAL,*LABELS),default=list(LABELS))
    p.add_argument('--stages',nargs='+',choices=('RAW','BASE_POST'),default=['RAW','BASE_POST'])
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    r=prepare() if a.action=='prepare' else run(a.labels,a.stages,a.workers)
    print({k:v for k,v in r.items() if k not in ('requests','files','source_hashes')},flush=True)
