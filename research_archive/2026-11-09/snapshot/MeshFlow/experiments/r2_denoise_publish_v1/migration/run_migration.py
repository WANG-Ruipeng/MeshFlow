"""Bounded CPU-only package migration against frozen 36 existing POSTs.

No research scoring/model calls. One orient run per 36 POST; one cached complete
pipeline per 12 RAW x {D0,RBF}. A failed started job is never replayed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT/'experiments/r2_denoise_publish_v1/migration'
REF = ROOT/'experiments/r2_source_winding_export_v1'
REG = REF/'registration.json'
KEY_ALIASES = {'C_source_corner_permutation':'C_source_corner_permutations'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,value):
    p=Path(p); assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def csv_write(p,rows):
    assert rows
    keys=list(dict.fromkeys(k for r in rows for k in r))
    p=Path(p); assert p.resolve().is_relative_to(OUT.resolve())
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def payload(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def snapshot():
    def call(args):
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    return dict(time_unix=time.time(),cpu_logical=os.cpu_count(),loadavg=os.getloadavg(),
       meminfo=Path('/proc/meminfo').read_text(),self_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       gpu=call(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),
       gpu_processes=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
       cpu_processes=call(['ps','-eo','pid,comm,pcpu,pmem,rss,args','--sort=-pcpu']))

def package_hashes():
    from meshflow_control.denoise import implementation_hashes
    h=implementation_hashes()
    h['r2_cli.py']=sha(ROOT/'r2/src/meshflow_control/r2_cli.py')
    return h

def checked_registration():
    r=read(OUT/'registration.json')
    assert r['code_hashes']==package_hashes(),'Package implementation changed after registration'
    assert r['runner_sha256']==sha(__file__),'Migration runner changed'
    return r

def prepare():
    assert not (OUT/'registration.json').exists(),'Already prepared'
    old=read(REG);records=old['records'];assert len(records)==36
    assert sorted({r['method'] for r in records})==['BLEND','D0_BASE','RBF']
    protected={str(REG):sha(REG)}
    jobs=[]
    for mode in ['orient','chain']:
      for r in records:
        if mode=='chain' and r['method']=='BLEND':continue
        rr=dict(r)
        rr['reference_oriented_path']=str(REF/'outputs'/r['method']/r['request_id']/'oriented.npz')
        rr['reference_restored_path']=str(REF/'outputs'/r['method']/r['request_id']/'restored_C.npz')
        rr['reference_receipt_path']=str(REF/'outputs'/r['method']/r['request_id']/'receipt.json')
        paths=[r['post_path'],r['condition_source'],rr['reference_oriented_path'],rr['reference_restored_path'],rr['reference_receipt_path']]
        if mode=='chain':paths += [r['raw_path'],r['D1_denoised_path'],r['D1_receipt_path'],r['denoised_path']]
        for p in paths:
            protected[p]=sha(p)
        assert protected[r['post_path']]==r['post_sha256']
        assert protected[r['condition_source']]==r['condition_source_sha256']
        if mode=='chain':
            assert protected[r['raw_path']]==r['raw_sha256']
            assert protected[r['D1_denoised_path']]==r['D1_denoised_sha256']
        rr.update(mode=mode,job_id=mode+'__'+r['method']+'__'+r['request_id'],protected_paths=sorted(set(paths)))
        jobs.append(rr)
    assert len(jobs)==60
    write(OUT/'registration.json',dict(status='REGISTERED',records=jobs,code_hashes=package_hashes(),
         runner_sha256=sha(__file__),fixed_L=old['fixed_L'],protected=protected,
         source_registration_sha256=sha(REG),model_forwards=0,quality_scores=0,
         budget=dict(orient_only=36,cached_chains=24,orientation_solves=60,GEO_calls=24,RBF_solves=12),
         array_comparison='Every old final/restored NPZ array byte-exact, except explicit singular->plural P key alias; package-only new metadata separately retained',
         threads_per_worker=2,max_workers=2,automatic_retry=False))
    write(OUT/'resources_preparation.json',snapshot())
    print('REGISTERED 60 jobs',flush=True)

def compare(path,ref,stage):
    a=payload(path);b=payload(ref);rows=[]
    for oldkey,value in b.items():
        key=KEY_ALIASES.get(oldkey,oldkey)
        ok=key in a and bits(a[key],value)
        row=dict(stage=stage,reference_key=oldkey,package_key=key,status='PASS' if ok else 'FAIL',
                 reference_shape=str(value.shape),reference_dtype=str(value.dtype),
                 actual_shape=str(a[key].shape) if key in a else 'MISSING',
                 actual_dtype=str(a[key].dtype) if key in a else 'MISSING')
        if key in a and a[key].shape==value.shape and value.dtype.kind in 'fiu' and a[key].dtype.kind in 'fiu':
            row['max_abs_difference']=float(np.max(np.abs(a[key].astype(np.float64)-value.astype(np.float64)),initial=0))
        rows.append(row)
    return rows

def work(job):
    from meshflow_control import denoise,source_winding,rbf_coordination
    reg=checked_registration()
    jid=job['job_id'];resultpath=OUT/'jobs'/f'{jid}.json';started=OUT/'jobs'/f'{jid}.started.json'
    if resultpath.exists():
        done=read(resultpath);assert done['status']=='PASS'
        for p,h in done['output_files'].items():assert sha(p)==h
        return done
    if started.exists():raise RuntimeError('Started migration job has no valid completion: '+jid)
    inputs={p:sha(p) for p in job['protected_paths']}
    assert all(reg['protected'][p]==h for p,h in inputs.items())
    write(started,dict(time_unix=time.time(),job_id=jid,inputs=inputs,pid=os.getpid()))
    dest=OUT/job['mode']/job['method']/job['request_id']
    counter={'orientation_solves':0,'GEO_calls':0,'RBF_solves':0,'model_forwards':0}
    oldori=source_winding.orient_fixed_C;oldgeo=denoise.geo_c.process;oldrbf=rbf_coordination.coordinate_restore;oldmodel=denoise.actual_proposal
    def wrap(fn,key):
        def call(*args,**kwargs):counter[key]+=1;return fn(*args,**kwargs)
        return call
    def blocked(*args,**kwargs):raise AssertionError('Model invocation forbidden in migration')
    source_winding.orient_fixed_C=wrap(oldori,'orientation_solves');denoise.geo_c.process=wrap(oldgeo,'GEO_calls')
    rbf_coordination.coordinate_restore=wrap(oldrbf,'RBF_solves');denoise.actual_proposal=blocked
    start=time.perf_counter()
    try:
        if job['mode']=='orient':
            receipt=denoise.orient(input=job['post_path'],condition=job['condition_source'],out=dest)
        else:
            receipt=denoise.run(input=job['raw_path'],known_faces=int(job['K']),out=dest,
                condition=job['condition_source'],proposal=job['D1_denoised_path'],proposal_receipt=job['D1_receipt_path'],
                coordination='rbf' if job['method']=='RBF' else 'none',
                reference_length=reg['fixed_L'] if job['method']=='RBF' else None,winding='source')
        expected=dict(orientation_solves=1,GEO_calls=int(job['mode']=='chain'),
                      RBF_solves=int(job['mode']=='chain' and job['method']=='RBF'),model_forwards=0)
        assert counter==expected,(counter,expected)
        assert 'torch' not in sys.modules,'CPU path imported torch'
        rows=compare(dest/'post.npz',job['reference_oriented_path'],'ORIENTED')
        rows+=compare(dest/'post_restored_C.npz',job['reference_restored_path'],'RESTORED_C_ONLY')
        if job['mode']=='chain':
            rows+=compare(dest/'post_canonical.npz',job['post_path'],'MODEL_POST')
            a=payload(dest/'denoised.npz');b=payload(job['denoised_path'])
            for key in ['output','C']:
                rows.append(dict(stage='PRE_GEO',reference_key=key,package_key=key,
                    status='PASS' if bits(a[key],b[key]) else 'FAIL',reference_shape=str(b[key].shape),
                    reference_dtype=str(b[key].dtype),actual_shape=str(a[key].shape),actual_dtype=str(a[key].dtype),
                    max_abs_difference=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)),initial=0))))
        csv_write(dest/'migration_array_differences.csv',rows)
        after={p:sha(p) for p in job['protected_paths']};assert after==inputs,'Original input changed'
        allpass=all(r['status']=='PASS' for r in rows)
        result=dict(status='PASS' if allpass else 'FAIL',job_id=jid,mode=job['mode'],method=job['method'],
            parent_id=job['parent_id'],condition_id=job['condition_id'],request_id=job['request_id'],noise=int(job['candidate']),
            seconds=time.perf_counter()-start,maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            counts=counter,arrays_compared=len(rows),failed_arrays=[r for r in rows if r['status']!='PASS'],
            before_hashes=inputs,after_hashes=after,torch_imported=False,
            output_files={str(p):sha(p) for p in sorted(dest.iterdir()) if p.is_file()},
            source_trust=receipt.get('restoration',receipt.get('source_export',{})),
            validation='Migration array equality, not new quality evidence')
        write(resultpath,result)
        assert allpass,jid+' array mismatch'
        print(jid,'PASS',round(result['seconds'],3),'seconds',flush=True)
        return result
    except BaseException as e:
        write(OUT/'jobs'/f'{jid}.failure.json',dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc(),counts=counter,seconds=time.perf_counter()-start))
        raise
    finally:
        source_winding.orient_fixed_C=oldori;denoise.geo_c.process=oldgeo;rbf_coordination.coordinate_restore=oldrbf;denoise.actual_proposal=oldmodel

def wave(mode,first,workers):
    reg=checked_registration();jobs=[r for r in reg['records'] if r['mode']==mode]
    if first:jobs=jobs[:1]
    else:jobs=jobs[1:]
    assert workers in (1,2) and (not first or workers==1)
    suffix=mode+('_first' if first else '_remaining')
    write(OUT/f'resources_{suffix}_before.json',snapshot());start=time.perf_counter()
    if workers==1:
        results=[work(j) for j in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(work,jobs))
    write(OUT/f'resources_{suffix}_after.json',snapshot())
    write(OUT/f'wave_{suffix}.json',dict(status='PASS',workers=workers,threads_per_worker=2,seconds=time.perf_counter()-start,
        jobs=[r['job_id'] for r in results],max_observed_worker_RSS_KiB=max(r['maxrss_KiB'] for r in results),
        timing_scope='Concurrent migration wall time including verification/I-O; not a model benchmark'))

def final():
    reg=checked_registration();results=[read(OUT/'jobs'/f"{j['job_id']}.json") for j in reg['records']]
    assert all(r['status']=='PASS' for r in results)
    after={p:sha(p) for p in reg['protected']};assert after==reg['protected']
    counts={k:sum(r['counts'][k] for r in results) for k in results[0]['counts']}
    assert counts==dict(orientation_solves=60,GEO_calls=24,RBF_solves=12,model_forwards=0)
    for r in results:
        for p,h in r['output_files'].items():assert sha(p)==h
    rows=[{k:r[k] for k in ['mode','method','parent_id','condition_id','request_id','noise','status','arrays_compared','seconds','maxrss_KiB']}|r['counts'] for r in results]
    csv_write(OUT/'validation_per_case.csv',rows)
    csv_write(OUT/'protected_hashes.csv',[dict(path=p,before_sha256=h,after_sha256=after[p],status='PASS') for p,h in reg['protected'].items()])
    write(OUT/'completion.json',dict(status='PASS',cases=60,orient_only=36,cached_complete_pipeline=24,
        arrays_compared=sum(r['arrays_compared'] for r in results),counts=counts,protected_files=len(after),
        code_hashes=reg['code_hashes'],runner_sha256=reg['runner_sha256'],new_quality_scores=0,new_noise=0,
        model_imports=0,bitwise_equal=True,reference_length=reg['fixed_L'],
        trust='Legacy mapping validated; old reference export independently authenticated against source parent; new installed package does not load GT/parent',
        scope='Code migration regression only; no new quality or research conclusions'))
    print('FINAL PASS',counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','orient-first','orient-rest','chain-first','chain-rest','final'],required=True);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES empty'
    assert all(os.environ.get(k)=='2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    if a.stage=='prepare':prepare()
    elif a.stage=='final':final()
    else:
        mode,part=a.stage.split('-');wave(mode,part=='first',a.workers)
