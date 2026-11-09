"""CPU-only, finite fixed-region assembly/fitting and unchanged scoring cascade."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse, json, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from r2.pilots.bench_regional_mix_dual_v1 import runtime as H

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_local_shape_mix_v1'
VERSION='bench_local_shape_mix_cpu_runtime_v1'
read,resolve,sha,arrays=H.read,H.resolve,H.sha,H.arrays
write,npz,csvwrite=H.write_new,H.npz,H.csvwrite
verify,clean,utc,resources=H.verify,H.clean,H.utc,H.resources

def load_jobs():
    path=OUT/'jobs.json';r=read(path)
    if r['status']!='FROZEN' or len(r['jobs'])!=12:raise ValueError('Expected frozen twelve jobs')
    if {j['label'] for j in r['jobs']}!={'LOCKED_RAW','LOCAL_SHAPE'}:raise ValueError('Unexpected arm')
    if len(r['metric_fields'])!=177:raise ValueError('Original metric schema required')
    verify(r['protected']);verify(r['source_hashes']);H.assert_cpu()
    return r,sha(path),{j['job_id']:j for j in r['jobs']}

def verify_result(receipt):
    if receipt['status']!='COMPLETE':raise ValueError('Incomplete producer')
    verify(receipt['files']);cp=read(receipt['artifacts']['cpu_complete']);verify(cp['files'])
    if cp['status']!='COMPLETE':raise ValueError('Incomplete original CPU cascade')
    return receipt

def definition_hashes():
    book=H.definition_hashes()
    for p in Path(__file__).parent.glob('*.py'):book[str(p)]=sha(p)
    return book

def run_one(job_id):
    reg,rh,jobs=load_jobs();job=jobs[job_id];dest=OUT/'runs'/job_id;done=dest/'complete.json'
    if done.exists():
        r=verify_result(read(done))
        if r['registration_sha256']!=rh or r['job_sha256']!=H.canonical_hash(job):raise ValueError('Changed completed identity')
        return r
    if dest.exists():raise RuntimeError('Partial job preserved; no implicit replay: '+str(dest))
    dest.mkdir(parents=True);tick=time.perf_counter();phase='ASSEMBLY';events=dest/'events.jsonl'
    write(dest/'attempt.json',dict(version=VERSION,utc=utc(),pid=os.getpid(),job=job,
        registration_sha256=rh,automatic_retry=False,scientific_budget=dict(assembly=1,fit=int(job['label']=='LOCAL_SHAPE'),GEO=0,score_rows=2)))
    try:
        write(dest/'resources_before.json',resources());H.event(events,'START',phase=phase)
        from . import operator as O
        result=O.build_from_job(job)
        post=result['post'];pre=result['pre'];q=job['request'];K=int(q['K']);raw=arrays(job['paths']['raw'])
        if pre.dtype!=np.float32 or pre.shape!=raw['raw'].shape:raise ValueError('Invalid full-slot reference')
        if pre[:K].tobytes()!=raw['raw'][:K].tobytes():raise ValueError('Canonical C changed')
        inp=arrays(q['inference_npz'])
        if post['C'].tobytes()!=inp['C_vertices'][inp['C_faces_original']].tobytes():raise ValueError('Source C changed')
        if not np.array_equal(post['output'],post['vertices'][post['faces']]):raise ValueError('Indexed output mismatch')
        npz(dest/'post.npz',post)
        npz(dest/'pre_geo.npz',dict(output=pre,known_faces=np.asarray(K),raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask'],
             stage=np.asarray('FIXED_ASSEMBLY_FULL_SLOT_REFERENCE_NO_GEO_EXECUTED')))
        if result.get('mappings'):npz(dest/'mapping.npz',result['mappings'])
        write(dest/'geometry.json',result['summary'])
        from meshflow_control.geo_io import write_indexed_ply
        write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        H.event(events,'COMPLETE',phase=phase,seconds=time.perf_counter()-tick)
        _,_,_,_,CPU,compare=H.modules()
        phase='BASE_PAIR_SCAN';H.event(events,'START',phase=phase)
        pairs=CPU.H.scan(post['output'],K);write(dest/'pairs.json',pairs);H.event(events,'COMPLETE',phase=phase)
        phase='ORIGINAL_CPU_CASCADE';H.event(events,'START',phase=phase)
        CPU.process_one(q,job['label'],dest/'post.npz',dest/'pre_geo.npz',job['paths']['raw'],dest/'pairs.json',dest/'cpu',False,rh)
        H.event(events,'COMPLETE',phase=phase)
        artifacts=dict(base=str(dest/'post.npz'),pre=str(dest/'pre_geo.npz'),pairs=str(dest/'pairs.json'),
            final=str(dest/'cpu/CLEAN_FINAL_POST/post.npz'),final_pairs=str(dest/'cpu/CLEAN_FINAL_POST/pairs.json'),
            cpu_dir=str(dest/'cpu'),cpu_complete=str(dest/'cpu/complete.json'),metric_rows=str(dest/'cpu/metric_rows.json'),
            geometry=str(dest/'geometry.json'),mapping=str(dest/'mapping.npz'))
        phase='CACHE_LEDGER'
        for stage,oldkey,oldpairkey,newkey,newpairkey in [('BASE_POST','original_base','original_base_pairs','base','pairs'),
            ('CLEAN_FINAL_POST','original_final','original_final_pairs','final','final_pairs')]:
            ledger=compare(arrays(job['paths'][oldkey]),arrays(artifacts[newkey]),read(job['paths'][oldpairkey]),read(artifacts[newpairkey]),K,CPU)
            lp=dest/('ledger_'+stage+'.json');write(lp,ledger);artifacts['ledger_'+stage]=str(lp)
        # PRE is the common, frozen HINGE displacement reference, not the fitted geometry.
        # Keep that distinction explicit in the independently versioned slot ledger.
        area=H._areas(raw['raw']);ref_area=H._areas(pre);slot_rows=[]
        for stage,key in [('BASE_POST','base'),('CLEAN_FINAL_POST','final')]:
            mesh=arrays(artifacts[key]);ids=mesh.get('parent_source_raw_slot',mesh.get('source_face_ids'))
            actual={int(s):float(a) for s,a in zip(ids,H._areas(mesh['output'])) if s>=0}
            for slot in range(K,len(pre)):
                slot_rows.append(dict(stage=stage,raw_slot=slot,retained=slot in actual,raw_area=float(area[slot]),
                    common_HINGE_reference_area=float(ref_area[slot]),actual_area=actual.get(slot)))
        csvwrite(dest/'per_slot_deletion_area.csv',slot_rows)
        rows=read(artifacts['metric_rows'])
        if len(rows)!=2 or any(any(f not in row for f in reg['metric_fields']) for row in rows):raise ValueError('Missing full metric rows')
        write(dest/'resources_after.json',resources());verify(reg['protected']);verify(reg['source_hashes']);H.assert_cpu()
        receipt=dict(status='COMPLETE',version=VERSION,utc=utc(),job_id=job_id,label=job['label'],request_id=q['request_id'],
            registration_sha256=rh,job_sha256=H.canonical_hash(job),artifacts=artifacts,seconds=time.perf_counter()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_scope='Fresh subprocess lifetime, all scientific phases',
            assembly_calls=1,fit_calls=int(job['label']=='LOCAL_SHAPE'),GEO_calls=0,base_pair_scan_calls=1,full_cascade_calls=1,new_score_rows=2,
            model_calls=0,D0_calls=0,training_updates=0,new_noise=0,new_queries=0,
            files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()})
        write(done,receipt);print(json.dumps(dict(job_id=job_id,status='COMPLETE',seconds=receipt['seconds'],peak_RSS_KiB=receipt['peak_RSS_KiB'])),flush=True)
        return receipt
    except BaseException as error:
        H.event(events,'FAILED',phase=phase,error=repr(error))
        write(dest/'failure.json',dict(status='STOPPED_PARTIAL_NO_REPLAY',phase=phase,error=repr(error),traceback=traceback.format_exc(),utc=utc()))
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('resources','code-hashes','run-one','status'));p.add_argument('--job-id');a=p.parse_args()
    if a.command=='resources':result=resources()
    elif a.command=='code-hashes':result=definition_hashes()
    elif a.command=='run-one':result=run_one(a.job_id);result={k:result[k] for k in ('status','job_id','seconds')}
    else:
        reg,rh,jobs=load_jobs();result=[dict(job_id=j,state='COMPLETE' if (OUT/'runs'/j/'complete.json').exists() else 'PARTIAL' if (OUT/'runs'/j).exists() else 'NOT_STARTED') for j in jobs]
    print(json.dumps(clean(result),ensure_ascii=False))

if __name__=='__main__':main()
