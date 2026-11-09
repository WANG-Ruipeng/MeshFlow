"""Thirty CPU GEO-only full production scores, with read-only old references."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,csv,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from r2.pilots.bench_adaptation_strength_v1.common import read,resolve,sha,write_new
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import aggregate,numeric

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_geo_only_acceptance_v1'
VERSION='bench_geo_only_original_full177_native_v1'
METRIC_SOURCE=ROOT/'experiments/r2_bench_seat_coverage_v1/results/completion.json'

def arrays(path):
    with np.load(resolve(path),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def csvwrite(path,rows):
    path=resolve(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Evidence changed: '+str(p))

def source_hashes():
    files=CPU.H.Q.definition_hashes()
    for mod in (CPU,CPU.M,CPU.M.holes,CPU.M.T,CPU.M.B):files[str(Path(mod.__file__))]=sha(mod.__file__)
    files[str(Path(__file__))]=sha(__file__)
    files[str(METRIC_SOURCE)]=sha(METRIC_SOURCE)
    return files

def tasks_from(path):
    m=read(path);tasks=m['tasks']
    if len(tasks)!=30 or len({t['request_id'] for t in tasks})!=30:raise ValueError('Exactly30 fixed requests required')
    if any(t['label']!='COVERAGE_FINAL' for t in tasks):raise ValueError('Only frozen Coverage GEO-only authorized')
    if sum(bool(t.get('reuse_geometry')) for t in tasks)!=4:raise ValueError('Expected26 new geometry and4 reused geometry')
    return m,tasks

def decorate(row,t,label,stage,origin):
    q=t['request'];out=dict(row)
    for k in ('request_id','condition_id','parent_id','split'):
        if out[k]!=q[k]:raise ValueError('Original scorer identity differs '+k)
    for k in ('N','K','noise_index'):
        if int(out[k])!=int(q[k]):raise ValueError('Original scorer domain differs '+k)
    out.update(label=label,stage=stage,eval_group=q['eval_group'],panel_origin=q['panel_origin'],
        part_type=q['part_type'],C_boundary_edges=q['boundary_edges'],closed_C=q['closed_C'],
        seam_applicability='NA_NOT_SEAM_REPRESENTATION',result_origin=origin)
    return out

def score_one(job):
    task,mh,code=job;dest=OUT/'scores'/task['request_id'];done=dest/'complete.json'
    meshpath=resolve(task['new_post_path']);identity=dict(version=VERSION,manifest_sha256=mh,
        request_id=task['request_id'],mesh_sha256=sha(meshpath),source_hashes=code)
    if done.exists():
        r=read(done)
        if r['identity']!=identity:raise ValueError('Completed score identity differs')
        verify(r['files']);return r
    if dest.exists():raise RuntimeError('Incomplete scoring attempt; explicit recovery required, no retry: '+str(dest))
    dest.mkdir(parents=True);write_new(dest/'attempt.json',dict(identity=identity,pid=os.getpid(),time=time.time(),model_calls=0))
    tick=time.perf_counter()
    try:
        mesh=arrays(meshpath);q=task['request']
        if np.asarray(mesh['raw_seam_slot_mask'],bool).any():raise ValueError('Native GEO-only cannot invent SEAM roles')
        payload=CPU.score_current(q,mesh,dest/'production','GEO_ONLY',seam_model=False)
        if payload['score']['seam_to_bulk']['status']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Wrong Native applicability')
        fields=read(METRIC_SOURCE)['metric_fields']
        if len(fields)!=177 or set(fields)-set(payload['row']):raise ValueError('Complete original177 fields required')
        row=decorate(payload['row'],task,'COVERAGE_FINAL','GEO_ONLY','NEW_FULL_SCORE_REUSED_GEOMETRY' if task['reuse_geometry'] else 'NEW_FULL_SCORE_NEW_GEOMETRY')
        comparisons=[]
        if task.get('old_subset_score_path'):
            old=read(task['old_subset_score_path']);oldrow=old['row']
            if old['identity']['mesh_sha256']!=identity['mesh_sha256']:raise ValueError('Reused geometry differs from prior subset score')
            keys=('CF_illegal_pair_count','CF_illegal_affected_free_fraction','FF_illegal_pair_count','FF_illegal_affected_free_fraction',
                'ConflictCov_C','ConflictCov_C_cross','ConflictCov_C_dup','ConflictCov_C_cross_and_dup','ConflictCov_C_status',
                'actual_free_faces','deleted_free_faces','free_total_surface_area','free_minimum_angle_lt5_fraction',
                'free_face_denominator','CF_tested_pair_denominator','FF_tested_pair_denominator','C_query_count','C_query_weight_denominator')
            for k in keys:
                value=payload['score'].get(k,payload['score']['metrics'].get(k))
                same=value==oldrow[k]
                comparisons.append(dict(request_id=q['request_id'],field=k,before=oldrow[k],after=value,exact_equal=same))
                if not same:raise AssertionError('Prior subset mismatch: '+k+': '+repr((oldrow[k],value)))
            write_new(dest/'prior_subset_exact_check.json',dict(status='PASS',source=str(task['old_subset_score_path']),
                source_sha256=sha(task['old_subset_score_path']),new_samples=0,comparisons=comparisons))
        write_new(dest/'row.json',row)
        r=dict(status='PASS',identity=identity,row=row,seconds=time.perf_counter()-tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,new_queries=0,
            full_production_score_calls=1,metric_fields=177,prior_subset_exact_checks=len(comparisons),
            files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()})
        if 'torch' in sys.modules:raise RuntimeError('Pure CPU score imported torch')
        write_new(done,r);print('FULL_SCORED',q['request_id'],round(r['seconds'],2),flush=True);return r
    except BaseException as exc:
        write_new(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,model_calls=0));raise

def reference_rows(tasks):
    rows=[];sources={}
    for t in tasks:
        for label,paths in (('COVERAGE_FINAL',t['original']),('NATIVE_S6000',t['native'])):
            for stage,key in (('RAW','raw_score_path'),('BASE_POST','base_score_path'),('CLEAN_FINAL_POST','final_score_path')):
                sp=resolve(paths[key]);folder=sp.parent;done=read(folder/'complete.json');verify(done['files'])
                score=read(sp)
                if done['status']!='PASS' or score['seam_to_bulk']['status']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Original fullscore not Native PASS')
                rowpath=folder/'row.json';r=decorate(read(rowpath),t,label,stage,'READ_ONLY_ORIGINAL_FULL_SCORE')
                r.update(source_score_path=str(sp),source_score_sha256=sha(sp))
                rows.append(r)
                sources[str(sp)]=sha(sp);sources[str(rowpath)]=sha(rowpath);sources[str(folder/'complete.json')]=sha(folder/'complete.json')
    if len(rows)!=180:raise ValueError('Expected180 readonly reference rows')
    return rows,sources

def collect(tasks,receipts):
    new=[r['row'] for r in receipts];refs,sources=reference_rows(tasks);rows=[*refs,*new]
    fields=read(METRIC_SOURCE)['metric_fields']
    conditions=aggregate(rows,('label','stage','eval_group','parent_id','condition_id'),fields)
    parents=aggregate(conditions,('label','stage','eval_group','parent_id'),fields)
    summary=aggregate(parents,('label','stage','eval_group'),fields)
    pairs=[]
    by={(r['label'],r['stage'],r['request_id']):r for r in rows}
    for t in tasks:
        rid=t['request_id'];b=by[('COVERAGE_FINAL','GEO_ONLY',rid)]
        for label,stage in (('COVERAGE_FINAL','RAW'),('COVERAGE_FINAL','BASE_POST'),('COVERAGE_FINAL','CLEAN_FINAL_POST'),('NATIVE_S6000','CLEAN_FINAL_POST')):
            a=by[(label,stage,rid)]
            r=dict(reference_label=label,reference_stage=stage,candidate_label='COVERAGE_FINAL',candidate_stage='GEO_ONLY',
                **{k:b[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group')})
            for f in fields:
                av,bv=a.get(f),b.get(f);r[f+'__reference']=av;r[f+'__candidate']=bv
                r[f+'__delta']=bv-av if numeric(av) and numeric(bv) else None
            pairs.append(r)
    tables=dict(per_output=rows,new_per_output=new,original_reference_scores=refs,per_condition=conditions,per_parent=parents,
        parent_macro_summary=summary,paired_per_output=pairs)
    for name,table in tables.items():csvwrite(OUT/'scores'/(name+'.csv'),table)
    return sources,fields

def run(manifest_path,workers=3):
    manifest_path=resolve(manifest_path);m,tasks=tasks_from(manifest_path);done=OUT/'scores/COMPLETE.json'
    if not 1<=int(workers)<=3:raise ValueError('Registered CPU score cap is3')
    if done.exists():
        r=read(done);verify(r['files']);return r
    if not (OUT/'core_complete.json').exists():raise RuntimeError('Wait for completed26+4 CPU geometry')
    core=read(OUT/'core_complete.json')
    if core['status']!='COMPLETE':raise RuntimeError('CPU geometry is incomplete')
    code=source_hashes();tick=time.perf_counter();jobs=[(t,sha(manifest_path),code) for t in tasks]
    # Incremental dispatch: first actual registered score establishes its RSS.
    first=score_one(jobs[0]);receipts=[first]
    write_new(OUT/'scores/dispatch.json',dict(workers=int(workers),threads_per_worker=1,
        first_peak_RSS_KiB=first['peak_RSS_KiB'],first_seconds=first['seconds'],model_calls=0,
        resource_scope='Root registered CPU/RAM allowance; no GPU worker',time=time.time()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers),mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=iter(jobs[1:]);active={}
        while True:
            while len(active)<int(workers):
                job=next(pending,None)
                if job is None:break
                active[pool.submit(score_one,job)]=job
            if not active:break
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                active.pop(future)
                try:receipts.append(future.result())
                except BaseException:
                    for other in active:other.cancel()
                    raise
    by={r['row']['request_id']:r for r in receipts};receipts=[by[t['request_id']] for t in tasks]
    sources,fields=collect(tasks,receipts);verify(code)
    result=dict(status='COMPLETE',new_full_scores=30,new_geometry=26,reused_geometry=4,old_full_scores_reused=180,
        unique_new_outputs=0,additional_model_calls=0,new_queries=0,metric_fields=fields,workers=int(workers),
        seconds=time.perf_counter()-tick,worker_seconds_sum=sum(r['seconds'] for r in receipts),
        peak_worker_RSS_KiB=max(r['peak_RSS_KiB'] for r in receipts),failures=0,
        subset_recomputed_exactly_verified=4,original_score_hashes=sources,source_hashes=code,
        manifest_sha256=sha(manifest_path),files={str(p):sha(p) for p in (OUT/'scores').rglob('*') if p.is_file()})
    write_new(done,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();r=run(a.manifest,a.workers);print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_hashes','original_score_hashes','metric_fields')}))
