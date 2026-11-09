"""Bounded CPU scorer: eight GEO(RAW) and two source GEO outputs, no models.

CF/FF/I call the unchanged production mathematics and its frozen C queries.
Existing RAW and BASE_POST scores are read-only references, never rescored.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,csv,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from r2.pilots.bench_adaptation_strength_v1.common import read,resolve,sha,write_new
from r2.tools.partial_edge_binding_v1 import metrics as ORIGINAL

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/r2_bench_backend_counterfactual_v1'
VERSION='bench_backend_original_CF_FF_I_subset_v1'
ILLEGAL={'proper_transverse_penetration','coplanar_positive_area_overlap'}


def arrays(p):
    with np.load(resolve(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def csvwrite(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def unchanged_C(a,b):
    return a.shape==b.shape and all(sorted(map(tuple,x))==sorted(map(tuple,y)) for x,y in zip(a,b))


def score_one(job):
    task,manifest_sha=job
    dest=OUT/'scores'/task['job_id'];done=dest/'complete.json'
    meshpath=resolve(task['new_post_path']);q=task['request'];K=int(task['K']);N=int(task['N'])
    identity=dict(version=VERSION,job_id=task['job_id'],manifest_sha256=manifest_sha,mesh_sha256=sha(meshpath))
    if done.exists():
        result=read(done)
        if result['identity']!=identity:raise ValueError('Completed score identity changed')
        for p,h in result['files'].items():
            if sha(p)!=h:raise ValueError('Completed score evidence changed')
        return result
    if dest.exists():raise RuntimeError('Incomplete scoring attempt retained; no silent retry: '+str(dest))
    dest.mkdir(parents=True);write_new(dest/'attempt.json',dict(identity=identity,pid=os.getpid(),time=time.time(),new_model_calls=0))
    start=time.perf_counter()
    try:
        mesh=arrays(meshpath);tri=np.asarray(mesh['output']);F=tri[K:]
        inp=arrays(q['inference_npz']);cc=inp['C_vertices'][inp['C_faces_model']]
        co=inp['C_vertices'][inp['C_faces_original']]
        if tri.dtype!=np.float32 or not np.isfinite(tri).all() or not unchanged_C(tri[:K],cc):raise ValueError('C/finite/dtype contract failed')
        if not np.array_equal(tri[:K],co):raise ValueError('Source C winding recovery differs')
        helpers=ORIGINAL.helpers();mod=helpers;L=ORIGINAL.L
        reference=helpers['old'].load_queries(q['query_descriptor'])
        if len(F):metrics,cf,ff=mod['geometry'].scan_geometry(cc,F,L)
        else:
            cf,ff=helpers['Q']._empty_scans(K);metrics={}
        trusted=mod['control'].trusted_cf(cc,F,L,cf_scan=cf)
        im,vectors=mod['control'].conflict_metrics(F,dict(C=cc,L0=L),reference,trusted)
        metrics.update(im)
        if abs(im['ConflictCov_C']-im['ConflictCov_C_cross']-im['ConflictCov_C_dup']+im['ConflictCov_C_cross_and_dup'])>1e-12:raise ValueError('I union identity failed')
        for kind,scan in (('CF',cf),('FF',ff)):
            bad=[r for r in scan['contacts'] if r['classification'] in ILLEGAL]
            affected={int(r['pair'][1]) for r in bad} if kind=='CF' else {int(x) for r in bad for x in r['pair']}
            metrics.update({kind+'_illegal_pair_count':len(bad),kind+'_illegal_affected_free_fraction':len(affected)/len(F) if len(F) else None,
                kind+'_proper_pair_count':scan['proper_pair_count'],kind+'_coplanar_pair_count':scan['coplanar_positive_area_pairs'],
                kind+'_unresolved_or_degenerate_pair_count':scan['unresolved_or_degenerate_pairs']})
        shapes={name:mod['shape'].shape_quality(part,L) for name,part in (('C',cc),('free',F),('whole',tri))}
        for name,s in shapes.items():
            metrics.update({name+'_total_surface_area':s['total_area'],name+'_minimum_angle_lt5_fraction':s['min_angle_lt5_fraction'],
                name+'_near_degenerate_count':s['near_degenerate_area_count'],name+'_exact_zero_area_count':s['exact_zero_area_count']})
        metrics.update(actual_free_faces=len(F),input_free_faces=N-K,deleted_free_faces=N-len(tri))
        geometry=read(task['new_geometry_path'])
        row=dict(job_id=task['job_id'],kind=task['kind'],label=task['label'],request_id=task['request_id'],
            condition_id=task['condition_id'],parent_id=task['parent_id'],noise_index=q.get('noise_index'),stage='GEO_RAW' if task['kind']=='RAW_GEO' else 'SOURCE_GEO',
            N=N,K=K,free_face_denominator=len(F),CF_tested_pair_denominator=K*len(F),FF_tested_pair_denominator=len(F)*(len(F)-1)//2,
            C_query_count=len(reference['arrays']['C_points']),C_query_weight_denominator=float(reference['arrays']['C_weights'].sum()),**metrics)
        result=dict(status='PASS',identity=identity,row=row,metric_status=dict(I=im['ConflictCov_C_status'],source_style='NA_NOT_UNIQUE_COMPLETION'),
            original_definitions_unchanged=True,original_C_queries_reused=True,source_hashes=ORIGINAL.definition_hashes(),
            geometry_receipt_path=task['new_geometry_path'],geometry_receipt_sha256=sha(task['new_geometry_path']),
            model_calls=0,new_queries=0,full_CF_scans=1,full_FF_scans=1,seconds=time.perf_counter()-start,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        for name,value in (('score',result),('pairs',dict(CF=cf,FF=ff)),('shape_per_face',shapes)):
            write_new(dest/(name+'.json'),value)
        with (dest/'vectors.npz').open('xb') as f:np.savez_compressed(f,**vectors)
        result['files']={str(p):sha(p) for p in dest.iterdir() if p.is_file()}
        if 'torch' in sys.modules:raise RuntimeError('CPU scorer imported torch')
        write_new(done,result)
        print('SCORED',task['job_id'],round(result['seconds'],2),flush=True)
        return result
    except BaseException as exc:
        write_new(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
        raise


def run(manifest_path,workers=2):
    manifest_path=resolve(manifest_path);manifest=read(manifest_path);tasks=manifest['tasks']
    if len(tasks)!=10 or sum(t['kind']=='RAW_GEO' for t in tasks)!=8:raise ValueError('Expected registered 8+2 CPU outputs')
    done=OUT/'scores/COMPLETE.json'
    if done.exists():
        receipt=read(done)
        for p,h in receipt['files'].items():
            if sha(p)!=h:raise ValueError('Completed scoring product changed')
        return receipt
    workers=int(workers)
    if not 1<=workers<=2:raise ValueError('CPU workers must remain within registered cap2')
    tick=time.perf_counter();mh=sha(manifest_path);jobs=[(t,mh) for t in tasks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=list(pool.map(score_one,jobs))
    rows=[r['row'] for r in receipts];references=[]
    for task in tasks:
        if task['kind']!='RAW_GEO':continue
        original=task['original'];label=task['label'];rid=task['request_id']
        for stage,key,default in (
            ('RAW','raw_score_path',ROOT/'experiments/r2_bench_seat_coverage_v1/evaluation/scores'/label/rid/'RAW/score.json'),
            ('BASE_POST','base_score_path',ROOT/'experiments/r2_bench_seat_coverage_v1/cpu'/label/rid/'BASE_POST/scores/score.json')):
            path=resolve(original.get(key,default));score=read(path)
            rr=dict(job_id=task['job_id'],kind=task['kind'],label=label,request_id=rid,parent_id=task['parent_id'],
                condition_id=task['condition_id'],stage=stage,origin='READ_ONLY_ORIGINAL_SCORE',source_score_path=str(path),source_score_sha256=sha(path))
            rr.update(score['metrics'])
            for k in ('free_face_denominator','CF_tested_pair_denominator','FF_tested_pair_denominator','C_query_weight_denominator','C_query_count'):rr[k]=score.get(k)
            references.append(rr)
    csvwrite(OUT/'scores/new_per_output.csv',rows)
    csvwrite(OUT/'scores/original_reference_scores.csv',references)
    comparisons=[]
    metrics=('CF_illegal_affected_free_fraction','FF_illegal_affected_free_fraction','ConflictCov_C','free_total_surface_area','actual_free_faces','deleted_free_faces','free_minimum_angle_lt5_fraction')
    for row in rows:
        if row['kind']!='RAW_GEO':continue
        for old in (r for r in references if r['job_id']==row['job_id']):
            rec=dict(job_id=row['job_id'],label=row['label'],request_id=row['request_id'],parent_id=row['parent_id'],baseline_stage=old['stage'])
            for k in metrics:
                a,b=old.get(k),row.get(k);rec[k+'_before']=a;rec[k+'_after']=b
                rec[k+'_delta']=b-a if a is not None and b is not None else None
            comparisons.append(rec)
    csvwrite(OUT/'scores/paired_differences.csv',comparisons)
    result=dict(status='COMPLETE',outputs=10,raw_geo_outputs=8,source_geo_outputs=2,old_scores_reused=16,workers=workers,
        model_calls=0,new_queries=0,seconds=time.perf_counter()-tick,failures=0,manifest_sha256=mh,
        files={str(p):sha(p) for p in (OUT/'scores').rglob('*') if p.is_file() and p.name!='COMPLETE.json'})
    write_new(done,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();r=run(a.manifest,a.workers);print(json.dumps({k:v for k,v in r.items() if k!='files'}))
