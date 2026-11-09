"""Frozen 12 RAW-coordinate targets -> original GEO_C and full CPU cascade.

No neural framework, D0, sampling, target selection, or metric redefinition.
Each producer is immutable and independently receipted. Completed work may be
verified and reused; a partial scientific attempt is never replayed silently.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, concurrent.futures, multiprocessing, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C
from r2.pilots.bench_backend_counterfactual_v1 import core as B
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
from r2.pilots.bench_geo_only_acceptance_v1 import evaluate as OLD_E
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import aggregate, numeric

ROOT,OUT=C.ROOT,C.OUT
LABEL='LOCAL_REPAIR'
LABELS={'target':LABEL,'identity':'RAW_IDENTITY'}
VERSION='bench_raw_local_repair_original_geometry_full177_v1'
MANIFEST=OUT/'data/manifest.json'
READY=OUT/'targets_READY.json'
METRIC_SOURCE=ROOT/'experiments/r2_bench_seat_coverage_v1/results/completion.json'
STAGES=('BASE_POST','CLEAN_FINAL_POST')
PREFIXES=('a8','27','838c','91e','fd355','2d')


def tasks_from(path=None):
    m=C.read(path or MANIFEST);tasks=m['tasks']
    if m['status']!='FROZEN' or len(tasks)!=12 or len({t['request_id'] for t in tasks})!=12:
        raise ValueError('Exactly the frozen12 asset requests are required')
    if len({t['parent_id'] for t in tasks})!=6:
        raise ValueError('Expected six fixed parents')
    for prefix in PREFIXES:
        group=[t for t in tasks if t['parent_id'].startswith(prefix)]
        if len(group)!=2 or sorted(int(t['noise_index']) for t in group)!=[0,1]:
            raise ValueError('Frozen parent/noise membership changed: '+prefix)
    for t in tasks:
        q=t['request']
        if any(t[k]!=q[k] for k in ('request_id','parent_id','condition_id','N','K','noise_index')):
            raise ValueError('Task/request identity differs')
        for name in ('RAW','SOURCE_C','INFERENCE','QUERY'):
            a=t['assets'][name]
            if a['status']!='AVAILABLE' or C.sha(a['path'])!=a['sha256']:
                raise ValueError('Frozen input asset unavailable/changed: '+name)
    return m,tasks


def locations(t,mode='target'):
    rid=t['request_id'];g=OUT/'geometry'/rid;p=OUT/'cpu'/LABELS[mode]/rid
    if mode=='identity':
        a=t['assets']
        return dict(target=C.resolve(a['RAW']['path']),geometry=None,
            pre=C.resolve(a['GEO_ONLY_PRE']['path']),base=C.resolve(a['GEO_ONLY_BASE']['path']),
            pairs=C.resolve(a['GEO_ONLY_PAIRS']['path']),cpu=p,final=p/'CLEAN_FINAL_POST/post.npz')
    return dict(target=OUT/'targets'/rid/'target.npz',geometry=g,
        pre=g/'pre_geo.npz',canonical=g/'post_canonical.npz',base=g/'post.npz',
        pairs=g/'pairs.json',cpu=p,final=p/'CLEAN_FINAL_POST/post.npz')


def source_hashes():
    _,_,_,h=B.dependencies();h.update(OLD_E.source_hashes())
    for mod in (B,CPU,CPU.H,CPU.alpha,CPU.holes,C):
        h[str(Path(mod.__file__))]=C.sha(mod.__file__)
    h[str(Path(__file__))]=C.sha(__file__)
    return h


def registration(manifest_path=None,ready_path=None,mode='target'):
    mp=C.resolve(manifest_path or MANIFEST);rp=C.resolve(ready_path or READY)
    m,tasks=tasks_from(mp);ready=C.read(rp) if mode=='target' else {}
    if mode=='target' and ready.get('status') not in ('READY','COMPLETE','PASS'):
        raise ValueError('Targets have not passed their explicit readiness barrier')
    if ready.get('files'):C.verify(ready['files'])
    assets_done=OUT/'data/assets_COMPLETE.json';ar=C.read(assets_done)
    if ar.get('status')!='COMPLETE' or ar.get('requests')!=12:
        raise ValueError('Frozen asset preparation is incomplete')
    C.verify(ar['files']);C.verify(m['protected_assets']);C.verify(m['derived_files'])
    protected={str(mp):C.sha(mp),str(assets_done):C.sha(assets_done)}
    if mode=='target':protected[str(rp)]=C.sha(rp)
    protected.update(m['protected_assets']);protected.update(m['derived_files'])
    protected.update(ready.get('files',{}))
    protocol=OUT/'PROTOCOL.md'
    if not protocol.is_file():raise ValueError('Frozen protocol is missing')
    protected[str(protocol)]=C.sha(protocol)
    addendum=OUT/'CPU_CONTROL_ADDENDUM.md'
    protected[str(addendum)]=C.sha(addendum)
    jobs=[]
    for t in tasks:
        if mode=='identity':
            loc=locations(t,mode)
            for key in ('pre','base','pairs'):protected[str(loc[key])]=C.sha(loc[key])
            old=C.resolve(t['assets']['GEO_ONLY_SCORE']['path']).parent.parent/'complete.json'
            prior=C.read(old);C.verify(prior['files']);protected[str(old)]=C.sha(old)
            jobs.append(dict(request_id=t['request_id'],raw_sha256=t['assets']['RAW']['sha256'],
                reused_GEO_ONLY_base_sha256=C.sha(loc['base']),original_score_receipt=str(old)))
            continue
        target=locations(t)['target'];protected[str(target)]=C.sha(target)
        a=C.arrays(target)
        if 'target' not in a:raise ValueError('target.npz must provide target')
        raw=C.arrays(t['assets']['RAW']['path'])['raw'];x=a['target'];k,n=int(t['K']),int(t['N'])
        if x.dtype!=np.float32 or x.shape!=(n,3,3) or not np.isfinite(x).all():
            raise ValueError('Target must be finite float32 in the original N33 slot domain')
        if x[:k].tobytes()!=raw[:k].tobytes():raise ValueError('Target moved original canonical C')
        jobs.append(dict(request_id=t['request_id'],target_sha256=C.sha(target),
            raw_sha256=t['assets']['RAW']['sha256'],all_requests_including_failed_or_KEEP_ONLY=True))
    value=dict(status='FROZEN',version=VERSION,mode=mode,manifest_path=str(mp),
        targets_ready_path=str(rp) if mode=='target' else None,
        candidate_label=LABELS[mode],jobs=jobs,protected=protected,source_hashes=source_hashes(),
        budget=dict(requests=12,GEO_calls=12 if mode=='target' else 0,source_orientation_calls=12 if mode=='target' else 0,base_scores=12,
            final_scores=12,model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0),
        chain=('RAW same-slot target -> restore C -> original GEO_C/source winding' if mode=='target' else
            'Unchanged RAW -> reuse original GEO_ONLY PRE/BASE/pairs')+' -> HINGE_OPP015 -> ALPHA -> HOLE')
    dest=OUT/'evaluation'/('CPU_'+mode+'_registration.json')
    if dest.exists():
        old=C.read(dest)
        if old!=value:raise ValueError('Frozen CPU registration changed')
        C.verify(old['protected']);C.verify(old['source_hashes']);return old
    C.write(dest,value);return value


def mapping_audit(post,n,k):
    sf=post['source_face_ids'];sc=post['source_corner_ids'];faces=post['faces'];m=len(sf)
    if sf.shape!=(m,) or np.any(sf<0) or np.any(sf>=n) or len(np.unique(sf))!=m:
        raise ValueError('Original RAW face identities are not one-to-one')
    if not np.array_equal(sf[:k],np.arange(k)) or sc.shape!=(m,3):
        raise ValueError('C face identity/corner shape changed')
    if not np.all(np.sort(sc,axis=1)==np.arange(3)):
        raise ValueError('Source corner IDs are not permutations')
    if not np.array_equal(faces,post['corner_vertex_ids'][sf[:,None],sc]):
        raise ValueError('RAW corner-to-vertex transport does not reproduce indexed mesh')
    if post['vertices'][faces].tobytes()!=post['output'].tobytes():
        raise ValueError('Indexed output differs from triangle output')
    kept=np.zeros(n,bool);kept[sf]=True
    if not np.array_equal(kept,post['source_face_kept']):raise ValueError('RAW retention mask mismatch')
    return dict(status='PASS',identity_domain='generated_RAW_slot_not_GT',input_faces=n,
        output_faces=m,source_face_ids_unique=True,source_corner_permutations=True,
        indexed_output_exact=True,full_original_corner_map_verified=True,
        original_C_face_rows_preserved=True,deleted_free_faces=int((~kept[k:]).sum()))


def geometry_one(t,registration_sha,code):
    loc=locations(t);dest=loc['geometry'];done=dest/'complete.json'
    rawpath=t['assets']['RAW']['path'];sourcepath=t['assets']['SOURCE_C']['path']
    identity=dict(version=VERSION,registration_sha256=registration_sha,request_id=t['request_id'],
        target_sha256=C.sha(loc['target']),raw_sha256=C.sha(rawpath),
        source_condition_sha256=C.sha(sourcepath),source_hashes=code)
    if done.exists():
        r=C.read(done)
        if r['identity']!=identity:raise ValueError('Completed geometry identity differs')
        C.verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial geometry retained; explicit recovery required: '+str(dest))
    C.write(dest/'attempt.json',dict(identity=identity,time=time.time(),pid=os.getpid(),model_calls=0))
    start=time.perf_counter();calls=0
    try:
        P,geo,io,_=B.dependencies();raw=C.arrays(rawpath);source=C.arrays(sourcepath)
        target=C.arrays(loc['target'])['target'];k,n=int(t['K']),int(t['N'])
        if raw['raw'].shape!=(n,3,3) or raw['raw_seam_slot_mask'].any():
            raise ValueError('Only the fixed Native RAW slot representation is authorized')
        if np.any(raw['raw_anchor_ids'][k:]>=0):raise ValueError('Unexpected free seam anchors')
        if target[:k].tobytes()!=raw['raw'][:k].tobytes():raise ValueError('Target changed C')
        before=raw['raw'].tobytes()
        pre,canonical,post,g,orient,anchors=P.geometry_from_proposal(raw,target,source);calls=1
        if raw['raw'].tobytes()!=before or pre.tobytes()!=target.tobytes():
            raise ValueError('RAW mutated or target silently changed during anchor restoration')
        if g['stats']['radius_model']!=geo.RADIUS:raise ValueError('Original GEO radius changed')
        ca=mapping_audit(canonical,n,k);pa=mapping_audit(post,n,k)
        if post['C'].tobytes()!=source['C_original'].tobytes():raise ValueError('Source C not restored exactly')
        C.npz(loc['pre'],dict(output=pre,C=raw['raw'][:k],known_faces=raw['known_faces'],
            raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        C.npz(loc['canonical'],canonical);C.npz(loc['base'],post)
        io.write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        C.write(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']))
        C.write(dest/'anchor_identity.json',anchors)
        C.write(dest/'mapping_audit.json',dict(canonical=ca,oriented=pa))
        C.write(dest/'source_restoration.json',orient['restoration_summary'])
        o=orient['orientation_result']
        C.write(dest/'orientation.json',dict(summary=o['summary'],components=o['component_rows'],edges=o['edge_rows']))
        C.write(loc['pairs'],CPU.H.scan(post['output'],k))
        if 'torch' in sys.modules:raise RuntimeError('CPU geometry imported torch')
        r=dict(status='COMPLETE',identity=identity,GEO_calls=1,source_orientation_calls=1,
            model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
            seconds=time.perf_counter()-start,files={str(p):C.sha(p) for p in dest.rglob('*') if p.is_file()})
        C.write(done,r);return r
    except BaseException as exc:
        C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            GEO_returned=calls,automatic_retry=False,model_calls=0));raise


def one(job):
    t,rh,code,mode=job;loc=locations(t,mode)
    g=geometry_one(t,rh,code) if mode=='target' else dict(seconds=0)
    result=CPU.process_one(t['request'],LABELS[mode],loc['base'],loc['pre'],t['assets']['RAW']['path'],
        loc['pairs'],loc['cpu'],seam_model=False,registration_sha=rh)
    rows=C.read(loc['cpu']/'metric_rows.json');fields=C.read(METRIC_SOURCE)['metric_fields']
    if len(rows)!=2 or len(fields)!=177 or {r['stage'] for r in rows}!=set(STAGES):
        raise ValueError('Expected original BASE/FINAL177 fields')
    for row in rows:
        if set(fields)-set(row) or row['seam_applicability']!='NA_NOT_SEAM_REPRESENTATION':
            raise ValueError('Incomplete fields or incorrect Native metric applicability')
    equality=None
    if mode=='identity':
        old=C.resolve(t['assets']['GEO_ONLY_SCORE']['path']).parent.parent/'complete.json'
        prior=C.read(old);C.verify(prior['files'])
        original=prior['row'];base=next(r for r in rows if r['stage']=='BASE_POST')
        changed=[f for f in fields if base[f]!=original[f]]
        if changed:raise ValueError('Recomputed identity BASE differs from frozen GEO_ONLY fields: '+repr(changed))
        equality=dict(status='PASS',compared_fields=len(fields),exact_equal_fields=len(fields),
            reference_receipt=str(old),reference_sha256=C.sha(old))
    if 'torch' in sys.modules:raise RuntimeError('CPU evaluation imported torch')
    return dict(request_id=t['request_id'],mode=mode,
        geometry_receipt=str(loc['geometry']/'complete.json') if mode=='target' else None,
        cpu_receipt=str(loc['cpu']/'complete.json'),rows=rows,geometry_seconds=g['seconds'],
        cpu_seconds=result['seconds'],identity_BASE_exact_check=equality,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def collect(tasks,receipts,rh,mode):
    dest=OUT/'scores'/mode;done=dest/'aggregation_complete.json'
    if done.exists():
        r=C.read(done)
        if r['registration_sha256']!=rh:raise ValueError('Aggregate identity changed')
        C.verify(r['files']);return r
    attempt=dest/'aggregation_attempt.json'
    if attempt.exists():raise RuntimeError('Partial aggregate requires explicit recovery; no overwrite')
    C.write(attempt,dict(registration_sha256=rh,time=time.time(),scientific_calls=0))
    fields=C.read(METRIC_SOURCE)['metric_fields'];rows=[]
    for t,r in zip(tasks,receipts):
        if t['request_id']!=r['request_id']:raise ValueError('Collection order changed')
        for row in r['rows']:
            rows.append(OLD_E.decorate(row,t,LABELS[mode],row['stage'],'NEW_FROZEN_RAW_LOCAL_REPAIR_FULL_SCORE'))
    if len(rows)!=24 or len({(r['request_id'],r['stage']) for r in rows})!=24:
        raise ValueError('Expected all24 score rows, including unsuccessful targets')
    conditions=aggregate(rows,('label','stage','eval_group','parent_id','condition_id'),fields)
    parents=aggregate(conditions,('label','stage','eval_group','parent_id'),fields)
    groups=aggregate(parents,('label','stage','eval_group'),fields)
    tables=dict(per_output=rows,per_condition=conditions,per_parent=parents,parent_macro_summary=groups)
    files={str(attempt):C.sha(attempt)}
    for name,table in tables.items():
        p=dest/(name+'.csv');C.csvwrite(p,table);files[str(p)]=C.sha(p)
    jp=dest/'metric_rows.json';C.write(jp,rows);files[str(jp)]=C.sha(jp)
    r=dict(status='COMPLETE',registration_sha256=rh,score_rows=24,metric_fields=fields,
        averaging='noise within condition -> condition within parent -> equal parents within group',files=files)
    C.write(done,r);return r


def verify_complete(mode='all'):
    done=OUT/'scores/COMPLETE.json' if mode=='all' else OUT/'scores'/mode/'COMPLETE.json';r=C.read(done)
    if r.get('status')!='COMPLETE' or r.get('requests')!=12 or r.get('new_scores')!=(48 if mode=='all' else 24):
        raise ValueError('Wrong completion scope')
    C.verify(r['files']);C.verify(r['source_hashes']);C.verify(r['protected'])
    if len(r['metric_fields'])!=177:raise ValueError('Metric schema changed')
    if any(r.get(k)!=0 for k in ('model_calls','D0_calls','GPU_calls','new_noise','new_queries')):
        raise ValueError('Zero-model execution contract violated')
    return r


def run_arm(manifest_path=None,ready_path=None,workers=3,mode='target'):
    if not 1<=int(workers)<=3:raise ValueError('CPU worker cap is3')
    if (OUT/'scores'/mode/'COMPLETE.json').exists():return verify_complete(mode)
    reg=registration(manifest_path,ready_path,mode);_,tasks=tasks_from(reg['manifest_path'])
    rh=C.sha(OUT/'evaluation'/('CPU_'+mode+'_registration.json'));tick=time.perf_counter()
    snap=C.resource_snapshot('before_local_repair_'+mode+'_cpu_dispatch')
    # Prior identical GEO/scoring jobs measured ~156 MiB per process. Retain
    # at least2GiB RAM and a1.2 multiplier; never change scientific settings.
    memory_slots=max(0,int((snap['ram']['available']-2*1024**3)/(1.2*156*1024**2)))
    actual=min(int(workers),int(snap['cpu_count'] or 1),memory_slots)
    if actual<1:raise RuntimeError('Insufficient measured CPU/RAM headroom; no jobs dispatched')
    attempt=OUT/'evaluation'/('cpu_'+mode+'_run_'+str(time.time_ns()))
    C.write(attempt/'attempt.json',dict(registration_sha256=rh,time=time.time(),pid=os.getpid(),
        requested_workers=int(workers),actual_workers=actual,threads_per_worker=1,
        comparable_peak_MiB=156,memory_headroom_bytes=2*1024**3,
        available_RAM_bytes=snap['ram']['available'],cpu_count=snap['cpu_count'],
        zero_GPU=True,scientific_budget=reg['budget']))
    receipts=[];jobs=[(t,rh,reg['source_hashes'],mode) for t in tasks]
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=iter(jobs);active={}
            while True:
                while len(active)<actual:
                    job=next(pending,None)
                    if job is None:break
                    active[pool.submit(one,job)]=job[0]['request_id']
                if not active:break
                ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in ready:
                    rid=active.pop(f)
                    try:receipts.append(f.result())
                    except BaseException:
                        for other in active:other.cancel()
                        raise
        indexed={r['request_id']:r for r in receipts}
        receipts=[indexed[t['request_id']] for t in tasks]
        aggregation=collect(tasks,receipts,rh,mode)
        C.verify(reg['protected']);C.verify(reg['source_hashes'])
        after=C.resource_snapshot('after_local_repair_'+mode+'_cpu_dispatch')
        ledger=dict(status='COMPLETE',requests=12,receipts=receipts,
            actual_workers=actual,threads_per_worker=1,seconds=time.perf_counter()-tick,
            worker_seconds_sum=sum(r['geometry_seconds']+r['cpu_seconds'] for r in receipts),
            peak_worker_RSS_KiB=max(r['peak_RSS_KiB'] for r in receipts),
            GEO_calls=12 if mode=='target' else 0,source_orientation_calls=12 if mode=='target' else 0,
            reused_GEO_results=12 if mode=='identity' else 0,base_scores=12,final_scores=12,
            model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
            replay_policy='Completed producer receipts reused; partial producers stop without replay',
            timing_scope='This invocation wall time; receipt stage sums may include previously completed reused jobs')
        C.write(attempt/'complete.json',ledger)
        ap=OUT/'scores'/mode/'aggregation_complete.json'
        files=dict(aggregation['files']);files[str(ap)]=C.sha(ap)
        files.update({str(p):C.sha(p) for p in attempt.rglob('*') if p.is_file()})
        for receipt in receipts:
            for name in ('geometry_receipt','cpu_receipt'):
                p=receipt[name]
                if p is not None:files[p]=C.sha(p);files.update(C.read(p)['files'])
        value=dict(status='COMPLETE',requests=12,parents=6,new_scores=24,metric_fields=aggregation['metric_fields'],
            candidate_label=LABELS[mode],mode=mode,stages=list(STAGES),workers=actual,threads_per_worker=1,
            seconds=ledger['seconds'],worker_seconds_sum=ledger['worker_seconds_sum'],
            peak_worker_RSS_KiB=ledger['peak_worker_RSS_KiB'],
            GEO_calls=12 if mode=='target' else 0,source_orientation_calls=12 if mode=='target' else 0,
            identity_BASE_exact_checks=12*177 if mode=='identity' else 0,
            model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
            registration_sha256=rh,source_hashes=reg['source_hashes'],protected=reg['protected'],files=files)
        C.write(OUT/'scores'/mode/'COMPLETE.json',value);return value
    except BaseException as exc:
        C.write(attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            returned_complete_request_ids=[r['request_id'] for r in receipts],
            in_flight_tasks_may_have_finished=True,automatic_retry=False,model_calls=0))
        raise


def combine():
    if (OUT/'scores/COMPLETE.json').exists():return verify_complete('all')
    arms={mode:verify_complete(mode) for mode in ('identity','target')}
    attempt=OUT/'scores/aggregation_attempt.json'
    if attempt.exists():raise RuntimeError('Partial combined collection requires explicit recovery')
    C.write(attempt,dict(time=time.time(),scientific_calls=0,identity_plus_target=True))
    fields=arms['identity']['metric_fields']
    if fields!=arms['target']['metric_fields']:raise ValueError('Arm metric schema differs')
    rows=C.read(OUT/'scores/identity/metric_rows.json')+C.read(OUT/'scores/target/metric_rows.json')
    if len(rows)!=48 or len({(r['label'],r['stage'],r['request_id']) for r in rows})!=48:
        raise ValueError('Expected48 unique arm/stage/request rows')
    cond=aggregate(rows,('label','stage','eval_group','parent_id','condition_id'),fields)
    parents=aggregate(cond,('label','stage','eval_group','parent_id'),fields)
    summary=aggregate(parents,('label','stage','eval_group'),fields)
    index={(r['label'],r['stage'],r['request_id']):r for r in rows};paired=[]
    for b in rows:
        if b['label']!=LABEL:continue
        a=index[('RAW_IDENTITY',b['stage'],b['request_id'])]
        p={k:b[k] for k in ('request_id','parent_id','condition_id','noise_index','eval_group','stage')}
        p.update(reference_label='RAW_IDENTITY',candidate_label=LABEL)
        for f in fields:
            av,bv=a[f],b[f];p[f+'__reference']=av;p[f+'__candidate']=bv
            p[f+'__delta']=bv-av if numeric(av) and numeric(bv) else None
        paired.append(p)
    tables=dict(per_output=rows,per_condition=cond,per_parent=parents,
        parent_macro_summary=summary,paired_per_output=paired)
    files={str(attempt):C.sha(attempt)}
    for name,table in tables.items():
        p=OUT/'scores'/(name+'.csv');C.csvwrite(p,table);files[str(p)]=C.sha(p)
    jp=OUT/'scores/metric_rows.json';C.write(jp,rows);files[str(jp)]=C.sha(jp)
    protected={};code={}
    for mode,r in arms.items():
        p=OUT/'scores'/mode/'COMPLETE.json';files[str(p)]=C.sha(p);files.update(r['files'])
        protected.update(r['protected']);code.update(r['source_hashes'])
    C.verify(protected);C.verify(code)
    r=dict(status='COMPLETE',requests=12,parents=6,new_scores=48,paired_rows=24,metric_fields=fields,
        labels=['RAW_IDENTITY',LABEL],stages=list(STAGES),GEO_calls=12,reused_identity_GEO=12,
        cascade_calls=24,base_scores=24,final_scores=24,identity_BASE_exact_checks=12*177,
        model_calls=0,D0_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
        comparison='LOCAL_REPAIR minus RAW_IDENTITY with identical stage and original query/scorer',
        source_hashes=code,protected=protected,files=files)
    C.write(OUT/'scores/COMPLETE.json',r);return r


def run(manifest_path=None,ready_path=None,workers=3,mode='all'):
    if mode in LABELS:return run_arm(manifest_path,ready_path,workers,mode)
    if mode!='all':raise ValueError('Unknown mode')
    run_arm(manifest_path,ready_path,workers,'identity')
    run_arm(manifest_path,ready_path,workers,'target')
    return combine()


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(MANIFEST))
    p.add_argument('--ready',default=str(READY));p.add_argument('--workers',type=int,default=3)
    p.add_argument('--mode',choices=('all','identity','target'),default='all')
    p.add_argument('--verify',action='store_true');a=p.parse_args()
    result=verify_complete(a.mode) if a.verify else run(a.manifest,a.ready,a.workers,a.mode)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected','source_hashes','metric_fields')},ensure_ascii=False))
