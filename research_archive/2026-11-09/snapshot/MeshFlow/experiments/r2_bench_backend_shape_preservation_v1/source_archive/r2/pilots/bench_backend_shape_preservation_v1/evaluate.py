"""CPU-only fixed candidate D0 proposal -> original geometry/cascade/full scores.

No neural imports or model calls. Each geometry and cascade task has a checked
receipt; incomplete attempts stop instead of being silently recomputed.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse, concurrent.futures, json, multiprocessing, resource, sys, time, traceback
from pathlib import Path
import numpy as np
from . import common as C
from r2.pilots.bench_backend_counterfactual_v1 import core as B
from r2.pilots.bench_clean_multipart_seam_v1 import post_cpu as CPU
from r2.pilots.bench_geo_only_acceptance_v1 import evaluate as OLD_E
from r2.pilots.bench_adaptation_strength_v1.collect_v2 import aggregate, numeric, csvwrite

ROOT,OUT,OLD=C.ROOT,C.OUT,C.OLD
read,resolve,sha,write_new,arrays,verify=C.read,C.resolve,C.sha,C.write,C.arrays,C.verify
LABEL='SHAPE_D0'
VERSION='bench_shape_preservation_original_geometry_cascade177_v1'
METRIC_SOURCE=ROOT/'experiments/r2_bench_seat_coverage_v1/results/completion.json'

def tasks_from(path=None):
    p=resolve(path or OLD/'manifest.json');m=read(p);tasks=m['tasks']
    if len(tasks)!=30 or len({t['request_id'] for t in tasks})!=30:
        raise ValueError('Exactly the original30 fixed requests are required')
    if any(t['label']!='COVERAGE_FINAL' for t in tasks):raise ValueError('Only fixed Coverage RAWs are authorized')
    return m,tasks

def locations(t):
    rid=t['request_id'];g=OUT/'geometry'/rid;p=OUT/'cpu'/LABEL/rid
    return dict(proposal=OUT/'evaluation/CANDIDATE_RAW'/rid/'proposal.npz',geometry=g,
        pre=g/'pre_geo.npz',base=g/'post.npz',pairs=g/'pairs.json',cpu=p,
        final=p/'CLEAN_FINAL_POST/post.npz')

def source_hashes():
    _,_,_,code=B.dependencies();code.update(OLD_E.source_hashes())
    code[str(Path(__file__))]=sha(__file__)
    code[str(Path(C.__file__))]=sha(C.__file__)
    return code

def register(manifest_path=None):
    mp=resolve(manifest_path or OLD/'manifest.json');m,tasks=tasks_from(mp)
    protected={str(mp):sha(mp),str(OUT/'PROTOCOL.md'):sha(OUT/'PROTOCOL.md')}
    jobs=[]
    for t in tasks:
        loc=locations(t)
        for p in (t['new_raw_path'],t['source_condition_path'],t['request']['inference_npz'],
                  t['request']['query_descriptor']['path'],loc['proposal']):protected[str(resolve(p))]=sha(p)
        # The model runner owns proposal provenance. Store any supplied receipt
        # without assuming its schema or inventing a new model invocation.
        for filename in ('complete.json','proposal_receipt.json','receipt.json'):
            rp=loc['proposal'].parent/filename
            if rp.exists():
                rr=read(rp)
                if 'files' in rr:verify(rr['files'])
                protected[str(rp)]=sha(rp)
        jobs.append(dict(request_id=t['request_id'],proposal_sha256=sha(loc['proposal']),
                         raw_sha256=sha(t['new_raw_path'])))
    value=dict(status='FROZEN',version=VERSION,manifest_path=str(mp),manifest_sha256=sha(mp),
        candidate_label=LABEL,jobs=jobs,protected=protected,source_hashes=source_hashes(),
        budget=dict(GEO_calls=30,source_orientation=30,base_scores=30,final_scores=30,
                    model_calls=0,new_noise=0,new_queries=0),
        chain='candidate proposal -> restore C -> original GEO_C/source winding -> HINGE_OPP015 -> ALPHA -> HOLE')
    dest=OUT/'evaluation/CPU_registration.json'
    if dest.exists():
        old=read(dest)
        if old!=value:raise ValueError('Frozen CPU registration changed')
        verify(old['protected']);verify(old['source_hashes']);return old
    write_new(dest,value);return value

def geometry_one(t,registration_sha,code):
    loc=locations(t);dest=loc['geometry'];done=dest/'complete.json'
    identity=dict(version=VERSION,registration_sha256=registration_sha,request_id=t['request_id'],
        proposal_sha256=sha(loc['proposal']),raw_sha256=sha(t['new_raw_path']),
        source_condition_sha256=sha(t['source_condition_path']),source_hashes=code)
    if done.exists():
        r=read(done)
        if r['identity']!=identity:raise ValueError('Completed geometry identity differs')
        verify(r['files']);return r
    if dest.exists():raise RuntimeError('Incomplete geometry retained; no automatic replay: '+str(dest))
    dest.mkdir(parents=True);write_new(dest/'attempt.json',dict(identity=identity,time=time.time(),model_calls=0))
    start=time.perf_counter();calls=0
    try:
        P,geo,io,_=B.dependencies();raw=arrays(t['new_raw_path']);source=arrays(t['source_condition_path'])
        prop=arrays(loc['proposal'])['official_unconstrained_proposal'];k,n=int(t['K']),int(t['N'])
        if raw['raw'].shape!=(n,3,3) or raw['raw_seam_slot_mask'].any():raise ValueError('Expected frozen Native RAW')
        before=raw['raw'].tobytes()
        pre,canonical,post,g,orient,anchors=P.geometry_from_proposal(raw,prop,source);calls=1
        if raw['raw'].tobytes()!=before or not np.array_equal(pre[k:],prop[k:]):raise ValueError('Free proposal changed during C restoration')
        if not np.array_equal(pre[:k],raw['raw'][:k]) or g['stats']['radius_model']!=geo.RADIUS:raise ValueError('C/geometry contract changed')
        C.npz(loc['pre'],dict(output=pre,C=raw['raw'][:k],known_faces=raw['known_faces'],
            raw_anchor_ids=raw['raw_anchor_ids'],raw_seam_slot_mask=raw['raw_seam_slot_mask']))
        C.npz(dest/'post_canonical.npz',canonical);C.npz(loc['base'],post)
        io.write_indexed_ply(dest/'post.ply',post['vertices'],post['faces'])
        write_new(dest/'geometry.json',dict(stats=g['stats'],face_records=g['face_records']))
        write_new(dest/'anchor_identity.json',anchors);write_new(dest/'source_restoration.json',orient['restoration_summary'])
        o=orient['orientation_result'];write_new(dest/'orientation.json',dict(summary=o['summary'],components=o['component_rows'],edges=o['edge_rows']))
        # Exact original detector cache, consumed by the original BASE scorer.
        write_new(loc['pairs'],CPU.H.scan(post['output'],k))
        face_rows=[];stage_rows=[];ids=np.arange(n)
        for name,soup,slots in (('RAW',raw['raw'],ids),('CANDIDATE_PROPOSAL',prop,ids),
                ('CANDIDATE_RESTORED_C',pre,ids),('CANDIDATE_GEO_MERGED',B.merged(pre,canonical),ids),
                ('CANDIDATE_BASE',canonical['output'],canonical['source_face_ids'])):
            stage_task=dict(t,label=LABEL,kind='CANDIDATE_D0_GEO',job_id=LABEL+'__'+t['request_id'])
            rr,ss=B.stage_records(stage_task,name,soup,slots,canonical['source_face_kept'],canonical['source_face_reason'])
            for row in rr:row['eval_group']=t['eval_group']
            ss['eval_group']=t['eval_group'];face_rows.extend(rr);stage_rows.append(ss)
        csvwrite(dest/'per_face_stages.csv',face_rows);csvwrite(dest/'stage_summary.csv',stage_rows)
        if 'torch' in sys.modules:raise RuntimeError('CPU geometry imported torch')
        r=dict(status='COMPLETE',identity=identity,GEO_calls=1,source_orientation_calls=1,model_calls=0,
            seconds=time.perf_counter()-start,files={str(p):sha(p) for p in dest.rglob('*') if p.is_file()})
        write_new(done,r);return r
    except BaseException as exc:
        write_new(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),GEO_returned=calls,automatic_retry=False));raise

def one(job):
    t,rh,code=job;loc=locations(t);g=geometry_one(t,rh,code);start=time.perf_counter()
    result=CPU.process_one(t['request'],LABEL,loc['base'],loc['pre'],t['new_raw_path'],loc['pairs'],loc['cpu'],
                           seam_model=False,registration_sha=rh)
    rows=read(loc['cpu']/'metric_rows.json');fields=read(METRIC_SOURCE)['metric_fields']
    if len(rows)!=2 or len(fields)!=177:raise ValueError('Expected BASE+FINAL with original177 fields')
    for row in rows:
        if set(fields)-set(row):raise ValueError('Incomplete original metric set')
        if row['seam_applicability']!='NA_NOT_SEAM_REPRESENTATION':raise ValueError('Wrong Native metric applicability')
    return dict(request_id=t['request_id'],geometry_receipt=str(loc['geometry']/'complete.json'),
        cpu_receipt=str(loc['cpu']/'complete.json'),rows=rows,geometry_seconds=g['seconds'],
        cpu_seconds=result['seconds'],peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        observed_call_seconds=time.perf_counter()-start)

def references(tasks):
    refs,files=OLD_E.reference_rows(tasks)
    for t in tasks:
        p=OLD/'scores'/t['request_id']/'complete.json';r=read(p);verify(r['files'])
        if r['row']['stage']!='GEO_ONLY' or r['row']['request_id']!=t['request_id']:raise ValueError('Wrong GEO_ONLY reference')
        refs.append(dict(r['row'],result_origin='READ_ONLY_ORIGINAL_GEO_ONLY_SCORE'))
        files[str(p)]=sha(p)
    if len(refs)!=210:raise ValueError('Expected210 original scores')
    return refs,files

def collect(tasks,receipts):
    refs,sources=references(tasks);fields=read(METRIC_SOURCE)['metric_fields'];new=[]
    for t,r in zip(tasks,receipts):
        if t['request_id']!=r['request_id']:raise ValueError('Non-deterministic result order')
        for row in r['rows']:new.append(OLD_E.decorate(row,t,LABEL,row['stage'],'NEW_CANDIDATE_ORIGINAL_FULL_SCORE'))
    rows=refs+new
    cond=aggregate(rows,('label','stage','eval_group','parent_id','condition_id'),fields)
    parents=aggregate(cond,('label','stage','eval_group','parent_id'),fields)
    summary=aggregate(parents,('label','stage','eval_group'),fields)
    by={(r['label'],r['stage'],r['request_id']):r for r in rows};pairs=[]
    for t in tasks:
        for stage in ('BASE_POST','CLEAN_FINAL_POST'):
            b=by[(LABEL,stage,t['request_id'])]
            for alabel,astage in (('COVERAGE_FINAL','RAW'),('COVERAGE_FINAL','BASE_POST'),
                    ('COVERAGE_FINAL','CLEAN_FINAL_POST'),('COVERAGE_FINAL','GEO_ONLY'),('NATIVE_S6000','CLEAN_FINAL_POST')):
                a=by[(alabel,astage,t['request_id'])]
                row=dict(reference_label=alabel,reference_stage=astage,candidate_label=LABEL,candidate_stage=stage,
                    **{k:b[k] for k in ('request_id','condition_id','parent_id','noise_index','eval_group')})
                for f in fields:
                    av,bv=a.get(f),b.get(f);row[f+'__reference']=av;row[f+'__candidate']=bv
                    row[f+'__delta']=bv-av if numeric(av) and numeric(bv) else None
                pairs.append(row)
    for name,table in dict(per_output=rows,new_per_output=new,original_reference_scores=refs,per_condition=cond,
            per_parent=parents,parent_macro_summary=summary,paired_per_output=pairs).items():
        csvwrite(OUT/'scores'/(name+'.csv'),table)
    return sources,fields

def run(manifest_path=None,workers=3):
    if not 1<=int(workers)<=3:raise ValueError('Registered CPU worker cap is3')
    done=OUT/'scores/COMPLETE.json'
    if done.exists():
        r=read(done);verify(r['files']);verify(r['source_hashes']);verify(r['reference_hashes']);return r
    reg=register(manifest_path);_,tasks=tasks_from(reg['manifest_path']);rh=sha(OUT/'evaluation/CPU_registration.json')
    start=time.perf_counter();jobs=[(t,rh,reg['source_hashes']) for t in tasks]
    # First real authorized request measures CPU workspaces; root owns resource GO.
    first=one(jobs[0]);receipts=[first];dp=OUT/'scores/dispatch.json'
    if not dp.exists():write_new(dp,dict(workers=int(workers),threads_per_worker=1,first_RSS_KiB=first['peak_RSS_KiB'],
                                       resource_authorization='root run-stage resource measurement',time=time.time()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers),mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=iter(jobs[1:]);active={}
        while True:
            while len(active)<int(workers):
                job=next(pending,None)
                if job is None:break
                active[pool.submit(one,job)]=job
            if not active:break
            ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in ready:
                active.pop(f)
                try:receipts.append(f.result())
                except BaseException:
                    for other in active:other.cancel()
                    raise
    indexed={r['request_id']:r for r in receipts};receipts=[indexed[t['request_id']] for t in tasks]
    sources,fields=collect(tasks,receipts);verify(reg['protected']);verify(reg['source_hashes'])
    files={str(p):sha(p) for p in (OUT/'scores').rglob('*') if p.is_file()}
    for r in receipts:
        for key in ('geometry_receipt','cpu_receipt'):
            files[r[key]]=sha(r[key]);files.update(read(r[key])['files'])
    value=dict(status='COMPLETE',requests=30,GEO_calls=30,new_scores=60,reference_scores=210,metric_fields=fields,
        candidate_label=LABEL,stages=['BASE_POST','CLEAN_FINAL_POST'],workers=int(workers),threads_per_worker=1,
        seconds=time.perf_counter()-start,worker_seconds_sum=sum(r['geometry_seconds']+r['cpu_seconds'] for r in receipts),
        peak_worker_RSS_KiB=max(r['peak_RSS_KiB'] for r in receipts),model_calls=0,new_noise=0,new_queries=0,
        registration_sha256=rh,source_hashes=reg['source_hashes'],reference_hashes=sources,files=files)
    write_new(done,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(OLD/'manifest.json'));p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();r=run(a.manifest,a.workers);print(json.dumps({k:v for k,v in r.items() if k not in ('files','source_hashes','reference_hashes','metric_fields')},ensure_ascii=False))
