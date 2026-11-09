"""Fixed Chair-back panel scoring, using the unchanged HINGE V2 math.

No models are imported. Query points come only from this run's frozen input
manifest. RAW, base POST and all CPU stages retain full original/bridge fields.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,collections,concurrent.futures,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from r2.pilots.seam_abc_v1 import evaluate as E,eval_common as C
from r2.tools.hinge_v2_cpu_v1 import scoring as M

ROOT=C.ROOT/'experiments/r2_chair_back_transfer_v1'
LABEL='S2000'
STAGES=('RAW','BASE_POST','HINGE_OPP015','ALPHA','FINAL_POST')
VERSION='chair_back_frozen_original_metrics_bridge_v1'


def requests(root=ROOT):
    m=C.read(C.resolve(root)/'eval_manifest.json')
    qq=m['requests']
    if len(qq)!=16 or len({q['request_id'] for q in qq})!=16 or len({q['condition_id'] for q in qq})!=8:
        raise ValueError('Expected exactly eight frozen C and sixteen requests')
    return qq


def mesh_path(root,q,stage):
    base=C.resolve(root)/'evaluation'/LABEL/q['request_id']
    if stage=='RAW':return base/'raw/raw.npz'
    if stage=='BASE_POST':return base/'post/post.npz'
    if stage in STAGES:return base/'cpu'/stage/'post.npz'
    raise ValueError('Unregistered stage')


def score_dir(root,q,stage):
    return C.resolve(root)/'evaluation/scores'/LABEL/q['request_id']/stage


def artifact(root,q,stage):
    p=mesh_path(root,q,stage)
    receipt=p.parent/('receipt.json' if stage in ('RAW','BASE_POST') else 'complete.json')
    r=C.read(receipt)
    if r.get('status') not in ('COMPLETE','PASS'):
        raise ValueError('Incomplete producer '+str(receipt))
    if r.get('request_id',r.get('row',{}).get('request_id'))!=q['request_id']:
        raise ValueError('Producer request identity differs')
    digest=r.get('raw_sha256' if stage=='RAW' else 'post_sha256')
    if digest is None:
        hh=[v for k,v in r.get('files',{}).items() if C.resolve(k)==p]
        if len(hh)!=1:raise ValueError('Producer does not authenticate mesh')
        digest=hh[0]
    if C.sha(p)!=digest:raise ValueError('Mesh SHA differs')
    if stage=='RAW' and r.get('generator_forwards')!=50:raise ValueError('Euler50 required')
    if stage=='BASE_POST' and (r.get('D0_forwards')!=1 or r.get('GEO_calls')!=1):
        raise ValueError('Exactly one official D0/GEO required')
    if r.get('files'):E.verify(r['files'])
    return p,receipt


def source_hashes():
    from r2.tools.hinge_v2_cpu_v1 import holes
    from r2.pilots.seam_hinge_quality_v1 import topology
    out=E.source_hashes()
    for mod in (M,holes,topology):out[str(Path(mod.__file__))]=C.sha(mod.__file__)
    out[str(Path(__file__))]=C.sha(__file__)
    return out


def prepare(root=ROOT):
    root=C.resolve(root);dest=root/'evaluation/score_registration.json'
    if dest.exists():
        r=C.read(dest);E.verify(r['protected']);E.verify(r['source_hashes']);return r
    qq=requests(root);old,A,_,_,_=E.helpers();protected={str(root/'eval_manifest.json'):C.sha(root/'eval_manifest.json')}
    audits=[]
    for q in qq:
        inp=C.arrays(q['inference_npz']);meta=C.arrays(q['metadata_npz'])
        cc=inp['C_vertices'][inp['C_faces_model']];original=inp['C_vertices'][inp['C_faces_original']]
        if cc.shape!=(int(q['K']),3,3) or cc.dtype!=np.float32:raise ValueError('C shape/dtype differs')
        if int(inp.get('total_N',inp.get('N')))!=q['N']:raise ValueError('N mismatch')
        desc=q['query_descriptor'];reference=old.load_queries(desc)
        if desc['identity']['C_sha256']!=old.array_sha(cc) or len(reference['arrays']['C_points'])!=2048:
            raise ValueError('Fixed query does not belong to this C')
        _,_,topology=A.topology(original,meta['C_source_vertex_ids'][inp['C_faces_original']])
        if topology['boundary_edge_count']<=0:raise ValueError('This back pilot uses open C only')
        for key in ('inference_npz','metadata_npz','source_npz','noise_npz'):
            protected[q[key]]=C.sha(q[key])
        protected[desc['path']]=desc['sha256']
        audits.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],boundary=topology,query_sha256=desc['sha256']))
    result=dict(status='FROZEN',version=VERSION,label=LABEL,requests=qq,stages=STAGES,
        protected=protected,source_hashes=source_hashes(),L=C.L,C_queries=2048,new_queries=0,
        quality_definition=old.cpu_math()['control'].frozen_definition(),
        input_audits=audits,metric_policy='Original135 plus existing bridge/fixed-parent fields; all fields retained',
        denominator_policy='Actual free count includes bridges; fixed-parent post denominator excludes bridges; RAW fixed parent means RAW free slots',
        semantic_policy='Open-ended completion; no source-style rank; actual nonblind AI_PROXY reviewed separately',
        model_calls=0)
    C.write(dest,result);return result


def pair_keys(pairs):
    return {kind:{(tuple(map(int,r['pair'])),r['classification']) for r in scan['contacts']} for kind,scan in pairs.items()}


def score_one(job):
    root,q,stage,regsha=job;root=C.resolve(root);dest=score_dir(root,q,stage)
    mesh,producer=artifact(root,q,stage)
    identity=dict(version=VERSION,label=LABEL,stage=stage,request_id=q['request_id'],
        mesh_path=str(mesh),mesh_sha256=C.sha(mesh),producer_sha256=C.sha(producer),registration_sha256=regsha)
    done=dest/'audit_receipt.json'
    if done.exists():
        r=C.read(done)
        if r['identity']!=identity:raise ValueError('Completed score identity changed')
        E.verify(r['files']);return r
    if dest.exists():raise RuntimeError('Unresolved score attempt preserved: '+str(dest))
    tick=time.perf_counter();a=C.arrays(mesh)
    # RAW has explicit shared indices but historically no index_domain key.
    # Add a descriptive tag to the scoring copy; do not alter saved geometry.
    if 'index_domain' not in a:a['index_domain']=np.asarray('ACTUAL_RAW_C_SEAM_SHARED_INDEX')
    existing=None
    if stage not in ('RAW','BASE_POST'):
        if not np.any(a.get('face_role',np.asarray([]))=='BRIDGE'):
            existing=C.read(mesh.parent/'pairs.json')
    result=M.score(q,a,dest,stage,existing_pairs=existing)
    if stage not in ('RAW','BASE_POST'):
        incremental=C.read(mesh.parent/'pairs.json')
        if pair_keys(incremental)!=pair_keys(result['pairs']):
            raise AssertionError('CPU incremental pair cache differs from full final score')
    r=dict(status='PASS',identity=identity,model_calls=0,new_queries=0,
        full_pair_scan=existing is None,seconds=time.perf_counter()-tick,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
    if 'torch' in sys.modules:raise RuntimeError('CPU scoring imported torch')
    C.write(done,r);print('SCORE',stage,q['request_id'],flush=True);return r


def worker_count(snapshot,first,maximum=2):
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in (snapshot.get('memory') or '').splitlines()
        if len(line.split())>1 and line.split()[1].isdigit()}
    available=memory.get('MemAvailable',0);cores=snapshot.get('cpu_logical') or 1
    load=(snapshot.get('loadavg') or [cores])[0]
    slots=int(max(0,available-(4<<30))/(1.2*max(1,first['peak_RSS_KiB']*1024)))
    return max(1,min(maximum,slots,max(1,int(cores-load-2))))


def run(root=ROOT,stages=STAGES,workers=2):
    root=C.resolve(root);reg=prepare(root);stages=tuple(stages)
    if any(s not in STAGES for s in stages):raise ValueError('Unregistered stage')
    for q in reg['requests']:
        for stage in stages:artifact(root,q,stage)
    done=root/'evaluation/completions'/('__'.join(stages)+'.json')
    if done.exists():
        saved=C.read(done);E.verify(saved['files']);return saved
    identity=C.sha(root/'evaluation/score_registration.json')
    jobs=[(str(root),q,s,identity) for q in reg['requests'] for s in stages]
    before=root/'evaluation/score_dispatch'/('__'.join(stages)+'_resources_before.json')
    if not before.exists():C.write(before,C.resources())
    tick=time.perf_counter();first=score_one(jobs[0]);snapshot=C.resources();count=worker_count(snapshot,first,workers)
    runpath=root/'evaluation/score_dispatch'/('__'.join(stages)+'.json')
    C.write(runpath,dict(workers=count,threads_per_worker=1,resources=snapshot,first_authorized_score=first['identity'],model_calls=0),replace=runpath.exists())
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*list(pool.map(score_one,jobs[1:]))]
    E.verify(reg['protected']);E.verify(reg['source_hashes'])
    result=dict(status='COMPLETE',stages=stages,rows=len(receipts),workers=count,seconds=time.perf_counter()-tick,
        files={str(score_dir(root,q,s)/'audit_receipt.json'):C.sha(score_dir(root,q,s)/'audit_receipt.json') for q in reg['requests'] for s in stages},model_calls=0)
    C.write(done,result);return result


def number(x):return isinstance(x,(int,float,bool)) and np.isfinite(float(x))


def aggregate(root=ROOT):
    root=C.resolve(root);qq=requests(root);rows=[];sources={}
    for q in qq:
        for stage in STAGES:
            p=score_dir(root,q,stage);r=C.read(p/'audit_receipt.json');E.verify(r['files'])
            row=C.read(p/'row.json');row.update(model=LABEL,stage=stage,part_type='BACK_FULL')
            rows.append(row);sources[str(p/'audit_receipt.json')]=C.sha(p/'audit_receipt.json')
    excluded=set(E.META)|{'label','stage','model','part_type'}
    fields=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in excluded and any(number(r.get(k)) for r in rows))
    # All historic135 remain present, including NA-only bridge/empty sets.
    directions=C.read(C.ROOT/'r2/tools/hinge_opposite_v1/metric_policy.json')['directions']
    fields=sorted(set(fields)|set(directions))
    summary=[];parents=[];conditions=[]
    for stage in STAGES:
        for split in (*sorted({q['split'] for q in qq}),'ALL'):
            rr=[r for r in rows if r['stage']==stage and (split=='ALL' or r['split']==split)]
            if not rr:continue
            for field in fields:
                cc=collections.defaultdict(list)
                for r in rr:cc[(r['parent_id'],r['condition_id'])].append(r.get(field))
                pp=collections.defaultdict(list)
                for (parent,cid),values in cc.items():
                    vv=[float(x) for x in values if number(x)];v=float(np.mean(vv)) if vv else None;pp[parent].append(v)
                    if split!='ALL':conditions.append(dict(stage=stage,split=split,parent_id=parent,condition_id=cid,metric=field,value=v,n_valid=len(vv),n_requests=len(values)))
                pv=[]
                for parent,values in pp.items():
                    vv=[v for v in values if v is not None];v=float(np.mean(vv)) if vv else None
                    if v is not None:pv.append(v)
                    if split!='ALL':parents.append(dict(stage=stage,split=split,parent_id=parent,metric=field,value=v,n_conditions=len(values),n_valid_conditions=len(vv)))
                vv=[float(r[field]) for r in rr if number(r.get(field))]
                summary.append(dict(stage=stage,split=split,metric=field,mean_parent_equal=float(np.mean(pv)) if pv else None,
                    mean_output=float(np.mean(vv)) if vv else None,n_outputs=len(rr),n_valid=len(vv),n_parents=len(pp),n_valid_parents=len(pv)))
    by={(r['request_id'],r['stage']):r for r in rows};paired=[]
    for q in qq:
        for a,b in (('RAW','BASE_POST'),('BASE_POST','HINGE_OPP015'),('HINGE_OPP015','ALPHA'),('ALPHA','FINAL_POST'),('BASE_POST','FINAL_POST')):
            x,y=by[q['request_id'],a],by[q['request_id'],b]
            for field in fields:
                av,bv=x.get(field),y.get(field)
                paired.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],parent_id=q['parent_id'],split=q['split'],
                    baseline=a,candidate=b,metric=field,baseline_value=av,candidate_value=bv,delta=float(bv)-float(av) if number(av) and number(bv) else None,
                    status='PASS' if number(av) and number(bv) else 'NA_NOT_COMMON'))
    out=root/'evaluation/tables';files={}
    for name,value in (('per_output',rows),('summary_parent_equal',summary),('per_condition',conditions),('per_parent',parents),('paired_per_output',paired)):
        p=out/(name+'.csv');C.csvwrite(p,value);files[str(p)]=C.sha(p)
    result=dict(status='COMPLETE',rows=len(rows),stages=STAGES,numeric_fields=fields,numeric_field_count=len(fields),
        source_receipts=sources,files=files,aggregation='noise -> condition -> parent equal; original FAMILIAR/DEV labels preserved',model_calls=0,new_scans=0)
    C.write(out/'completion.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','score','aggregate'))
    p.add_argument('--root',default=str(ROOT));p.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES);p.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    a=p.parse_args();r=prepare(a.root) if a.action=='prepare' else aggregate(a.root) if a.action=='aggregate' else run(a.root,a.stages,a.workers)
    print(json.dumps({k:v for k,v in r.items() if k not in ('requests','files','protected','source_hashes','source_receipts','numeric_fields')}),flush=True)
