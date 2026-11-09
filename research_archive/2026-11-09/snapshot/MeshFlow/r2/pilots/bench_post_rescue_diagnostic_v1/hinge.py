"""Frozen Bench saved-output HINGE_OPP015 diagnostic; CPU only.

The original decoder and original Bench quality scorer are reused unchanged.
No model import, new noise/query, ALPHA, HOLE, threshold sweep or face deletion.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse,collections,concurrent.futures,json,multiprocessing,resource,sys,time,traceback
from pathlib import Path
import numpy as np
from r2.pilots.bench_parts_convergence_v1 import evaluation as B
from r2.pilots.seam_abc_v1 import evaluate as E,eval_common as C
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
from r2.pilots.seam_hinge_quality_v1 import topology as T
from r2.tools.hinge_opposite_v1 import engine as H

SOURCE=C.ROOT/'experiments/r2_bench_parts_convergence_v1'
OUT=C.ROOT/'experiments/r2_bench_post_rescue_diagnostic_v1/hinge'
LABELS=('SEAM_S15000','SEAM_S21000')
VERSION='bench_saved_post_hinge_opposite_015_original_scorer_v1'


def mesh_dir(label,rid):return OUT/'evaluation'/label/rid/'post'
def score_dir(label,rid):return OUT/'evaluation/scores'/label/rid/'POST'
def old_score_dir(label,rid):return SOURCE/'evaluation/scores'/label/rid/'POST'


def source_hashes():
    out={**B.source_hashes(),**H.Q.definition_hashes()}
    for p in (H.__file__,T.__file__,__file__):out[str(Path(p))]=C.sha(p)
    return out


def freeze():
    path=OUT/'registration.json'
    if path.exists():
        r=C.read(path);E.verify(r['protected']);E.verify(r['source_hashes']);return r
    jobs=[];protected={};panels={};inputs={}
    for label in LABELS:
        rp=SOURCE/'evaluation/score_registrations'/(label+'.json');reg=C.read(rp)
        if not reg['seam_model']:raise ValueError('Requires SEAM representation')
        protected[str(rp)]=C.sha(rp)
        qq={q['request_id']:q for q in reg['requests']};actual=[]
        for d in sorted((SOURCE/'evaluation'/label).iterdir()):
            if not (d/'post/receipt.json').exists():continue
            q=qq[d.name];base,br=B.artifact(SOURCE,label,d.name,'POST');raw,rr=B.artifact(SOURCE,label,d.name,'RAW')
            sp=old_score_dir(label,d.name)/'score.json';score=C.read(sp);E.verify(score['files'])
            if score['identity']['mesh_sha256']!=C.sha(base):raise ValueError('Baseline score geometry differs')
            paths=[base,br,raw,rr,base.parent/'pre_geo.npz',sp,*map(Path,score['files'])]
            for k in ('inference_npz','metadata_npz','source_npz','noise_npz'):paths.append(C.resolve(q[k]))
            paths.append(C.resolve(q['query_descriptor']['path']))
            for p in paths:protected[str(p)]=C.sha(p)
            key=q['request_id'];identity={k:q[k] for k in ('condition_id','N','K','seed','noise_sha256','inference_npz','metadata_npz')}
            if key in inputs and inputs[key]!=identity:raise ValueError('Same ID has different input/noise')
            inputs[key]=identity;actual.append(key);jobs.append(dict(label=label,q=q))
        panels[label]=actual
    common=sorted(set.intersection(*(set(x) for x in panels.values())))
    r=dict(status='FROZEN',version=VERSION,source_root=str(SOURCE),labels=LABELS,jobs=jobs,
        panels=panels,common_request_ids=common,panel_differences={a:sorted(set(panels[a])-set(common)) for a in LABELS},
        source_hashes=source_hashes(),protected=protected,L=C.L,delta_fraction=.015,
        require_opposite_winding=True,new_model_calls=0,new_noise=0,new_queries=0,
        prior_route='Chair S2000 HINGE_OPP015; bounded positive, zero additional illegal pairs and preserved original legal edges; now test fixed Bench saved outputs without algorithm changes',
        decoder='Original engine.decode, existing real GEO clusters; no ALPHA/HOLE, no new threshold, no reorientation or deletion',
        score='Original Bench private_score/E.score_one and metric_row; same frozen 2048 queries, units and pair detector',
        closed_C='Original closed-C seam and boundary scores NA; free quality and generic fragmentation applicable',
        interpretation='Exact common-request pairing; saved endpoint comparison is descriptive, no checkpoint selection or unique GT ranking',
        resources=C.resources())
    C.write(path,r);return r


def save_npz(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    t=p.with_suffix('.npz.partial')
    with t.open('xb') as f:np.savez_compressed(f,**a);f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def candidate_artifact(root,label,rid,stage):
    if stage!='POST':raise ValueError('Only saved POST diagnostic is authorized')
    p=mesh_dir(label,rid)/'post.npz';r=mesh_dir(label,rid)/'receipt.json';x=C.read(r)
    if x['status']!='COMPLETE' or x['label']!=label or x['request_id']!=rid:raise ValueError('Candidate identity differs')
    E.verify(x['files']);return p,r


def pair_keys(pairs):
    return {k:{(tuple(r['pair']),r['classification']) for r in s['contacts']} for k,s in pairs.items()}


def work(job):
    label,q,regsha=job;rid=q['request_id'];dest=mesh_dir(label,rid);done=dest/'receipt.json'
    basep=SOURCE/'evaluation'/label/rid/'post/post.npz'
    prepath=basep.parent/'pre_geo.npz';rawpath=basep.parent.parent/'raw/raw.npz'
    pairpath=old_score_dir(label,rid)/'pairs.json'
    identity=dict(label=label,request_id=rid,registration_sha256=regsha,base_sha256=C.sha(basep),
        pre_geo_sha256=C.sha(prepath),raw_sha256=C.sha(rawpath),pairs_sha256=C.sha(pairpath))
    start=time.perf_counter()
    if done.exists():
        rec=C.read(done)
        if rec['identity']!=identity:raise ValueError('Completed geometry identity differs')
        E.verify(rec['files'])
    else:
        if dest.exists():raise RuntimeError('Partial attempt preserved: '+str(dest))
        C.write(dest/'started.json',dict(identity=identity,utc=C.utc(),pid=os.getpid()))
        try:
            base=C.arrays(basep);pre=C.arrays(prepath)['output'];raw=C.arrays(rawpath)['output'];pairs=C.read(pairpath)
            if pre.shape!=raw.shape or pre.shape!=(q['N'],3,3):raise ValueError('PRE/RAW face domain differs')
            out,records,summary,before,after,newpairs=H.decode(base,pre,raw,pairs,.015,require_opposite_winding=True)
            if not np.array_equal(out['output'][:q['K']].view(np.uint32),base['output'][:q['K']].view(np.uint32)):raise AssertionError('C changed')
            paired=H.Q.compare_ledgers(before,after,pairs,newpairs,summary['affected_source_raw_slots'])
            for k in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                if paired['summary'][k]:raise AssertionError('Frozen guard violated '+k)
            save_npz(dest/'post.npz',out)
            for name,data in (('pairs',newpairs),('summary',summary),('candidates',records),('baseline_connectivity',before),('connectivity',after),('paired_ledger',paired)):
                C.write(dest/(name+'.json'),data)
            C.csvwrite(dest/'candidates.csv',records)
            rec=dict(status='COMPLETE',identity=identity,label=label,request_id=rid,post_sha256=C.sha(dest/'post.npz'),
                wall_seconds=time.perf_counter()-start,model_calls=0,D0_calls=0,new_noise=0,summary=summary,
                files={str(p):C.sha(p) for p in dest.iterdir() if p.is_file()})
            C.write(done,rec)
        except BaseException as exc:
            C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),model_calls=0));raise
    # A private routing namespace retains the original numerical scorer and closed-C NA.
    api=B.private_score(q,label,True);api['artifact']=candidate_artifact
    score_receipt=api['score_one']((str(OUT),label,q,'POST',regsha))
    scored=C.read(score_dir(label,rid)/'pairs.json')
    if pair_keys(scored)!=pair_keys(C.read(dest/'pairs.json')):raise AssertionError('Full original score differs from decoder pair cache')
    if 'torch' in sys.modules:raise RuntimeError('CPU diagnostic imported neural framework')
    result=dict(label=label,request_id=rid,geometry_receipt=str(done),geometry_receipt_sha256=C.sha(done),
        score=score_receipt,wall_seconds=time.perf_counter()-start,geometry_seconds=rec['wall_seconds'],
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0)
    print('HINGE_COMPLETE',label,rid,rec['summary']['accepted'],flush=True);return result


def number(x):return isinstance(x,(int,float,bool)) and np.isfinite(float(x))


def aggregate(reg):
    rows=[];ledgers=[];sources={};roles=[]
    for job in reg['jobs']:
        label,q=job['label'],job['q'];rid=q['request_id']
        for stage in ('BASE_POST','HINGE_OPP015'):
            sp=(old_score_dir(label,rid) if stage=='BASE_POST' else score_dir(label,rid))/'score.json'
            s=C.read(sp);E.verify(s['files']);mesh=C.arrays(s['identity']['mesh_path']);pairs=C.read(sp.parent/'pairs.json')
            row,role,_=metric_row(s,mesh,pairs);diag,faces=T.measure(mesh,q['K'],T.edges_file(sp.parent/'valid_edges.csv'),C.L)
            row.update(diag,stage=stage,label=label,part_type=q['part_type'],part_family=q['part_family'],closed_C=q['closed_C'],boundary_edges=q['boundary_edges'],
                seam_applicability=s['seam_to_bulk']['status'],graph_scope='generic exact-index C connectivity is separate from seam_tip-only seed analysis')
            rows.append(row);roles.extend(dict(r,stage=stage) for r in role);sources[str(sp)]=C.sha(sp)
            # All graph data remains descriptive; seam-specific metrics stay NA for closed C.
            C.csvwrite(OUT/'topology'/label/rid/(stage+'_per_face.csv'),faces)
        lr=C.read(mesh_dir(label,rid)/'paired_ledger.json')['summary'];summ=C.read(mesh_dir(label,rid)/'summary.json')
        ledgers.append(dict(label=label,request_id=rid,parent_id=q['parent_id'],condition_id=q['condition_id'],split=q['split'],part_type=q['part_type'],closed_C=q['closed_C'],accepted=summ['accepted'],**lr))
    meta=set(E.META)|{'stage','label','part_type','part_family','closed_C','boundary_edges','seam_applicability','graph_scope'}
    fields=sorted(k for k in set().union(*(r.keys() for r in rows)) if k not in meta and any(number(r.get(k)) for r in rows))
    lookup={(r['label'],r['request_id'],r['stage']):r for r in rows};paired=[]
    for job in reg['jobs']:
        label,q=job['label'],job['q'];a=lookup[label,q['request_id'],'BASE_POST'];b=lookup[label,q['request_id'],'HINGE_OPP015']
        for field in fields:
            x,y=a.get(field),b.get(field);paired.append(dict(label=label,request_id=q['request_id'],parent_id=q['parent_id'],condition_id=q['condition_id'],split=q['split'],part_type=q['part_type'],metric=field,baseline=x,candidate=y,delta=float(y)-float(x) if number(x) and number(y) else None,status='PASS' if number(x) and number(y) else 'NA'))
    summaries=[]
    for label in LABELS:
        for stage in ('BASE_POST','HINGE_OPP015'):
            rr=[r for r in rows if r['label']==label and r['stage']==stage]
            for dim,groups in (('split',('ALL','TRAIN','DEV')),('part_type',sorted({r['part_type'] for r in rr}))):
                for group in groups:
                    xx=[r for r in rr if group=='ALL' or r[dim]==group]
                    if not xx:continue
                    for field in fields:
                        cc=collections.defaultdict(list)
                        for r in xx:cc[r['parent_id'],r['condition_id']].append(r.get(field))
                        pp=collections.defaultdict(list)
                        for (parent,cid),values in cc.items():
                            vv=[float(v) for v in values if number(v)]
                            if vv:pp[parent].append(float(np.mean(vv)))
                        vv=[float(r[field]) for r in xx if number(r.get(field))]
                        summaries.append(dict(label=label,stage=stage,grouping=dim,group=group,metric=field,
                            mean_parent_equal=float(np.mean([np.mean(v) for v in pp.values()])) if pp else None,
                            mean_output=float(np.mean(vv)) if vv else None,n_requests=len(xx),n_valid=len(vv),n_valid_parents=len(pp),
                            closed_C_requests=sum(bool(r['closed_C']) for r in xx)))
    cross=[]
    for rid in reg['common_request_ids']:
        for stage in ('BASE_POST','HINGE_OPP015'):
            a,b=lookup[LABELS[0],rid,stage],lookup[LABELS[1],rid,stage]
            for field in fields:
                x,y=a.get(field),b.get(field);cross.append(dict(request_id=rid,condition_id=a['condition_id'],parent_id=a['parent_id'],split=a['split'],stage=stage,metric=field,step15000=x,step21000=y,delta=float(y)-float(x) if number(x) and number(y) else None))
    files={}
    for name,data in (('per_output',rows),('paired_per_output',paired),('summary_parent_equal',summaries),('connectivity_ledger',ledgers),('common_request_checkpoint_pairing',cross),('illegal_pairs_by_role',roles)):
        p=OUT/'tables'/(name+'.csv');C.csvwrite(p,data);files[str(p)]=C.sha(p)
    r=dict(status='COMPLETE',rows=len(rows),candidate_cells=len(reg['jobs']),numeric_fields=fields,numeric_field_count=len(fields),
        original_124_fields_plus_generic_indexed_graph11=True,source_score_hashes=sources,files=files,model_calls=0,new_queries=0,
        same_panel=all(not x for x in reg['panel_differences'].values()),common_requests=len(reg['common_request_ids']))
    C.write(OUT/'tables/completion.json',r);return r


def run():
    reg=freeze();done=OUT/'completion.json'
    if done.exists():
        r=C.read(done);E.verify(r['files']);return r
    path=OUT/'registration.json';jobs=[(j['label'],j['q'],C.sha(path)) for j in reg['jobs']]
    start=time.perf_counter();first=work(jobs[0]);snapshot=C.resources()
    mem={x.split(':')[0]:int(x.split()[1])*1024 for x in snapshot['memory'].splitlines() if len(x.split())>1 and x.split()[1].isdigit()}
    workers=2 if mem.get('MemAvailable',0)>max(4<<30,3*first['peak_RSS_KiB']*1024) and snapshot['loadavg'][0]<max(1,(os.cpu_count() or 1)-2) else 1
    C.write(OUT/'dispatch.json',dict(workers=workers,threads_per_worker=1,resources=snapshot,first_authorized_cell=first))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first,*list(pool.map(work,jobs[1:]))]
    result=aggregate(reg);E.verify(reg['protected']);E.verify(reg['source_hashes'])
    C.write(OUT/'execution_cells.json',receipts)
    files={str(OUT/'execution_cells.json'):C.sha(OUT/'execution_cells.json'),str(OUT/'tables/completion.json'):C.sha(OUT/'tables/completion.json')}
    for x in receipts:files[x['geometry_receipt']]=x['geometry_receipt_sha256'];files[x['score']['path']]=x['score']['sha256']
    r=dict(status='COMPLETE',cells=len(receipts),full_original_candidate_scores=len(receipts),baseline_scores_reused=len(receipts),workers=workers,
        wall_seconds=time.perf_counter()-start,files=files,model_calls=0,GPU_calls=0,new_noise=0,new_queries=0,
        source_assets_verified=len(reg['protected']),summary=result['files'])
    C.write(done,r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','aggregate'),default='run',nargs='?');a=p.parse_args()
    r=freeze() if a.action=='prepare' else aggregate(freeze()) if a.action=='aggregate' else run()
    print(json.dumps({k:v for k,v in r.items() if k not in ('jobs','protected','source_hashes','files','source_score_hashes','numeric_fields','resources')},ensure_ascii=False),flush=True)
