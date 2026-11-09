"""Resumable 140-cell frozen cache replay; no neural framework or new randomness."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,collections,concurrent.futures,inspect,json,multiprocessing,resource,sys,time,traceback,types
from pathlib import Path
import numpy as np
from . import pipeline as P,floor,guard_geo,test_geometry
from r2.pilots.seam_abc_v1 import evaluate as E,eval_common as C
from r2.pilots.chair_back_transfer_v1.post_cpu import save_npz
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
from r2.pilots.seam_pool47_couple_v1.endpoint_scoring import metric_row
from r2.pilots.seam_hinge_quality_v1 import topology as T
from r2.tools.hinge_v2_cpu_v1 import scoring as M
H=P.H
ROOT=C.ROOT/'experiments/r2_unified_rescue_post_v1'
STAGES=('GUARD_GEO','HINGE_OPP015','ALPHA','FINAL')

def active_contract(reg=None):
    reg=C.read(ROOT/'registration.json') if reg is None else reg
    path=ROOT/'execution_revision_02.json'
    if not path.exists():return dict(revision='CORRECTED_GEO_V1',path=str(ROOT/'registration.json'),sha256=C.sha(ROOT/'registration.json'),code=reg['code'])
    amendment=C.read(path)
    if amendment['original_registration_sha256']!=C.sha(ROOT/'registration.json'):raise ValueError('Original registration changed')
    return dict(revision=amendment['revision'],path=str(path),sha256=C.sha(path),code=amendment['code'],prior_code_snapshots=amendment['prior_code_snapshots'])

def verify_code(reg):
    contract=active_contract(reg);E.verify(contract['code'])
    if 'prior_code_snapshots' in contract:E.verify(contract['prior_code_snapshots'])
    else:E.verify(reg['code'])
    return contract

def describe_sources():
    jobs=[]
    bench=C.ROOT/'experiments/r2_bench_parts_convergence_v1'
    for label in ('SEAM_S15000','SEAM_S21000'):
        manifest=bench/'evaluation/score_registrations'/(label+'.json')
        for q in C.read(manifest)['requests']:
            base=bench/'evaluation'/label/q['request_id'];score=bench/'evaluation/scores'/label/q['request_id']/'POST'
            end=C.ROOT/'experiments/r2_bench_post_rescue_diagnostic_v1/hinge/evaluation'/label/q['request_id']/'post'
            es=C.ROOT/'experiments/r2_bench_post_rescue_diagnostic_v1/hinge/evaluation/scores'/label/q['request_id']/'POST'
            jobs.append(dict(panel='BENCH_'+label,q=q,manifest=str(manifest),base_dir=str(base),old_base=str(base/'post/post.npz'),
                old_base_score=str(score),old_final=str(end/'post.npz'),old_final_score=str(es),old_final_chain='OLD_GEO_C + HINGE_OPP015_ONLY'))
    back=C.ROOT/'experiments/r2_chair_back_transfer_v1';manifest=back/'eval_manifest.json'
    for q in C.read(manifest)['requests']:
        base=back/'evaluation/S2000'/q['request_id'];score=back/'evaluation/scores/S2000'/q['request_id']
        jobs.append(dict(panel='CHAIR_BACK_S2000',q=q,manifest=str(manifest),base_dir=str(base),old_base=str(base/'post/post.npz'),
            old_base_score=str(score/'BASE_POST'),old_final=str(base/'cpu/FINAL_POST/post.npz'),old_final_score=str(score/'FINAL_POST'),old_final_chain='OLD_GEO_C + HINGE_OPP015 + ALPHA + HOLE'))
    seat=C.ROOT/'experiments/r2_seam_quality_wave1_wave2_v1';manifest=seat/'protocol.json'
    for q in C.read(manifest)['evaluation']['requests']:
        base=seat/'evaluation/PLAIN_S2000'/q['request_id'];end=C.ROOT/'experiments/r2_hinge_v2_cpu_v1'
        jobs.append(dict(panel='CHAIR_SEAT_S2000',q=q,manifest=str(manifest),base_dir=str(base),old_base=str(base/'post/post.npz'),
            old_base_score=str(seat/'evaluation/scores/PLAIN_S2000'/q['request_id']/'POST'),old_final=str(end/'cells/ALPHA_HOLE'/q['request_id']/'post.npz'),
            old_final_score=str(end/'scores/ALPHA_HOLE'/q['request_id']),old_final_chain='OLD_GEO_C + HINGE_OPP015 + ALPHA + HOLE'))
    if len(jobs)!=140 or collections.Counter(j['panel'] for j in jobs)!=dict(BENCH_SEAM_S15000=48,BENCH_SEAM_S21000=48,CHAIR_BACK_S2000=16,CHAIR_SEAT_S2000=28):raise ValueError('Fixed panel mismatch')
    return jobs

def freeze():
    dest=ROOT/'registration.json'
    if dest.exists():
        r=C.read(dest);E.verify(r['protected']);verify_code(r);return r
    jobs=describe_sources();protected={};audits=[]
    for job in jobs:
        q=job['q'];base=C.resolve(job['base_dir']);paths=[C.resolve(job['manifest']),base/'raw/raw.npz',base/'raw/source_condition.npz',
            base/'raw/receipt.json',base/'post/proposal.npz',base/'post/pre_geo.npz',base/'post/receipt.json',C.resolve(job['old_base']),C.resolve(job['old_final'])]
        for name in ('old_base_score','old_final_score'):
            sp=C.resolve(job[name]);paths.extend([sp/'score.json',sp/'pairs.json'])
            if (sp/'row.json').exists():paths.append(sp/'row.json')
        for key in ('inference_npz','metadata_npz'):paths.append(C.resolve(q[key]))
        paths.append(C.resolve(q['query_descriptor']['path']))
        for p in paths:protected[str(p)]=C.sha(p)
        rr=C.arrays(base/'raw/raw.npz');pr=C.arrays(base/'post/proposal.npz');source=C.arrays(base/'raw/source_condition.npz')
        if set(pr)!=set(('official_unconstrained_proposal',)):raise ValueError('Unexpected D0 cache keys '+str(pr.keys()))
        p=pr['official_unconstrained_proposal'];anchors=rr['raw_anchor_ids'];restore=p.copy();restore[anchors>=0]=rr['C_vertices'][anchors[anchors>=0]]
        if restore.tobytes()!=C.arrays(base/'post/pre_geo.npz')['output'].tobytes():raise ValueError('Original hard restoration differs')
        if rr['raw'].shape!=(q['N'],3,3) or rr['C'].tobytes()!=source['C_original'].tobytes():raise ValueError('RAW/source identity differs')
        if 'boundary_edges' not in q:
            inp=C.arrays(q['inference_npz']);meta=C.arrays(q['metadata_npz']);_,A,_,_,_=E.helpers()
            _,_,top=A.topology(inp['C_vertices'][inp['C_faces_original']],meta['C_source_vertex_ids'][inp['C_faces_original']]);q['boundary_edges']=top['boundary_edge_count']
        audits.append(dict(panel=job['panel'],request_id=q['request_id'],N=q['N'],K=q['K'],C_boundary_edges=q['boundary_edges'],proposal_original_hard_restore_exact=True))
    code={str(p):C.sha(p) for p in Path(__file__).parent.glob('*.py')}
    for mod in (P.H,P.alpha,P.holes,M,T):code[str(Path(mod.__file__))]=C.sha(mod.__file__)
    code.update(H.Q.definition_hashes());code.update(E.source_hashes())
    for p in (C.ROOT/'r2/src/meshflow_control/geo_c.py',C.ROOT/'r2/src/meshflow_control/geo_io.py',C.ROOT/'r2/src/meshflow_control/source_winding.py'):
        code[str(p)]=C.sha(p)
    C.write(ROOT/'synthetic_unit_tests.json',test_geometry.run());C.csvwrite(ROOT/'input_audit.csv',audits)
    r=dict(status='FROZEN',version=P.VERSION,utc=C.utc(),jobs=jobs,protected=protected,code=code,inputs=140,
        stages=STAGES,R_model=guard_geo.RADIUS,L_model=C.L,
        floor='RAW shortest >=1.5R AND hard-restored D0 shortest <1.25R; exact first quadratic crossing of all3edges; one shared face alpha',
        anticollapse='RAW longest >3R AND shortest >R; deterministic original GEO order; no two corners of protected face may join one existing coordinate cluster',
        prior_degeneracy='Does not separate already coincident PRE_GEO corners or repair pre-existing collinearity; retained accounting',
        post='GUARD_GEO -> one HINGE_OPP015 -> one original ALPHA -> one original HOLE',
        baseline='OLD_BASE and actual OLD_FINAL reused read-only; Bench OLD_FINAL is HINGE only, not an old ALPHA/HOLE chain',
        metrics='Original full135, existing179 where bridge scorer available; same queries and parent-equal summaries; no unique GT style rank',
        closed_C='Boundary and seam-specific metrics NA, free quality applicable',stop='140 cache requests only; Step2 separate authorized registry',
        GPU_calls=0,model_calls=0,new_D0=0,new_queries=0,new_noise=0)
    C.write(dest,r);return r

def cell(job):return ROOT/'cells'/job['panel']/job['q']['request_id']

def score_current(q,mesh,dest,label,pairs=None):
    # Original math cloned only to retain Bench's existing closed-C NA contract.
    ns=dict(vars(M))
    for name,obj in vars(M).items():
        if inspect.isfunction(obj) and obj.__globals__ is vars(M):
            fn=types.FunctionType(obj.__code__,ns,obj.__name__,obj.__defaults__,obj.__closure__);fn.__kwdefaults__=obj.__kwdefaults__;ns[name]=fn
    prior=ns['role_reach']
    def reach(*args,**kwargs):
        internal,stats,rows=prior(*args,**kwargs)
        if q['boundary_edges']==0:internal=dict(status='NA_CLOSED_C_NO_SEAM_INTERFACE',**{k:None for k in E.INTERNAL_FIELDS})
        return internal,stats,rows
    ns['role_reach']=reach
    a=dict(mesh)
    if 'index_domain' not in a:a['index_domain']=np.asarray('SOURCE_C_SHARED_GUARDED_GEO')
    if np.any(a.get('face_role',np.asarray([]))=='BRIDGE'):pairs=None
    return ns['score'](q,a,dest,label,existing_pairs=pairs)

def baseline_row(job,stage):
    key='old_base' if stage=='OLD_BASE' else 'old_final';sp=C.resolve(job[key+'_score'])
    if (sp/'row.json').exists():row=C.read(sp/'row.json')
    else:
        score=C.read(sp/'score.json');mesh=C.arrays(job[key]);pairs=C.read(sp/'pairs.json')
        row=metric_row(score,mesh,pairs)[0]
        _,_,G,_,_=E.helpers();_,edges,_=G.actual_valid_graph(mesh['vertices'],mesh['faces'],job['q']['K'],pairs)
        row.update(T.measure(mesh,job['q']['K'],edges,C.L)[0])
    row=dict(row);row.update(panel=job['panel'],stage=stage,old_final_chain=job['old_final_chain'],C_boundary_edges=job['q']['boundary_edges'])
    return row

def work(job):
    q=job['q'];dest=cell(job);done=dest/'complete.json'
    if done.exists():r=C.read(done);E.verify(r['files']);return r
    if dest.exists():raise RuntimeError('Partial attempt preserved; explicit engineering recovery required '+str(dest))
    tick=time.perf_counter();contract=active_contract()
    C.write(dest/'started.json',dict(utc=C.utc(),pid=os.getpid(),panel=job['panel'],request_id=q['request_id'],
        execution_revision=contract['revision'],execution_registration_sha256=contract['sha256']))
    try:
        base=C.resolve(job['base_dir']);raw=C.arrays(base/'raw/raw.npz');proposal=C.arrays(base/'post/proposal.npz')['official_unconstrained_proposal']
        source=C.arrays(base/'raw/source_condition.npz');old=C.arrays(job['old_base']);oldpairs=C.read(C.resolve(job['old_base_score'])/'pairs.json')
        pre,canon,geo,g,w,anchors,fr=P.geometry_from_cached(raw,proposal,source)
        save_npz(dest/'guarded_pre_geo.npz',dict(output=pre,raw_anchor_ids=raw['raw_anchor_ids']))
        save_npz(dest/'guarded_canonical.npz',canon)
        C.write(dest/'floor_faces.json',fr);C.csvwrite(dest/'floor_faces.csv',[{k:v for k,v in x.items() if k!='quadratics'} for x in fr])
        C.write(dest/'geo.json',dict(stats=g['stats'],face_records=g['face_records'],guard_records=g['guard_records']))
        C.write(dest/'source_restoration.json',w['restoration_summary']);C.write(dest/'orientation.json',w['orientation_result']);C.write(dest/'anchor_identity.json',anchors)
        baseline=H.Q.connectivity_ledger(old['vertices'],old['faces'],q['K'],oldpairs,old['source_face_ids'],old['source_corner_ids'])
        stages=[];current=geo;pairs=None;scored={}
        for stage in STAGES:
            p=dest/stage;now=time.perf_counter()
            if stage=='GUARD_GEO':
                result=score_current(q,current,p/'scores',stage);pairs=result['pairs'];records=[];summary=g['stats']
                after=H.Q.connectivity_ledger(current['vertices'],current['faces'],q['K'],pairs,current['source_face_ids'],current['source_corner_ids'])
                paired=H.Q.compare_ledgers(baseline,after,oldpairs,pairs,list(range(q['K'],q['N'])))
                C.write(p/'old_base_paired_ledger.json',paired);C.write(p/'connectivity.json',after)
                scored[stage]=result['row']
            else:
                previous=current;out,records,summary,before,after,npairs=P.finish_stage(stage,current,pre,raw['raw'],pairs)
                if out['C'].tobytes()!=geo['C'].tobytes():raise AssertionError('CPU chain changed C')
                current,pairs=out,npairs
                if before is not None:
                    paired=H.Q.compare_ledgers(before,after,C.read(dest/stages[-1]['stage']/'pairs.json'),pairs,summary['affected_source_raw_slots'])
                    C.write(p/'paired_ledger.json',paired)
                    for k in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                        if paired['summary'][k]:raise AssertionError('Original fixed CPU guard violated '+k)
                if stage=='FINAL':
                    result=score_current(q,current,p/'scores',stage,pairs);scored[stage]=result['row']
                    # Full original detector runs if BRIDGE exists; verify every contact identity/classification.
                    for name in pairs:
                        a={(tuple(x['pair']),x['classification']) for x in pairs[name]['contacts']};b={(tuple(x['pair']),x['classification']) for x in result['pairs'][name]['contacts']}
                        if a!=b:raise AssertionError('Final independent scan disagrees')
            save_npz(p/'post.npz',current);C.write(p/'pairs.json',pairs);C.write(p/'summary.json',summary);C.write(p/'candidate_records.json',records)
            stages.append(dict(stage=stage,seconds=time.perf_counter()-now,summary=summary))
        rows=[baseline_row(job,s) for s in ('OLD_BASE','OLD_FINAL')]
        for stage,row in scored.items():rows.append(dict(row,panel=job['panel'],stage=stage,old_final_chain=job['old_final_chain'],C_boundary_edges=q['boundary_edges']))
        C.write(dest/'metric_rows.json',rows)
        # Source-slot relation accounting uses the unchanged prior POST plus a
        # final graph with BRIDGE faces carrying new negative IDs, never GT IDs.
        ids=current['parent_source_raw_slot'].copy() if 'parent_source_raw_slot' in current else current['source_face_ids'].copy();bridge=ids<0
        ids[bridge]=-1-np.arange(bridge.sum())
        corners=np.vstack((geo['source_corner_ids'],np.tile(np.arange(3),(int(bridge.sum()),1))))
        finalledger=H.Q.connectivity_ledger(current['vertices'],current['faces'],q['K'],pairs,ids,corners)
        paired=H.Q.compare_ledgers(baseline,finalledger,oldpairs,pairs,list(range(q['K'],q['N'])))
        C.write(dest/'old_base_final_paired_ledger.json',paired)
        fixed=raw['raw_anchor_ids']>=0
        displacement=dict(max_guard_pre_vs_RAW=float(np.linalg.norm(pre.astype(float)-raw['raw'],axis=2).max()),
            max_guard_pre_vs_D0_restored=float(max(r['max_guard_minus_proposal'] for r in fr)),
            floor_faces=sum(r['floor_triggered'] for r in fr),protected_faces=sum(r['anti_collapse_protected'] for r in fr),
            all_hard_anchor_coordinates_exact=bool(pre[fixed].tobytes()==raw['C_vertices'][raw['raw_anchor_ids'][fixed]].tobytes()))
        if 'torch' in sys.modules:raise RuntimeError('CPU job imported neural framework')
        rec=dict(status='COMPLETE',panel=job['panel'],request_id=q['request_id'],seconds=time.perf_counter()-tick,stages=stages,
            execution_revision=contract['revision'],execution_registration_sha256=contract['sha256'],
            displacement=displacement,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,GPU_calls=0,new_noise=0,
            files={str(p):C.sha(p) for p in dest.rglob('*') if p.is_file()})
        C.write(done,rec);print('COMPLETE',job['panel'],q['request_id'],round(rec['seconds'],2),flush=True);return rec
    except BaseException as exc:
        C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),model_calls=0));raise

def run(workers=4):
    reg=freeze();done=ROOT/'completion.json'
    if done.exists():r=C.read(done);E.verify(r['files']);return r
    if not (ROOT/'resources_before.json').exists():C.write(ROOT/'resources_before.json',C.resources())
    tick=time.perf_counter();first=work(reg['jobs'][0]);snapshot=C.resources();count=worker_count(snapshot,first,workers)
    C.write(ROOT/'dispatch.json',dict(workers=count,threads_per_worker=1,resources=snapshot,first_seconds=first['seconds'],first_peak_RSS_KiB=first['peak_RSS_KiB'],GPU_workers=0),replace=(ROOT/'dispatch.json').exists())
    # Start the second real cell, then add one independent CPU slot at a time.
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        receipts=[first];todo=iter(reg['jobs'][1:]);active={};slots=1
        active[pool.submit(work,next(todo))]=True
        while active:
            completed,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in completed:receipts.append(f.result());del active[f]
            if slots<count:
                snap=C.resources();available=worker_count(snap,first,workers)
                slots=min(slots+1,available,count)
            while len(active)<slots:
                try:job=next(todo)
                except StopIteration:break
                active[pool.submit(work,job)]=True
    E.verify(reg['protected']);contract=verify_code(reg)
    C.csvwrite(ROOT/'execution_cells.csv',[dict(panel=r['panel'],request_id=r['request_id'],seconds=r['seconds'],
        execution_revision=r.get('execution_revision','CORRECTED_GEO_V1'),**r['displacement']) for r in receipts])
    r=dict(status='COMPLETE',requests=len(receipts),workers=count,seconds=time.perf_counter()-tick,model_calls=0,GPU_calls=0,new_noise=0,
        seconds_scope='LAST_RESUME_CONTROLLER_WALL_NOT_TOTAL',sum_completed_cell_seconds=sum(x['seconds'] for x in receipts),
        execution_revisions=dict(collections.Counter(x.get('execution_revision','CORRECTED_GEO_V1') for x in receipts)),
        final_execution_registration_sha256=contract['sha256'],
        files={str(cell(j)/'complete.json'):C.sha(cell(j)/'complete.json') for j in reg['jobs']})
    C.write(done,r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run'));p.add_argument('--workers',type=int,choices=(1,2,3,4,5,6),default=4);a=p.parse_args()
    r=freeze() if a.action=='freeze' else run(a.workers);print(json.dumps({k:v for k,v in r.items() if k not in ('jobs','files','protected','code')}),flush=True)
