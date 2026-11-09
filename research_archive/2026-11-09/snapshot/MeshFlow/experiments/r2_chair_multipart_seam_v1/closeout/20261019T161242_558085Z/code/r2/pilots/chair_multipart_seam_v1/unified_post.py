"""CPU-only unified rescue on the 44 final mixed-part cached D0 proposals.

Preparation freezes the exact Step1 code/parameters before caches exist. Actual
execution requires all RAW, single-D0, BASE_POST scores and source-C receipts.
No original BASE pipeline or BASE score is recomputed. Failed partial cells are
preserved and never silently retried; completed cells can be verified/reused.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import concurrent.futures
import copy
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import traceback
import numpy as np
from r2.pilots.unified_rescue_post_v1 import run as U
from r2.pilots.unified_rescue_post_v1 import pipeline as P
from r2.pilots.chair_back_transfer_v1.post_cpu import save_npz
from r2.pilots.chair_back_transfer_v1.evaluate import worker_count
from .eval_prepare import OUT as EXPERIMENT, LABEL

C,E,H=U.C,U.E,P.H
ROOT=EXPERIMENT/'unified'
SOURCE_REGISTRATION=C.ROOT/'experiments/r2_unified_rescue_post_v1/registration.json'
STAGES=('GUARD_GEO','HINGE_OPP015','ALPHA','FINAL')


def cell(job):
    return C.resolve(job['output_root'])/'cells'/job['panel']/job['q']['request_id']


def prepare():
    path=ROOT/'registration.json'
    if path.exists():
        r=C.read(path);E.verify(r['protected']);E.verify(r['code']);return r
    source=C.read(SOURCE_REGISTRATION);contract=U.verify_code(source)
    assert source['status']=='FROZEN' and source['version']==P.VERSION
    assert tuple(source['stages'])==STAGES
    manifest=EXPERIMENT/'eval_manifest.json';m=C.read(manifest)
    if len(m['requests'])!=44 or len({q['request_id'] for q in m['requests']})!=44:
        raise ValueError('Exactly 44 original requests required')
    E.verify(m['protected'])
    jobs=[]
    for q in m['requests']:
        part=q['part_type'];base=EXPERIMENT/'evaluation'/LABEL/q['request_id']
        jobs.append(dict(q=copy.deepcopy(q),panel=part,output_root=str(ROOT),
                         execution_revision=contract['revision'],execution_registration_sha256=contract['sha256'],
                         source_request_id=q['request_id'],source_manifest=str(manifest),
                         base_dir=str(base),old_base=str(base/'post/post.npz'),
                         old_base_score=str(EXPERIMENT/'evaluation/scores'/LABEL/q['request_id']/'BASE_POST'),
                         old_comparison_panel='CHAIR_BACK_S2000' if part=='BACK_FULL' else 'CHAIR_SEAT_S2000'))
    code=dict(contract['code']);code[str(Path(__file__))]=C.sha(__file__)
    parameters={k:source[k] for k in ('version','R_model','L_model','floor','anticollapse','prior_degeneracy','post')}
    result=dict(schema='chair_multipart_unified_post_v1',status='FROZEN',jobs=jobs,inputs=44,
                label=LABEL,parameters=parameters,stages=list(STAGES),scored_new_stages=['GUARD_GEO','FINAL'],
                baseline='Read existing BASE_POST score only; no BASE geometry or scoring rerun',
                source_pipeline_registration=str(SOURCE_REGISTRATION),
                source_pipeline_registration_sha256=C.sha(SOURCE_REGISTRATION),
                source_execution_revision=contract['revision'],
                source_execution_registration=contract['path'],source_execution_registration_sha256=contract['sha256'],
                protected={str(manifest):C.sha(manifest),str(SOURCE_REGISTRATION):C.sha(SOURCE_REGISTRATION),
                           contract['path']:contract['sha256'],**contract.get('prior_code_snapshots',{}),**m['protected']},
                code=code,CPU_workers_max=2,threads_per_worker=1,
                model_calls=0,GPU_calls=0,new_D0=0,new_queries=0,new_noise=0,
                readiness='All44 existing RAW/proposal/BASE producer+score receipts required before first geometry call',
                failure_policy='No automatic retry of partial cell; preserve failure and source caches',
                scope='Same frozen Step1 floor/same-faceguard/HINGE_OPP015/ALPHA/HOLE; no extra heuristic, radius scan or source-owner input')
    C.write(path,result);return result


def freeze_inputs():
    reg=prepare();path=ROOT/'input_registration.json'
    if path.exists():
        r=C.read(path);E.verify(r['protected']);return r
    missing=[]
    for job in reg['jobs']:
        base=C.resolve(job['base_dir']);score=C.resolve(job['old_base_score'])
        for p in (base/'raw/receipt.json',base/'post/receipt.json',base/'post/proposal_receipt.json',score/'audit_receipt.json'):
            if not p.exists():missing.append(str(p))
    if missing:
        return dict(status='WAITING_FOR_EXISTING_CACHES',missing_count=len(missing),missing=missing,model_calls=0)
    protected={};jobs=[];audits=[]
    for original in reg['jobs']:
        job=copy.deepcopy(original);q=job['q'];base=C.resolve(job['base_dir']);score=C.resolve(job['old_base_score'])
        rr=C.read(base/'raw/receipt.json');pr=C.read(base/'post/receipt.json');dr=C.read(base/'post/proposal_receipt.json')
        sr=C.read(score/'audit_receipt.json')
        if rr.get('request_id')!=q['request_id'] or pr.get('request_id')!=q['request_id']:
            raise ValueError('Producer request identity differs')
        if rr['generator_forwards']!=50 or pr['D0_forwards']!=1 or pr['GEO_calls']!=1 or dr['D0_forwards']!=1:
            raise ValueError('Expected original Euler50 and single D0/GEO')
        if sr['identity']['label']!=LABEL or sr['identity']['stage']!='BASE_POST' or sr['identity']['request_id']!=q['request_id']:
            raise ValueError('BASE score identity differs')
        for rec in (rr,pr,dr,sr):E.verify(rec['files'])
        raw=C.arrays(base/'raw/raw.npz');proposal=C.arrays(base/'post/proposal.npz')
        source=C.arrays(base/'raw/source_condition.npz');pre=C.arrays(base/'post/pre_geo.npz')['output']
        if set(proposal)!={'official_unconstrained_proposal'}:raise ValueError('Unexpected D0 proposal schema')
        restore=proposal['official_unconstrained_proposal'].copy();a=raw['raw_anchor_ids'];fixed=a>=0
        restore[fixed]=raw['C_vertices'][a[fixed]]
        if restore.tobytes()!=pre.tobytes():raise ValueError('Hard-restored cached D0 differs')
        if raw['raw'].shape!=(q['N'],3,3) or raw['C'].tobytes()!=source['C_original'].tobytes():
            raise ValueError('RAW/source original C differs')
        inp=C.arrays(q['inference_npz']);meta=C.arrays(q['metadata_npz']);_,A,_,_,_=E.helpers()
        _,_,top=A.topology(inp['C_vertices'][inp['C_faces_original']],meta['C_source_vertex_ids'][inp['C_faces_original']])
        q['boundary_edges']=top['boundary_edge_count']
        if q['boundary_edges']!=len(raw['edge_records']):raise ValueError('Original C boundary identity differs')
        paths=[base/'raw/raw.npz',base/'raw/source_condition.npz',base/'raw/receipt.json',base/'post/proposal.npz',
               base/'post/proposal_receipt.json',base/'post/pre_geo.npz',base/'post/post.npz',base/'post/receipt.json',
               score/'audit_receipt.json',score/'score.json',score/'row.json',score/'pairs.json']
        for p in paths:protected[str(p)]=C.sha(p)
        audits.append(dict(panel=job['panel'],request_id=q['request_id'],source_request_id=q['request_id'],
                           N=q['N'],K=q['K'],C_boundary_edges=q['boundary_edges'],hard_restore_exact=True,
                           existing_BASE_score_reused=True))
        jobs.append(job)
    C.csvwrite(ROOT/'input_audit.csv',audits)
    result=dict(status='FROZEN',jobs=jobs,protected=protected,registration_sha256=C.sha(ROOT/'registration.json'),
                inputs=44,model_calls=0,new_geometry_calls=0,new_score_calls=0)
    C.write(path,result);return result


def interface_probe(job):
    assert 'torch' not in sys.modules
    return dict(cell=str(cell(job)),panel=job['panel'],source_request_id=job['source_request_id'],
                old_comparison_panel=job['old_comparison_panel'],new_scores=[str(cell(job)/s/'scores/row.json') for s in ('GUARD_GEO','FINAL')],
                model_calls=0,geometry_calls=0)


def check_cpu():
    reg=prepare()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        result=pool.submit(interface_probe,reg['jobs'][0]).result()
    assert Path(result['cell']).is_relative_to(ROOT)
    value=dict(status='PASS',production_cell_path_in_spawn=result,requests=len(reg['jobs']),
               same_Step1_parameters=True,source_registration_sha256=reg['source_pipeline_registration_sha256'],
               source_execution_revision=reg['source_execution_revision'],
               source_execution_registration_sha256=reg['source_execution_registration_sha256'],
               model_calls=0,geometry_calls=0,new_scores=0,resources=C.resources())
    path=ROOT/'cpu_interface_check.json'
    if not path.exists():C.write(path,value)
    return value


def work(job):
    q=job['q'];dest=cell(job);done=dest/'complete.json'
    if done.exists():
        rec=C.read(done);E.verify(rec['files']);return rec
    if dest.exists():raise RuntimeError('Partial CPU attempt preserved; no automatic retry: '+str(dest))
    tick=time.perf_counter();C.write(dest/'started.json',dict(utc=C.utc(),pid=os.getpid(),panel=job['panel'],request_id=q['request_id'],
        execution_revision=job['execution_revision'],execution_registration_sha256=job['execution_registration_sha256']))
    try:
        base=C.resolve(job['base_dir']);raw=C.arrays(base/'raw/raw.npz')
        proposal=C.arrays(base/'post/proposal.npz')['official_unconstrained_proposal'];source=C.arrays(base/'raw/source_condition.npz')
        old=C.arrays(job['old_base']);oldpairs=C.read(C.resolve(job['old_base_score'])/'pairs.json')
        pre,canonical,geo,g,w,anchors,floor_rows=P.geometry_from_cached(raw,proposal,source)
        save_npz(dest/'guarded_pre_geo.npz',dict(output=pre,raw_anchor_ids=raw['raw_anchor_ids']))
        save_npz(dest/'guarded_canonical.npz',canonical)
        C.write(dest/'floor_faces.json',floor_rows);C.csvwrite(dest/'floor_faces.csv',[{k:v for k,v in row.items() if k!='quadratics'} for row in floor_rows])
        C.write(dest/'geo.json',dict(stats=g['stats'],face_records=g['face_records'],guard_records=g['guard_records']))
        C.write(dest/'source_restoration.json',w['restoration_summary']);C.write(dest/'orientation.json',w['orientation_result'])
        C.write(dest/'anchor_identity.json',anchors)
        baseline=H.Q.connectivity_ledger(old['vertices'],old['faces'],q['K'],oldpairs,old['source_face_ids'],old['source_corner_ids'])
        oldrow=C.read(C.resolve(job['old_base_score'])/'row.json')
        rows=[dict(oldrow,panel=job['panel'],part_type=job['panel'],stage='BASE_POST',label=LABEL,
                   source_request_id=q['request_id'],C_boundary_edges=q['boundary_edges'])]
        stages=[];current=geo;pairs=None
        for stage in STAGES:
            path=dest/stage;started=time.perf_counter()
            if stage=='GUARD_GEO':
                result=U.score_current(q,current,path/'scores',stage);pairs=result['pairs'];records=[];summary=g['stats']
                after=H.Q.connectivity_ledger(current['vertices'],current['faces'],q['K'],pairs,current['source_face_ids'],current['source_corner_ids'])
                paired=H.Q.compare_ledgers(baseline,after,oldpairs,pairs,list(range(q['K'],q['N'])))
                C.write(path/'old_base_paired_ledger.json',paired);C.write(path/'connectivity.json',after)
            else:
                out,records,summary,before,after,newpairs=P.finish_stage(stage,current,pre,raw['raw'],pairs)
                if out['C'].tobytes()!=geo['C'].tobytes():raise AssertionError('Unified CPU chain changed C')
                if before is not None:
                    paired=H.Q.compare_ledgers(before,after,pairs,newpairs,summary['affected_source_raw_slots'])
                    for key in ('CF_lost_legal_edges','FF_lost_legal_edges','CF_new_illegal_pairs','FF_new_illegal_pairs','deleted_source_faces'):
                        if paired['summary'][key]:raise AssertionError('Original fixed CPU guard violated '+key)
                    C.write(path/'paired_ledger.json',paired)
                current,pairs=out,newpairs
                if stage=='FINAL':
                    result=U.score_current(q,current,path/'scores',stage,pairs)
                    for kind in pairs:
                        a={(tuple(r['pair']),r['classification']) for r in pairs[kind]['contacts']}
                        b={(tuple(r['pair']),r['classification']) for r in result['pairs'][kind]['contacts']}
                        if a!=b:raise AssertionError('Final full scorer contacts differ')
            if stage in ('GUARD_GEO','FINAL'):
                rows.append(dict(result['row'],panel=job['panel'],part_type=job['panel'],stage=stage,label=LABEL,
                                 source_request_id=q['request_id'],C_boundary_edges=q['boundary_edges']))
            save_npz(path/'post.npz',current);C.write(path/'pairs.json',pairs);C.write(path/'summary.json',summary)
            C.write(path/'candidate_records.json',records)
            stages.append(dict(stage=stage,seconds=time.perf_counter()-started,summary=summary))
        C.write(dest/'metric_rows.json',rows)
        ids=current['parent_source_raw_slot'].copy() if 'parent_source_raw_slot' in current else current['source_face_ids'].copy()
        bridge=ids<0;ids[bridge]=-1-np.arange(bridge.sum())
        corners=np.vstack((geo['source_corner_ids'],np.tile(np.arange(3),(int(bridge.sum()),1))))
        final=H.Q.connectivity_ledger(current['vertices'],current['faces'],q['K'],pairs,ids,corners)
        C.write(dest/'old_base_final_paired_ledger.json',H.Q.compare_ledgers(baseline,final,oldpairs,pairs,list(range(q['K'],q['N']))))
        fixed=raw['raw_anchor_ids']>=0
        displacement=dict(max_guard_pre_vs_RAW=float(np.linalg.norm(pre.astype(float)-raw['raw'],axis=2).max()),
            max_guard_pre_vs_D0_restored=float(max(r['max_guard_minus_proposal'] for r in floor_rows)),
            floor_faces=sum(r['floor_triggered'] for r in floor_rows),protected_faces=sum(r['anti_collapse_protected'] for r in floor_rows),
            all_hard_anchor_coordinates_exact=bool(pre[fixed].tobytes()==raw['C_vertices'][raw['raw_anchor_ids'][fixed]].tobytes()))
        if 'torch' in sys.modules:raise RuntimeError('CPU adapter imported a neural framework')
        rec=dict(status='COMPLETE',panel=job['panel'],request_id=q['request_id'],source_request_id=q['request_id'],
            execution_revision=job['execution_revision'],execution_registration_sha256=job['execution_registration_sha256'],
            old_comparison_panel=job['old_comparison_panel'],seconds=time.perf_counter()-tick,stages=stages,displacement=displacement,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_calls=0,GPU_calls=0,new_D0=0,new_noise=0,
            new_query_arrays=0,BASE_rescored=False,full_new_score_stages=['GUARD_GEO','FINAL'],
            files={str(p):C.sha(p) for p in dest.rglob('*') if p.is_file()})
        C.write(done,rec);print('COMPLETE',job['panel'],q['request_id'],round(rec['seconds'],2),flush=True);return rec
    except BaseException as exc:
        C.write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),model_calls=0));raise


def run(workers=2):
    reg=prepare();inputs=freeze_inputs()
    if inputs['status']!='FROZEN':return inputs
    done=ROOT/'completion.json'
    if done.exists():r=C.read(done);E.verify(r['files']);return r
    if not (ROOT/'resources_before.json').exists():C.write(ROOT/'resources_before.json',C.resources())
    tick=time.perf_counter();jobs=inputs['jobs'];first=work(jobs[0]);snapshot=C.resources()
    count=worker_count(snapshot,first,workers)
    C.write(ROOT/'dispatch.json',dict(workers=count,threads_per_worker=1,resources=snapshot,first_seconds=first['seconds'],
            first_peak_RSS_KiB=first['peak_RSS_KiB'],GPU_workers=0),replace=(ROOT/'dispatch.json').exists())
    receipts=[first]
    with concurrent.futures.ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
        todo=iter(jobs[1:]);active={pool.submit(work,next(todo)):True};slots=1
        while active:
            completed,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:receipts.append(future.result());del active[future]
            if slots<count:slots=min(slots+1,worker_count(C.resources(),first,workers),count)
            while len(active)<slots:
                try:job=next(todo)
                except StopIteration:break
                active[pool.submit(work,job)]=True
    E.verify(reg['protected']);E.verify(reg['code']);E.verify(inputs['protected'])
    C.csvwrite(ROOT/'execution_cells.csv',[dict(panel=r['panel'],request_id=r['request_id'],seconds=r['seconds'],**r['displacement']) for r in receipts])
    result=dict(status='COMPLETE',requests=len(receipts),workers=count,seconds=time.perf_counter()-tick,
                source_execution_revision=reg['source_execution_revision'],
                source_execution_registration_sha256=reg['source_execution_registration_sha256'],
                model_calls=0,GPU_calls=0,new_D0=0,new_noise=0,new_queries=0,BASE_rescored=False,
                files={str(cell(job)/'complete.json'):C.sha(cell(job)/'complete.json') for job in jobs})
    C.write(done,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','check-cpu','inputs','run'))
    parser.add_argument('--workers',type=int,choices=(1,2),default=2);args=parser.parse_args()
    result=prepare() if args.action=='prepare' else check_cpu() if args.action=='check-cpu' else freeze_inputs() if args.action=='inputs' else run(args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','files','protected','code','missing')},ensure_ascii=False,indent=2))
