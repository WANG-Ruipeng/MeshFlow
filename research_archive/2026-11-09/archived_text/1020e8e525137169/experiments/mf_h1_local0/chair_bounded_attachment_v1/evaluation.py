"""CPU-only scoring after the entire registered intervention cohort is saved."""
from pathlib import Path
import argparse
import json
import math
import time
import traceback
import numpy as np
from .common import OUT,LABELS,read_json,resolve,digest,array_hash,atomic_json,freeze_json,append_event,arrays

VERSION='bounded_attachment_evaluation_v1'
QUALITY_IDENTITY_KEYS=('D_sym2048','GT_to_F_MSE2048_L0','F_to_GT_MSE2048_L0',
    'fixed_query_MSE','fixed_query_p95_L','coverage_fraction','CF_proper_pair_count',
    'CF_proper_affected_free_fraction','CF_coplanar_pair_count','CF_unresolved_or_degenerate_pair_count',
    'FF_illegal_pair_count','FF_illegal_affected_free_fraction','FF_unresolved_or_degenerate_pair_count',
    'free_exact_zero_area_count','free_near_degenerate_count','free_minimum_angle_lt5_fraction',
    'free_total_surface_area','free_area_over_GT_free_area',
    'Q_Y2048_sha256','Q_F2048_sha256','legacy_query_sha256')


def barrier(out,requests,phase):
    """No GT files are opened here. All sixteen coordinates must exist first."""
    label='ENDPOINT' if phase==1 else 'TRAJECTORY'
    jobs=[r for r in requests if r['label']==label]
    failures=[];saved=[]
    if len(jobs)!=16 or len({r['base_request_id'] for r in jobs})!=16:
        failures.append('Expected sixteen distinct registered '+label+' requests')
    for request in jobs:
        p=Path(out)/'outputs'/request['request_id']/'result.json'
        if not p.exists():failures.append(request['request_id']+': result not saved');continue
        result=read_json(p)
        expected='PROJECTED' if phase==1 else 'SAMPLED'
        if result.get('status')!=expected or not result.get('raw_path'):
            failures.append(request['request_id']+': no completed '+label+' coordinates');continue
        raw=resolve(result['raw_path'])
        if not raw.is_file() or digest(raw)!=result.get('raw_sha256'):
            failures.append(request['request_id']+': saved geometry hash mismatch');continue
        saved.append(dict(request_id=request['request_id'],result_sha256=digest(p),raw_sha256=result['raw_sha256']))
    if phase==2:
        gate=Path(out)/'phase1_gate.json'
        if not gate.exists() or read_json(gate).get('status')!='PASS':failures.append('Phase2 requires preregistered phase1 PASS')
    value=dict(status='PASS' if not failures else 'BLOCKED',phase=phase,cohort=label,
        expected=16,saved=len(saved),rows=saved,failures=failures,GT_files_read=0)
    atomic_json(Path(out)/'scoring'/f'phase{phase}_save_barrier.json',value)
    return value


def load_condition(out,request):
    p=resolve(request['condition_npz'])
    if digest(p)!=request['condition_sha256']:raise ValueError('Condition changed')
    metadata=read_json(Path(out)/'registration/conditions.json')
    if isinstance(metadata,dict):metadata=metadata.get('rows',metadata.get('conditions',[]))
    records=[x for x in metadata if x['condition_id']==request['condition_id']]
    if len(records)!=1:raise ValueError('Expected exactly one registered condition')
    value=dict(records[0]);value.update(arrays(p));return value


def _same(a,b):
    if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b


def raw_identity_check(out,request,row,control_value):
    from . import metrics_adapter as adapter
    prior=read_json(adapter.checked_path(request,'legacy_boundary_joined_path'))
    prior_control=read_json(adapter.checked_path(request,'legacy_control_score_path'))['metrics']
    if prior['raw_sha256']!=row['raw_sha256']:raise ValueError('RAW differs from original same-request bytes')
    failures=[];count=0;max_abs=0.;exact=0
    for key in QUALITY_IDENTITY_KEYS:
        count+=1
        if key not in row or key not in prior or not _same(row[key],prior[key]):failures.append(key)
        elif row[key]==prior[key]:exact+=1
    for key,value in prior_control.items():
        count+=1;observed=control_value['metrics'].get(key)
        if not _same(observed,value):failures.append(key)
        elif observed==value:exact+=1
        if isinstance(value,(float,int)) and not isinstance(value,bool) and isinstance(observed,(float,int)):
            max_abs=max(max_abs,abs(observed-value))
    receipt=dict(status='PASS' if not failures else 'BLOCKED',request_id=request['request_id'],
        historical_request_id=request['legacy_boundary_request_id'],compared_fields=count,exactly_equal_fields=exact,
        failed_fields=failures,max_control_absolute_difference=max_abs,comparison_rtol=1e-12,comparison_atol=1e-12,
        original_taus=[.005,.01,.02],strict_tau_is_new=True,old_results_modified=False)
    atomic_json(Path(out)/'scoring/raw_identity'/(request['request_id']+'.json'),receipt)
    if failures:raise ValueError('RAW old score identity mismatch: '+','.join(failures))
    return receipt


def _finite_number(value):
    return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)


def score_one(out,request):
    from . import metrics_adapter as adapter
    from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_json
    out=Path(out);rid=request['request_id'];p=out/'outputs'/rid/'result.json'
    result=read_json(p);condition=load_condition(out,request)
    identity=dict(version=VERSION,request_sha256=__import__('hashlib').sha256(json.dumps(request,sort_keys=True).encode()).hexdigest(),
        result_sha256=digest(p),condition_sha256=request['condition_sha256'],sources=adapter.source_identity())
    target=out/'scoring/joined'/(rid+'.json');cache=target.with_suffix('.identity.json')
    if target.exists():
        saved=read_json(cache)
        if saved['identity']!=identity or digest(target)!=saved['score_sha256']:raise ValueError('Completed evaluation identity changed')
        return read_json(target)
    row=adapter.score_quality(out,request)
    row.update(label=request['label'],base_request_id=request['base_request_id'],base_id=request.get('base_id',request['base_request_id']),
        noise_index=int(request['candidate']),generation_status=result['status'])
    if row.get('status')!='PASS':return row
    raw=arrays(result['raw_path'])['raw'];n,k=int(condition['N']),int(condition['K'])
    C=np.asarray(condition['C']).reshape(k,3,3)
    shape=raw.shape==(n,9) and raw.dtype==np.float32
    preserved=shape and raw[:k].tobytes()==C.reshape(k,9).tobytes()
    row.update(N_preserved=bool(shape),C_preserved=bool(preserved),finite=bool(np.isfinite(raw).all()))
    if not shape or not preserved or not row['finite']:return dict(row,status='INVALID_OUTPUT')
    free=raw[k:].reshape(-1,3,3)
    reference=adapter.load_reference(request,condition)
    value=adapter.score_control(free,condition,reference,cf_scan=adapter.cf_from_quality(row))
    row.update(value['metrics'])
    vectors=out/'scoring/control_vectors'/(rid+'.npz');vectors.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(vectors,**value['arrays'])
    score_path=out/'scoring/control'/(rid+'.json')
    atomic_json(score_path,dict(metrics=value['metrics'],statuses=value['statuses'],audit=value['audit'],
        vectors_path=str(vectors),vectors_sha256=digest(vectors),reference_path=request['legacy_control_reference_npz_path'],
        reference_sha256=request['legacy_control_reference_npz_sha256']))
    row.update(control_score_path=str(score_path),control_vectors_path=str(vectors),control_vectors_sha256=digest(vectors),
        control_reference_path=request['legacy_control_reference_npz_path'])
    gt=np.asarray(condition['GT_full']).reshape(-1,3).astype(np.float64)
    lo,hi=gt.min(0),gt.max(0);center=(lo+hi)/2;half=max((hi-lo).max()/2,1e-5)*1.08
    row['free_out_of_GT_cube_corner_fraction']=float((np.abs(free.reshape(-1,3)-center)>half).any(1).mean())
    L_C=float(np.linalg.norm(np.ptp(C.astype(np.float64).reshape(-1,3),axis=0)));delta=.01*L_C
    base_result=read_json(out/'outputs'/('RAW__'+request['base_request_id'])/'result.json')
    if digest(resolve(base_result['raw_path']))!=base_result['raw_sha256']:raise ValueError('RAW comparison bytes changed')
    baseline=arrays(base_result['raw_path'])['raw'];base_free=baseline[k:].reshape(-1,3,3)
    movement=np.linalg.norm(free.astype(np.float64)-base_free.astype(np.float64),axis=-1)
    move_path=out/'scoring/displacements'/(rid+'.npz');move_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(move_path,per_corner_displacement=movement)
    row.update(L_C=L_C,delta=delta,delta_over_L0=delta/float(condition['L0']),
        RAW_max_corner_displacement=float(movement.max()),RAW_RMS_corner_displacement=float(np.sqrt((movement**2).mean())),
        changed_corner_fraction=float((movement>0).mean()),changed_face_fraction=float((movement>0).any(1).mean()),
        displacement_path=str(move_path),displacement_sha256=digest(move_path))
    log={};logpath=result.get('intervention_log_path',result.get('log_path'))
    if logpath:
        logpath=resolve(logpath);expected=result.get('intervention_log_sha256',result.get('log_sha256'))
        if not expected or digest(logpath)!=expected:raise ValueError('Intervention log identity required')
        log=read_json(logpath);row.update(intervention_log_path=str(logpath),intervention_log_sha256=digest(logpath))
    contracts=dict(result.get('geometry_contracts',{}))
    row.update(geometry_contracts=contracts,intervention_status=log.get('status',result.get('intervention_status')),
        accepted_count=log.get('accepted_count',result.get('accepted_count')),
        maximum_constraint_residual=log.get('maximum_constraint_residual',result.get('maximum_constraint_residual')),
        raw_generation_seconds=base_result.get('source_generation_seconds',base_result.get('metadata',{}).get('seconds')),
        project_seconds=(log.get('seconds') or {}).get('total'),projection_timing=log.get('seconds'),
        source_generation_seconds=base_result.get('source_generation_seconds',base_result.get('metadata',{}).get('seconds')))
    if request['label']=='RAW':
        check=raw_identity_check(out,request,row,value);row['RAW_original_score_identity']=check['status']
        row['evaluation_contract_pass']=row['N_preserved'] and row['C_preserved'] and row['finite']
        row['deployment_seconds']=row['raw_generation_seconds']
    else:
        budget_actual=(L_C>0 and float(movement.max())<=delta) if request['label']=='ENDPOINT' else True
        row['endpoint_actual_budget_pass']=budget_actual if request['label']=='ENDPOINT' else None
        row['trajectory_RAW_drift_pass']=float(movement.max())<=3*delta if request['label']=='TRAJECTORY' else None
        required=('C','N','finite','budget','accepted_equations','no_new_degenerate','no_new_illegal_events')
        row['geometry_contracts_complete']=all(isinstance(contracts.get(key),bool) for key in required)
        row['evaluation_contract_pass']=(row['C_preserved'] and row['N_preserved'] and row['finite'] and budget_actual
            and row['geometry_contracts_complete'] and all(contracts[key] for key in required)
            and row['intervention_status']!='IMPLEMENTATION_ERROR')
        if request['label']=='ENDPOINT':
            row['deployment_seconds']=(row['raw_generation_seconds']+row['project_seconds']) if all(_finite_number(row[k]) for k in ('raw_generation_seconds','project_seconds')) else None
        else:row['deployment_seconds']=result.get('metadata',{}).get('seconds')
    row.update(status='PASS',evaluation_version=VERSION)
    write_json(target,row);atomic_json(cache,dict(identity=identity,score_sha256=digest(target)))
    return row


def _score_job(out,request):
    tick=time.perf_counter()
    try:row=score_one(out,request)
    except Exception as exc:
        row={k:request.get(k) for k in ('request_id','base_request_id','base_id','label','uid','candidate','noise_index','condition_id')}
        row.update(status='SCORING_BLOCKED',reason=repr(exc))
        atomic_json(Path(out)/'scoring/errors'/(request['request_id']+'.json'),dict(row,traceback=traceback.format_exc()))
    return row,time.perf_counter()-tick


def run(out=OUT,phase=1):
    if phase not in (1,2):raise ValueError('Only fixed phases 1 and 2')
    out=Path(out);requests=read_json(out/'requests.json');saved=barrier(out,requests,phase)
    if saved['status']!='PASS':return saved
    # This is the first possible GT-reading call in the whole evaluation entry.
    from . import metrics_adapter as adapter
    from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv
    freeze_json(out/'scoring/definition.json',adapter.definition())
    append_event(out/'scoring/cpu_ledger.jsonl',dict(event='GT_EVALUATION_BEGIN',phase=phase,all_interventions_saved=16,barrier_sha256=digest(out/'scoring'/f'phase{phase}_save_barrier.json')))
    labels=('RAW','ENDPOINT') if phase==1 else ('TRAJECTORY',)
    import os
    import psutil
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    resources=dict(logical_cores=os.cpu_count(),available_RAM_bytes=psutil.virtual_memory().available,
        CPU_utilization_percent=psutil.cpu_percent(interval=.2),CPU_threads_per_worker=1)
    workers=2 if resources['logical_cores']>=2 and resources['available_RAM_bytes']>=4*2**30 else 1
    resources.update(workers=workers,reason='Independent per-output writers; original references read-only; two workers capped by authorization',model_calls=0)
    atomic_json(out/'scoring'/f'phase{phase}_CPU_dispatch.json',resources)
    for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='1'
    jobs=[r for r in requests if r['label'] in labels];rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for request,(row,seconds) in zip(jobs,pool.map(_score_job,[out]*len(jobs),jobs)):
            rows.append(row)
            append_event(out/'scoring/cpu_ledger.jsonl',dict(event='SCORE_RETURN',phase=phase,request_id=request['request_id'],status=row['status'],seconds=seconds,model_forwards=0))
            print('CPU',phase,len(rows),request['request_id'],row['status'],flush=True)
    current={r['request_id']:r for r in rows};combined=[]
    for request in requests:
        p=out/'scoring/joined'/(request['request_id']+'.json')
        if request['request_id'] in current:value=current[request['request_id']]
        elif p.exists():value=read_json(p)
        else:value={**{k:request.get(k) for k in ('request_id','base_request_id','base_id','label','uid','candidate','noise_index','condition_id')},'status':'NOT_RUN'}
        combined.append(value)
    atomic_json(out/'scoring/per_output.json',combined);write_csv(out/'per_output_metrics.csv',combined)
    atomic_json(out/'scoring'/f'phase{phase}_rows.json',rows)
    from .aggregation import finalize
    return finalize(out,phase=phase)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--phase',type=int,choices=(1,2),default=1)
    args=p.parse_args();print(json.dumps(run(args.out,args.phase)))
