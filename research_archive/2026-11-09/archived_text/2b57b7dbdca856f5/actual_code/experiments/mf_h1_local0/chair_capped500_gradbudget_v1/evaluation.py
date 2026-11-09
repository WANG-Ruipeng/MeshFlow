"""CPU scoring with exact historical cache reuse and immutable old formulas.

No model loading, sampling, optimizer, or CONFIRM access. Existing original
references are mandatory; this adapter never redraws a GT evaluation query.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import types

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import numpy as np
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import evaluation as old
from experiments.mf_h1_local0.chair_control_feedback_auto_v1 import aggregation as gates

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'native_t1/runs/chair_capped500_gradbudget_v1'
OLD = ROOT/'native_t1/runs/chair_control_feedback_auto_v1'
VERSION = 'chair_capped500_gradbudget_evaluation_v1'
LABELS = ('START','HIST_TEACHER500','HIST_CAPPED500','T500','T1000','L500','L1000','G500','G1000')
PANELS = ('OLD','FRESH')
HISTORICAL = {'START':'START','HIST_TEACHER500':'explore_REPLAY_FM_CONTROL_500',
              'HIST_CAPPED500':'explore_TRAJECTORY_REPLAY_CAPPED_500'}
quality, control = old.quality, old.control
resolve, digest, read_json, arrays = old.resolve, old.digest, old.read_json, old.arrays
write_json, write_csv = old.write_json, old.write_csv
freeze_json, freeze_npz = old.freeze_json, old.freeze_npz


def assert_out(out):
    out=resolve(out).resolve()
    if out == OLD.resolve() or OLD.resolve() in out.parents:
        raise ValueError('Historical run is read-only')
    return out


def manifest(out=OUT,panel='OLD'):
    out=assert_out(out)
    if panel=='ALL':
        rows=manifest(out,'OLD')+manifest(out,'FRESH')
    elif panel in PANELS:
        data=read_json(out/'manifests'/(panel+'.json'));rows=data['rows']
        if data.get('panel',data.get('pool',panel))!=panel:raise ValueError('Manifest panel mismatch')
        rows=[dict(r,split=panel,panel=panel,id=r.get('id',r['request_id'])) for r in rows]
        if len(rows)!=16:raise ValueError('Each OLD/FRESH panel must have 16 fixed requests')
    else:raise ValueError('Only OLD/FRESH/ALL evaluation is authorized')
    if len({r['request_id'] for r in rows})!=len(rows) or len({r['uid'] for r in rows})!=8:
        raise ValueError('Invalid fixed parent/request cohort')
    expected=4 if panel=='ALL' else 2
    if any(sum(q['uid']==r['uid'] for q in rows)!=expected for r in rows):raise ValueError('Unequal fixed noises')
    return rows


def sources():
    return dict(version=VERSION,adapter_sha256=digest(__file__),scientific_sources=old.identities(),
                formulas='Unchanged original control v1r1, quality scorer and fixed reference arrays')


def prepare_references(out=OUT):
    out=assert_out(out);started=time.perf_counter();seen={};identity=sources()
    for request in manifest(out,'ALL'):
        cid=request['condition_id']
        if cid in seen:
            if seen[cid]['condition_sha256']!=request['condition_sha256']:raise ValueError('OLD/FRESH condition mismatch')
            continue
        condition=old._condition(request,out)
        # Every condition is an existing DEV condition. Never permit the old
        # helper's fallback that would generate a new input reference.
        target=out/'references'/cid
        if not (target/'control.json').exists():
            if not old._legacy_pointer(request,condition,'legacy_control_reference','json'):
                raise ValueError('Original control reference required; no new reference allowed')
        if not (target/'quality.json').exists():
            if not old._legacy_pointer(request,condition,'legacy_quality_query','json'):
                raise ValueError('Original quality reference required; no GT query redraw allowed')
        ref=old._reference(out,request,condition);qb=old._queries(out,request,condition)
        seen[cid]=dict(condition_id=cid,condition_sha256=request['condition_sha256'],
            control_npz_sha256=digest(target/'control.npz'),quality_npz_sha256=digest(target/'quality.npz'),
            Q_Y2048_sha256=quality.array_sha(qb['Q_Y2048']),legacy_query_sha256=quality.array_sha(condition['legacy_query_points']),
            S_applicable=ref['audits']['gamma']['status']=='PASS',I_applicable=True,
            E_reference_points=int(np.any(ref['arrays']['GT_visibility']==control.VISIBLE,axis=1).sum()))
    if len(seen)!=8:raise ValueError('Exactly eight existing DEV conditions required')
    definition=dict(sources=identity,conditions=list(seen.values()),control_definition=control.frozen_definition(),
        quality_definition=quality.DEFINITIONS,new_GT_queries=0,model_calls=0,
        eligibility='Input-only immutable references; PARTIAL bounds and NA retained; all parents retained')
    freeze_json(out/'references/definition_and_eligibility.json',definition)
    return dict(status='PASS',conditions=8,seconds=time.perf_counter()-started,new_GT_queries=0)


def result_for(out,label,request):
    path=Path(out)/'runtime'/label/request['request_id']/'result.json'
    result=read_json(path) if path.exists() else {}
    raw=resolve(result['raw_path']) if result.get('raw_path') else path.with_name('raw.npz')
    return path,result,raw


def _cache_pointer(request,label):
    return request.get('historical_cache',{}).get(label)


def verify_historical(request,label,out=OUT):
    """Verify one old score, raw, Gaussian, C, references and vectors; no scoring."""
    if request['split']!='OLD' or label not in HISTORICAL:raise ValueError('Historical cache only for registered OLD labels')
    pointer=_cache_pointer(request,label)
    if not pointer:raise ValueError('Historical cache registration missing')
    joined=resolve(pointer['joined_path'])
    if digest(joined)!=pointer['joined_sha256']:raise ValueError('Historical joined score changed')
    saved=read_json(joined);row=saved['metrics'];identity=saved['identity']
    if row['status']!='PASS' or row.get('scoring_sources')!=old.identities():raise ValueError('Historical scientific identity not reusable')
    expected_id=pointer.get('source_request_id',request['request_id'])
    if row['request_id']!=expected_id or row['label']!=HISTORICAL[label]:raise ValueError('Historical label/request mismatch')
    if any(row.get(k)!=request.get(k) for k in ('uid','condition_id','candidate','noise_seed','N','K')):
        raise ValueError('Historical pair metadata mismatch')
    if identity['condition_sha256']!=request['condition_sha256'] or identity['bank_sha256']!=request['bank_sha256']:
        raise ValueError('Historical condition/noise file identity mismatch')
    checked={str(joined):digest(joined)}
    for name in ('raw','result','vectors','control_vectors'):
        path=resolve(pointer.get(name+'_path',row.get(name+'_path')))
        expected=pointer.get(name+'_sha256',row.get(name+'_sha256'))
        if not expected or digest(path)!=expected:raise ValueError('Historical '+name+' identity changed')
        checked[str(path)]=expected
    rawpath=resolve(pointer.get('raw_path',row['raw_path']));raw=old._adapted_arrays(rawpath)
    condition=old._condition(request,out);oldresult=read_json(resolve(pointer.get('result_path',row['result_path'])))
    facts=old._contracts(request,raw,condition,oldresult)
    if not all(facts.get(k) is True for k in gates.HARD):raise ValueError('Historical hard contract not established')
    ref=old._reference(out,request,condition);qb=old._queries(out,request,condition)
    if row['Q_Y2048_sha256']!=quality.array_sha(qb['Q_Y2048']) or row['legacy_query_sha256']!=quality.array_sha(condition['legacy_query_points']):
        raise ValueError('Historical query identity mismatch')
    if quality.array_sha(raw['raw'])!=row['geometry_sha256']:raise ValueError('Historical raw-array identity mismatch')
    checkpoint=pointer.get('checkpoint_sha256')
    if checkpoint and row['checkpoint_sha256']!=checkpoint:raise ValueError('Historical checkpoint identity mismatch')
    return row,dict(status='PASS',source_files=checked,source_request_id=expected_id,source_label=row['label'],
                    raw_and_C_and_noise_and_all51_and_exports=True,query_arrays_verified=True,
                    reference_identity=ref['identity'],scores_recomputed=0)


def evaluate_request(request,raw_path=None,out_dir=OUT,label='START'):
    out=assert_out(out_dir)
    if label not in LABELS or request['split'] not in PANELS:raise ValueError('Unknown registered label/panel')
    panel=request['split'];rid=request['request_id'];folder=out/'evaluation'/label/panel
    dest=folder/'scoring/joined'/(rid+'.json');tick=time.perf_counter()
    resultpath,result,registered_raw=result_for(out,label,request)
    rawpath=resolve(raw_path) if raw_path else registered_raw
    cache=_cache_pointer(request,label) if panel=='OLD' else None
    use_historical=bool(cache and result.get('status') not in ('SAMPLED','PASS','DONE'))
    identity=dict(version=VERSION,label=label,request_id=rid,panel=panel,condition_sha256=request['condition_sha256'],
        bank_sha256=request['bank_sha256'],sources=sources(),historical_joined_sha256=cache['joined_sha256'] if use_historical else None,
        raw_sha256=None if use_historical else digest(rawpath) if rawpath.exists() else None,
        result_sha256=digest(resultpath) if resultpath.exists() else None)
    if dest.exists():
        saved=read_json(dest)
        if saved['identity']!=identity:raise ValueError('Completed evaluation identity changed; preserve before repair')
        row=saved['metrics']
        for key in ('vectors_path','control_vectors_path'):
            if row.get(key) and digest(resolve(row[key]))!=row[key.replace('_path','_sha256')]:raise ValueError('Cached score vectors changed')
        return row
    base={k:request.get(k) for k in ('request_id','uid','condition_id','candidate','noise_seed','N','K')}
    base.update(label=label,panel=panel,split=panel,model_calls=0,human_usable=None,user_study='NOT_RUN')
    if use_historical:
        original,audit=verify_historical(request,label,out)
        row=dict(original,**base,historical_source_label=original['label'],historical_source_request_id=original['request_id'],
            generation_status='CACHED',generation_time_source='historical_generation',new_scoring_seconds=0.,
            historical_score_seconds=original.get('adapter_seconds'),cache_verification_seconds=time.perf_counter()-tick)
        freeze_json(dest,dict(identity=identity,metrics=quality.clean(row),cache_audit=audit))
        return quality.clean(row)
    if not rawpath.exists() or result.get('status') in ('FAILED','FAIL','BLOCKED','NOT_RUN'):
        failed=result.get('status') in ('FAILED','FAIL') or rawpath.with_name('failure.json').exists()
        row=dict(base,status='GENERATION_FAILED' if failed else 'NOT_RUN',reason=result.get('reason','Registered output not complete'))
        return quality.clean(quality.failure_metrics(row)) if failed else row
    if not result: return dict(base,status='NOT_RUN',reason='RAW has no completed result receipt')
    if result.get('raw_sha256')!=identity['raw_sha256']:raise ValueError('New RAW/result identity mismatch')
    condition=old._condition(request,out);raw=old._adapted_arrays(rawpath)
    facts=old._contracts(request,raw,condition,result)
    if any(facts.get(k) is False for k in gates.HARD):
        row=quality.clean(quality.failure_metrics(dict(base,**facts,status='INVALID_OUTPUT',reason='Hard coordinate/runtime/export contract failed')))
        freeze_json(dest,dict(identity=identity,metrics=row));return row
    # References must be prepared by one coordinator before parallel scoring.
    if not (out/'references'/request['condition_id']/'control.json').exists():raise ValueError('Run prepare_references before scoring')
    reference=old._reference(out,request,condition);qb=old._queries(out,request,condition)
    qrequest=dict(request,arm=label,panel=panel,panels=[panel],sensitivity_check=False)
    qrequest.pop('legacy_score_path',None)
    bridge=dict(result,status='SAMPLED',raw_path=str(rawpath),raw_sha256=identity['raw_sha256'])
    freeze_json(folder/'outputs'/rid/'result.json',bridge)
    namespace=dict(quality.__dict__);namespace.update(arrays=old._adapted_arrays,query_bank=lambda *_:qb)
    qscore=types.FunctionType(quality.score_request.__code__,namespace,quality.score_request.__name__,quality.score_request.__defaults__)
    qscore.__kwdefaults__=quality.score_request.__kwdefaults__
    row=qscore(folder,qrequest,cf_ff=True);row.update(base);row.update(facts)
    if row['status']=='PASS':
        value=control.score_control(raw['raw'][int(request['K']):].reshape(-1,3,3),condition,reference,cf_scan=old._cf_scan(row))
        cp=folder/'scoring/control'/rid;freeze_npz(cp.with_suffix('.npz'),value['arrays'])
        freeze_json(cp.with_suffix('.json'),dict(identity=identity,metrics=value['metrics'],statuses=value['statuses'],
            audit=value['audit'],vectors_sha256=digest(cp.with_suffix('.npz'))))
        row.update(value['metrics'],control_statuses=value['statuses'],control_vectors_path=str(cp.with_suffix('.npz')),
                   control_vectors_sha256=digest(cp.with_suffix('.npz')))
    row.update(new_scoring_seconds=time.perf_counter()-tick,adapter_seconds=time.perf_counter()-tick,
        scoring_sources=old.identities(),checkpoint_sha256=result.get('checkpoint_sha256'),
        result_path=str(resultpath),result_sha256=identity['result_sha256'],raw_path=str(rawpath),raw_sha256=identity['raw_sha256'],
        geometry_sha256=quality.array_sha(raw['raw']),generation_status='NEW_GENERATED',
        generation_time_source='actual_new_generation',generation_seconds=result.get('metadata',{}).get('seconds',result.get('seconds')))
    freeze_json(dest,dict(identity=identity,metrics=quality.clean(row)))
    return quality.clean(row)


def _job(args):
    request,out,label=args;tick=time.perf_counter()
    try:row=evaluate_request(request,out_dir=out,label=label)
    except Exception as exc:
        row={k:request.get(k) for k in ('request_id','uid','condition_id','candidate','noise_seed','N','K','split')}
        row.update(label=label,status='SCORING_BLOCKED',reason=repr(exc),traceback=traceback.format_exc())
    return row,time.perf_counter()-tick


def score_set(out=OUT,label='START',panel='OLD',*,workers=2,available_only=False):
    out=assert_out(out);requests=manifest(out,panel)
    if label not in LABELS or workers not in (1,2):raise ValueError('Invalid label/CPU worker count')
    if panel=='ALL':
        result=[score_set(out,label,p,workers=workers,available_only=available_only) for p in PANELS]
        rows=[]
        for p in PANELS:rows.extend(read_json(out/'evaluation'/label/p/'per_output.json'))
        elapsed=sum(s['invocation_seconds'] for s in result);actual_workers=max(s['cpu_workers'] for s in result)
    else:
        prepare_references(out);tick=time.perf_counter()
        jobs=[(r,str(out),label) for r in requests]
        if workers==1:values=list(map(_job,jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:values=list(executor.map(_job,jobs))
        rows=[r for r,_ in values];elapsed=time.perf_counter()-tick;actual_workers=workers
        audit=out/'evaluation'/label/panel/'invocations.jsonl';audit.parent.mkdir(parents=True,exist_ok=True)
        with audit.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),label=label,panel=panel,workers=workers,
                wall_seconds=elapsed,worker_seconds=sum(t for _,t in values),completed=sum(r['status']=='PASS' for r in rows),
                model_calls=0,available_only=available_only,denominator_retained=len(requests)))+'\n')
    folder=out/'evaluation'/label/panel
    summary=gates.summarize(rows,requests);summary.update(label=label,panel=panel,cpu_workers=actual_workers,
        invocation_seconds=elapsed,model_calls=0,historical_scores_reused=sum(r.get('generation_status')=='CACHED' for r in rows),
        new_scored_outputs=sum(r.get('generation_status')=='NEW_GENERATED' for r in rows),
        original_scoring_worker_seconds=sum(r.get('new_scoring_seconds',0) for r in rows))
    write_json(folder/'per_output.json',rows);write_csv(folder/'per_output_metrics.csv',rows)
    write_json(folder/'summary.json',summary);write_csv(folder/'per_parent_metrics.csv',summary['parents']);write_csv(folder/'per_noise_metrics.csv',summary['noise'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--prepare',action='store_true')
    p.add_argument('--label',choices=LABELS,default='START');p.add_argument('--panel',choices=(*PANELS,'ALL'),default='OLD')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--available-only',action='store_true');a=p.parse_args()
    result=prepare_references(a.out) if a.prepare else score_set(a.out,a.label,a.panel,workers=a.workers,available_only=a.available_only)
    print(json.dumps(quality.clean(result),ensure_ascii=False))
