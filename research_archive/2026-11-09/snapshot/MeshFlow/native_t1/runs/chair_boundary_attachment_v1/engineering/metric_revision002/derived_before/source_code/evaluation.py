"""CPU evaluation: original quality plus the three registered control metrics."""
from pathlib import Path
import argparse
import json
import time
import traceback
import numpy as np
from .common import OUT, ROOT, QUICK, HYBRID, LABELS, read_json, resolve, digest, array_hash, atomic_json, freeze_json, append_event
from .registration import _arrays, bits, freeze_npz
from experiments.mf_h1_local0.chair_control_usability_quick_v1 import scoring as quality

VERSION='boundary_attachment_evaluation_v1'


def load_condition(out, request):
    metadata={x['condition_id']:x for x in read_json(Path(out)/'registration/conditions.json')}[request['condition_id']]
    path=resolve(metadata['condition_npz'])
    if digest(path)!=metadata['condition_sha256']:raise ValueError('Condition file changed')
    condition=dict(metadata);condition.update(_arrays(path))
    return condition


def _start_result(out, request):
    path=Path(out)/'outputs'/request['request_id']/'result.json'
    if path.exists():return read_json(path)
    if request['status']!='CACHED':return None
    if request['cache_status']!='PASS' or digest(resolve(request['legacy_raw']))!=request['legacy_raw_sha256']:
        raise ValueError('Unverified START cache')
    result=dict(request,status='CACHED',metadata=request['legacy_metadata'],model_calls=0,
        raw_path=request['legacy_raw'],raw_sha256=request['legacy_raw_sha256'],
        raw_origin='historical_START_identity_verified',sampling_attempts=0)
    freeze_json(path,result)
    return result


def _historical_score_path(request):
    if request['label']!='START_HYBRID':return None
    if request.get('source_family')=='quick':
        path=QUICK/'scoring/outputs'/(request['source_request_id']+'.json')
        if path.exists():
            metrics=read_json(path)['metrics']
            return metrics.get('legacy_score_path')
    if request.get('legacy_case_id'):
        path=HYBRID/'evaluation/OT_HYBRID/scoring/scores'/(request['legacy_case_id']+'.json')
        if path.exists():return str(path)
    return None


def reference_for(out, condition, density):
    from . import control_metrics as control
    directory=Path(out)/'scoring/control_references'
    name=condition['condition_id']+'__'+density
    path=directory/(name+'.npz');meta_path=directory/(name+'.json')
    expected=dict(condition_sha256=condition['condition_sha256'],density=density,
        source_identity=control.source_identity(),definition=control.frozen_definition())
    if meta_path.exists():
        metadata=read_json(meta_path)
        if metadata['cache_identity']!=expected or digest(path)!=metadata['npz_sha256']:
            raise ValueError('Frozen control reference identity differs')
        reference=dict(metadata['reference']);reference['arrays']=_arrays(path)
        for key,value in reference['arrays'].items():
            if array_hash(value)!=metadata['arrays_sha256'][key]:raise ValueError('Frozen reference array changed')
        return reference,metadata
    tick=time.perf_counter()
    reference=control.prepare_reference(condition,condition['condition_id'],density=density)
    freeze_npz(path,reference['arrays'])
    payload={k:v for k,v in reference.items() if k!='arrays'}
    metadata=dict(cache_identity=expected,reference=payload,npz_path=str(path),npz_sha256=digest(path),
        arrays_sha256={k:array_hash(v) for k,v in reference['arrays'].items()},seconds=time.perf_counter()-tick)
    freeze_json(meta_path,metadata)
    return reference,metadata


def _cf_scan(quality_row):
    if quality_row.get('legacy_score_path'):
        path=resolve(quality_row['legacy_score_path'])
        if path.exists():
            data=read_json(path);data=data.get('score',data)
            value=data.get('detail',{}).get('CF')
            if value and value.get('independent_verification') is True:return value
    return None


def _control_score(out,request,condition,free,quality_row,density):
    from . import control_metrics as control
    base=Path(out)/'scoring/control'/density/request['request_id']
    path=base.with_suffix('.json');vectors=base.with_suffix('.npz')
    identity=dict(raw_sha256=quality_row['raw_sha256'],condition_sha256=condition['condition_sha256'],
        source_identity=control.source_identity(),definition=control.frozen_definition(),density=density)
    if path.exists():
        saved=read_json(path)
        if saved['identity']!=identity or digest(vectors)!=saved['vectors_sha256']:
            raise ValueError('Control score cache identity differs')
        return saved
    reference,ref_meta=reference_for(out,condition,density)
    value=control.score_control(free,condition,reference,cf_scan=_cf_scan(quality_row))
    freeze_npz(vectors,value['arrays'])
    saved=dict(identity=identity,metrics=value['metrics'],statuses=value['statuses'],audit=value['audit'],
        vectors_path=str(vectors),vectors_sha256=digest(vectors),
        reference_path=ref_meta['npz_path'],reference_sha256=ref_meta['npz_sha256'],
        arrays_sha256={k:array_hash(v) for k,v in value['arrays'].items()})
    freeze_json(path,saved)
    return saved


def score_request(out,request,*,quality_only=False):
    out=Path(out);row={k:request.get(k) for k in ('request_id','base_request_id','label','arm','uid','panel',
        'condition_id','context_level','candidate','noise_index','N','K','L0')}
    if request['status']=='NOT_APPLICABLE':
        return dict(row,status='NOT_APPLICABLE',generation_status='NOT_APPLICABLE',reason=request.get('reason'))
    result=_start_result(out,request)
    if result is None:
        p=out/'outputs'/request['request_id']/'result.json'
        result=read_json(p) if p.exists() else None
    if not result:return dict(row,status='BLOCKED',generation_status='NOT_RUN',reason='Registered output not complete')
    qrequest=dict(request,sensitivity_check=False)
    historical=_historical_score_path(request)
    if historical:qrequest['legacy_score_path']=historical
    q=quality.score_request(out,qrequest,cf_ff=True)
    row.update(q);row['label']=request['label'];row['base_request_id']=request['base_request_id']
    if q.get('status')!='PASS':return row
    condition=load_condition(out,request)
    raw=_arrays(resolve(result['raw_path']))['raw']
    n,k=int(condition['N']),int(condition['K'])
    shape=raw.shape==(n,9) and raw.dtype==np.float32
    preserved=shape and bits(raw[:k],condition['C'].reshape(k,9))
    row.update(C_preserved=bool(preserved),N_preserved=bool(shape),finite=bool(np.isfinite(raw).all()))
    if not shape or not preserved or not row['finite']:
        return dict(row,status='INVALID_OUTPUT',reason='Exact FP32 N/C/finite check failed')
    free=raw[k:].reshape(-1,3,3)
    gt=condition['GT_full'].reshape(-1,3).astype(np.float64)
    lo,hi=gt.min(0),gt.max(0);center=(lo+hi)/2;half=max((hi-lo).max()/2,1e-5)*1.08
    row['free_out_of_GT_cube_corner_fraction']=float((np.abs(free.reshape(-1,3)-center)>half).any(1).mean())
    meta=result.get('metadata') or result.get('legacy_metadata') or {}
    row['peak_cuda_allocated_bytes']=meta.get('peak_cuda_allocated_bytes')
    row['peak_cuda_reserved_bytes']=meta.get('peak_cuda_reserved_bytes')
    row['sampling_peak_scope']=meta.get('peak_scope')
    if quality_only:return dict(row,status='QUALITY_COMPLETE_CONTROL_NOT_RUN')
    value=_control_score(out,request,condition,free,q,'main')
    row.update(value['metrics'])
    row.update(control_vectors_path=value['vectors_path'],control_vectors_sha256=value['vectors_sha256'],
        control_reference_path=value['reference_path'],control_reference_sha256=value['reference_sha256'],
        control_score_path=str((out/'scoring/control/main'/request['request_id']).with_suffix('.json')),
        status='PASS')
    frozen_ids={x['request_id'] for x in read_json(out/'registration/sensitivity_units.json')}
    if request['request_id'] in frozen_ids:
        dense=_control_score(out,request,condition,free,q,'dense')
        row.update({'dense_'+k:v for k,v in dense['metrics'].items()})
        row.update(sensitivity_status='COMPLETE',dense_control_vectors_path=dense['vectors_path'],
            dense_control_reference_path=dense['reference_path'])
    atomic_json(out/'scoring/joined'/(request['request_id']+'.json'),row)
    return row


def run(out=OUT,*,label=None,available_only=False,quality_only=False):
    from . import control_metrics as control
    out=Path(out)
    if label is not None and label not in LABELS:raise ValueError('Unregistered label')
    rows=[];requests=read_json(out/'requests.json')
    atomic_json(out/'scoring/definitions.json',dict(version=VERSION,new_control=control.frozen_definition(),
        original_quality=quality.DEFINITIONS,original_quality_source=digest(quality.__file__),
        new_control_sources=control.source_identity(),human_ratings='NOT_RUN',model_forwards=0,
        panel_denominators=dict(P1=32,P2=8,P3=16,P4=16,P5=8),evaluation_script_sha256=digest(__file__)))
    for request in requests:
        if label is not None and request['label']!=label:continue
        if available_only and request['status']!='CACHED' and request['status']!='NOT_APPLICABLE' and not (out/'outputs'/request['request_id']/'result.json').exists():continue
        tick=time.perf_counter()
        try:row=score_request(out,request,quality_only=quality_only)
        except Exception as exc:
            row={k:request.get(k) for k in ('request_id','base_request_id','label','arm','panel','uid','condition_id','candidate','noise_index')}
            row.update(status='SCORING_BLOCKED',reason=repr(exc))
            atomic_json(out/'scoring/errors'/(request['request_id']+'.json'),dict(row,traceback=traceback.format_exc()))
        rows.append(row)
        append_event(out/'scoring/cpu_ledger.jsonl',dict(event='SCORE_RETURN',request_id=request['request_id'],
            status=row['status'],seconds=time.perf_counter()-tick,model_forwards=0,backwards=0,optimizer_updates=0))
        print('CPU',request['label'],len(rows),request['request_id'],row['status'],flush=True)
    suffix=label or 'all'
    atomic_json(out/'scoring'/('per_output_'+suffix+'.json'),rows)
    quality.write_csv(out/('per_output_metrics_'+suffix+'.csv'),rows)
    # Merge only completed, persisted per-output joined scores; never erase other arms.
    combined=[]
    for request in requests:
        p=out/'scoring/joined'/(request['request_id']+'.json')
        if p.exists():combined.append(read_json(p))
        elif request['status']=='NOT_APPLICABLE':
            combined.append({**{k:request.get(k) for k in ('request_id','base_request_id','label','arm','panel','uid','condition_id','candidate','noise_index')},
                'status':'NOT_APPLICABLE','reason':request.get('reason')})
        else:
            current=next((r for r in rows if r['request_id']==request['request_id']),None)
            combined.append(current or {**{k:request.get(k) for k in ('request_id','base_request_id','label','arm','panel','uid','condition_id','candidate','noise_index')},'status':'BLOCKED','reason':'Not scored yet'})
    atomic_json(out/'scoring/per_output.json',combined);quality.write_csv(out/'per_output_metrics.csv',combined)
    summary=dict(status='COMPLETE' if all(x['status'] in ('PASS','NOT_APPLICABLE') for x in rows) else 'INCOMPLETE',
        label=label,rows=len(rows),passed=sum(x['status']=='PASS' for x in rows),
        not_applicable=sum(x['status']=='NOT_APPLICABLE' for x in rows),
        scoring_blocked=sum(x['status']=='SCORING_BLOCKED' for x in rows),model_forwards=0)
    atomic_json(out/'scoring'/('summary_'+suffix+'.json'),summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--label',choices=LABELS)
    parser.add_argument('--available-only',action='store_true');parser.add_argument('--quality-only',action='store_true')
    a=parser.parse_args();print(json.dumps(run(a.out,label=a.label,available_only=a.available_only,quality_only=a.quality_only)))
