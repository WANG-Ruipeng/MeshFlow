"""Independent read-only call/output audit. No torch, model or score execution.

Live snapshots tolerate one incomplete final JSONL line and in-flight calls.
Final audits require closed calls/occupancy and verify protected file bytes.
Checkpoint tensors are not opened: saved validation metadata is explicitly the
evidence for AdamW/model state; only the protected original START is hashed once
per final invocation. Device VRAM is never reported as process memory.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from . import assets

OUT=assets.OUT
KINDS=('training_forward','training_backward','optimizer_update','sampling','sampling_forward','precheck_forward','precheck_backward')
LIMITS=dict(sampling=512,train_feedback_rollouts=128,sampling_forward=25600,optimizer_update=6000,
            sample_equivalent_forwards=160000,precheck_forward=64,precheck_backward=32,gpu_occupancy_seconds=28800)
FORWARDS=('training_forward','sampling_forward','precheck_forward')


def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def _rows(path,live,issues):
    if not path.exists():return [],None
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);rows=[]
    for i,line in enumerate(lines):
        try:
            if not line.strip():continue
            row=json.loads(line)
            if i==len(lines)-1 and not line.endswith(b'\n'):
                issues.append(dict(kind='UNFINISHED_LAST_LINE',path=str(path),line=i+1,allowed_in_live=live))
                if live:break
            rows.append(row)
        except (ValueError,UnicodeDecodeError):
            allowed=live and i==len(lines)-1 and not line.endswith(b'\n')
            issues.append(dict(kind='INVALID_JSONL',path=str(path),line=i+1,allowed_in_live=allowed))
            if not allowed:raise ValueError('Malformed ledger JSONL: '+str(path)+':'+str(i+1))
    return rows,dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),rows=len(rows))


def _cell():return {key:Counter() for key in ('attempt','return','fail','pending')}


def _utc(timestamp):return datetime.fromtimestamp(timestamp,timezone.utc).isoformat() if timestamp is not None else None


def audit_calls(out,live,issues):
    events=[];sources=[]
    for p in sorted((out/'ledger').glob('calls_*.jsonl')):
        rows,source=_rows(p,live,issues);sources.append(source)
        for row in rows:row['_stage']=p.stem[len('calls_'):];events.append(row)
    events.sort(key=lambda r:(r.get('time_unix',0),0 if r.get('event')=='CALL_BEGIN' else 1))
    begins={};ends={};total=_cell();arms=defaultdict(_cell);stages=defaultdict(_cell);sample_groups=defaultdict(_cell)
    sample_ids=defaultdict(list);times=Counter();failures=[]
    for r in events:
        cid=r.get('call_id');kind=r.get('kind');event=r.get('event');context=r.get('context',{})
        if kind not in KINDS:issues.append(dict(kind='UNKNOWN_CALL_KIND',event=r));continue
        arm=context.get('phase','NA')+'__'+context.get('arm',context.get('label','NA'))
        scope=[total,arms[arm],stages[r['_stage']]]
        if context.get('label') and context.get('split'):scope.append(sample_groups[(context['label'],context['split'])])
        if event=='CALL_BEGIN':
            if cid in begins:issues.append(dict(kind='DUPLICATE_BEGIN',call_id=cid))
            begins[cid]=r
            for group in scope:group['attempt'][kind]+=1
            if kind=='sampling':sample_ids[(context.get('label'),context.get('request_id'))].append(cid)
        elif event in ('CALL_RETURN','CALL_FAILED'):
            if cid not in begins or cid in ends:issues.append(dict(kind='UNMATCHED_OR_DUPLICATE_END',call_id=cid))
            if cid in begins and (begins[cid]['kind']!=kind or begins[cid].get('context')!=context):
                issues.append(dict(kind='CALL_END_IDENTITY_MISMATCH',call_id=cid))
            ends[cid]=r;field='return' if event=='CALL_RETURN' else 'fail'
            for group in scope:group[field][kind]+=1
            times[kind]+=r.get('wall_seconds',0)
            if field=='fail':failures.append(r)
        else:issues.append(dict(kind='UNKNOWN_EVENT',event=r))
    pending=[]
    for cid,r in begins.items():
        if cid in ends:continue
        pending.append(r);kind=r['kind'];context=r.get('context',{})
        arm=context.get('phase','NA')+'__'+context.get('arm',context.get('label','NA'))
        for group in (total,arms[arm],stages[r['_stage']]):group['pending'][kind]+=1
        if context.get('label') and context.get('split'):sample_groups[(context['label'],context['split'])]['pending'][kind]+=1
    if pending and not live:issues.append(dict(kind='PENDING_REAL_CALLS_IN_FINAL',count=len(pending)))
    equivalents=sum(r.get('batch_size',1) for r in begins.values() if r['kind'] in FORWARDS)
    feedback=sum(r['kind']=='sampling' and r.get('context',{}).get('split')=='TRAIN_FEEDBACK' for r in begins.values())
    def serial(cell):return {field:{kind:cell[field][kind] for kind in KINDS} for field in cell}
    public=dict(total=serial(total),by_arm={k:serial(v) for k,v in arms.items()},
        by_stage={k:serial(v) for k,v in stages.items()},
        by_label_split=[dict(label=k[0],split=k[1],**serial(v)) for k,v in sorted(sample_groups.items())],
        sample_equivalent_forwards=equivalents,train_feedback_rollouts=feedback,
        pending=pending,model_call_failures=failures,call_wall_seconds=dict(times),
        timing_warning='sampling wall seconds already contain its nested sampling_forward calls. Neither these sums nor F/B/update seconds are GPU occupancy.',
        first_call_unix=min((r['time_unix'] for r in events),default=None),
        last_observed_call_unix=max((r['time_unix'] for r in events),default=None),sources=sources)
    return public,begins,ends,sample_ids


def audit_occupancy(out,live,issues,now):
    rows,source=_rows(out/'ledger/gpu_occupancy.jsonl',live,issues);opened={};intervals=[]
    for r in rows:
        key=(r.get('stage'),r.get('pid'))
        if r['event']=='GPU_ACQUIRE':
            if key in opened:issues.append(dict(kind='REPEATED_GPU_ACQUIRE',key=key))
            opened[key]=r
        elif r['event']=='GPU_RELEASE':
            begin=opened.pop(key,None)
            if begin is None:issues.append(dict(kind='UNMATCHED_GPU_RELEASE',row=r));continue
            intervals.append(dict(stage=key[0],pid=key[1],start=begin['time_unix'],end=r['time_unix'],
                measured_seconds=r['seconds'],status=r['status'],complete=True))
    for key,r in opened.items():intervals.append(dict(stage=key[0],pid=key[1],start=r['time_unix'],end=now,
        measured_seconds=max(0,now-r['time_unix']),status='RUNNING',complete=False))
    if opened and not live:issues.append(dict(kind='OPEN_GPU_OCCUPANCY_IN_FINAL',count=len(opened)))
    points=[]
    for x in intervals:points.extend([(x['start'],1),(x['end'],-1)])
    count=maximum=0;previous=None;union=0.
    for t,delta in sorted(points,key=lambda x:(x[0],x[1])):
        if previous is not None and count:union+=t-previous
        count+=delta;maximum=max(maximum,count);previous=t
    if maximum>1:issues.append(dict(kind='MULTIPLE_GPU_OWNERS',maximum=maximum))
    return dict(intervals=intervals,maximum_concurrent_GPU_workers=maximum,
        measured_closed_seconds=sum(x['measured_seconds'] for x in intervals if x['complete']),
        including_live_open_seconds=sum(x['measured_seconds'] for x in intervals),
        timestamp_union_seconds=union,open_workers=len(opened),source=source,
        note='Occupancy is the locked GPU-owner lifetime, including load/save/CPU waits. Do not add forward timings to this value.')


def audit_outputs(out,sample_ids,begins,ends,live,issues):
    requests={}
    for pool in ('TRAIN_FEEDBACK','DEV','CONFIRM'):
        p=out/'manifests'/(pool+'.json')
        if p.exists():requests.update({r['request_id']:r for r in assets.read_json(p)['rows']})
    rows=[];saved=set()
    for p in sorted((out/'runtime').glob('*/*/result.json')):
        result=assets.read_json(p);label=result['label'];rid=result['request_id'];request=requests[rid]
        raw=p.with_name('raw.npz');rp=dict(label=label,split=request['split'],request_id=rid,
            status=result['status'],result_sha256=assets.digest(p),raw_exists=raw.exists(),
            result_path=str(p),checkpoint_sha256=result.get('checkpoint_sha256'))
        facts={}
        try:
            assert result['status']=='PASS'
            assert raw.exists() and assets.digest(raw)==result['raw_sha256']
            z=assets.arrays(raw);bank=assets.arrays(assets.resolve(request['bank_npz']))
            assert assets.digest(assets.resolve(request['bank_npz']))==request['bank_sha256']
            n,k=request['N'],request['K'];c=bank['C'].reshape(k,9);output=z['output'];path=z['path']
            facts['N']=output.dtype==np.float32 and output.shape==(n,9)
            facts['finite']=bool(np.isfinite(output).all() and np.isfinite(path).all())
            facts['C_all51']=path.dtype==np.float32 and path.shape==(51,n,9) and all(assets.bits(s[:k],c) for s in path)
            facts['endpoint_matches_path']=assets.bits(path[-1],output)
            facts['noise']=assets.bits(z['initial_gaussian'],bank['z'])
            export=result['export'];npz=assets.resolve(export['npz']);ply=assets.resolve(export['ply'])
            assert assets.digest(npz)==export['npz_sha256'] and assets.digest(ply)==export['ply_sha256']
            ex=assets.arrays(npz);mesh,faces=assets.read_binary_ply(ply)
            facts['NPZ_output']=assets.bits(ex['output'].reshape(n,9),output)
            facts['PLY_output']=assets.bits(mesh.reshape(n,9),output)
            facts['NPZ_C']=assets.bits(ex['output'][ex['known_mask']].reshape(k,9),c)
            facts['PLY_C']=assets.bits(mesh[faces['known'].astype(bool)].reshape(k,9),c)
            facts['recorded_NFE50']=result['metadata']['nfe']==50
            facts['all']=all(facts.values())
            if not facts['all']:issues.append(dict(kind='RAW_CONTRACT_FAILURE',request_id=rid,label=label,facts=facts))
            rp.update(raw_sha256=result['raw_sha256'],recorded_NFE=result['metadata']['nfe'],
                generation_seconds=result['metadata'].get('seconds'),
                cuda_peak_allocated_bytes=result['metadata'].get('peak_cuda_allocated_bytes'),
                cuda_peak_reserved_bytes=result['metadata'].get('peak_cuda_reserved_bytes'),facts=facts)
            saved.add((label,rid))
        except Exception as exc:
            rp.update(audit_error=repr(exc),facts=facts);issues.append(dict(kind='OUTPUT_AUDIT_ERROR',path=str(p),reason=repr(exc)))
        rows.append(rp)
    discrepancies=[];failures=[];retries=0
    for key,cids in sample_ids.items():
        label,rid=key;folder=out/'runtime'/label/rid;complete=[cid for cid in cids if ends.get(cid,{}).get('event')=='CALL_RETURN']
        missing=len(complete)-int(key in saved)
        failure=assets.read_json(folder/'failure.json') if (folder/'failure.json').exists() else None
        used=assets.read_json(folder/'engineering_retry_used.json') if (folder/'engineering_retry_used.json').exists() else None
        authorization=assets.read_json(folder/'engineering_retry_authorization.json') if (folder/'engineering_retry_authorization.json').exists() else None
        if failure:failures.append(dict(label=label,request_id=rid,error=failure.get('error'),retry_used=used is not None))
        if used:retries+=1
        if len(cids)>1:
            if not used or not authorization or len(cids)-1>authorization.get('max_retries',0):
                issues.append(dict(kind='UNAUTHORIZED_OR_EXCESS_ROLLOUT_RETRY',label=label,request_id=rid,attempts=len(cids)))
        if missing:
            known=used.get('completed_lost_rollouts',0) if used else 0
            state='DOCUMENTED_IO_LOSS' if missing==known and failure else 'LIVE_SAVE_PENDING' if live and missing>0 else 'UNEXPLAINED_MISSING_OUTPUT'
            discrepancies.append(dict(label=label,request_id=rid,returned_rollouts=len(complete),saved_raw=int(key in saved),
                completed_without_retained_output=missing,status=state,return_call_ids=complete,
                documented_lost_NFE=used.get('completed_lost_NFE') if used else None))
            if state=='UNEXPLAINED_MISSING_OUTPUT':issues.append(dict(kind=state,label=label,request_id=rid,count=missing))
    for key in saved:
        if key not in sample_ids:issues.append(dict(kind='RAW_WITHOUT_CALL_LEDGER',label=key[0],request_id=key[1]))
    groups=[]
    for label,pool in sorted({(r['label'],r['split']) for r in rows}):
        group=[r for r in rows if r['label']==label and r['split']==pool]
        times=[r['generation_seconds'] for r in group if isinstance(r.get('generation_seconds'),(int,float))]
        groups.append(dict(label=label,split=pool,retained_outputs=len(group),
            complete_contract_outputs=sum(r.get('facts',{}).get('all',False) for r in group),
            retained_recorded_NFE=sum(r.get('recorded_NFE',0) for r in group),
            generation_seconds_sum=sum(times),generation_seconds_mean=statistics.mean(times) if times else None))
    return dict(retained_outputs=len(rows),rows=rows,by_label_split=groups,discrepancies=discrepancies,
        completed_without_retained_output=sum(x['completed_without_retained_output'] for x in discrepancies),
        engineering_or_scientific_failure_files=failures,failure_file_count=len(failures),engineering_retries_used=retries,
        note='Model-call failures and post-return save failures are separate. A returned rollout lost before saving remains charged, even after successful same-input retry.')


def audit_checkpoints(out,begins,ends,issues):
    steps=defaultdict(set)
    for cid,r in begins.items():
        if r['kind']=='optimizer_update' and ends.get(cid,{}).get('event')=='CALL_RETURN':
            c=r['context'];steps[(c['phase'],c['arm'])].add(c['pilot_step'])
    values=[]
    for p in sorted((out/'checkpoints').glob('*.json')):
        side=assets.read_json(p)
        if 'metadata' not in side:continue
        m=side['metadata'];step=m['feedback_step'];key=(m['phase'],m['arm']);pt=assets.resolve(side['path'])
        checks=dict(file_exists=pt.exists(),step_consistent=m['conditional_total_step']==5000+step,
            source_START=m['source_checkpoint_sha256']==assets.START_SHA,lambda_correct=m['lambda_variance']==.25,
            geo=m['context_encoder']=='geo',full_recovery=m['full_recovery_state'] is True,
            optimizer_not_reset=m['optimizer_reset'] is False,update_returned=step in steps[key] or step==0,
            state_identities_recorded=all(m.get(k) for k in ('model_state_sha256','optimizer_state_sha256','rng_sha256','stream_state_sha256')))
        if not all(checks.values()):issues.append(dict(kind='CHECKPOINT_METADATA_MISMATCH',path=str(p),checks=checks))
        values.append(dict(sidecar_path=str(p),sidecar_sha256=assets.digest(p),path=str(pt),
            checkpoint_sha256=side['checkpoint_sha256'],metadata=m,checks=checks,
            tensor_inspection='NOT_REPEATED: checkpoint.save/load validate full model/AdamW/RNG/stream and optimizer step 5000+feedback_step before these receipts; this audit reads metadata only'))
    return dict(rows=values,saved_checkpoint_count=len(values),returned_update_steps={phase+'__'+arm:dict(count=len(v),max=max(v,default=0)) for (phase,arm),v in steps.items()},
                large_checkpoint_byte_hashes_recomputed=0)


def audit_resources(out,live,issues):
    stages=[]
    for p in sorted((out/'resources').glob('*_monitor.jsonl')):
        rows,source=_rows(p,live,issues);device=[];own=[];util=[];nonnumeric=empty=0
        for r in rows:
            try:
                parts=r['gpu']['stdout'].split(',');device.append(float(parts[2]));util.append(float(parts[4]))
            except (KeyError,ValueError,IndexError):pass
            text=r.get('processes',{}).get('stdout','')
            if not text:empty+=1
            for line in text.splitlines():
                parts=line.split(',')
                if parts[0].strip()==str(r.get('worker_pid')):
                    try:own.append(float(parts[-1].strip()))
                    except ValueError:nonnumeric+=1
        stages.append(dict(stage=p.stem.removesuffix('_monitor'),telemetry_samples=len(rows),
            device_used_peak_MiB=max(device,default=None),own_process_peak_MiB=max(own,default=None),
            own_process_numeric_samples=len(own),own_process_nonnumeric_samples=nonnumeric,empty_process_rows=empty,
            device_utilization_max=max(util,default=None),source=source))
    return dict(stages=stages,maximum_device_used_MiB=max((s['device_used_peak_MiB'] for s in stages if s['device_used_peak_MiB'] is not None),default=None),
        maximum_observed_own_process_MiB=max((s['own_process_peak_MiB'] for s in stages if s['own_process_peak_MiB'] is not None),default=None),
        warning='Global device usage includes unrelated processes. Missing WSL/WDDM process accounting is NA, not zero; PyTorch allocated and reserved are separate, not summed.')


def audit_phases(out,live,issues):
    stages=[]
    for p in sorted((out/'training').glob('*_metrics.jsonl')):
        rows,source=_rows(p,live,issues)
        stages.append(dict(kind='training',stage=p.stem.removesuffix('_metrics'),completed_metric_rows=len(rows),
            primary_plus_aux_forward_seconds=sum(r.get('forward_seconds',0) for r in rows),
            backward_seconds=sum(r.get('backward_seconds',0) for r in rows),
            optimizer_seconds=sum(r.get('optimizer_seconds',0) for r in rows),
            update_wall_seconds=sum(r.get('update_total_seconds',0) for r in rows),
            OT_and_input_seconds=sum(r.get('OT_and_input_seconds',0) for r in rows),GPU_workers=1,
            cuda_allocated_peak_bytes=max((r.get('cuda_peak_allocated_bytes',0) for r in rows),default=None),
            cuda_reserved_peak_bytes=max((r.get('cuda_peak_reserved_bytes',0) for r in rows),default=None),
            peak_scope='per update reset; resident model+AdamW+backward+geometry; not process VRAM',source=source))
    for p in sorted((out/'evaluation').glob('*/*/summary.json')):
        r=assets.read_json(p);stages.append(dict(kind='CPU_scoring',label=r.get('label'),split=r.get('pool'),
            wall_seconds=r.get('invocation_seconds'),summed_original_worker_seconds=r.get('original_scoring_worker_seconds'),
            CPU_workers=r.get('cpu_workers'),completed=r.get('completed'),path=str(p),sha256=assets.digest(p),
            note='Workers overlap. Do not add their summed CPU worker seconds to wall time; a later cache invocation is separate.'))
    for p in sorted((out/'visuals').glob('*/manifest.json')):
        r=assets.read_json(p);stages.append(dict(kind='CPU_render',labels=r.get('labels'),split=r.get('split'),
            wall_seconds=r.get('wall_seconds'),mesh_render_seconds=r.get('mesh_render_seconds'),
            GL_workers=r.get('GL_workers'),CPU_threads=r.get('CPU_threads'),path=str(p),sha256=assets.digest(p)))
    return stages


def audit_protected(out,live,issues):
    path=out/'registration/source_files.json';registered=assets.read_json(path);rows=[]
    if len(registered)!=167:issues.append(dict(kind='PROTECTED_SOURCE_COUNT_CHANGED',actual=len(registered),expected=167))
    for name,record in registered.items():
        p=assets.resolve(name);sha=assets.digest(p) if p.exists() else None
        expected=record['sha256'] if isinstance(record,dict) else record
        row=dict(path=str(p),expected=expected,actual=sha,unchanged=sha==expected);rows.append(row)
        if not row['unchanged']:issues.append(dict(kind='PROTECTED_SOURCE_CHANGED',path=str(p)))
    start=assets.ROOT/'native_t1/runs/chair_hybrid_coupling_v1/checkpoints/OT_HYBRID_pilot1000_total5000.pt'
    if live:
        start_result=dict(path=str(start),expected_sha256=assets.START_SHA,status='DEFERRED_FINAL_BYTE_HASH',
            exists=start.exists(),bytes=start.stat().st_size if start.exists() else None,
            note='Avoid repeatedly streaming the protected large checkpoint during live training; final mode hashes its bytes once.')
    else:
        actual=assets.digest(start) if start.exists() else None
        start_result=dict(path=str(start),expected_sha256=assets.START_SHA,actual_sha256=actual,
            status='PASS' if actual==assets.START_SHA else 'FAIL',full_byte_hash_passes=1)
        if actual!=assets.START_SHA:issues.append(dict(kind='PROTECTED_START_CHANGED'))
    return dict(source_file_count=len(rows),unchanged_source_file_count=sum(r['unchanged'] for r in rows),
        source_manifest_sha256=assets.digest(path),files=rows,START=start_result)


def run(out=OUT,*,live=False):
    out=assets.resolve(out);tick=time.perf_counter();now=time.time();issues=[]
    calls,begins,ends,sample_ids=audit_calls(out,live,issues)
    occupancy=audit_occupancy(out,live,issues,now)
    outputs=audit_outputs(out,sample_ids,begins,ends,live,issues)
    checkpoints=audit_checkpoints(out,begins,ends,issues)
    resources=audit_resources(out,live,issues);phases=audit_phases(out,live,issues)
    protected=audit_protected(out,live,issues)
    actual=dict(calls['total']['attempt'],sample_equivalent_forwards=calls['sample_equivalent_forwards'],
        train_feedback_rollouts=calls['train_feedback_rollouts'],gpu_occupancy_seconds=occupancy['including_live_open_seconds'])
    limits={k:dict(actual=actual[k],limit=v,within=actual[k]<=v) for k,v in LIMITS.items()}
    for k,v in limits.items():
        if not v['within']:issues.append(dict(kind='BUDGET_EXCEEDED',budget=k,**v))
    protocol=assets.read_json(out/'protocol.json');started=datetime.fromisoformat(protocol['start_utc'].replace('Z','+00:00')).timestamp()
    errors=[x for x in issues if not (live and x.get('allowed_in_live'))]
    report=dict(version='chair_feedback_accounting_audit_v1',mode='LIVE' if live else 'FINAL',
        status='BLOCKED' if errors else 'LIVE_CONSISTENT' if live else 'PASS',
        experiment_completion_not_asserted=True,created_utc=_utc(now),audit_seconds=time.perf_counter()-tick,
        calls=calls,outputs=outputs,checkpoints=checkpoints,GPU_occupancy=occupancy,resources=resources,phases=phases,
        budget_limits=limits,protected_sources=protected,issues=issues,
        wall=dict(protocol_start_utc=protocol['start_utc'],audit_snapshot_end_utc=_utc(now),elapsed_since_protocol_start_seconds=now-started,
            first_real_call_utc=_utc(calls['first_call_unix']),last_observed_real_call_utc=_utc(calls['last_observed_call_unix']),
            definition='Live/final snapshot wall is elapsed wall time, not sum of nested model/CPU/render durations.'),
        engineering_failure_count=outputs['failure_file_count'],model_call_failure_count=len(calls['model_call_failures']),
        no_new_model_calls=True,no_checkpoint_loaded=True,no_geometric_scoring=True,
        script_sha256=assets.digest(__file__))
    folder=out/'derived/audit';name='live' if live else 'final';stamp=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    _write(folder/(name+'_'+stamp+'.json'),report);_write(folder/(name+'_latest.json'),report)
    print(json.dumps(dict(status=report['status'],mode=report['mode'],attempts=calls['total']['attempt'],
        pending=len(calls['pending']),retained_outputs=outputs['retained_outputs'],
        completed_without_retained_output=outputs['completed_without_retained_output'],
        engineering_failures=report['engineering_failure_count'],model_call_failures=report['model_call_failure_count'],
        protected_sources_unchanged=protected['unchanged_source_file_count'],issues=len(errors),
        path=str(folder/(name+'_latest.json')))))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));parser.add_argument('--live',action='store_true')
    args=parser.parse_args();result=run(args.out,live=args.live)
    raise SystemExit(0 if result['status']!='BLOCKED' else 2)
