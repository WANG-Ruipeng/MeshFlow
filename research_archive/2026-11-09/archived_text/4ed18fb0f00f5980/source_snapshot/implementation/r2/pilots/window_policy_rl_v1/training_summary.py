"""Model-free snapshots of original online training/accounting evidence.

Reads small JSON/JSONL only: no checkpoint deserialization, model construction,
GPU query, rescoring, or semantic review. Completed logical work and physical
attempts/returns remain separate, including partial and failed batches.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter,defaultdict
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import uuid
from .bootstrap import bind,OUT
D,E=bind()

VERSION='window_policy_rl_model_free_training_summary_v1'
DEST=OUT/'analysis'
METRICS=('loss','gradient_norm_before_clip','gradient_norm_after_clip','adapter_delta_l2',
         'preupdate_old_logp_max_abs_error','postupdate_trajectory_KL_mean',
         'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')


class Inputs:
    def __init__(self):
        self.sources={};self.cache={};self.notes=[]
    def blob(self,path):
        path=D.resolve(path);key=str(path)
        if key in self.cache:return self.cache[key]
        if not path.exists():return None
        before=path.stat();data=path.read_bytes();after=path.stat()
        self.sources[key]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),mtime_ns=after.st_mtime_ns)
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            self.notes.append(dict(kind='CHANGED_DURING_READ',path=key,read_bytes=len(data)))
        self.cache[key]=data
        return data
    def read(self,path):
        data=self.blob(path)
        if data is None:return None
        try:return json.loads(data.decode('utf-8-sig'))
        except (UnicodeError,json.JSONDecodeError) as exc:
            self.notes.append(dict(kind='PARTIAL_OR_INVALID_JSON',path=str(path),error=str(exc)))
            return None
    def events(self,path):
        data=self.blob(path)
        if data is None:return []
        rows=[];lines=data.splitlines(keepends=True)
        for index,line in enumerate(lines):
            if not line.strip():continue
            if not line.endswith(b'\n'):
                self.notes.append(dict(kind='UNTERMINATED_EVENT_TAIL',path=str(path),line=index+1,bytes=len(line)))
                continue
            try:rows.append(json.loads(line))
            except (UnicodeError,json.JSONDecodeError) as exc:
                self.notes.append(dict(kind='INVALID_EVENT_LINE',path=str(path),line=index+1,error=str(exc)))
        return rows


def counts(events):
    out=Counter()
    for event in events:
        name=event.get('event','MISSING')
        kind=event.get('kind','')
        out[name+(':'+kind if kind else '')]+=1
    return dict(out)


def numeric(value):
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def range_stats(values):
    values=[v for x in values if (v:=numeric(x)) is not None]
    return dict(count=len(values),minimum=min(values) if values else None,
        maximum=max(values) if values else None,mean=statistics.fmean(values) if values else None)


def event_counts(sample,d0,update):
    s,d,u=map(counts,(sample,d0,update))
    result=dict(rollout_attempts=s.get('ROLLOUT_ATTEMPT',0),rollout_returns=s.get('ROLLOUT_COMPLETE',0),
        sampling_F_attempts=s.get('FORWARD_START:sampling_forward',0),sampling_F_returns=s.get('FORWARD_RETURN:sampling_forward',0),
        D0_F_attempts=d.get('D0_FORWARD_ATTEMPT',0),D0_F_returns=d.get('D0_FORWARD_RETURN',0),
        backward_attempts=u.get('BACKWARD_START',0),backward_returns=u.get('BACKWARD_RETURN',0),
        optimizer_step_attempts=u.get('OPTIMIZER_STEP_START',0),optimizer_step_returns=u.get('OPTIMIZER_STEP_RETURN',0),
        optimizer_skip_events=u.get('OPTIMIZER_STEP_SKIPPED',0))
    for kind in ('training','postupdate_kl','zero_lora_validation'):
        result[kind+'_F_attempts']=u.get('FORWARD_START:'+kind,0)
        result[kind+'_F_returns']=u.get('FORWARD_RETURN:'+kind,0)
    result['all_model_F_attempts']=sum(result[k] for k in ('sampling_F_attempts','D0_F_attempts','training_F_attempts','postupdate_kl_F_attempts','zero_lora_validation_F_attempts'))
    result['all_model_F_returns']=sum(result[k] for k in ('sampling_F_returns','D0_F_returns','training_F_returns','postupdate_kl_F_returns','zero_lora_validation_F_returns'))
    return result


def resource_rows(value,batch,phase,path):
    if not value:return []
    snapshots=[value['initial']] if 'initial' in value else []
    snapshots+=value.get('samples',[])
    if 'final' in value:snapshots.append(value['final'])
    rows=[]
    for index,s in enumerate(snapshots):
        if 'monitor_error' in s:
            rows.append(dict(batch_index=batch,phase=phase,path=str(path),index=index,utc=s.get('utc'),monitor_error=s['monitor_error']))
            continue
        gpu_lines=(s.get('gpu',{}).get('stdout') or '').splitlines()
        fields=[x.strip() for x in gpu_lines[0].split(',')] if gpu_lines else []
        memory={}
        for line in (s.get('meminfo') or '').splitlines():
            if line.startswith(('MemTotal:','MemAvailable:')):memory[line.split(':')[0]]=int(line.split()[1])/1024
        process=[]
        for line in (s.get('processes',{}).get('stdout') or '').splitlines():
            f=[x.strip() for x in line.split(',')]
            if len(f)>=3:process.append(dict(pid=f[0],name=','.join(f[1:-1]),used_memory_MiB=numeric(f[-1]),raw_memory=f[-1]))
        rows.append(dict(batch_index=batch,phase=phase,path=str(path),index=index,utc=s.get('utc'),monitor_pid=s.get('pid'),
            gpu_name=fields[0] if len(fields)>=5 else None,
            global_GPU_total_MiB=numeric(fields[-4]) if len(fields)>=5 else None,
            global_GPU_used_MiB=numeric(fields[-3]) if len(fields)>=5 else None,
            global_GPU_free_MiB=numeric(fields[-2]) if len(fields)>=5 else None,
            global_GPU_utilization_percent=numeric(fields[-1]) if len(fields)>=5 else None,
            CPU_count=s.get('cpu_count'),load_average=s.get('load_average'),RAM_available_MiB=memory.get('MemAvailable'),
            GPU_processes=process,process_memory_NA_rows=sum(p['used_memory_MiB'] is None for p in process),
            scope='Global device observations, not model-owned VRAM; WDDM process memory can be NA'))
    return rows


def elapsed(start,end):
    if not start or not end:return None
    return (datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds()


def one_batch(inp,group,requests):
    batch=group['batch_index'];stage=OUT/'training'/f'batch_{batch:03d}'
    sample_events=inp.events(stage/'sample_ledger/calls.jsonl')
    d0_events=inp.events(stage/'d0_ledger/events.jsonl')
    update_events=inp.events(stage/'update/events.jsonl')
    count=event_counts(sample_events,d0_events,update_events)
    markers={name:inp.read(path) for name,path in dict(sample=stage/'sample_ledger/completion.json',
        d0=stage/'d0_ledger/completion.json',cpu=stage/'cpu_ledger/completion.json',update=stage/'update/completion.json').items()}
    sample_attempt=inp.read(stage/'sample_ledger/attempt.json')
    metrics=inp.read(stage/'update/training_metrics.json');checkpoint=inp.read(stage/'update/checkpoint.json')
    rewards=inp.read(stage/'rewards.json');failures=[]
    for phase,path in (('sample',stage/'sample_ledger/failure.json'),('d0',stage/'d0_ledger/failure.json'),
                       ('d0_phase',stage/'d0_ledger/phase/failure.json'),('cpu',stage/'cpu_ledger/failure.json'),('update',stage/'update/failure.json')):
        failure=inp.read(path)
        if failure:failures.append(dict(phase=phase,path=str(path),error=failure.get('error'),status=failure.get('status')))
    complete=all(value and value.get('status')=='COMPLETE' for value in markers.values())
    complete=bool(complete and metrics and metrics.get('status')=='COMPLETE' and checkpoint and checkpoint.get('status')=='COMPLETE')
    state='FAILED' if failures else 'COMPLETE' if complete else 'IN_PROGRESS' if stage.exists() else 'NOT_STARTED'
    q=requests[group['requests'][0]]
    row=dict(batch_index=batch,cycle=q['cycle'],condition_id=group['condition_id'],parent_id=q['parent_id'],
        status=state,registered_requests=8,logical_batch_complete=complete and not failures,
        failure_records=failures,**count)
    for phase,value in markers.items():
        row[phase+'_status']=value.get('status') if value else 'NOT_COMPLETE'
        row[phase+'_wall_seconds']=value.get('seconds') if value else None
    row['overall_start_utc']=(sample_attempt or {}).get('resources',{}).get('utc')
    row['overall_end_utc']=(markers['update'] or {}).get('utc') if row['logical_batch_complete'] else None
    row['overall_wall_seconds']=elapsed(row['overall_start_utc'],row['overall_end_utc'])
    row['phase_wall_seconds_sum']=sum(row[phase+'_wall_seconds'] or 0 for phase in markers)
    row['overall_scope']='First sampling admission snapshot to committed update phase completion; includes inter-phase handoff gaps'
    reward_values=[]
    if rewards:
        records=rewards['records']
        if [r['request_id'] for r in records]!=group['requests'] or any(r['reward'] not in (0,1) for r in records):
            raise ValueError('Original reward coverage/binary definition differs in batch '+str(batch))
        reward_values=[int(r['reward']) for r in records]
        row.update(reward_count=len(reward_values),reward_0=reward_values.count(0),reward_1=sum(reward_values),
            reward_mean=statistics.fmean(reward_values),active_reward_group=0<sum(reward_values)<8,
            inactive_kind='ALL_ZERO' if sum(reward_values)==0 else 'ALL_ONE' if sum(reward_values)==8 else 'ACTIVE')
    else:row.update(reward_count=0,reward_0=None,reward_1=None,reward_mean=None,active_reward_group=None,inactive_kind='PENDING')
    for name in METRICS:row[name]=metrics.get(name) if metrics else None
    row['actual_optimizer_updates_reported']=metrics.get('actual_optimizer_updates') if metrics else None
    row['optimizer_steps_cumulative']=metrics.get('optimizer_steps') if metrics else None
    row['committed_optimizer_updates']=int(metrics['actual_optimizer_updates']) if row['logical_batch_complete'] else 0
    row['opportunities_completed_checkpoint']=checkpoint.get('opportunities_completed') if checkpoint else None
    row['checkpoint_path']=checkpoint.get('path') if checkpoint else None
    row['checkpoint_sha256']=checkpoint.get('sha256') if checkpoint else None
    for name in ('current_hash_before','current_hash_after','old_hash','postupdate_conditional_KL_sum_by_trajectory','forward_counts'):
        row[name]=metrics.get(name) if metrics else None
    row['training_reported_seconds']=metrics.get('seconds') if metrics else None
    row['D0_model_loads_this_batch']=markers['d0'].get('resident_models_loaded') if markers['d0'] else None
    row['D0_load_seconds']=markers['d0'].get('load_seconds') if markers['d0'] else None
    for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
        row['D0_'+name]=markers['d0'].get(name) if markers['d0'] else None
    raw_receipts=[]
    for rid in group['requests']:
        receipt=inp.read(stage/'raw'/rid/'receipt.json')
        if receipt:raw_receipts.append(receipt)
    row['RAW_receipts_complete']=sum(r.get('status')=='COMPLETE' for r in raw_receipts)
    for name in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
        values=[r['sampling'][name] for r in raw_receipts if name in r.get('sampling',{})]
        row['sampling_'+name+'_max']=max(values) if values else None
    observed=[]
    for phase,path in (('sample',stage/'sample_ledger/resources.json'),('D0',stage/'d0_ledger/phase/resources.json'),
                       ('update',stage/'update/resources.json')):
        value=inp.read(path);observed+=resource_rows(value,batch,phase,path)
        row[phase+'_continuous_resource_record_present']=value is not None
    for phase in ('sample','D0','update'):
        own=[r for r in observed if r['phase']==phase]
        for metric in ('global_GPU_used_MiB','global_GPU_free_MiB','global_GPU_utilization_percent'):
            stats=range_stats(r.get(metric) for r in own)
            row[phase+'_'+metric+'_range']=stats
    violations=[]
    if row['logical_batch_complete']:
        expected=dict(rollout_attempts=8,rollout_returns=8,sampling_F_attempts=400,sampling_F_returns=400,
            D0_F_attempts=8,D0_F_returns=8,training_F_attempts=160,training_F_returns=160,
            postupdate_kl_F_attempts=160,postupdate_kl_F_returns=160,backward_attempts=160,backward_returns=160,
            zero_lora_validation_F_attempts=2 if batch==0 else 0,zero_lora_validation_F_returns=2 if batch==0 else 0)
        for key,value in expected.items():
            if row[key]!=value:violations.append(dict(batch_index=batch,field=key,expected=value,actual=row[key]))
        if row['active_reward_group'] is None:
            violations.append(dict(batch_index=batch,field='COMPLETED_BATCH_MISSING_REWARDS'))
            return row,observed,violations
        active=int(row['active_reward_group'])
        for key,value in (('optimizer_step_attempts',active),('optimizer_step_returns',active),
                          ('optimizer_skip_events',1-active),('actual_optimizer_updates_reported',active),
                          ('opportunities_completed_checkpoint',batch+1),('RAW_receipts_complete',8)):
            if row[key]!=value:violations.append(dict(batch_index=batch,field=key,expected=value,actual=row[key]))
        if not active and (row['adapter_delta_l2']!=0 or row['postupdate_trajectory_KL_mean']!=0):
            violations.append(dict(batch_index=batch,field='INACTIVE_ADAPTER_OR_KL_CHANGED'))
    return row,observed,violations


def csv_bytes(rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def write_csv(path,rows):
    payload=csv_bytes(rows);path=D.checked(path)
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Immutable training summary CSV differs')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(payload)


def run():
    if 'torch' in sys.modules:raise RuntimeError('Training summary must be model-free')
    inp=Inputs();training=inp.read(OUT/'inputs/training_manifest.json');config=inp.read(OUT/'training_config.json')
    if training is None or config is None:raise ValueError('Frozen training/config required')
    if training['batches']!=64 or len(training['training_requests'])!=512:raise ValueError('Expected frozen64/512 training panel')
    requests={q['request_id']:q for q in training['training_requests']}
    inp.blob(__file__)
    rows=[];resources=[];violations=[]
    for group in training['groups']:
        row,observed,errors=one_batch(inp,group,requests);rows.append(row);resources+=observed;violations+=errors
    committed=0
    for row in rows:
        if row['logical_batch_complete']:
            committed+=row['committed_optimizer_updates']
            if row['optimizer_steps_cumulative']!=committed:
                violations.append(dict(batch_index=row['batch_index'],field='CUMULATIVE_OPTIMIZER_CLOCK',expected=committed,actual=row['optimizer_steps_cumulative']))
    service_rows=[]
    for path in sorted((OUT/'training/resident_d0_sessions').glob('*/started.json')):
        directory=path.parent;start=inp.read(path);events=inp.events(directory/'events.jsonl');c=counts(events)
        closed=inp.read(directory/'closed.json');failure=inp.read(directory/'failure.json');model=inp.read(directory/'resident_model.json')
        service_rows.append(dict(service_dir=str(directory),pid=start.get('pid') if start else None,
            status='FAILED' if failure else 'CLOSED_EOF' if closed else 'ACTIVE_OR_NOT_YET_CLOSED',
            model_load_attempts=c.get('MODEL_LOAD_ATTEMPT',0),model_load_returns=c.get('MODEL_LOAD_RETURN',0),
            load_seconds=model.get('load_seconds') if model else None,
            mirrored_D0_F_attempts=c.get('D0_FORWARD_ATTEMPT',0),mirrored_D0_F_returns=c.get('D0_FORWARD_RETURN',0),
            D0_counter_scope='Mirrors per-batch D0 events; never add both to total',
            first_event_utc=events[0].get('utc') if events else None,last_event_utc=events[-1].get('utc') if events else None,
            failure=failure))
    conditions=[]
    for cid in sorted({r['condition_id'] for r in rows}):
        own=[r for r in rows if r['condition_id']==cid];done=[r for r in own if r['logical_batch_complete']]
        reward=[r for r in own if r['reward_count']==8]
        conditions.append(dict(condition_id=cid,parent_id=own[0]['parent_id'],registered_opportunities=len(own),
            completed_opportunities=len(done),committed_optimizer_updates=sum(r['committed_optimizer_updates'] for r in done),
            reward_completed_batches=len(reward),reward_observations=sum(r['reward_count'] for r in reward),
            reward_1=sum(r['reward_1'] for r in reward),reward_0=sum(r['reward_0'] for r in reward),
            active_reward_groups=sum(bool(r['active_reward_group']) for r in reward),
            all_zero_groups=sum(r['inactive_kind']=='ALL_ZERO' for r in reward),all_one_groups=sum(r['inactive_kind']=='ALL_ONE' for r in reward),
            cycles=[dict(batch_index=r['batch_index'],cycle=r['cycle'],reward_mean=r['reward_mean'],status=r['status']) for r in own],
            interpretation='Fresh noise and changing policy each cycle; descriptive online coverage, not a fixed-test improvement'))
    physical_keys=list(event_counts([],[],[]))
    totals={key:sum(row[key] for row in rows) for key in physical_keys}
    final=inp.read(OUT/'training/final.json');completed=sum(r['logical_batch_complete'] for r in rows)
    if final is not None and (final.get('status')!='COMPLETE' or final.get('batches')!=64 or final.get('rollouts')!=512):
        violations.append(dict(field='FINAL_CONTROLLER_SCHEMA',value=final.get('status')))
    finished=completed==64 and final is not None and final.get('status')=='COMPLETE' and not inp.notes
    failures=[r['batch_index'] for r in rows if r['status']=='FAILED']
    status='FAIL' if violations or failures else 'COMPLETE' if finished else 'PARTIAL'
    summary=dict(version=VERSION,status=status,registered_opportunities=64,registered_training_RAW=512,
        completed_logical_opportunities=completed,committed_optimizer_updates=committed,
        actual_physical_counts=totals,batch_status_counts=dict(Counter(r['status'] for r in rows)),
        partial_or_failed_batches=[r['batch_index'] for r in rows if r['status'] in ('IN_PROGRESS','FAILED')],
        inactive_completed_batches=sum(r['logical_batch_complete'] and r['inactive_kind'] in ('ALL_ZERO','ALL_ONE') for r in rows),
        service_model_load_attempts=sum(r['model_load_attempts'] for r in service_rows),
        service_model_load_returns=sum(r['model_load_returns'] for r in service_rows),
        service_load_seconds_sum=sum(r['load_seconds'] or 0 for r in service_rows),
        per_phase_wall_seconds={phase:range_stats(r[phase+'_wall_seconds'] for r in rows) for phase in ('sample','d0','cpu','update')},
        per_phase_wall_seconds_sum={phase:sum(r[phase+'_wall_seconds'] or 0 for r in rows) for phase in ('sample','d0','cpu','update')},
        overall_wall_seconds=range_stats(r['overall_wall_seconds'] for r in rows),
        overall_wall_seconds_sum=sum(r['overall_wall_seconds'] or 0 for r in rows),
        first_three_batches=[{k:v for k,v in r.items() if k in ('batch_index','status','condition_id','reward_0','reward_1','committed_optimizer_updates','overall_start_utc','overall_end_utc','overall_wall_seconds','phase_wall_seconds_sum','overall_scope') or '_wall_seconds' in k or k.endswith('_range')} for r in rows[:3]],
        training_metric_ranges={key:range_stats(r[key] for r in rows) for key in METRICS},
        resource_ranges={key:range_stats(r.get(key) for r in resources) for key in ('global_GPU_used_MiB','global_GPU_free_MiB','global_GPU_utilization_percent','RAM_available_MiB')},
        resource_observations=len(resources),process_memory_NA_observations=sum(r.get('process_memory_NA_rows',0) for r in resources),
        resource_scope='GPU device values include resident D0 and other processes; CUDA allocated/reserved are per-process phase high-water marks, not total process VRAM. WDDM NA remains unknown. CPU phase has admission snapshots but no invented continuous telemetry.',
        conditions=conditions,violations=violations,input_read_notes=inp.notes,
        interpretation=['Changing C/noise and online-policy reward curves do not establish fixed-test improvement',
            'Conditional KL is the full Gaussian KL summed on sampled old states, averaged across eight trajectories; not terminal mesh KL',
            'Adapter deltas are original reported successive-update L2 values, not a recomputed final cumulative displacement',
            'Physical optimizer returns can precede a failed checkpoint/phase; only completed logical batches count as committed updates',
            'Concurrent/resident wall times are not isolated compute benchmarks; service idle lifetime is not summed as active D0 time'],
        model_calls_performed_by_summary=0,checkpoint_loads=0,geometry_rescores=0,torch_imported=False,
        final_controller_present=final is not None,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload=json.dumps(dict(sources=inp.sources,summary=summary),sort_keys=True,allow_nan=False).encode()
    tag=hashlib.sha256(payload).hexdigest()[:20];destination=DEST/'training_snapshots'/tag
    for name,data in (('training_per_batch.csv',rows),('training_per_condition.csv',conditions),
                      ('training_resources.csv',resources),('training_resident_services.csv',service_rows)):
        write_csv(destination/name,data)
    D.write(destination/'training_summary.json',summary)
    D.write(destination/'training_input_sources.json',dict(files=inp.sources,read_notes=inp.notes))
    pointer=dict(status=status,snapshot_path=str(destination),summary_path=str(destination/'training_summary.json'),
        summary_sha256=D.sha(destination/'training_summary.json'),completed_logical_opportunities=completed,
        committed_optimizer_updates=committed,source_sha256=summary['source_sha256'])
    p=D.checked(DEST/'training_latest.json');temp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf8') as f:json.dump(pointer,f,sort_keys=True,indent=2);f.write('\n')
    os.replace(temp,p)
    return pointer


def self_test():
    checks={}
    sample=[{'event':'ROLLOUT_ATTEMPT'},{'event':'FORWARD_START','kind':'sampling_forward'}]
    partial=event_counts(sample,[],[{'event':'OPTIMIZER_STEP_START'}])
    checks['attempts_not_returns']=partial['rollout_attempts']==1 and partial['rollout_returns']==0 and partial['sampling_F_returns']==0
    checks['optimizer_attempt_not_commit']=partial['optimizer_step_attempts']==1 and partial['optimizer_step_returns']==0
    d0=[{'event':'D0_FORWARD_ATTEMPT'},{'event':'D0_FORWARD_RETURN'}]
    update=[{'event':'FORWARD_START','kind':'training'},{'event':'FORWARD_RETURN','kind':'training'},
            {'event':'FORWARD_START','kind':'postupdate_kl'},{'event':'FORWARD_RETURN','kind':'postupdate_kl'},
            {'event':'BACKWARD_START'},{'event':'BACKWARD_RETURN'},{'event':'OPTIMIZER_STEP_SKIPPED'}]
    values=event_counts([],d0,update)
    checks['sum_separate_kinds_once']=values['all_model_F_attempts']==3 and values['all_model_F_returns']==3
    checks['backward_and_skip_distinct']=values['backward_returns']==1 and values['optimizer_step_returns']==0 and values['optimizer_skip_events']==1
    checks['overall_wall_includes_handoffs']=elapsed('2026-10-15T00:00:00+00:00','2026-10-15T00:02:01+00:00')==121
    checks['incomplete_overall_wall_remains_unknown']=elapsed('2026-10-15T00:00:00+00:00',None) is None
    checks['NA_not_zero']=range_stats([None,'N/A',3])==dict(count=1,minimum=3.,maximum=3.,mean=3.)
    resource={'initial':{'utc':'synthetic','gpu':{'stdout':'GPU, 24576, 3500, 21076, 82'},
        'processes':{'stdout':'1, python, [N/A]'},'meminfo':'MemAvailable: 1048576 kB\n','cpu_count':24}}
    row=resource_rows(resource,0,'test','synthetic')[0]
    checks['global_MiB_and_WDDM_NA']=row['global_GPU_used_MiB']==3500 and row['RAM_available_MiB']==1024 and row['process_memory_NA_rows']==1
    assert all(checks.values()),checks
    result=dict(status='PASS',checks=checks,source_sha256=D.sha(__file__),model_calls=0,checkpoint_loads=0,
        scope='Synthetic parser/accounting cases only; no actual training completeness claim')
    D.write(OUT/'engineering'/('training_summary_cpu_checks_'+D.sha(__file__)[:12]+'.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args();print(json.dumps(self_test() if args.self_test else run(),sort_keys=True))
