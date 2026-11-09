"""Read-only endpoint, call budget and resource audit. No tensor/model loads.

The separate training_closeout.py owns exposure/loss/curve accounting. This
script verifies serialized file bytes and recorded state hashes, rather than
loading the large live generator or optimizer a second time.
"""
from __future__ import annotations
import argparse,collections,json,math,statistics,time
from pathlib import Path
from .common import OUT,SOURCE,read,sha,resolve,write_new

DEST=OUT/'execution_closeout'
ARM='W_MATCHED'
EXPECTED=dict(train_forward=12000,train_backward=12000,optimizer_update=1500)


def require(value,message):
    if not value:raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def audit_calls(rows,updates=1500):
    attempted=collections.Counter();successful=collections.Counter();pending={};begins=[];seconds=collections.defaultdict(float)
    wanted=dict(train_forward=8*updates,train_backward=8*updates,optimizer_update=updates)
    for row in rows:
        require(row['arm']==ARM and row['kind'] in wanted,'Unexpected arm/kind in scientific ledger')
        call_id=row['call_id'];kind=row['kind']
        if row['event']=='CALL_BEGIN':
            require(not pending and call_id==len(begins)+1,'Overlapping, duplicated or skipped scientific call')
            pending[call_id]=row;begins.append(row);attempted[kind]+=1
        elif row['event']=='CALL_DONE':
            require(call_id in pending,'Call completed without matching begin')
            first=pending.pop(call_id)
            require(all(row[k]==first[k] for k in ('kind','context','attempt_id')),'Call begin/end identity differs')
            require(math.isfinite(row['seconds']) and row['seconds']>=0,'Invalid call timing')
            successful[kind]+=1;seconds[kind]+=row['seconds']
        else:raise ValueError('Failure/pending event present; do not call complete')
    require(not pending and dict(attempted)==wanted and dict(successful)==wanted,'Scientific budget differs')
    order=[]
    for step in range(1,updates+1):
        order.extend((step,mi,kind) for mi in range(8) for kind in ('train_forward','train_backward'))
        order.append((step,None,'optimizer_update'))
    for row,(step,mi,kind) in zip(begins,order):
        c=row['context']
        require((c['stage_step'],c.get('microbatch'),row['kind'])==(step,mi,kind),'Exact scientific call order differs')
        require(c['global_step']==21000+step and c['local_step']==step and c['native_total_step']==9000+step
            and c['curriculum_step']==3000+step and c['seam_total_step']==0,'Call cumulative cursor differs')
    return dict(attempted=dict(attempted),successful=dict(successful),failed=0,pending=0,
        exact_step_microbatch_order=True,head_calls=0,probe_calls=0,scientific_call_seconds=dict(seconds),events=len(rows))


def parse_samples(samples):
    result=[]
    for row in samples:
        gpu=row.get('gpu',{})
        if gpu.get('returncode')!=0:continue
        parts=[v.strip() for v in gpu.get('stdout','').strip().split(',')]
        if len(parts)!=5:continue
        try:values=[float(v) for v in parts[1:]]
        except ValueError:continue
        result.append(dict(unix=row['time_unix'],name=parts[0],total_mib=values[0],used_mib=values[1],
            free_mib=values[2],util_percent=values[3]))
    return result


def run():
    folder=OUT/'training'/ARM;state=read(folder/'status.json')
    require(state.get('status')=='COMPLETE','Only run after the fixed endpoint is COMPLETE')
    if (DEST/'complete.json').exists():
        result=read(DEST/'complete.json')
        for path,h in result['outputs'].items():require(sha(path)==h,'Previous closeout output changed')
        return result
    require(not DEST.exists(),'Partial closeout exists; inspect before explicit engineering recovery')
    started=time.perf_counter();reg=read(OUT/'training_registration.json');regsha=sha(OUT/'training_registration.json')
    hashes={}
    def authenticate(path,expected):
        p=resolve(path);key=str(p)
        if key not in hashes:hashes[key]=sha(p)
        require(hashes[key]==expected,'Frozen file changed: '+key)
    for mapping in ('training_code_hashes','gate_hashes','protected'):
        for path,h in reg[mapping].items():authenticate(path,h)
    for field in ('manifest','eval_manifest','source_checkpoint','coefficient_schedule'):
        authenticate(reg[field],reg[field+'_sha256'])
    for spec in reg['schedules'].values():authenticate(spec['path'],spec['sha256'])
    require(state['registration_sha256']==regsha and state['arm']==ARM,'Endpoint registration differs')
    require((state['local_step'],state['stage_step'],state['global_step'],state['native_total_step'],state['curriculum_step'])
        ==(1500,1500,22500,10500,4500),'Endpoint cumulative steps differ')
    require(state['head_calls']==state['head_step']==state['probe_calls']==state['seam_total_step']==0,'Unexpected head/probe/SEAM activity')
    logs=jsonl(folder/'training_log.jsonl');require(len(logs)==1500,'1500 actual update records required')
    ledger=audit_calls(jsonl(folder/'calls.jsonl'))
    require(not state['calls']['failed'] and not state['calls']['pending_call_ids'],'Status contains failed/pending calls')
    for key in ('attempted','successful'):require(state['calls'][key]==EXPECTED,'Status budget differs from ledger')
    init=read(folder/'branch_initialization.json');source_side=read(SOURCE.with_suffix('.json'))
    for key in ('source_checkpoint_sha256','source_generator_sha256','source_optimizer_sha256','source_rng_sha256'):
        require(init[key]==reg[key],'Source initialization hash differs')
    for key,sourcekey in (('sha256','source_checkpoint_sha256'),('generator_state_sha256','source_generator_sha256'),
        ('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')):
        require(source_side[key]==reg[sourcekey],'Retained source recovery sidecar differs')
    require(init['inherited_optimizer'] and init['inherited_rng'] and not init['fresh_AdamW'] and not init['head_used'],
        'Matching AdamW/RNG continuation was not recorded')
    require(init['coefficient_schedule_sha256']==reg['coefficient_schedule_sha256'],'Initialization coefficient identity differs')
    artifacts={}
    for key,filename,schema in (('checkpoint','checkpoint_step1500.pt','bench_group_weight_match_training_v1'),
        ('inference_export','generator_step1500.pt','bench_group_weight_match_generator_v1')):
        path=folder/filename;side=read(path.with_suffix('.json'))
        require(side==state[key],'Final status and artifact sidecar differ')
        require((side['schema'],side['arm'],side['stage'],side['stage_step'],side['local_step'],side['global_step'],side['curriculum_step'])
            ==(schema,ARM,'NATIVE',1500,1500,22500,4500),'Final artifact identity differs')
        require(path.stat().st_size==side['bytes'],'Final artifact byte size differs');authenticate(path,side['sha256'])
        artifacts[key]=side
    require(artifacts['checkpoint']['generator_state_sha256']==artifacts['inference_export']['generator_state_sha256'],
        'Recovery/inference generator state identities differ')
    require(bool(artifacts['checkpoint']['optimizer_state_sha256']) and bool(artifacts['checkpoint']['rng_sha256']),
        'Final recovery optimizer/RNG hash missing')
    attempts=[];gpu=[]
    require(not list((folder/'attempts').glob('*/failure.json')),'Failed training attempt exists')
    for path in sorted((folder/'attempts').iterdir()):
        final=read(path/'final_status.json');resource=read(path/'resources.json')
        require(final['status'] in ('COMPLETE','INTERRUPTED_SAFE'),'Unresolved attempt exists')
        gpu.extend(parse_samples(resource['samples']))
        attempts.append(dict(attempt=path.name,status=final['status'],local_step=final['local_step'],
            active_wall_seconds=final['elapsed_seconds'],resource_summary=resource['summary'],
            resources_sha256=sha(path/'resources.json'),resources_path=str(path/'resources.json')))
    require(bool(gpu),'No valid full-training GPU observer samples')
    timing=dict(attempts=attempts,attempt_count=len(attempts),active_attempt_wall_seconds=sum(x['active_wall_seconds'] for x in attempts),
        optimizer_updates=1500,sum_update_seconds=sum(x['update_seconds'] for x in logs),
        sum_input_seconds=sum(x['input_seconds'] for x in logs),
        first_update_time_unix=logs[0]['time_unix'],last_update_time_unix=logs[-1]['time_unix'],
        effective_updates_per_active_second=1500/sum(x['active_wall_seconds'] for x in attempts),
        allocator_peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in logs),
        allocator_peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in logs),
        aggregate_device_samples=len(gpu),aggregate_device_peak_used_mib=max(x['used_mib'] for x in gpu),
        aggregate_device_min_free_mib=min(x['free_mib'] for x in gpu),
        aggregate_device_mean_util_percent=statistics.mean(x['util_percent'] for x in gpu),
        process_VRAM='Unavailable under WDDM; aggregate values include other user processes',
        timing_scope='Active attempt wall includes model load and checkpoint IO; update durations exclude input and most checkpoint IO. One training worker; no claim of isolated GPU benchmark.',
        scientific_call_seconds=ledger['scientific_call_seconds'])
    # No copied tensors or raw logs: summaries link the immutable original evidence.
    DEST.mkdir()
    write_new(DEST/'resource_and_attempt_wallclock.json',timing)
    record=dict(status='PASS',experiment=reg['experiment'],arm=ARM,local_step=1500,global_step=22500,
        native_total_step=10500,curriculum_step=4500,ledger=ledger,artifacts=artifacts,
        source_initialization=init,source_checkpoint_unchanged=True,
        recovery_evidence_scope='Model/AdamW/RNG state hashes from runtime-validated serialization and authenticated final bytes; no repeat tensor load/state rehash.',
        exact_scientific_replay_updates=0,source_stream_and_loss_audit='../training_closeout/complete.json (separate independently owned audit)',
        protected_and_frozen_sha256=hashes,
        evidence_hashes={str(folder/name):sha(folder/name) for name in ('status.json','calls.jsonl','training_log.jsonl','branch_initialization.json')},
        training_registration_sha256=regsha,model_calls=0,tensor_loads=0,OT_calls=0)
    write_new(DEST/'execution_audit.json',record)
    result=dict(status='PASS',schema='bench_group_weight_match_execution_closeout_v1',arm=ARM,
        total_updates=1500,total_forwards=12000,total_backwards=12000,failed=0,pending=0,
        head_calls=0,probe_calls=0,replayed_updates=0,attempts=len(attempts),
        active_wall_seconds=timing['active_attempt_wall_seconds'],device_peak_used_mib=timing['aggregate_device_peak_used_mib'],
        device_min_free_mib=timing['aggregate_device_min_free_mib'],final_checkpoint=artifacts['checkpoint'],
        inference_export=artifacts['inference_export'],training_registration_sha256=regsha,
        audit_code_sha256=sha(__file__),outputs={str(p):sha(p) for p in sorted(DEST.iterdir()) if p.is_file()},
        audit_seconds=time.perf_counter()-started,model_calls=0,tensor_loads=0,OT_calls=0)
    write_new(DEST/'complete.json',result);return result


def cpu_check():
    rows=[];call_id=0
    for step in range(1,3):
        for mi,kind in [(m,k) for m in range(8) for k in ('train_forward','train_backward')]+[(None,'optimizer_update')]:
            call_id+=1;c=dict(stage_step=step,local_step=step,global_step=21000+step,native_total_step=9000+step,
                curriculum_step=3000+step,seam_total_step=0)
            if mi is not None:c['microbatch']=mi
            row=dict(arm=ARM,kind=kind,call_id=call_id,attempt_id='fixture',context=c)
            rows.extend((dict(row,event='CALL_BEGIN'),dict(row,event='CALL_DONE',seconds=.01)))
    result=audit_calls(rows,2);require(result['attempted']['train_forward']==16,'Fixture call count differs')
    for invalid in (rows[:-1],rows+[dict(rows[-1],event='CALL_FAILED')]):
        try:audit_calls(invalid,2)
        except ValueError:pass
        else:raise ValueError('Accepted pending/failed call fixture')
    parsed=parse_samples([dict(time_unix=1,gpu=dict(returncode=0,stdout='RTX, 16000, 7000, 9000, 20'))])
    require(parsed[0]['free_mib']==9000,'Resource parse differs')
    return dict(status='PASS',checks=4,model_calls=0,tensor_loads=0,production_OT_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args();result=cpu_check() if args.cpu_check else run()
    print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','final_checkpoint','inference_export')}))
