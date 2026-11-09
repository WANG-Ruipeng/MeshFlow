"""CPU-only final accounting; no model construction, geometry scoring or blind key read.

This tool consumes already-completed receipts and byte hashes. It does not repair
or infer missing calls. Run only after the root has authorized final accounting.
"""
from __future__ import annotations
import argparse, collections, csv, json, math
from pathlib import Path
from . import common as C

OUT=C.OUT
EV=OUT/'evaluation'
DEST=OUT/'accounting'
NEW=('CREF_C0','CREF_C200')
CACHED=('EXP67_FROZEN','SMALL_C200')
EXPECTED_TRAIN={'train_forward':1600,'train_backward':1600,'optimizer_update':200}


def lines(path):
    return [json.loads(line) for line in C.resolve(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def require(value,message):
    if not value:raise ValueError(message)


def training_calls(path):
    rows=lines(path);pending={};begun={};done={};attempted=collections.Counter();successful=collections.Counter();failed=collections.Counter();seconds=collections.Counter()
    for row in rows:
        ident,kind=row['call_id'],row['kind'];require(kind in EXPECTED_TRAIN,'Unknown training call')
        require(row['arm']=='CREF_C200','Training ownership changed')
        if row['event']=='CALL_BEGIN':
            require(ident not in begun,'Duplicate training reservation');begun[ident]=row;pending[ident]=row;attempted[kind]+=1
        elif row['event'] in ('CALL_DONE','CALL_FAILED'):
            require(ident in pending and pending[ident]['kind']==kind,'Unmatched training completion')
            require(pending[ident]['context']==row['context'],'Training context changed during call')
            pending.pop(ident);(successful if row['event']=='CALL_DONE' else failed)[kind]+=1
            if row['event']=='CALL_DONE':done[ident]=row;seconds[kind]+=float(row['seconds'])
        else:raise ValueError('Unknown training event')
    require(not pending and not failed,'Unresolved or failed training calls')
    require(dict(attempted)==dict(successful)==EXPECTED_TRAIN,'Training call budget not exact')
    updates=[r['context']['global_step'] for r in begun.values() if r['kind']=='optimizer_update']
    require(updates==list(range(6001,6201)),'Training update cursor skipped or replayed')
    for kind in ('train_forward','train_backward'):
        selected=[r for r in begun.values() if r['kind']==kind]
        require([r['context']['sample_index'] for r in selected]==list(range(1600)),'Training sample order changed')
        require(all(r['context']['local_step']==i//8+1 and r['context']['microbatch']==i%8 for i,r in enumerate(selected)),'Microbatch identity changed')
    return dict(attempted=dict(attempted),successful=dict(successful),failed=dict(failed),pending_call_ids=[],
                measured_call_seconds=dict(seconds),first_unix=min(r['time_unix'] for r in rows),last_unix=max(r['time_unix'] for r in rows),
                ledger_sha256=C.sha(path),attempt_ids=sorted({r['attempt_id'] for r in rows}))


def sampling_calls(path,arm,rid):
    rows=lines(path)
    require(len(rows)==103 and rows[0]['event']=='CELL_REGISTERED','Euler ledger length/registration')
    context=rows[0]['context'];require(context['label']==arm and context['request_id']==rid,'Euler owner identity')
    require(rows[1]['event']=='RESERVE' and rows[1]['kind']=='rollout_attempt','Missing rollout reservation')
    seconds=0.
    for step in range(50):
        a,b=rows[2+step*2:4+step*2]
        require((a.get('event'),a.get('kind'),a.get('ordinal'),a.get('context',{}).get('euler_step'))==('RESERVE','sampling_forward',step+1,step),'Euler reservation cursor changed')
        require((b.get('event'),b.get('kind'))==('RETURN','sampling_forward'),'Unresolved Euler call')
        seconds+=float(b['seconds'])
    require(rows[-1]['event']=='ROLLOUT_COMPLETE' and rows[-1]['nfe']==50,'Euler completion missing')
    return dict(reserved=50,returned=50,rollouts=1,measured_forward_seconds=seconds,ledger_sha256=C.sha(path))


def effective_evaluation_hashes(registration):
    """Accept only the explicit, preserved import-order amendment; no broad waiver."""
    path=OUT/'engineering/001_import_path/amendment.json'
    before=dict(registration['code_hashes'])
    if not path.exists():return before,[]
    a=C.read(path)
    require(a.get('status')=='APPLIED_ENGINEERING_ONLY','Engineering amendment not applied')
    require(a['original_registration_sha256']==C.sha(EV/'registration.json'),'Amendment parent registration changed')
    require(a['previous_code_hashes']==before,'Amendment parent code set differs')
    current=a['current_code_hashes'];require(set(current)==set(before),'Amendment changed source set')
    changed=[p for p in before if before[p]!=current[p]]
    require(len(changed)==1 and C.resolve(changed[0]).resolve()==(C.CODE/'evaluate.py').resolve(),'Only registered evaluator import patch is allowed')
    p=changed[0];saved=C.resolve(a['preserved_original_path'])
    require(saved.resolve()==(OUT/'engineering/001_import_path/evaluate_before.py').resolve(),'Original evaluator preservation path differs')
    require(a['previous_evaluate_sha256']==before[p]==C.sha(saved),'Preserved original evaluator hash differs')
    require(a.get('training_or_sampling_definition_changed') is False,'Science change is not this engineering amendment')
    for key in ('affected_completed_outputs','actual_new_D0_before_error','scientific_calls_replayed'):
        require(a.get(key)==0,'Engineering recovery reports scientific replay/change: '+key)
    for p,h in current.items():require(C.sha(p)==h,'Amended evaluation source differs: '+p)
    return current,[dict(path=str(path),sha256=C.sha(path),preserved_original_path=str(saved),preserved_original_sha256=C.sha(saved),
                         parent_registration_sha256=a['original_registration_sha256'],changed_source=changed[0],old_sha256=before[changed[0]],new_sha256=current[changed[0]],
                         diff_path=str(path.parent/'code.diff'),diff_sha256=C.sha(path.parent/'code.diff'),scientific_calls_replayed=0,status='PASS_EXPLICIT_PARENT_CHAIN')]


def check_hashes(groups):
    expected={};owners=collections.defaultdict(set)
    for group,mapping in groups:
        for path,h in mapping.items():
            path=str(C.resolve(path))
            require(path not in expected or expected[path]==h,'Conflicting frozen hashes: '+path)
            expected[path]=h;owners[path].add(group)
    rows=[]
    for path,h in sorted(expected.items()):
        actual=C.sha(path);require(actual==h,'Frozen asset changed: '+path)
        p=C.resolve(path)
        category='weight' if p.suffix.lower() in ('.pt','.pth','.ckpt') else 'source_code' if p.suffix.lower()=='.py' else 'other_frozen_input'
        rows.append(dict(path=path,kind=category,registration_groups=';'.join(sorted(owners[path])),expected_sha256=h,actual_sha256=actual,status='PASS',bytes=p.stat().st_size))
    return rows


def gpu_snapshot_rows(paths):
    result=[]
    for path in sorted(set(paths)):
        if path.suffix=='.jsonl':items=lines(path)
        else:
            obj=C.read(path);items=obj.get('samples',[obj])
        for i,item in enumerate(items):
            gpu=item.get('gpu',{})
            if not isinstance(gpu,dict):continue
            output=gpu.get('stdout','').strip()
            if not output:continue
            parts=[x.strip() for x in output.splitlines()[0].split(',')]
            if len(parts)<5:continue
            def number(x):
                try:return float(x)
                except (TypeError,ValueError):return None
            result.append(dict(path=str(path),sample_index=i,gpu=parts[0],total_MiB=number(parts[1]),device_used_MiB=number(parts[2]),device_free_MiB=number(parts[3]),device_compute_percent=number(parts[4]),
                               timestamp=item.get('utc',item.get('unix',item.get('time_unix'))),cpu_cores=item.get('cpu_logical',item.get('cpu_cores')),ram_available_bytes=item.get('ram_available_bytes'),
                               process_observation=item.get('processes'),load_average=item.get('loadavg',item.get('load_average')),
                               scope='Device totals include unrelated processes; never a worker VRAM measurement'))
    return result


def maximum(rows,key):
    values=[float(x[key]) for x in rows if isinstance(x.get(key),(int,float)) and math.isfinite(float(x[key]))]
    return max(values) if values else None


def build():
    reg=C.read(OUT/'registration.json');ev=C.read(EV/'registration.json');status=C.read(OUT/'training/CREF_C200/status.json')
    require(status['status']=='COMPLETE' and (status['local_step'],status['global_step'])==(200,6200),'Await fixed final training endpoint')
    completion=C.read(EV/'score_completion.json');aggregation=C.read(EV/'aggregation.json')
    require(completion['status']==aggregation['status']=='COMPLETE' and aggregation['outputs']==384,'Await complete registered scoring/aggregation')
    require((completion['new_stages'],completion['reused_stages'])==(192,192),'Quality-call reuse accounting differs')
    effective_codes,amendments=effective_evaluation_hashes(ev)
    source_groups=[('training_registration',reg['source_hashes']),('evaluation_code',effective_codes),('evaluation_protected',ev['protected']),('scoring_definition',completion['definition_hashes'])]
    for amendment in amendments:
        source_groups.append(('preserved_engineering_parent',{amendment['preserved_original_path']:amendment['preserved_original_sha256'],amendment['path']:amendment['sha256'],amendment['diff_path']:amendment['diff_sha256']}))
    inputs=C.read(OUT/'inputs/manifest.json')
    source_groups.append(('inherited_input_sources',inputs.get('source_hashes',{})))
    source_groups.append(('frozen_manifest',{OUT/'inputs/manifest.json':ev['manifest_sha256']}))
    source_groups.append(('evaluation_weight_bindings',{b['path']:b['sha256'] for b in ev['models'].values() if 'sha256' in b}))
    source_groups.append(('official_D0',{ev['d0_checkpoint']['path']:ev['d0_checkpoint']['sha256']}))
    source_groups.append(('new_weight_outputs',{status[key]['path']:status[key]['sha256'] for key in ('checkpoint','inference_export')}))
    for path in (OUT/'checkpoints').glob('*.json'):
        receipt=C.read(path)
        if 'path' in receipt and 'sha256' in receipt:source_groups.append(('checkpoint_receipts',{receipt['path']:receipt['sha256']}))
    protected=check_hashes(source_groups)
    train=training_calls(OUT/'training/CREF_C200/calls.jsonl')
    require(all(train[key]==status['calls'][key] for key in ('attempted','successful','failed','pending_call_ids')),'Status/actual ledger mismatch')
    logs=lines(OUT/'training/CREF_C200/training_log.jsonl')
    require([x['step'] for x in logs]==list(range(1,201)),'Training logs incomplete or replayed')
    init=C.read(OUT/'training/CREF_C200/initialization.json');baseline=C.read(C.QUICK/'training/SMALL_C200/initialization.json')
    # A resume record may name an intermediate state. The immutable baseline
    # identity still proves the registered first-state contract in that case.
    if init['restored_global_step']==6000:
        require(all(init[key]==baseline[key] for key in ('loaded_generator_sha256','loaded_optimizer_sha256','loaded_rng_sha256')),'Initial generator/AdamW/RNG differs from SMALL')
    plan=C.read(OUT/'plans/CREF_C200.json');prior_plan=C.read(C.QUICK/'plans/SMALL_C200.json')
    require(plan['records']==prior_plan['records'] and len(plan['records'])==1600,'Baseline training records changed')
    expected={(label,q['request_id']) for label in CACHED+NEW for q in ev['requests']};indices={}
    for stage in ('raw','post'):
        rows=C.read(EV/(stage+'_index.json'))['rows'];mapping={(r['label'],r['request_id']):r for r in rows}
        require(len(rows)==192 and set(mapping)==expected,'Incomplete '+stage+' index');indices[stage]=mapping
    raw_audit=[];post_audit=[];execution=[]
    for arm in NEW:
        for q in ev['requests']:
            rid=q['request_id'];raw=indices['raw'][arm,rid];post=indices['post'][arm,rid]
            folder=EV/'raw'/arm/rid;calls=sampling_calls(folder/'calls.jsonl',arm,rid)
            require(raw['status']=='COMPLETE' and raw['new_rollouts']==1 and raw['new_sampling_F']==50,'Incorrect new RAW counters')
            run=C.read(raw['result_path']);timing=run.get('sampling',run)
            raw_audit.append(dict(label=arm,request_id=rid,**calls,sampling_seconds=timing.get('seconds'),raw_path=raw['raw_path'],raw_sha256=raw['raw_sha256'],receipt_sha256=C.sha(folder/'receipt.json')))
            d0path=C.resolve(post['proposal_receipt']);d0=C.read(d0path);reserved=C.read(d0path.parent/'D0_started.json')
            require(d0['status']=='PASS' and d0['model_forwards']==1,'Not exactly one actual D0 return')
            require(reserved['status']=='RESERVED' and reserved['raw_sha256']==d0['raw_sha256']==raw['raw_sha256'],'D0 reserve/input differs')
            require(d0['checkpoint_sha256']==reserved['checkpoint_sha256']==ev['d0_checkpoint']['sha256'],'D0 checkpoint differs')
            require(post['status']=='COMPLETE' and post['GEO_calls']==1,'Missing geometry completion')
            # Count this experiment's actual proposal receipts, not new_D0_F
            # in a later engineering continuation that may correctly say zero.
            post_audit.append(dict(label=arm,request_id=rid,D0_reserved=1,D0_returned=1,GEO_completed=1,seconds=d0['seconds'],peak_allocated_bytes=d0.get('peak_allocated_bytes'),peak_reserved_bytes=d0.get('peak_reserved_bytes'),proposal_receipt=str(d0path),proposal_receipt_sha256=C.sha(d0path),post_sha256=post['post_sha256']))
        for stage in ('generator','D0'):
            p=EV/'execution'/arm/(stage+'_0_1_complete.json');x=C.read(p)
            require(x['status']=='COMPLETE','Worker missing completion')
            execution.append(dict(label=arm,stage=stage,path=str(p),**x))
    require(len(raw_audit)==len(post_audit)==96,'New evaluation budget differs')
    # No extra cell directories or reservations are hidden by the final index.
    actual_raw=set((p.parent.parent.name,p.parent.name) for p in (EV/'raw').glob('*/*/calls.jsonl'))
    actual_post=set((p.parent.parent.name,p.parent.name) for p in (EV/'post').glob('*/*/D0_started.json'))
    new_expected={(label,q['request_id']) for label in NEW for q in ev['requests']}
    require(actual_raw==actual_post==new_expected,'Unindexed/extra sampling or D0 attempt')
    raw_events=[]
    for path in (OUT/'execution').glob('*events.jsonl'):raw_events.extend(lines(path))
    for event in ('D0_FORWARD_RESERVED','D0_FORWARD_RETURNED'):
        ids=collections.Counter(x['request_id'] for x in raw_events if x['event']==event)
        require(ids==collections.Counter({q['request_id']:2 for q in ev['requests']}),'D0 event ledger differs from 96 proposals: '+event)
    cached_rows=[]
    previous={stage:{(x['label'],x['request_id']):x for x in C.read(C.NOISE/f'evaluation/{stage}_index.json')['rows']} for stage in ('raw','post')}
    for arm in CACHED:
        for q in ev['requests']:
            for stage in ('raw','post'):
                row=indices[stage][arm,q['request_id']]
                require(row==previous[stage][arm,q['request_id']],'Cached index row silently rewritten')
                cached_rows.append(dict(label=arm,request_id=q['request_id'],stage=stage.upper(),actual_new_model_calls=0,status='VERIFIED_HISTORICAL_REUSE',artifact_sha256=row[stage+'_sha256']))
    resources=list((OUT/'resources').glob('*.json'))+list(OUT.glob('resources*.json'))+list((OUT/'training/CREF_C200').glob('resources*.json'))+[OUT/'training/CREF_C200/resource_checks.jsonl']
    resources=[p for p in resources if p.is_file()];resource_rows=gpu_snapshot_rows(resources)
    observer=C.read(OUT/'training/CREF_C200/resources.json')['summary']
    phase_ends=[dict(x) for x in lines(OUT/'ledger.jsonl') if x['event']=='PHASE_END']
    failed_phases=[x for x in phase_ends if x['returncode']!=0]  # Retained for the final independent audit, never erased by a later success.
    per_step=dict(updates=200,sum_update_seconds=sum(x['update_seconds'] for x in logs),sum_input_wait_seconds=sum(x['input_seconds'] for x in logs),
                  peak_allocated_bytes=maximum(logs,'cuda_peak_allocated_bytes'),peak_reserved_bytes=maximum(logs,'cuda_peak_reserved_bytes'),
                  counters_scope='Reset at each actual update; includes forward/backward/optimizer allocator, excludes CUDA context and other processes')
    budget=dict(training_forwards=1600,training_backwards=1600,optimizer_updates=200,new_rollouts=96,new_generator_forwards=sum(x['returned'] for x in raw_audit),new_D0_forwards=sum(x['D0_returned'] for x in post_audit),
                reused_RAW=96,reused_POST=96,reused_generator_or_D0_calls_this_experiment=0,new_scored_stages=192,reused_scored_stages=192)
    require(budget['new_generator_forwards']==4800 and budget['new_D0_forwards']==96,'New model-call budget mismatch')
    result=dict(status='PASS',experiment='R2_C_REFERENCE_FRAME_V1',budget=budget,training=train,training_update_resources=per_step,
                training_status_wall_seconds=status['wall_seconds'],training_observer=observer,execution_completions=execution,phase_ends=phase_ends,failed_phase_records=failed_phases,
                D0_peak_allocated_bytes=maximum(post_audit,'peak_allocated_bytes'),D0_peak_reserved_bytes=maximum(post_audit,'peak_reserved_bytes'),
                generator_peak_allocated_bytes=maximum([r for r in execution if r['stage']=='generator'],'peak_allocated_bytes'),generator_peak_reserved_bytes=maximum([r for r in execution if r['stage']=='generator'],'peak_reserved_bytes'),
                source_hashes_verified=len(protected),engineering_amendments=amendments,source_weight_files_verified=sum(x['kind']=='weight' for x in protected),
                actual_call_timing=dict(training=train['measured_call_seconds'],generator_sum_seconds=sum(x['measured_forward_seconds'] for x in raw_audit),D0_sum_seconds=sum(x['seconds'] for x in post_audit)),
                caveats=['PyTorch allocated/reserved counters exclude CUDA context; WDDM may not expose per-process memory. A 5GiB allocator cap is not a measured 6GiB process guarantee.',
                         'Device VRAM/compute include unrelated processes. Concurrent wall time must not be compared to earlier isolated wall time as a compute benchmark.',
                         'PHASE_END durations include subprocess startup and phase overhead. Training status wall_seconds covers the most recent trainer invocation if resumed; phase ledger and calls retain all actual invocations.',
                         'Cached RAW/POST receipts may contain historical new_* counters; none are added to this experiment model-call budget.'],
                finalization_model_calls=0,finalization_quality_calls=0)
    C.csvwrite(DEST/'source_protection.csv',protected);C.csvwrite(DEST/'new_raw_calls.csv',raw_audit);C.csvwrite(DEST/'new_D0_calls.csv',post_audit);C.csvwrite(DEST/'reused_outputs.csv',cached_rows);C.csvwrite(DEST/'resource_snapshots.csv',resource_rows);C.write(DEST/'summary.json',result);C.write(OUT/'call_accounting.json',result)
    table='\n'.join('| '+k+' | '+str(v)+' |' for k,v in budget.items())
    note='# Actual execution accounting\n\nAll registered call counts and protected file hashes pass. No model or geometry scorer was run by this tool.\n\n| Item | Actual |\n|---|---:|\n'+table+'\n\n'
    note+='Training update sum: %.3f s; input wait sum: %.3f s. Last trainer invocation wall time: %.3f s.\n\n'%(per_step['sum_update_seconds'],per_step['sum_input_wait_seconds'],status['wall_seconds'])
    note+='Full actual phase durations, worker timings and memory scopes are in summary.json. '+str(len(protected))+' frozen files were independently rehashed.\n\n'+'\n\n'.join(result['caveats'])+'\n'
    C._write_bytes(DEST/'ACCOUNTING.md',note.encode('utf8'))
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    result=build();print(json.dumps(dict(status=result['status'],budget=result['budget'],source_hashes_verified=result['source_hashes_verified']),sort_keys=True),flush=True)


if __name__=='__main__':main()
