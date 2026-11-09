"""Inventory actual checkpoints and calls; never infer completion from filenames."""
from pathlib import Path
import csv
from .common import read_json, atomic_json, digest, LIMITS, REPO

def finalize(source,out):
    source,out=Path(source),Path(out)
    state=read_json(out/'run_state.json') if (out/'run_state.json').exists() else {}
    checkpoints=[]
    index=read_json(source/'checkpoints/checkpoints.json')
    for row in index['checkpoints']:
        path=Path(row['path'])
        checkpoints.append(dict(origin='protected_source',path=str(path),exists=path.is_file(),
            checkpoint_sha256=row['checkpoint_sha256'],metadata=row['metadata'],
            hash_verified=path.is_file() and digest(path)==row['checkpoint_sha256']))
    for sidecar in sorted((out/'checkpoints').glob('*.json')):
        row=read_json(sidecar)
        if 'checkpoint_sha256' not in row:continue
        path=Path(row['path'])
        checkpoints.append(dict(origin='current_run',path=str(path),exists=path.is_file(),
            checkpoint_sha256=row['checkpoint_sha256'],metadata=row['metadata'],
            hash_verified=path.is_file() and digest(path)==row['checkpoint_sha256'],
            purpose='recoverable_latest' if path.stem.endswith('_latest') else 'fixed_registered_state'))
    manifest=dict(schema='control_mvp48_budget_inventory_v1',restore_mode='RESTORE_ALL',
        probe_restore_mode='FRESH_PROBE',checkpoints=checkpoints,
        note='MVP48 conditional fine-tuning steps only; official pretraining excluded; weights are not repackaged')
    atomic_json(out/'checkpoint_manifest.json',manifest)
    training=[];micro_attempt=micro_return=back_attempt=back_return=step_attempt=steps=0
    for arm in ('T1','Geo','Full','T1_fit2_probe'):
        path=out/'training'/f'{arm}_status.json'
        row=read_json(path) if path.exists() else dict(arm=arm,status='NOT_RUN',counts={})
        training.append(row);c=row.get('counts',{})
        micro_attempt+=c.get('forward_attempts',0);micro_return+=c.get('forward_returns',0)
        back_attempt+=c.get('backward_attempts',0);back_return+=c.get('backward_returns',0)
        step_attempt+=c.get('optimizer_step_attempts',0);steps+=c.get('optimizer_updates',0)
    evaluation=[];gen_attempt=gen_return=gen_fwd_attempt=gen_fwd_return=teach_attempt=teach_return=0
    resources=[]
    for path in sorted((out/'evaluation').glob('*/*/ledger.json')):
        row=read_json(path);identity=row.get('identity',{});a=row.get('sampling_counts',{});b=row.get('teacher_counts',{})
        evaluation.append(dict(identity=identity,status=row['status'],sampling_counts=a,teacher_counts=b))
        gen_attempt+=a.get('rollout_attempts',0);gen_return+=a.get('complete_rollouts',0)
        gen_fwd_attempt+=a.get('model_forward_attempts',0);gen_fwd_return+=a.get('model_forward_returns',0)
        teach_attempt+=b.get('model_forward_attempts',0);teach_return+=b.get('model_forward_returns',0)
        cases=row.get('rows',[]);teachers=row.get('teacher_rows',[])
        resources.append(dict(phase=identity.get('phase'),worker=identity.get('model_id'),status=row['status'],
            gpu_workers=row.get('actual_gpu_workers'),worker_seconds=row.get('worker_seconds'),lock_wait_seconds=row.get('lock_wait_seconds'),
            forward_seconds=sum(r.get('audit',{}).get('seconds',0) for r in cases)+sum(r.get('seconds',0) for r in teachers),
            cuda_peak_allocated_bytes=max([r.get('audit',{}).get('peak_cuda_allocated_bytes',0) for r in cases]+[r.get('peak_cuda_allocated_bytes',0) for r in teachers]+[0]),
            cuda_peak_reserved_bytes=max([r.get('audit',{}).get('peak_cuda_reserved_bytes',0) for r in cases]+[r.get('peak_cuda_reserved_bytes',0) for r in teachers]+[0]),
            memory_scope='max of per-generation/teacher scopes; includes resident model; not summed',
            process_gpu_used_peak_mib=row.get('process_gpu_used_peak_mib'),global_gpu_used_peak_mib=row.get('global_gpu_used_peak_mib')))
    ot_attempt=ot_return=ot_hits=0
    for path in sorted((out/'training').glob('*_to*_summary.json')):
        r=read_json(path);c=r.get('OT_counts',{})
        ot_attempt+=c.get('actual_OT_attempts',0);ot_return+=c.get('actual_OT_returns',0);ot_hits+=c.get('OT_cache_hits',0)
        resources.append(dict(phase='training',worker=f"{r['arm']}_to{r['target']}",status=r['status'],gpu_workers=1,
            worker_seconds=r.get('worker_wall_seconds'),lock_wait_seconds=r.get('lock_wait_seconds'),
            forward_seconds=r.get('worker_train_seconds'),OT_seconds=r.get('worker_OT_seconds'),save_seconds=r.get('worker_save_seconds'),
            load_seconds=r.get('load_seconds'),cuda_peak_allocated_bytes=r.get('cuda_peak_allocated_bytes'),
            cuda_peak_reserved_bytes=r.get('cuda_peak_reserved_bytes'),memory_scope=r.get('memory_scope'),
            process_gpu_used_peak_mib=r.get('process_gpu_used_peak_mib'),global_gpu_used_peak_mib=r.get('global_gpu_used_peak_mib'),
            updates_per_train_second=r.get('throughput_updates_per_train_second')))
    wiring=read_json(out/'registration/wiring_precheck.json') if (out/'registration/wiring_precheck.json').exists() else {}
    prepare_path=out/'registration/prepare_events.jsonl'
    import json
    prep=[json.loads(line) for line in prepare_path.read_text().splitlines()] if prepare_path.exists() else []
    wa=wiring.get('forwards_attempted',0);wr=wiring.get('forwards_returned',0)
    total=micro_attempt+gen_fwd_attempt+teach_attempt+wa
    record=dict(status=state.get('status','NOT_RUN'),limits=LIMITS,training=training,evaluation=evaluation,
        optimizer_step_attempts=step_attempt,optimizer_updates_returned=steps,
        training_microbatch_forward_attempts=micro_attempt,training_microbatch_forward_returns=micro_return,
        training_backward_attempts=back_attempt,training_backward_returns=back_return,
        generation_attempts=gen_attempt,complete_generations=gen_return,
        generation_forward_attempts=gen_fwd_attempt,generation_forward_returns=gen_fwd_return,
        teacher_forward_attempts=teach_attempt,teacher_forward_returns=teach_return,
        wiring_forward_attempts=wa,wiring_forward_returns=wr,zero_update_backward_returns=wiring.get('zero_update_backwards',0),
        zero_update_output_gradient_calls=wiring.get('zero_update_output_gradient_calls',0),
        all_model_forward_attempts=total,
        CPU_OT=dict(training_attempts=ot_attempt,training_returns=ot_return,cache_hits=ot_hits,
            teacher_prepare_attempts=sum(r.get('action')=='teacher_OT_attempt' for r in prep),
            teacher_prepare_returns=sum(r.get('action')=='teacher_OT_return' for r in prep),
            wiring=wiring.get('cpu_OT',{})),
        within_hard_limits=steps<=13000 and gen_return<=1196 and teach_attempt<=738 and wa<=80 and total<=164618,
        timing_note='Worker wall times include their own waiting/loading/IO; overlapping observations and resource peaks are not additive. No isolated throughput equivalence claimed.',
        process_VRAM_unavailable='null/NA denotes WSL unavailable; allocated and reserved are not added',
        restart_command=state.get('restart_command'))
    atomic_json(out/'execution_accounting.json',record)
    fields=list(dict.fromkeys(k for r in resources for k in r))
    with (out/'resource_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in resources:w.writerow({k:'NA' if v is None else v for k,v in r.items()})
    protected=read_json(source/'TRAIN_READY.json')['source_hashes']
    checks={name:digest(REPO/name)==sha for name,sha in protected.items()}
    atomic_json(out/'protected_assets_after.json',dict(public_code_hashes=checks,
        source_checkpoints_unchanged=all(r['hash_verified'] for r in checkpoints if r['origin']=='protected_source'),
        no_default_model_change=all(checks.values()),commit_or_push=False))
    (out/'restart.txt').write_text((state.get('restart_command') or 'Run the same run_all --source --out command')+'\n',encoding='utf-8')
    return record
