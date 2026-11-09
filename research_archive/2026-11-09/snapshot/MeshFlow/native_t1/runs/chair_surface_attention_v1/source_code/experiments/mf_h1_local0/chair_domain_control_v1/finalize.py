"""Actual execution accounting and protected checkpoint inventory; no model calls."""
from pathlib import Path
import csv
import json
from .common import REPO,LIMITS,atomic_json,read_json,digest
from .event_audit import audit_training_events


def finalize(source,out):
    source,out=Path(source),Path(out)
    state=read_json(out/'run_state.json');checkpoints=[]
    for row in read_json(source/'checkpoints/checkpoints.json')['checkpoints']:
        path=Path(row['path']);meta=row['metadata']
        checkpoints.append(dict(origin='protected_source',path=str(path),checkpoint_sha256=row['checkpoint_sha256'],metadata=meta,
            phase_step=0 if meta['logical_step']==1000 else None,
            purpose='own_arm_phase0_source' if meta['logical_step']==1000 else 'protected_lineage_only',
            hash_verified=path.is_file() and digest(path)==row['checkpoint_sha256']))
    for sidecar in sorted((out/'checkpoints').glob('*.json')):
        row=read_json(sidecar)
        if 'checkpoint_sha256' not in row: continue
        path=Path(row['path'])
        checkpoints.append(dict(origin='current_run',path=str(path),checkpoint_sha256=row['checkpoint_sha256'],metadata=row['metadata'],
            phase_step=row['metadata']['phase_step'],purpose='recoverable_latest' if path.stem.endswith('_latest') else 'fixed_registered_state',
            hash_verified=path.is_file() and digest(path)==row['checkpoint_sha256']))
    if any(not r['hash_verified'] for r in checkpoints): raise ValueError('Checkpoint inventory hash failure')
    atomic_json(out/'checkpoint_manifest.json',dict(schema='chair_domain_checkpoint_manifest_v1',restore_mode='RESTORE_ALL',
        checkpoints=checkpoints,note='Phase0 references original own-arm step1000 bytes. Conditional cumulative steps exclude official pretraining; weights are not repackaged.'))
    audit=read_json(out/'registration/execution_budget_audit.json')
    events=audit_training_events(out)
    if audit['status']!='PASS' or events['status']!='PASS': raise ValueError('Actual execution budget/event audit not PASS')
    resources=[];evaluation=[];scoring={}
    for path in sorted((out/'evaluation').glob('phase*/*/ledger.json')):
        row=read_json(path);ident=row['identity']
        evaluation.append(dict(identity=ident,status=row['status'],sampling_counts=row['sampling_counts'],teacher_counts=row['teacher_counts'],scoring_counts=row['scoring_counts']))
        for key,value in row['scoring_counts'].items():
            if isinstance(value,(int,float)):scoring[key]=scoring.get(key,0)+value
        resources.append(dict(phase='evaluation_worker',arm=ident['arm'],phase_step=ident['phase_step'],gpu_workers=1,
            worker_seconds=row.get('worker_seconds'),lock_wait_seconds=row.get('lock_wait_seconds'),load_seconds=row.get('model_load_seconds'),
            scope='Whole evaluation worker including lock wait, loading, sampling, teacher and CPU scoring; nested stage times not additive',
            process_gpu_used_peak_mib=row.get('process_gpu_used_peak_mib'),global_gpu_used_peak_mib=row.get('global_gpu_used_peak_mib')))
        stages=row.get('stage_resource_attempts',row.get('stage_resources',{}))
        stage_items=stages.items() if isinstance(stages,dict) else ((v.get('stage','UNKNOWN'),v) for v in stages)
        for name,value in stage_items:
            resources.append(dict(phase=name,arm=ident['arm'],phase_step=ident['phase_step'],gpu_workers=1,**value))
    learning=[]
    for arm in ('T1','Geo','Full'):
        path=out/'training'/f'{arm}_events.jsonl'
        for r in (json.loads(line) for line in path.read_text().splitlines()):
            if r['event']=='UPDATE_VALIDATED':
                learning.append({k:r.get(k) for k in ('arm','phase_step','conditional_total_step','FM','geometry','loss','free_coordinate_denominator','active_samples','grad_norm_before_clip','mean_active_D_edge','mean_active_L5','OT_seconds','train_seconds','batch_sha256')})
            if r['event']!='SEGMENT_END': continue
            resources.append(dict(phase='training',arm=arm,phase_step=r['phase_step'],target_phase=r['target'],gpu_workers=1,
                worker_seconds=r.get('worker_wall_seconds'),lock_wait_seconds=r.get('lock_wait_seconds'),load_seconds=r.get('load_seconds'),
                training_update_seconds=r.get('worker_training_update_seconds'),OT_seconds=r.get('worker_OT_seconds'),save_seconds=r.get('worker_save_seconds'),
                actual_updates=r.get('actual_updates_this_worker'),cuda_peak_allocated_bytes=r.get('cuda_peak_allocated_bytes'),cuda_peak_reserved_bytes=r.get('cuda_peak_reserved_bytes'),
                global_gpu_used_peak_mib=r.get('global_gpu_used_peak_mib'),process_gpu_used_peak_mib=r.get('process_gpu_used_peak_mib'),
                updates_per_training_update_second=r.get('throughput_updates_per_train_second'),scope=r.get('timing_scope'),memory_scope=r.get('memory_scope')))
    with (out/'training_learning_curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(learning[0]));writer.writeheader();writer.writerows(learning)
    wiring=read_json(out/'registration/wiring_precheck.json');prep=read_json(out/'registration/data_audit.json')
    analysis=read_json(out/'analysis_ledger.json')
    resources.append(dict(phase='wiring',gpu_workers=1,worker_seconds=wiring.get('elapsed_seconds'),lock_wait_seconds=wiring.get('lock_wait_seconds'),
        cuda_peak_allocated_bytes=wiring.get('peak_allocated_bytes'),cuda_peak_reserved_bytes=wiring.get('peak_reserved_bytes'),memory_scope=wiring.get('peak_scope')))
    record=dict(schema='chair_domain_actual_accounting_v1',status='PASS',limits=LIMITS,
        optimizer_step_attempts=audit['optimizer_updates'],optimizer_updates_returned=audit['optimizer_updates'],
        training_microbatch_forward_attempts=audit['training_forward_attempts'],training_microbatch_forward_returns=audit['training_forward_attempts'],
        training_backward_attempts=events['counts']['training_backward_attempts'],training_backward_returns=events['counts']['training_backward_attempts'],
        generation_attempts=audit['complete_rollouts'],complete_generations=audit['complete_rollouts'],
        generation_forward_attempts=audit['sampling_forward_attempts'],generation_forward_returns=audit['sampling_forward_attempts'],
        teacher_forward_attempts=audit['teacher_forward_attempts'],teacher_forward_returns=audit['teacher_returns'],
        wiring_forward_attempts=wiring['forwards_attempted'],wiring_forward_returns=wiring['forwards_returned'],
        zero_update_backward_returns=wiring['zero_update_backwards'],zero_update_output_gradient_calls=wiring.get('zero_update_output_gradient_calls',0),
        all_model_forward_attempts=audit['model_forward_attempts'],planned_repetitions=0,actual_repetitions=0,
        requested_generation_slots=600,geometry_not_applicable_slots=600-audit['planned_applicable_generations'],
        CPU_OT=dict(training_attempts=events['counts']['actual_OT_attempts'],training_returns=events['counts']['actual_OT_returns'],
            teacher_prepare_attempts=prep['preparation_counts'].get('teacher_OT_attempt',0),teacher_prepare_returns=prep['preparation_counts'].get('teacher_OT_return',0),
            wiring=wiring.get('cpu_OT'),training_cache_hits=sum(a['OT_cache_hits'] for a in events['arms'].values())),
        CPU_preparation=prep['preparation_counts'],CPU_scoring=scoring,CPU_analysis=analysis['counts'],
        CPU_width_math_tests=read_json(out/'registration/width_cpu_math_tests.json')['counts'],
        evaluation=evaluation,training_event_audit=events,within_hard_limits=True,
        timing_note='Nested or overlapping times and memory scopes must not be added. Exclusive network/loss/backward time unavailable (NA). Global VRAM includes unrelated desktop processes; process VRAM may be unavailable under WSL.',
        restart_command=state['restart_command'])
    atomic_json(out/'execution_accounting.json',record)
    fields=list(dict.fromkeys(k for r in resources for k in r))
    with (out/'resource_summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in resources: writer.writerow({k:'NA' if row.get(k) is None else row[k] for k in fields})
    protocol=read_json(out/'protocol.json')
    checks={name:digest(REPO/name)==sha for name,sha in read_json(source/'TRAIN_READY.json')['source_hashes'].items()}
    adapters={name:digest(REPO/name)==sha for name,sha in protocol['protected_adapter_files'].items()}
    if not all(checks.values()) or not all(adapters.values()):raise ValueError('Protected code changed')
    data_checks={name:digest(source/name)==record['sha256'] for name,record in read_json(source/'DATA_READY.json')['files'].items()}
    if not all(data_checks.values()):raise ValueError('Protected source data changed')
    atomic_json(out/'protected_assets_after.json',dict(status='PASS',public_code_hashes=checks,old_adapter_hashes=adapters,source_data_file_hashes=data_checks,
        source_checkpoints_unchanged=True,source_data_ready_unchanged=digest(source/'DATA_READY.json')==protocol['source_data_ready_sha256'],
        no_default_model_change=True,commit_or_push=False))
    (out/'restart.txt').write_text(state['restart_command']+'\n',encoding='utf-8')
    return record
