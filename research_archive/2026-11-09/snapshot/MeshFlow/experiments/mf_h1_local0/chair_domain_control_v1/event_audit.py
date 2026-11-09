"""CPU-only independent audit of actual training events, inputs and costs."""
from pathlib import Path
import json

ARMS=('T1','Geo','Full')
COUNT_FIELDS=('forward_attempts','forward_returns','backward_attempts','backward_returns')


def _rows(path):
    path=Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def audit_event_records(by_arm,shared,expected_steps=3000):
    """The small expected_steps override is solely for synthetic CPU fixtures."""
    errors=[];arms={};all_hashes={};expected=list(range(1,expected_steps+1))
    for arm in ARMS:
        rows=by_arm.get(arm,[])
        input_begins=[r for r in rows if r.get('event')=='INPUT_BEGIN']
        input_ready=[r for r in rows if r.get('event')=='INPUT_READY']
        begins=[r for r in rows if r.get('event')=='UPDATE_BEGIN']
        dones=[r for r in rows if r.get('event')=='UPDATE_VALIDATED']
        segments=[r for r in rows if r.get('event')=='SEGMENT_END']
        failures=[r for r in rows if r.get('event')=='FAILURE']
        if failures:errors.append(arm+': failure attempt recorded; not a complete single run')
        for name,selected in (('INPUT_BEGIN',input_begins),('INPUT_READY',input_ready),('BEGIN',begins),('VALIDATED',dones)):
            steps=[r.get('phase_step') for r in selected]
            if steps!=expected:errors.append(arm+': '+name+' must be ordered, continuous and unique 1..'+str(expected_steps))
            for r in selected:
                phase=r.get('phase_step',-1)
                if r.get('conditional_total_step')!=1000+phase or r.get('logical_step')!=1000+phase:
                    errors.append(arm+': '+name+' phase/total identity differs');break
        begun={r.get('phase_step'):r.get('batch_sha256') for r in begins}
        hashes={r.get('phase_step'):r.get('batch_sha256') for r in dones}
        if begun!=hashes:errors.append(arm+': attempted and validated input hashes differ')
        if {r.get('phase_step'):r.get('batch_sha256') for r in input_ready}!=hashes:
            errors.append(arm+': constructed and validated input hashes differ')
        if any(r.get('OT_delta')!=dict(actual_OT_attempts=8,actual_OT_returns=8,OT_cache_hits=0) or
               r.get('stream_batch_index_after')!=r.get('phase_step') for r in input_ready):
            errors.append(arm+': per-batch actual OT delta/cursor differs')
        if any(r.get('stream_batch_index_before')!=r.get('phase_step',0)-1 for r in input_begins):
            errors.append(arm+': input-attempt stream cursor differs')
        all_hashes[arm]=hashes
        for r in begins:
            if r.get('counts_before',{}).get('optimizer_updates',0)!=r.get('phase_step',0)-1:
                errors.append(arm+': BEGIN actual optimizer cursor differs');break
        for r in dones:
            phase=r.get('phase_step',-1);counts=r.get('counts',{})
            if counts.get('optimizer_updates')!=phase or counts.get('optimizer_step_attempts')!=phase or r.get('new_updates')!=phase:
                errors.append(arm+': VALIDATED optimizer counts differ');break
            if any(counts.get(k)!=8*phase for k in COUNT_FIELDS):
                errors.append(arm+': VALIDATED microbatch counters differ');break
            if counts.get('optimizer_step_in_progress') is not False or counts.get('update_validated') is not True:
                errors.append(arm+': incomplete optimizer boundary');break
            sample=r.get('sample_diagnostics',[])
            if len(sample)!=8 or sum(x['free_coordinates'] for x in sample)!=r.get('free_coordinate_denominator'):
                errors.append(arm+': variable free-coordinate denominator differs');break
            if sum(x.get('alpha')==1. for x in sample)!=4:
                errors.append(arm+': frozen four/four alpha schedule differs');break
            if arm=='Full' and any(x.get('Joint',{}).get('ramp')!=1. or
                    x.get('Joint',{}).get('active')!=(x['t']>=.5) for x in sample):
                errors.append(arm+': Full ramp/gate differs');break
        actual_updates=sum(int(r.get('actual_updates_this_worker',-100000)) for r in segments)
        OT_attempts=sum(int(r.get('OT_counts',{}).get('actual_OT_attempts',-100000)) for r in segments+failures)
        OT_returns=sum(int(r.get('OT_counts',{}).get('actual_OT_returns',-100000)) for r in segments+failures)
        cache_hits=sum(int(r.get('OT_counts',{}).get('OT_cache_hits',-100000)) for r in segments+failures)
        if actual_updates!=expected_steps:errors.append(arm+': segment actual update sum differs')
        if OT_attempts!=8*expected_steps or OT_returns!=OT_attempts or cache_hits!=0:
            errors.append(arm+': actual OT attempts/returns or cache count differs')
        if segments and any(r.get('status') not in ('SEGMENT_COMPLETE','INTERRUPTED_SAFE') for r in segments):
            errors.append(arm+': invalid segment completion state')
        final=dones[-1].get('counts',{}) if dones else {}
        arms[arm]=dict(unique_validated_updates=len(set(hashes)),validated_updates=len(dones),begin_events=len(begins),
            actual_segment_updates=actual_updates,actual_OT_attempts=OT_attempts,actual_OT_returns=OT_returns,OT_cache_hits=cache_hits,
            training_microbatch_forwards=final.get('forward_attempts',0),training_microbatch_backwards=final.get('backward_attempts',0),
            input_begin_events=len(input_begins),input_ready_events=len(input_ready),
            segment_attempts=len(segments),failed_attempts=len(failures),
            timing=dict(training_update_seconds=sum(float(r.get('worker_training_update_seconds',r.get('worker_train_seconds',0))) for r in segments),
                OT_seconds=sum(float(r.get('worker_OT_seconds',0)) for r in segments),
                save_seconds=sum(float(r.get('worker_save_seconds',0)) for r in segments),
                worker_wall_seconds=sum(float(r.get('worker_wall_seconds',0)) for r in segments)))
    if any(all_hashes[a]!=all_hashes['T1'] for a in ARMS):errors.append('Three arms actual batch hashes differ')
    shared_steps=[r.get('phase_step') for r in shared]
    if shared_steps!=expected:errors.append('Shared input ledger is not unique, ordered and complete')
    shared_hashes={r.get('phase_step'):r.get('batch_sha256') for r in shared}
    if shared_hashes!=all_hashes['T1']:errors.append('Registered shared inputs differ from actual arm inputs')
    counts=dict(optimizer_updates=sum(a['actual_segment_updates'] for a in arms.values()),
        training_forward_attempts=sum(a['training_microbatch_forwards'] for a in arms.values()),
        training_backward_attempts=sum(a['training_microbatch_backwards'] for a in arms.values()),
        actual_OT_attempts=sum(a['actual_OT_attempts'] for a in arms.values()),
        actual_OT_returns=sum(a['actual_OT_returns'] for a in arms.values()))
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,counts=counts,arms=arms,
        shared_batches=len(shared_hashes),cross_arm_input_identity=all(all_hashes[a]==all_hashes['T1'] for a in ARMS),
        timing_scope='Per process SEGMENT_END events are summed, including safe interrupted segments; train/update includes forward, loss, backward, clip, optimizer and validation, not exclusive network time')


def audit_training_events(out):
    try:
        out=Path(out)
        return audit_event_records({a:_rows(out/'training'/f'{a}_events.jsonl') for a in ARMS},
            _rows(out/'training/shared_batch_hashes.jsonl'))
    except (ValueError,KeyError,TypeError,OSError) as exc:
        return dict(status='FAIL',errors=['Malformed/missing training ledger: '+repr(exc)],counts={},arms={})
