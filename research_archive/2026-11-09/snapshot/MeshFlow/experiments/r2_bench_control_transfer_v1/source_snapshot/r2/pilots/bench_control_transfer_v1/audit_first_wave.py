"""Read-only accounting of the first already-authorized 25 formal updates."""
from collections import Counter
import json,statistics,time
from . import train as T

def lines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    root=T.DEFAULT_ROOT;out=root/'training/PLAIN';limit=25
    ledger=[r for r in lines(out/'ledger.jsonl') if int(r.get('step',0))<=limit]
    metrics=[r for r in lines(out/'training_log.jsonl') if r['step']<=limit]
    exposure=[r for r in lines(out/'exposure.jsonl') if r['step']<=limit]
    counts=Counter(r['event'] for r in ledger)
    assert counts['UPDATE_STARTED']==counts['UPDATE_COMPLETE']==limit
    assert counts['FORWARD_STARTED']==counts['FORWARD_COMPLETE']==counts['BACKWARD_STARTED']==counts['BACKWARD_COMPLETE']==8*limit
    assert counts['OPTIMIZER_STARTED']==counts['OPTIMIZER_COMPLETE']==limit
    assert not any('HEAD_' in r['event'] for r in ledger)
    assert len(exposure)==limit*8
    schedule=T.read(root/'input_schedule.json')['steps']
    reg=T.read(root/'registration.json')
    for r in exposure:
        s=schedule[r['step']-1]['samples'][r['micro']]
        for k,v in s.items():assert r[k]==v
        assert r['parent_id']==reg['records'][s['index']]['parent_id']
    values=metrics[1:]
    facts=dict(status='PASS',audited_step_range=[1,25],update_calls=25,train_forwards=200,backwards=200,occupancy_calls=0,
        exposure_rows=len(exposure),distinct_parents=len({r['parent_id'] for r in exposure}),
        clean_count=sum(r['alpha']==1. for r in exposure),width_augmented_count=sum(r['alpha']!=1. for r in exposure),
        schedule_identity_exact=True,median_update_seconds=statistics.median(r['update_seconds'] for r in values),
        mean_update_seconds=statistics.mean(r['update_seconds'] for r in values),mean_CPU_input_seconds=statistics.mean(r['input_seconds'] for r in values),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in metrics),peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in metrics),
        device_before=T.read(out/'resources_before.json'),device_after_25=T.read(out/'resources_step25.json'),
        scope='No training, generation, or replay. Per-process GPU VRAM unavailable under WDDM; device delta is a conservative estimate with background variation.')
    T.write(root/'first_wave_independent_audit.json',facts)
    events=dict(status='RECORDED',scientific_changes=False,events=[
        dict(id='PRE_GPU_SHELL_QUOTING_01',stage='CPU_INPUT_CHECK_INVOCATION',symptom='Nested Python exec string quoting rejected by bash before Python launch',generator_forwards=0,optimizer_updates=0,OT_calls=0,resolution='Run standalone check_inputs.py; CPU16 OT checks passed, no GPU calls'),
        dict(id='PRE_GPU_ROOT_INLINE_PRINT_01',time_local_as_reported='2026-10-18 09:21 Asia/Shanghai',stage='REGISTRATION_INSPECTION',symptom='PowerShell inline dict-print NameError after checked() completed',generator_forwards=0,optimizer_updates=0,source='Root agent report',resolution='No model invocation or data mutation; continue unchanged registered run')])
    T.write(root/'engineering_invocation_events.json',events)
    print({k:v for k,v in facts.items() if not isinstance(v,dict)})
if __name__=='__main__':main()
