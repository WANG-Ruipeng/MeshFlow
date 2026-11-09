"""Second authorized recovery replay: CPU log comparison only."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
AUDIT=ROOT/'recovery2_cpu_audit_20261019_1792345993997925922'
LIVE=ROOT/'training/COMMON'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def snapshot(p):
    raw=Path(p).read_bytes();lines=raw.decode('utf-8-sig').splitlines();rows=[]
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
            return None,None
    return rows,dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),snapshot_unix=time.time())
def immutable(path,data):
    with path.open('x',encoding='utf-8') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

final=AUDIT/'replay_verification.json'
if final.exists():print(json.dumps(read(final)));raise SystemExit(0)
metrics,mh=snapshot(LIVE/'training_log.jsonl');exposure,eh=snapshot(LIVE/'exposure.jsonl');ledger,lh=snapshot(LIVE/'ledger.jsonl')
if any(v is None for v in (metrics,exposure,ledger)):
    print(json.dumps(dict(status='WAIT_FOR_COMPLETE_JSONL_LINE')));raise SystemExit(0)
expected_metrics=read(AUDIT/'lost_training_metrics.json');expected_exposure=read(AUDIT/'lost_exposure_rows.json')
maxstep=max(r['step'] for r in metrics)
new_metrics={r['step']:r for r in metrics if 5251<=r['step']<=5362}
new_exposure={(r['step'],r['micro']):r for r in exposure if 5251<=r['step']<=5362}
if len(new_metrics)!=112 or len(new_exposure)!=896:
    print(json.dumps(dict(status='WAIT_FOR_REPLAY',last_logged_update=maxstep,metrics_ready=len(new_metrics),exposures_ready=len(new_exposure))));raise SystemExit(0)
assert len(metrics)==len({r['step'] for r in metrics})
assert len(exposure)==len({(r['step'],r['micro']) for r in exposure})
timing={'unix','input_seconds','update_seconds','peak_allocated_bytes','peak_reserved_bytes'}
differences=[];stats=collections.defaultdict(lambda:dict(comparisons=0,mismatches=0,max_abs_delta=0.))
def compare(kind,key,old,new,excluded):
    if old.keys()!=new.keys():
        differences.append(dict(kind=kind,key=str(key),field='SCHEMA',original=sorted(old),replayed=sorted(new)))
    for field in sorted((old.keys()&new.keys())-excluded):
        a,b=old[field],new[field];s=stats[kind+'.'+field];s['comparisons']+=1
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)):
            delta=abs(float(a)-float(b));assert math.isfinite(delta);s['max_abs_delta']=max(s['max_abs_delta'],delta)
        if type(a)!=type(b) or a!=b:
            s['mismatches']+=1;differences.append(dict(kind=kind,key=str(key),field=field,original=a,replayed=b))
for r in expected_metrics:compare('update',r['step'],r,new_metrics[r['step']],timing)
for r in expected_exposure:compare('exposure',(r['step'],r['micro']),r,new_exposure[r['step'],r['micro']],{'unix'})
counts=collections.Counter(r['event'] for r in ledger if 5251<=r.get('step',0)<=5362)
assert counts['FORWARD_COMPLETE']==counts['BACKWARD_COMPLETE']==896
assert counts['OPTIMIZER_COMPLETE']==counts['UPDATE_COMPLETE']==112 and counts['PROBE_FORWARD_COMPLETE']==0
skip=[r for r in ledger if r.get('step')==5250 and r['event']=='PROBE_NOT_SCHEDULED']
assert len(skip)==1
assert skip[0]['status']=='NOT_SCHEDULED' and skip[0]['reason']=='EXACT_OFF_GRID_RESUME_CURSOR'
assert skip[0]['generator_forwards']==skip[0]['generator_forward_budget']==skip[0]['head_forwards']==0
assert skip[0]['checkpoint_sha256']==read(AUDIT/'receipt.json')['checkpoint_file_sha256']
assert not (LIVE/'fixed_loss/step5250.json').exists()
assert not any(r.get('step')==5250 and r['event'] in ('PROBE_STARTED','PROBE_FORWARD_STARTED','PROBE_FORWARD_COMPLETE','PROBE_COMPLETE') for r in ledger)
grid=LIVE/'fixed_loss/step6000.json'
if grid.exists():
    probe=read(grid)
    assert probe['step']==6000 and len(probe['rows'])==96
    pcounts=collections.Counter(r['event'] for r in ledger if r.get('step')==6000)
    assert pcounts['PROBE_FORWARD_COMPLETE']==96 and pcounts['PROBE_COMPLETE']==1
    scheduled=dict(status='PASS_REGISTERED6000_PROBE',path=str(grid),sha256=hashfile(grid),generator_forward_returns=96)
else:scheduled=dict(status='NOT_YET_AVAILABLE',expected_step=6000,observation_live_step=maxstep,not_a_failure=True)
with (AUDIT/'replay_field_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['field','comparisons','mismatches','max_abs_delta']);writer.writeheader()
    for field,value in sorted(stats.items()):writer.writerow(dict(field=field,**value))
immutable(AUDIT/'replay_differences.json',differences)
result=dict(status='PASS_EXACT_SAVED_SCALARS_AND_EXPOSURE_REPLAY' if not differences else 'FAIL_REPLAY_NUMERIC_MISMATCH',
    replay_incident=2,range=[5251,5362],updates_compared=112,exposure_rows_compared=896,
    all_compared_values_exactly_equal=not differences,mismatch_count=len(differences),
    comparison='Exact JSON scalar equality, no tolerance or rounding; only timing/allocator fields excluded',
    excluded_update_fields=sorted(timing),excluded_exposure_fields=['unix'],
    offgrid_resume_probe=dict(status='PASS_NO_EXTRA_FORWARD',record=skip[0],artifact_exists=False),
    next_registered_probe=scheduled,
    explicit_limit='Original5362 parameters were not saved. Exact agreement of saved trajectory diagnostics is not direct parameter-by-parameter5362 identity.',
    live_snapshots=[mh,eh,lh],last_live_logged_update=maxstep,model_forwards=0,GPU_calls=0,verified_unix=time.time())
immutable(final,result)
first=ROOT/'recovery_cpu_audit_20261019_1792341018591983696/recovery_compute_accounting.json'
assert read(first)['status']=='RECOVERY_REPLAY_VERIFIED'
accounting=dict(schema='bench_parts_two_authorized_replays_compute_accounting_v1',
    status='BOTH_REPLAYS_VERIFIED' if not differences else 'SECOND_REPLAY_MISMATCH_REQUIRES_REVIEW',
    scientific_registration_changed=False,unique_endpoint_extension=0,
    incidents=[dict(incident=1,optimizer_updates=77,forward_returns=620,backward_returns=619,backward_result_unknown=1,
        source=str(first)),dict(incident=2,optimizer_updates=112,forward_returns=896,backward_returns=896,
        backward_result_unknown=0,extra_probe_forwards=0,source=str(AUDIT/'receipt.json'))],
    total_discarded_attempt_extra=dict(optimizer_updates=189,training_forward_returns=1516,
        training_backward_returns=1515,training_backward_starts=1516,backward_result_unknown=1,
        head_optimizer_updates=0,extra_fixed_probe_forwards=0),
    final_common_if_unique12000=dict(unique_optimizer_updates=12000,physical_optimizer_updates=12189,
        training_forward_returns=97516,training_backward_returns=97515,training_backward_starts=97516,
        excludes_registered_fixed_probe_forwards=True),
    all_arms_formula='unique optimizer updates=12000+2*(paired_final_global_step-12000); physical optimizer updates=unique+189',
    log_policy='Canonical logs keep one committed record per global step. Add archived discarded-attempt counts for physical compute; do not merge discarded rows into learning curves.',
    replay_verification=str(final),first_incident_accounting=str(first),created_unix=time.time())
immutable(AUDIT/'recovery_compute_accounting.json',accounting)
print(json.dumps(dict(status=result['status'],mismatches=len(differences),updates=112,exposures=896,
    skip5250='PASS_0_FORWARD',probe6000=scheduled['status'],receipt=str(final))))
