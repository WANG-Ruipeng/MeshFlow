"""Compare authorized replay to archived observations; no model/GPU calls."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
AUDIT=ROOT/'recovery_cpu_audit_20261019_1792341018591983696'
ARCHIVE=ROOT/'audits/dispatch_cpu_gate_handoff_20261019'
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
    data=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x',encoding='utf-8') as f:f.write(data)

if (AUDIT/'replay_verification.json').exists():
    print(json.dumps(read(AUDIT/'replay_verification.json')));raise SystemExit(0)
metrics,mh=snapshot(LIVE/'training_log.jsonl');exposure,eh=snapshot(LIVE/'exposure.jsonl');ledger,lh=snapshot(LIVE/'ledger.jsonl')
if metrics is None or exposure is None or ledger is None:
    print(json.dumps(dict(status='WAIT_FOR_COMPLETE_JSONL_LINE')));raise SystemExit(0)
expected_metrics=read(AUDIT/'lost_training_metrics.json');expected_exposure=read(AUDIT/'lost_exposure_rows.json')
maxstep=max(r['step'] for r in metrics)
actual_metrics={r['step']:r for r in metrics if 2001<=r['step']<=2077}
actual_exposure={(r['step'],r['micro']):r for r in exposure if 2001<=r['step']<=2078}
if len(actual_metrics)!=77 or not all((r['step'],r['micro']) in actual_exposure for r in expected_exposure):
    print(json.dumps(dict(status='WAIT_FOR_REPLAY',last_logged_update=maxstep,metrics_ready=len(actual_metrics),exposures_ready=len(actual_exposure))));raise SystemExit(0)
assert len(metrics)==len({r['step'] for r in metrics})
assert len(exposure)==len({(r['step'],r['micro']) for r in exposure})
timing={'unix','input_seconds','update_seconds','peak_allocated_bytes','peak_reserved_bytes'}
diffs=[];stats=collections.defaultdict(lambda:dict(comparisons=0,mismatches=0,max_abs_delta=0.))
def compare(kind,key,old,new,excluded):
    assert old.keys()==new.keys(),(kind,key,'field schema differs')
    for field in sorted(old.keys()-excluded):
        a,b=old[field],new[field];s=stats[kind+'.'+field];s['comparisons']+=1
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)):
            delta=abs(float(a)-float(b));assert math.isfinite(delta);s['max_abs_delta']=max(s['max_abs_delta'],delta)
        if type(a)!=type(b) or a!=b:
            s['mismatches']+=1;diffs.append(dict(kind=kind,key=str(key),field=field,original=a,replayed=b))
for r in expected_metrics:compare('update',r['step'],r,actual_metrics[r['step']],timing)
for r in expected_exposure:compare('exposure',(r['step'],r['micro']),r,actual_exposure[r['step'],r['micro']],{'unix'})
counts=collections.Counter(r['event'] for r in ledger if 2001<=r.get('step',0)<=2077)
assert counts['FORWARD_COMPLETE']==counts['BACKWARD_COMPLETE']==77*8
assert counts['OPTIMIZER_COMPLETE']==counts['UPDATE_COMPLETE']==77
assert counts['PROBE_FORWARD_COMPLETE']==0
with (AUDIT/'replay_field_comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['field','comparisons','mismatches','max_abs_delta']);writer.writeheader()
    for key,value in sorted(stats.items()):writer.writerow(dict(field=key,**value))
immutable(AUDIT/'replay_differences.json',diffs)
receipt=dict(status='PASS_EXACT_SAVED_SCALARS_AND_EXPOSURE_REPLAY' if not diffs else 'FAIL_REPLAY_NUMERIC_MISMATCH',
    range=[2001,2077],updates_compared=77,exposure_rows_compared=619,
    all_compared_values_exactly_equal=not diffs,mismatch_count=len(diffs),
    comparison='Exact JSON scalar equality, no tolerance and no field rounding; times and allocator peaks excluded',
    excluded_update_fields=sorted(timing),excluded_exposure_fields=['unix'],
    partial2078_comparison='micro0..2 exposure SSE/counts and input identities compared; micro3 had no saved numerical result',
    explicit_limit='No saved2077 model state exists. This verifies all saved trajectory diagnostics, not parameter-by-parameter2077 identity.',
    source_checkpoint_sha256=read(AUDIT/'receipt.json')['checkpoint_file_sha256'],
    immutable_old_logs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ARCHIVE/'interrupted.training_log.jsonl',ARCHIVE/'interrupted.exposure.jsonl',ARCHIVE/'interrupted.ledger.jsonl']},
    live_snapshots=[mh,eh,lh],last_live_logged_update=maxstep,model_forwards=0,GPU_calls=0,
    fail_action='Root must stop additional dispatch and assess exact mismatch; no second replay authorized' if diffs else None,
    verified_unix=time.time())
immutable(AUDIT/'replay_verification.json',receipt)
old=read(AUDIT/'receipt.json')['discarded_attempt']
accounting=dict(schema='bench_parts_one_authorized_replay_compute_accounting_v1',
    status='RECOVERY_REPLAY_VERIFIED' if not diffs else 'REPLAY_MISMATCH_REQUIRES_REVIEW',
    reason='WSL lifecycle interrupted COMMON process during controller handoff; saved cursor2000, lost completed2001..2077 plus partial2078',
    user_authorized_replays=1,scientific_registration_changed=False,endpoint_extension=0,
    discarded_attempt_extra=dict(generator_forward_started=620,generator_forward_returned=620,
        generator_backward_started=620,generator_backward_returned=619,generator_backward_result_unknown=1,
        generator_optimizer_updates=77,head_optimizer_updates=0,fixed_probe_forwards=0,
        logged_exposure_rows=619,confirmed_training_forward_exposures=620),
    committed_path='Active logs contain each global optimizer step once after explicitly archived recovery; they define the scientific training curve.',
    total_physical_compute_rule='Final actual counts = active committed and in-flight operation ledgers + archived discarded-attempt counters above. Never add to unique update cursor.',
    common_at_unique12000=dict(unique_optimizer_updates=12000,physical_optimizer_updates=12077,
        training_forward_returns=96620,training_backward_returns=96619,training_backward_starts=96620,
        extra_unresolved_backward_calls=1,excludes_fixed_probe_forwards=True),
    total_two_arm_formula='Unique generator optimizer updates =12000+2*(final_global_step-12000); actual physical updates = unique+77. Both arm budgets remain equal.',
    exposure_policy='Report canonical per-parent/part exposure and the archived extra619 observed rows separately; the extra fourth forward input at2078 is known from frozen schedule but lacks an EXPOSURE completion row.',
    fixed_probe_policy='Existing2000 fixed probe reused:0 extra probe forwards during resumed segment.',
    replay_verification=str(AUDIT/'replay_verification.json'),source_interruption_receipt=str(AUDIT/'receipt.json'),
    original_event_span_seconds=old['observed_event_span_seconds'],
    timing_limit='98.825s is observed update-event span only; do not describe as total lost process wall/GPU time.',
    created_unix=time.time())
immutable(AUDIT/'recovery_compute_accounting.json',accounting)
print(json.dumps(dict(status=receipt['status'],mismatches=len(diffs),updates=77,exposures=619,receipt=str(AUDIT/'replay_verification.json'))))
