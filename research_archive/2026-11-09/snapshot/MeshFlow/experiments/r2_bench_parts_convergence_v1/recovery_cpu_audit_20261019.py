"""Read-only checkpoint/ledger audit after the WSL process-lifetime incident."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from meshflow_control.checkpoints import state_hash

ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
FOLDER=ROOT/'training/COMMON'
ARCHIVE=ROOT/'audits/dispatch_cpu_gate_handoff_20261019'
OUT=ROOT/('recovery_cpu_audit_20261019_'+str(time.time_ns()))
OUT.mkdir()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

start=time.perf_counter();status=read(ARCHIVE/'interrupted.status.json');sidecar=read(ARCHIVE/'interrupted.rolling.json');reg=read(ROOT/'registration.json')
source_paths=[ARCHIVE/('interrupted.'+x) for x in ('status.json','rolling.json','ledger.jsonl','training_log.jsonl','exposure.jsonl')]+[FOLDER/'rolling.pt']
protected={str(p):sha(p) for p in source_paths}
assert protected[str(FOLDER/'rolling.pt')]==sidecar['sha256']
cp=torch.load(FOLDER/'rolling.pt',map_location='cpu',weights_only=True,mmap=True)
assert cp['schema']=='bench_parts_training_v1' and cp['arm']=='COMMON'
assert cp['step']==cp['global_step']==cp['stage_step']==sidecar['step']==2000
assert not cp['EMA'] and cp['head_state'] is None and cp['head_optimizer'] is None and cp['head_step']==0
assert cp['registration_sha256']==sha(ROOT/'registration.json')
actual_state=state_hash(cp['generator_state'])
assert actual_state==cp['generator_state_sha256']==sidecar['generator_state_sha256']
assert sum(x.numel() for x in cp['generator_state'].values())==130401155
assert all(torch.isfinite(x).all().item() for x in cp['generator_state'].values())
optimizer=cp['optimizer'];parameter_ids=[x for g in optimizer['param_groups'] for x in g['params']]
assert len(parameter_ids)==len(set(parameter_ids))==len(optimizer['state'])
assert set(parameter_ids)==set(optimizer['state'])
cursors={float(s['step']) for s in optimizer['state'].values()};assert cursors=={2000.}
for s in optimizer['state'].values():
    assert s['exp_avg'].shape==s['exp_avg_sq'].shape
    assert s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32
    assert torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item()
for g in optimizer['param_groups']:
    assert g['lr']==reg['lr'] and list(g['betas'])==reg['betas'] and g['weight_decay']==reg['weight_decay'] and g['eps']==reg['eps']
rng=cp['rng'];random.Random().setstate(rng['python']);n=rng['numpy']
np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached']))
torch.Generator(device='cpu').set_state(rng['torch_cpu'])
assert len(rng['torch_cuda'])==1 and all(t.device.type=='cpu' and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0 for t in rng['torch_cuda'])
for key in ('manifest','input_schedule','probe_manifest','evaluation_manifest'):
    assert sha(reg[key])==reg[key+'_sha256']
for path,digest in reg['training_code_hashes'].items():assert sha(path)==digest
manifest=read(reg['manifest'])
for row in manifest['rows']:assert sha(row['case_path'])==row['case_sha256']
ledger=rows(ARCHIVE/'interrupted.ledger.jsonl');metrics=rows(ARCHIVE/'interrupted.training_log.jsonl');exposure=rows(ARCHIVE/'interrupted.exposure.jsonl')
outside=[r for r in ledger if int(r.get('step',0))>2000];counts=collections.Counter(r['event'] for r in outside)
finished=[r['step'] for r in outside if r['event']=='UPDATE_COMPLETE']
optimizers=[r['step'] for r in outside if r['event']=='OPTIMIZER_COMPLETE']
assert finished==optimizers==list(range(2001,2078))
assert [r['step'] for r in metrics if r['step']>2000]==finished
partial=[r for r in outside if r['step']==2078]
assert [r['micro'] for r in partial if r['event']=='FORWARD_COMPLETE']==[0,1,2,3]
assert [r['micro'] for r in partial if r['event']=='BACKWARD_COMPLETE']==[0,1,2]
assert not any(r['event'].startswith('OPTIMIZER') for r in partial)
assert counts['FORWARD_COMPLETE']==620 and counts['BACKWARD_COMPLETE']==619 and counts['BACKWARD_STARTED']==620
schedule=read(reg['input_schedule'])['steps'];outside_exposure=[r for r in exposure if r['step']>2000]
assert len(outside_exposure)==619
for r in outside_exposure:
    source=schedule[r['step']-1]['samples'][r['micro']]
    assert all(r[k]==v for k,v in source.items())
per_update=[]
for step in range(2001,2079):
    c=collections.Counter(r['event'] for r in outside if r['step']==step)
    per_update.append(dict(step=step,**c))
fields=sorted({key for r in per_update for key in r})
with (OUT/'lost_update_call_counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(per_update)
write(OUT/'lost_training_metrics.json',[r for r in metrics if r['step']>2000])
write(OUT/'lost_exposure_rows.json',outside_exposure)
write(OUT/'last_incomplete_update.json',partial)
input2001=schedule[2000];input2078=schedule[2077]
write(OUT/'boundary_schedule_rows.json',dict(next_checkpoint_update=input2001,incomplete_attempt_update=input2078))
# Exact checkpoint-origin comparison against already exported inference/probe.
inference=read(FOLDER/'generator_step2000.json')
assert sha(FOLDER/'generator_step2000.pt')==inference['sha256']
assert inference['generator_state_sha256']==actual_state
probes=list((FOLDER/'fixed_loss').glob('*.json'))
probe2000=[read(p) for p in probes if read(p).get('step')==2000]
assert len(probe2000)==1 and probe2000[0]['generator_state_sha256']==actual_state
assert probe2000[0]['probe_manifest_sha256']==reg['probe_manifest_sha256']
assert not torch.cuda.is_initialized()
unchanged={str(p):sha(p)==protected[str(p)] for p in source_paths};assert all(unchanged.values())
receipt=dict(status='PASS_CHECKPOINT_AND_INTERRUPTION_ACCOUNTING_CPU',checkpoint_step=2000,
    last_confirmed_optimizer_step=2077,incomplete_attempt_step=2078,
    checkpoint_file_sha256=sidecar['sha256'],generator_state_sha256=actual_state,
    schema_model_AdamW_RNG_registration_code_all316_cases_verified=True,
    optimizer_parameter_states=len(optimizer['state']),optimizer_cursors=sorted(cursors),
    cuda_rng_states_present=len(rng['torch_cuda']),CUDA_initialized=False,model_calls=0,
    discarded_attempt=dict(completed_updates=77,started_updates=78,confirmed_generator_forwards=620,
        completed_backwards=619,backwards_started=620,unresolved_backward=dict(step=2078,micro=3,
        interpretation='Started; completion unobserved. Some or all device work may have occurred; do not claim zero or completed.'),
        optimizer_updates=77,head_forwards=0,head_updates=0,new_fixed_probe_forwards=0,
        exposure_rows=619,event_counts=dict(counts),first_event_unix=outside[0]['unix'],last_event_unix=outside[-1]['unix'],
        observed_event_span_seconds=outside[-1]['unix']-outside[0]['unix']),
    schedule_next_index=2000,next_optimizer_step=2001,no_completed_2077_weight_snapshot=True,
    interpretation='step2077 cannot be resumed directly. Saved exact restart is2000; replay77 completed updates plus the partial2078 needs explicit authorization.',
    replay_check_plan=dict(authorization='Root reports user explicitly authorized one replay; this audit starts nothing',
        frozen_inputs='Same source hashes, code, schedule[2000:], per-row seeds, batch8/micro1, same AdamW/RNG2000',
        compare_completed_update_fields=['FM','generator_gradient_norm','free_coordinate_denominator','all_free_velocity_sse','all_free_scalar_count','bulk_velocity_sse','bulk_scalar_count'],
        compare_range=[2001,2077],compare_exposure_count=619,
        exclude_fields=['unix','update_seconds','input_seconds','peak_allocated_bytes','peak_reserved_bytes'],
        do_not_claim='No live2077 parameter snapshot exists; loss agreement is trajectory evidence, not direct parameter bitwise comparison',
        mismatch='Record exact difference and stop further dispatch for review; no silent second replay'),
    final_accounting=dict(scientific_common_unique_updates=12000,
        actual_common_optimizer_updates_if_one_replay_completes=12077,
        common_training_forwards_if_one_replay_completes=12000*8+620,
        common_completed_backwards_if_one_replay_completes=12000*8+619,
        common_backward_started_if_one_replay_completes=12000*8+620,
        paired_scientific_budget='Unchanged: both branches shareCOMMON12000 and equal further3000..9000',
        lost_attempt_summary_separate=True,canonical_curve='One committed update row per global step; archived original logs remain immutable',
        training_exposure_rule='Separate committed optimizer-path exposures from discarded-attempt exposures and all physical calls'),
    source_hashes=protected,source_unchanged=unchanged,seconds=time.perf_counter()-start,
    peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(OUT/'receipt.json',receipt)
print(json.dumps(dict(status=receipt['status'],checkpoint_step=2000,discarded=receipt['discarded_attempt'],receipt=str(OUT/'receipt.json'))))
