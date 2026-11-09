"""Second interruption: CPU-only saved5250 payload and immutable tail accounting."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
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
OUT=ROOT/('recovery2_cpu_audit_20261019_'+str(time.time_ns()));OUT.mkdir()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8-sig').splitlines() if x.strip()]

started=time.perf_counter();snapshots={}
# Snapshot cheap source evidence first; no mutation of production files.
for source in [*[FOLDER/x for x in ('status.json','rolling.json','ledger.jsonl','training_log.jsonl','exposure.jsonl')],
               ROOT/'registration.json',ROOT/'dispatch/failure.json']:
    raw=source.read_bytes();target=OUT/('original.'+source.name)
    with target.open('xb') as f:f.write(raw)
    snapshots[str(source)]=dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
status=read(OUT/'original.status.json');sidecar=read(OUT/'original.rolling.json');reg=read(OUT/'original.registration.json')
assert status['step']==5362 and sidecar['step']==5250
ledger=rows(OUT/'original.ledger.jsonl');metrics=rows(OUT/'original.training_log.jsonl');exposure=rows(OUT/'original.exposure.jsonl')
tail=[r for r in ledger if r.get('step',0)>5250];counts=collections.Counter(r['event'] for r in tail)
done=[r['step'] for r in tail if r['event']=='UPDATE_COMPLETE'];assert done==list(range(5251,5363))
assert [r['step'] for r in tail if r['event']=='UPDATE_STARTED']==done
assert [r['step'] for r in tail if r['event']=='OPTIMIZER_COMPLETE']==done
assert counts['FORWARD_STARTED']==counts['FORWARD_COMPLETE']==counts['BACKWARD_STARTED']==counts['BACKWARD_COMPLETE']==896
tailmetrics=[r for r in metrics if r['step']>5250];tailexposure=[r for r in exposure if r['step']>5250]
assert [r['step'] for r in tailmetrics]==done and len(tailexposure)==896
assert len({(r['step'],r['micro']) for r in tailexposure})==896
assert max(r.get('step',0) for r in ledger)==5362
perstep=[]
for step in done:
    c=collections.Counter(r['event'] for r in tail if r['step']==step)
    assert c['FORWARD_COMPLETE']==c['BACKWARD_COMPLETE']==8 and c['OPTIMIZER_COMPLETE']==1
    perstep.append(dict(step=step,**c))
with (OUT/'lost_update_call_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for r in perstep for k in r}));w.writeheader();w.writerows(perstep)
write(OUT/'lost_training_metrics.json',tailmetrics);write(OUT/'lost_exposure_rows.json',tailexposure)
write(OUT/'tail_ledger.json',tail)
print(json.dumps(dict(phase='TAIL_FROZEN',output=str(OUT),completed_updates=112,forwards=896,backwards=896,partial_updates=0)),flush=True)

checkpoint=FOLDER/'rolling.pt';assert sha(checkpoint)==sidecar['sha256']
cp=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert cp['schema']=='bench_parts_training_v1' and cp['arm']=='COMMON'
assert cp['step']==cp['global_step']==cp['stage_step']==5250
assert cp['EMA'] is False and cp['head_state'] is None and cp['head_optimizer'] is None and cp['head_step']==0
assert cp['registration_sha256']==snapshots[str(ROOT/'registration.json')]['sha256']==sha(ROOT/'registration.json')
modelhash=state_hash(cp['generator_state']);assert modelhash==sidecar['generator_state_sha256']==cp['generator_state_sha256']
assert sum(v.numel() for v in cp['generator_state'].values())==130401155
assert all(v.device.type=='cpu' and torch.isfinite(v).all().item() for v in cp['generator_state'].values())
opt=cp['optimizer'];ids=[i for g in opt['param_groups'] for i in g['params']]
assert len(ids)==len(set(ids))==len(opt['state']) and set(ids)==set(opt['state'])
assert {float(s['step']) for s in opt['state'].values()}=={5250.}
for s in opt['state'].values():
    assert s['exp_avg'].shape==s['exp_avg_sq'].shape and s['exp_avg'].dtype==s['exp_avg_sq'].dtype==torch.float32
    assert torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item()
for g in opt['param_groups']:
    assert g['lr']==reg['lr'] and list(g['betas'])==reg['betas'] and g['weight_decay']==reg['weight_decay'] and g['eps']==reg['eps']
rng=cp['rng'];random.Random().setstate(rng['python']);n=rng['numpy']
np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],np.uint32),n['pos'],n['has_gauss'],n['cached']))
torch.Generator(device='cpu').set_state(rng['torch_cpu'])
assert len(rng['torch_cuda'])==1 and all(t.device.type=='cpu' and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>0 for t in rng['torch_cuda'])
trainfile=Path('/mnt/e/MeshFlow/r2/pilots/bench_parts_convergence_v1/train.py')
assert sha(trainfile)==reg['training_code_hashes'][str(trainfile)]
assert sha(reg['input_schedule'])==reg['input_schedule_sha256']
schedule=read(reg['input_schedule'])['steps']
for r in tailexposure:
    s=schedule[r['step']-1]['samples'][r['micro']]
    assert all(r[k]==v for k,v in s.items())
assert not torch.cuda.is_initialized()
# Small-file hashes detect concurrent edits; avoid a second huge-weight read.
unchanged={p:sha(p)==s['sha256'] for p,s in snapshots.items()}
assert all(unchanged.values()),'Production evidence changed during audit'
write(OUT/'receipt.json',dict(status='PASS_STEP5250_PAYLOAD_AND_TAIL_ACCOUNTING',checkpoint=str(checkpoint),
    checkpoint_file_sha256=sidecar['sha256'],generator_state_sha256=modelhash,checkpoint_step=5250,
    generator_parameters=130401155,optimizer_state_count=len(opt['state']),optimizer_step=5250,
    saved_RNG_CPU_states_loadable=True,saved_CUDA_RNG_present=True,CUDA_initialized=False,model_forwards=0,
    registration_sha256=cp['registration_sha256'],training_code_sha256=sha(trainfile),
    input_schedule_sha256=reg['input_schedule_sha256'],all896_tail_inputs_match_schedule=True,
    cases_reaudited=0,scope='Necessary checkpoint payload, optimizer/RNG, registration, unchanged trainer and fixed schedule only; no repeat full316case audit',
    tail=dict(completed_updates=112,first_step=5251,last_step=5362,partial_updates=0,
        forward_started=896,forward_completed=896,backward_started=896,backward_completed=896,
        unknown_backward=0,optimizer_updates=112,head_updates=0,new_fixed_probe_forwards=0,
        exposure_rows=896,event_counts=dict(counts),observed_event_span_seconds=tail[-1]['unix']-tail[0]['unix']),
    authorization='AUDIT_ONLY; second recovery requires fresh explicit authorization, no replay started here',
    replay_plan='If authorized: exact5250 model/AdamW/RNG and schedule5251; compare112 update records and896 exposures to these immutable snapshots, excluding timing/allocator fields only',
    final_compute_accounting_if_second_replay_is_authorized=dict(
        prior_incident_updates=77,this_incident_updates=112,total_extra_optimizer_updates=189,
        total_extra_forward_returns=1516,total_extra_backward_returns=1515,total_extra_backward_unknown=1,
        common_unique12000_physical_updates=12189,scientific_unique_budget_unchanged=True),
    snapshots=snapshots,source_unchanged=unchanged,seconds=time.perf_counter()-started,
    peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(json.dumps(dict(status='PASS_STEP5250_PAYLOAD_AND_TAIL_ACCOUNTING',receipt=str(OUT/'receipt.json'),updates=112,forwards=896,backwards=896)),flush=True)
