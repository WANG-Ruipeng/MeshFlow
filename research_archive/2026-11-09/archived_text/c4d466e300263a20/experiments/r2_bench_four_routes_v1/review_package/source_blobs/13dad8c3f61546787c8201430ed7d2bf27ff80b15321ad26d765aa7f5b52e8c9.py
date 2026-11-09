"""Freeze the authorized M+3000 continuation and its original46 panel."""
import json
from pathlib import Path
from . import common as C,stream as Q


def freeze():
    old=C.read(C.M_ROOT/'training_registration.json')
    gate=C.read(C.OUT/'training_cpu_gate.json');streamgate=C.read(C.OUT/'data/stream_gate.json')
    if gate['status']!='PASS' or streamgate['status']!='PASS':raise ValueError('CPU preparation not passed')
    m=C.read(C.MANIFEST);s=C.read(C.SCHEDULE)
    Q.validate_manifest(m);Q.validate(s,m['train']);Q.checked_coefficients(C.COEFFICIENTS,s,m)
    if len(C.read(C.EVAL_MANIFEST)['requests'])!=46:raise ValueError('Fixed panel changed')
    files={C.resolve(p) for p in old['training_code_hashes']}
    for p,h in old['training_code_hashes'].items():
        if C.sha(p)!=h:raise ValueError('Historical dependency changed: '+p)
    files.update(Path(__file__).parent.glob('*.py'))
    for p,h in gate['source_hashes'].items():
        if C.sha(p)!=h:raise ValueError('CPU-tested source changed')
    if C.sha(C.SOURCE)!=C.SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('M source changed')
    gatepaths=[C.OUT/'training_cpu_gate.json',C.OUT/'data/stream_gate.json',C.MASTER/'PROTOCOL.md',C.MASTER/'AUTHORIZATION.json']
    r=dict(schema='BENCH_FOUR_ROUTES_EXPOSURE_REGISTRATION_V1',status='FROZEN',experiment=C.EXPERIMENT,
        arms=list(C.ARMS),updates=3000,full_steps=[3000],checkpoint_every=250,
        source_global_step=22500,final_global_step=25500,source_checkpoint=str(C.SOURCE),**C.SOURCE_IDENTITIES,
        model_config=old['model_config'],batch=8,microbatch=1,hybrid_lambda=.25,
        optimizer='Retained M live/AdamW/RNG; lr1e-5 betas(.9,.95) wd0 clip1; no reset',
        runtime_version=old['runtime_version'],N_contract=[128,799],loss_normalization=C.LOSS_NORMALIZATION,
        schedules={C.ARMS[0]:dict(path=str(C.SCHEDULE),sha256=C.sha(C.SCHEDULE))},
        manifest=str(C.MANIFEST),manifest_sha256=C.sha(C.MANIFEST),eval_manifest=str(C.EVAL_MANIFEST),eval_manifest_sha256=C.sha(C.EVAL_MANIFEST),
        coefficient_schedule=str(C.COEFFICIENTS),coefficient_schedule_sha256=C.sha(C.COEFFICIENTS),
        training_code_hashes={str(p):C.sha(p) for p in sorted(files)},gate_hashes={str(p):C.sha(p) for p in gatepaths},
        protected={str(p):C.sha(p) for p in (C.SOURCE,C.SOURCE_INFERENCE,C.MANIFEST,C.EVAL_MANIFEST,C.ORIGINAL_SCHEDULE,C.ORIGINAL_COEFFICIENTS)},
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        total_budget=dict(train_forward=24000,train_backward=24000,optimizer_update=3000,new_RAW=46,sampling_forward=2300,new_D0=46),
        exposure=streamgate,scientific_boundary='Additional compute and per-parent exposure; OLD/FOCUS also increase, not a pure parent-count causal contrast',
        failure_contract='No silent scientific replay; exact safely checkpointed cursor continuation only',
        no_cleanup_commit_push_default_change=True)
    C.write_new(C.OUT/'training_registration.json',r)
    return r


if __name__=='__main__':
    r=freeze();print(json.dumps({k:r[k] for k in ('status','arms','updates','final_global_step','total_budget')}))
