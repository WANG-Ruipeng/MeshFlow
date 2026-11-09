"""Freeze the paired exposure experiment before any training call."""
from __future__ import annotations
import json
from .common import *
from . import stream


def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        r=read(dest)
        for p,h in r['training_code_hashes'].items():
            if sha(p)!=h:raise ValueError('Frozen training source changed: '+p)
        return r
    paths=dict(manifest=OUT/'data/manifest.json',eval_manifest=OUT/'data/eval_manifest.json',
        data_gate=OUT/'data/cpu_gate.json',training_gate=OUT/'training_cpu_gate.json',
        cache_gate=OUT/'audits/P0_cache_audit.json',protocol=OUT/'PROTOCOL.md')
    m,ev=read(paths['manifest']),read(paths['eval_manifest'])
    if not m.get('actual_data_ready') and m.get('status')!='READY':raise ValueError('Data not ready')
    for key in ('data_gate','training_gate','cache_gate'):
        if read(paths[key]).get('status')!='PASS':raise ValueError('Gate not passed: '+key)
    gate=read(paths['training_gate'])
    if gate['source_identities']!=SOURCE_IDENTITIES:raise ValueError('CPU gate source differs')
    for p,h in gate['source_hashes'].items():
        if sha(p)!=h:raise ValueError('CPU-tested source changed')
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:raise ValueError('Source file changed')
    original=read(SOURCE_ROOT/'data/manifest.json')
    if m!=original:raise ValueError('Data differs from Coverage24')
    req=ev['requests']
    if ev.get('status')!='FROZEN' or len(req)!=30 or len({q['request_id'] for q in req})!=30:
        raise ValueError('Fixed30 evaluation required')
    parents={r['parent_id'] for r in m['train']}
    if any(q['parent_id'] in parents for q in req if q['report_group'] in ('DEV','HOLDOUT')):
        raise ValueError('Heldout parent leakage')
    schedules={}
    for arm in ARMS:
        p=OUT/'data'/('schedule_'+arm+'.json')
        stream.validate(read(p),m['train'],1500,arm)
        schedules[arm]=dict(path=str(p),sha256=sha(p))
    prior=read(SOURCE_ROOT/'training_registration.json')
    files={resolve(p) for p in prior['training_code_hashes']}
    files|={Path(__file__).with_name(n) for n in ('__init__.py','common.py','training.py','stream.py','train_tests.py','prepare.py','register.py')}
    # Inherited source is frozen transitively; evaluation is frozen separately
    # once its CPU implementation is complete, before its first model call.
    for p,h in prior['training_code_hashes'].items():
        if sha(p)!=h:raise ValueError('Inherited source changed since Coverage: '+p)
    r=dict(schema='BENCH_EXPOSURE_ALLOCATION_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=1500,full_steps=[1500],checkpoint_every=250,
        source_global_step=BASE_STEP,final_global_step=21000,source_checkpoint=str(SOURCE),
        **SOURCE_IDENTITIES,model_config=prior['model_config'],batch=8,microbatch=1,hybrid_lambda=.25,
        optimizer='Inherited matching Coverage live AdamW and RNG; lr1e-5 betas(.9,.95) wd0 clip1',
        training_seed=stream.DEFAULT_SEED,schedules=schedules,
        manifest=str(paths['manifest']),manifest_sha256=sha(paths['manifest']),
        eval_manifest=str(paths['eval_manifest']),eval_manifest_sha256=sha(paths['eval_manifest']),
        training_code_hashes={str(p):sha(p) for p in sorted(files)},
        gate_hashes={str(paths[k]):sha(paths[k]) for k in ('data_gate','training_gate','cache_gate','protocol')},
        protocol_sha256=sha(paths['protocol']),data_counts=dict(old_parents=22,focus_parents=2,conditions=24),
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        per_arm_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500),
        total_budget=dict(train_forward=24000,train_backward=24000,optimizer_update=3000,
            sampling_forward=3000,new_RAW=60,new_D0=60),
        all_RAW_before_D0=True,sampling_steps=50,
        final_selection='Predeclared1500 endpoint only; no best checkpoint or extension',
        failure_contract='No silent retry/replay; only exact safe saved boundary may resume explicitly',
        scientific_contrast='Fixed Coverage24 pool, 50/50 versus75/25 group exposure; no extra loss',
        data_expansion_policy='CPU-only independent preparation, excluded from both training and evaluation',
        no_cleanup_commit_push_default_change=True)
    write_new(dest,r);write_new(OUT/'registration.json',r)
    return r


if __name__=='__main__':
    r=freeze();print(json.dumps({k:r[k] for k in ('status','arms','updates','final_global_step','total_budget')}))
