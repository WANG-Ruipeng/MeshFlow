"""Freeze the authorized single original-SEAM branch; GPU still needs root GO."""
from pathlib import Path
import json
from .common import *
from . import stream, runtime


def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        r=read(dest)
        for p,h in r['training_code_hashes'].items():
            if sha(p)!=h:raise ValueError('Frozen training source changed')
        return r
    gates=[OUT/'training_cpu_gate.json',OUT/'runtime_cpu_gate.json',OUT/'data/owner_compatibility.json',OUT/'data/stream_gate.json']
    for p in gates:
        if read(p)['status']!='PASS':raise ValueError('CPU gate not passed: '+str(p))
    gate=read(gates[0])
    for p,h in gate['source_hashes'].items():
        if sha(p)!=h:raise ValueError('CPU-qualified source changed')
    prior=read(SOURCE_ROOT/'training_registration.json')
    files={resolve(p) for p in prior['training_code_hashes']}
    for p,h in prior['training_code_hashes'].items():
        if sha(p)!=h:raise ValueError('Inherited model/training dependency changed')
    files|={Path(__file__).with_name(n) for n in ('__init__.py','common.py','runtime.py','training.py','stream.py','train_tests.py','prepare.py','register.py')}
    files|={ROOT/'r2/pilots/seam_abc_v1/slot_data_extension.py',ROOT/'r2/src/meshflow_control/seam/model_adapter.py'}
    protocol=OUT.parent/'PROTOCOL.md';authorization=OUT.parent/'AUTHORIZATION.json'
    if not protocol.is_file() or not authorization.is_file():raise ValueError('Root scientific protocol and authorization must exist')
    schedule=OUT/'data/schedule_SEAM1500.json';manifest=OUT/'data/manifest.json';ev=OUT/'data/eval_manifest.json'
    stream.validate(read(schedule),read(manifest)['train'])
    if len(read(ev)['requests'])!=46:raise ValueError('Fixed46 evaluation')
    r=dict(schema='BENCH_FOUR_ROUTES_SEAM_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=1500,full_steps=[1500],checkpoint_every=250,
        source_global_step=BASE_STEP,final_global_step=24000,source_checkpoint=str(SOURCE),**SOURCE_IDENTITIES,
        model_config=prior['model_config'],batch=8,microbatch=1,hybrid_lambda=.25,
        optimizer='Inherited narrow B live AdamW/RNG, lr1e-5 betas(.9,.95) wd0 eps1e-8 clip1',
        runtime_version=runtime.VERSION,N_contract=[128,799],loss_normalization='whole_effective_batch_free_coordinates',
        training_seed=stream.DEFAULT_SEED,schedules={'SEAM1500':dict(path=str(schedule),sha256=sha(schedule))},
        manifest=str(manifest),manifest_sha256=sha(manifest),eval_manifest=str(ev),eval_manifest_sha256=sha(ev),
        training_code_hashes={str(p):sha(p) for p in sorted(files)},
        gate_hashes={str(p):sha(p) for p in gates+[protocol,authorization]},
        protocol=str(protocol),protocol_sha256=sha(protocol),authorization=str(authorization),authorization_sha256=sha(authorization),
        data_counts=dict(old_parents=22,focus_parents=2,high_parents=2,conditions=26),
        instance_budget=dict(OLD=9000,FOCUS=1500,HIGH=1500),head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        training_budget=dict(train_forward=12000,train_backward=12000,optimizer_update=1500,head_forward=0,head_backward=0,head_optimizer_update=0),
        head=dict(policy='FROZEN_HISTORICAL_OLD_S15000',parameters=833,threshold_logit=0.,
            state_sha256='21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4',
            initialized_new=False,source_owner_training_only=True,C_only_inference=True,no_oracle=True,
            inference_forwards=21,closed_C_conditions=2),
        evaluation_budget=dict(requests=46,new_RAW=46,generator_forwards=2300,new_D0=46,new_GEO=46,head_forwards=21),
        original_owner_only_representation=True,extra_loss=False,source_shared_noise=False,
        final_selection='Fixed1500 endpoint; no best selection',
        failure_contract='No silent retry or replay; exact saved boundary and ledger only; preserve failed attempts',
        no_cleanup_commit_push_default_change=True)
    write_new(dest,r);write_new(OUT/'registration.json',r)
    print(json.dumps(dict(status=r['status'],updates=1500,global_step=24000,training_budget=r['training_budget'],head=r['head'])))
    return r

if __name__=='__main__':freeze()
