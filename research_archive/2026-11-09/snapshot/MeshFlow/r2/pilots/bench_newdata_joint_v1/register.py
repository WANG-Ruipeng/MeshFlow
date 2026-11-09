"""Freeze authorized1500updates and28pairedrequests before any new neural calls."""
from pathlib import Path
import json
from .common import *

def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists(): raise FileExistsError('Already frozen; validate instead')
    old=read(BASELINE_ROOT/'training_registration.json')
    start=BASELINE_ROOT/'training/A_COMMON_LR1E5/checkpoint_step0.pt'
    sr=read(start.with_suffix('.json'))
    assert sha(start)==sr['sha256']
    gate=read(OUT/'cpu_gate.json');assert gate['status']=='PASS' and gate['neural_forwards']==0
    inv=read(OUT/'data/inventory.json');m=read(OUT/'data/manifest.json')
    evaluation=OUT/'eval_manifest.json';ev=read(evaluation)
    assert len(ev['requests'])==28
    assert sum(bool(q['baseline_reuse']) for q in ev['requests'])==8
    head=ROOT/'experiments/r2_bench_three_family_v1/production_contract/old_s15000_head_only.pt'
    assert sha(head)=='9802355787ff2abf9c02f983f9ca8547bec2182c32162a309cb5048927d51ce1'
    paths=[resolve(p) for p in old['training_code_hashes']]
    for p,h in old['training_code_hashes'].items():assert sha(p)==h,p
    # Freeze training dependencies only; evaluation wrappers are frozen at their own stage.
    paths += [Path(__file__).with_name(n) for n in ('__init__.py','common.py','data.py','prepare.py','training.py','register.py','dispatch.py')]
    paths += [ROOT/'r2/src/meshflow_control/seam/occupancy.py',ROOT/'r2/pilots/bench_relation_training_v1/data.py']
    labels={REFERENCE_LABEL:dict(arm='RELATION_V1_SEAM',subset='full',local_step=1500,global_step=13500,
        generator_path=str(REFERENCE_ROOT/'training/RELATION_V1_SEAM/generator_step1500.pt')),
        LABEL:dict(arm=ARM,subset='full',local_step=1500,global_step=13500,
        generator_path=str(OUT/'training'/ARM/'generator_step1500.pt'))}
    reg=dict(schema='BENCH_NEWDATA_JOINT_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        arms=list(ARMS),updates=1500,batch=8,microbatch=1,hybrid_lambda=.25,
        source_checkpoint=str(start),source_checkpoint_sha256=sr['sha256'],
        source_generator_sha256=sr['generator_state_sha256'],source_optimizer_sha256=sr['optimizer_state_sha256'],
        source_rng_sha256=sr['rng_sha256'],source_global_step=12000,final_global_step=13500,training_seed=20261020,
        model_config=old['model_config'],training_code_hashes={str(p):sha(p) for p in paths},
        manifest=str(OUT/'data/manifest.json'),manifest_sha256=sha(OUT/'data/manifest.json'),
        input_schedule=str(OUT/'data/schedule.json'),input_schedule_sha256=sha(OUT/'data/schedule.json'),
        eval_manifest=str(evaluation),eval_manifest_sha256=sha(evaluation),
        evaluation_manifest=str(evaluation),evaluation_manifest_sha256=sha(evaluation),
        data_counts=dict(rows=len(m['rows']),train=92,dev=39,train_parents=27),
        head_source_checkpoint=str(head),head_source_sha256=sha(head),
        head_state_sha256='21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4',
        head_source_legacy_state_sha256='e8276bd60887c7d9e740693025d1d4a40f34f0149e8609535526d605f041c710',
        head_hash_definition='head_state_sha256 uses meshflow_control.checkpoints.state_hash; legacy uses earlier helper',
        head_class_weights=inv['head_class_weights'],head_forward_budget=inv['head_forward_budget'],
        head_optimizer_updates=1500,head_optimizer=dict(lr=.001,betas=[.9,.95],eps=1e-8,weight_decay=0,clip=1.,fresh=True),
        head_initial_historical_updates=2996,head_threshold=0.,head_architecture='24-32-SiLU-1,833 existing parameters',
        protocol_sha256=sha(OUT/'PROTOCOL.md'),cpu_gate_sha256=sha(OUT/'cpu_gate.json'),
        optimizer='Exact A-step0 fresh AdamW,lr1e-5,betas(.9,.95),eps1e-8,wd0,clip1',
        scientific_contrast='Whole new-data+head-adaptation recipe versus RELATION; not a causal single-new-parent ablation',
        fresh_training_forwards_per_arm=12000,fresh_backwards_per_arm=12000,fresh_updates_per_arm=1500,
        fixed_probe_forwards=0,new_parameters=0,
        expected_new_raw=48,expected_new_d0=48,reused_raw=8,reused_d0=8,
        generator_forward_sampling=2400,occupancy_eval_head_forward_budget=19,
        evaluation_labels=labels,final_selection='1500only',
        failure_contract='Preserve failures; no silent replay or scientific setting changes; exact completed cursor resume allowed',
        precision='BF16forward,FP32parameters/integration,deterministic warn_onlyFalse,SDPA math,TF32off',
        no_commit_push_default_change=True)
    write_new(OUT/'resource_initial.json',resources())
    write_new(OUT/'registration.json',reg);write_new(dest,reg)
    return dict(status='FROZEN',updates=1500,head_forward_budget=inv['head_forward_budget'],eval_requests=28)

if __name__=='__main__':print(json.dumps(freeze()))
