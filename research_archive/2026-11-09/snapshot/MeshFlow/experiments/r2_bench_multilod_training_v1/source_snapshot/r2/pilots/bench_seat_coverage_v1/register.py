"""Freeze the one-arm data-coverage experiment after data/panel review gates."""
from __future__ import annotations
import collections, json
from pathlib import Path
from .common import *
from . import schedule as Q


def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        r=read(dest)
        for p,h in r['training_code_hashes'].items():
            if sha(p)!=h: raise ValueError('Frozen training source changed: '+p)
        if read(OUT/'registration.json')!=r:raise ValueError('Registration aliases disagree')
        return r
    paths={k:OUT/v for k,v in dict(manifest='data/manifest.json',input_schedule='data/schedule.json',
        eval_manifest='eval_manifest.json',budget='budget.json',data_gate='data/cpu_gate.json',
        training_gate='training_cpu_gate.json',review='data/root_input_review.json').items()}
    values={k:read(p) for k,p in paths.items()}
    m,ev,budget=values['manifest'],values['eval_manifest'],values['budget']
    if m.get('status')!='READY' and m.get('actual_data_ready') is not True:
        raise ValueError('Reviewed training manifest is not ready')
    if any(values[k].get('status')!='PASS' for k in ('data_gate','training_gate')):
        raise ValueError('Data/CPU/review PASS gates required before registration')
    review=values['review']; observations=review.get('records',[])
    if review.get('reviewer_kind')!='AI_PROXY' or len(observations)!=5 or not all(r.get('actually_viewed') for r in observations):
        raise ValueError('Five actual input-image observations required')
    reviewed={r['condition_id'] for r in observations if r['decision']=='ACCEPT_COMPLETE_SEAT'}
    chosen={r['condition_id'] for r in m['train'] if r['replay_group']=='NEW'} | {r['condition_id'] for r in m.get('dev',[])}
    if reviewed!=chosen:raise ValueError('Selected new train/holdout do not match actual review')
    for r in observations:
        if sha(r['image_path'])!=r['image_sha256']:raise ValueError('Reviewed input image changed')
    if ev.get('status')!='FROZEN' or not ev.get('requests'):
        raise ValueError('Fixed evaluation inputs required')
    if len({r['request_id'] for r in ev['requests']})!=len(ev['requests']):
        raise ValueError('Repeated evaluation request ID')
    updates=budget['updates']
    if type(updates) is not int or updates not in (1500,3000):
        raise ValueError('This finite pilot supports the agreed1500 or explicitly revised3000 budget')
    endpoint=budget.get('final_global_step',budget.get('endpoint_global_step'))
    full_steps=budget.get('full_steps',[0,updates])
    checkpoint_every=budget.get('checkpoint_every',250)
    if budget['source_global_step']!=BASE_STEP or endpoint!=BASE_STEP+updates:
        raise ValueError('Branch global cursor differs')
    if full_steps!=[0,updates] or checkpoint_every!=250:
        raise ValueError('Only local0/endpoint immutable checkpoints plus250 rolling saves')
    Q.validate(values['input_schedule'],m['train'],updates)
    groups=Q.groups_for(m['train'])
    old=read(SOURCE_ROOT/'data/manifest.json')['train']
    if {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in old} != {r['condition_id']:(r['parent_id'],r['case_sha256']) for r in m['train'] if r['replay_group']=='OLD'}:
        raise ValueError('Original22-condition replay pool changed')
    trainparents={r['parent_id'] for r in m['train']}
    heldout={r['parent_id'] for r in m.get('dev',[])}
    heldout.update(r['parent_id'] for r in ev['requests'] if r.get('eval_group',r.get('group')) in ('DEV','HOLDOUT','NEW_DEV','INDEPENDENT_DEV','NEW_PARENT_HOLDOUT'))
    if trainparents&heldout:raise ValueError('Evaluation parent leakage')
    for r in m.get('rows',m['train']):
        if sha(r['case_path'])!=r['case_sha256']:raise ValueError('Frozen case bytes changed')
    if any(not 128<=int(r['N'])<=256 or not 2<=int(r['K'])<int(r['N']) for r in m['train']):
        raise ValueError('N128..256/K>=2 completion contract differs')
    receipt=read(SOURCE.with_suffix('.json'))
    if sha(SOURCE)!=SOURCE_IDENTITIES['source_checkpoint_sha256']:
        raise ValueError('Retained Native6000 checkpoint bytes changed')
    for k,hk in (('generator_state_sha256','source_generator_sha256'),('optimizer_state_sha256','source_optimizer_sha256'),('rng_sha256','source_rng_sha256')):
        if receipt[k]!=SOURCE_IDENTITIES[hk]:raise ValueError('Source sidecar identity differs')
    prior=read(SOURCE_ROOT/'training_registration.json')
    files={resolve(p) for p in prior['training_code_hashes']}
    files|={Path(__file__).with_name(n) for n in ('__init__.py','common.py','train.py','training_state.py','register.py','schedule.py','test_training_cpu.py')}
    files|={ROOT/'r2/pilots/bench_full_seat_curriculum_v1/train.py',ROOT/'r2/pilots/bench_full_seat_curriculum_v1/training_state.py'}
    gatekeys=('budget','data_gate','training_gate','review')
    labels=dict(
        NATIVE_S6000=dict(mode='NATIVE',arm='NATIVE',local_step=6000,global_step=18000,
            generator_path=str(SOURCE_ROOT/'training/NATIVE/generator_step6000.pt')),
        COVERAGE_FINAL=dict(mode='NATIVE',arm='NATIVE',local_step=updates,global_step=BASE_STEP+updates,
            generator_path=str(OUT/f'training/NATIVE/generator_step{updates}.pt')),
        SEAM_S3000=dict(mode='SEAM',arm='SEAM',local_step=3000,global_step=21000,
            generator_path=str(SOURCE_ROOT/'training/SEAM/generator_step3000.pt')))
    reg=dict(schema='BENCH_SEAT_COVERAGE_REGISTRATION_V1',status='FROZEN',experiment=EXPERIMENT,
        stage='NATIVE',updates=updates,full_steps=full_steps,checkpoint_every=250,
        source_global_step=BASE_STEP,final_global_step=BASE_STEP+updates,
        source_checkpoint=str(SOURCE),**SOURCE_IDENTITIES,model_config=prior['model_config'],
        batch=8,microbatch=1,hybrid_lambda=.25,optimizer='Matching Native6000 AdamW, moments/parameter order inherited',
        new_data_stream=True,old_data_stream_resumed=False,training_seed=values['input_schedule']['seed'],
        manifest=str(paths['manifest']),manifest_sha256=sha(paths['manifest']),
        input_schedule=str(paths['input_schedule']),input_schedule_sha256=sha(paths['input_schedule']),
        eval_manifest=str(paths['eval_manifest']),eval_manifest_sha256=sha(paths['eval_manifest']),
        data_counts=dict(old_parents=len(groups['OLD']),new_parents=len(groups['NEW']),train_conditions=len(m['train'])),
        training_code_hashes={str(p):sha(p) for p in sorted(files)},gate_hashes={str(paths[k]):sha(paths[k]) for k in gatekeys},
        protocol_sha256=sha(OUT/'PROTOCOL.md'),
        head_forward_budget=0,head_optimizer_updates=0,probe_forward_budget=0,
        generator_training_forwards=8*updates,generator_backwards=8*updates,optimizer_updates=updates,
        augmentation='OLD2+NEW2 distinct parents x(identity+x-width U(.9,1.1))',
        representation='NATIVE_ALL_FREE_OT',head_policy='No training/inference head for this Native candidate',
        inference_export_schema='bench_seat_coverage_generator_v1',
        inference_export=str(OUT/f'training/NATIVE/generator_step{updates}.pt'),
        evaluation_labels=labels,expected_new_raw=ev['new_RAW'],expected_new_d0=ev['new_D0'],
        generator_sampling_forwards=ev['generator_forwards'],occupancy_eval_head_forward_budget=0,
        sampling_steps=50,postprocess='single Bench D0 -> GEO_C/source orientation -> HINGE_OPP015 -> ALPHA -> HOLE',
        final_selection='Predeclared endpoint, no best selection, automatic SEAM or extension',
        failure_contract='Preserve failures; only exact INTERRUPTED_SAFE cursor can resume without replay',
        scientific_contrast='Joint data-coverage/replay continuation, not an isolated data-only causal control',
        no_cleanup_commit_push_default_change=True)
    write_new(OUT/'registration.json',reg);write_new(dest,reg)
    return reg


if __name__=='__main__':
    r=freeze();print(json.dumps({k:r[k] for k in ('status','updates','data_counts','final_global_step')},indent=2))
