"""Freeze finite training/evaluation identities before any experiment forwards."""
import json
from pathlib import Path
from .common import *

def freeze():
    dest=OUT/'training_registration.json'
    if dest.exists():
        old=read(dest)
        for p,h in old['training_code_hashes'].items():
            if sha(p)!=h: raise ValueError('Frozen training code changed: '+p)
        return old
    manifest_path=OUT/'data/manifest.json'; schedule_path=OUT/'data/schedule.json'
    eval_path=OUT/'eval_manifest.json'
    m=read(manifest_path); schedule=read(schedule_path); ev=read(eval_path)
    gate=read(OUT/'data/cpu_gate.json'); training_gate=read(OUT/'training_cpu_gate.json')
    review=read(OUT/'data/root_input_review.json')
    assert gate['status']=='PASS' and training_gate['status']=='PASS'
    assert review['status']=='PASS' and review['actual_images_viewed']>0
    budgets={'NATIVE':6000,'SEAM':3000}
    assert len(schedule['steps'])==sum(budgets.values())
    train=m['train']; parents={r['parent_id'] for r in train}
    assert len(parents)==len(train) and len(train)>=4
    assert all(int(r['K'])>=2 and 128<=int(r['N'])<=256 for r in train)
    assert len(ev['requests'])==24
    assert len({q['request_id'] for q in ev['requests']})==24
    assert len({q['condition_id'] for q in ev['requests']})==12
    assert sha(SOURCE)==SOURCE_SHA256 and sha(HEAD)==HEAD_SHA256
    prior=read(ROOT/'experiments/r2_bench_adaptation_strength_v1/training_registration.json')
    paths={resolve(p) for p in prior['training_code_hashes']}
    paths |= set((ROOT/'r2/src/meshflow_control').rglob('*.py'))
    paths |= {Path(__file__).with_name(n) for n in ('__init__.py','common.py','data.py',
              'train.py','training_state.py','register.py','dispatch.py')}
    paths |= {ROOT/'r2/pilots/bench_parts_convergence_v1/train.py',
              ROOT/'r2/pilots/bench_newdata_joint_v1/training.py'}
    labels={
      'COMMON_S12000':dict(mode='NATIVE',arm='NATIVE',local_step=0,global_step=12000,
                          generator_path=str(OUT/'training/NATIVE/generator_step0.pt')),
      'NATIVE_S6000':dict(mode='NATIVE',arm='NATIVE',local_step=6000,global_step=18000,
                        generator_path=str(OUT/'training/NATIVE/generator_step6000.pt')),
      'SEAM_S3000':dict(mode='SEAM',arm='SEAM',local_step=3000,global_step=21000,
                      generator_path=str(OUT/'training/SEAM/generator_step3000.pt'))}
    reg=dict(schema='BENCH_FULL_SEAT_CURRICULUM_REGISTRATION_V1',status='FROZEN',
      experiment=EXPERIMENT,stage_updates=budgets,stages=list(STAGES),
      stage_full_steps={'NATIVE':[0,1000,3000,6000],'SEAM':[0,1000,3000]},
      batch=8,microbatch=1,hybrid_lambda=.25,training_seed=20261023,
      source_checkpoint=str(SOURCE),source_checkpoint_sha256=SOURCE_SHA256,
      source_generator_sha256='7f96d80f8ed4cd6f803a4d5b94fb91560497cb1ea134de69eebf5916de57e5ba',
      source_optimizer_sha256='0d55263e81a89c9ddd351db52110647d833486767f18f73e85ba42c2fe597da5',
      source_rng_sha256='824e3e71a9ca6cd66d3f042a4c5b0746b63968c8f185d377af10f044e5de66e8',
      source_global_step=12000,final_global_step=21000,model_config=prior['model_config'],
      head_source_checkpoint=str(HEAD),head_source_sha256=HEAD_SHA256,
      head_state_sha256='21229acce25e3e070ae3adfa580798e1534e704cc6512e006de1654d2fefb9c4',
      head_policy='Frozen OLD_S15000; training no head calls/updates; C-only inference',
      head_threshold=0.,optimizer='Matching COMMON AdamW restored, continuous across both stages',
      augmentation='4 parents x (identity + x-width U(.9,1.1)); no XYZ or translation',
      manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
      input_schedule=str(schedule_path),input_schedule_sha256=sha(schedule_path),
      eval_manifest=str(eval_path),eval_manifest_sha256=sha(eval_path),
      evaluation_manifest=str(eval_path),evaluation_manifest_sha256=sha(eval_path),
      data_counts=dict(train_parents=len(parents),train_conditions=len(train)),
      training_code_hashes={str(p):sha(p) for p in sorted(paths)},
      protocol_sha256=sha(OUT/'PROTOCOL.md'),data_cpu_gate_sha256=sha(OUT/'data/cpu_gate.json'),
      training_cpu_gate_sha256=sha(OUT/'training_cpu_gate.json'),
      input_visual_review_sha256=sha(OUT/'data/root_input_review.json'),
      evaluation_labels=labels,expected_new_raw=72,expected_new_d0=72,
      generator_training_forwards=72000,generator_backwards=72000,optimizer_updates=9000,
      training_head_forwards=0,training_head_updates=0,probe_forwards=0,
      generator_sampling_forwards=3600,occupancy_eval_head_forward_budget=12,
      sampling_steps=50,postprocess='single Bench D0 -> GEO_C/source orientation -> HINGE_OPP015 -> ALPHA -> HOLE',
      stage_dependency='All72 RAW before any D0; NATIVE complete before SEAM training',
      final_selection='Registered6000/3000 endpoints only; no best selection/automatic extension',
      failure_contract='Preserve errors; exact complete cursor only; no silent replay/scientific setting changes',
      scientific_contrast='Whole complete-seat curriculum; no isolated causal attribution or guaranteed category transfer',
      no_cleanup_commit_push_default_change=True)
    write_new(OUT/'registration.json',reg);write_new(dest,reg)
    return reg

if __name__=='__main__':
    r=freeze();print(json.dumps({k:r[k] for k in ('status','stage_updates','data_counts','expected_new_raw')},indent=2))
