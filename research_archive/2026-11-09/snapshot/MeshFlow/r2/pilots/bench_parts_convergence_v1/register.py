"""Freeze data, budgets, loss convergence and evaluation before the first update."""
from __future__ import annotations
import argparse, collections, json, shutil
from pathlib import Path
from . import data
from .train import DEFAULT_ROOT, ROOT, resources, write, sha, resolve
from meshflow_control.config import MODEL_CONFIG

def register(root=DEFAULT_ROOT):
    root=resolve(root)
    if (root/'registration.json').exists():raise RuntimeError('Registration already frozen; do not overwrite')
    manifest=data.read(root/'data/manifest.json')
    verification=data.read(root/'data/cpu_verification.json')
    if verification.get('status')!='PASS':raise ValueError('All-case CPU verification required')
    positive=negative=0
    for row in manifest['train']:
        labels=data.load_case(row)['occupied_labels']
        positive+=int(labels.sum());negative+=int((~labels).sum())
    weights=[(positive+negative)/(2*negative),(positive+negative)/(2*positive)] if positive and negative else [1.,1.]
    code_paths=[ROOT/'r2/pilots/bench_parts_convergence_v1'/name for name in ('train.py','data.py','convergence.py','register.py')]
    code_paths += [ROOT/p for p in ('r2/pilots/bench_control_transfer_v1/train.py','r2/pilots/seam_abc_v1/slot_data_extension.py',
        'r2/src/meshflow_control/seam/slot_data.py','r2/src/meshflow_control/seam/model_adapter.py',
        'r2/src/meshflow_control/seam/occupancy.py','r2/src/meshflow_control/models/native.py',
        'r2/src/meshflow_control/training/coupling.py','r2/src/meshflow_control/training/schedule40.py',
        'r2/src/meshflow_control/runtime.py')]
    official=ROOT/'output/120m-ot-v-bench/checkpoints/01000000.pt'
    expected='00fc5ddb5bc5690cd53fe4f092ee3320e23437a287464fa5639120c2bb82a505'
    if sha(official)!=expected:raise ValueError('Official Bench identity changed')
    reg=dict(schema='R2_BENCH_MULTIPART_CHAIR_MATCHED_CONVERGENCE_V1',status='FROZEN',category='bench',synset='02828884',
        user_authorization='2026-10-18: use multipart dataset, record loss, observe convergence; explicitly official Bench from step 0 even if slow',
        arms=['COMMON','PLAIN','SEAM'],common_updates=12000,minimum_stage_updates=3000,maximum_stage_updates=9000,
        maximum_generator_optimizer_updates_total=30000,common_global_interval=[0,12000],paired_global_interval=[12000,21000],
        model_config=MODEL_CONFIG,model_seed=20261018,official=str(official),official_sha256=expected,
        initialization='Official Bench EMA + Native/T1 role/Geo zero exits; fresh AdamW; no old Bench/Chair continuation',
        fork='COMMON12000 model and matching AdamW/RNG cloned to PLAIN and SEAM; schedule global cursor 12001',
        batch=8,microbatch=1,lr=1e-5,betas=[.9,.95],eps=1e-8,weight_decay=0.,clip=1.,hybrid_lambda=.25,
        objective='Unchanged free-corner FM; whole-batch effective scalar denominator; role-constrained OT before HYBRID',
        augmentation='Four distinct parents/update, each same C at alpha1 plus width U[.9,1.1]; parent-uniform part/variant cycles',
        precision='BF16 forward; FP32 parameters/AdamW/integration; strict deterministic algorithms; TF32 off; SDPA math',
        head_lr=.001,head_seed=20261016,head_class_weights=weights,head_counts=dict(owner=positive,inactive=negative),
        head_policy='Existing 833-parameter C-only per-edge MLP; balanced BCE frozen from TRAIN pool; logit>=0; joint independent optimizer throughout SEAM; skip closed C. No generator gradient from head.',
        head_Chair_difference='Historical Chair PLAIN_S2000 used frozen LONG C-only head/layout; this new multipart Bench head trains on its own TRAIN C throughout. No DEV/source-label layout at inference.',
        checkpoint_policy='Atomic rolling resume every250; fixed inference snapshot every1000; named full-state12000,15000,21000 and actual final. No best-checkpoint selection.',
        stopping=dict(minimum_updates=3000,maximum_updates=9000,probe_interval=1000,adjacent_intervals=3,absolute_relative_change_lt=.02,
                      strata=['ALL_free','bulk','seam_tip'],major_part_worsening_gt=.05,major_part_worsening_intervals=3,
                      paired_rule='Both arms plateau at same current stage endpoint; otherwise advance both1000 up to cap. DEV is diagnostic only.',
                      finite_cap_not_convergence=True,extra_updates_beyond_cap=False),
        major_parts=['SEAT_SURFACE','LEG_SUPPORT','BACK_SURFACE','SEAT_SUPPORT','SEAT_BACK','PART_ASSEMBLY'],
        evaluation=dict(full_requests=48,process_requests=8,noises_per_C=2,loss_probe_C=32,loss_probe_times=[.2,.5,.8],
                        loss_probe_interval=1000,loss_probe_generator_forwards_per_checkpoint=96,
                        process_common_steps=[2000,6000,10000],process_paired_steps=[13000,17000,19000],
                        full_steps=[12000,15000,'actual_final'],include_both_arms=True,
                        generation='Euler50 fixed standard Gaussian; C-only head layout; no source completion at inference',
                        post='One official Bench D0 fixed_std=.05; restore C and seam anchors; original GEO_C; source winding restore + fixed-C free orientation',
                        D0_sha256='67765c728c503b61c21624c05997ea026c9de486d1529052fafa134be3336b14',
                        original_metrics='Existing complete 124-field metric bank via independent wrapper; RAW and POST separate; closed-C interface metrics NA; open-C all-null remains zero',
                        visual_review='actual AI_PROXY; same scale, dual mesh views and C-centered front/side sections; per-part and parent summaries',
                        no_GT_unique_style_ranking=True,no_model_oracle_layout=True,raw_barrier='All registered RAW in a comparison phase saved before D0'),
        comparison_limits='Within Bench, matched Native-vs-SEAM from same common fork; cross-category Chair is a process reference, not a category-only causal experiment.',
        failure_policy='Stop new dispatch on OOM/numerical error; no silent update replay or scientific setting change. Preserve failed artifacts; explicit engineering-recovery record required.',
        dispatch_policy='Measure real training first updates incl backward/AdamW; headroom=max(2GiB,10% total). Incrementally launch independent arms only when measured memory and compute justify.',
        training_code_hashes={str(p):sha(p) for p in code_paths},no_commit_push_or_default_change=True)
    paths=dict(manifest=root/'data/manifest.json',input_schedule=root/'data/input_schedule.json',
               probe_manifest=root/'fixed_loss_probe/manifest.json',evaluation_manifest=root/'evaluation_inputs/manifest.json')
    for key,p in paths.items():reg[key]=str(p);reg[key+'_sha256']=sha(p)
    r=resources();r['disk']=shutil.disk_usage(root)._asdict();r['dispatch']='COMMON has a single shared lineage: one training worker. Paired independent arms are evaluated for concurrency after the common phase.'
    write(root/'resources_before_training.json',r)
    write(root/'registration.json',reg)
    return reg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(DEFAULT_ROOT));a=p.parse_args()
    reg=register(a.root);print(json.dumps(dict(status=reg['status'],common=12000,paired_min=3000,paired_max=9000,head_counts=reg['head_counts'])))
