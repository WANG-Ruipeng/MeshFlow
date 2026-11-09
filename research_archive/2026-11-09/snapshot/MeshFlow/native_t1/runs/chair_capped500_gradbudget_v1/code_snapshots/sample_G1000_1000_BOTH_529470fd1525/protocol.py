"""Freeze this experiment before the first model call, never revise a frozen contract."""
from pathlib import Path
from datetime import datetime,timezone
from .common import *
from .ledger import LIMITS
from .stream import SEED


def freeze(out=OUT):
    out=Path(out)
    if (out/'protocol.json').exists():return read_json(out/'protocol.json')
    registration=read_json(out/'registration/receipt.json');assert registration['status']=='PASS'
    old=read_json(OLD/'protocol.json')
    p=dict(task='CHAIR_CAPPED500_REPLICATION_AND_GRADIENT_BUDGET_V1',version=1,
        registered_utc=datetime.now(timezone.utc).isoformat(),task_contract_sha256=digest(out/'task_contract.txt'),
        start_checkpoint_sha256=START_SHA,arms=ARM_NAMES,rho=1.,GPU_workers=1,
        hard_GPU_occupancy_seconds=14400,limits=LIMITS,top_level_forward_limit=41524,
        normal_new_generations=240,historical_cached_outputs=48,full_evaluation_cells=288,
        endpoints=dict(primary=500,stability=1000),panels=['OLD','FRESH'],
        manifest_sha256={name:digest(out/'manifests'/f'{name}.json') for name in ('OLD','FRESH','ALL','TRAINING','TRAIN_FEEDBACK','CONFIRM')},
        training=dict(seed=SEED,seed_namespace='chair_capped500_gradbudget_v1/train_repeat_1',
            auxiliary_frozen_teacher_seed=AUX_SEED,auxiliary_seed_note='New primary continuation random stream; original fixed teacher-line states and epsilon preserved from pool, no new feedback data.',
            updates_per_arm=1000,batch=8,microbatch=1,parents=32,ready_tasks=453,
            primary='Unchanged HYBRID .25/free-only nested OT/logit-normal t/global free scalar denominator',
            width='4 alpha1 + 4 independent U[.9,1.1], original slot shuffle',
            optimizer='Restored START AdamW step5000 moments, lr1e-5 betas(.9,.95) wd0 clip1',
            precision='FP32 parameters and geometry; BF16 forward, original eval-mode dropout; deterministic strict, TF32 off, SDPA math',
            gradient_carrier='8 primary F/B -> full FP32 gF buffer -> clear .grad -> raw aux1F/B -> detached scalar -> combine -> global clip1 -> AdamW',
            true_parameter_update_steps=[s for s in range(1,1001) if s==1 or s%25==0 or s==501]),
        coefficient=dict(loss='min(0.1, detached Python LF / max(detached FP32 LA,1e-30))',
            gradient='rho*||gF||/(||gA||+1e-12)',G='min(lambda_loss,lambda_grad)',
            zero_F='G coefficient zero',zero_A='zero auxiliary contribution',
            norms='All actual trainable parameters, complete unweighted gradients, pre total-clip; None=zero, detached',
            no_AdamW_update_bound_claim=True),
        gates=dict(A_replication=dict(OLD=dict(S_mean_min=.03,S_median_positive=True,S_improved_parents_min=6),
            FRESH=dict(S_mean_positive=True,S_median_positive=True,S_improved_parents_min=5)),
            B_quality_protection=dict(S_mean_min=-.01,MSE512_relative_max=-.05,MSE2048_relative_max=-.05,
                MSE2048_improved_parents_min=5,I_mean_max=.01,E_mean_min=-.01,Dsym_relative_max=.05,
                require_control_effect_vs_T=True),promotion=old['gates']),
        real_preflight_tolerances=dict(combined_gradient_relative_L2_max=.01,combined_gradient_cosine_min=.9999,
            auxiliary_gradient_relative_L2_max=.05,normalization_absolute_max=1e-8),
        preflight_plan='One fixed first batch:8 primaryF/B + T2auxF/B + L2auxF/B; one discarded isolated optimizer copy; retain0 updates.',
        original_scorers={str(q):digest(q) for q in [ROOT/'experiments/mf_h1_local0/evaluate.py',
            ROOT/'experiments/mf_h1_local0/chair_boundary_attachment_v1/control_metrics.py',
            ROOT/'experiments/mf_h1_local0/chair_control_usability_quick_v1/scoring.py']},
        default_model_changed=False,CONFIRM_evaluation=False,feedback_pool_refresh=False,
        forbidden=['rho scan','new loss','time reweighting','endpoint residual replacement','optimizer change','new module','guidance','postprocess','automatic commit/push'],
        failure='Preserve attempt/log/state; engineering repair only within remaining counts. Scientific definition change stops this version.')
    p['scientific_sources']={str(Path(__file__).with_name(name)):digest(Path(__file__).with_name(name)) for name in ('training.py','stream.py','gradients.py','checkpoint.py','preflight.py')}
    freeze_json(out/'protocol.json',p)
    for arm in ARMS:
        freeze_json(out/'recipes'/f'{arm}.json',dict(status='FROZEN',arm=arm,name=ARM_NAMES[arm],protocol_sha256=json_hash(p),
            start_sha256=START_SHA,auxiliary_pool_sha256=digest(OLD/'feedback/pool_v1/manifest.json'),
            target='teacher-line' if arm=='T' else 'frozen real START states and matched GT velocity',
            loss_cap=True,gradient_budget=arm=='G',rho=1.,primary_seed=SEED,auxiliary_teacher_seed=AUX_SEED))
    freeze_json(out/'hypotheses.json',dict(A='L500 vs T500 replicates development seam effect on OLD and FRESH separately',
        B='G500 preserves seam gains while reducing missing-GT surface error vs L500',
        C='G reduces 500-to-1000 regression only if L actually exhibits it',
        interpretation='Finite fresh-START continuation replication; fixed feedback pool; FRESH is new noise on observed DEV parents, not new-shape confirmation.'))
    return p


if __name__=='__main__':
    result=freeze();print(result['task'],result['training']['seed'],json_hash(result))
