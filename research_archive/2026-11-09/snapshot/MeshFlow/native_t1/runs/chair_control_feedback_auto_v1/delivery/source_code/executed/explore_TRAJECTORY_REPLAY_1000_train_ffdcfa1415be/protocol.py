"""Freeze decisions before observing new feedback or DEV results."""
from .common import *

def register(out=OUT):
    out=Path(out)
    p=dict(task='CHAIR_CONTROL_FEEDBACK_AUTONOMOUS_OPTIMIZATION_V1',version=1,
        start_utc=START_UTC,soft_deadline_utc=SOFT_DEADLINE_UTC,hard_GPU_hours=8,
        start_checkpoint_sha256=START_SHA,default_model_changed=False,GPU_workers=1,
        max_updates=6000,max_rollouts=512,max_feedback_rollouts=128,max_NFE=25600,
        max_sample_equivalent_forwards=160000,max_precheck_forward=64,max_precheck_backward=32,
        training=dict(parents=32,batch=8,microbatch=1,width='4 alpha1 + 4 U[.9,1.1]',
            primary='Original pre-scale free-only OT, HYBRID lambda.25, original logit-normal t, globally normalized free FM',
            optimizer='restored original AdamW step5000 lr1e-5 betas(.9,.95) wd0 clip1',
            precision='FP32 parameters, BF16 forward, FP32 geometry/integration, strict deterministic, SDPA math, TF32 off',
            updates_per_arm=1000,dev_steps=[500,1000],explore_seed=20261008,repeat_seed=20261009),
        arms=list(ARMS),extra_recipe_slots=1,
        feedback=dict(source='START real Euler50 on frozen TRAIN_FEEDBACK16, all 8 training parents and both noises',
            queue='cycle S/I/E by (step-1)%3; sorted UID parent by ((step-1)//3)%8; choose worst of its two noises using S lower / I upper / E global lower; tie lower candidate index',
            times='0.6 and 0.8 alternate every complete 24-step queue cycle',lambda_aux=.1,
            trajectory='free target velocity (GT matched to final generated free slots - real Xt)/(1-t); nested face/corner bijection at final model scale',
            matched_control='same queued parent/C/Z/t and extra forward/backward; original pre_ot_free to Z OT before HYBRID .25 with independent frozen epsilon; teacher-line FM',
            interpretation='Compound real-state and endpoint-GT correspondence intervention; not pure causal state-only ablation',
            forbidden='No self-generated pseudo-GT, no detached S/I/E scalar pretending to be differentiable loss, no GT/Gamma to model'),
        dev=dict(parents=8,noises=2,outputs=16,source='original A20 parents,C,candidate0/1'),
        confirm=dict(parents=8,noises=4,outputs_per_model=32,source='first C in original holdout manifest order; historically observed shapes; independent frozen new noise',
            release='Only after favorable DEV, fixed candidate/formula/step, and independent fresh-START repeat candidate + matched control',
            labels=['START','explore_matched','explore_candidate','repeat_matched','repeat_candidate']),
        gates=dict(quality_references=['START','matching_control'],Dsym_relative_max=.05,MSE512_relative_max=.05,
            p95_relative_max=.10,FF_affected_free_absolute_max=.02,new_hard_failure_allowed=0,
            A=dict(S_vs_matched_min=.03,parent_median_positive=True,parents_improved_min=6,S_vs_START_min=0,I_max_increase=.01,E_max_decrease=.01),
            B=dict(S_min_vs_both=-.005,I_vs_matched_reduction_min=.02,E_vs_matched_increase_min=.03,other_max_regression=.01,parent_median_positive=True,parents_improved_min=6),
            severe_parent_IE_regression=.10,intervals='Conservative lower/upper; PARTIAL affecting decision -> INCOMPLETE; NA never zero',
            visual='AI_VISUAL_REVIEW only after real images read; human usability NOT_RUN'),
        stop='No default switch, no extra arms beyond budget, no solver scan, stop after report or wall window; no promotion without evidence')
    p['manifest_sha256']={name:digest(out/'manifests'/f'{name}.json') for name in ('TRAINING','TRAIN_FEEDBACK','DEV','CONFIRM')}
    sha=freeze_json(out/'protocol.json',p)
    freeze_json(out/'hypotheses.json',dict(protocol_sha256=sha,hypotheses=[
        dict(id='H1',status='REGISTERED',claim='Real late generated-state supervision can improve control beyond original FM continuation without violating shape constraints',arm='TRAJECTORY_REPLAY',falsifier='No DEV A/B gate or shape/visual regression'),
        dict(id='H2',status='REGISTERED',claim='Any benefit must exceed equal extra teacher FM supervision under the identical scored feedback queue',arm='REPLAY_FM_CONTROL',falsifier='Trajectory does not beat matched teacher control'),
        dict(id='H3',status='UNUSED_RESERVED',claim='One evidence-motivated modification may be registered only after complete DEV review and before its training')]))
    return p

if __name__=='__main__':
    print(register())
