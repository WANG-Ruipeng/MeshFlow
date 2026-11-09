"""Optional final exploration recipe. Registration precedes any model call."""
import argparse
import datetime
from .common import *
from .loss_balance import ARM, frozen_recipe


def register(out=OUT):
    out=Path(out)
    for arm in ('FM_CONTROL','TRAJECTORY_REPLAY','REPLAY_FM_CONTROL'):
        done=read_json(out/'training'/f'explore_{arm}_complete.json')
        assert done['steps']==1000 and done['status']=='PASS'
    original_proof_path=out/'cap_control_equivalence/final/receipt.json'
    proof=read_json(original_proof_path)
    # Preserve the exact successful proof and executed verifier source before later audits.
    archive=out/'decisions/H3_control_equivalence_freeze'
    proof_path=archive/'receipt.json'
    assert proof['status']=='PASS_CAP_INACTIVE_1000'
    assert proof['safe_inactive_steps']==1000 and not proof['cap_active_steps']
    view_path=out/'audit/TRAJECTORY_REPLAY_1000_full_visual_review_vs_START.json'
    review=read_json(view_path)
    assert review['reviewer_type']=='AI_VISUAL_REVIEW' and len(review['rows'])==16
    assert all(r['actually_viewed'] for r in review['rows'])
    gate_path=out/'comparisons/DEV/explore_TRAJECTORY_REPLAY_1000__vs__explore_REPLAY_FM_CONTROL_1000/gate.json'
    gate=read_json(gate_path)
    assert gate['numeric_status']=='FAIL'
    post=out/'derived/posttraining_feedback/complete.json'
    assert read_json(post)['status']=='PASS'
    accounting=read_json(out/'accounting.json')
    assert accounting['returns']['optimizer_update']==3000 and not accounting['pending']
    copies={original_proof_path:proof_path,
        original_proof_path.with_name('per_step.csv'):archive/'per_step.csv',
        Path(__file__).with_name('cap_control_equivalence.py'):archive/'cap_control_equivalence.py',
        Path(__file__).with_name('training_diagnostics.py'):archive/'training_diagnostics.py',
        out/'decisions/training_starts.jsonl':archive/'training_starts_at_freeze.jsonl'}
    archive.mkdir(parents=True,exist_ok=True)
    for source,target in copies.items():
        data=source.read_bytes()
        if target.exists():assert target.read_bytes()==data
        else:
            with target.open('xb') as handle:handle.write(data)
    evidence=[proof_path,view_path,gate_path,post,
        out/'training_diagnostics/live/TR1000_learning_summary.json',
        out/'evaluation/explore_TRAJECTORY_REPLAY_500/DEV/summary.json',
        out/'evaluation/explore_TRAJECTORY_REPLAY_1000/DEV/summary.json']
    evidence_records=[]
    evidence_archive=out/'decisions/H3_evidence_freeze'
    evidence_archive.mkdir(parents=True,exist_ok=True)
    for index,source in enumerate(evidence):
        target=evidence_archive/f'{index:02d}_{source.name}'
        data=source.read_bytes()
        if target.exists():assert target.read_bytes()==data
        else:
            with target.open('xb') as handle:handle.write(data)
        evidence_records.append(dict(original_path=str(source),path=str(target),sha256=digest(target)))
    recipe=dict(status='FROZEN',arm=ARM,hypothesis='H3',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        START_sha256=START_SHA,parent_protocol_sha256=digest(out/'protocol.json'),
        feedback_pool_sha256=digest(out/'feedback/pool_v1/manifest.json'),
        coefficient_formula='min(0.1, detached_python_FM_sum / max(detached_FP32_aux_MSE, 1e-30))',
        primary_stream_arm_alias='TRAJECTORY_REPLAY',auxiliary_arm_alias='TRAJECTORY_REPLAY',
        inactive_branch='loss=.1*raw; exact original FP32 multiply retained',
        active_branch='loss=lambda_eff*raw; detached Python scalar, no gradient through FM or ratio',
        loss_balance_source_sha256=digest(Path(__file__).with_name('loss_balance.py')),
        exploration_matched_control_equivalence=dict(path=str(proof_path),sha256=digest(proof_path)),
        evidence=evidence_records,
        observed_DEV1000_vs_matched=gate['vs_matched']['stats'],
        observed_DEV1000_vs_START=gate['vs_START']['stats'],
        reason='TR1000 fits all frozen auxiliary positions and improves actual TRAIN_FEEDBACK S versus START by 0.211476, but DEV S drops by 0.052553 and quality/control fail. The matched training-pool S gain 0.079714 has zero parent median and only 3/8 improving parents. Its .8-time weighted auxiliary remains large and frequently dominates clipping. Test whether limiting loss scale preserves local benefits while reducing these costs; this does not assume the cap resolves the observed transfer gap.',
        claim_scope='A detached loss-value cap up to FP32 multiplication rounding, not a gradient norm cap. Compared with original trajectory replay it changes only this coefficient rule. Compared with teacher control it remains a compound state/correspondence/cap intervention.',
        control='Existing REPLAY_FM_CONTROL exploration path is exactly the inactive-cap teacher path by complete 1000-step audited induction; no extra teacher updates are claimed.',
        repeat_control='If promoted, fresh-seed candidate and teacher each run 1000; the repeat teacher also requires complete inactive-cap verification before CONFIRM. An active or inconclusive cap forbids the same-algorithm equivalence claim.',
        unchanged=['Original batch8 FM','All primary inputs and update order','Same START/full AdamW','Same 16 feedback trajectories and S/I/E queue','Same .6/.8 times and GT correspondence','Original Euler50, output count, scoring and gates','No new trainable parameters or deployment inputs'],
        possible_failure=['Loss cap need not bound auxiliary gradients','Stale START trajectories may not transfer to the current policy','Limited feedback parents can induce specialization','Primary FM continuation itself can degrade this DEV panel','Original TR improves MSE relative to the matched teacher while failing START quality; reducing auxiliary strength could lose that partial shape benefit'],
        stop='No success unless original quality and A/B gates plus actual visual review pass; no fourth hypothesis or further exploration recipe.',
        selection_step=1000,intermediate_DEV_step=500,selection_rule='Only the preselected 1000-step endpoint can trigger repeat; 500-step signals remain intermediate evidence.',
        budgets=dict(updates=1000,DEV_rollouts=32,posttraining_feedback_rollouts=16,aux_forward_backward_per_update=1),
        postfeedback='Same original frozen TRAIN_FEEDBACK16 and Euler50 at final endpoint; diagnostic only, not new training targets or promotion panel.',
        sources={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('training.py'),Path(__file__).with_name('checkpoint.py'),Path(__file__).with_name('loss_balance.py')]})
    freeze_json(out/'decisions/H3_recipe.json',recipe)
    frozen_recipe(out)
    append_event(out/'decisions/research_decisions.jsonl',dict(event='H3_FROZEN',arm=ARM,recipe_sha256=digest(out/'decisions/H3_recipe.json'),optimizer_updates_before=3000))
    return recipe


def diagnostic(out=OUT):
    from .run import call_module,ready,score
    from .aggregation import paired
    from experiments.mf_h1_local0.chair_control_usability_quick_v1.scoring import write_csv
    out=Path(out);recipe,sha=frozen_recipe(out)
    assert recipe['budgets']['posttraining_feedback_rollouts']==16
    label=f'explore_{ARM}_1000';identity=f'explore_{ARM}'
    assert read_json(out/'training'/f'{identity}_complete.json')['steps']==1000
    manifest=out/'manifests/TRAIN_FEEDBACK.json'
    if not ready(out,label,'TRAIN_FEEDBACK'):
        p,h=call_module('worker',['sample','--out',str(out),'--label',label,'--checkpoint',str(out/'checkpoints'/f'{label}.pt'),'--manifest',str(manifest)],out/'logs/H3_postfeedback.log')
        code=p.wait();h.close()
        if code:raise RuntimeError('H3 diagnostic failed; preserve attempts and inspect before recovery')
    if not (out/'evaluation'/label/'TRAIN_FEEDBACK/summary.json').exists():score(out,label,'TRAIN_FEEDBACK')
    rows=read_json(out/'evaluation'/label/'TRAIN_FEEDBACK/per_output.json');requests=read_json(manifest)['rows']
    folder=out/'derived/posttraining_feedback_H3'
    for ref in ('START','explore_REPLAY_FM_CONTROL_1000','explore_TRAJECTORY_REPLAY_1000'):
        comparison=paired(rows,read_json(out/'evaluation'/ref/'TRAIN_FEEDBACK/per_output.json'),requests)
        freeze_json(folder/f'vs_{ref}.json',dict(role='TRAINING_COHORT_DIAGNOSTIC_ONLY',**comparison))
        for key in ('outputs','parents','noise'):write_csv(folder/f'vs_{ref}_{key}.csv',comparison[key])
    freeze_json(folder/'complete.json',dict(status='PASS',outputs=16,recipe_sha256=sha,training_targets_modified=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','train','diagnostic']);args=parser.parse_args()
    if args.stage=='register':print(register()['status'])
    elif args.stage=='train':
        from .run import train_stage
        frozen_recipe(OUT);train_stage(OUT,'explore',ARM,1000)
    else:diagnostic()
