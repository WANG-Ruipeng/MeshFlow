"""Freeze source-resolution data partitions before C/Y construction, no training."""
from __future__ import annotations
import csv
import json
from .split_audit import ROOT, OUT, CURRENT, read, sha, save, parent_set

BASE = ROOT / 'experiments/r2_bench_deployment_task_data_v1'
REVIEWED = BASE / 'asset_inventory/reviewed_candidates.json'
NOTES = BASE / 'asset_inventory/visual_notes.json'
HELDOUT_PREFIXES = ('617fa8', '756108')
HOLD_PREFIX = '750cec'


def main():
    source = read(REVIEWED)['rows']; notes = read(NOTES); current = read(CURRENT)
    inventory = {r['parent_id']: r for r in csv.DictReader((OUT/'parent_inventory560.csv').open(encoding='utf-8'))}
    assert len(source) == 16 and all(r['actually_viewed'] for r in source)
    rows=[]
    for r in sorted(source, key=lambda x:x['parent_id']):
        uid=r['parent_id']; prefix=r['parent_prefix']
        assert r['source_finite_pass'] and r['illegal_pairs']==r['unresolved_pairs']==0
        assert sha(r['clean_case_path'])==r['clean_case_sha256']
        assert inventory[uid]['candidate_adaptation_unseen_parent']=='True', uid
        role=('HOLD_SOURCE_COMPLETENESS' if prefix==HOLD_PREFIX else
              'ADAPTATION_UNSEEN_PARENT_STAGING' if prefix in HELDOUT_PREFIXES else 'TRAIN_STAGING')
        if role!='HOLD_SOURCE_COMPLETENESS': assert r['source_complete']=='PLAUSIBLE'
        rows.append(dict(parent_id=uid,parent_prefix=prefix,split_role=role,
            visual_family_id=r['family'],family_group_authority='INHERITED_ACTUAL_AI_PROXY_VISUAL_REVIEW_NOT_VERIFIED_SOURCE_GENEALOGY',
            family_independence_status='PROVISIONAL_VISUAL_GROUPING_NEAR_DUPLICATES_NOT_EXHAUSTIVELY_EXCLUDED',
            source_structure=r['source_structure'],structural_increment=r['structural_increment'],
            source_complete=r['source_complete'],source_case_path=r['clean_case_path'],source_case_sha256=r['clean_case_sha256'],
            source_geometry_hash=r['exact_clean_geometry_sha256'],source_N=r['N_clean'],
            official_split=r['official_split'],official_checkpoint_actual_exposure='UNKNOWN_NO_PER_PARENT_LOG',
            known_adaptation_exposure='NOT_IN_AUDITED_96_PARENT_MANIFEST_UNION',
            new_training_authorized=False,training_ready=False,adapter_status='BLOCKED_N_OUTSIDE_128_256',
            review_image_path=r['review_image_path'],review_image_sha256=r['review_image_sha256']))
    train=[r for r in rows if r['split_role']=='TRAIN_STAGING']; test=[r for r in rows if r['split_role']=='ADAPTATION_UNSEEN_PARENT_STAGING']
    hold=[r for r in rows if r['split_role']=='HOLD_SOURCE_COMPLETENESS']
    assert (len(train),len(test),len(hold))==(13,2,1)
    assert not {r['visual_family_id'] for r in train}&{r['visual_family_id'] for r in test}
    old=parent_set(current['train']); assert len(old)==26
    split=dict(schema='BENCH_DEPLOYMENT_STAGING_PARENT_FAMILY_SPLIT_V1',status='FROZEN_STAGING_NOT_TRAINING',
        freeze_scope='Parent/family roles before any new source-resolution C/Y task construction; not a training or evaluation execution registration.',
        training_ready=False,training_authorized=False,new_model_results_considered=False,new_noise_registered=False,
        source_target_policy='Use authenticated clean source resolution without new geometry mutation. N outside128..256 remains blocked.',
        holdout_selection_reason='Reserve two distinct plausible structural relations with no explicit old-pool near-style caution: padded seat/base frame/round legs and full grid-back/arm/leg organization. No generated mesh result was consulted.',
        conservative_grouping_policy='All ornamental-end-frame variants 1d/4eb/d7 remain TRAIN; rectangular-end-frame variants945/ddd/f540 remain TRAIN. bcb,585 and27 stay TRAIN because source review records possible old-pool family/structure overlap.',
        family_scope_limit='Family labels are inherited AI_PROXY visual groups, not certified source genealogy or exhaustive shape-family independence. All15 plausible sources are official TRAIN partition; actual checkpoint exposure remains UNKNOWN.',
        current_reused_training_parent_ids=sorted(old),current_reused_training_condition_ids=sorted(r['condition_id'] for r in current['train']),
        current_reused_training_manifest=str(CURRENT),current_reused_training_manifest_sha256=sha(CURRENT),
        new_training_parent_ids=[r['parent_id'] for r in train],
        adaptation_unseen_parent_ids=[r['parent_id'] for r in test],
        held_parent_ids=[r['parent_id'] for r in hold],
        counts=dict(existing_training_parents=26,existing_training_conditions=91,new_train_staging_parents=13,
                    new_eval_staging_parents=2,new_eval_visual_families=2,new_plausible_source_parents=15,held_source_parents=1,
                    baseline_N128_256_admissible_new_parents=0),
        rows=rows,source_review_path=str(REVIEWED),source_review_sha256=sha(REVIEWED),
        source_family_notes_path=str(NOTES),source_family_notes_sha256=sha(NOTES),
        original_split_audit_path=str(OUT/'summary.json'),original_split_audit_sha256=sha(OUT/'summary.json'),
        immutable_after_condition_construction=True)
    save(OUT/'new_data_split.json',split)
    contract=dict(schema='BENCH_DEPLOYMENT_STAGED_EVALUATION_CONTRACT_V1',status='FROZEN_STAGING_NOT_TRAINING',
        parent_family_split_path=str(OUT/'new_data_split.json'),parent_family_split_sha256=sha(OUT/'new_data_split.json'),
        source_resolution_task_root=str(BASE/'dataset/source_tasks_pending_adapter'),
        training_ready=False,training_authorized=False,evaluation_execution_authorized=False,
        initialization_checkpoint=None,training_budget=None,noise_files=None,request_count=None,
        active_baseline_N_contract=[128,256],all_new_source_tasks_adapter_status='BLOCKED_N_OUTSIDE_128_256',
        source_geometry_mutation_allowed=False,
        existing_training=dict(parent_count=26,condition_count=91,manifest=str(CURRENT),manifest_sha256=sha(CURRENT),
            use='Retained training inputs and potential TRAIN_KNOWN_C reference after future training registration; no new same-parent holdout identity is invented for these already-exposed91C.'),
        recipe_order=[
            dict(recipe_id='R1',intended_group_for_new_TRAIN_parent='TRAIN_KNOWN_C',training_pool_membership=True,
                 geometry='Complete seat surface, seat board or curved seat shell; exact coherent region chosen from authenticated source before writing C/Y.'),
            dict(recipe_id='R2',intended_group_for_new_TRAIN_parent='SAME_PARENT_HELDOUT_C',training_pool_membership=False,
                 geometry='Complete end frame, pedestal, leg or back region distinct from R1; real whole component or complete meaningful surface, not random faces.'),
            dict(recipe_id='R3',intended_group_for_new_TRAIN_parent='TRAIN_KNOWN_C',training_pool_membership=True,optional=True,
                 geometry='Meaningful union of R1 with another whole component; omit if the union would contain all R2 faces. Do not invent a substitute or duplicate R1.')],
        heldout_parent_rule='All constructed C on617fa8 and756108 belong only to ADAPTATION_UNSEEN_PARENT_STAGING; none enters training or same-parent-heldout groups.',
        condition_selection_freeze='For each source, freeze selection.json with exact source face IDs, recipe IDs, intended groups and split hash before materializing C/Y. Model results play no role.',
        heldout_C_checks=['R2 excluded from training manifest and training exposure schedule',
            'R2 C geometry/topology distinct from every training C under vertex/face reorder invariance',
            'R3 omitted if it contains all R2 source face IDs',
            'No reclassification of existing historical91C as heldout',
            'Source full target still used in training for that same parent; this intentionally is condition transfer, not unseen parent or unseen target geometry'],
        evaluation_groups=dict(
            TRAIN_KNOWN_C='Potential future evaluation of actual trained C on current or staged training parents; no samples/noise registered now.',
            SAME_PARENT_HELDOUT_C='Reserved R2 on new TRAIN_STAGING parents only; source complete M is shared with training, C presentation is reserved.',
            ADAPTATION_UNSEEN_PARENT='Both heldout visual families remain out of current and future staging training; known historical96-parent adaptation union absent. Future chosen initialization ancestry must be rechecked.',
            REGRESSION='Existing48 request panel and old39DEV conditions retain historical identity and fixed inputs; not relabeled as new test.'),
        provenance_limits=['Two heldout groups are provisional AI_PROXY visual families, not certified genealogically independent families.',
            'All15 sources belong to official train partition; actual official checkpoint exposure is UNKNOWN. No claim of completely unseen pretraining data.',
            'Exact source geometry hashes are retained; near-duplicate equivalence has not been exhaustively established.',
            'No new N128..256 parent has passed this stage; source-resolution C/Y construction does not imply baseline-ready training data.'],
        required_actual_task_fields=['condition_id','parent_id','family_id','recipe_id','intended_eval_group','training_pool_membership','selection_path','selection_sha256','parent_family_split_sha256','source_case_path','source_case_sha256','source_face_ids','N','K','input_path','input_sha256','paired_path','paired_sha256','C_geometry_sha256','C_topology_sha256','partition_exact_roundtrip','training_ready','adapter_status','review_status'],
        target_usage='Source completeM/Y are supervision and validity evidence only; they are never the unique required output style or deployment input.',
        missing_condition_rule='Unknown meaning, incomplete region, non-distinctR2 or invalidC/Y staysHOLD/NA with reason; do not fill quotas.',
        upstream_no_model_or_noise=True)
    save(OUT/'staged_eval_contract.json',contract)
    print(json.dumps(dict(status=split['status'],path=str(OUT/'new_data_split.json'),sha256=sha(OUT/'new_data_split.json'),counts=split['counts']),ensure_ascii=False))


if __name__=='__main__': main()
