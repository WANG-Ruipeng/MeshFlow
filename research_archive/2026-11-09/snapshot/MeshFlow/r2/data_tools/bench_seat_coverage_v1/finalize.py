"""Freeze the root-reviewed complete-seat scope; CPU and immutable files only."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from r2.data_tools.bench_user_v2 import build as b
from r2.data_tools.bench_seat_coverage_v1.inspect import ROOT, OUT

DECISIONS = {
    'a8': ('TRAIN', 'ACCEPT_COMPLETE_SEAT', 'Complete regular set of seat slats and perimeter rails. Retain source nonconforming boundary contacts; do not claim manifold C.'),
    '27': ('TRAIN', 'ACCEPT_COMPLETE_SEAT', 'Complete curved seating bowl, with the selected upper surface ending at its bend into the back.'),
    'df': ('NEW_PARENT_HOLDOUT', 'ACCEPT_COMPLETE_SEAT', 'Complete seating surfaces on both sides of a central back divider. Hold out from this experiment; retain the nonconforming boundary contact.'),
    '3e': ('HOLD', 'HOLD_IRREGULAR_NOTCH', 'Irregular central notch and boundary make this unsuitable for this round of regular full-seat inputs. No repair or difficult-face removal.'),
    'd9': ('HOLD', 'HOLD_PARTIAL_SEAT_BAND', 'The selected narrow seat band is only a partial region, not the full-seat task for this experiment.'),
}

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def ref(path):
    return {'path': str(path), 'sha256': b.sha(path)}

def main():
    oldpath = ROOT/'experiments/r2_bench_full_seat_curriculum_v1/data/manifest.json'
    oldeval = ROOT/'experiments/r2_bench_full_seat_curriculum_v1/eval_manifest.json'
    old, cand = b.read(oldpath), b.read(OUT/'candidate_manifest.json')
    validation = b.read(OUT/'validation/summary.json')
    links = {r['condition_id']: r for r in b.read(OUT/'vertex_link_audit.json')}
    gallery = {r['condition_id']: r for r in b.read(OUT/'gallery_manifest.json')}
    assert len(old['train']) == 22
    assert all(r['status'] == 'PASS' for r in validation['conditions'])
    policy = dict(
        version='BENCH_COMPLETE_SEAT_NATIVE_INPUT_V1',
        target='Preserved finite triangles forming a meaningful complete seating surface; C/Y are an exact partition of one unchanged target version.',
        N_range=[128, 256], K_min=2,
        finite_C_illegal_pairs_allowed=0, unresolved_C_pairs_allowed=0,
        degenerate_C_faces_allowed=0, exact_duplicate_C_groups_allowed=0,
        C_manifold_required=False,
        permitted_source_limitation='Original nonconforming T-junction / partial collinear boundary contacts may remain after actual whole-seat review; no topology-clean or shared-index connectivity certification.',
        original_quarantine_records_retained=True,
        old_head_prediction_is_admission_gate=False,
        old_assets_modified=False,
        SEAM_compatibility_claim=False,
        source_labels_available_to_model=False,
    )
    reviewed, newtrain, newdev, held = [], [], [], []
    for source in cand['conditions']:
        row = copy.deepcopy(source)
        cid, uid = row['condition_id'], row['parent_id']
        split, decision, notes = DECISIONS[uid[:2]]
        g = gallery[cid]
        assert b.sha(g['image_path']) == g['image_sha256']
        assert not row['old_DEV_regression'] and not row['current_course22_parent']
        geometry = links[cid]
        assert all(geometry[k] == 0 for k in ('C_illegal_finite_pairs', 'C_unresolved_pairs', 'C_degenerate', 'C_exact_duplicate_groups', 'C_nonmanifold_edge_count'))
        rv = dict(condition_id=cid, parent_id=uid, review_kind='AI_PROXY', reviewer='AI_PROXY_root',
                  actually_viewed=True, view_tool='functions.view_image', nonblind=True,
                  evidence_source='Root explicitly confirmed actual viewing of all five generated M/C/Y three-view pages and supplied these decisions in collaboration messages.',
                  image_path=g['image_path'], image_sha256=g['image_sha256'],
                  viewed_columns=g['columns'], viewed_views=g['views'],
                  decision=decision, split=split, observations=notes,
                  C_vertex_link_failures=geometry['C_vertex_link_failures'],
                  old_geometry_status=row['geometry_status'], old_geometry_reasons=row['geometry_reasons'])
        reviewed.append(rv)
        row.update(group=split, replay_group='NEW', review_status=decision,
                   eligibility='ACCEPTED_NATIVE_COMPLETE_SEAT' if split != 'HOLD' else 'HOLD',
                   coverage_native_admission=rv,
                   coverage_native_policy_version=policy['version'],
                   history_scope=dict(official_partition=row['official_split'],
                       official_checkpoint_actual_exposure=row['official_pretraining_exposure'],
                       COMMON12000_seen=row['COMMON12000_exposed'],
                       current_Native_course22_seen=row['current_course22_parent'],
                       historical_RELATION26_seen=row['historical_RELATION26_exposed'],
                       historical_NEWDATA27_seen=row['historical_NEWDATA27_exposed'],
                       completely_new_project_parent=False,
                       claim='Recovered/new complete-seat task coverage; current COMMON/Native lineage unseen for a8/27/df, but other historical branches used these IDs. Not pretraining-unseen generalization.'))
        if split == 'TRAIN': newtrain.append(row)
        elif split == 'NEW_PARENT_HOLDOUT': newdev.append(row)
        else: held.append(row)
    newtrain.sort(key=lambda r: ['a8', '27'].index(r['parent_id'][:2]))
    assert len(newtrain) == 2 and len(newdev) == 1 and len(held) == 2
    assert all(not r['COMMON12000_exposed'] for r in newtrain + newdev)
    reviewpath = OUT/'root_input_review.json'
    b.save(reviewpath, dict(schema='BENCH_SEAT_COVERAGE_ROOT_ACTUAL_INPUT_REVIEW_V1',
         reviewer_kind='AI_PROXY', human_review=False, records=reviewed,
         policy=policy, generated_output_review=False, decisions_before_model_sampling=True))
    train = [dict(copy.deepcopy(r), replay_group='OLD') for r in old['train']] + newtrain
    old_exact = all({k:v for k,v in r.items() if k != 'replay_group'} == source for r, source in zip(train[:22], old['train']))
    assert old_exact and len({r['parent_id'] for r in train}) == 24
    assert not ({r['parent_id'] for r in train} & {r['parent_id'] for r in newdev})
    protected = copy.deepcopy(old['protected_inputs'])
    assert isinstance(protected, dict)
    protected[str(oldpath)] = b.sha(oldpath)
    protected[str(oldeval)] = b.sha(oldeval)
    for path in (OUT/'candidate_manifest.json', OUT/'validation/summary.json', OUT/'vertex_link_audit.json', reviewpath):
        protected[str(path)] = b.sha(path)
    for r in newtrain + newdev:
        for pk, hk in [('case_path','case_sha256'),('input_path','input_sha256'),('paired_path','paired_sha256'),('quality_file','quality_sha256'),('supervision_file','supervision_sha256'),('source_target_path','source_target_sha256')]:
            assert b.sha(r[pk]) == r[hk]
            protected[r[pk]] = r[hk]
    manifestpath = OUT/'manifest.json'
    b.save(manifestpath, dict(schema='BENCH_SEAT_COVERAGE_24TRAIN_1HOLDOUT_V1',
        train=train, dev=newdev, rows=train+newdev, rejected_candidates=held,
        train_parents=[r['parent_id'] for r in train], actual_data_ready=True,
        source_manifest=str(oldpath), source_manifest_sha256=b.sha(oldpath),
        old22_row_payload_hash=digest(old['train']), old22_preserved_except_replay_group=old_exact,
        candidate_manifest=ref(OUT/'candidate_manifest.json'), actual_review=ref(reviewpath),
        policy=policy, protected_inputs=protected, protected_hashes=protected,
        summary=dict(train_parents=24, train_conditions=24, old_parents=22, new_train_parents=2,
                     new_holdout_parents=1, new_holdout_conditions=1, held_candidates=2,
                     new_train_N=[r['N'] for r in newtrain], new_train_K=[r['K'] for r in newtrain]),
        old_DEV_reference=dict(manifest=ref(oldpath), conditions=len(old['dev']), purpose='Retained historical regression; not relabeled as new independent test.'),
        split_policy='Root froze a8/27 TRAIN and df6 HOLDOUT after input-only review; no generated result used. Distinct UIDs are not proof of independent shape families.',
        model_input_policy='Production Native C,N and reversible C identity metadata only; source Y and topology labels reserved for matching/supervision.',
        source_target_role='One plausible completion, not unique ground-truth success criterion.',
        training_budget_defined=False, training_updates=0, model_forwards=0))
    oldrequests = b.read(oldeval)['requests']
    assert len(oldrequests) == 24
    selected=[]
    for r in newtrain+newdev:
        for ni in range(2):
            selected.append(dict(condition_id=r['condition_id'], parent_id=r['parent_id'],
                request_id=f"{r['condition_id']}__n{ni}", noise_index=ni, N=r['N'], K=r['K'],
                eval_group='NEW_TRAIN' if r['group']=='TRAIN' else 'NEW_PARENT_HOLDOUT',
                case_path=r['case_path'], case_sha256=r['case_sha256'],
                input_path=r['input_path'], input_sha256=r['input_sha256'],
                paired_path=r['paired_path'], paired_sha256=r['paired_sha256'],
                history_scope=r['history_scope']))
    evalpath=OUT/'eval_selection.json'
    b.save(evalpath, dict(schema='BENCH_SEAT_COVERAGE_EVAL_SELECTION30_V1',
        status='CONDITION_SELECTION_FROZEN_PENDING_ROOT_NOISE_AND_QUERY_MANIFEST',
        original24_manifest=ref(oldeval), original24_request_ids=[r['request_id'] for r in oldrequests],
        original24_inputs_and_noise_policy='Reuse exact existing request C/N/noise arrays and hashes; no replacement noise.',
        new_requests=selected, new_request_count=6, total_requests_per_model=30,
        new_noise_policy='Root prepare/register creates two fixed standard Gaussian arrays per new C and shares exact arrays across all evaluated models; no evaluation noise is created by this data script.',
        validation_noise_is_not_evaluation_noise=True,
        history_scope='New means unseen by the selected COMMON12000/Native-course initialization; all three IDs are official train partition and appeared in other historical branches. df6 is excluded from this round training, not claimed unseen throughout project or official pretraining.',
        source_targets_for_display_or_supervision_only=True))
    protected_source_ok=all(b.sha(p)==h for p,h in b.read(OUT/'protected_source_hashes.json').items())
    assert protected_source_ok
    b.save(OUT/'cpu_gate.json',dict(schema='BENCH_SEAT_COVERAGE_CPU_GATE_V1', status='PASS',
        manifest=ref(manifestpath), input_review=ref(reviewpath), eval_selection=ref(evalpath),
        old22_preserved_except_replay_group=old_exact,
        old22_prior_cpu_gate=ref(oldpath.parent/'cpu_gate.json'),
        new_candidates_actual_validation=ref(OUT/'validation/summary.json'),
        prior_source_targets_hash_unchanged=protected_source_ok,
        target_reaudits_this_finalize=0, model_forwards=0, training_updates=0,
        actual_CPU_OT_total_from_build=validation['total_CPU_OT'], additional_CPU_OT_this_finalize=0,
        accepted_conditions=[r['condition_id'] for r in newtrain+newdev],
        topology_limitations={r['condition_id']:links[r['condition_id']] for r in newtrain+newdev},
        old_topology_quarantine_not_overwritten=True,
        note='PASS is this new Native finite-surface input policy, not manifold certification or SEAM deployment validation.'))
    print(json.dumps(dict(train=len(train), old=22, new=2, holdout=len(newdev), eval_requests=30,
          manifest=ref(manifestpath), cpu_gate=ref(OUT/'cpu_gate.json')), indent=2))

if __name__=='__main__':main()
