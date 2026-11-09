"""Independent CPU-only checks of emitted scaling data; no geometry rescans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_user_data_scaling_12h_v1'
sys.path.insert(0, str(ROOT / 'r2/src'))
from meshflow_control.source_winding import validate_source


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def run():
    paths = {key: OUT / path for key, path in {
        'freeze': 'data/data_freeze.json', 'train': 'data/train_manifest.json',
        'panel': 'data/panel_manifest.json', 'qualified': 'data/qualified_condition_manifest.json',
        'eligibility': 'data/condition_eligibility.json',
        'exclusions': 'inventory/historical_exclusions.json',
        'candidate_rules': 'data/candidate_selection_rules.json',
    }.items()}
    before = {key: sha(p) for key, p in paths.items()}
    docs = {key: read(p) for key, p in paths.items()}
    freeze, train, panel, qualified, eligibility, exclusions = [docs[k] for k in
        ('freeze', 'train', 'panel', 'qualified', 'eligibility', 'exclusions')]
    parents = {p['uid']: p for p in qualified['parents']}
    tasks = qualified['conditions']
    assert len(parents) == len(qualified['parents'])
    assert len({p['group_uid'] for p in parents.values()}) == len(parents)
    assert all(128 <= p['N'] <= 256 and p['official_split'] == 'train' for p in parents.values())
    dev_ids = {p['uid'] for p in panel['dev_parents']}
    large_label = freeze['large_label']
    train_ids, small_ids = set(train['pools'][large_label]), set(train['pools']['USER_SMALL'])
    assert small_ids <= train_ids and len(train_ids) == len(train['pools'][large_label])
    assert len(small_ids) == len(train['pools']['USER_SMALL'])
    assert not dev_ids & train_ids and train_ids | dev_ids == set(parents)
    dev_groups = {parents[u]['group_uid'] for u in dev_ids}
    train_groups = {parents[u]['group_uid'] for u in train_ids}
    assert not dev_groups & train_groups
    assert not dev_groups & set(exclusions['new_dev_excluded_groups'])
    assert not (dev_groups | train_groups) & (set(exclusions['old_development_groups']) | set(exclusions['sealed_historical_test_groups']))
    assert all(parents[u]['new_dev_eligible'] for u in dev_ids)
    assert set(train['parents']) == train_ids
    assert {t['uid'] for t in train['tasks'].values()} == train_ids
    assert {t['uid'] for t in panel['conditions']} == dev_ids
    for uid in dev_ids:
        picked = [t for t in panel['conditions'] if t['uid'] == uid]
        assert len(picked) == 2 and len({tuple(t['source_face_ids']) for t in picked}) == 2
    assert len({t['task_id'] for t in tasks}) == len(tasks)
    assert len({(t['uid'], tuple(t['source_face_ids'])) for t in tasks}) == len(tasks)

    accepted = {'ACCEPT', 'ACCEPT_AI_PROXY', 'ACCEPTED', 'ACCEPTED_AI_PROXY', 'ACCEPTED_WITH_RISK_AI_PROXY'}
    options = {(r['uid'], r['variant_uid'], r['kind']+'_'+r['tier'], tuple(sorted(r['ids']))): r
               for r in eligibility['rows'] if r['eligible']}
    geometry = {}
    for uid, p in parents.items():
        assert sha(p['npz']) == p['sha256']
        qp = OUT / 'data/quality' / (uid+'__'+p['variant_uid']+'.json')
        q = read(qp)
        assert q['status'] == 'COMPLETE' and q['independent_verification']
        assert q['input_npz_sha256'] == p['sha256']
        assert not q['basic_geometry']['nonfinite_source_face_ids']
        assert not q['basic_geometry']['exact_zero_area_source_face_ids']
        bad = set()
        for pair in q['pair_contacts']:
            if pair['category'] in ('proper', 'coplanar_positive_overlap', 'UNKNOWN'):
                bad.update(pair['pair'])
        geometry[uid] = (q, bad, sha(qp))
    ff_risk = 0
    for t in tasks:
        p = parents[t['uid']]
        assert p['variant_uid'] == t['variant_uid']
        ids = t['source_face_ids']; free = t['free_source_ids']; n = p['N']
        assert len(ids) == t['K'] and len(set(ids)) == len(ids)
        assert 0 < len(ids) < n and not set(ids) & set(free)
        assert sorted(ids+free) == list(range(n))
        assert set(ids) <= set(t['region_source_face_ids'])
        q, bad, qhash = geometry[t['uid']]
        assert t['quality_sha256'] == qhash
        assert not set(ids) & bad
        option = options[t['uid'], t['variant_uid'], t['condition_type'], tuple(sorted(ids))]
        assert option['review_status'] in accepted and not option['eligibility_reasons']
        for key in ('CC', 'CF'):
            assert t['CC_CF_FF'][key]['illegal'] == t['CC_CF_FF'][key]['unknown'] == 0
        ff_risk += int(t['CC_CF_FF']['FF']['illegal'] > 0 or t['CC_CF_FF']['FF']['unknown'] > 0)

    # Stable selection uses identifiers only, no random stream and no model outputs.
    sample = sorted(tasks, key=lambda t: hashlib.sha256(('INDEPENDENT_DATA_CHECK_V1|'+t['task_id']).encode()).hexdigest())[:3]
    examples = []
    for t in sample:
        p = parents[t['uid']]
        assert sha(t['npz']) == t['sha256']
        with np.load(t['npz'], allow_pickle=False) as z: c = {k: z[k].copy() for k in z.files}
        with np.load(p['npz'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
        ids = np.asarray(t['source_face_ids'], dtype=np.int64)
        free = np.asarray(t['free_source_ids'], dtype=np.int64)
        full = a['full_target'].reshape(-1, 3, 3)
        assert bits(c['C'], full[ids])
        assert bits(a['pre_ot_full'] * 2, full)
        assert bits(a['model_vertices'][a['source_face_vertex_ids']], full)
        assert np.array_equal(c['source_face_ids'], ids)
        assert np.array_equal(c['free_source_ids'], free)
        assert np.array_equal(free, np.setdiff1d(np.arange(len(full)), ids))
        restored = validate_source(c, full[ids])
        original_faces = a['source_faces'][ids]
        assert bits(restored['C_original'], a['model_vertices'][original_faces])
        assert np.array_equal(restored['C_original_source_vertex_ids'], original_faces)
        assert np.array_equal(restored['C_original_source_face_ids'], a['retained_to_original_face'][ids])
        assert bits(c['C_original_raw_coordinates'], a['source_vertices'][original_faces].astype(np.float32))
        assert np.array_equal(c['C_source_corner_permutations'], a['source_corner_permutations'][ids])
        assert len(full[free]) == len(full)-len(ids)
        examples.append(dict(condition_id=t['task_id'], parent_uid=t['uid'], variant_uid=t['variant_uid'],
            N=len(full), K=len(ids), condition_sha256=t['sha256'], parent_sha256=p['sha256'],
            canonical_C_bitwise_matches_parent=True, source_C_original_corners_and_IDs_match=True,
            source_raw_coordinates_match=True, Y_exact_source_face_complement=True,
            source_restore_summary=restored['summary']))

    assert before == {key: sha(p) for key, p in paths.items()}
    assert freeze['train_manifest']['sha256'] == before['train']
    assert freeze['panel_manifest']['sha256'] == before['panel']
    gate = len(train_ids) >= 128 and len(dev_ids) == 32 and len(small_ids) == 32
    assert (freeze['status'] == 'FROZEN') == gate
    assert train['no_scientific_run'] == (not gate)
    result = dict(schema='r2_user_scaling_independent_data_verification_v1', status='PASS',
        data_gate=freeze['status'], verified_qualified_parents=len(parents), verified_conditions=len(tasks),
        train_parents=len(train_ids), small_parents=len(small_ids), dev_parents=len(dev_ids),
        train_dev_disjoint_by_uid_and_known_group=True, small_nested_in_large=True,
        new_DEV_excludes_historical_conditional_and_development_groups=True,
        full_M_nonfinite_and_zero_area_absent_in_recorded_complete_scans=True,
        selected_C_touches_no_illegal_or_UNKNOWN_pair_in_frozen_contacts=True,
        selected_conditions_with_Y_only_pair_risk_retained=ff_risk,
        semantics_and_geometry_independently_required=True,
        input_manifest_hashes=before, code_sha256=sha(__file__), examples=examples,
        review_findings='No concrete contract defect found in current emitted data. DATA_SCALE_BLOCKED is enforced; 99 qualified parents is a current reviewed-pool count, not an absolute local-data upper bound.',
        scope='All qualified metadata/splits and existing scan contact records; stable identifier-based three-condition array verification. No new geometry detector, random noise, OT, model calls, training, or generation.',
        model_calls=0, detector_calls=0, OT_calls=0, noise_arrays=0)
    dest = OUT / 'data/verification_data.json'
    with dest.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(path=str(dest), sha256=sha(dest), status=result['status'], data_gate=freeze['status'], sampled_conditions=[e['condition_id'] for e in examples], Y_only_risk_conditions=ff_risk)))


if __name__ == '__main__':
    run()
