"""Read-only local Chair inventory for R2_USER_DATA_SCALING_12H_V1.

No model imports, geometry edits, downloads, semantic acceptance, or test sampling.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path('/mnt/e/MeshFlow')
OUT = ROOT / 'experiments/r2_user_data_scaling_12h_v1/inventory'
ART = Path('/mnt/e/MeshFlow-Control-artifacts')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text('utf-8') != text:
            raise RuntimeError('Refuse overwrite: ' + str(path))
    else:
        path.write_text(text, encoding='utf-8')


def main():
    t0 = time.monotonic()
    if OUT.is_symlink():
        raise RuntimeError('Symlink output refused')
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'experiments/mf_h1_local0/assets/shapenet-03001627'
    old = ROOT / 'native_t1/runs/control_mvp48_v1/data'
    input_paths = [old / n for n in ('candidate_audit.json', 'repair_candidate_audit.json', 'splits.json', 'manifest.json')]
    input_paths += [ART / 'data/train_manifest.json', ART / 'experiments/chair_r2_user_conditions_v1/active_conditions.json', base / 'split/train.npz', base / 'split/test.npz']
    before = {str(p): sha(p) for p in input_paths}
    orig, repair, legacy_splits, legacy_manifest, r2, active = map(read, input_paths[:6])
    groups = repair['grouping']['groups']
    group_of = {uid: group for group, members in groups.items() for uid in members}
    official = {}
    # Official trusted local split is a published object-array NPZ. Geometry NPZ
    # below always uses allow_pickle=False. This is not an unknown external file.
    for name in ('train', 'test'):
        with np.load(base / 'split' / (name + '.npz'), allow_pickle=True) as z:
            for row in z['npz_list']:
                uid = str(row['uid'])
                if uid in official:
                    raise ValueError('Overlapping official split: ' + uid)
                official[uid] = name
    historical_groups = set(repair['grouping']['historically_exposed_group_ids'])
    r2_train = {r['uid'] for r in r2['tasks']}
    user_train = {r['uid'] for r in active['rows'] if r['split'] == 'train'}
    old_train = set(legacy_splits['train'])
    old_dev = set(legacy_splits['validation'])
    old_test = set(legacy_splits['test'])
    train_groups = {group_of[u] for u in r2_train | user_train | old_train}
    dev_groups = {group_of[u] for u in old_dev}
    test_groups = {group_of[u] for u in old_test}
    exclusion_groups = historical_groups | train_groups | dev_groups | test_groups
    orig_by_uid = {r['uid']: r for r in orig['objects']}
    records = []
    for uid, prior in sorted(orig_by_uid.items()):
        n = prior['N']
        if not 128 <= n <= 256:
            continue
        path = ROOT / prior['source_path']
        h = sha(path)
        if h != prior['source_sha256']:
            raise RuntimeError('Protected source changed: ' + uid)
        with np.load(path, allow_pickle=False) as z:
            v, f = z['vertices'], z['faces']
            if v.dtype != np.float32 or f.ndim != 2 or f.shape[1] != 3:
                raise ValueError('Unexpected geometry schema: ' + uid)
            if str(z['uid'].item()) != uid or len(f) != n:
                raise ValueError('Source identity mismatch: ' + uid)
            valid = np.isfinite(v).all() and np.all((f >= 0) & (f < len(v)))
            tri = v[f].astype(np.float64)
            areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
            exact_zero = int(np.sum(areas == 0))
        group = group_of[uid]
        reasons = []
        if group in historical_groups: reasons.append('PRIOR_REPEATED_DEVELOPMENT_GROUP')
        if group in train_groups: reasons.append('PRIOR_CONDITIONAL_TRAIN_GROUP')
        if group in dev_groups: reasons.append('EXISTING_DEV_GROUP')
        if group in test_groups: reasons.append('RESERVED_HISTORICAL_TEST_GROUP')
        if official[uid] != 'train': reasons.append('OFFICIAL_TEST_SEALED')
        records.append(dict(parent_uid=uid, group_uid=group, official_split=official[uid],
            variant_uid='official_raw_indexed', N=n, source_file=str(path), source_sha256=h,
            exact_geometry_sha256=prior['exact_geometry_sha256'], vertices_key='vertices', faces_key='faces',
            source_vertex_count=len(v), source_corner_order='ORIGINAL_INDEXED_FACES',
            source_vertex_identity='ORIGINAL_INDEXED_VERTEX_IDS', finite_and_valid_indices=bool(valid),
            exact_zero_area_faces=exact_zero, exact_duplicate_triangles=prior['duplicate_source_triangle_count'],
            raw_train_asset=official[uid]=='train', new_dev_candidate=not reasons,
            prior_exposure_reasons=reasons, known_condition_train=group in train_groups,
            known_development=group in (historical_groups | dev_groups),
            official_generator_pretraining_seen='UNKNOWN', official_denoiser_pretraining_seen='UNKNOWN',
            semantic_status='NOT_YET_REVIEWED', source_quality_status=('INVALID_COMPLETE_M_ZERO_AREA' if exact_zero else 'LOCAL_CC_CF_REVIEW_REQUIRED'),
            variant_rank_key=[abs(n-192), 'official_raw_indexed']))
    # Existing repaired versions are inventoried separately; the old all-corpus
    # repair recipe is not an already-materialized alternative mesh for new UIDs.
    variants = []
    for row in legacy_manifest['objects']:
        path = ROOT / 'native_t1/runs/control_mvp48_v1' / row['prepared_path']
        if not path.exists():
            raise FileNotFoundError(path)
        h = sha(path)
        if h != row['prepared_sha256']:
            raise RuntimeError('Protected existing variant changed: ' + str(path))
        variants.append(dict(parent_uid=row['uid'], group_uid=group_of[row['uid']],
            official_split=official[row['uid']], variant_uid='existing_repair_v2',
            N=row['N'], source_file=str(path), source_sha256=h,
            coordinate_status='ALREADY_R2_MODEL_SPACE',
            source_corner_order='SAVED_ORIGINAL_INDEXED_SOURCE_PLUS_CANONICAL_PERMUTATION',
            source_vertex_identity='SAVED_ORIGINAL_INDEXED_VERTEX_IDS',
            exact_geometry_sha256=row['exact_geometry_sha256'], vertices_key='model_vertices', faces_key='source_faces',
            exact_zero_area_faces=0, finite_and_valid_indices=True, original_source_file=str(ROOT / row['source_path']),
            original_source_sha256=row['source_sha256'], original_N=row['repair']['original_N'],
            lineage=row['repair'], existing_only=True, variant_rank_key=[abs(row['N']-192), 'existing_repair_v2']))
    train_rows = [r for r in records if r['official_split']=='train']
    new_rows = [r for r in train_rows if r['new_dev_candidate']]
    summary = dict(schema='r2_scaling_local_asset_inventory_v1',
        state='ASSETS_AVAILABLE_SEMANTICS_AND_LOCAL_INTERFACE_QUALIFICATION_PENDING',
        official_root=str(base), official_counts=dict(collections.Counter(official.values())),
        official_validation_present=False, official_test_used=False,
        N128_256_counts=dict(collections.Counter(r['official_split'] for r in records)),
        train_known_groups=len({r['group_uid'] for r in train_rows}),
        new_dev_unexposed_asset_candidates=len(new_rows),
        new_dev_unexposed_candidate_groups=len({r['group_uid'] for r in new_rows}),
        train_exact_zero_area_free_objects=sum(r['exact_zero_area_faces']==0 for r in train_rows),
        new_dev_exact_zero_area_free_objects=sum(r['exact_zero_area_faces']==0 for r in new_rows),
        existing_repair_v2_variants=len(variants),
        existing_part_semantics=dict(source=str(input_paths[5]), train_parents=len(user_train),
            train_conditions=sum(r['split']=='train' for r in active['rows']),
            status='HISTORICAL_AI_PROXY_ONLY_NOT_NEW_DATA_LABELS'),
        partnet_status='NOT_FOUND_IN_EXISTING_ASSET_ROOTS_OR_DOWNLOADS',
        caveats=['Counts are assets, not qualified USER regions or conditions.',
            'No validation split exists; new DEV must be held out from verified official train.',
            'Use saved known source groups; no new all-corpus near-duplicate search.',
            'Do not infer new DEV eligibility from worker completion order.',
            'Zero/duplicate faces are inventoried; local CC/CF qualification is a separate stage.',
            'Repair audit recipes do not make thousands of repaired meshes existing variants.',
            'No new normalization, repair, QEM, subdivision, source vertex merge or model call.'])
    exclusions = dict(schema='r2_scaling_historical_group_exclusions_v1',
        known_group_mapping_source=str(input_paths[1]),
        historical_groups=sorted(historical_groups),
        historical_conditional_train_uids=sorted(r2_train | user_train | old_train),
        historical_conditional_train_groups=sorted(train_groups),
        old_development_uids=sorted(old_dev), old_development_groups=sorted(dev_groups),
        sealed_historical_test_uids=sorted(old_test), sealed_historical_test_groups=sorted(test_groups),
        new_dev_excluded_groups=sorted(exclusion_groups),
        groups=groups, semantics='Known groups are conservative inherited source groups, not proven full ancestry.')
    dump(OUT/'asset_summary.json', summary)
    dump(OUT/'raw_candidates.json', dict(schema='r2_scaling_raw_asset_candidates_v1', rows=records))
    dump(OUT/'existing_variants.json', dict(schema='r2_scaling_existing_saved_variants_v1', rows=variants))
    dump(OUT/'historical_exclusions.json', exclusions)
    dump(OUT/'input_hashes.json', before)
    assert all(sha(p)==h for p,h in before.items())
    csvpath = OUT/'raw_candidates.csv'
    if not csvpath.exists():
        with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in records)
    print(json.dumps(summary,indent=2))
    print('inventory_wall_seconds', time.monotonic()-t0)


if __name__=='__main__':
    main()
