"""Freeze the authorized 32-vs-67 roles, then reuse immutable qualified assets.

No source reconstruction, geometry scan, image review, model or random-noise call.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

ROOT = Path('/mnt/e/MeshFlow')
OLD = ROOT / 'experiments/r2_user_data_scaling_12h_v1'
OUT = ROOT / 'experiments/r2_user_data_scaling_32_67_v1'
TASK = 'R2_USER_DATA_SCALING_32_67_V1'
BIG = 'USER_EXPANDED_67'


def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()


def write_bytes(p, value):
    p = Path(p)
    if not p.resolve().is_relative_to(OUT.resolve()) or p.is_symlink():
        raise ValueError('New experiment output only')
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.read_bytes() != value: raise FileExistsError(str(p))
        return
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    with tmp.open('xb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
    if p.exists(): raise FileExistsError(str(p))
    os.replace(tmp, p)


def save(p, value):
    write_bytes(p, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8'))


def csv_save(p, rows):
    s = io.StringIO(newline=''); keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(s, keys); w.writeheader()
    for r in rows: w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    write_bytes(p, ('\ufeff'+s.getvalue()).encode('utf-8'))


def coverage(label, parents, tasks, training=True):
    by_uid = defaultdict(list)
    for t in tasks: by_uid[t['uid']].append(t)
    assert set(by_uid) == set(parents)
    m = len(parents); rows = []; parent_rows = []
    for uid in sorted(parents):
        opts = by_uid[uid]; p = parents[uid]
        tags = sorted({str(tag) for t in opts for tag in (t.get('structure_tags') or [])}) or ['UNKNOWN']
        parent_rows.append(dict(pool=label, uid=uid, group_uid=p['group_uid'], N=p['N'],
            conditions=len(opts), parent_probability=1/m, structure_tags=tags,
            expected_exposure_step4000_batch8=32000/m if training else None,
            expected_exposure_step6000_batch8=48000/m if training else None))
        for t in sorted(opts, key=lambda r:r['task_id']):
            prob = 1/(m*len(opts))
            rows.append(dict(pool=label, uid=uid, condition_id=t['task_id'], condition_type=t['condition_type'],
                N=t['N'], K=t['K'], free_faces=t['N']-t['K'], free_scalars=9*(t['N']-t['K']),
                K_over_N=t['K']/t['N'], area_C_over_M=t.get('area_C_over_M'),
                source_quality=t.get('source_quality', 'UNKNOWN'),
                FF_illegal=t['CC_CF_FF']['FF']['illegal'], FF_unknown=t['CC_CF_FF']['FF']['unknown'],
                parent_condition_count=len(opts), probability=prob,
                expected_exposure_step4000_batch8=32000*prob if training else None,
                expected_exposure_step6000_batch8=48000*prob if training else None,
                structure_tags=t.get('structure_tags') or ['UNKNOWN']))
    weights = [r['probability'] for r in rows]
    assert abs(sum(weights)-1) < 1e-12
    type_rows = Counter(r['condition_type'] for r in rows)
    type_exposure = defaultdict(float); risk_exposure = defaultdict(float)
    for r in rows:
        type_exposure[r['condition_type']] += r['probability']
        risk_exposure[r['source_quality']] += r['probability']
    structure = Counter(tag for r in parent_rows for tag in r['structure_tags'])
    numeric = {}
    for key in ('N','K','free_faces','free_scalars','K_over_N','area_C_over_M'):
        usable = [r for r in rows if r[key] is not None]
        denom = sum(r['probability'] for r in usable)
        numeric[key] = dict(min=min(r[key] for r in usable), max=max(r[key] for r in usable),
            condition_row_mean=sum(r[key] for r in usable)/len(usable),
            parent_uniform_condition_uniform_mean=sum(r[key]*r['probability'] for r in usable)/denom,
            known_probability=denom)
    result = dict(parent_count=m, condition_count=len(rows), conditions_per_parent=dict(Counter(len(v) for v in by_uid.values())),
        condition_row_counts=dict(type_rows), condition_row_fraction={k:v/len(rows) for k,v in type_rows.items()},
        expected_condition_type_exposure_fraction=dict(type_exposure),
        source_risk_row_counts=dict(Counter(r['source_quality'] for r in rows)),
        expected_source_risk_exposure_fraction=dict(risk_exposure),
        parent_uniform_structure_fraction={k:v/m for k,v in structure.items()},
        structure_parent_counts=dict(structure), structure_policy='Inherited literal known tags; parent union; missing=UNKNOWN; overlapping labels need not sum to one. No semantic reannotation.',
        numeric=numeric, training_exposure_projection=training,
        expected_per_parent_exposures={str(step):8*step/m for step in (4000,6000)} if training else None,
        expected_total_free_scalars={str(step):8*step*numeric['free_scalars']['parent_uniform_condition_uniform_mean'] for step in (4000,6000)} if training else None)
    return result, rows, parent_rows


def run():
    tick = time.monotonic()
    inherited_files = {str(OLD / p): sha(OLD / p) for p in (
        'data/train_manifest.json', 'data/panel_manifest.json', 'data/data_freeze.json',
        'data/qualified_condition_manifest.json', 'data/verification_data.json',
        'data/verification_scope_note.json', 'inventory/historical_exclusions.json')}
    train0 = read(OLD / 'data/train_manifest.json'); panel0 = read(OLD / 'data/panel_manifest.json')
    previous = read(OLD / 'data/data_freeze.json'); excl = read(OLD / 'inventory/historical_exclusions.json')
    assert previous['status'] == 'DATA_SCALE_BLOCKED' and train0['status'] == 'DATA_SCALE_BLOCKED'
    small, large = train0['pools']['USER_SMALL'], train0['pools']['USER_LARGE_67']
    tasks = train0['tasks']; dev_tasks = panel0['conditions']; parents = train0['parents']
    dev = {p['uid']:p for p in panel0['dev_parents']}; all_parents = dict(parents, **dev)
    assert len(small) == len(set(small)) == 32 and len(large) == len(set(large)) == 67
    assert set(small) < set(large) == set(parents)
    assert len(tasks) == 194 and sum(t['uid'] in small for t in tasks.values()) == 91
    assert len(dev) == 32 and len(dev_tasks) == 64
    assert len(all_parents) == 99 and not set(parents) & set(dev)
    train_groups = {p['group_uid'] for p in parents.values()}; dev_groups = {p['group_uid'] for p in dev.values()}
    assert len(train_groups) == 67 and len(dev_groups) == 32 and not train_groups & dev_groups
    assert not dev_groups & set(excl['new_dev_excluded_groups'])
    assert not (train_groups | dev_groups) & (set(excl['old_development_groups']) | set(excl['sealed_historical_test_groups']))
    assert all(p['official_split']=='train' and 128<=p['N']<=256 for p in all_parents.values())
    assert all(p['new_dev_eligible'] for p in dev.values())

    # This explicit new-protocol role freeze MUST precede new formal manifests.
    role_base = dict(schema='r2_user_scaling_32_67_parent_roles_v1', experiment=TASK, status='FROZEN',
        role_policy='Existing qualified input-subset roles are the explicitly authorized starting point of this new experiment; no parent is moved or reselected.',
        earlier_order_deviation='Original run reviewed/derived C before final parent role assignment; preserved, not retroactively repaired. No model output was used. New protocol accepts these existing roles before training.',
        applicability='Qualified USER input subset, not unscreened official Chair population; official pretrain seen UNKNOWN.',
        pools={'USER_SMALL':small, BIG:large}, DEV=list(dev),
        rows=[dict(uid=uid, group_uid=p['group_uid'], variant_uid=p['variant_uid'], N=p['N'],
                   role='TRAIN' if uid in parents else 'DEV', in_small=uid in small,
                   npz=p['npz'], sha256=p['sha256']) for uid,p in sorted(all_parents.items())],
        inherited_manifest_hashes=inherited_files)
    role_path = OUT / 'data/parent_roles.json'
    if role_path.exists():
        roles = read(role_path)
        assert {k:v for k,v in roles.items() if k!='created_utc'} == role_base
    else:
        roles = dict(role_base, created_utc=now()); save(role_path, roles)
    role_hash = sha(role_path)

    # Bounded checks hash only small prepared parent/condition NPZ assets.
    asset_rows = []; unique_assets = {}
    for kind, values in [('parent', all_parents.values()), ('condition', list(tasks.values())+dev_tasks)]:
        for r in values:
            path = r['npz']; expected = r['sha256']
            if path in unique_assets:
                assert unique_assets[path] == expected; continue
            actual = sha(path); assert actual == expected
            unique_assets[path] = expected
            asset_rows.append(dict(kind=kind, path=path, sha256=actual, bytes=Path(path).stat().st_size))
    combined = list(tasks.values())+dev_tasks
    assert len({t['task_id'] for t in combined}) == 258
    assert Counter(t['uid'] for t in dev_tasks) == Counter({u:2 for u in dev})
    for t in combined:
        p = all_parents[t['uid']]; ids=t['source_face_ids']; free=t['free_source_ids']
        assert p['N']==t['N'] and p['variant_uid']==t['variant_uid'] and p['group_uid']==t['group_uid']
        assert 0 < len(ids)==t['K'] < p['N'] and len(set(ids))==len(ids)
        assert not set(ids)&set(free) and sorted(ids+free)==list(range(p['N']))
        assert t['condition_source']==t['npz'] and t['condition_source_sha256']==t['sha256']
        assert all(t['CC_CF_FF'][k]['illegal']==t['CC_CF_FF'][k]['unknown']==0 for k in ('CC','CF'))
    inheritance = dict(schema='r2_user_scaling_32_67_readonly_inheritance_v1', experiment=TASK,
        parent_roles_sha256=role_hash, source_manifest_hashes=inherited_files,
        prepared_asset_hash_checks=asset_rows, bytes_hashed=sum(r['bytes'] for r in asset_rows),
        weight_files_hashed=0, source_reconstruction_calls=0, geometry_scans=0, new_reviews=0,
        statement='All NPZ remain in the old read-only run. Only new metadata points to exact existing arrays; no NPZ copied or rewritten.')
    save(OUT / 'data/raw_inheritance_hashes.json', inheritance)

    train = dict(train0, experiment=TASK, status='FROZEN', data_status='DATA_READY32v67',
        pools={'USER_SMALL':small, BIG:large}, no_scientific_run=False,
        parent_roles=dict(path=str(role_path),sha256=role_hash),
        inherited_source_manifest=dict(path=str(OLD/'data/train_manifest.json'),sha256=inherited_files[str(OLD/'data/train_manifest.json')]))
    panel = dict(panel0, experiment=TASK, status='FROZEN', data_status='DATA_READY32v67',
        parent_roles=dict(path=str(role_path),sha256=role_hash),
        gaussian_status='NOT_MATERIALIZED_IN_DATA_REUSE_STAGE',
        inherited_source_manifest=dict(path=str(OLD/'data/panel_manifest.json'),sha256=inherited_files[str(OLD/'data/panel_manifest.json')]))
    assert train['parents']==train0['parents'] and train['tasks']==train0['tasks']
    assert panel['conditions']==panel0['conditions'] and panel['dev_parents']==panel0['dev_parents']
    save(OUT/'data/train_manifest.json',train); save(OUT/'data/panel_manifest.json',panel)
    summary={}; all_rows=[]; parent_rows=[]
    for label, ps, ts, training in [
        ('USER_SMALL',{u:parents[u] for u in small},[t for t in tasks.values() if t['uid'] in small],True),
        (BIG,parents,list(tasks.values()),True),('DEV',dev,dev_tasks,False)]:
        stats,rr,pp=coverage(label,ps,ts,training);summary[label]=stats;all_rows+=rr;parent_rows+=pp
    save(OUT/'data/coverage_summary.json',dict(experiment=TASK,pools=summary,
        policy='Parent-uniform, condition-uniform-within-parent expected exposure. 4000/6000 projections are both descriptive; common training budget must be locked by the controller after authorized preflight.',
        actual_exposure='To be recorded by training; these values are expectations, not executed updates.'))
    csv_save(OUT/'data/coverage_per_condition.csv',all_rows); csv_save(OUT/'data/coverage_per_parent.csv',parent_rows)
    assert inherited_files=={p:sha(p) for p in inherited_files}
    freeze=dict(schema='r2_user_scaling_data_freeze_v1',experiment=TASK,status='FROZEN',data_status='DATA_READY32v67',
        train_manifest=dict(path=str(OUT/'data/train_manifest.json'),sha256=sha(OUT/'data/train_manifest.json')),
        panel_manifest=dict(path=str(OUT/'data/panel_manifest.json'),sha256=sha(OUT/'data/panel_manifest.json')),
        parent_roles=dict(path=str(role_path),sha256=role_hash),large_label=BIG,
        train_parent_count=67,small_parent_count=32,dev_parent_count=32,train_conditions=194,small_conditions=91,dev_conditions=64,
        qualified_total_parents=99,source_qualified_total_conditions=286,active_unique_conditions=258,
        earlier_run_status='DATA_SCALE_BLOCKED_RETAINED_UNCHANGED',
        protocol='New authorized 32-vs-67 comparison; original >=128/256 experiment is not claimed completed.',
        qualification_timing='Existing input-qualified parent roles explicitly accepted and frozen before this new protocol; previous preparation order deviation remains disclosed.',
        no_training_no_new_generation_yet=True,scientific_endpoint=None)
    save(OUT/'data/data_freeze.json',freeze)
    receipt_path=OUT/'data/verification_data.json'
    if not receipt_path.exists():
        save(receipt_path,dict(schema='r2_user_scaling_32_67_reuse_verification_v1',experiment=TASK,status='PASS',
            created_utc=now(),wall_seconds=time.monotonic()-tick,code_sha256=sha(__file__),
            role_freeze_precedes_new_manifest=True,parent_roles_sha256=role_hash,
            train_dev_known_groups_disjoint=True,DEV_history_exclusions_pass=True,small_exact_nested=True,
            train_parents=67,small_parents=32,dev_parents=32,train_conditions=194,small_conditions=91,dev_conditions=64,
            parent_and_condition_rows_exactly_inherited=True,prepared_parent_files_checked=99,prepared_condition_files_checked=258,
            existing_sourceC_winding_verification_inherited=inherited_files[str(OLD/'data/verification_data.json')],
            metadata_CY_partition_all_pass=True,CC_CF_recorded_illegal_unknown_zero=True,
            all_old_manifests_unchanged=True,hash_scope='Only metadata and small prepared parent/condition NPZ; no weights.',
            bytes_hashed=inheritance['bytes_hashed'],model_calls=0,OT_calls=0,noise_arrays=0,
            geometry_scans=0,source_reconstructions=0,new_reviews=0,
            data_freeze_sha256=sha(OUT/'data/data_freeze.json')))
    print(json.dumps(dict(status='DATA_READY32v67',output=str(OUT/'data'),parent_roles_sha256=role_hash,
        data_freeze_sha256=sha(OUT/'data/data_freeze.json'),bytes_hashed=inheritance['bytes_hashed'],
        train_parent_count=67,small_parent_count=32,dev_parent_count=32,train_conditions=194,small_conditions=91,dev_conditions=64)))


if __name__=='__main__':run()
