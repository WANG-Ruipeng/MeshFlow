"""CPU-only immutable registration for CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
from .common import ROOT, OUT, SOURCE, CHAIR, HYBRID, QUICK, START, START_SHA, LABELS, ARMS
from .common import read_json, digest, array_hash, json_hash, freeze_json, atomic_json, resolve
from experiments.mf_h1_local0.chair_control_usability_quick_v1.registration import (
    freeze_npz, bits, true_interface, area, components)
from experiments.mf_h1_local0.chair_domain_control_v1.prepare import load_task_sample, _topology, _api

SCHEMA = 'chair_boundary_attachment_registration_v1'
BANK_KEYS = ('z', 'C', 'known_mask', 'valid_mask', 'N', 'K')
PANELS = ('P1', 'P2', 'P3', 'P4', 'P5')


def _arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def _old_path(value):
    p = Path(str(value))
    return resolve(value) if p.is_absolute() or ':' in str(value) else CHAIR / p


def _record_path(record):
    return resolve(record['input_npz']) if 'input_npz' in record else _old_path(record['input_path'])


def _attach_topology(payload, base, ids):
    interface, edges, status, ambiguous = true_interface(base, ids, _topology(base))
    trusted_count = len(edges)
    known = {int(s): i for i, s in enumerate(ids)}
    use = interface if status == 'PASS' else {k: [] for k in interface}
    payload.update(
        gamma_edges=edges.copy() if status == 'PASS' else np.empty((0,2,3), np.float32),
        gamma_known_face_indices=np.asarray([known[int(i)] for i in use['known_source_face_ids']], np.int64),
        gamma_known_source_face_ids=np.asarray(use['known_source_face_ids'], np.int64),
        gamma_free_source_face_ids=np.asarray(use['free_source_face_ids'], np.int64),
        gamma_edge_source_vertex_ids=np.asarray(use['edge_source_vertex_ids'], np.int64).reshape(-1,2),
        gamma_known_corner_ids=np.asarray(use['known_corner_ids'], np.int64).reshape(-1,2),
        gamma_free_corner_ids=np.asarray(use['free_corner_ids'], np.int64).reshape(-1,2),
        model_vertices=np.asarray(base['model_vertices']).copy(),
        source_face_vertex_ids=np.asarray(base['source_face_vertex_ids']).copy(),
        retained_to_original_face=np.asarray(base['retained_to_original_face']).copy())
    return dict(gamma_status=status, gamma_edge_count=len(payload['gamma_edges']),
        trusted_edge_count_before_full_boundary_gate=trusted_count,
        gamma_ambiguities=ambiguous, gamma_interface=interface,
        gamma_primary_never_partial_subset=True)


def _legacy_condition(record, manifest):
    if record['panel'] == 'P4':
        base = _api(str(SOURCE)).load_sample(record['uid'], record['patch_id'], SOURCE/'data')
        L0 = float(base['bbox_diagonal'])
    else:
        base = load_task_sample(SOURCE, manifest, record['task_id'], 1.)
        L0 = float(base['bbox_L0'])
    n, k = int(base['N']), int(base['K'])
    ids = np.asarray(base['source_face_ids'], np.int64)
    free = np.asarray(base['free_source_ids'], np.int64)
    full = np.asarray(base['full_target'], np.float32).reshape(n,3,3)
    reference = _arrays(_old_path(record['ref_npz']))
    prefix = 'reference_' if record['panel'] == 'P4' else ''
    payload = dict(C=full[ids].copy(), GT_full=full.copy(), GT_free=full[free].copy(),
        source_face_ids=ids.copy(), known_source_face_ids=ids.copy(), free_source_ids=free.copy(),
        original_known_source_face_ids=np.asarray(base['retained_to_original_face'])[ids].copy(),
        original_free_source_face_ids=np.asarray(base['retained_to_original_face'])[free].copy(),
        N=np.asarray(n,np.int64), K=np.asarray(k,np.int64), L0=np.asarray(L0,np.float64),
        legacy_query_points=reference[prefix+'coverage_points'].copy(),
        legacy_pairwise_points=reference[prefix+'pairwise_target_points'].copy(),
        legacy_gamma_points=reference[prefix+'gamma_points'].copy(),
        legacy_gamma_weights=reference[prefix+'gamma_weights'].copy())
    assert bits(payload['C'], reference['C'])
    assert bits(payload['GT_free'].reshape(-1,9), reference['gt_free'].reshape(-1,9))
    info = _attach_topology(payload, base, ids)
    sizes = components(ids, _topology(base)[2])
    info.update(known_components=len(sizes), known_component_sizes=sizes,
        known_face_ratio=k/n, known_area_ratio=float(area(payload['C'])/area(full)),
        legacy_ref_npz=str(_old_path(record['ref_npz'])),
        legacy_ref_json=str(_old_path(record['ref_json'])),
        reference_audit=read_json(_old_path(record['ref_json'])),
        source_task_id=record['task_id'], source_patch_id=record['patch_id'],
        common_area=None)
    return payload, info


def _quick_condition(record, quick_conditions, manifest):
    meta = quick_conditions[record['condition_id']]
    path = resolve(meta['condition_npz'])
    assert digest(path) == meta['condition_sha256']
    payload = _arrays(path)
    base_task = record['uid']+'_A20'
    base = load_task_sample(SOURCE, manifest, base_task, 1.)
    original_gamma = payload['gamma_edges'].copy()
    info = _attach_topology(payload, base, payload['source_face_ids'])
    assert bits(original_gamma, payload['gamma_edges']), 'Old quick Gamma changed'
    assert info['gamma_status'] == meta['gamma_status']
    info.update({k:meta.get(k) for k in ('known_components','known_component_sizes',
        'known_face_ratio','known_area_ratio','legacy_ref_npz','legacy_ref_json',
        'reference_audit','source_task_id','common_area')})
    info['source_condition_npz'] = str(path)
    info['source_condition_sha256'] = digest(path)
    info['source_quick_condition_identity'] = meta['condition_identity']
    return payload, info


def _check_cache(record, bank, cache, cache_path):
    path = resolve(cache['raw_path'])
    assert cache['status'] in ('SAMPLED','CACHED'), 'No historical completed raw'
    assert cache['checkpoint_sha256'] == START_SHA, 'Wrong START checkpoint identity'
    actual_hash = digest(path)
    assert actual_hash == cache['raw_sha256'], 'Historical raw file changed'
    raw = _arrays(path)
    n, k = int(bank['N']), int(bank['K'])
    assert raw['raw'].dtype == np.float32 and raw['raw'].shape == (n,9)
    assert np.isfinite(raw['raw']).all() and bits(raw['raw'][:k],bank['C'].reshape(k,9))
    assert bits(raw['noise'],bank['z']) and bits(raw['C'].reshape(k,3,3),bank['C'])
    assert bits(raw['known_mask'].reshape(n),bank['known_mask'])
    if cache.get('raw_array_sha256'):
        assert array_hash(raw['raw']) == cache['raw_array_sha256']
    meta = cache.get('metadata') or cache.get('legacy_metadata')
    if not meta:
        raise ValueError('Missing sampling identity in '+str(cache_path))
    expected = dict(solver='original_Euler50',nfe=50,accepted_states=51,
        forward_dtype='bfloat16',integration_dtype='float32',
        known_all51_bitwise=True,known_state_checks=51,actual_Geo_recomputations=50,
        GT_model_inputs=False,geometry_repairs=0)
    for key,val in expected.items():
        assert meta.get(key) == val, (key,meta.get(key),val,str(cache_path))
    assert meta['noise_sha256'] == array_hash(bank['z'])
    assert meta['condition_sha256'] == array_hash(bank['C'].reshape(k,9))
    if 'states' in raw:
        assert raw['states'].shape == (51,n,9)
        assert all(bits(s[:k],bank['C'].reshape(k,9)) for s in raw['states'])
    return dict(cache_status='PASS',legacy_case_json=str(cache_path),
        legacy_case_sha256=digest(cache_path),legacy_raw=str(path),
        legacy_raw_sha256=actual_hash,raw_path=str(path),raw_sha256=actual_hash,
        raw_array_sha256=array_hash(raw['raw']),legacy_metadata=meta,
        historical_trajectory_present='states' in raw,
        source_generation_seconds=meta.get('seconds'), checkpoint_sha256=START_SHA)


def _selection(panels, quick_requests):
    val = sorted(set(x['uid'] for x in panels['A'] if x['split']=='validation'))
    train = sorted(set(x['uid'] for x in panels['A'] if x['split']=='train'))[:4]
    holdout = sorted(set(x['uid'] for x in panels['C']))[:4]
    records=[]
    for x in quick_requests:
        if x['arm'] != 'OT_HYBRID':
            continue
        panel = 'P1' if x['context_level']==1 else 'P5'
        y=dict(x,panel=panel,source_family='quick',source_request_id=x['request_id'],
            noise_index=int(x['candidate']),split='validation',alpha=1.)
        assert x['uid'] in val
        if panel=='P5':
            assert x['uid'] in val[:2] and x['candidate'] in (0,1) and x['context_level'] in (2,3)
        records.append(y)
    for x in panels['A']:
        panel = ('P2' if x['split']=='train' and x['uid'] in train and x['ratio']==.2
                 else 'P3' if x['split']=='validation' and x['ratio']==.4 else None)
        if panel:
            records.append(dict(x,panel=panel,source_family='hybrid',legacy_case_id=x['case_id'],
                condition_id=x['task_id'],candidate=int(x['noise_index']),
                context_level=1 if panel=='P2' else 2))
    for x in panels['C']:
        if x['uid'] in holdout:
            records.append(dict(x,panel='P4',source_family='hybrid',legacy_case_id=x['case_id'],
                condition_id=x['task_id'],candidate=int(x['noise_index']),context_level=None))
    records.sort(key=lambda x:(PANELS.index(x['panel']),x['uid'],x['condition_id'],x['candidate']))
    counts={p:sum(x['panel']==p for x in records) for p in PANELS}
    assert counts == dict(P1=32,P2=8,P3=16,P4=16,P5=8), counts
    assert len({(x['panel'],x['condition_id'],x['candidate']) for x in records}) == 80
    assert sum(x['status']=='NOT_APPLICABLE' for x in records) == 4
    return records,dict(P1=val,P2=train,P3=val,P4=holdout,P5=val[:2])


def prepare(out=OUT):
    out=Path(out); (out/'registration').mkdir(parents=True,exist_ok=True)
    source_hashes={}
    def track(path):
        p=resolve(path)
        source_hashes[str(p)]=digest(p)
        return p
    panels=read_json(track(CHAIR/'panels.json'))
    manifest=read_json(track(CHAIR/'task_manifest.json'))
    quick=read_json(track(QUICK/'requests.json'))
    quick_conditions={x['condition_id']:x for x in read_json(track(QUICK/'registration/conditions.json'))}
    hybrid_index=read_json(track(HYBRID/'evaluation/OT_HYBRID/generated_rows.json'))
    if isinstance(hybrid_index,dict):hybrid_index=hybrid_index['rows']
    hybrid_index={x['case_id']:x for x in hybrid_index}
    assert digest(START)==START_SHA
    source_hashes[str(START)]=START_SHA
    track(SOURCE/'data/manifest.json');track(SOURCE/'data/patches.json');track(SOURCE/'data/load_sample.py')
    records,parents=_selection(panels,quick)
    base_rows=[];conditions={};cache_audits=[]
    for record in records:
        cid=record['condition_id']; panel=record['panel']
        base_id=panel+'__'+cid+'__n'+str(record['candidate'])
        if record['status']=='NOT_APPLICABLE':
            reason=record.get('reason')
            condition=dict(condition_id=cid,uid=record['uid'],panel=panel,
                status='NOT_APPLICABLE',reason=reason,N=record['N'],K=record['K'],
                context_level=record['context_level'],gamma_status='NOT_APPLICABLE',
                condition_npz=None,condition_sha256=None)
            conditions.setdefault(cid,condition)
            base_rows.append(dict(base_request_id=base_id,condition_id=cid,panel=panel,
                uid=record['uid'],split=record['split'],candidate=record['candidate'],
                noise_index=record['candidate'],context_level=record['context_level'],
                N=record['N'],K=record['K'],alpha=1.,status='NOT_APPLICABLE',reason=reason,
                legacy_case_id=record['case_id'],input_npz=None,condition_npz=None))
            continue
        original_input=track(_record_path(record))
        assert digest(original_input)==record['input_sha256']
        b=_arrays(original_input);n=int(b['N']);k=len(b['C'])
        bank={name:b[name].copy() for name in ('z','C','known_mask','valid_mask','N')}
        bank['K']=np.asarray(k,np.int64)
        assert set(bank)==set(BANK_KEYS)
        assert bank['z'].shape==(1,n,9) and bank['z'].dtype==np.float32
        assert bank['C'].shape==(k,3,3) and bank['C'].dtype==np.float32
        assert bank['known_mask'].shape==(n,) and bank['valid_mask'].shape==(n,)
        assert bank['known_mask'].dtype==np.bool_ and bank['valid_mask'].dtype==np.bool_
        assert bank['valid_mask'].all() and np.array_equal(bank['known_mask'],np.arange(n)<k)
        assert np.isfinite(bank['z']).all() and np.isfinite(bank['C']).all()
        bank_path=out/'inputs/inference'/f'{base_id}.npz';freeze_npz(bank_path,bank)
        if cid not in conditions:
            if record['source_family']=='quick':
                payload,info=_quick_condition(record,quick_conditions,manifest)
            else:
                payload,info=_legacy_condition(record,manifest)
            assert bits(payload['C'],bank['C'])
            assert sorted(np.r_[payload['source_face_ids'],payload['free_source_ids']].tolist())==list(range(n))
            assert bits(payload['GT_full'][payload['source_face_ids']],bank['C'])
            cp=out/'inputs/conditions'/f'{cid}.npz';freeze_npz(cp,payload)
            identity=json_hash({name:array_hash(a) for name,a in sorted(payload.items())})
            condition=dict(condition_id=cid,uid=record['uid'],panel=panel,
                context_level=record['context_level'],status='READY',N=n,K=k,
                L0=float(payload['L0']),condition_npz=str(cp),condition_sha256=digest(cp),
                condition_identity=identity,C_sha256=array_hash(bank['C']),
                source_face_ids_sha256=array_hash(payload['source_face_ids']),**info)
            conditions[cid]=condition
            for key in ('legacy_ref_npz','legacy_ref_json','source_condition_npz'):
                if info.get(key):track(info[key])
        condition=conditions[cid]
        assert condition['C_sha256']==array_hash(bank['C'])
        if record['source_family']=='quick':
            cache_path=track(QUICK/'outputs'/record['source_request_id']/'result.json')
            cache=read_json(cache_path)
        else:
            cache_path=track(HYBRID/'evaluation/OT_HYBRID/cases'/f"{record['legacy_case_id']}.json")
            cache=read_json(cache_path)
            assert cache==hybrid_index[record['legacy_case_id']]
        cache_identity=_check_cache(record,bank,cache,cache_path)
        source_hashes[cache_identity['legacy_raw']]=cache_identity['legacy_raw_sha256']
        row=dict(base_request_id=base_id,condition_id=cid,panel=panel,uid=record['uid'],
            split=record['split'],candidate=record['candidate'],noise_index=record['candidate'],
            context_level=record['context_level'],alpha=1.,N=n,K=k,L0=condition['L0'],
            input_npz=str(bank_path),input_sha256=digest(bank_path),
            condition_npz=condition['condition_npz'],condition_sha256=condition['condition_sha256'],
            condition_identity=condition['condition_identity'],gaussian_seed=record['gaussian_seed'],
            gaussian_sha256=array_hash(bank['z']),C_array_sha256=array_hash(bank['C']),
            known_mask_sha256=array_hash(bank['known_mask']),valid_mask_sha256=array_hash(bank['valid_mask']),
            legacy_input_npz=str(original_input),legacy_input_sha256=digest(original_input),
            legacy_case_id=record.get('legacy_case_id',record.get('case_id')),
            source_family=record['source_family'],source_request_id=record.get('source_request_id'),
            status='READY',reason=None,**cache_identity)
        base_rows.append(row);cache_audits.append(dict(base_request_id=base_id,**cache_identity))
    # Protect prepared geometry and source index manifests actually used above.
    object_manifest=read_json(SOURCE/'data/manifest.json')
    objects=object_manifest.get('objects',object_manifest.get('records',[]))
    if isinstance(objects,dict):objects=list(objects.values())
    used={x['uid'] for x in records}
    for obj in objects:
        if obj.get('uid') in used and obj.get('prepared_path'):
            path=Path(obj['prepared_path'])
            if not path.is_absolute():path=SOURCE/path
            track(path)
    requests=[]
    for label in LABELS:
        arm='START_HYBRID' if label=='START_HYBRID' else label[:-4]
        for base in base_rows:
            row=dict(base,label=label,arm=arm,request_id=label+'__'+base['base_request_id'])
            if base['status']=='NOT_APPLICABLE':
                row['cache_status']='NOT_APPLICABLE'
            elif label=='START_HYBRID':
                row['status']='CACHED'
            else:
                row.update(status='READY',cache_status='NOT_REQUESTED',
                    raw_path=None,raw_sha256=None,checkpoint_sha256=None)
            requests.append(row)
    assert len(requests)==400 and len({x['request_id'] for x in requests})==400
    assert sum(x['status']=='NOT_APPLICABLE' for x in requests)==20
    assert sum(x['status']=='CACHED' for x in requests)==76
    assert len(conditions)==32
    replays=[]
    for panel in ('P2','P3'):
        first=next(x for x in requests if x['label']=='START_HYBRID' and x['panel']==panel and x['status']=='CACHED' and x['candidate']==0)
        replays.append(dict(request_id=first['request_id'],base_request_id=first['base_request_id'],
            panel=panel,uid=first['uid'],condition_id=first['condition_id'],candidate=0,
            input_npz=first['input_npz'],input_sha256=first['input_sha256'],
            expected_raw=first['legacy_raw'],expected_raw_sha256=first['legacy_raw_sha256'],
            model_arm='B_BOUNDARY',initial_boundary_vector='all_zero',
            comparison='all_raw_float32_bytes',required_NFE=50,
            D_reuses_identical_zero_path=True))
    sensitivity=[dict(request_id=x['request_id'],uid=x['uid'],label=x['label'],
        condition_id=x['condition_id'],candidate=0,panel='P1')
        for x in requests if x['panel']=='P1' and x['uid'] in parents['P1'][:2] and
        x['candidate']==0 and x['label'] in ('START_HYBRID','D_BOUNDARY_SEAM_END')]
    assert len(sensitivity)==4
    main_definition=dict(C_area_samples=2048,gamma_midpoints_per_edge=64,directions=26)
    dense_definition=dict(C_area_samples=8192,gamma_midpoints_per_edge=256,directions=74,
        construction='normalized original 26 integer directions plus all signed permutations of (2,1,0) and (2,1,1), lexicographic order')
    freeze_json(out/'registration/conditions.json',list(conditions.values()))
    freeze_json(out/'registration/base_requests.json',base_rows)
    freeze_json(out/'requests.json',requests)
    freeze_json(out/'registration/identity_replays.json',replays)
    freeze_json(out/'registration/sensitivity_units.json',sensitivity)
    freeze_json(out/'registration/cache_audit.json',dict(status='PASS',rows=cache_audits,
        cached_outputs=76,identity_requirement='START exact checkpoint/C/z/masks/Euler50/FP32 raw/hash',
        missing_cache_action='BLOCKED, no substitute noise or hidden sampling'))
    freeze_json(out/'registration/metric_sampling.json',dict(main=main_definition,dense=dense_definition,
        C_seed="SHA256('CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_V1|20261007|C_AREA|condition_id|count') first 4 bytes little endian",
        sensitivity_unit_count=4,density_never_model_selected=True))
    freeze_json(out/'registration/source_hashes_before.json',source_hashes)
    local_files=[out/'requests.json']+list((out/'inputs').rglob('*.npz'))+[
        out/'registration'/s for s in ('conditions.json','base_requests.json','identity_replays.json','sensitivity_units.json','metric_sampling.json','cache_audit.json')]
    freeze_json(out/'registration/input_file_hashes.json',{str(p):digest(p) for p in sorted(local_files)})
    summary=dict(schema=SCHEMA,status='PASS',requests=400,applicable=380,not_applicable=20,
        per_label_requests=80,per_label_applicable=76,START_cache=76,formal_new_samples=304,
        base_requests=80,inference_banks=76,conditions=32,applicable_conditions=30,
        panel_counts=dict(P1=32,P2=8,P3=16,P4=16,P5=8),parents=parents,
        no_GT_in_inference=True,inference_bank_allowlist=list(BANK_KEYS),
        model_calls=0,new_Gaussians=0,geometry_repairs=0,identity_replays=2,
        sensitivity_units=4,START_sha256=START_SHA,
        whole_gamma_na_conditions=[x['condition_id'] for x in conditions.values() if x['status']=='READY' and x['gamma_status']!='PASS'],
        paths=dict(requests=str(out/'requests.json'),conditions=str(out/'registration/conditions.json'),
            identity_replays=str(out/'registration/identity_replays.json'),
            sensitivity=str(out/'registration/sensitivity_units.json')))
    freeze_json(out/'registration/summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT))
    args=parser.parse_args()
    print(json.dumps(prepare(Path(args.out)),sort_keys=True))
