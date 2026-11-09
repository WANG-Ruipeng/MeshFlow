"""Immutable CPU registration; never imports a model or projection implementation."""
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from .common import (ROOT, OUT, BOUNDARY, QUICK, HYBRID, START, START_SHA, LABELS,
    read_json, digest, array_hash, json_hash, freeze_json, atomic_json, resolve, arrays, write_mesh)

SCHEMA = 'chair_bounded_attachment_registration_v1'
BANK_KEYS = {'z','C','known_mask','valid_mask','N','K'}
EXPECTED_UIDS = (
    '33ec1e64a02ff718faaadcdc0221eb72','341bcae1e7f366a233d93f57729688db',
    '534b0e6e2c53a8e1a7cdfd3874ed3723','866e7248434968051cc7750d0afca796',
    '86ec9c796a6c1789e792598ed58dd1cc','bdd29e651e5f6fb2b079317292bdc5d4',
    'c1b8c2f0c4f3cc43aff7c08b060f5ed6','f59627a077dcc2566c63b8b48495261a')


def _bits(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _copy_exact(source, target, expected):
    source, target = resolve(source), Path(target)
    assert digest(source) == expected, ('Source identity changed',str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert digest(target) == expected, ('Existing destination changed',str(target))
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024)
    assert digest(target) == expected
    return target


def _validate(record, bank, raw, result):
    n,k=int(bank['N']),int(bank['K'])
    assert set(bank) == BANK_KEYS and 128 <= n <= 256 and 0 < k < n
    assert (n,k)==(record['N'],record['K'])
    assert bank['z'].dtype==np.float32 and bank['z'].shape==(1,n,9)
    assert bank['C'].dtype==np.float32 and bank['C'].shape==(k,3,3)
    for key in ('known_mask','valid_mask'):
        assert bank[key].dtype==np.bool_ and bank[key].shape==(n,)
    assert bank['valid_mask'].all() and np.array_equal(bank['known_mask'],np.arange(n)<k)
    assert np.isfinite(bank['C']).all() and np.isfinite(bank['z']).all()
    for name,field in [('C','C_array_sha256'),('z','gaussian_sha256'),
                       ('known_mask','known_mask_sha256'),('valid_mask','valid_mask_sha256')]:
        assert array_hash(bank[name]) == record[field]
    assert result['status']=='CACHED' and result['checkpoint_sha256']==START_SHA
    assert raw['raw'].dtype==np.float32 and raw['raw'].shape==(n,9)
    assert np.isfinite(raw['raw']).all()
    assert _bits(raw['raw'][:k],bank['C'].reshape(k,9))
    assert _bits(raw['C'].reshape(k,3,3),bank['C']) and _bits(raw['noise'],bank['z'])
    assert _bits(raw['known_mask'].reshape(n),bank['known_mask'])
    assert array_hash(raw['raw'])==record['raw_array_sha256']==result['raw_array_sha256']
    meta=result.get('metadata') or result['legacy_metadata']
    for name,value in dict(solver='original_Euler50',nfe=50,accepted_states=51,
        integration_dtype='float32',forward_dtype='bfloat16',known_all51_bitwise=True,
        known_state_checks=51,actual_backbone_calls=50,actual_Geo_recomputations=50,
        GT_model_inputs=False,geometry_repairs=0,initial_gaussian_scaled_by_alpha=False).items():
        assert meta.get(name)==value,(name,meta.get(name),value)
    assert meta['observed_raw_output_dtypes']==['torch.bfloat16']
    assert meta['condition_sha256']==array_hash(bank['C'].reshape(k,9))
    assert meta['noise_sha256']==array_hash(bank['z'])
    runtime_expected=dict(cublas_workspace_config=':4096:8',parameter_dtype='torch.float32',
        cuda_autocast_enabled=True,cuda_autocast_dtype='torch.bfloat16',
        deterministic_algorithms=True,deterministic_warn_only=False,
        tf32_matmul=False,tf32_cudnn=False,sdpa_math_enabled=True,
        sdpa_flash_enabled=False,sdpa_mem_efficient_enabled=False,sdpa_cudnn_enabled=False,
        gradient_enabled=False,trainable_parameters=0,all_modules_eval=True)
    for stage in ('runtime_before','runtime_after'):
        for name,value in runtime_expected.items():
            assert meta[stage].get(name)==value,(stage,name,meta[stage].get(name),value)
    path_states=raw.get('states')
    if path_states is not None:
        assert path_states.shape==(51,n,9)
        assert all(_bits(state[:k],bank['C'].reshape(k,9)) for state in path_states)
    return dict(status='PASS',N=n,K=k,C_bitwise=True,noise_bitwise=True,
        masks_bitwise=True,raw_finite=True,raw_dtype='float32',checkpoint_sha256=START_SHA,
        solver='original_Euler50',historical_nfe=50,historical_states_checked=51,
        stored_states_rechecked=51 if path_states is not None else 0,
        runtime_evidence='Historical before/after metadata; no current model load or forward',
        inference_bank_keys=sorted(BANK_KEYS),GT_in_model_bank=False)


def prepare(out=OUT):
    out=Path(out);reg=out/'registration';reg.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();source_hashes={};errors=[]
    def track(path,expected=None):
        p=resolve(path);sha=digest(p)
        if expected is not None:assert sha==expected,('Protected file changed',str(p))
        source_hashes[str(p)]=sha
        return p
    old_requests=read_json(track(BOUNDARY/'requests.json'))
    selected=sorted((r for r in old_requests if r['label']=='START_HYBRID'
        and r['panel']=='P1' and r['candidate'] in (0,1)), key=lambda r:(r['uid'],r['candidate']))
    assert len(selected)==16 and tuple(sorted(set(r['uid'] for r in selected)))==EXPECTED_UIDS
    assert all([r['candidate'] for r in selected if r['uid']==uid]==[0,1] for uid in EXPECTED_UIDS)
    assert all(r['alpha']==1 and r['context_level']==1 and r['split']=='validation' for r in selected)
    conditions={r['condition_id']:r for r in read_json(track(BOUNDARY/'registration/conditions.json'))}
    manifest=read_json(track(HYBRID/'checkpoint_manifest.json'))
    endpoint=next(r for r in manifest['endpoints'] if r['metadata']['arm']=='OT_HYBRID')
    assert endpoint['checkpoint_sha256']==START_SHA
    for key,value in dict(pilot_step=1000,conditional_total_step=5000,lambda_variance=.25,
                          loss_recipe='fm',context_encoder='geo').items():
        assert endpoint['metadata'][key]==value
    track(START,START_SHA)
    official=read_json(track(ROOT/'native_t1/recipes/chair_hybrid_start.json'))
    assert official['checkpoint_sha256']==START_SHA and official['model_state_sha256']==endpoint['metadata']['model_state_sha256']
    for rel in ('native_t1/docs/chair_start.md','experiments/mf_h1_local0/chair_hybrid_coupling_v1/sampling.py',
                'native_t1/maintenance/pair_cleanup_20261002_100559/evidence/GP_CLOSURE_REPORT.md',
                'native_t1/runs/chair_boundary_attachment_v1/CHAIR_BOUNDARY_ATTACHMENT_FACTORIAL_REPORT.md'):
        track(ROOT/rel)
    base_rows=[];new_conditions={};audits=[]
    for old in selected:
        base_id=old['base_request_id'];cid=old['condition_id'];record=dict(old,base_id=base_id,
            legacy_boundary_request_id=old['request_id'],legacy_boundary_label=old['label'],
            legacy_boundary_input_npz=old['input_npz'],legacy_boundary_input_sha256=old['input_sha256'],
            legacy_boundary_condition_npz=old['condition_npz'],legacy_boundary_condition_sha256=old['condition_sha256'])
        try:
            source_input=track(old['input_npz'],old['input_sha256']);bank=arrays(source_input)
            source_raw=track(old['raw_path'],old['raw_sha256']);raw=arrays(source_raw)
            old_result_path=track(BOUNDARY/'outputs'/old['request_id']/'result.json')
            old_result=read_json(old_result_path)
            assert old_result['raw_sha256']==old['raw_sha256'] and old_result['input_sha256']==old['input_sha256']
            track(old['legacy_case_json'],old['legacy_case_sha256'])
            track(old['legacy_input_npz'],old['legacy_input_sha256'])
            check=_validate(old,bank,raw,old_result)
            cp=conditions[cid];source_condition=track(cp['condition_npz'],cp['condition_sha256'])
            condition=arrays(source_condition)
            assert _bits(condition['C'],bank['C'])
            assert int(condition['N'])==old['N'] and int(condition['K'])==old['K']
            assert cp['condition_identity']==old['condition_identity']
            assert cp['condition_identity']==json_hash({k:array_hash(v) for k,v in sorted(condition.items())})
            input_path=_copy_exact(source_input,out/'inputs/inference'/f'{base_id}.npz',old['input_sha256'])
            condition_path=_copy_exact(source_condition,out/'inputs/conditions'/f'{cid}.npz',cp['condition_sha256'])
            if cid not in new_conditions:
                new_conditions[cid]=dict(cp,condition_npz=str(condition_path),
                    legacy_boundary_condition_npz=str(source_condition),
                    legacy_boundary_condition_sha256=cp['condition_sha256'],copied_bitwise=True)
            joined_path=track(BOUNDARY/'scoring/joined'/f"{old['request_id']}.json")
            joined=read_json(joined_path)
            assert joined['status']=='PASS' and joined['raw_sha256']==old['raw_sha256']
            assert joined['condition_id']==cid and joined['candidate']==old['candidate']
            reference_npz=track(joined['control_reference_path'],joined['control_reference_sha256'])
            reference_meta=track(reference_npz.with_suffix('.json'))
            reference=read_json(reference_meta)
            assert reference['npz_sha256']==digest(reference_npz)
            assert reference['cache_identity']['condition_sha256']==old['condition_sha256']
            reference_arrays=arrays(reference_npz)
            assert all(array_hash(reference_arrays[k])==v for k,v in reference['arrays_sha256'].items())
            query_npz=track(BOUNDARY/'scoring/queries'/f'{cid}.npz')
            query_meta=track(query_npz.with_suffix('.json'));qmeta=read_json(query_meta)
            assert qmeta['npz_sha256']==digest(query_npz) and qmeta['identity']['condition_id']==cid
            query_arrays=arrays(query_npz)
            assert all(array_hash(query_arrays[k])==v for k,v in qmeta['array_hashes'].items())
            pointers=dict(legacy_boundary_joined_path=joined_path,
                legacy_boundary_result_path=old_result_path,
                legacy_control_score_path=track(joined['control_score_path']),
                legacy_control_vectors_path=track(joined['control_vectors_path'],joined['control_vectors_sha256']),
                legacy_control_reference_meta_path=reference_meta,
                legacy_control_reference_npz_path=reference_npz,
                legacy_quality_query_npz_path=query_npz,legacy_quality_query_meta_path=query_meta,
                legacy_quality_vectors_path=track(joined['vectors_path'],joined['vectors_sha256']),
                legacy_score_path=track(joined['legacy_score_path']))
            for name,path in pointers.items():
                record[name]=str(path);record[name.removesuffix('_path')+'_sha256']=digest(path)
            raw_id='RAW__'+base_id;raw_dir=out/'outputs'/raw_id
            raw_path=_copy_exact(source_raw,raw_dir/'raw.npz',old['raw_sha256'])
            ply_path=raw_dir/'mesh.ply'
            if not ply_path.exists():write_mesh(ply_path,raw['raw'])
            record.update(input_npz=str(input_path),condition_npz=str(condition_path),
                raw_path=str(raw_path),raw_sha256=old['raw_sha256'],mesh_path=str(ply_path),
                mesh_sha256=digest(ply_path),registration_status='PASS',
                original_raw_path=str(source_raw),original_raw_sha256=old['raw_sha256'])
            result=dict(record,label='RAW',arm='RAW',request_id=raw_id,status='CACHED',
                raw_origin='historical_START_bitwise_file_copy',metadata=old_result['metadata'],
                source_identity=dict(boundary_request_id=old['request_id'],source_raw_path=str(source_raw),
                    source_raw_sha256=old['raw_sha256'],source_result_path=str(old_result_path),
                    source_result_sha256=digest(old_result_path),same_condition_id=True,
                    same_noise=True,identity_runtime_scope='historical'),
                model_calls=0,sampling_attempts=0,training_updates=0,geometry_repairs=0)
            freeze_json(raw_dir/'result.json',result)
            audits.append(dict(base_id=base_id,uid=old['uid'],candidate=old['candidate'],
                raw_path=str(raw_path),raw_sha256=old['raw_sha256'],source_raw_sha256=old['raw_sha256'],
                source_and_copy_sha256_equal=True,queries_reused_bitwise=True,**check))
        except (FileNotFoundError,KeyError,ValueError,AssertionError) as exc:
            record.update(registration_status='ASSET_BLOCKED',reason=repr(exc))
            errors.append(dict(base_id=base_id,status='ASSET_BLOCKED',error=repr(exc)))
            audits.append(dict(base_id=base_id,status='ASSET_BLOCKED',error=repr(exc)))
        base_rows.append(record)
    requests=[]
    for label in LABELS:
        for base in base_rows:
            row=dict(base,request_id=label+'__'+base['base_id'],label=label,arm=label,
                status=('CACHED' if label=='RAW' and base['registration_status']=='PASS' else
                        'ASSET_BLOCKED' if base['registration_status']!='PASS' else 'NOT_RUN'))
            if label!='RAW':row.update(raw_path=None,raw_sha256=None,mesh_path=None,mesh_sha256=None)
            requests.append(row)
    assert len(requests)==48 and len({r['request_id'] for r in requests})==48
    freeze_json(out/'base_requests.json',base_rows);freeze_json(out/'requests.json',requests)
    freeze_json(reg/'conditions.json',[new_conditions[k] for k in sorted(new_conditions)])
    freeze_json(reg/'raw_identity_audit.json',dict(status='PASS' if not errors else 'ASSET_BLOCKED',rows=audits,errors=errors))
    replay=dict(status='FROZEN_NOT_RUN',requires_phase1='PASS',
        base_ids=[base_rows[0]['base_id'],base_rows[-1]['base_id']],
        reason='First sorted UID candidate0 and last sorted UID candidate1; no outcome selection',
        sampling_attempt_limit=2)
    freeze_json(reg/'identity_replays.json',replay)
    freeze_json(reg/'source_hashes.json',source_hashes)
    freeze_json(reg/'model_input_manifest.json',dict(schema=SCHEMA,checkpoint=str(START),
        checkpoint_sha256=START_SHA,checkpoint_file_actually_hashed=True,metadata=endpoint['metadata'],
        model_loaded=False,model_forward_calls=0,projection_calls=0,bank_keys=sorted(BANK_KEYS),
        GT_excluded_from_bank=True,rows=[{k:r.get(k) for k in ('base_id','uid','candidate','N','K',
            'input_npz','input_sha256','gaussian_seed','gaussian_sha256','C_array_sha256',
            'known_mask_sha256','valid_mask_sha256','registration_status')} for r in base_rows]))
    summary=dict(schema=SCHEMA,status='PASS' if not errors else 'ASSET_BLOCKED',base_count=16,
        request_count=48,parent_count=8,condition_count=len(new_conditions),
        cached_raw_count=sum(a['status']=='PASS' for a in audits),errors=errors,
        new_model_calls=0,new_sampling_attempts=0,projection_calls=0,
        seconds=time.perf_counter()-tick,source_files_hashed=len(source_hashes),
        all_results_development_objects=True,noise_replicates_not_independent_parents=True,
        base_requests_sha256=digest(out/'base_requests.json'),requests_sha256=digest(out/'requests.json'))
    atomic_json(reg/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT));args=parser.parse_args()
    print(json.dumps(prepare(args.out),indent=2))