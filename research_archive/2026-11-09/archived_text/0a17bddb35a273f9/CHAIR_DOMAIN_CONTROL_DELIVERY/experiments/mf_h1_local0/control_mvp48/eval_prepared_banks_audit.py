"""Audit existing C banks and export their inventory; no OT, scoring or models."""
from __future__ import annotations
import csv
import json
import os
import time
from collections import defaultdict
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import numpy as np
from .eval_common import RUN,EVALUATION,REGISTRATION,read_json,sha256,snapshot,write_json
from .eval_prepare import array_hash,load_data_api


def same(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def saved(row):
    assert sha256(RUN/row['path']) == row['sha256'],row['path']
    with np.load(RUN/row['path'],allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def main():
    start=time.perf_counter();resources=snapshot()
    ledger_path=REGISTRATION/'eval_preparation_ledger.json'
    ledger_hash=sha256(ledger_path);ledger=read_json(ledger_path)
    protocol_path=REGISTRATION/'eval_protocol.json'
    protocol_hash=sha256(protocol_path);protocol=read_json(protocol_path)
    requests_path=REGISTRATION/'eval_requests.json'
    requests_hash=sha256(requests_path);requests=read_json(requests_path)
    ready_hash=sha256(RUN/'DATA_READY.json')
    assert ledger['status']=='PASS' and ledger['protocol_sha256']==protocol_hash
    assert protocol['data_identity']['data_ready_sha256']==ready_hash
    assert ledger['OT_attempts']==ledger['OT_returns']==8
    assert ledger['model_forwards']==0 and ledger['CUDA_initialized'] is False
    assert len(ledger['test_inputs'])==32 and len(ledger['conditions'])==16 and len(ledger['teacher_inputs'])==8
    assert len(requests)==96 and len({r['request_id'] for r in requests})==96
    api=load_data_api();sample_cache={};inventory=[];ref_audit=[];teacher_audit=[]
    def sample(row):
        key=(row['parent_id'],row['condition_id'])
        if key not in sample_cache:
            sample_cache[key]=api.load_sample(*key)
        return sample_cache[key]
    def inventory_row(row,kind,input_id,c_hash,noise_hash,pairing_hash='',model_hash=''):
        s=sample(row);repair=s['repair_metadata']
        return dict(kind=kind,input_id=input_id,source_case_id=row['source_case_id'],parent_id=row['parent_id'],condition_id=row['condition_id'],noise_id=row['noise_id'],gaussian_seed=row['gaussian_seed'],N=s['N'],K=s['K'],free_count=s['N']-s['K'],original_N=s['original_N'],removed_exact_zero_faces=len(repair['removed_zero_original_face_ids']),removed_duplicate_faces=len(repair['removed_duplicate_original_face_ids']),path=row['path'],sha256=row['sha256'],C_sha256=c_hash,noise_sha256=noise_hash,pairing_sha256=pairing_hash,teacher_model_input_sha256=model_hash,data_ready_sha256=ready_hash,protocol_sha256=protocol_hash)
    by_case=defaultdict(list)
    for row in requests:by_case[row['case_id']].append(row)
    by_parent_noise=defaultdict(list);by_condition=defaultdict(list)
    for row in ledger['test_inputs']:
        arrays=saved(row);s=sample(row);n,k=row['N'],row['K']
        assert set(arrays)=={'C','z','N','known_mask','valid_mask'}
        assert n==s['N'] and k==s['K'] and int(arrays['N'])==n
        assert arrays['z'].shape==(1,n,9) and arrays['z'].dtype==np.float32
        assert arrays['C'].shape==(k,3,3) and same(arrays['C'],s['constraints'])
        assert np.isfinite(arrays['z']).all() and (arrays['z'][:,:k]==0).all()
        assert np.array_equal(arrays['known_mask'],np.arange(n)<k) and arrays['valid_mask'].all()
        for name in ('C','z','known_mask','valid_mask'):assert array_hash(arrays[name])==row[name+'_sha256']
        noise=arrays['z'][0,k:].reshape(n-k,3,3)
        assert array_hash(noise)==row['noise_sha256']
        matches=by_case[row['case_id']]
        assert len(matches)==3 and {m['model'] for m in matches}=={'T1','Geo','Full'}
        for m in matches:
            assert m==dict(row,model=m['model'],request_id=f"{row['case_id']}_{m['model']}")
        by_parent_noise[(row['parent_id'],row['noise_id'])].append(row)
        by_condition[(row['parent_id'],row['condition_id'])].append(row)
        inventory.append(inventory_row(row,'test',row['case_id'],row['C_sha256'],row['noise_sha256']))
    assert len(by_parent_noise)==16 and len(by_condition)==16
    for rows in by_parent_noise.values():
        assert len(rows)==2 and {r['condition_id'] for r in rows}=={'p0','p1'}
        for field in ('N','K','gaussian_seed','noise_sha256','z_sha256'):
            assert len({r[field] for r in rows})==1
    for rows in by_condition.values():
        assert len(rows)==2 and {r['noise_id'] for r in rows}=={0,1}
        assert len({r['C_sha256'] for r in rows})==1
    from .eval_scoring import _check_reference
    from experiments.mf_h1_local0.native_inpainting_metrics import array_sha
    for row in ledger['conditions']:
        arrays=saved(row);s=sample(row)
        assert same(arrays['C'],s['constraints']) and same(arrays['gt_free'],s['free_target']) and same(arrays['full_target'],s['full_target'])
        assert int(arrays['N'])==s['N'] and int(arrays['K'])==s['K'] and float(arrays['L'])==s['bbox_diagonal']
        reference={'audit':row['reference_metadata'],'arrays':{key[10:]:val for key,val in arrays.items() if key.startswith('reference_')}}
        _check_reference(reference,arrays['C'].reshape(-1,3,3),arrays['gt_free'].reshape(-1,3,3),float(arrays['L']))
        assert reference['arrays']['pairwise_target_points'].shape==(512,3)
        assert reference['arrays']['coverage_points'].shape==(512,3)
        interface=s['interface'];full=s['full_target'].reshape(-1,3,3)
        segments=full[interface['known_source_face_ids'][:,None],interface['known_corner_ids']]
        assert array_sha(segments)==reference['audit']['gamma_segment_sha256']
        assert array_sha(interface['edge_source_vertex_ids'])==reference['audit']['gamma_source_vertex_ids_sha256']
        ref_audit.append(dict(parent_id=row['parent_id'],condition_id=row['condition_id'],path=row['path'],sha256=row['sha256'],own_complement_exact=True,reference_array_hashes='PASS',GT_queries=512))
    from .eval_teacher import _unpack_bank
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    for row in ledger['teacher_inputs']:
        arrays=saved(row);s=sample(row);identity=row['identity']
        _unpack_bank((arrays,identity))
        n,k=s['N'],s['K']
        assert identity['N']==n and identity['K']==k
        assert same(arrays['C'],s['constraints']) and same(arrays['free_target'],s['free_target']) and same(arrays['pre_ot_free'],s['pre_ot_free'])
        assert identity['actual_OT_attempts']==identity['actual_OT_returns']==1
        assert same(arrays['x0'][k:],arrays['paired_noise'].astype(np.float32).reshape(n-k,9))
        assert same(arrays['x1'][k:],s['free_target']) and same(arrays['u'],arrays['x1']-arrays['x0'])
        expected= np.float32(.5)*arrays['x0'] + np.float32(.5)*arrays['x1']
        expected[:k]=arrays['C'].reshape(k,9)
        assert same(expected,arrays['xt'])
        assert np.isfinite(arrays['xt']).all() and np.isfinite(arrays['u']).all()
        inventory.append(inventory_row(row,'teacher',row['teacher_id'],identity['array_sha256']['C'],identity['array_sha256']['raw_gaussian'],identity['pairing_sha256'],identity['model_input_sha256']))
        teacher_audit.append(dict(teacher_id=row['teacher_id'],path=row['path'],sha256=row['sha256'],array_hashes='PASS',model_input_hash=identity['model_input_sha256'],pairing_hash=identity['pairing_sha256'],OT_regenerated=False))
    assert len(inventory)==40 and len({r['path'] for r in inventory})==40
    control=read_json(REGISTRATION/'eval_control.json')
    same_noise_groups=defaultdict(set)
    for row in ledger['test_inputs']:same_noise_groups[(row['N'],row['K'],row['noise_sha256'])].add(row['parent_id'])
    assert not any(len(parents)>1 for parents in same_noise_groups.values())
    assert control['status']=='NOT_RUN'
    inventory_path=EVALUATION/'eval_input_inventory.csv'
    with inventory_path.open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    assert sha256(ledger_path)==ledger_hash and sha256(protocol_path)==protocol_hash and sha256(requests_path)==requests_hash
    assert sha256(RUN/'DATA_READY.json')==ready_hash and not torch.cuda.is_initialized()
    audit=dict(schema='control_mvp48_eval_prepared_banks_audit_v1',status='PASS',owner='Session C',data_ready_sha256=ready_hash,protocol_sha256=protocol_hash,ledger_sha256=ledger_hash,requests_sha256=requests_hash,test_inputs_checked=32,teacher_inputs_checked=8,reference_banks_checked=16,formal_requests_checked=96,exact_three_model_shared_inputs=True,same_parent_noise_across_C=True,own_complement_reference_audit=ref_audit,teacher_hash_audit=teacher_audit,inventory={'path':str(inventory_path.relative_to(RUN)),'sha256':sha256(inventory_path),'rows':40},optional_control_status='NOT_RUN',optional_control_reason='Same-N pair exists but frozen cross-parent Gaussian bytes differ; no extra generation.',resource_snapshot=resources,cpu_workers=1,numerical_threads=1,OT_calls_this_audit=0,OT_calls_preparation_ledger=8,model_forwards=0,formal_scores=0,reference_samples_regenerated=0,CUDA_initialized=False,seconds=time.perf_counter()-start)
    write_json(REGISTRATION/'eval_prepared_banks_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ('status','test_inputs_checked','teacher_inputs_checked','reference_banks_checked','formal_requests_checked','inventory','seconds')},indent=2))


if __name__=='__main__':main()
