"""CPU-only contract tests; real optimizer cost is exercised by the separate smoke command."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ..io_utils import array_hash,file_hash
from ..layout_value.optimizer import optimize_layout as original_optimizer
from ..legacy.geometry_task import fibonacci,hull_faces
from . import warm_start as ws


def result_fixture(iterations=4,snapshot_ids=(0,1,4),best=1):
    u=np.eye(3);f=np.array([[0,1,2]],np.int64)
    trace=[{'iteration':i,'phase':'surface','accepted':True} for i in range(iterations+1)]
    return {'u':u*(best+1),'vertices':u*(best+1),'faces':f,
        'stats':{'iterations':iterations,'stop_reason':'max_seconds','best_feasible_iteration':best},
        'trace':trace,'attempts':[{'iteration':i,'accepted':True} for i in range(1,iterations+1)],
        'snapshots':[{**trace[i],'u':u*(i+1),'vertices':u*(i+1),'faces':f} for i in snapshot_ids]}


def test_original_optimizer_and_fixed_budget_contract():
    assert ws.optimize_layout is original_optimizer
    assert ws.MAX_STEPS==16 and ws.MAX_SECONDS==3600 and ws.SNAPSHOTS==(0,1,4,16)
    assert len(ws.UIDS)==4 and len(set(ws.UIDS))==4 and len(ws.MODELS)==5
    assert ws.SEED_BASES==(1593501,1693501) and ws.POINTS==8192


def test_early_termination_and_missing_snapshots_are_not_fake16():
    result=result_fixture(3,(0,1,3));rows,extra=ws.snapshot_records(result)
    assert [r['status'] for r in rows]==['available','available','terminated_before_snapshot','terminated_before_snapshot']
    assert rows[-1]['actual_last_iteration']==3 and [x['iteration'] for x in extra]==[3]
    result=result_fixture(16,(0,1,16));rows,_=ws.snapshot_records(result)
    assert rows[2]['status']=='missing_snapshot' and rows[3]['status']=='available'
    result['snapshots'].append(copy.deepcopy(result['snapshots'][0]))
    with pytest.raises(ValueError,match='Duplicate'):ws.snapshot_records(result)


def test_best_feasible_final_is_saved_separately_from_last_snapshot(tmp_path):
    result=result_fixture();ws.serialize_optimizer(result,tmp_path)
    with np.load(tmp_path/'OPTIMIZER_ARRAYS.npz') as z:
        assert np.array_equal(z['final_u'],result['u'])
        assert not np.array_equal(z['final_u'],z['snapshot_4_u'])
        assert np.array_equal(z['snapshot_4_u'],result['snapshots'][-1]['u'])
    record=ws.read_json(tmp_path/'OPTIMIZER.json')
    assert record['stats']['best_feasible_iteration']==1 and record['no_fixed_fallback']
    assert record['arrays_sha256']==file_hash(tmp_path/'OPTIMIZER_ARRAYS.npz')


def test_GT_reference_is_evaluation_only_not_replaced_by_mainfield(tmp_path,monkeypatch):
    uid=ws.UIDS[0];u=fibonacci(258);f=hull_faces(u).astype(np.int32)
    refv=np.array([[3.,0,0],[0,4.,0],[0,0,5.]]);reff=np.array([[0,1,2]],np.int32)
    d={'logr':np.zeros(2050),'base_u':u,'faces':f,'ref_vertices':refv,'ref_faces':reff}
    class Support:
        def query(self,y,q):assert np.array_equal(y,d['logr']);return np.ones(len(q))
        def mesh(self,y):raise AssertionError('Must not use field as primary GT metric reference')
    quality={'face_count':512,'min_triangle_angle_deg':20.,'triangle_min_angle_p05_deg':30.,'degenerate_count':0}
    monkeypatch.setattr(ws,'named_quality',lambda v,f:quality.copy())
    monkeypatch.setattr(ws,'audit_mesh',lambda v,f,exact:{'valid':True})
    seen=[]
    def metric(run,parent,v,f,rv,rf,seeds,points):
        assert np.array_equal(rv,refv) and np.array_equal(rf,reff)
        assert points==8192;seen.extend(seeds)
        return [{'metrics':{'seed':seed,'rms_bbox_percent':1.}} for seed in seeds]
    monkeypatch.setattr(ws,'metric_cache',metric)
    row=ws.evaluate_u(tmp_path,{'all16_sorted_uids':[uid]},uid,'fixed','raw',u,d,Support(),quality,u)
    assert row['status']=='success' and row['GT_reference_hash']==array_hash(refv,reff)
    assert row['pure_network_result'] is False and row['raw_initialization_only'] is True
    assert seen==list(ws.SEED_BASES) and row['final_radius_requery_exact']
    bad=ws.evaluate_u(tmp_path,{'all16_sorted_uids':[uid]},uid,'fixed','bad',u,d,Support(),quality,u*2)
    assert bad['status']=='failed' and 'stale' in bad['error']


def test_prepare_refuses_partial_source_before_creating_output(tmp_path,monkeypatch):
    monkeypatch.setattr(ws,'source_files',lambda:{})
    def incomplete(*args):raise ValueError('Only fixed240k final checkpoints may initialize')
    monkeypatch.setattr(ws,'validate_final',incomplete)
    with pytest.raises(ValueError,match='fixed240k'):ws.prepare(tmp_path/'long',tmp_path/'balanced',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_validate_final_rejects_20k_even_if_labeled_final(tmp_path):
    folder=tmp_path/'training/joint_10';folder.mkdir(parents=True)
    (tmp_path/'RUN_PROTOCOL.json').write_text('{}')
    state={'update':20000,'config':{'total_updates':20000},'is_predeclared_final':True}
    torch.save(state,folder/'final.pt');(folder/'TRAIN_REPORT.json').write_text(json.dumps({'update':20000}))
    with pytest.raises(ValueError,match='fixed240k'):ws.validate_final(tmp_path,'longfit',10)


def full_attempts():
    answer=[]
    for uid in ws.UIDS:
        for model in ws.MODELS:
            rows=[]
            for kind in ws.ROW_KINDS:
                rows.append({'uid':uid,'model':model,'kind':kind,'status':'success','qualified':True,
                    'mainfield_hash':'field_'+uid,'GT_reference_hash':'GT_'+uid,
                    'metrics':[{'seed':seed,'rms_bbox_percent':2 if model=='fixed' else 1} for seed in ws.SEED_BASES]})
            answer.append({'uid':uid,'model':model,'rows':rows,'optimizer_stats':{'iterations':16}})
    return answer


def test_summary_fixed_denominators_and_GT_pairing():
    attempts=full_attempts();target=attempts[1]['rows'][4]
    target['status']='terminated_before_snapshot';target.pop('qualified')
    value=ws.summary(attempts)
    assert value['task_count']==20 and value['primary_row_count']==120 and value['missing_or_failed_primary_rows']==1
    group=next(x for x in value['groups'] if x['model']=='longfit_10')
    assert group['parents'][0]['final_vs_Fixed_optimized_gain_percent']==[50.,50.]
    assert value['pure_A_claim_permitted'] is False
    left=copy.deepcopy(attempts[1]['rows'][-1]);right=attempts[0]['rows'][-1]
    left['GT_reference_hash']='different'
    with pytest.raises(ValueError,match='GT reference'):ws.gain(left,right)
    with pytest.raises(ValueError,match='task'):ws.summary(attempts[:-1])


def test_raw_initializer_exact_replay_rejects_dtype_or_endpoint_change(tmp_path):
    mesh={'u':np.eye(3),'r':np.ones(3),'vertices':np.eye(3),'faces':np.array([[0,1,2]],np.int64),'alpha':np.ones(32)}
    path=tmp_path/'raw.npz';np.savez(path,**mesh);record={'path':str(path),'sha256':file_hash(path)}
    assert ws.verify_raw_initializer(mesh,record)['exact_source_replay']
    for key in ('u','r','vertices','faces','alpha'):
        changed={k:v.copy() for k,v in mesh.items()};changed[key].flat[0]+=1
        with pytest.raises(ValueError,match='replay mismatch'):ws.verify_raw_initializer(changed,record)
    changed={**mesh,'faces':mesh['faces'].astype(np.int32)}
    with pytest.raises(ValueError,match='replay mismatch'):ws.verify_raw_initializer(changed,record)
