"""Regression tests for evidence accounting, hash dtypes and exclusive sealing."""
import ast
import copy
import numpy as np
import pytest
import torch
from . import package_results as pack
from ..io_utils import array_hash


def partial_ledger():
    uid='example'
    row={'uid':uid,'model':'fixed','policy':'raw','status':'success'}
    success={'uid':uid,'model':'fixed','status':'success','rows':[row]}
    failed={'uid':uid,'model':'c1_10','status':'error','rows':[],'error':'retained'}
    expected={(uid,m,p) for m in pack.MODEL_NAMES for p in pack.POLICIES[m]}
    missing={(uid,m) for m in pack.MODEL_NAMES if m not in ('fixed','c1_10')}
    ledger={'rows':[row],'errors':[failed],'attempt_count':2,'expected_attempt_count':10,'expected_row_count':18,
            'failed_row_slots':[(uid,'c1_10',p) for p in pack.POLICIES['c1_10']],
            'unattempted':sorted(missing),'missing_success_rows':sorted(expected-{(uid,'fixed','raw')})}
    return ledger,[success,failed]


def test_failure_and_missing_attempts_stay_in_full_denominator():
    ledger,attempts=partial_ledger()
    result=pack.attempt_accounting(ledger,attempts,['example'])
    assert result['planned_attempts']==10 and result['planned_row_slots']==18
    assert result['saved_success_rows']==1 and result['error_accounted_row_slots']==2
    assert len(result['missing_attempts_retained'])==8 and len(result['missing_row_slots_retained'])==15
    assert not result['all_planned_attempts_completed']
    bad=copy.deepcopy(ledger);bad['errors']=[]
    with pytest.raises(ValueError,match='discarded errors'):pack.attempt_accounting(bad,attempts,['example'])
    bad=copy.deepcopy(ledger);bad['failed_row_slots']=[]
    with pytest.raises(ValueError,match='Failed slots'):pack.attempt_accounting(bad,attempts,['example'])


def test_success_cannot_omit_declared_policy_or_duplicate_rows():
    ledger,attempts=partial_ledger()
    bad=copy.deepcopy(attempts);bad[1]['status']='success'
    with pytest.raises(ValueError,match='omits/duplicates'):pack.attempt_accounting(ledger,bad,['example'])
    bad_ledger=copy.deepcopy(ledger);bad_ledger['rows'].append(bad_ledger['rows'][0])
    with pytest.raises(ValueError,match='Duplicate aggregate'):pack.attempt_accounting(bad_ledger,attempts,['example'])


def test_logical_and_storage_connectivity_hashes_are_not_interchangeable():
    vertices=np.eye(3,dtype=np.float64);faces=np.array([[0,1,2]],np.int32)
    old={'vertices':vertices.copy(),'faces':faces.astype(np.int64),'u':vertices.copy(),'r':np.ones(3)}
    new={**old,'faces':faces.copy()}
    logical=array_hash(vertices,faces);stored_old=array_hash(vertices,old['faces'])
    prior={'mesh_hash':logical};identity={'stored_array_hash':stored_old}
    current={'mesh_hash':logical,'stored_array_hash':logical}
    assert logical!=stored_old
    report=pack.compare_prior_mesh(old,new,faces,prior,identity,current)
    assert report['source_faces_dtype']=='int64' and report['saved_faces_dtype']=='int32'
    assert report['geometry_bitwise_equal']
    with pytest.raises(ValueError,match='Prior stored'):
        pack.compare_prior_mesh(old,new,faces,prior,{'stored_array_hash':logical},current)
    changed=copy.deepcopy(new);changed['u'][0,0]+=.001
    with pytest.raises(ValueError,match='Frozen prior geometry changed'):
        pack.compare_prior_mesh(old,changed,faces,prior,identity,current)


def sequence_states():
    state={'trace':[{'uids':['a','b','c','d'],'rng_matched_k':[1,2,3,4]}],
           'sampler':{'exposures':[1,1,1,1]},'rng':{'torch_cpu':torch.tensor([1,2],dtype=torch.uint8),
           'numpy':np.array([3,4]),'python':(1,2)}}
    return {(arm,seed):copy.deepcopy(state) for arm in ('c0','c1','c2') for seed in (10,11)}


def test_equal_updates_require_matched_parents_draws_and_saved_rng():
    states=sequence_states();assert pack.check_paired_sequences(states)['10']['equal_compute_claim'] is False
    states[('c2',10)]['trace'][0]['rng_matched_k'][0]=2
    with pytest.raises(ValueError,match='Parent/RNG sequence'):pack.check_paired_sequences(states)
    states=sequence_states();states[('c1',11)]['rng']['torch_cpu'][0]=7
    with pytest.raises(ValueError,match='RNG progression'):pack.check_paired_sequences(states)


def test_failed_preflight_creates_no_partial_sealing_artifacts(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'original.txt').write_text('preserve')
    before=sorted(str(p) for p in tmp_path.rglob('*'))
    def failure(*args):raise ValueError('sentinel failed scientific audit')
    monkeypatch.setattr(pack,'audit_run',failure)
    with pytest.raises(ValueError,match='sentinel'):pack.package_run(run,tmp_path/'smoke')
    assert sorted(str(p) for p in tmp_path.rglob('*'))==before
    assert (run/'original.txt').read_text()=='preserve'


def test_fresh_replay_is_valid_code_and_writes_actual_json_newline():
    module=ast.parse(pack.REPLAY_SCRIPT)
    compile(module,'<fresh-replay>','exec')
    writes=[node for node in ast.walk(module) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='write' and node.args and isinstance(node.args[0],ast.Constant)]
    assert any(node.args[0].value=='\n' for node in writes)
    assert "'training/c2/seed_10/final.pt'" in pack.REPLAY_SCRIPT
    assert "uid=sorted(protocol['uids'])[0]" in pack.REPLAY_SCRIPT
