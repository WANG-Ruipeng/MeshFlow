import json
import pytest
from .evaluate import MODEL_NAMES, POLICIES, expected_slots, registered_uids, aggregate


def protocol():
    uids=[f'p{i:02}' for i in range(16)]
    return {'uids':uids, 'train_uids':uids[:12], 'internal_check_uids':uids[12:],
        'jobs':{'single_energy_10':{'active_training_uids':[uids[0]]},'single_prior_10':{'active_training_uids':[uids[1]]}},
        'evaluation':{'model_uids':{n:([uids[0]] if n=='single_energy_10' else [uids[1]] if n=='single_prior_10' else uids) for n in MODEL_NAMES}}}


def test_full_plan_keeps_single_object_scope_and_fixed_denominators():
    p=protocol(); slots=expected_slots(p)
    assert len(slots)==164
    assert len({(u,n) for u,n,_ in slots})==98
    assert all(u in p['train_uids'] for u,n,_ in slots if n.startswith('single_'))
    p['evaluation']['model_uids']['single_prior_10']=[p['internal_check_uids'][0]]
    with pytest.raises(ValueError,match='training parent'):expected_slots(p)


def test_joint_omitted_parent_is_rejected():
    p=protocol();p['evaluation']['model_uids']['joint_10']=p['uids'][:-1]
    with pytest.raises(ValueError,match='omits parents'):registered_uids(p,'joint_10')


def test_attempt_error_and_unattempted_keep_all_planned_slots(tmp_path):
    p=protocol();(tmp_path/'RUN_PROTOCOL.json').write_text(json.dumps(p))
    folder=tmp_path/'attempts'/'p00'/'joint_10';folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps({'uid':'p00','model':'joint_10','status':'error','error':'synthetic checkpoint failure','rows':[]}))
    ledger=aggregate(tmp_path)
    assert ledger['expected_attempt_count']==98 and ledger['expected_row_count']==164
    assert ledger['attempt_count']==1 and len(ledger['errors'])==1
    assert len(ledger['failed_row_slots'])==2 and len(ledger['unattempted'])==97
    assert len(ledger['missing_success_rows'])==164

def test_single_other_training_parent_is_rejected_even_when_train_split_matches():
    p=protocol()
    p['evaluation']['model_uids']['single_energy_10']=p['jobs']['single_prior_10']['active_training_uids']
    with pytest.raises(ValueError,match='own registered training parent'):
        registered_uids(p,'single_energy_10')
