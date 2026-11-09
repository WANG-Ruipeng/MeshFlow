import copy
from .candidate_gate import classify,UIDS


def rows():
    records=[]
    for i,uid in enumerate(UIDS):
        for name,values,q in [('fixed',[1.,1.],True),('old_final',[.97,.97] if i==0 else [.99,.99],i==0),
                              ('margin_final',[.94,.94] if i<2 else [.99,.99],i<2)]:
            records.append({'uid':uid,'endpoint':name,'status':'success','metrics':[{'seed':s,'rms_bbox_percent':v} for s,v in zip((1,2),values)],
                'audit':{'valid':True,'exact_executed':True},'quality':{'face_count':512},'qualified_original':{'qualified':q}})
    return records


def test_positive_budget_gate_is_not_training_authorization():
    result=classify(rows());assert result['candidate_go'] is True
    assert result['training_authorized_by_this_report'] is False
    assert len(result['old_joint_success_uids'])==1 and len(result['new_joint_success_uids'])==2


def test_cannot_sacrifice_previous_joint_success():
    data=rows();data[2]['qualified_original']['qualified']=False
    result=classify(data)
    assert not result['candidate_go'] and result['lost_old_joint_success_uids']==[UIDS[0]]


def test_geometry_missing_or_one_MC_gain_fails_independently():
    data=rows();data[-1]['audit']['valid']=False
    assert not classify(data)['conditions']['all_four_new_finals_exact_valid_and_F512']
    data=rows();data[5]['metrics'][1]['rms_bbox_percent']=.995
    assert not classify(data)['conditions']['at_least_two_new_vs_old_dual_MC_ge2pct']
    data=rows();data[-1]={'uid':UIDS[-1],'endpoint':'margin_final','status':'error'}
    result=classify(data);assert not result['candidate_go'] and result['per_uid'][-1]['margin_gain_over_old_fraction'] is None
