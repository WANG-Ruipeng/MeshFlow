import numpy as np
from . import report as r
from .prepare import UIDS
from ..io_utils import write_json,file_hash


def test_fractional_change_does_not_invent_zero_denominator():
    assert r.fractional_change(0,1) is None
    assert r.fractional_change(2,1)==.5
    assert r.fractional_change(2,3)==-.5


def test_summary_preserves_missing_negative_and_zero_target(tmp_path):
    p={'evaluation':{'reference_surface':'same frozen full radial field'},
       'target_displacement_energy':{uid:{'old':0. if i==3 else .1} for i,uid in enumerate(UIDS)}}
    write_json(tmp_path/'RUN_PROTOCOL.json',p);sha=file_hash(tmp_path/'RUN_PROTOCOL.json')
    rows=[]
    for name,rms in [('fixed',[2.,2.]),('old_final',[1.,1.]),('margin_final',[1.2,.8])]:
        rows.append({'uid':UIDS[0],'endpoint':name,'metrics':[{'rms_bbox_percent':x} for x in rms],
          'qualified_original':{'qualified':False},'qualified_strict':{'qualified':False},
          'actual_margin_vs_fixed_deg':{},'direction_errors':{},'quality':{'face_count':512},'audit':{'valid':True}})
    write_json(tmp_path/'results.json',{'protocol_sha256':sha,'rows':rows,'success_count':3,
      'errors':[{'uid':UIDS[1],'endpoint':'old_final','status':'error'}],
      'missing_success_slots':[(UIDS[1],'old_final')]})
    write_json(tmp_path/'TRAIN_DIAGNOSTICS.json',{'protocol_sha256':sha,'rows':[],
      'success_count':0,'missing_success_slots':[(uid,u) for uid in UIDS for u in (0,5000,20000)]})
    write_json(tmp_path/'PAIRING_AUDIT.json',{'status':'PASS'})
    result=r.summarize(tmp_path)
    assert len(result['paired'])==4 and result['planned_endpoints']==20 and result['planned_observations']==24
    assert len(result['errors'])==1 and result['paired'][3]['old_teacher_zero_motion']
    assert np.allclose(result['paired'][0]['margin_target_final_RMS_improvement_over_old_target_final'],[-.2,.2])
    assert result['paired'][1]['margin_target_final_RMS_improvement_over_old_target_final'] is None
    assert all(x['status']=='missing_or_failed' for x in result['paired'][0]['observations']['old'])
    assert result['reference_raw_asset_claim'] is False and result['checkpoint_selection'] is False
