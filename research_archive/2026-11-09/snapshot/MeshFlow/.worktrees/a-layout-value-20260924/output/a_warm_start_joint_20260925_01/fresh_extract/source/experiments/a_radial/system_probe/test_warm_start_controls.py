"""Light CPU controls: fixed tasks, source identity, pairing, and no hidden search."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from . import warm_start as ws
from . import warm_start_controls as controls
from ..io_utils import file_hash
from ..layout_value.optimizer import optimize_layout


def attempts(models):
    values=[]
    for uid in ws.UIDS:
        for name in models:
            rows=[]
            for kind in ws.ROW_KINDS:
                rows.append({'uid':uid,'model':name,'kind':kind,'status':'success','qualified':True,
                    'mainfield_hash':'field_'+uid,'GT_reference_hash':'reference_'+uid,
                    'metrics':[{'seed':seed,'points_per_direction':8192,'rms_bbox_percent':
                        2. if name=='fixed' else 1.5 if name in controls.NEW_MODELS else 1.} for seed in ws.SEED_BASES]})
            values.append({'uid':uid,'model':name,'status':'completed_optimizer_call','rows':rows,'optimizer_stats':{'iterations':16}})
    return values


def parent_protocol():
    return {'uids':list(ws.UIDS),'models':list(ws.MODELS),'max_steps':16,'max_seconds':3600,
        'snapshot_steps':[0,1,4,16],'threads':4,'interop_threads':1,'optimizer_seed_base':1493501,
        'evaluation_seed_bases':[1593501,1693501],'points_each_direction':8192}


def test_existing_optimizer_task_and_all_settings_reused():
    assert controls.ws.task is ws.task and controls.ws.optimize_layout is optimize_layout
    assert controls.NEW_MODELS==('original_A_10','early_v1_10')
    assert controls.ALL_MODELS==tuple(ws.MODELS)+controls.NEW_MODELS
    assert len(controls.verify_parent_state(parent_protocol(),{'attempts':attempts(ws.MODELS)}))==120
    assert ws.OPT_SEED_BASE==1493501 and ws.SEED_BASES==(1593501,1693501)


@pytest.mark.parametrize('alter',['omit','duplicate','misowned'])
def test_registered_slots_cannot_be_lost_duplicated_or_reassigned(alter):
    new=attempts(controls.NEW_MODELS)
    if alter=='omit':new[0]['rows'].pop()
    elif alter=='duplicate':new[1]=copy.deepcopy(new[0])
    else:new[0]['rows'][0],new[1]['rows'][0]=new[1]['rows'][0],new[0]['rows'][0]
    with pytest.raises(ValueError):controls.verify_attempt_slots(new,controls.NEW_MODELS)


def test_combined_summary_retains_failed_denominator_and_never_recomputes_old():
    old=attempts(ws.MODELS);new=attempts(controls.NEW_MODELS);before=copy.deepcopy(old)
    new[0]['rows'][-1]['status']='unavailable_runtime_failure';new[0]['rows'][-1].pop('qualified')
    result=controls.summarize_combined(old,new)
    assert old==before and result['task_count']==28 and result['primary_row_count']==168
    assert result['new_task_count']==8 and result['new_primary_row_count']==48
    assert result['new_missing_or_failed_rows']==1 and result['parent_outputs_recomputed'] is False
    assert len(result['paired_long_vs_new_controls'])==32
    group=next(x for x in result['groups'] if x['model']=='original_A_10')
    assert group['planned_parents']==4 and group['joint_vs_Fixed_opt16']==3
    assert sum(x['pair_available'] for x in result['paired_long_vs_new_controls'])==28
    assert result['pure_A_claim_permitted'] is False
    assert result['arbitrary_initializers_or_random_perturbation_mechanism_excluded'] is False


@pytest.mark.parametrize('alter',['field','reference','MC'])
def test_pairing_rejects_different_field_reference_or_MC(alter):
    old=attempts(ws.MODELS);new=attempts(controls.NEW_MODELS);row=new[0]['rows'][-1]
    if alter=='field':row['mainfield_hash']='different'
    elif alter=='reference':row['GT_reference_hash']='different'
    else:row['metrics'][0]['seed']+=1
    with pytest.raises(ValueError,match='Different'):controls.summarize_combined(old,new)


def test_early_model_is_exact_registered3000_model_only():
    state={'source_model_only':True,'source_stored_update':3000,'model_sha256':controls.EARLY_V1_HASH,
           'optimizer_rng_scheduler_sampler_from_source_omitted':True}
    controls.validate_early_metadata(state)
    for key,value in [('source_stored_update',3001),('model_sha256','0'*64),('optimizer',{})]:
        changed={**state,key:value}
        with pytest.raises(ValueError):controls.validate_early_metadata(changed)


def test_parent_budget_change_and_bound_file_change_are_rejected(tmp_path):
    p=parent_protocol();p['max_steps']=17
    with pytest.raises(ValueError,match='Parent fixed setting'):controls.verify_parent_state(p,{'attempts':attempts(ws.MODELS)})
    source=tmp_path/'frozen.txt';source.write_text('before');bound={str(source):file_hash(source)}
    controls.verify_bound(bound);source.write_text('after')
    with pytest.raises(ValueError,match='Frozen bound file changed'):controls.verify_bound(bound)


def test_initializer_failure_preserves_all_six_slots_without_optimizer(tmp_path):
    result=controls.failed_task(tmp_path,ws.UIDS[0],controls.NEW_MODELS[0],ValueError('guard failed'))
    assert result['optimizer_call_executed'] is False and result['optimizer_actual_steps']==0
    assert len(result['rows'])==6 and {r['kind'] for r in result['rows']}==set(ws.ROW_KINDS)
    assert all(r['status']=='unavailable_initializer_identity_replay_failure' for r in result['rows'])


def test_adapter_executes_exactly_one_setup_and_one_original_task_per_new_slot(tmp_path,monkeypatch):
    p={'uids':list(ws.UIDS),'models':list(controls.NEW_MODELS),'model_bindings':{name:{'fixed_identity':name} for name in controls.NEW_MODELS},
       'parent_run':str(tmp_path/'old'),'parent_results_sha256':'frozen'}
    (tmp_path/'RUN_PROTOCOL.json').write_text(json.dumps(p));old={'attempts':attempts(ws.MODELS)}
    monkeypatch.setattr(controls,'verify_protocol',lambda run:(p,old))
    monkeypatch.setattr(controls,'load_model',lambda protocol,name:name)
    monkeypatch.setattr(controls,'RadialSupport',lambda:object())
    monkeypatch.setattr(ws,'load_data',lambda protocol,uid:{'logr':np.zeros(3),'base_u':np.eye(3),'faces':np.array([[0,1,2]],np.int32)})
    predictions=[];tasks=[]
    mesh={'u':np.eye(3),'r':np.ones(3),'vertices':np.eye(3),'faces':np.array([[0,1,2]],np.int64),'alpha':np.ones(32)}
    def predict(model,*args):predictions.append(model);return mesh,{'generation_seconds':.01}
    monkeypatch.setattr(ws,'predict_raw',predict)
    fixtures={(a['uid'],a['model']):a for a in attempts(controls.NEW_MODELS)}
    def original_task_adapter(run,runtime,uid,name,model,support):
        tasks.append((uid,name));raw,_=predict(model)
        assert ws.verify_raw_initializer(raw,runtime['raw_initializer_references'][name][uid])['exact_source_replay']
        return copy.deepcopy(fixtures[uid,name])
    monkeypatch.setattr(ws,'task',original_task_adapter)
    monkeypatch.setattr(controls,'write_report',lambda *args:None)
    result=controls.run_study(tmp_path)
    assert len(tasks)==8 and len(set(tasks))==8 and len(predictions)==16
    assert {model for _,model in tasks}==set(controls.NEW_MODELS)
    assert result['new_task_count']==8 and (tmp_path/'COMPLETE.json').exists()
    with pytest.raises(FileExistsError):controls.run_study(tmp_path)
