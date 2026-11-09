"""Pure-CPU state gates and dry orchestration; never launch a training process."""
from copy import deepcopy
import json
import os
import pickle
import random
import numpy as np
from pathlib import Path
import pytest
import torch
from . import overnight_queue as q


def completion():
    return {'status':'COMPLETE','protocol_sha256':'balanced_sha','driver_sha256':'driver_sha',
        'new_optimizer_updates_total':440000,'global_updates_sum_including_imported_parent_traces':480000,
        'jobs':[{'job':name,'returncode':0} for name in q.JOBS]}


def test_balanced_completion_requires_both_exit0_and_exact_new_and_global_budgets():
    value=completion();q.validate_completion_record(value,'balanced_sha','driver_sha')
    bad=[{**value,'new_optimizer_updates_total':480000},{**value,'global_updates_sum_including_imported_parent_traces':440000},
        {**value,'jobs':value['jobs'][:1]},{**value,'driver_sha256':'changed'},
        {**value,'jobs':[{'job':'joint_10','returncode':0},{'job':'joint_11','returncode':1}]},
        {**value,'jobs':[{'job':'joint_10','returncode':0}]*2}]
    for changed in bad:
        with pytest.raises(ValueError):q.validate_completion_record(changed,'balanced_sha','driver_sha')


class ChronologicalTrace:
    def __len__(self):return 240000
    def __iter__(self):return ({'update':i} for i in range(1,240001))


def state_fixture():
    uids=[f'parent{i}' for i in range(12)]
    config={'protocol_sha256':'balanced_sha','job_id':'joint_10','job':'joint_10','seed':10,
        'arm':'c0_parent_weighted','total_updates':240000,'start_update':20000,'active_training_uids':uids,
        'initial_checkpoint_sha256':'initial','training_labels_sha256':'labels'}
    model={'weight':torch.tensor([1.,2.])};mh=q.tensor_model_hash(model)
    state={'protocol_sha256':'balanced_sha','update':240000,'is_predeclared_final':True,'job_id':'joint_10','config':config,
        'initial_checkpoint_sha256':'initial','training_labels_sha256':'labels','model':model,'model_sha256':mh,
        'trace':ChronologicalTrace(),'scheduler':{'last_epoch':240000},
        'optimizer':{'state':{0:{'step':torch.tensor(240000.)}}},'rng':{'torch_cuda':[torch.zeros(1,dtype=torch.uint8)],
            'numpy':np.random.RandomState(1).get_state(),'python':random.Random(1).getstate(),
            'torch_cpu':torch.Generator().manual_seed(1).get_state()},
        'sampler':{'exposures':[80000]*12}}
    report={'status':'COMPLETE','job':'joint_10','seed':10,'update':240000,'final_budget':240000,
        'imported_parent_updates':20000,'new_optimizer_updates_since_import':220000,'protocol_sha256':'balanced_sha',
        'model_sha256':mh,'parent_exposures':{uid:80000 for uid in uids},'global_parent_exposures_total':960000}
    protocol={'train_uids':uids,'hashes':{'training_labels_sha256':'labels'},
        'jobs':{'joint_10':{'seed':10,'active_training_uids':uids,'initial_checkpoint_sha256':'initial'}}}
    return state,report,config,protocol


def test_final240k_state_checks_model_counters_sampler_protocol_and_parent_identity():
    state,report,config,protocol=state_fixture();q.validate_balanced_state(state,report,config,protocol,'joint_10','balanced_sha')
    for key,value in [('update',20000),('is_predeclared_final',False),('protocol_sha256','another'),('model_sha256','changed'),('job_id','joint_11')]:
        with pytest.raises(ValueError):q.validate_balanced_state({**state,key:value},report,config,protocol,'joint_10','balanced_sha')
    wrong={**state,'sampler':{'exposures':[79999]+[80000]*11}}
    with pytest.raises(ValueError,match='sampler'):q.validate_balanced_state(wrong,report,config,protocol,'joint_10','balanced_sha')
    wrong={**state,'scheduler':{'last_epoch':239999}}
    with pytest.raises(ValueError,match='counters'):q.validate_balanced_state(wrong,report,config,protocol,'joint_10','balanced_sha')
    changed={**report,'parent_exposures':{**report['parent_exposures'],'parent0':79999}}
    with pytest.raises(ValueError,match='exposure'):q.validate_balanced_state(state,changed,config,protocol,'joint_10','balanced_sha')
    assert not torch.cuda.is_initialized()


def test_failure_sentinel_wins_even_when_completion_sentinel_exists(tmp_path):
    q.put(tmp_path/'TRAINING_COMPLETE.json',completion());q.put(tmp_path/'training/joint_11/FAILED.json',{'error':'saved original failure'})
    observed=q.balanced_status(tmp_path)
    assert observed['status']=='FAILED' and observed['failures'][0]['record']['error']=='saved original failure'


def test_waiting_status_reports_actual_log_progress_and_saved_checkpoint(tmp_path):
    root=tmp_path/'training/joint_10';root.mkdir(parents=True);(root/'step_50000.pt').write_bytes(b'not loaded while waiting')
    log=tmp_path/'training_logs/joint_10.log';log.parent.mkdir();log.write_text('{"job":"joint_10","update":57000,"loss":0.123}\npartial trailing output')
    observed=q.balanced_status(tmp_path)
    assert observed['status']=='WAITING' and observed['jobs']['joint_10']['last_logged']['update']==57000
    assert observed['jobs']['joint_10']['last_saved_intermediate_update']==50000
    assert not observed['jobs']['joint_10']['final_checkpoint_exists']


def test_wait_loop_emits30second_heartbeats_and_requires_final_evidence_gate():
    now=[0.];events=[];checks=[];states=iter([{'status':'WAITING'},{'status':'WAITING'},{'status':'COMPLETION_PRESENT_REQUIRES_VALIDATION'}])
    plan={'balanced_run':'unused','balanced_wait_timeout_seconds':10800,'heartbeat_seconds':30}
    result=q.wait_for_balanced(plan,lambda s,d:events.append((s,d)),lambda _:next(states),
        lambda p:checks.append(p) or {'status':'PASS'},lambda:now[0],lambda seconds:now.__setitem__(0,now[0]+seconds))
    assert result['status']=='PASS' and len(checks)==1
    assert [x[1]['elapsed_wait_seconds'] for x in events]==[0.,30.,60.]


def test_wait_timeout_and_failure_never_launch_or_call_completion_checker():
    def prohibited(*args):pytest.fail('Cannot validate/launch after missing completion or failure')
    plan={'balanced_run':'unused','balanced_wait_timeout_seconds':70,'heartbeat_seconds':30};now=[0.];events=[]
    with pytest.raises(TimeoutError):q.wait_for_balanced(plan,lambda s,d:events.append(d),lambda _:{'status':'WAITING'},prohibited,
        lambda:now[0],lambda seconds:now.__setitem__(0,now[0]+seconds))
    assert [e['elapsed_wait_seconds'] for e in events]==[0.,30.,60.,70.]
    with pytest.raises(ValueError,match='failure sentinel'):q.wait_for_balanced(plan,lambda *a:None,
        lambda _:{'status':'FAILED','failures':['original FAILED.json']},prohibited,lambda:0.,prohibited)


def fake_repository(tmp_path,monkeypatch):
    names=list(q.code_files(q.REPOSITORY));repo=tmp_path/'repo';repo.mkdir()
    for name in names:
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixed source '+name)
    balanced=repo/q.BALANCED_REL;formal=repo/q.FORMAL_REL;balanced.mkdir(parents=True);formal.mkdir(parents=True)
    bp={'jobs':{name:{'seed':seed,'start_update':20000,'total_updates':240000,'new_optimizer_updates':220000} for name,seed in zip(q.JOBS,(10,11))},
        'training':{'updates':240000}}
    q.put(balanced/'RUN_PROTOCOL.json',bp);bsha=q.sha(balanced/'RUN_PROTOCOL.json')
    gate=repo/'output/registered_gate';gate.mkdir();fp={'seeds':[10,11],'formal_updates':50000,'primary_weight':'ema','gate_output':str(gate)}
    q.put(formal/'RUN_PROTOCOL.json',fp);fsha=q.sha(formal/'RUN_PROTOCOL.json')
    q.put(formal/'SAMPLING_PROTOCOL.json',{'fixed':'sampling'});ssha=q.sha(formal/'SAMPLING_PROTOCOL.json')
    q.put(gate/'COMPLETE.json',{'status':'PASS','formal_protocol_sha256':fsha})
    q.put(balanced/'TRAINING_GATE.json',{'status':'APPROVED_BALANCED_OBJECTIVE_FIXED_BUDGET','protocol_sha256':bsha,'training_source_hashes':{}})
    for name in ['SOURCE_MANIFEST.json','CONFIG.json','FORMAL_CONFIG.yaml','SAMPLING_CPU_CHECKS.json']:(formal/name).write_text('{}')
    monkeypatch.setattr(q,'REPOSITORY',repo);monkeypatch.setattr(q,'BALANCED_SHA',bsha);monkeypatch.setattr(q,'FORMAL_SHA',fsha);monkeypatch.setattr(q,'SAMPLING_SHA',ssha)
    monkeypatch.setattr(q.subprocess,'Popen',lambda *a,**kw:pytest.fail('CPU tests cannot launch child processes'))
    return repo,balanced,formal


def test_prepare_only_creates_exclusive_plan_and_source_change_or_other_run_is_rejected(tmp_path,monkeypatch):
    repo,balanced,formal=fake_repository(tmp_path,monkeypatch);queue=repo/'output/queue'
    plan=q.prepare(queue);assert q.load_plan(queue)==plan
    assert q.read(queue/'STATUS.json')['status']=='PREPARED_NOT_STARTED' and not (formal/'OVERNIGHT_QUEUE_CLAIM.json').exists()
    with pytest.raises(FileExistsError):q.prepare(queue)
    bad={**plan,'official_run':str(repo/'output/other')};q.put(queue/'PLAN.json',bad,replace=True)
    with pytest.raises(ValueError,match='only the two'):q.load_plan(queue)
    q.put(queue/'PLAN.json',plan,replace=True)
    (repo/'experiments/a_radial/system_formal/finish.py').write_text('changed source')
    with pytest.raises(ValueError,match='source changed'):q.load_plan(queue)
    assert not torch.cuda.is_initialized()


def test_timeout_writes_failure_preserves_claim_and_refuses_automatic_restart(tmp_path,monkeypatch):
    repo,balanced,formal=fake_repository(tmp_path,monkeypatch);queue=repo/'output/queue';q.prepare(queue)
    def timeout(plan,emit):emit('WAIT_BALANCED240K',{'elapsed_wait_seconds':10800,'balanced':{'status':'WAITING'}});raise TimeoutError('registered timeout')
    monkeypatch.setattr(q,'wait_for_balanced',timeout)
    monkeypatch.setattr(q,'run_process',lambda *a:pytest.fail('Timeout must not launch official training'))
    with pytest.raises(TimeoutError):q.run(queue)
    assert q.read(queue/'FAILED.json')['stage']=='WAIT_BALANCED240K' and (formal/'OVERNIGHT_QUEUE_CLAIM.json').exists()
    with pytest.raises(ValueError,match='already attempted'):q.run(queue)


def test_dry_dispatch_orders_official_then_finish_and_keeps_unqualified_status(tmp_path,monkeypatch):
    repo,balanced,formal=fake_repository(tmp_path,monkeypatch);queue=repo/'output/queue';q.prepare(queue)
    monkeypatch.setattr(q,'wait_for_balanced',lambda plan,emit:{'status':'PASS','new_optimizer_updates_total':440000})
    monkeypatch.setattr(q,'formal_queue_evidence',lambda plan:{'status':'PASS','scientific_qualification':False})
    calls=[]
    def fake_process(queue,plan,stage,module,emit):
        calls.append(module);emit(stage,{'dry_test':True})
        if stage=='FROZEN_FINISH':q.put(formal/'posttraining/FINISH_COMPLETE.json',{
            'status':'FIXED_BUDGET_SAMPLING_AND_MAIN_TABLE_DONE_UNQUALIFIED','training_protocol_sha256':q.FORMAL_SHA,'sampling_protocol_sha256':q.SAMPLING_SHA})
        return {'stage':stage,'returncode':0}
    monkeypatch.setattr(q,'run_process',fake_process)
    result=q.run(queue)
    assert calls==['experiments.a_radial.system_formal.run_all','experiments.a_radial.system_formal.finish']
    assert result['status']=='COMPLETE_UNQUALIFIED' and not result['scientific_convergence_or_generalization_claim']
    assert not torch.cuda.is_initialized()


def test_formal_failure_prevents_finish_dispatch_and_is_recorded(tmp_path,monkeypatch):
    repo,balanced,formal=fake_repository(tmp_path,monkeypatch);queue=repo/'output/queue';q.prepare(queue)
    monkeypatch.setattr(q,'wait_for_balanced',lambda plan,emit:{'status':'PASS'})
    calls=[]
    def failed(queue,plan,stage,module,emit):calls.append(module);emit(stage,{});raise RuntimeError('returncode1')
    monkeypatch.setattr(q,'run_process',failed)
    with pytest.raises(RuntimeError,match='returncode1'):q.run(queue)
    assert calls==['experiments.a_radial.system_formal.run_all']
    assert q.read(queue/'FAILED.json')['stage']=='OFFICIAL_RUN_ALL_2X50000'


def test_prelaunch_hash_failure_is_explicit_without_creating_formal_claim(tmp_path,monkeypatch):
    repo,balanced,formal=fake_repository(tmp_path,monkeypatch);queue=repo/'output/queue';q.prepare(queue)
    (repo/'experiments/a_radial/system_formal/run_all.py').write_text('changed after prepared')
    with pytest.raises(ValueError,match='source changed'):q.run(queue)
    assert q.read(queue/'FAILED.json')['stage']=='PRELAUNCH_VALIDATION'
    assert not (formal/'OVERNIGHT_QUEUE_CLAIM.json').exists()



def test_numpy_rng_safe_loader_preserves_original_array_without_allowing_arbitrary_globals(tmp_path):
    value={'rng':{'numpy':np.random.RandomState(372).get_state()},'weight':torch.tensor([1.,2.])}
    path=tmp_path/'numpy_rng.pt';torch.save(value,path)
    before=list(torch.serialization.get_safe_globals())
    loaded=q.load_balanced_checkpoint_cpu(path)
    assert torch.equal(loaded['weight'],value['weight'])
    assert loaded['rng']['numpy'][0]==value['rng']['numpy'][0]
    assert np.array_equal(loaded['rng']['numpy'][1],value['rng']['numpy'][1])
    assert loaded['rng']['numpy'][1].dtype==np.uint32
    assert loaded['rng']['numpy'][2:]==value['rng']['numpy'][2:]
    assert list(torch.serialization.get_safe_globals())==before
    class Unregistered:
        def __reduce__(self):return (os.getcwd,())
    bad=tmp_path/'unregistered.pt';torch.save({'payload':Unregistered()},bad)
    with pytest.raises(pickle.UnpicklingError):q.load_balanced_checkpoint_cpu(bad)
    assert list(torch.serialization.get_safe_globals())==before
    assert not torch.cuda.is_initialized()
