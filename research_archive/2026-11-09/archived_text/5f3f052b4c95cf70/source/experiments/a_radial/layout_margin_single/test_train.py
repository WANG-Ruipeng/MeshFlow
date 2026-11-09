"""CPU paired-target identity, frozen-core semantics and exact resume contracts."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ..legacy.geometry_task import fibonacci,hull_faces
from ..legacy.networks import DomainVelocity,point_features
from .train import (config_digest,configure_cpu,file_sha256,model_sha256,read_json,
                    write_json,train_run,make_session,validate_state,step_sequence_hash)
from .smoke import smoke_run


def fixture_run(path):
    path.mkdir();(path/'initial').mkdir();configure_cpu(10)
    base=fibonacci(258);base/=np.linalg.norm(base,axis=1,keepdims=True)
    theta=.02;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    old=base@r.T;old/=np.linalg.norm(old,axis=1,keepdims=True)
    margin=(base+old)*.5;margin/=np.linalg.norm(margin,axis=1,keepdims=True)
    uids=['parent_0','parent_1','parent_2','parent_3'];proxies=np.stack([fibonacci(256)]*4)
    directions0=np.stack([base]*4);directions1_old=np.stack([old]*4);directions1_margin=np.stack([margin]*4)
    directions1_margin[1]=directions0[1]
    np.savez(path/'train_labels.npz',train_uids=uids,directions0=directions0,
        directions1_old=directions1_old,directions1_margin=directions1_margin,
        proxy_points=proxies,point_features=point_features(proxies),faces=hull_faces(base).astype(np.int32))
    weights=DomainVelocity().state_dict();model_hash=model_sha256(weights)
    torch.save({'model':weights,'model_sha256':model_hash,'source':'synthetic CPU fixture'},path/'initial/common_v1_model.pt')
    initial_sha=file_sha256(path/'initial/common_v1_model.pt')
    files={x.relative_to(path).as_posix():file_sha256(x) for x in path.rglob('*') if x.is_file()}
    write_json(path/'DATA_MANIFEST.json',{'files':files})
    jobs={}
    for i,uid in enumerate(uids):
        for target in ['old','margin']:
            jobs[target+'_'+uid]={'uid':uid,'index':i,'seed':10,'target_kind':target,
                'target_key':'directions1_'+target,'initial_checkpoint':'initial/common_v1_model.pt',
                'initial_checkpoint_sha256':initial_sha,'initial_model_sha256':model_hash,
                'start_update':0,'total_updates':8,'initialization_mode':'shared_model_fresh_cpu_AdamW',
                'pair_id':uid,'diagnostic_uids':[uid],'active_training_uids':[uid]}
    p={'scope':'SMOKE_ONLY','train_uids':uids,'internal_check_uids':['forbidden_check'],
        'endpoint_scale_squared':float(np.mean((directions0-directions1_old)**2)),
        'training':{'batch_size':4,'learning_rate':1e-4,'weight_decay':1e-4,'checkpoint_updates':[0,8],
            'device':'cpu','torch_threads':1,'interop_threads':1,'total_updates':8},
        'loss':{'name':'c0_unscaled_discrete_bridge_velocity_MSE','bridge_steps':32,
            'endpoint_scale_unused_by_c0':True},'jobs':jobs,
        'paths':{'training_labels':'train_labels.npz','data_manifest':'DATA_MANIFEST.json',
            'initial_checkpoint':'initial/common_v1_model.pt'},
        'hashes':{'training_labels_sha256':files['train_labels.npz'],
            'data_manifest_sha256':file_sha256(path/'DATA_MANIFEST.json'),
            'initial_checkpoint_sha256':initial_sha,'initial_model_sha256':model_hash},'source_sha256':{}}
    p['hashes']['config_sha256']=config_digest(p);write_json(path/'RUN_PROTOCOL.json',p)
    return path,p


def rebind_archive(run,p,arrays):
    np.savez(run/'train_labels.npz',**arrays)
    digest=file_sha256(run/'train_labels.npz');p['hashes']['training_labels_sha256']=digest
    manifest=read_json(run/'DATA_MANIFEST.json');manifest['files']['train_labels.npz']=digest
    (run/'DATA_MANIFEST.json').write_text(json.dumps(manifest))
    p['hashes']['data_manifest_sha256']=file_sha256(run/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p);(run/'RUN_PROTOCOL.json').write_text(json.dumps(p))


def test_all_eight_jobs_real_cpu_resume_manual_original_and_matched_times(tmp_path):
    run,p=fixture_run(tmp_path/'source');before=file_sha256(run/'RUN_PROTOCOL.json')
    result=smoke_run(run,tmp_path/'smoke')
    assert result['status']=='PASS' and result['actual_optimizer_updates']==144
    assert len(result['jobs'])==8 and result['paired_k_sequences_equal']
    assert result['full_semantic_state_bitwise_equal'] and result['manual_frozen_C0_next_bitwise_equal']
    assert result['cuda_rng_calls']==0 and result['cuda_context_initialized'] is False
    assert result['torch_threads']==result['interop_threads']==1
    assert result['jobs']['margin_parent_1']['whole']['zero_target'] is True
    assert result['jobs']['margin_parent_1']['whole']['relative_error_denominator_defined'] is False
    assert file_sha256(run/'RUN_PROTOCOL.json')==before and not (run/'training').exists()


def test_mainfield_archive_rejected_even_if_manifest_rebound(tmp_path):
    run,p=fixture_run(tmp_path/'source')
    with np.load(run/'train_labels.npz') as z:arrays={key:z[key].copy() for key in z.files}
    arrays['logr']=np.zeros((4,2050));rebind_archive(run,p,arrays)
    with pytest.raises(ValueError,match='Unexpected training input'):train_run(run,'old_parent_0')
    assert not (run/'training').exists()


def test_internal_check_parent_rejected_before_writes(tmp_path):
    run,p=fixture_run(tmp_path/'source');p['internal_check_uids']=[p['train_uids'][0]]
    (run/'RUN_PROTOCOL.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='Internal-check label leakage'):train_run(run,'old_parent_0')
    assert not (run/'training').exists()


def test_resume_other_target_is_rejected(tmp_path):
    run,p=fixture_run(tmp_path/'source')
    first=train_run(run,'old_parent_0',stop_after=3,output_dir=run/'first')
    with pytest.raises(ValueError,match='Resume config'):train_run(run,'margin_parent_0',first['checkpoint'],output_dir=run/'wrong')
    assert not (run/'wrong').exists()


@pytest.mark.parametrize('tamper',['cuda_rng','adam_counter','adam_lr','scheduler_counter','state_data','time_sequence'])
def test_resume_corruption_rejected(tmp_path,tamper):
    run,p=fixture_run(tmp_path/'source');first=train_run(run,'old_parent_0',stop_after=3,output_dir=run/'first')
    state=torch.load(first['checkpoint'],map_location='cpu',weights_only=False)
    if tamper=='cuda_rng':state['rng']['torch_cuda']=[]
    elif tamper=='adam_counter':next(iter(state['optimizer']['state'].values()))['step'].add_(1)
    elif tamper=='adam_lr':state['optimizer']['param_groups'][0]['lr']=2e-4
    elif tamper=='scheduler_counter':state['scheduler']['_step_count']+=1
    elif tamper=='state_data':state['training_labels_sha256']='0'*64
    else:state['trace'][0]['rng_matched_k'][0]=(state['trace'][0]['rng_matched_k'][0]+1)%32
    corrupt=run/'corrupt.pt';torch.save(state,corrupt)
    with pytest.raises(ValueError):train_run(run,'old_parent_0',corrupt,output_dir=run/'wrong')
    assert not (run/'wrong').exists()


def test_fresh_source_must_not_carry_old_adam_or_rng(tmp_path):
    run,p=fixture_run(tmp_path/'source');path=run/'initial/common_v1_model.pt'
    source=torch.load(path,map_location='cpu',weights_only=False);source['optimizer']={};torch.save(source,path)
    sha=file_sha256(path);p['hashes']['initial_checkpoint_sha256']=sha
    for job in p['jobs'].values():job['initial_checkpoint_sha256']=sha
    manifest=read_json(run/'DATA_MANIFEST.json');manifest['files']['initial/common_v1_model.pt']=sha
    (run/'DATA_MANIFEST.json').write_text(json.dumps(manifest));p['hashes']['data_manifest_sha256']=file_sha256(run/'DATA_MANIFEST.json')
    p['hashes']['config_sha256']=config_digest(p);(run/'RUN_PROTOCOL.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='inherited training state'):train_run(run,'old_parent_0')


def test_k_sequence_hash_is_order_sensitive_and_rejects_invalid_steps():
    a=[{'update':1,'rng_matched_k':[0,1,2,3]},{'update':2,'rng_matched_k':[4,5,6,7]}]
    b=copy.deepcopy(a);b[0]['rng_matched_k'].reverse()
    assert step_sequence_hash(a)!=step_sequence_hash(b)
    b[1]['rng_matched_k'][0]=32
    with pytest.raises(ValueError,match='Invalid bridge'):step_sequence_hash(b)
