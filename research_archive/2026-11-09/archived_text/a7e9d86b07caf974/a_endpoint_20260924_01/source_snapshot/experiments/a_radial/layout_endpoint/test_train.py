"""Endpoint-training correctness; fixtures do not inspect development outcomes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from ..exporter import ASSETS,RadialSupport
from ..legacy.geometry_task import fibonacci,hull_faces
from ..legacy.networks import point_features
from ..layout_student.train import file_sha256,model_sha256
from ..layout_value.optimizer import _np_quality
from .smoke import exact_same
from .train import load_train_archive,quality_penalty,train_run


def _write_json(path,value):
    path.write_text(json.dumps(value))


def fixture_run(path,updates=2):
    path.mkdir();(path/'initial').mkdir();support=RadialSupport()
    base=fibonacci(258);faces=hull_faces(base).astype(np.int32);uids=[f'train_{i}' for i in range(4)]
    rotation=np.array([[np.cos(.03),-np.sin(.03),0],[np.sin(.03),np.cos(.03),0],[0,0,1.]])
    teacher=base@rotation.T;teacher/=np.linalg.norm(teacher,axis=1,keepdims=True)
    logr=np.zeros((4,2050));proxy_u=fibonacci(256)
    proxy=proxy_u*support.query(logr[0],proxy_u)[:,None]
    quality=_np_quality(base*support.query(logr[0],base)[:,None],faces)
    u0=np.stack([base]*4);u1=np.stack([teacher]*4)
    proxies=np.stack([proxy]*4)
    np.savez(path/'train_labels.npz',train_uids=uids,directions0=u0,directions1=u1,
        proxy_points=proxies,point_features=point_features(proxies),logr=logr,faces=faces,
        quality_floors_deg=np.tile([quality['min_angle_deg'],quality['angle_q05_deg']],(4,1)))
    weights=torch.load(ASSETS/'original_A_10.pt',map_location='cpu',weights_only=True)
    torch.save({'model':weights,'model_sha256':model_sha256(weights)},path/'initial/seed_10.pt')
    files={p.relative_to(path).as_posix():file_sha256(p) for p in path.rglob('*') if p.is_file()}
    _write_json(path/'DATA_MANIFEST.json',{'files':files})
    scale=float(np.mean((u1-u0)**2))
    training={'arms':['c0','c1','c2'],'seeds':[10],'updates':updates,'batch_size':4,
        'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':1e-4,'rollout_steps':32}
    loss={'endpoint_scale_squared':scale,'quality_scale_deg':.1,'weights':{'endpoint':1.,'min_angle':1.,'p05_angle':1.}}
    protocol={'train_uids':uids,'internal_check_uids':['forbidden_check'],
        'paths':{'training_labels':'train_labels.npz','initial_checkpoints':{'10':'initial/seed_10.pt'}},
        'training':training,'loss':loss,'endpoint_scale_squared':scale,
        'hashes':{'training_labels_sha256':files['train_labels.npz'],
            'data_manifest_sha256':file_sha256(path/'DATA_MANIFEST.json'),
            'initial_checkpoint_sha256':{'10':files['initial/seed_10.pt']},
            'initial_model_sha256':{'10':model_sha256(weights)},
            'config_sha256':hashlib.sha256(json.dumps({'training':training,'loss':loss},sort_keys=True,separators=(',',':')).encode()).hexdigest()}}
    _write_json(path/'RUN_PROTOCOL.json',protocol)
    return path,protocol


def test_quality_penalty_uses_degrees_exact_fixed_scale_and_backprop():
    u=torch.eye(3,dtype=torch.float64)[None].requires_grad_(True)
    f=torch.tensor([[0,1,2]],dtype=torch.long)
    class IdentityField:
        def vertices(self,q):return q
    fields=[IdentityField()]
    from ..layout_value.optimizer import torch_face_quality
    actual=torch.stack(torch_face_quality(u[0],f)).detach()[None]
    zero,_,_=quality_penalty(u,[0],fields,f,actual)
    assert zero.item()==0.
    value,_,deficit=quality_penalty(u,[0],fields,f,actual+torch.tensor([[.1,.2]],dtype=torch.float64))
    assert value.item()==pytest.approx(5.,abs=1e-10)
    assert torch.allclose(deficit,torch.tensor([[.1,.2]],dtype=torch.float64),atol=1e-12,rtol=0)
    value.backward();assert torch.isfinite(u.grad).all() and torch.count_nonzero(u.grad)>0


@pytest.mark.parametrize('arm',['c0','c1','c2'])
def test_actual_cpu_training_resume_and_train_only_reads(tmp_path,monkeypatch,arm):
    run,protocol=fixture_run(tmp_path/'run')
    load=np.load;opened=[]
    def whitelist(path,*args,**kwargs):
        assert Path(path).resolve() in {(run/'train_labels.npz').resolve(),(ASSETS/'support.npz').resolve()}
        opened.append(str(path));return load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',whitelist)
    whole=train_run(run,arm,10,2,device='cpu',checkpoint_every=1,output_dir=run/'whole')
    first=train_run(run,arm,10,2,device='cpu',checkpoint_every=1,stop_after=1,output_dir=run/'first')
    resumed=train_run(run,arm,10,2,Path(first['checkpoint']),device='cpu',checkpoint_every=1,output_dir=run/'resumed')
    a=torch.load(whole['checkpoint'],map_location='cpu',weights_only=False)
    b=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=False)
    for key in ['model','optimizer','scheduler','rng','sampler','config','trace']:exact_same(a[key],b[key])
    assert a['model_sha256']==b['model_sha256']
    assert a['config']['optimizer_initialization']=='fresh_empty_no_v1_Adam_inheritance'
    assert a['config']['mainfield_network_input'] is False
    assert a['config']['parameter_count']==29907
    assert all(x==2 for x in whole['parent_exposures'].values())
    assert whole['internal_check_teacher_files_opened']==0
    assert sum(Path(p).name=='train_labels.npz' for p in opened)==3
    assert a['is_predeclared_final'] and not a['diagnostic_only']
    if arm!='c0':assert all(row['rollout_steps']==32 and len(row['alphas'])==4 for row in a['trace'])


def test_archive_rejects_check_leakage_and_changed_training_scale(tmp_path):
    run,protocol=fixture_run(tmp_path/'run')
    protocol['internal_check_uids']=['train_0']
    with pytest.raises(ValueError,match='leakage'):load_train_archive(run,protocol)
    protocol['internal_check_uids']=['forbidden_check'];protocol['endpoint_scale_squared']*=2
    with pytest.raises(ValueError,match='scale differs'):load_train_archive(run,protocol)


def test_same_balanced_parent_and_time_sequence_across_arms(tmp_path):
    run,_=fixture_run(tmp_path/'run',updates=1)
    states=[]
    for arm in ['c0','c1','c2']:
        report=train_run(run,arm,10,1,device='cpu',checkpoint_every=1,output_dir=run/arm)
        states.append(torch.load(report['checkpoint'],map_location='cpu',weights_only=False))
    for state in states[1:]:
        assert state['trace'][0]['uids']==states[0]['trace'][0]['uids']
        assert state['trace'][0]['rng_matched_k']==states[0]['trace'][0]['rng_matched_k']
        exact_same(state['rng'],states[0]['rng']);exact_same(state['sampler'],states[0]['sampler'])
