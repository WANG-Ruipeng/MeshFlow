"""Registration boundaries and control identity; no live experiments or GPU."""
import copy
import json
from pathlib import Path
import pytest
import torch
from .test_prepare import control_fixture
from .train import config_digest
from .register_controls import CONTROL_MODELS,control_plan,load_registry,register_controls
from .evaluate import validate_control_checkpoint
from ..io_utils import file_hash
from ..legacy.networks import DomainVelocity
from ..layout_student.train import model_sha256


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def registry_fixture(tmp_path):
    run=tmp_path/'balanced';live=tmp_path/'longfit';run.mkdir();live.mkdir()
    cp,source,hashes,models=control_fixture()
    jobs=copy.deepcopy(cp['jobs']);manifest={}
    for name,job in jobs.items():
        rel=f'initial/{name}.pt';path=run/rel;path.parent.mkdir(exist_ok=True);path.write_bytes((name+' fullstate20k').encode())
        sha=file_hash(path);job.update(initial_checkpoint=rel,initial_checkpoint_sha256=sha)
        cp['jobs'][name]['initial_checkpoint_sha256']=sha
        source_final=live/'training'/name/'final.pt';source_final.parent.mkdir(parents=True)
        source_final.write_bytes((name+' frozen240k').encode())
        put(source_final.parent/'TRAIN_CONFIG.json',{'same':'config'})
        put(source_final.parent/'IMPORT_RECORD.json',{'same':'import'})
    put(run/'initial/source_protocol.json',source)
    put(live/'DATA_MANIFEST.json',{'files':manifest})
    cp['hashes'].update(config_sha256=config_digest(cp),data_manifest_sha256=file_hash(live/'DATA_MANIFEST.json'))
    put(live/'RUN_PROTOCOL.json',cp);put(run/'control_reference/control.json',cp)
    ch=file_hash(live/'RUN_PROTOCOL.json')
    plan={'source_run':str(live),'source_protocol_sha256':ch,'source_protocol_file':str(live/'RUN_PROTOCOL.json'),
        'copied_protocol_file':'control_reference/control.json','registry_file':'CONTROL_REGISTRY.json',
        'registration_exclusive_once':True,'run_protocol_never_rewritten':True,'registry_required_before_control_evaluation':True,'models':{}}
    for name in CONTROL_MODELS:
        seed=int(name.rsplit('_',1)[1]);job=f'joint_{seed}'
        plan['models'][name]={'job_id':job,'seed':seed,'update':240000,'required_arm':'c0','required_global_exposures_per_parent':80000,
            'source_checkpoint':str(live/'training'/job/'final.pt'),'local_checkpoint':f'controls/{name}_final.pt',
            'source_protocol_sha256':ch,'initial_checkpoint_sha256':jobs[job]['initial_checkpoint_sha256'],
            'initial_model_sha256':jobs[job]['initial_model_sha256']}
    p={**copy.deepcopy(source),'experiment':'A-BALANCED-v1','jobs':jobs,'training':copy.deepcopy(cp['training']),
        'paths':{'source_protocol':'initial/source_protocol.json','control_registry':'CONTROL_REGISTRY.json','unweighted_control_protocol':'control_reference/control.json'},
        'unweighted_controls':plan,
        'evaluation':{'unweighted_control_models':list(CONTROL_MODELS),'control_registry':'CONTROL_REGISTRY.json',
            'model_checkpoints':{n:b['local_checkpoint'] for n,b in plan['models'].items()},
            'model_seeds':{n:b['seed'] for n,b in plan['models'].items()}},
        'hashes':{'training_labels_sha256':'labels','source_protocol_sha256':file_hash(run/'initial/source_protocol.json'),
            'unweighted_control_protocol_sha256':ch}}
    p['hashes']['config_sha256']=config_digest(p);put(run/'RUN_PROTOCOL.json',p)
    def successful_audit(actual_run,actual_cp,actual_manifest):
        assert actual_run==live and actual_cp==cp and actual_manifest==manifest
        training={n:{'initial_source_sha256':j['initial_checkpoint_sha256'],'full_resumable_state_verified':True,
            'parent_trace_preserved_exactly':True} for n,j in jobs.items()}
        identities={n:{'checkpoint_sha256':file_hash(live/'training'/n/'final.pt'),'model_sha256':j['initial_model_sha256']} for n,j in jobs.items()}
        return training,identities,{}
    return run,live,p,cp,successful_audit


def test_control_plan_forbids_intermediate_or_changed_initial(tmp_path):
    run,live,p,_,_=registry_fixture(tmp_path);control_plan(run,p)
    bad=copy.deepcopy(p);bad['unweighted_controls']['models']['unweighted_joint_10']['source_checkpoint']=str(live/'training/joint_10/step_100000.pt')
    with pytest.raises(ValueError,match='substituted'):control_plan(run,bad)
    bad=copy.deepcopy(p);bad['unweighted_controls']['models']['unweighted_joint_10']['initial_checkpoint_sha256']='another'
    with pytest.raises(ValueError,match='different initial'):control_plan(run,bad)


def test_all_audits_pass_before_exclusive_writes(tmp_path,monkeypatch):
    run,live,p,cp,audit=registry_fixture(tmp_path)
    from ..layout_longfit import package_results as old
    def reject(*args):raise ValueError('incomplete full state')
    monkeypatch.setattr(old,'training_audit',reject)
    ph=file_hash(run/'RUN_PROTOCOL.json')
    with pytest.raises(ValueError,match='incomplete full state'):register_controls(run)
    assert not (run/'controls').exists() and not (run/'CONTROL_REGISTRY.json').exists()
    assert file_hash(run/'RUN_PROTOCOL.json')==ph
    monkeypatch.setattr(old,'training_audit',audit)
    value=register_controls(run)
    assert value['protocol_sha256']==ph and value['source20k_full_import_verified']
    assert file_hash(run/'RUN_PROTOCOL.json')==ph
    with pytest.raises(ValueError,match='exclusive once'):register_controls(run)
    # Consumers are portable: all control bytes and protocol are inside this run.
    (live/'RUN_PROTOCOL.json').unlink()
    for name in ('joint_10','joint_11'):(live/'training'/name/'final.pt').unlink()
    loaded,_=load_registry(run,p);assert loaded==value
    copied=run/p['evaluation']['model_checkpoints']['unweighted_joint_10'];copied.write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):load_registry(run,p)


def test_control_state_cannot_be_weighted_intermediate_or_another_protocol(tmp_path):
    run,live,p,cp,_=registry_fixture(tmp_path);name='unweighted_joint_10';binding=p['unweighted_controls']['models'][name]
    model=DomainVelocity().state_dict();mh=model_sha256(model)
    cfg={'total_updates':240000,'job':'joint_10','job_id':'joint_10','kind':'joint','arm':'c0','seed':10,
        'protocol_sha256':binding['source_protocol_sha256'],'config_sha256':cp['hashes']['config_sha256'],
        'active_training_uids':p['train_uids'],'train_uids':p['train_uids'],'network_extra_inputs':[],
        'initial_checkpoint_sha256':binding['initial_checkpoint_sha256'],'initial_model_sha256':binding['initial_model_sha256'],
        'training_labels_sha256':'labels','source_sha256':{}}
    state={'config':cfg,'model':model,'model_sha256':mh,'update':240000,'is_predeclared_final':True,'diagnostic_only':True,
        'job_id':'joint_10','protocol_sha256':binding['source_protocol_sha256'],
        'initial_checkpoint_sha256':binding['initial_checkpoint_sha256'],'training_labels_sha256':'labels',
        'sampler':{'count':12,'exposures':[80000]*12}}
    entry={'binding':binding,'model_sha256':mh};validate_control_checkpoint(state,name,p,cp,entry)
    for changed in ('weighted','budget','protocol','exposure','source'):
        bad=copy.deepcopy(state)
        if changed=='weighted':bad['config']['arm']='c0_parent_weighted'
        elif changed=='budget':bad['update']=100000
        elif changed=='protocol':bad['protocol_sha256']='another'
        elif changed=='exposure':bad['sampler']['exposures'][0]-=1
        else:bad['config']['source_sha256']={'legacy/networks.py':'wrong-sha'}
        with pytest.raises(ValueError):validate_control_checkpoint(bad,name,p,cp,entry)
