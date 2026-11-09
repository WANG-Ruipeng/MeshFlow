"""Light CPU contract tests; actual CUDA numerical gates live in gate.py."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import textwrap
import numpy as np
import pytest
import torch
import yaml
from ..system_probe.official import Session,validate_config,exact_same
from ..layout_student.train import model_sha256
from .train import ParallelSession,FORMAL_SCOPE,PROBE_SCOPE,BOUNDARY,validate_checkpoint,validate_final_budget,dataset_manifest,new_invocation


def test_update_core_changes_only_the_loader_call():
    original=ast.parse(textwrap.dedent(inspect.getsource(Session.update)))
    actual=ast.parse(textwrap.dedent(inspect.getsource(ParallelSession._update_core)))
    original.body[0].name='_update_core'
    class ReplaceLoader(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='collate_fn':
                self.count+=1
                return ast.parse('b,_ot_audit=load_microbatch(self.ds,ids,self.pool)').body[0]
            return self.generic_visit(node)
    replacement=ReplaceLoader();original=replacement.visit(original)
    assert replacement.count==1
    assert ast.dump(original,include_attributes=False)==ast.dump(actual,include_attributes=False)


def state_fixture(scope=FORMAL_SCOPE,step=4):
    model={'layer.weight':torch.tensor([[1.,2.],[3.,4.]])};ema={'layer.weight':torch.tensor([[.1,.2],[.3,.4]])}
    if step==0:ema=deepcopy(model)
    return {'identity':{'protocol_sha256':'bound_protocol','seed':10,'data_manifest_sha256':'bound_data','source_manifest_sha256':'bound_source'},'config':{'frozen':'config'},'scope':scope,'boundary':deepcopy(BOUNDARY),
        'step':step,'ema_updates':step,'scheduler':{'last_epoch':step},'parent_exposures':torch.full((16,),step,dtype=torch.int64),
        'trace':[{'step':i,'ema_updates':i,'scheduler_updates':i} for i in range(1,step+1)],
        'optimizer':{'state':{} if step==0 else {0:{'step':torch.tensor(float(step))}}},
        'fresh_seed':10,'formal_budget':50000,'initialization':'fresh_original_model_no_probe_or_pretrained_checkpoint',
        'timing_trace':[{'step':i} for i in range(1,step+1)],'is_predeclared_final':validate_final_budget(step,scope),
        'model':model,'ema':ema,'model_sha256':model_sha256(model),'ema_sha256':model_sha256(ema)}


def test_checkpoint_scope_prevents_importing_probe_weights():
    state=state_fixture(PROBE_SCOPE)
    with pytest.raises(AssertionError,match='probe checkpoint'):
        validate_checkpoint(state,state['identity'],state['config'],FORMAL_SCOPE)


@pytest.mark.parametrize('change',['pending_OT','parent_exposure','EMA_count','Adam_count','trace','model_bytes','EMA_bytes','final_marker'])
def test_checkpoint_rejects_partial_or_stale_state(change):
    state=state_fixture();identity=deepcopy(state['identity']);cfg=deepcopy(state['config'])
    validate_checkpoint(state,identity,cfg,FORMAL_SCOPE)
    if change=='pending_OT':state['boundary']['pending_OT_futures']=1
    elif change=='parent_exposure':state['parent_exposures'][0]-=1
    elif change=='EMA_count':state['ema_updates']-=1
    elif change=='Adam_count':state['optimizer']['state'][0]['step']-=1
    elif change=='trace':state['trace'].pop(0)
    elif change=='model_bytes':state['model']['layer.weight'][0,0]+=1
    elif change=='EMA_bytes':state['ema']['layer.weight'][0,0]+=1
    elif change=='final_marker':state['is_predeclared_final']=True
    with pytest.raises(AssertionError):validate_checkpoint(state,identity,cfg,FORMAL_SCOPE)


def test_incomplete_update_cannot_be_saved():
    session=object.__new__(ParallelSession);session.safe_boundary=False
    with pytest.raises(AssertionError,match='partial-update checkpoint'):session.state()


def test_fixed_budget_and_final_selection_are_not_changed_for_probe():
    path=Path(__file__).resolve().parents[1]/'configs/meshflow_closed16_512.yaml'
    cfg=yaml.safe_load(path.read_text());validate_config(cfg)
    assert not validate_final_budget(100,PROBE_SCOPE)
    assert not validate_final_budget(49999,FORMAL_SCOPE)
    assert validate_final_budget(50000,FORMAL_SCOPE)
    assert not validate_final_budget(50000,PROBE_SCOPE)
    changed=deepcopy(cfg);changed['closed16_runner']['updates']=5000
    with pytest.raises(AssertionError):validate_config(changed)


def test_dataset_manifest_binds_all19_exact_files_and_detects_changed_bytes(tmp_path):
    uids=[f'uid_{i:02}' for i in range(16)]
    (tmp_path/'DATASET.json').write_text(json.dumps({'rows':[{'uid':uid} for uid in reversed(uids)]}))
    folder=tmp_path/'meshflow512';meshes=folder/'objaverse_occ_v5_ids';meshes.mkdir(parents=True)
    (folder/'split').mkdir()
    for name in ['train.npz','test.npz']:(folder/'split'/name).write_bytes(name.encode())
    for uid in uids:(meshes/(uid+'.npz')).write_bytes(uid.encode())
    before=dataset_manifest(tmp_path);assert before['parent_uids']==uids and len(before['files'])==19
    target=meshes/(uids[0]+'.npz');target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(AssertionError):exact_same(before,dataset_manifest(tmp_path))


def test_resume_invocation_is_new_and_preserves_old_logs(tmp_path):
    first=new_invocation(tmp_path);(first/'FAILED.json').write_text('original failure')
    second=new_invocation(tmp_path)
    assert first.name=='0001' and second.name=='0002'
    assert (first/'FAILED.json').read_text()=='original failure'


def test_step0_fresh_checkpoint_has_empty_adam_and_valid_zero_counters():
    state=state_fixture(step=0)
    validate_checkpoint(state,state['identity'],state['config'],FORMAL_SCOPE)
    assert state['optimizer']['state']=={} and state['trace']==[] and not state['is_predeclared_final']
    assert not state['parent_exposures'].any() and state['model_sha256']==state['ema_sha256']
    state['optimizer']['state'][0]={'step':torch.tensor(0.)}
    with pytest.raises(AssertionError,match='Adam state initialization count'):
        validate_checkpoint(state,state['identity'],state['config'],FORMAL_SCOPE)


@pytest.mark.parametrize('key',['seed','data_manifest_sha256','source_manifest_sha256','protocol_sha256'])
def test_step0_refuses_wrong_seed_data_source_or_protocol(key):
    state=state_fixture(step=0);expected=deepcopy(state['identity'])
    expected[key]=11 if key=='seed' else 'other_bytes'
    with pytest.raises(AssertionError,match='resume_identity'):
        validate_checkpoint(state,expected,state['config'],FORMAL_SCOPE)


def test_real_official_split_locations_and_all16_file_hashes():
    repo=Path(__file__).resolve().parents[3]
    data=repo/'downloaded_data/layout_closed16_512_20260924'
    result=dataset_manifest(data)
    assert len(result['parent_uids'])==16 and len(result['files'])==19
    assert 'meshflow512/split/train.npz' in result['files']
    assert 'meshflow512/split/test.npz' in result['files']
