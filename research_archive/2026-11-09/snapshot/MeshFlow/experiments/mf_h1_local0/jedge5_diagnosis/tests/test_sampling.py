"""CPU-only mock integration contracts; no actual MeshFlow network is constructed."""
import json
import time
import numpy as np
import torch
from torch import nn
from ..sampling import sample, new_counts
from ..stages_ab import plan_A,plan_B


class _Attention(nn.Module):
    def __init__(self):
        super().__init__();self.qkv_proj=nn.Identity();self.out_proj=nn.Identity()
    def forward(self,x):
        return self.out_proj(self.qkv_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__();self.attn=_Attention();self.mlp=nn.Identity()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.x_embedder=nn.Identity();self.layers=nn.ModuleList([_Block()])


class _Mock(nn.Module):
    calls=0
    def __init__(self,mode='normal'):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(1),requires_grad=False)
        self.backbone=_Backbone();self.experiment_trainable=False;self.mode=mode;self.times=[];self.eval()
    def forward(self,x,t,y,valid,known):
        type(self).calls+=1;self.times.append(t.clone())
        assert y.tolist()==[112] and valid.all() and known.sum()==12
        x=self.backbone.x_embedder(x)
        x=self.backbone.layers[0].attn(x)
        x=self.backbone.layers[0].mlp(x)
        v=x*.125+t[:,None,None]+.03
        if self.mode=='nonfinite':v=v.clone();v[0,12,0]=float('nan')
        if self.mode=='wrong_dtype':v=v.to(torch.bfloat16)
        return v


def _inputs():
    rng=np.random.RandomState(73)
    z=torch.from_numpy(rng.normal(size=(1,112,9)).astype(np.float32))
    C=torch.from_numpy(rng.normal(size=(12,3,3)).astype(np.float32));C[0,0,0]=-0.0
    return z,C


def run_cpu_tests():
    assert not torch.cuda.is_initialized()
    before=_Mock.calls;start=time.perf_counter();rows=[];expected_rejections=0
    seeds=(10401,10402)
    A=plan_A('B',seeds)+plan_A('J',seeds);B=plan_B('B',seeds)+plan_B('J',seeds)
    assert len(A)==168 and sum(x['nfe'] for x in A)==13600
    assert len(B)==78 and sum(100-x['start_step'] for x in B)==3900
    assert sum(x['same_model_replay'] for x in B)==6
    assert len({tuple(sorted(x.items())) for x in A})==168
    rows.append(dict(name='exact_A_B_matrices_and_budget',status='PASS',A_full=168,A_forwards=13600,B_suffix=78,B_forwards=3900))
    z,C=_inputs();z0=z.clone();c0=C.clone();m=_Mock();counts=new_counts()
    raw,path,audit=sample(m,z,C,nfe=100,precision='fp32',counts=counts,cpu_mock=True)
    assert path.shape==(101,112,9) and counts['complete_rollouts']==1 and counts['model_forward_attempts']==100
    for cut in (20,50,80):
        init=torch.from_numpy(path[cut][None].copy());saved=init.clone()
        out,suffix,sa=sample(m,z,C,nfe=100,precision='fp32',counts=counts,initial_state=init,start_step=cut,cpu_mock=True)
        assert np.array_equal(suffix.view(np.uint32),path[cut:].view(np.uint32))
        assert torch.equal(init.view(torch.int32),saved.view(torch.int32))
        assert sa['absolute_time_values_FP32'][0]==float(np.float32(cut/100))
        assert len(sa['absolute_time_values_FP32'])==100-cut
    assert counts['model_forward_attempts']==250 and counts['suffix_returns']==3
    assert torch.equal(z.view(torch.int32),z0.view(torch.int32)) and torch.equal(C.view(torch.int32),c0.view(torch.int32))
    assert np.all(path[:,0,0].view(np.uint32)==np.uint32(0x80000000))
    rows.append(dict(name='absolute_suffix_all_states_C_signedzero_input_immutability',status='PASS',mock_forwards=250))
    m=_Mock();c=new_counts();raw50,path50,_=sample(m,z,C,nfe=50,precision='bf16',counts=c,cpu_mock=True)
    expected=z.clone();expected[:,:12]=C.reshape(1,12,9);known=torch.zeros((1,112),dtype=torch.bool);known[:,:12]=True
    with torch.no_grad():
        for k in range(50):
            v=m(expected,torch.full((1,),k/50,dtype=torch.float32),torch.tensor([112]),torch.ones_like(known),known).float()
            nxt=expected.clone();nxt[:,12:]=expected[:,12:]+v[:,12:]*(1/50);nxt[:,:12]=C.reshape(1,12,9);expected=nxt
    assert np.array_equal(raw50.view(np.uint32),expected[0].numpy().view(np.uint32))
    rows.append(dict(name='Euler50_arithmetic_against_direct_CPU_reference',status='PASS',mock_forwards=100,
                     limitation='No claim of real BF16 equality; production CUDA replay is separately budgeted'))
    m=_Mock();_,path200,a=sample(m,z,C,nfe=200,precision='fp32',cpu_mock=True)
    assert path200.shape==(201,112,9) and a['state_indices'][40]==40 and a['state_indices'][100]==100 and a['state_indices'][160]==160
    assert a['dt']==1/200 and a['absolute_time_values_FP32'][199]==float(np.float32(199/200))
    rows.append(dict(name='Euler200_direct_FP32_times_and_midpoints',status='PASS',mock_forwards=200))
    m=_Mock();_,on,_=sample(m,z,C,nfe=50,precision='fp32',cpu_mock=True)
    _,off,_=sample(m,z,C,nfe=50,precision='fp32',cpu_mock=True,observe_dtypes=False)
    assert np.array_equal(on.view(np.uint32),off.view(np.uint32))
    assert not any(x._forward_hooks or x._forward_pre_hooks for x in m.modules())
    rows.append(dict(name='observation_nonintrusive_and_hooks_removed',status='PASS',mock_forwards=100))
    for change in ('C_sign','bad_start','missing_state','wrong_NFE','wrong_precision'):
        m=_Mock();c=new_counts();init=torch.from_numpy(path[20][None].copy());kwargs=dict(nfe=100,precision='fp32',counts=c,initial_state=init,start_step=20,cpu_mock=True)
        if change=='C_sign':init[0,0,0]=0.0
        if change=='bad_start':kwargs['start_step']=100
        if change=='missing_state':kwargs['initial_state']=None
        if change=='wrong_NFE':kwargs['nfe']=75
        if change=='wrong_precision':kwargs['precision']='fp16'
        try:sample(m,z,C,**kwargs)
        except (ValueError,RuntimeError):expected_rejections+=1
        else:raise AssertionError('Invalid request accepted: '+change)
        assert c['model_forward_attempts']==0
    rows.append(dict(name='invalid_suffix_and_unregistered_settings_fail_before_forward',status='PASS',expected_rejections=5))
    for mode in ('nonfinite','wrong_dtype'):
        m=_Mock(mode);c=new_counts()
        try:sample(m,z,C,nfe=50,precision='fp32',counts=c,cpu_mock=True)
        except RuntimeError as exc:
            expected_rejections+=1;assert exc.partial_path.shape==(1,112,9)
        else:raise AssertionError('Numerical failure was accepted')
        assert c['model_forward_attempts']==c['model_forward_returns']==1 and c['sampling_failures']==1
        assert not any(x._forward_hooks or x._forward_pre_hooks for x in m.modules())
    rows.append(dict(name='finite_and_FP32_dtype_failures_preserve_partial_and_counts',status='PASS',mock_forwards=2,expected_rejections=2))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',tests_run=len(rows),tests=rows,seconds=time.perf_counter()-start,
                counts=dict(CPU_mock_forward_calls=_Mock.calls-before,actual_MeshFlow_models_constructed=0,
                            actual_model_forwards=0,model_backwards=0,leaf_autograd_calls=0,OT_calls=0,
                            optimizer_updates=0,expected_rejected_requests=expected_rejections),CUDA_initialized=False)


def run_global_gate_cpu_tests():
    """Metadata/IO-only new guard tests; do not rerun the seven sampler fixtures."""
    from copy import deepcopy
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    from .. import common
    from ..stages_ab import _validate_B_gate_payload,require_all_B_gates,run_B
    from native_t1.artifacts import file_sha256
    seeds=[10401,10402];rows=[];rejections=0
    def payload(label):
        counts=new_counts();counts.update(model_forward_attempts=150,model_forward_returns=150,euler_updates=150,suffix_attempts=3,suffix_returns=3)
        return dict(stage='B',mode='gate',model=label,status='PASS',seeds=seeds,
            state_before=common.ASSETS[label]['state'],state_after=common.ASSETS[label]['state'],counts=counts,rows=[],
            same_model_replays=[dict(source_model=label,continuation_model=label,parent=0,patch=0,seed=10401,start_step=cut,
                same_model_replay=True,same_model_full_suffix_bitwise=True,precision='fp32',nfe=100,
                audit=dict(known_all_states_bitwise=True,integration_steps=100-cut)) for cut in (20,50,80)])
    for label in ('B','J'):
        assert _validate_B_gate_payload(payload(label),label,seeds)
    rows.append(dict(name='two_valid_model_gate_payloads',status='PASS'))
    for defect in ('failed','wrong_state','missing_replay','wrong_count','wrong_seed','bitwise_false','C_false'):
        x=payload('B')
        if defect=='failed':x['status']='STOPPED'
        if defect=='wrong_state':x['state_after']='0'*64
        if defect=='missing_replay':x['same_model_replays'].pop()
        if defect=='wrong_count':x['counts']['model_forward_attempts']=151
        if defect=='wrong_seed':x['same_model_replays'][0]['seed']=10402
        if defect=='bitwise_false':x['same_model_replays'][0]['same_model_full_suffix_bitwise']=False
        if defect=='C_false':x['same_model_replays'][0]['audit']['known_all_states_bitwise']=False
        try:_validate_B_gate_payload(x,'B',seeds)
        except RuntimeError:rejections+=1
        else:raise AssertionError('Invalid global B gate accepted '+defect)
    rows.append(dict(name='failed_incomplete_wrong_identity_or_replay_rejected',status='PASS',expected_rejections=7))
    with tempfile.TemporaryDirectory(prefix='jedge5_B_gate_cpu_') as d:
        out=Path(d)
        with patch.object(common,'registered_seeds',return_value=seeds), patch.object(common,'load_model',side_effect=AssertionError('model load forbidden')) as lm, patch.object(common,'load_cases',side_effect=AssertionError('case load forbidden')) as lc, patch.object(common,'noise_bank',side_effect=AssertionError('array load forbidden')) as ln:
            try:run_B(out,'B',mode='cross')
            except RuntimeError:rejections+=1
            else:raise AssertionError('Cross stage started with absent gates')
            assert lm.call_count==lc.call_count==ln.call_count==0 and not (out/'B').exists()
        for label in ('B','J'):
            folder=out/'B'/label;folder.mkdir(parents=True)
            x=payload(label)
            for row in x['same_model_replays']:
                f=folder/('synthetic_proof_'+str(row['start_step'])+'.bin');f.write_bytes(b'CPU fixture only')
                row.update(path=str(f.relative_to(out)),sha256=file_sha256(f))
            (folder/'gate_results.json').write_text(json.dumps(x))
            if label=='B':
                try:require_all_B_gates(out,seeds)
                except RuntimeError:rejections+=1
                else:raise AssertionError('One model gate incorrectly sufficient')
        evidence=require_all_B_gates(out,seeds)
        assert evidence['all_six_same_model_suffixes_bitwise'] and len(evidence['models'])==2
        (out/'B'/'J'/'synthetic_proof_20.bin').write_bytes(b'changed fixture')
        try:require_all_B_gates(out,seeds)
        except RuntimeError:rejections+=1
        else:raise AssertionError('Mutated replay proof accepted')
    rows.append(dict(name='both_models_required_before_any_array_or_model_load_and_proof_hash_checked',status='PASS',expected_rejections=3))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',tests_run=3,tests=rows,counts=dict(expected_rejected_requests=rejections,
        CPU_mock_forward_calls=0,actual_model_forwards=0,model_backwards=0,OT_calls=0,optimizer_updates=0),CUDA_initialized=False)


if __name__=='__main__':
    print(json.dumps(run_cpu_tests(),indent=2,allow_nan=False))
