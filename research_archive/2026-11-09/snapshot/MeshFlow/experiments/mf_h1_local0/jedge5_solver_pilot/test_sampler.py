"""Bounded CPU fixtures. No real MeshFlow model, CUDA, OT or training."""
from native_t1.runtime import configure_stable_runtime
import math,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from .sampler import sample,new_counts,METHODS,_validate_velocity


class AnalyticModel(torch.nn.Module):
    def __init__(self,kind,C,fail_on=None):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.experiment_trainable=False;self.kind=kind;self.C=C.reshape(12,9).clone();self.fail_on=fail_on;self.calls=0;self.times=[]
        self.eval()
    def forward(self,x,t,y,valid,known):
        self.calls+=1;self.times.append(float(t[0]))
        assert x.dtype==t.dtype==torch.float32
        assert y.tolist()==[112] and valid.all() and known[:,:12].all() and not known[:,12:].any()
        assert torch.equal(x[0,:12].view(torch.int32),self.C.view(torch.int32))
        if self.calls==self.fail_on:raise RuntimeError('fixture injected model failure')
        v=torch.ones_like(x)*.75 if self.kind=='constant' else x.clone() if self.kind=='linear' else torch.ones_like(x)*t[0]
        v[:,:12]=123. # The solver must ignore raw known-slot velocity, not require zero.
        return v


def run_cpu_tests():
    torch.set_num_threads(1);tick=time.perf_counter()
    counts=dict(real_model_forwards=0,model_backwards=0,optimizer_updates=0,OT_calls=0,geometry_autograd_calls=0,
                CPU_mock_forward_attempts=0,CPU_mock_forward_returns=0,CPU_mock_complete_rollouts=0,expected_rejections=0)
    rows=[];C=torch.arange(108,dtype=torch.float32).reshape(12,3,3)/100.;C[0,0,0]=-0.
    noise=torch.ones((1,112,9),dtype=torch.float32)
    linear_errors={}
    for kind in ('constant','linear','time'):
        for method,spec in METHODS.items():
            model=AnalyticModel(kind,C);ledger=new_counts();n0=noise.clone();c0=C.clone()
            raw,path,audit=sample(model,noise,C,method,counts=ledger,cpu_mock=True)
            counts['CPU_mock_forward_attempts']+=ledger['model_forward_attempts'];counts['CPU_mock_forward_returns']+=ledger['model_forward_returns']
            counts['CPU_mock_complete_rollouts']+=ledger['complete_rollouts']
            assert ledger['model_forward_attempts']==ledger['model_forward_returns']==model.calls==spec['nfe']
            assert path.shape==(spec['intervals']+1,112,9)
            assert audit['known_stages_all_bitwise'] and audit['known_accepted_all_bitwise']
            assert np.all(audit['_arrays']['stage_velocities'][:,:12]==123.)
            assert torch.equal(noise.view(torch.int32),n0.view(torch.int32)) and torch.equal(C.view(torch.int32),c0.view(torch.int32))
            assert audit['actual_endpoint_t1_forward_calls']==(1 if spec['solver']=='Heun' else 0)
            assert len(audit['actual_stage_times_FP32'])==spec['nfe']
            value=float(raw[12,0]);error=None
            ideal=np.asarray(audit['ideal_accepted_times']);h=np.diff(ideal)
            if kind=='constant':assert abs(value-1.75)<3e-6
            elif kind=='time':
                expected=1+(.5-.5*float(np.sum(h*h)) if spec['solver']=='Euler' else .5)
                assert abs(value-expected)<3e-6
            else:
                error=abs(value-math.e);linear_errors[method]=error
                discrete=float(np.prod(1+h)) if spec['solver']=='Euler' else float(np.prod(1+h+.5*h*h))
                assert abs(value-discrete)<3e-6
            rows.append(dict(kind=kind,method=method,calls=model.calls,endpoint=value,linear_exact_error=error,
                actual_final_forward_time=model.times[-1],endpoint_t1_calls=audit['actual_endpoint_t1_forward_calls']))
    # Observed Heun25 -> Heun50 order on the same x'=x analytic ODE.
    ratio=linear_errors['Heun25']/linear_errors['Heun50'];assert 3.5<ratio<4.5
    assert abs(linear_errors['Midpoint25']-linear_errors['Heun25'])<3e-6
    # Fixed arithmetic first-order witness for the quadratic Euler mesh. No model calls.
    euler_exact=[]
    for M in (25,50,100):
        h=np.diff((np.arange(M+1,dtype=np.float64)/M)**2)
        euler_exact.append(abs(float(np.prod(1+h))-math.e))
    assert 1.8<euler_exact[0]/euler_exact[1]<2.2 and 1.8<euler_exact[1]/euler_exact[2]<2.2
    # Error bookkeeping includes a returned first stage and a failed second stage.
    ledger=new_counts();model=AnalyticModel('constant',C,fail_on=2)
    try:sample(model,noise,C,'Heun25',counts=ledger,cpu_mock=True)
    except RuntimeError as e:
        assert 'fixture injected' in str(e);assert len(e.partial_states)==1
        assert len(e.partial_stage_states)==2 and len(e.partial_stage_velocities)==1
    else:raise AssertionError('Expected deliberate CPU mock failure')
    assert ledger['model_forward_attempts']==2 and ledger['model_forward_returns']==1 and ledger['complete_rollouts']==0 and ledger['sampling_failures']==1
    counts['CPU_mock_forward_attempts']+=2;counts['CPU_mock_forward_returns']+=1;counts['expected_rejections']+=1
    # CPU inputs must never implicitly become the production BF16 CUDA path.
    for action in (lambda:sample(AnalyticModel('constant',C),noise,C,'Heun25',cpu_mock=False),
                   lambda:sample(AnalyticModel('constant',C),noise,C,'RK45',cpu_mock=True),
                   lambda:_validate_velocity(torch.zeros_like(noise),noise,cpu_mock=False)):
        try:action()
        except (ValueError,RuntimeError):counts['expected_rejections']+=1
        else:raise AssertionError('Expected fail-closed rejection')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',tests_run=6,fixture_groups=['constant all4','linear all4 and order','time-dependent all4',
        'C stage preservation and input identity','failure attempt/return accounting','no silent precision/device fallback'],
        rows=rows,Heun_order_error_ratio=ratio,quadratic_Euler_FP64_discrete_errors=euler_exact,
        counts=counts,CUDA_initialized=False,seconds=time.perf_counter()-tick,
        note='12 successful analytic CPU mock trajectories, one intentionally failed mock. FP64 arithmetic order witness does not call a model. No real model constructed.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);a=p.parse_args();path=Path(a.record)
    if path.exists():raise FileExistsError(path)
    r=run_cpu_tests();r['sources']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (Path(__file__),Path(__file__).with_name('sampler.py'))}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r),flush=True)
