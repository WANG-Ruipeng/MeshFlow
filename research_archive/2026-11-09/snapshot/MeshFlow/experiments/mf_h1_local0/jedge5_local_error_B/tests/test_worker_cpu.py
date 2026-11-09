"""In-memory CPU exercise of the actual worker control flow, using analytic fields.

No real checkpoint/model, CUDA, lock, OT, geometry scorer, or training is used.
The fake forward counters test orchestration, not scientific neural-call counts.
"""
from native_t1 import runtime as _stable
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from experiments.mf_h1_local0.jedge5_local_error_B import worker as w
from experiments.mf_h1_local0.jedge5_local_error_B.math_ops import euler_free, bits_equal


class NPZ:
    def __init__(self, values): self.values=values; self.files=list(values)
    def __getitem__(self, key): return self.values[key]
    def __enter__(self): return self
    def __exit__(self, *args): return None


class Field:
    def __init__(self, label): self.label=label; self.cpu_analytic_calls=0
    def __call__(self, x, t, y, valid, known):
        assert x.device.type=='cpu' and x.dtype==torch.float32
        assert x.shape==(1,112,9) and known.shape==valid.shape==(1,112)
        assert bool(valid.all()) and bool(known[:,:12].all()) and not bool(known[:,12:].any())
        assert y.tolist()==[112] and t.shape==(1,)
        self.cpu_analytic_calls+=1
        return (1+.2*t[:,None,None])*x + (0.01 if self.label=='B' else 0.02)


class Event:
    def __init__(self, **kwargs): pass
    def record(self): pass
    def elapsed_time(self, other): return 0.0


class Recorder:
    def __init__(self, model): pass
    def finish(self, velocity, precision): assert precision=='fp32'; return {'CPU_mock':True}
    def close(self): pass


def run_fixture():
    started=time.perf_counter()
    virtual=Path('/mnt/e/MeshFlow/experiments/mf_h1_local0/jedge5_local_error_B/tests/_virtual_unwritten')
    arrays={}; records={}; models={m:Field(m) for m in ('B','J')}
    known=torch.zeros((1,112),dtype=torch.bool); known[:,:12]=True
    valid=torch.ones_like(known); y=torch.tensor([112])
    noise=torch.arange(1008,dtype=torch.float32).reshape(1,112,9)/1000
    noise[0,0,0]=-0.0
    C=noise[0,:12].reshape(12,3,3).numpy().copy()
    trajectories=[]
    for m,model in models.items():
        state=noise.clone();path=[state[0].numpy().copy()]
        for k in range(100):
            v=model(state,torch.tensor([k/100],dtype=torch.float32),y,valid,known)
            next_state=state.clone()
            next_state[:,12:]=state[:,12:]+v[:,12:]*.01
            state=next_state;path.append(state[0].numpy().copy())
        key=str(virtual/f'{m}_archive.npz')
        arrays[key]={'states':np.stack(path),'C':C,'raw':path[-1]}
        trajectories.append(dict(case_id=0,model=m,parent=0,patch=0,seed=10401,bbox=3.0,path=key,sha256='fake'))
        model.cpu_analytic_calls=0
    noise_key=str(virtual/'noise.npz'); arrays[noise_key]={'seed_10401':noise.numpy().copy()}
    records[str(virtual/'registration/protocol.json')]={'models':[{'label':m,'state_sha256':m} for m in models],
                                                     'trajectories':trajectories,'noise_path':noise_key}
    def write(path,value): records[str(path)]=deepcopy(value)
    def save(path,**values):
        arrays[str(path)]={k:v.detach().numpy().copy() if torch.is_tensor(v) else np.array(v,copy=True)
                           for k,v in values.items()}
        return str(path.relative_to(virtual)),'fake'
    def sample(model,noise,C,*,nfe,precision,counts,initial_state,start_step):
        assert precision=='fp32' and nfe==100 and start_step in (21,51,81)
        counts['suffix_attempts']+=1
        x=initial_state.clone();path=[x[0].numpy().copy()]
        for k in range(start_step,nfe):
            counts['model_forward_attempts']+=1
            v=model(x,torch.tensor([k/100],dtype=torch.float32),y,valid,known)
            counts['model_forward_returns']+=1
            x=euler_free(x,v,.01,known);counts['euler_updates']+=1
            assert bits_equal(x[known],noise[known])
            path.append(x[0].numpy().copy())
        counts['suffix_returns']+=1
        return path[-1],np.stack(path),dict(known_all_states_bitwise=True,OT_calls=0,model_forward_cuda_event_seconds=0)
    # Global torch constructors remain CPU-only while the actual worker asks for CUDA.
    def cpu_constructor(original):
        def call(*args,**kwargs):
            if kwargs.get('device')=='cuda': kwargs['device']='cpu'
            return original(*args,**kwargs)
        return call
    orig={name:getattr(torch,name) for name in ('zeros','tensor','full')}
    before_cuda=torch.cuda.is_initialized()
    with ExitStack() as stack:
        replacements={
            'OUT':virtual,'read_json':lambda path:deepcopy(records[str(path)]),'write_json':write,
            'file_sha256':lambda path:'fake','save_npz':save,'tensor':lambda a:torch.from_numpy(a.copy()),
            'load_model':lambda label:(models[label],dict(state_sha256=label)),
            'state_sha256':lambda model:model.label,'DTypeRecorder':Recorder,'sample':sample,
            'resource_snapshot':lambda:dict(gpu_csv=dict(stdout='Mock GPU, 16000, 1000, 15000, 0')),
            'native_execution':lambda *args,**kwargs:nullcontext(dict(CPU_mock=True)),
            'aggregate':lambda stage:None,
        }
        for name,value in replacements.items(): stack.enter_context(patch.object(w,name,value))
        stack.enter_context(patch.object(w._runtime,'configure_stable_runtime',lambda:None))
        stack.enter_context(patch.object(w.fcntl,'flock',lambda *args:None))
        stack.enter_context(patch('builtins.open',lambda *args,**kwargs:io.StringIO()))
        stack.enter_context(patch.object(Path,'mkdir',lambda *args,**kwargs:None))
        stack.enter_context(patch.object(Path,'exists',lambda *args:False))
        stack.enter_context(patch.object(np,'load',lambda path,**kwargs:NPZ(arrays[str(path)])))
        for name,value in orig.items(): stack.enter_context(patch.object(torch,name,cpu_constructor(value)))
        stack.enter_context(patch.object(torch.cuda,'Event',Event))
        for name in ('reset_peak_memory_stats','synchronize'):
            stack.enter_context(patch.object(torch.cuda,name,lambda *args:None))
        for name in ('max_memory_allocated','max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda,name,lambda *args:0))
        w.execute('local',0,attempt=2)
        local=records[str(virtual/'workers/local_case0_attempt2.json')]
        assert local['status']=='PASS' and local['counts']['model_forward_attempts']==148
        assert len(local['rows'])==24 and len(local['supplemental_rows'])==2
        assert all(r['next_state_bitwise'] for r in local['rows']+local['supplemental_rows'])
        assert local['raw_known_velocity_max_observed']>0 # raw C velocity is allowed and ignored.
        records[str(virtual/'raw/local_rows.json')]={'rows':local['rows']*6,'supplemental_rows':local['supplemental_rows']*6}
        w.execute('suffix',0)
        suffix=records[str(virtual/'workers/suffix_case0.json')]
        assert suffix['status']=='PASS' and suffix['counts']['model_forward_attempts']==318
        assert suffix['counts']['suffix_returns']==6 and suffix['counts']['euler_updates']==294
        assert len(suffix['rows'])==6 and len(suffix['probes'])==12
        assert all(p['status']=='RESOLVED' for p in suffix['probes'])
        assert sorted(r['start_step'] for r in suffix['rows'])==[21,21,51,51,81,81]
        assert suffix['counts']['full_rollout_attempts']==0
    assert before_cuda is False and torch.cuda.is_initialized() is False
    return dict(status='PASS',CPU_analytic_calls={'local':148,'suffix_and_direction':318},
                local_states=24,supplemental_states=2,suffixes=6,direction_probes=12,
                suffix_Euler_updates=294,known_negative_zero_preserved=True,
                raw_known_velocity_allowed=True,real_neural_forwards=0,CUDA_initialized=False,
                optimizer_updates=0,model_backwards=0,OT_calls=0,
                mock_IO='all worker filesystem, GPU, lock, and checkpoint operations mocked in memory',
                seconds=time.perf_counter()-started)


if __name__=='__main__':
    result=run_fixture()
    Path(__file__).with_name('worker_cpu_test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
