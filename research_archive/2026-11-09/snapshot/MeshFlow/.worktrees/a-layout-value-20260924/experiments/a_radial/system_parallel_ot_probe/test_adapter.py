from concurrent.futures import ThreadPoolExecutor
import random
import threading
import time
import numpy as np
import pytest
import torch
from .adapter import load_microbatch,OTContractError
from ..system_probe.io_profile import seed_cpu,cpu_rng
from ..system_probe.official import exact_same


class SyntheticDataset:
    noise_sort='ot'
    mode='ordinary'
    def __getitem__(self,index):
        import datasets.mesh_dataset as module
        data=np.full((512,3,3),float(index))
        noise=np.random.randn(512,3,3)
        marker=random.random()+float(torch.rand(()))
        if self.mode=='missing':result=noise
        elif self.mode=='special':result=module.optimal_sum_numpy(data,noise,optimal=False)
        else:
            result=module.optimal_sum_numpy(data,noise,optimal=True)
            if self.mode=='multiple':module.optimal_sum_numpy(data,noise,optimal=True)
        if self.mode=='mutated':result=result+1
        return {'uid':str(index),'num_faces':512,'len':512,'coords':data.reshape(512,9)+marker,
                'noise':result.reshape(512,9)}


def setup_ot(monkeypatch,function=None):
    import datasets.mesh_dataset as module
    import utils.ot_utils as ot
    if function is None:
        def function(data,noise,optimal=True):return noise[::-1].copy()
    monkeypatch.setattr(ot,'optimal_sum_numpy',function)
    monkeypatch.setattr(module,'optimal_sum_numpy',function)
    return module,function


def test_sequential_rng_and_all_collated_arrays_identical(monkeypatch):
    module,original=setup_ot(monkeypatch);ds=SyntheticDataset()
    seed_cpu(19)
    expected=module.collate_fn([ds[i] for i in range(4)],max_seq_length=800);state=cpu_rng()
    seed_cpu(19)
    with ThreadPoolExecutor(max_workers=4) as pool:actual,audit=load_microbatch(ds,[0,1,2,3],pool)
    exact_same(expected,actual);exact_same(state,cpu_rng())
    assert [x['uid'] for x in audit]==['0','1','2','3']
    assert module.optimal_sum_numpy is original


@pytest.mark.parametrize('mode',['multiple','missing','special','mutated'])
def test_unsupported_data_flow_rejected_and_binding_restored(monkeypatch,mode):
    module,original=setup_ot(monkeypatch);ds=SyntheticDataset();ds.mode=mode
    with ThreadPoolExecutor(max_workers=4) as pool:
        with pytest.raises(OTContractError):load_microbatch(ds,[0,1,2,3],pool)
    assert module.optimal_sum_numpy is original


def test_failed_worker_is_drained_and_binding_restored(monkeypatch):
    completed=[];lock=threading.Lock()
    def fail_first(data,noise,optimal=True):
        index=int(data[0,0,0])
        if index!=0:time.sleep(.015)
        with lock:completed.append(index)
        if index==0:raise RuntimeError('intentional worker failure')
        return noise
    module,original=setup_ot(monkeypatch,fail_first)
    with ThreadPoolExecutor(max_workers=4) as pool:
        with pytest.raises(RuntimeError,match='intentional worker failure'):
            load_microbatch(SyntheticDataset(),[0,1,2,3],pool)
        assert sorted(completed)==[0,1,2,3]
    assert module.optimal_sum_numpy is original
