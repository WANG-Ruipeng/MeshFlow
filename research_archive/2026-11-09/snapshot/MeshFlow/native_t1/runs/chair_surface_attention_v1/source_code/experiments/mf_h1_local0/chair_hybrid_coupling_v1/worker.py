"""One short-lived GPU worker under the pre-existing shared flock."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='2'
from native_t1 import runtime as _runtime
import argparse
from pathlib import Path
import time
import traceback
import torch
from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor,dispatch_check
from experiments.mf_h1_local0.chair_domain_control_v1.training import process_gpu_lock
from .common import *
from .ledger import Budget
from .checkpoint import load,tensor_state_hash
from .registration import verify_identity


def run(stage,out=OUT):
    out=Path(out);status_path=out/'stages'/f'{stage}.json'
    if status_path.exists():
        prior=read_json(status_path)
        if prior['status']=='DONE':return prior
        if prior['status'] in ('BLOCKED','RUNNING'):
            raise RuntimeError('Affected stage has prior failed/interrupted work; inspect ledger, no automatic replay')
    protocol=read_json(out/'protocol.json');verify_identity(out)
    assert read_json(out/'cpu_tests/summary.json')['status']=='PASS'
    if stage!='precheck':assert read_json(out/'stages/precheck.json')['status']=='DONE'
    if stage.startswith('train_'):
        assert read_json(out/'stages/eval_START.json')['status']=='DONE'
        assert read_json(out/'evaluation/START/scoring/summary.json')['status']=='COMPLETE'
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before shared GPU lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);_runtime.configure_stable_runtime()
    (out/'training').mkdir(parents=True,exist_ok=True)
    before=resource_snapshot();admission=dispatch_check(before)
    atomic_json(out/'resources'/f'{stage}_before.json',dict(resources=before,admission=admission))
    state=dict(status='RUNNING',stage=stage,started_unix=time.time(),gpu_workers=1)
    atomic_json(status_path,state)
    budget=None;monitor=None;tick=time.perf_counter()
    try:
        with process_gpu_lock(out) as lock_wait:
            locked=resource_snapshot();dispatch_check(locked)
            atomic_json(out/'resources'/f'{stage}_locked.json',locked)
            budget=Budget(out,stage)
            label=stage.removeprefix('train_').removeprefix('eval_')
            trainable=stage=='precheck' or stage.startswith('train_')
            cp=START
            if stage.startswith('eval_') and label!='START':
                assert read_json(out/'stages'/f'train_{label}.json')['status']=='DONE'
                cp=out/'checkpoints'/f'{label}_pilot1000_total5000.pt'
            elif stage.startswith('train_') and (out/'training'/f'{label}_pointer.json').exists():
                cp=Path(read_json(out/'training'/f'{label}_pointer.json')['path'])
            monitor=GPUMonitor(out);monitor.start()
            load_tick=time.perf_counter()
            model,optimizer,meta,hashes,stream_state=load(cp,'cuda:0',trainable=trainable,optimizer=trainable)
            load_seconds=time.perf_counter()-load_tick
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            if stage=='precheck':
                from .precheck import run as operation
                result=operation(out,model,optimizer,meta,budget)
            elif stage.startswith('train_'):
                from .training import train_arm
                result=train_arm(out,label,protocol,budget,model,optimizer,meta,stream_state)
            elif stage.startswith('eval_'):
                from .evaluation import run as operation
                result=operation(out,label,model,budget,hashes)
                if tensor_state_hash(model.state_dict())!=meta['model_state_sha256']:
                    raise ValueError('Evaluation changed model state')
            else:raise ValueError('Unknown stage')
            torch.cuda.synchronize();budget.snapshot()
            monitor.stop();monitor=None
            completion='INTERRUPTED_SAFE' if result.get('status')=='INTERRUPTED_SAFE' else 'DONE'
            state.update(status=completion,result=result,lock_wait_seconds=lock_wait,load_seconds=load_seconds,
                         wall_seconds=time.perf_counter()-tick,finished_unix=time.time(),
                         peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                         peak_scope='Phase after model loading; evaluation sampler also reports per-rollout reset peaks; nvidia-smi records process/global peaks separately')
            if digest(START)!=START_SHA:raise ValueError('Protected starting checkpoint changed')
            atomic_json(out/'resources'/f'{stage}_after.json',resource_snapshot())
            atomic_json(status_path,state)
            print(stage+' '+completion,flush=True)
            return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-tick)
        atomic_json(status_path,state)
        if budget is not None:budget.snapshot()
        raise
    finally:
        if monitor is not None:monitor.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();run(a.stage,a.out)
