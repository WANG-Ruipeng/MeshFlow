"""One registered real-model audit worker, protected before CUDA initialization."""
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
from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor
from experiments.mf_h1_local0.chair_domain_control_v1.training import process_gpu_lock
from .common import *
from .ledger import Budget
from .registration import verify as verify_identity


def admission(snapshot,protocol):
    fields=snapshot['gpu']['stdout'].strip().split(',')
    total,used,free=map(float,fields[-4:-1]);reserve=max(.15*total,2048.)
    # Comparable old real training process used about 6.8 GiB including context;
    # admission only starts one authorized audit, and its actual peaks decide cost.
    required=1.2*6.8*1024
    result=dict(total_mib=total,used_mib=used,free_mib=free,reserve_mib=reserve,
        comparable_training_process_estimate_mib=6.8*1024,required_initial_mib=required,
        gpu_workers=1,explicit_task_serial=True,status='PASS' if free-reserve>=required else 'WAIT_RESOURCE')
    if result['status']!='PASS':raise RuntimeError('Resource headroom insufficient: '+str(result))
    return result


def run(out=OUT):
    out=Path(out);stage='precheck';statepath=out/'stages/precheck.json'
    if statepath.exists():
        prior=read_json(statepath)
        if prior['status']=='DONE':return prior
        raise RuntimeError('Precheck already attempted; no automatic replay')
    protocol=read_json(out/'protocol.json');verify_identity(out)
    assert read_json(out/'tests/monitor/result.json')['status']=='PASS'
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);_runtime.configure_stable_runtime()
    before=resource_snapshot();plan=admission(before,protocol)
    atomic_json(out/'resources/precheck_before.json',dict(resources=before,admission=plan))
    (out/'training').mkdir(parents=True,exist_ok=True)
    state=dict(status='RUNNING',stage=stage,started_unix=time.time(),gpu_workers=1)
    atomic_json(statepath,state);monitor=None;tick=time.perf_counter()
    try:
        with process_gpu_lock(out) as wait:
            locked=resource_snapshot();admission(locked,protocol)
            atomic_json(out/'resources/precheck_locked.json',locked)
            monitor=GPUMonitor(out);monitor.start()
            from .precheck import run as operation
            result=operation(out)
            monitor.stop();monitor=None
            Budget(out,'precheck_finalize').snapshot()
            complete=result.get('actual_forwards')==76 and result.get('actual_zero_update_backwards')==20
            state.update(status='DONE' if complete else 'INCOMPLETE',result=result,lock_wait_seconds=wait,wall_seconds=time.perf_counter()-tick,
                finished_unix=time.time(),meaning='Finite checks executed; DONE does not imply a passing numerical or cost gate')
            assert digest(START)==START_SHA
            atomic_json(out/'resources/precheck_after.json',resource_snapshot());atomic_json(statepath,state)
            print('PRECHECK_EXECUTED '+str(result.get('status')),flush=True);return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-tick)
        atomic_json(statepath,state)
        try:Budget(out,'precheck_finalize').snapshot()
        except RuntimeError:pass
        raise
    finally:
        if monitor is not None:monitor.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();run(a.out)
