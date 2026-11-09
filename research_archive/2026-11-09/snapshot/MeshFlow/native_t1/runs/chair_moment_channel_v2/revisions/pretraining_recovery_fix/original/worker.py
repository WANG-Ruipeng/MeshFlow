"""One GPU worker; resource admission and shared lock precede CUDA initialization."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='2'
from native_t1 import runtime as _runtime
import argparse
import time
import traceback
import torch
from experiments.mf_h1_local0.control_mvp48.train_io import resource_snapshot
from experiments.mf_h1_local0.control_mvp48.train_run import GPUMonitor
from experiments.mf_h1_local0.chair_domain_control_v1.training import process_gpu_lock
from .common import *
from .ledger import Budget

def admission(snapshot,protocol):
    fields=snapshot['gpu']['stdout'].strip().split(',');total,used,free=map(float,fields[-4:-1])
    reserve=protocol['cost_gate']['reserve_mib'];required=1.2*6.8*1024
    result=dict(total_mib=total,used_mib=used,free_mib=free,reserve_mib=reserve,
        prior_training_process_estimate_mib=6.8*1024,required_mib=required,gpu_workers=1,
        reason='Explicit single GPU worker; resource estimate includes context/optimizer/backward, actual peaks monitored',
        status='PASS' if free-reserve>=required else 'WAIT_RESOURCE')
    if result['status']!='PASS':raise RuntimeError('Resource headroom unavailable: '+str(result))
    return result

def run(out,stage,arm=None):
    out=Path(out);key=stage if arm is None else stage+'_'+arm
    path=out/'stages'/f'{key}.json';protocol=read_json(out/'protocol.json')
    if path.exists() and read_json(path)['status']=='DONE':return read_json(path)
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before shared lock')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);_runtime.configure_stable_runtime()
    before=resource_snapshot();plan=admission(before,protocol)
    atomic_json(out/'resources'/f'{key}_before.json',dict(resources=before,admission=plan))
    state=dict(status='RUNNING',stage=key,started_unix=time.time(),gpu_workers=1);atomic_json(path,state)
    monitor=None;tick=time.perf_counter()
    try:
        with process_gpu_lock(out) as wait:
            locked=resource_snapshot();admission(locked,protocol);atomic_json(out/'resources'/f'{key}_locked.json',locked)
            (out/'training').mkdir(parents=True,exist_ok=True);monitor=GPUMonitor(out);monitor.start()
            if stage=='precheck':
                from .precheck import run as operation
                result=operation(out)
            else:
                from .checkpoint import load_start,load
                from .model_adapter import set_profile
                budget=Budget(out,key)
                if stage=='initial_replay':
                    from .evaluation import initial_replay
                    model,opt,meta,hashes,stream=load_start('MOMENT_XY','cuda',optimizer=False)
                    set_profile(model,False,collect_stats=False);result=initial_replay(out,model,budget)
                elif stage=='train':
                    from .training import train_arm
                    assert arm in ARMS and read_json(out/'gate_results.json')['status']=='PASS'
                    assert read_json(out/'initial_replay/result.json')['status']=='PASS'
                    from .registration import verify_identity
                    verify_identity(out,code=True)
                    candidates=[p for p in (out/'checkpoints').glob(f'{arm}_moment*_total*.pt')]
                    if candidates:
                        selected=max(candidates,key=lambda p:read_json(p.with_suffix('.json'))['metadata']['moment_step'])
                        model,opt,meta,hashes,stream=load(selected,'cuda',trainable=True,optimizer=True)
                    else:model,opt,meta,hashes,stream=load_start(arm,'cuda',optimizer=True)
                    result=train_arm(out,arm,protocol,budget,model,opt,meta,stream)
                elif stage=='eval':
                    from .evaluation import run as evaluate
                    p=out/'checkpoints'/f'{arm}_moment1000_total6000.pt'
                    model,opt,meta,hashes,stream=load(p,'cuda',trainable=False,optimizer=False)
                    assert meta['moment_step']==1000 and meta['conditional_total_step']==6000
                    set_profile(model,False,collect_stats=False);result=evaluate(out,arm+'_END',model,budget,digest(p))
                else:raise ValueError(stage)
            monitor.stop();monitor=None
            state.update(status='DONE',result=result,wall_seconds=time.perf_counter()-tick,lock_wait_seconds=wait,finished_unix=time.time())
            Budget(out,key+'_finish').snapshot();atomic_json(out/'resources'/f'{key}_after.json',resource_snapshot());atomic_json(path,state)
            print('STAGE_DONE '+key,flush=True);return state
    except BaseException as exc:
        state.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-tick)
        atomic_json(path,state)
        try:Budget(out,key+'_failure').snapshot()
        except RuntimeError:pass
        raise
    finally:
        if monitor is not None:monitor.stop()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--stage',choices=('precheck','initial_replay','train','eval'),required=True);ap.add_argument('--arm',choices=ARMS)
    a=ap.parse_args();run(a.out,a.stage,a.arm)
