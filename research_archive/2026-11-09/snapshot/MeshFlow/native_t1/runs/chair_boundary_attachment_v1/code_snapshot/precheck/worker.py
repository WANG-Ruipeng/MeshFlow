"""One locked CUDA owner. No CUDA initialization before the shared flock."""
from native_t1 import runtime
import argparse
import fcntl
import gc
import os
import time
import traceback
import torch
from .common import *
from .ledger import Budget
from .resources import Monitor,before_dispatch,resource_snapshot


def run(stage,out=OUT,arm=None):
    from . import protocol,checkpoint
    out=Path(out);protocol.verify(out,training=stage in ('train','sample'))
    runtime.configure_stable_runtime();torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized before shared lock')
    if stage in ('train','sample') and arm not in ARMS:raise ValueError('Unregistered arm')
    name=stage+('_'+arm if arm else '')
    status_path=out/'workers'/f'{name}.json'
    if status_path.exists() and read_json(status_path)['status']=='DONE':return read_json(status_path)
    budget=Budget(out,name);started=time.perf_counter()
    with open(GPU_LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        append_event(out/'ledger/lifecycle.jsonl',dict(event='GPU_LOCK_ACQUIRED',stage=name,path=GPU_LOCK,pid=os.getpid()))
        before_dispatch(out,name);monitor=Monitor(out,name).start()
        atomic_json(status_path,dict(status='RUNNING',stage=name,pid=os.getpid()))
        try:
            if stage=='precheck':
                from .precheck import run_precheck
                result=run_precheck(out,budget)
            elif stage=='replay':
                from .sampling import replay_initial
                result=replay_initial(out,budget)
            elif stage=='train':
                from .training import train_arm
                p=read_json(out/'protocol.json')
                if p['arm_status'][arm]!='PASS':return dict(status='BLOCKED',arm=arm,reason=p['arm_status'][arm])
                pointer=out/'training'/f'{arm}_pointer.json'
                path=resolve(read_json(pointer)['path']) if pointer.exists() else START
                model,opt,meta,stream_state=checkpoint.load(path,arm,'cuda',trainable=True,optimizer=True,out=out)
                resource_snapshot(out,'loaded_'+name)
                result=train_arm(out,arm,p,budget,model,opt,meta,stream_state)
                del model,opt;gc.collect();torch.cuda.empty_cache()
            elif stage=='sample':
                from .sampling import endpoint
                result=endpoint(out,arm,budget)
            else:raise ValueError(stage)
            worker_status=result['status'] if stage=='train' and result['status']!='DONE' else 'DONE'
            state=dict(status=worker_status,stage=name,wall_seconds=time.perf_counter()-started,result=result)
            atomic_json(status_path,state);return state
        except BaseException as exc:
            atomic_json(status_path,dict(status='BLOCKED',stage=name,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started))
            raise
        finally:
            budget.snapshot();monitor.close();resource_snapshot(out,'completed_'+name)
            append_event(out/'ledger/lifecycle.jsonl',dict(event='GPU_LOCK_RELEASE',stage=name,pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['precheck','replay','train','sample'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--out',default=str(OUT))
    args=parser.parse_args();print(run(args.stage,args.out,args.arm),flush=True)
