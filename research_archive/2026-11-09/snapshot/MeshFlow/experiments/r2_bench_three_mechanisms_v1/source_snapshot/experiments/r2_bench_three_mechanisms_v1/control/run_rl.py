"""External resource observer; does not modify the frozen RL implementation."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'r2/src'))
from r2.pilots.bench_three_mechanisms_v1 import common as C


def resource_guard(snapshot,estimate):
    if float(estimate)<=0:raise ValueError('Positive observed/comparable process estimate required')
    headroom=max(2048.,.1*float(snapshot['total_mib']))
    available=float(snapshot['free_mib'])-headroom
    if available<1.2*float(estimate):raise RuntimeError('RESOURCE_WAIT: insufficient current free VRAM')
    return dict(headroom_mib=headroom,free_mib=snapshot['free_mib'],estimated_process_peak_mib=float(estimate),
        admissible_incremental_slots=int(max(0,available)//(1.2*float(estimate))),
        worker_count=1,controller_owns_parallel_dispatch=True)


def check_identity(go_path=None):
    rr=C.OUT/'rl/registration_v2.json';reg=C.read(rr)
    if reg['root_registration_sha256']!=C.sha(C.REGISTRATION):raise ValueError('Root registration changed')
    if reg['schedule_sha256']!=C.sha(C.OUT/'data/rl_schedule.json'):raise ValueError('RL schedule changed')
    for path,digest in reg['source_code_sha256'].items():
        if C.sha(ROOT/path)!=digest:raise ValueError('Frozen source changed: '+path)
    gate=C.read(C.OUT/'rl/cpu_contract_v2.json')
    if gate['status']!='PASS' or gate['registration_sha256']!=C.sha(rr):raise ValueError('CPU gate mismatch')
    if go_path is not None:
        go=C.read(go_path)
        if (go.get('status')!='GO' or go.get('arm')!='RL16' or
            go.get('registration_sha256')!=C.sha(C.REGISTRATION) or
            go.get('rl_registration_sha256')!=C.sha(rr)):raise ValueError('Explicit matching resource GO required')
    return dict(status='PASS',root_registration_sha256=C.sha(C.REGISTRATION),rl_registration_sha256=C.sha(rr),
        frozen_sources=len(reg['source_code_sha256']),model_calls=0,CUDA_initialized=False)


def run(go_path,estimated_peak_mib):
    identity=check_identity(go_path)
    complete=C.OUT/'rl/training/complete.json'
    if complete.exists():
        return dict(status='ALREADY_COMPLETE',completion_sha256=C.sha(complete),model_calls=0)
    snapshot=C.resources();dispatch=resource_guard(snapshot,estimated_peak_mib)
    folder=C.OUT/'rl/resources'/('run_'+str(time.time_ns()));folder.mkdir(parents=True)
    C.write(folder/'dispatch.json',dict(**identity,**dispatch,resource_before=snapshot,
        GO_path=str(C.resolve(go_path)),GO_sha256=C.sha(go_path),pid=os.getpid(),
        wrapper_sha256=C.sha(__file__),
        scope='Observer starts before load_policy; includes actual rollout/reward waits/backward/optimizer/checkpoint work'))
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    class PersistentObserver(GPUObserver):
        def sample(self):
            super().sample()
            C.event(folder/'samples.jsonl',self.samples[-1])
    observer=PersistentObserver(C.EXPERIMENT+'_RL16',interval=1.)
    started=time.perf_counter();observer.__enter__()
    try:
        from r2.pilots.bench_three_mechanisms_v1 import rl
        result=rl.train(go_path)
        C.write(folder/'complete.json',dict(status='COMPLETE',seconds=time.perf_counter()-started,
            training_completion=str(complete),training_completion_sha256=C.sha(complete),
            generator_forwards=result['generator_forwards'],backwards=result['backwards'],
            opportunities=result['opportunities'],optimizer_updates=result['optimizer_updates']))
        return result
    except BaseException as exc:
        C.write(folder/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            seconds=time.perf_counter()-started,no_automatic_retry=True))
        raise
    finally:
        observer.__exit__(None,None,None)
        C.write(folder/'observation.json',dict(status='RECORDED',summary=observer.summary(),samples=observer.samples,
            seconds=time.perf_counter()-started,resource_after=C.resources(),
            process_VRAM_scope='Matching PID only when nvidia-smi exposes it; never substitute device usage for process usage',
            allocator_scope='Separate per-opportunity allocated/reserved peaks are in training complete records',
            samples_jsonl_sha256=C.sha(folder/'samples.jsonl')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--resource-go');p.add_argument('--estimated-peak-mib',type=float)
    args=p.parse_args()
    if args.check and not args.run:
        assert resource_guard(dict(total_mib=16000,free_mib=12000),4000)['admissible_incremental_slots']==2
        try:resource_guard(dict(total_mib=16000,free_mib=5000),4000)
        except RuntimeError:pass
        else:raise AssertionError('Resource guard accepted insufficient headroom')
        result=check_identity(args.resource_go)
    elif args.run and not args.check and args.resource_go and args.estimated_peak_mib:
        result=run(args.resource_go,args.estimated_peak_mib)
    else:p.error('Use --check or explicit --run --resource-go PATH --estimated-peak-mib VALUE')
    print(json.dumps(result))


if __name__=='__main__':main()
