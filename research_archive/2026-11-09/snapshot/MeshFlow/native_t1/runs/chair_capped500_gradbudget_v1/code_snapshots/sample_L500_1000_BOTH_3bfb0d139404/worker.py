"""Single locked GPU stage with immutable source snapshots and call journal."""
from native_t1 import runtime
from pathlib import Path
import argparse,fcntl,time,traceback
import torch
from experiments.mf_h1_local0.chair_boundary_attachment_v1.resources import Monitor,before_dispatch,resource_snapshot
from experiments.mf_h1_local0.chair_control_feedback_auto_v1.worker import sample_manifest
from .common import *
from .ledger import Budget
from . import checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('preflight','train','sample'))
    parser.add_argument('--out',default=str(OUT));parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--until',type=int,choices=(500,1000),default=1000)
    parser.add_argument('--checkpoint',default=str(START));parser.add_argument('--label',choices=LABELS)
    parser.add_argument('--panel',choices=('OLD','FRESH','BOTH'),default='BOTH')
    args=parser.parse_args();out=Path(args.out)
    stage=args.mode+'_'+(args.label or args.arm or 'real')+'_'+str(args.until)+'_'+args.panel
    runtime.configure_stable_runtime();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    protocol=read_json(out/'protocol.json');assert protocol['start_checkpoint_sha256']==START_SHA
    for name,sha in protocol['manifest_sha256'].items():assert digest(out/'manifests'/f'{name}.json')==sha
    for source,sha in protocol['scientific_sources'].items():assert digest(source)==sha,'Frozen scientific code changed: '+source
    for source,sha in protocol['original_scorers'].items():assert digest(source)==sha,'Original scorer changed: '+source
    if args.mode=='train':assert read_json(out/'preflight/real_complete.json')['status']=='PASS'
    sources={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
    snapshot=out/'code_snapshots'/(stage+'_'+json_hash(sources)[:12]);snapshot.mkdir(parents=True,exist_ok=True)
    for name,sha in sources.items():
        dest=snapshot/Path(name).name
        if not dest.exists():dest.write_bytes(Path(name).read_bytes())
        assert digest(dest)==sha
    freeze_json(snapshot/'manifest.json',sources)
    occupancy_rows=ledger_rows(out/'ledger/gpu_occupancy.jsonl')
    if sum(r['event']=='GPU_ACQUIRE' for r in occupancy_rows)!=sum(r['event']=='GPU_RELEASE' for r in occupancy_rows):raise RuntimeError('Unclosed GPU occupation requires engineering reconciliation')
    before_dispatch(out,stage)
    used=sum(r.get('seconds',0) for r in ledger_rows(out/'ledger/gpu_occupancy.jsonl') if r['event']=='GPU_RELEASE')
    if used>=14400:raise RuntimeError('Four GPU-hour occupancy ceiling reached')
    with open(GPU_LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        acquired=time.time();tick=time.perf_counter();append_event(out/'ledger/gpu_occupancy.jsonl',dict(event='GPU_ACQUIRE',stage=stage))
        monitor=Monitor(out,stage).start();budget=Budget(out,stage,acquired_at=acquired);status='FAILED'
        try:
            model,opt,meta,hashes,ss=checkpoint.load(args.checkpoint,'cuda',args.mode!='sample',args.mode!='sample')
            atomic_json(out/'audit'/f'{stage}_load.json',dict(metadata=meta,hashes=hashes,optimizer_restored=opt is not None))
            if args.mode=='preflight':
                from .preflight import run
                run(out,model,opt,protocol,budget)
            elif args.mode=='train':
                from .training import train
                train(out,args.arm,args.until,model,opt,meta,ss,protocol,budget)
            else:
                panels=('OLD','FRESH') if args.panel=='BOTH' else (args.panel,)
                for panel in panels:sample_manifest(out,args.label,out/'manifests'/f'{panel}.json',model,budget,hashes['checkpoint_sha256'])
            status='PASS'
        except BaseException as exc:
            atomic_json(out/'failures'/f'{stage}.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:
            budget.snapshot();monitor.close()
            append_event(out/'ledger/gpu_occupancy.jsonl',dict(event='GPU_RELEASE',stage=stage,seconds=time.perf_counter()-tick,status=status))
            resource_snapshot(out,'after_'+stage)
            atomic_json(out/'status'/f'{stage}.json',dict(status=status,seconds=time.perf_counter()-tick))
            fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
