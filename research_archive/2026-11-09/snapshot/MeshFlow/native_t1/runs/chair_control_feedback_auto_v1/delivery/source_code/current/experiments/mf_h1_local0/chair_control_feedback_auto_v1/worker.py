"""One locked GPU owner; actual calls journaled before execution."""
from native_t1 import runtime
import argparse,fcntl,gc,os,time,traceback
import numpy as np
import torch
from experiments.mf_h1_local0.chair_boundary_attachment_v1.resources import Monitor,before_dispatch,resource_snapshot
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.sampling import sample_case
from .common import *
from .ledger import Budget
from . import checkpoint
from .assets import export_mesh

def sample_manifest(out,label,manifest,model,budget,checkpoint_hash,release=None):
    data=read_json(manifest);split=data['pool']
    if split=='CONFIRM':
        from .guards import confirm
        confirm(out,label,checkpoint_hash,release)
    model.set_trainable(False).eval()
    for p in model.parameters():p.grad=None
    results=[]
    for req in data['rows']:
        rid=req['id'];folder=out/'runtime'/label/rid;rp=folder/'result.json'
        assert digest(req['bank_npz'])==req['bank_sha256'] and digest(req['condition_npz'])==req['condition_sha256']
        if rp.exists():
            r=read_json(rp);assert r['status']=='PASS' and r['checkpoint_sha256']==checkpoint_hash
            saved_attempt=read_json(folder/'attempt.json')
            assert saved_attempt['bank_sha256']==req['bank_sha256'] and saved_attempt['checkpoint_sha256']==checkpoint_hash
            assert digest(folder/'raw.npz')==r['raw_sha256'];results.append(r);continue
        attempt=folder/'attempt.json'
        if attempt.exists():
            receipt=folder/'engineering_retry_authorization.json'
            if not receipt.exists() or (folder/'engineering_retry_used.json').exists():raise RuntimeError('Unfinished rollout cannot automatically retry: '+rid)
            freeze_json(folder/'engineering_retry_used.json',read_json(receipt))
        assert req['status']=='READY' and digest(req['bank_npz'])==req['bank_sha256']
        bank=dict(np.load(req['bank_npz'],allow_pickle=False));context=dict(label=label,request_id=rid,split=split)
        freeze_json(attempt,dict(context=context,checkpoint_sha256=checkpoint_hash,bank_sha256=req['bank_sha256']))
        try:
            result=budget.call('sampling',sample_case,model,bank,True,budget,context,context=context)
            C=bank['C'];N=req['N'];K=req['K'];known=np.arange(N)<K
            raw_sha=save_npz(folder/'raw.npz',output=result['raw'],path=result['states'],velocities=result['velocities'],
                C=C,z=bank['z'],initial_gaussian=bank['z'],known_mask=known,valid_mask=np.ones(N,bool))
            ex=export_mesh(result['raw'],C,known,folder/'mesh',source_condition=req['condition_npz'])
            r=dict(status='PASS',request_id=rid,label=label,split=split,checkpoint_sha256=checkpoint_hash,
                raw_sha256=raw_sha,metadata=result['metadata'],export=ex,
                C_all_steps_bitwise=True,NPZ_C_bitwise=True,PLY_C_bitwise=True,N_preserved=True,finite=True)
            freeze_json(rp,r);results.append(r)
            print('SAMPLE',label,split,len(results),len(data['rows']),rid,round(result['metadata']['seconds'],3),flush=True)
        except BaseException as exc:
            partial=getattr(exc,'partial_sampling',None)
            if partial:save_npz(folder/'partial.npz',**{k:v for k,v in partial.items() if isinstance(v,np.ndarray)})
            atomic_json(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
        finally:budget.snapshot()
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sample','train']);p.add_argument('--out',default=str(OUT))
    p.add_argument('--label');p.add_argument('--manifest',action='append');p.add_argument('--checkpoint',default=str(START))
    p.add_argument('--phase',default='explore');p.add_argument('--arm');p.add_argument('--until',type=int,default=1000);p.add_argument('--confirm-release')
    a=p.parse_args();out=Path(a.out);stage=(a.label or f'{a.phase}_{a.arm}_{a.until}')+'_'+a.mode
    runtime.configure_stable_runtime();torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    protocol=read_json(out/'protocol.json');assert protocol['start_checkpoint_sha256']==START_SHA
    for name,sha in protocol['manifest_sha256'].items():assert digest(out/'manifests'/f'{name}.json')==sha
    sources={str(p):digest(p) for p in Path(__file__).parent.glob('*.py')}
    snapshot=out/'code_snapshots'/(stage+'_'+json_hash(sources)[:12]);snapshot.mkdir(parents=True,exist_ok=True)
    for name,sha in sources.items():
        dest=snapshot/Path(name).name
        if not dest.exists():dest.write_bytes(Path(name).read_bytes())
        assert digest(dest)==sha
    freeze_json(snapshot/'manifest.json',sources)
    if a.mode=='train' and a.phase=='repeat':
        from .guards import repeat
        repeat(out,a.arm)
    before_dispatch(out,stage)
    previous=sum(r.get('seconds',0) for r in ledger_rows(out/'ledger/gpu_occupancy.jsonl') if r['event']=='GPU_RELEASE')
    if previous>=8*3600:raise RuntimeError('Eight GPU-hour occupancy limit reached')
    with open(GPU_LOCK,'a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tick=time.perf_counter();append_event(out/'ledger/gpu_occupancy.jsonl',dict(event='GPU_ACQUIRE',stage=stage))
        monitor=Monitor(out,stage).start();budget=Budget(out,stage);status='FAILED'
        try:
            model,opt,meta,hashes,ss=checkpoint.load(a.checkpoint,'cuda',a.mode=='train',a.mode=='train')
            atomic_json(out/'audit'/f'{stage}_load.json',dict(metadata=meta,hashes=hashes,optimizer_restored=opt is not None))
            if a.mode=='sample':
                for manifest in a.manifest or []:sample_manifest(out,a.label,manifest,model,budget,hashes['checkpoint_sha256'],a.confirm_release)
            else:
                from .training import train
                train(out,a.phase,a.arm,a.until,model,opt,meta,ss,protocol,budget)
            status='PASS'
        finally:
            budget.snapshot();monitor.close()
            append_event(out/'ledger/gpu_occupancy.jsonl',dict(event='GPU_RELEASE',stage=stage,seconds=time.perf_counter()-tick,status=status))
            resource_snapshot(out,'after_'+stage)
            atomic_json(out/'status'/f'{stage}.json',dict(status=status,seconds=time.perf_counter()-tick))
            fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':main()
