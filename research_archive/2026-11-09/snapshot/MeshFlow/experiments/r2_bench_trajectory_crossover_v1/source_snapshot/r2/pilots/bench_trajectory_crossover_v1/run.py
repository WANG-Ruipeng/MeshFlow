"""Exactly 25 authorized suffix forwards per request; no implicit retry."""
from .common import *
import argparse, copy, traceback
import numpy as np

def run(combo):
    reg=registration(); assert combo in reg['new_combinations']
    dest=OUT/'execution'/combo
    if (dest/'complete.json').exists():
        verify(read(dest/'complete.json')['files']); print(combo,'already complete'); return
    verify(reg['source_hashes'])
    model_name=MODELS[combo[1]]; prefix=MODELS[combo[0]]
    verify({reg['checkpoints'][model_name]:reg['files'][reg['checkpoints'][model_name]]})
    r=resources(); save(dest/'resources_before.json',r)
    if r['gpu']['returncode'] != 0: raise RuntimeError('GPU measurement unavailable')
    fields=r['gpu']['stdout'].splitlines()[0].split(',')
    total=float(fields[1]); free=float(fields[3]); headroom=max(2048,total*.1)
    assert free >= headroom+1.2*1536, ('No measured memory headroom',r)
    if list(dest.glob('calls.jsonl')): raise RuntimeError('Unfinished GPU attempt: explicit recovery decision required')
    import torch
    from meshflow_control import runtime
    from meshflow_control.config import MODEL_CONFIG
    from meshflow_control.models.native import build_model, native_execution
    from meshflow_control.seam.model_adapter import seam_forward
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays
    runtime.configure_stable_runtime(); torch.set_num_threads(1)
    p=torch.load(reg['checkpoints'][model_name],map_location='cpu',weights_only=True,mmap=True)
    model=build_model(copy.deepcopy(p.get('model_config',MODEL_CONFIG)),readout_mode='none',trainable=False).eval()
    model.load_state_dict(p['generator_state'],strict=True); model.cuda().set_trainable(False); del p
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(); save(dest/'resources_loaded.json',resources())
    rng_cpu=torch.get_rng_state().clone(); rng_gpu=torch.cuda.get_rng_state().clone()
    tick=time.perf_counter(); calls=0; receipts={}; ledger=dest/'calls.jsonl'
    for qi,q in enumerate(reg['requests']):
        rid=q['request_id']; d=OUT/'evaluation'/combo/rid/'raw'
        if d.exists(): raise RuntimeError('Refuse replay/overwrite '+str(d))
        save(d/'attempt.json',dict(combo=combo,request_id=rid,new_generator_calls=25,cached_prefix_calls=25))
        for path in (q['starts'][prefix],q['layout_npz'],q['C_export_metadata']): verify({path:reg['files'][path]})
        lay=arrays(q['layout_npz']); lay['N']=int(lay['N']); lay['K']=int(lay['K'])
        x=arrays(q['starts'][prefix])['state']; fixed=lay['anchor_ids']>=0; anchor=x[fixed].copy()
        states=[x.copy()]; velocities=[]; n=len(x)
        known=torch.from_numpy(lay['known_mask'][None].astype(bool)).cuda()
        valid=torch.from_numpy(lay['valid_mask'][None].astype(bool)).cuda()
        fm=torch.from_numpy(fixed[None]).cuda(); y=torch.tensor([n],dtype=torch.int64,device='cuda')
        try:
            for step in reg['absolute_steps']:
                xt=torch.from_numpy(x.reshape(1,n,9)).cuda(); tt=torch.tensor([step/50],dtype=torch.float32,device='cuda')
                event(ledger,stage='STARTED',request_id=rid,absolute_step=step,call=calls+1)
                with torch.no_grad(),native_execution(model,coordinates=xt):
                    v=seam_forward(model,xt,tt,y,valid,known,fm)
                torch.cuda.synchronize(); calls+=1
                event(ledger,stage='RETURNED',request_id=rid,absolute_step=step,call=calls)
                assert v.dtype==torch.bfloat16 and bool(torch.isfinite(v).all())
                vv=v.float().cpu().numpy().reshape(n,3,3); vv[fixed]=0
                x=x+np.float32(.02)*vv; x[fixed]=anchor
                assert np.isfinite(x).all(); states.append(x.copy()); velocities.append(vv)
        except BaseException:
            array_file(d/'partial_trajectory.npz',dict(states=np.stack(states),velocities=np.stack(velocities) if velocities else np.empty((0,n,3,3),np.float32)))
            raise
        assert len(states)==26 and all(np.array_equal(s[fixed],anchor) for s in states)
        assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_gpu,torch.cuda.get_rng_state())
        array_file(d/'trajectory.npz',dict(states=np.stack(states),velocities=np.stack(velocities),times=np.arange(25,51)/50))
        array_file(d/'raw.npz',raw_arrays(x,lay,lay['occupancy_logits'],arrays(q['C_export_metadata'])))
        save(d/'receipt.json',dict(status='COMPLETE',combo=combo,request_id=rid,prefix_model=prefix,suffix_model=model_name,
            generator_forwards=25,cached_prefix_steps=25,start_t=.5,initial_state_sha256=reg['files'][q['starts'][prefix]],
            fixed_anchors_exact=True,RNG_unchanged=True,source_Y_read=False,
            files={str(p):sha(p) for p in d.glob('*.npz')}))
        receipts[str(d/'receipt.json')]=sha(d/'receipt.json')
        print(combo,qi+1,'/16',rid,'calls',calls,flush=True)
        if qi in (0,7,15):
            save(dest/f'resources_after{qi+1}.json',dict(**resources(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_scope='suffix inference after model load, no backward/optimizer'))
    assert calls==400
    save(dest/'complete.json',dict(status='COMPLETE',generator_calls=calls,D0=0,head=0,updates=0,outputs=16,
        seconds=time.perf_counter()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),files=receipts))
    print(combo,'COMPLETE',calls,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('combo',choices=['CU','UC']);a=p.parse_args()
    try:run(a.combo)
    except BaseException as exc:
        event(OUT/'engineering_events.jsonl',stage='GPU_FAILURE',combo=a.combo,error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
