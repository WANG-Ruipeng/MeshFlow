"""Fixed16 full Euler rollouts and a single official D0 per saved RAW."""
from .common import *
import argparse,copy,json,time,traceback,types
import numpy as np

def sample(shard):
    import torch
    from meshflow_control import runtime
    from meshflow_control.models.native import build_model,native_execution
    from meshflow_control.config import MODEL_CONFIG
    from meshflow_control.seam.model_adapter import seam_forward
    from meshflow_control.checkpoints import rng_state
    from r2.pilots.bench_adaptation_strength_v1.training import base
    from r2.pilots.seam_slot_c200_v1.generate import raw_arrays,dispatch_guard
    reg=registration();ereg=read(OUT/'evaluation/registration.json');verify(ereg['files']);verify(ereg['source_hashes'])
    status=read(OUT/'training'/ARM/'status.json');assert status['status']=='COMPLETE' and status['local_step']==1500
    cp=resolve(status['inference_export']['path']);assert sha(cp)==status['inference_export']['sha256']
    qs=reg['requests'][shard*8:(shard+1)*8];assert len(qs)==8
    dest=OUT/'execution'/f'raw_shard{shard}';done=dest/'complete.json'
    if done.exists():verify(read(done)['files']);print('ALREADY_COMPLETE',shard);return
    if (dest/'calls.jsonl').exists():raise RuntimeError('Previous incomplete GPU attempt; inspect before any replay')
    snap=resources();save(dest/'resources_before.json',snap);dispatch_guard(snap,1600)
    torch.set_num_threads(1);runtime.configure_stable_runtime()
    p=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
    model=build_model(copy.deepcopy(p.get('model_config',MODEL_CONFIG)),readout_mode='none',trainable=False).eval()
    model.load_state_dict(p['generator_state'],strict=True);model.cuda().set_trainable(False);del p
    torch.cuda.reset_peak_memory_stats();save(dest/'resources_loaded.json',resources())
    ledger=dest/'calls.jsonl';calls=0;start=time.perf_counter();receipts=[]
    try:
        for qi,q in enumerate(qs):
            rid=q['request_id'];target=OUT/'evaluation/EARLY'/rid/'raw'
            if target.exists():raise RuntimeError('Raw path exists, no silent overwrite '+rid)
            save(target/'attempt.json',dict(request_id=rid,budget=50,checkpoint_sha256=sha(cp)))
            lay=arrays(q['layout_npz']);lay['N']=int(lay['N']);lay['K']=int(lay['K'])
            fixed=lay['anchor_ids']>=0;x=np.zeros((q['N'],3,3),np.float32)
            x[q['K']:]=arrays(q['noise_npz'])['z'];x[fixed]=lay['C_vertices'][lay['anchor_ids'][fixed]]
            assert np.array_equal(x,arrays(q['baseline']['USER']['trajectory'])['states'][0])
            anchors=x[fixed].copy();path=[x.copy()];vel=[];rh=base.tree_hash(rng_state())
            known=torch.from_numpy(lay['known_mask'][None].astype(bool)).cuda()
            valid=torch.from_numpy(lay['valid_mask'][None].astype(bool)).cuda()
            fx=torch.from_numpy(fixed[None]).cuda();y=torch.tensor([q['N']],dtype=torch.int64,device='cuda')
            for step in range(50):
                xt=torch.from_numpy(x.reshape(1,q['N'],9)).cuda();t=torch.tensor([step/50],dtype=torch.float32,device='cuda')
                assert calls<400
                event(ledger,stage='STARTED',request_id=rid,step=step,call=calls+1)
                with torch.no_grad(),native_execution(model,coordinates=xt):v=seam_forward(model,xt,t,y,valid,known,fx)
                torch.cuda.synchronize();calls+=1
                event(ledger,stage='RETURNED',request_id=rid,step=step,call=calls)
                assert v.dtype==torch.bfloat16 and bool(torch.isfinite(v).all())
                vv=v.float().cpu().numpy().reshape(q['N'],3,3);vv[fixed]=0
                x=x+np.float32(.02)*vv;x[fixed]=anchors;path.append(x.copy());vel.append(vv)
            assert rh==base.tree_hash(rng_state()),'Sampling changed RNG'
            source_path=next(r['source_C_metadata'] for r in ereg['rows'] if r['request_id']==rid)
            source=arrays(source_path);raw=raw_arrays(x,lay,lay['occupancy_logits'],source)
            array_file(target/'raw.npz',raw);array_file(target/'source_condition.npz',source)
            array_file(target/'trajectory.npz',dict(states=np.stack(path),velocities=np.stack(vel),times=np.arange(51)/50))
            save(target/'receipt.json',dict(status='COMPLETE',label='EARLY',arm=ARM,request_id=rid,generator_forwards=50,
                raw_sha256=sha(target/'raw.npz'),RNG_unchanged=True,source_Y_read=False,initial_noise_exact=True,
                all_anchors_exact=all(np.array_equal(xx[fixed],anchors) for xx in path),
                files={str(p):sha(p) for p in target.glob('*.npz')}))
            receipts.append(target/'receipt.json')
            save(dest/f'resources_output{qi+1}.json',resources());print('RAW',shard,qi+1,rid,flush=True)
        assert calls==400
        save(done,dict(status='COMPLETE',shard=shard,generator_calls=calls,seconds=time.perf_counter()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_scope='Inference only including resident model; reset after loading; not whole-process VRAM',
            files={str(p):sha(p) for p in receipts}));print('SHARD_COMPLETE',shard,flush=True)
    except BaseException as exc:
        save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),returned_calls=calls,no_automatic_retry=True));raise

def barrier(root,labels,subset):
    assert root==OUT and labels==['EARLY'] and subset=='full'
    for shard in (0,1):
        r=read(OUT/'execution'/f'raw_shard{shard}'/'complete.json');verify(r['files']);assert r['generator_calls']==400
    receipts=[]
    for q in registration()['requests']:
        p=OUT/'evaluation/EARLY'/q['request_id']/'raw/receipt.json';r=read(p);verify(r['files'])
        assert r['generator_forwards']==50 and r['all_anchors_exact'] and r['RNG_unchanged'];receipts.append(p)
    save(OUT/'evaluation/EARLY/raw_complete_full.json',dict(status='COMPLETE',outputs=16,files={str(p):sha(p) for p in receipts}))
    return registration()['requests'],receipts

def post():
    from r2.pilots.bench_parts_convergence_v1 import postprocess as legacy
    ereg=read(OUT/'evaluation/registration.json');verify(ereg['files']);verify(ereg['source_hashes'])
    namespace=dict(vars(legacy));namespace['verify_raw_barrier']=barrier
    fn=types.FunctionType(legacy.run.__code__,namespace,legacy.run.__name__,legacy.run.__defaults__,legacy.run.__closure__)
    fn.__kwdefaults__=legacy.run.__kwdefaults__
    r=fn(OUT,'EARLY',['EARLY'],'full',legacy.OLD.D0,3500)
    print({k:v for k,v in r.items() if k!='files'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['sample','post']);p.add_argument('--shard',type=int,choices=(0,1),default=0);a=p.parse_args()
    sample(a.shard) if a.phase=='sample' else post()
