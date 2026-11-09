"""Fixed-state frozen-weight forwards only; all geometric reduction is CPU-side."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,time,traceback
import numpy as np
from .common import *

def event(path,typ,**fields):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(type=typ,utc=time.time(),**fields),ensure_ascii=False)+'\n'); f.flush();os.fsync(f.fileno())

def run(label,manifest_path):
    import torch
    from meshflow_control import runtime
    from meshflow_control.checkpoints import state_hash
    from meshflow_control.models.native import build_model,native_execution
    from meshflow_control.seam.model_adapter import seam_forward
    from r2.pilots.user_condition_finetune_v1.trainer import GPUObserver
    reg=read(OUT/'registration.json'); spec=reg['models'][label]
    package=read(manifest_path)
    rows=[r for r in package['rows'] if r['representation']==('SEAM' if label=='SEAM_S3000' else 'NATIVE')]
    expected=spec['probe_forwards']
    assert len(rows)*5==expected,(label,len(rows))
    dest=OUT/'probes'/label
    if (dest/'completion.json').exists():
        done=read(dest/'completion.json')
        for p,h in done['files'].items():assert sha(p)==h
        return done
    if dest.exists():raise RuntimeError('Unresolved probe attempt; no silent forward replay: '+str(dest))
    dest.mkdir(parents=True)
    write_new(dest/'attempt.json',dict(label=label,budget=expected,manifest_sha256=sha(manifest_path),weights=spec,probe_code_sha256=sha(__file__)))
    before=resources();write_new(dest/'resources_before.json',before)
    vals=before['gpu']['stdout'].strip().split(',')
    total=float(vals[1].strip().split()[0]); free=float(vals[3].strip().split()[0])
    assert free>max(2048,.1*total)+1.2*2800,'Insufficient VRAM; no model started'
    ledger=dest/'calls.jsonl'; count=0;model=None;tick=time.perf_counter()
    observer=GPUObserver('flow_integrity_'+label,interval=.5)
    observer.__enter__()
    try:
        torch.set_num_threads(1);runtime.configure_stable_runtime()
        assert sha(spec['path'])==spec['sha256'],'Frozen weight file changed'
        payload=torch.load(resolve(spec['path']),map_location='cpu',weights_only=True,mmap=True)
        assert int(payload['global_step'])=={'COMMON_S12000':12000,'NATIVE_S6000':18000,'COVERAGE_S1500':19500,'SEAM_S3000':21000}[label]
        assert not payload.get('EMA',False)
        assert state_hash(payload['generator_state'])==payload['generator_state_sha256']
        model=build_model(copy.deepcopy(payload['model_config']),readout_mode='none',trainable=False).eval()
        model.load_state_dict(payload['generator_state'],strict=True);model.to('cuda').set_trainable(False)
        frozen_hash=payload['generator_state_sha256'];del payload
        torch.cuda.reset_peak_memory_stats()
        write_new(dest/'resources_loaded.json',resources())
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
        for row in rows:
            path=resolve(row['trace_path']);assert sha(path)==row['trace_sha256']
            with np.load(path,allow_pickle=False) as z:s={k:z[k].copy() for k in z.files}
            times=s['times'];assert np.allclose(times,[.1,.3,.5,.7,.9],atol=1e-7)
            n=int(s['N']); known=torch.as_tensor(s['known_mask'][None],device='cuda'); valid=torch.as_tensor(s['valid_mask'][None],device='cuda')
            anchor=torch.as_tensor((s['anchor_ids']>=0)[None],device='cuda');y=torch.tensor([n],device='cuda',dtype=torch.int64)
            predictions=[]
            for ti,t in enumerate(times):
                assert s['x_by_time'][ti].shape in ((n,9),(n,3,3))
                x=torch.from_numpy(s['x_by_time'][ti].reshape(1,n,9).astype(np.float32)).cuda()
                tt=torch.tensor([float(t)],device='cuda',dtype=torch.float32)
                identity=dict(request_id=row['request_id'],representation=row['representation'],time_index=ti,t=float(t))
                event(ledger,'FORWARD_STARTED',**identity)
                with torch.no_grad(),native_execution(model,coordinates=(x,tt)):
                    pred=(seam_forward(model,x,tt,y,valid,known,anchor) if label=='SEAM_S3000' else model(x,tt,y,valid_mask=valid,known_mask=known)).float()
                torch.cuda.synchronize();event(ledger,'FORWARD_COMPLETE',**identity);count+=1
                assert torch.isfinite(pred).all(),'Nonfinite frozen probe'
                predictions.append(pred[0].cpu().numpy().copy())
                del pred,x,tt
                if count==5:
                    write_new(dest/'representative_peak.json',dict(completed=count,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),resources=resources(),observer=observer.summary()))
                    print(json.dumps(dict(label=label,completed=count,representative_peak=True)),flush=True)
            target=dest/(row['request_id']+'.npz')
            with target.open('xb') as f:np.savez_compressed(f,pred=np.stack(predictions),times=times)
            print(json.dumps(dict(label=label,completed=count,request_id=row['request_id'])),flush=True)
        assert count==expected
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        assert state_hash(model.state_dict())==frozen_hash
        result=dict(status='COMPLETE',label=label,forwards=count,backwards=0,updates=0,head_forwards=0,
            seconds=time.perf_counter()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            frozen_tensor_sha256=frozen_hash,manifest_sha256=sha(manifest_path),
            files={str(p):sha(p) for p in dest.glob('*.npz')})
        write_new(dest/'completion.json',result);return result
    except BaseException as e:
        write_new(dest/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_forwards=count,no_silent_retry=True));raise
    finally:
        observer.__exit__(None,None,None)
        write_new(dest/'resource_observation.json',dict(summary=observer.summary(),samples=observer.samples))
        del model;torch.cuda.empty_cache()
        write_new(dest/'resources_finished.json',resources())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=list(MODELS),required=True);p.add_argument('--manifest',required=True)
    a=p.parse_args();print(run(a.label,a.manifest))
