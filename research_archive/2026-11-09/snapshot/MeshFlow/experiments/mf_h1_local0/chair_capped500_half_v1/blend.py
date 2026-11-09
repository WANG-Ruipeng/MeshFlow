"""CPU-only arithmetic midpoint. This artifact has no training continuation state."""
from native_t1 import runtime
import gc,math,os,time
import torch
from experiments.mf_h1_local0.chair_hybrid_coupling_v1.checkpoint import build_model
from experiments.mf_h1_local0.chair_domain_control_v1.checkpoint import tensor_state_hash
from .common import *
SCHEMA='chair_capped500_half_inference_v1'
def run():
    runtime.configure_stable_runtime();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    if WEIGHT.exists():
        side=read_json(WEIGHT.with_suffix('.json'));assert digest(WEIGHT)==side['checkpoint_sha256']
        assert side['schema']==SCHEMA and side['alpha']==.5
        return dict(status='PASS',cached=True,**side)
    tick=time.perf_counter()
    assert digest(START)==START_SHA and digest(CAPPED)==CAPPED_SHA
    a=torch.load(START,map_location='cpu',weights_only=True,mmap=True)
    b=torch.load(CAPPED,map_location='cpu',weights_only=True,mmap=True)
    sa,sb=a['model'],b['model'];assert sa.keys()==sb.keys()
    assert tensor_state_hash(sa)==a['metadata']['model_state_sha256']
    assert tensor_state_hash(sb)==b['metadata']['model_state_sha256']
    model=build_model(a,'cpu',False);param=set(dict(model.named_parameters()));buffers=set(dict(model.named_buffers()))
    assert set(sa)==param|buffers
    state={};records=[];delta2=0.;half2=0.;rounding_max=0.
    for k,x in sa.items():
        y=sb[k];assert x.shape==y.shape and x.dtype==y.dtype
        if k in param:
            assert x.dtype==torch.float32 and torch.isfinite(x).all() and torch.isfinite(y).all()
            xd,yd=x.double(),y.double();ideal=(xd+yd)*.5;z=ideal.float()
            residual=(z.double()-ideal).abs().max().item()
            delta2+=(yd-xd).square().sum().item();half2+=(z.double()-xd).square().sum().item()
            rounding_max=max(rounding_max,residual)
            records.append(dict(name=k,role='parameter',elements=x.numel(),dtype=str(x.dtype),same_endpoints=torch.equal(x,y),
                source_START_sha256=array_hash(x.numpy()),source_CAPPED_sha256=array_hash(y.numpy()),
                midpoint_sha256=array_hash(z.numpy()),rounding_max_abs=residual))
        else:
            assert torch.equal(x,y),'Non-parameter buffer differs: '+k
            z=x.clone();records.append(dict(name=k,role='equal_buffer_copied',elements=x.numel(),dtype=str(x.dtype),same_endpoints=True))
        assert torch.isfinite(z).all();state[k]=z
    metadata=dict(alpha=.5,start_checkpoint_sha256=START_SHA,capped_checkpoint_sha256=CAPPED_SHA,
        source_START_model_sha256=a['metadata']['model_state_sha256'],source_CAPPED_model_sha256=b['metadata']['model_state_sha256'],
        model_state_sha256=tensor_state_hash(state),context_encoder='geo',lambda_variance=.25,training_updates=0,
        optimizer_present=False,training_step=None,inference_only=True,
        arithmetic='(START.double()+HIST_CAPPED500.double())*0.5 then one FP32 cast; equal buffers copied',
        parameter_tensors=len(param),parameter_elements=sum(sa[k].numel() for k in param),buffer_tensors=len(buffers),
        full_parameter_delta_norm=math.sqrt(delta2),midpoint_parameter_delta_norm=math.sqrt(half2),
        midpoint_delta_ratio=math.sqrt(half2/delta2),maximum_rounding_abs=rounding_max)
    payload=dict(schema=SCHEMA,model=state,metadata=metadata)
    WEIGHT.parent.mkdir(parents=True,exist_ok=True);tmp=WEIGHT.with_suffix('.pt.tmp');torch.save(payload,tmp);os.replace(tmp,WEIGHT)
    saved=torch.load(WEIGHT,map_location='cpu',weights_only=True,mmap=True)
    assert tensor_state_hash(saved['model'])==metadata['model_state_sha256']
    loaded=build_model(saved,'cpu',False);assert all(not p.requires_grad for p in loaded.parameters())
    assert tensor_state_hash(loaded.state_dict())==metadata['model_state_sha256']
    assert not torch.cuda.is_initialized()
    side=dict(schema=SCHEMA,alpha=.5,path=str(WEIGHT),checkpoint_sha256=digest(WEIGHT),bytes=WEIGHT.stat().st_size,
        metadata=metadata,source_files_unchanged=(digest(START)==START_SHA and digest(CAPPED)==CAPPED_SHA))
    assert side['source_files_unchanged'];freeze_json(WEIGHT.with_suffix('.json'),side)
    atomic_json(OUT/'audit/interpolation.json',dict(status='PASS',**side,parameter_records=records,
        CPU_seconds=time.perf_counter()-tick,model_forward_calls=0,backward_calls=0,optimizer_updates=0,CUDA_initialized=False))
    del a,b,model,loaded,state,saved;gc.collect()
    return dict(status='PASS',cached=False,path=str(WEIGHT),sha256=side['checkpoint_sha256'],metadata=metadata)
if __name__=='__main__':
    import json
    print(json.dumps(run()))
