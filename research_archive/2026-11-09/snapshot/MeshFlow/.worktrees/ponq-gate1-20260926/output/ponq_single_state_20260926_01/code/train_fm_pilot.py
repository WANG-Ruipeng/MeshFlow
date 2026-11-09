"""Two-arm, fixed-budget unconditional CLOSED-SET flow-matching pilot.

data.npz: states[M,N,22] PHYSICAL, mean[22], scale[22]. Each x/n/F block
shares one scalar scale. Normalization occurs here; object/frame IDs never enter
the model. All targets/noise/t/couplings follow arm-independent RNG streams.
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fm_model import DeepSetsVelocity,total_velocity,validate_chart

SEED=260926
BATCH=8
WALL_LIMIT_SECONDS=600.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


class BalancedSchedule:
    def __init__(self,count,seed):
        self.count=count;self.rng=np.random.default_rng(seed);self.order=[];self.cursor=0;self.cycles=0
    def take(self,count):
        result=[]
        while len(result)<count:
            if self.cursor==len(self.order):
                self.order=self.rng.permutation(self.count).tolist();self.cursor=0;self.cycles+=1
            take=min(count-len(result),len(self.order)-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take]);self.cursor+=take
        return result
    def state(self):
        return dict(count=self.count,order=self.order,cursor=self.cursor,cycles=self.cycles,
                    numpy_generator_state=self.rng.bit_generator.state)


def coupled_batch(states,mean,scale,schedule,generator):
    """Physical xyz Hungarian; a target token's entire 22-vector moves together."""
    ids=schedule.take(BATCH)
    noise=torch.randn(BATCH,states.shape[1],22,generator=generator,dtype=torch.float32)
    times=torch.rand(BATCH,1,1,generator=generator,dtype=torch.float32)
    source_x=noise[...,:3].numpy()*scale[:3]+mean[:3]
    targets=np.empty((BATCH,states.shape[1],22),dtype=np.float32)
    for b,index in enumerate(ids):
        xyz=states[index,:,:3].astype(np.float64)
        cost=((source_x[b,:,None,:].astype(np.float64)-xyz[None,:,:])**2).sum(axis=-1)
        row,col=linear_sum_assignment(cost)
        if not np.array_equal(row,np.arange(states.shape[1])):
            raise AssertionError('Unexpected Hungarian row ordering')
        targets[b]=((states[index,col]-mean)/scale).astype(np.float32)
    return noise,times,torch.from_numpy(targets),ids


def diagnostic_summary(diag):
    cond=diag['design_condition'].detach();finite=cond[torch.isfinite(cond)]
    return dict(rank_deficient_elements=int((diag['design_rank']<3).sum()),
        design_condition_max=float(finite.max()) if finite.numel() else None,
        design_condition_nonfinite=int((~torch.isfinite(cond)).sum()),
        h_min=float(diag['h'].detach().min()),ridge_lambda_min=float(diag['ridge_lambda'].detach().min()),
        L_rms=float(diag['L'].detach().square().mean().sqrt()),
        factor_transport_rms=float(diag['transport'][...,6:].detach().square().mean().sqrt()),
        factor_raw_velocity_rms=float(diag['raw_velocity'][...,6:].detach().square().mean().sqrt()))


def main(args):
    start=time.perf_counter();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    data_path=Path(args.data).resolve()
    with np.load(data_path,allow_pickle=False) as data:
        states=np.array(data['states'],dtype=np.float64,copy=True)
        mean_np=np.array(data['mean'],dtype=np.float64,copy=True)
        scale_np=np.array(data['scale'],dtype=np.float64,copy=True)
    if states.ndim!=3 or states.shape[-1]!=22 or states.shape[1]<=16 or len(states)<1 or not np.isfinite(states).all():
        raise ValueError('states must be finite physical (M,N,22), M>=1,N>16')
    validate_chart(torch.from_numpy(mean_np),torch.from_numpy(scale_np))
    mean=torch.tensor(mean_np,dtype=torch.float32,device=device)
    scale=torch.tensor(scale_np,dtype=torch.float32,device=device)
    model=DeepSetsVelocity(width=128).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    schedule=BalancedSchedule(len(states),SEED+2)
    generator=torch.Generator(device='cpu');generator.manual_seed(SEED+1)
    source_files=[Path(__file__).resolve(),Path(__file__).with_name('fm_model.py').resolve()]
    protocol=dict(version='CLOSED_SET_FM_PILOT_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arm=args.arm,seed=SEED,steps_requested=args.steps,batch_size=BATCH,width=128,
        optimizer=dict(name='AdamW',lr=3e-4,weight_decay=1e-4,gradient_clip_norm=1.,scheduler=None),
        wall_limit_seconds=WALL_LIMIT_SECONDS,wall_limit_scope='from training entry; checked at update boundaries and before optimizer.step; final persistence is additional',
        data_path=str(data_path),data_sha256=sha(data_path),states_shape=list(states.shape),states_space='physical; normalized inside this trainer',
        state_order='x3,n3,centered factor F16 (row-major); C=F F^T',mean=mean_np.tolist(),scale=scale_np.tolist(),
        prior='independent standard Gaussian in all standardized chart coordinates',
        coupling='Hungarian squared physical xyz; reorder all 22 target channels together; same noise/time/schedule for both arms',
        probability_path='z_t=(1-t)*noise+t*assigned_target; u=assigned_target-noise; t Uniform[0,1)',
        loss='mean squared TOTAL chart velocity error over all 22 components; block weights 3/22,3/22,16/22',
        transport='current physical x and predicted physical vx; current 16NN; h=16th distance; normalized exp(-(d/h)^2); ridge=1e-3 trace(G)/3; Fdot=-diag(L,0)^T F',
        gradient_policy='backpropagate through predicted vx, current coordinates, ridge solve and factor; discrete neighbor identities are not differentiated',
        direct_control='same parameter structure/init; also computes L diagnostics; transport is not added to direct velocity or loss',
        compute_caveat='L forward diagnostics shared; residual arm has additional backward dependencies, so compute budgets are not exactly equal. Record wall time.',
        condition_inputs='FM time only; no object ID, shape frame time, endpoint state, reference mesh, or analytic motion',
        model_selection='final completed update only; no validation selection, scheduler, or loss-based early stopping',
        parameter_count=sum(p.numel() for p in model.parameters()),device=str(device),torch_version=torch.__version__,
        numpy_version=np.__version__,source_files=[dict(path=str(p),sha256=sha(p)) for p in source_files])
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    torch.save(model.state_dict(),out/'initial_model.pt')
    exposure=np.zeros(len(states),dtype=np.int64)
    completed=0;attempted=0;status='RUNNING';error=None;last_metrics=None
    trace=out/'TRAIN_TRACE.jsonl'
    stream_digest=bytes(32)
    try:
        with trace.open('x',buffering=1) as log:
            while completed<args.steps:
                if time.perf_counter()-start>=WALL_LIMIT_SECONDS:
                    status='WALL_BUDGET_EXHAUSTED';break
                attempted+=1;step_start=time.perf_counter()
                noise,times,target,ids=coupled_batch(states,mean_np,scale_np,schedule,generator)
                batch_hash=hashlib.sha256(noise.numpy().tobytes()+times.numpy().tobytes()+target.numpy().tobytes()+np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
                stream_digest=hashlib.sha256(stream_digest+bytes.fromhex(batch_hash)).digest()
                noise=noise.to(device);times=times.to(device);target=target.to(device)
                z=(1-times)*noise+times*target;true_velocity=target-noise
                optimizer.zero_grad(set_to_none=True)
                predicted,diag=total_velocity(model,z,times,mean,scale,args.arm)
                squared=(predicted-true_velocity).square();loss=squared.mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite total-velocity loss')
                loss.backward()
                gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                # The expensive computation may finish after the deadline; never
                # apply another optimizer update once that boundary is observed.
                if device.type=='cuda':torch.cuda.synchronize(device)
                if time.perf_counter()-start>=WALL_LIMIT_SECONDS:
                    status='WALL_BUDGET_EXHAUSTED';break
                optimizer.step();completed+=1
                for index in ids:exposure[index]+=1
                if device.type=='cuda':torch.cuda.synchronize(device)
                last_metrics=dict(step=completed,attempted_batches=attempted,batch_sha256=batch_hash,elapsed_seconds=time.perf_counter()-start,
                    step_seconds=time.perf_counter()-step_start,loss=float(loss.detach()),
                    x_mse=float(squared[...,:3].detach().mean()),n_mse=float(squared[...,3:6].detach().mean()),
                    F_mse=float(squared[...,6:].detach().mean()),gradient_norm_before_clip=float(gradnorm),
                    **diagnostic_summary(diag))
                log.write(json.dumps(json_safe(last_metrics),allow_nan=False)+'\n')
                if completed==1 or completed%100==0:
                    print(json.dumps(json_safe(last_metrics),allow_nan=False),flush=True)
            if completed==args.steps:status='COMPLETED'
    except Exception as exc:
        status='FAILED';error=dict(exception=repr(exc),traceback=traceback.format_exc())
    training_elapsed=time.perf_counter()-start
    rng=dict(python=random.getstate(),numpy_global=np.random.get_state(),torch_cpu=torch.get_rng_state(),
             torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             batch_generator_cpu=generator.get_state(),balanced_schedule=schedule.state())
    checkpoint=dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),rng=rng,
        protocol=protocol,protocol_sha256=sha(out/'PROTOCOL.json'),mean=mean.detach().cpu(),scale=scale.detach().cpu(),
        completed_steps=completed,attempted_batches=attempted,training_elapsed_seconds=training_elapsed,
        status=status,exposure_counts=exposure.tolist(),last_metrics=last_metrics,error=error,attempted_stream_sha256=stream_digest.hex())
    torch.save(checkpoint,out/'final.pt')
    result=dict(status=status,completed_steps=completed,requested_steps=args.steps,attempted_batches=attempted,
        finished_requested_budget=completed==args.steps,training_elapsed_seconds=training_elapsed,
        total_wall_seconds_including_save=time.perf_counter()-start,wall_limit_seconds=WALL_LIMIT_SECONDS,
        exposure_counts=exposure.tolist(),last_metrics=last_metrics,error=error,
        attempted_stream_sha256=stream_digest.hex(),final_checkpoint_sha256=sha(out/'final.pt'),initial_model_sha256=sha(out/'initial_model.pt'),
        protocol_sha256=sha(out/'PROTOCOL.json'))
    (out/'RESULT.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(json_safe(result),indent=2,allow_nan=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--arm',choices=['direct','transport_residual'],required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--steps',type=int,default=3000)
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    result=main(args)
    raise SystemExit(1 if result['status']=='FAILED' else 0)
