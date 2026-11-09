"""One lock-owning case worker; no global solver, training, OT or backward."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from native_t1 import runtime as _runtime
import argparse,fcntl,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from native_t1.model import native_execution
from native_t1.artifacts import state_sha256
from experiments.mf_h1_local0.jedge5_diagnosis.common import load_model,read_json,write_json,file_sha256
from experiments.mf_h1_local0.jedge5_diagnosis.sampling import sample,new_counts,DTypeRecorder
from .registration import OUT,CASES,INDICES,CUTS,resource_snapshot
from .math_ops import *

def tensor(a):return torch.from_numpy(a.copy()).to('cuda')
def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('Refusing existing raw path')
    np.savez_compressed(path,**{k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in arrays.items()})
    return str(path.relative_to(OUT)),file_sha256(path)

def execute(stage,cid,attempt=1):
    suffix = f'_attempt{attempt}' if attempt > 1 else ''
    target=OUT/f'workers/{stage}_case{cid}{suffix}.json'
    if target.exists():raise RuntimeError('Worker already attempted; no silent replay')
    target.parent.mkdir(parents=True,exist_ok=True)
    protocol=read_json(OUT/'registration/protocol.json')
    if stage=='suffix':
        aggregate('local');gate=read_json(OUT/'raw/local_rows.json')
        assert len(gate['rows'])==144 and all(r['next_state_bitwise'] for r in gate['rows']+gate['supplemental_rows'])
    counts=new_counts();rows=[];supplemental=[];probes=[]
    info=dict(stage=stage,case_id=cid,attempt=attempt,status='WAITING_LOCK',counts=counts,rows=rows,supplemental_rows=supplemental,probes=probes,resources_before_lock=resource_snapshot(),training_updates=0,OT_calls=0,model_backward=0)
    write_json(target,info)
    assert not torch.cuda.is_initialized()
    lock=open('/tmp/wrp_meshflow_jedge5_diag_gpu0.lock','a+')
    waiting=time.perf_counter();print(f'WAIT_LOCK {stage} case{cid}',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
    info['lock_wait_seconds']=time.perf_counter()-waiting;info['resources_after_lock']=resource_snapshot();info['CUDA_initialized_before_lock']=torch.cuda.is_initialized()
    assert not torch.cuda.is_initialized()
    gpu=info['resources_after_lock']['gpu_csv']['stdout'].strip().split(',');free=float(gpu[3]);total=float(gpu[1])
    # Two frozen inference models + context; no optimizer or gradients. Conservative 3GiB reserve.
    if free<3072+max(2048,.1*total):
        info['status']='RESOURCE_BLOCKED';write_json(target,info);return
    started=time.perf_counter();models={};identities={};events=[];recorder=None
    try:
        _runtime.configure_stable_runtime();torch.set_num_threads(1)
        tick=time.perf_counter()
        for m in ('B','J'):
            models[m],identities[m]=load_model(m);assert identities[m]['state_sha256']==next(i['state_sha256'] for i in protocol['models'] if i['label']==m)
        info['load_seconds']=time.perf_counter()-tick;info['identities']=identities
        info['resources_models_loaded']=resource_snapshot();torch.cuda.reset_peak_memory_stats()
        known=torch.zeros((1,112),device='cuda',dtype=torch.bool);known[:,:12]=True
        valid=torch.ones_like(known);y=torch.tensor([112],device='cuda')
        with np.load(protocol['noise_path'],allow_pickle=False) as z:noise=tensor(z[f'seed_{CASES[cid][2]}'])
        seen=set();info['dtypes']={};info['runtimes']={};info['model_forward_host_seconds']=0.0
        def forward(label,x,t):
            nonlocal recorder
            if counts['model_forward_attempts']>= (148 if stage=='local' else 318):raise RuntimeError('Per-case forward ceiling')
            recorder=DTypeRecorder(models[label]) if label not in seen else None
            with native_execution(models[label],coordinates=x,autocast=False) as runtime:
                info['runtimes'][label]=runtime
                ti=torch.full((1,),t,device='cuda',dtype=torch.float32)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                clock=time.perf_counter();counts['model_forward_attempts']+=1
                velocity=models[label](x,ti,y,valid,known)
                counts['model_forward_returns']+=1;info['model_forward_host_seconds']+=time.perf_counter()-clock
                end.record();events.append((begin,end))
                assert velocity.dtype==torch.float32 and not velocity.requires_grad and bool(torch.isfinite(velocity).all())
                info['raw_known_velocity_max_observed']=max(info.get('raw_known_velocity_max_observed',0.0),float(velocity[known].abs().max()))
                if recorder is not None:info['dtypes'][label]=recorder.finish(velocity,'fp32');recorder=None;seen.add(label)
                return velocity
        with torch.no_grad():
            for m in ('B','J'):
                rec=next(r for r in protocol['trajectories'] if r['case_id']==cid and r['model']==m)
                assert file_sha256(rec['path'])==rec['sha256']
                with np.load(rec['path'],allow_pickle=False) as z:arch={k:z[k].copy() for k in z.files}
                C=tensor(arch['C']);bbox=rec['bbox'];meta={k:rec[k] for k in ('model','parent','patch','seed','bbox')}
                if stage=='local':
                    for k in INDICES+[80]:
                        x=tensor(arch['states'][k:k+1]);t=k/100;v0=forward(m,x,t);pred=local_predictors(x,v0,known)
                        expected=tensor(arch['states'][k+1:k+2]);same=bits_equal(pred['x_E'],expected)
                        if not same:
                            save_npz(OUT/f'raw/mismatch/{m}_case{cid}_k{k}.npz',x_E=pred['x_E'],archived=expected,v0=v0,x=x)
                            raise RuntimeError(f'ARCHIVED_NEXT_STATE_MISMATCH {m} case{cid} k{k}')
                        vmid=forward(m,pred['x_half'],t+.005)
                        if k==80:
                            x2=euler_free(pred['x_half'],vmid,.005,known);dh=free_difference(x2,pred['x_E'],known)
                            data=dict(**pred,x_2E=x2,delta_h=dh)
                            row=dict(**meta,t=t,index=k,local=dict(delta_h=vector_stats(dh,known,bbox),h=.01,delta_h_resolution_floor=resolution_floor(pred['x_E'],x2,known=known)),next_state_bitwise=True,supplemental=True)
                        else:
                            vq=forward(m,pred['x_quarter'],t+.0025)
                            data=local_differences(x,v0,vmid,vq,known,bbox=bbox)
                            vs=forward(m,pred['x_half'],t);vt=forward(m,x,t+.005);vo=forward('J' if m=='B' else 'B',x,t)
                            decomp=decompose_velocity(v0,vmid,vs,vt,known,bbox);assert decomp['summary']['identity_within_tolerance']
                            data.update(v_quarter=vq,v_space=vs,v_time=vt,v_other=vo,**{f'a_{key}':val for key,val in decomp['terms'].items()})
                            row=dict(**meta,t=t,index=k,local=data['summary'],decomposition=decomp['summary'],same_state_model_difference=vector_stats(free_difference(vo,v0,known),known,bbox),next_state_bitwise=True,supplemental=False)
                        arrays={key:val for key,val in data.items() if torch.is_tensor(val)};arrays.update(x=x,v0=v0,v_mid=vmid,C=C)
                        path,sha=save_npz(OUT/f'raw/local/{m}_case{cid}_k{k}.npz',**arrays);row.update(states_path=path,states_sha256=sha)
                        (supplemental if k==80 else rows).append(row);write_json(target,info)
                else:
                    for k in CUTS:
                        local_path=OUT/f'raw/local/{m}_case{cid}_k{k}.npz'
                        with np.load(local_path,allow_pickle=False) as z:local={key:tensor(z[key]) for key in ('x','x_E','x_2E','delta_h')}
                        x,xE,x2,delta=[local[key] for key in ('x','x_E','x_2E','delta_h')]
                        assert bits_equal(xE,tensor(arch['states'][k+1:k+2]))
                        for eta in (1e-4,5e-5):
                            probe=make_direction_probe(x,delta,known,eta,reference_states=(xE,x2))
                            public={key:val for key,val in probe.items() if not torch.is_tensor(val) and key not in ('x_plus','x_minus','direction')}
                            if probe['status']=='RESOLVED':
                                fp=forward(m,probe['x_plus'],k/100);fm=forward(m,probe['x_minus'],k/100)
                                public.update(direction_gain(fp,fm,probe,known,bbox))
                            probes.append(dict(**meta,t=k/100,index=k,**public));write_json(target,info)
                        raw,path,audit=sample(models[m],noise,C,nfe=100,precision='fp32',counts=counts,initial_state=x2,start_step=k+1)
                        assert audit['known_all_states_bitwise'] and audit['OT_calls']==0
                        save,sha=save_npz(OUT/f'raw/suffix/{m}_case{cid}_replace{k}.npz',raw=raw,states=path,state_indices=np.arange(k+1,101),C=arch['C'])
                        diff=free_difference(tensor(raw[None]),tensor(arch['raw'][None]),known)
                        summary=propagation_summary(delta,diff,known,bbox,reference_states=(xE,x2))
                        growth=[]
                        for j,state in enumerate(path):
                            index=k+1+j;d=free_difference(tensor(state[None]),tensor(arch['states'][index:index+1]),known)
                            growth.append(dict(index=index,t=index/100,**vector_stats(d,known,bbox)))
                        rows.append(dict(**meta,t=k/100,index=k,replacement_step=k,start_step=k+1,path=save,sha256=sha,propagation=summary,growth=growth,audit=audit,original_path=rec['path'],original_sha256=rec['sha256']))
                        write_json(target,info)
                print(f'DONE {stage} case{cid} model{m} calls={counts["model_forward_attempts"]}',flush=True)
        torch.cuda.synchronize();info['local_probe_cuda_event_seconds']=sum(a.elapsed_time(b) for a,b in events)/1000
        info['network_cuda_event_seconds']=info['local_probe_cuda_event_seconds']+sum(r.get('audit',{}).get('model_forward_cuda_event_seconds',0) for r in rows)
        info['peak_cuda_allocated']=torch.cuda.max_memory_allocated();info['peak_cuda_reserved']=torch.cuda.max_memory_reserved()
        info['memory_peak_scope']='Models simultaneously resident; local worker reset after loading; suffix inherited per-suffix reset, audit retains each suffix peak'
        info['resources_after_work']=resource_snapshot()
        info['state_after']={m:state_sha256(models[m]) for m in models}
        assert all(info['state_after'][m]==identities[m]['state_sha256'] for m in models)
        expected=148 if stage=='local' else 294+sum(2 for p in probes if p['status']=='RESOLVED')
        assert counts['model_forward_attempts']==counts['model_forward_returns']==expected
        info['status']='PASS'
    except BaseException as exc:
        info.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if recorder is not None:recorder.close()
        info['lock_owned_work_seconds']=time.perf_counter()-started;write_json(target,info)
        # Process exit immediately after return releases CUDA context and flock together.
        print(f'FINAL {stage} case{cid} {info["status"]}',flush=True)

def aggregate(stage):
    workers=[]
    for i in range(6):
        suffix='_attempt2' if stage=='local' and i==0 else ''
        workers.append(read_json(OUT/f'workers/{stage}_case{i}{suffix}.json'))
    if not all(w['status']=='PASS' for w in workers):raise RuntimeError('All case groups must PASS before aggregation')
    fields=('rows','supplemental_rows','probes');out={key:sum([w[key] for w in workers],[]) for key in fields}
    out['counts']={key:sum(w['counts'][key] for w in workers) for key in new_counts()}
    write_json(OUT/f'raw/{stage}_rows.json',out)
    if stage=='suffix':write_json(OUT/'raw/sensitivity_rows.json',dict(rows=out['probes']))
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['local','suffix','aggregate_local','aggregate_suffix']);p.add_argument('--case',type=int);p.add_argument('--attempt',type=int,default=1);args=p.parse_args()
    if args.stage.startswith('aggregate_'):aggregate(args.stage.removeprefix('aggregate_'))
    else:execute(args.stage,args.case,args.attempt)

