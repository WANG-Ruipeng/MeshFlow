"""Finite local-action Gaussian policy gradient, distinct from masked FM/DPO.

The edit mask is fixed from a same-noise frozen reference RAW before any policy
action. Outside it, the conditional transition mean uses only the frozen base.
Later frozen attention can propagate local changes; endpoint protection is
therefore checked separately. No claim of preserving the ODE's marginals.
"""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse, copy, hashlib, json, math, subprocess, sys, time
import numpy as np
from . import common as C
sys.path.insert(0,str(C.ROOT/'r2/src'))
from meshflow_control import runtime as stable
import torch
from meshflow_control.config import MODEL_CONFIG
from meshflow_control.checkpoints import state_hash
from meshflow_control.models.native import native_block_forward,native_final_forward
from r2.pilots.bench_natural_n_recipe_v1.runtime import build_model,inference_layout
from r2.pilots.terminal_reward_nft_v1.model import (LoRAPolicy,AdapterLinear,TARGET_SUFFIXES,
    nft_execution,assert_nft_runtime,EXPECTED_TRAINABLE)

VERSION='bench_local_quality_rl_v1'
WINDOW=tuple(range(30,40));STEPS=50;DT=1/50;SIGMA=.005;GROUP=4;OPPORTUNITIES=16
BASE_SHA='76c42158b9016e58cc235b859f6cd5bc68ff1f4e19f3c70e4f94f676447170f4'
OPTIMIZER=dict(lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
RL_REGISTRATION=C.OUT/'rl/registration.json'


def registration_contract():
    from . import reward as Q
    files=('r2/pilots/bench_three_mechanisms_v1/rl.py','r2/pilots/bench_three_mechanisms_v1/reward.py',
        'r2/pilots/bench_three_mechanisms_v1/test_rl.py','r2/pilots/bench_three_mechanisms_v1/common.py',
        'r2/pilots/terminal_reward_nft_v1/model.py','r2/pilots/bench_natural_n_recipe_v1/runtime.py',
        'r2/tools/hinge_opposite_v1/engine.py','r2/tools/partial_edge_binding_v1/metrics.py',
        'native_t1/metric_geometry.py')
    return dict(schema=VERSION,root_registration_sha256=C.sha(C.REGISTRATION),
        schedule_sha256=C.sha(C.OUT/'data/rl_schedule.json'),source_SHA256=BASE_SHA,
        source_code_sha256={p:C.sha(C.ROOT/p) for p in files},
        opportunities=16,group=4,base_references=64,policy_rollouts=64,steps=50,
        window=list(WINDOW),sigma_state=SIGMA,max_edit_faces=8,optimizer=OPTIMIZER,
        rank=4,alpha=4,trainable_parameters=EXPECTED_TRAINABLE,
        density='joint Gaussian on all selected9M coordinates; sum10 window logp; no dimension normalization',
        update='one on-policy RLOO update; no ratio epochs; continuous[-1,1]; zero advantages skip',
        reward_version=Q.VERSION,retention=dict(surface_tolerance=Q.SURFACE_TOLERANCE,
            protected_coverage=Q.MIN_PROTECTED_COVERAGE,relative_area_change=Q.MAX_AREA_RELATIVE_CHANGE,
            new_degenerates=False,CF_count_increase=False,outside_FF_count_increase=False),
        generator_forwards_max=8320,backwards_max=640,D0_forwards=0,
        interpretation='reference surface preservation is a proxy; support/C-purpose may remain incorrect',
        failure='stop and preserve partial attempts; no implicit retry, replacement noise or budget extension')


def register_cpu():
    import unittest
    suite=unittest.defaultTestLoader.loadTestsFromName('r2.pilots.bench_three_mechanisms_v1.test_rl')
    result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite)
    if not result.wasSuccessful() or torch.cuda.is_initialized():raise RuntimeError('CPU contract gate failed')
    contract=registration_contract()
    C.write(RL_REGISTRATION,contract)
    C.write(C.OUT/'rl/cpu_contract.json',dict(status='PASS',tests=result.testsRun,
        CUDA_initialized=False,model_constructions=0,model_forwards=0,optimizer_updates=0,
        original_exact_detector_in_separate_process=True,registration_sha256=C.sha(RL_REGISTRATION)))
    return dict(status='PASS',tests=result.testsRun,rl_registration_sha256=C.sha(RL_REGISTRATION))


class LocalLoRAPolicy(LoRAPolicy):
    """Literal Native/LoRA arithmetic, with maintained natural N128..799 guard."""
    def forward(self,x,t,y,valid_mask=None,known_mask=None,*,observer=None):
        assert_nft_runtime(self,coordinates=(x,t))
        if x.ndim!=3 or x.shape[-1]!=9:raise ValueError('Expected FP32[B,N,9]')
        b,n,_=x.shape
        if t.shape!=(b,) or y.shape!=(b,) or y.dtype!=torch.int64:raise ValueError('Time/count shapes')
        if not bool(torch.isfinite(x).all() and torch.isfinite(t).all()) or bool(((t<0)|(t>1)).any()):raise ValueError('State/time finite bounds')
        for mask in (valid_mask,known_mask):
            if mask is None or mask.dtype!=torch.bool or mask.shape!=(b,n) or mask.device!=x.device:raise ValueError('Mask shape/device')
        if bool((known_mask&~valid_mask).any()) or not torch.equal(valid_mask.sum(1),y) or bool(((y<128)|(y>799)).any()):raise ValueError('N128..799')
        if bool((known_mask.sum(1)==0).any()) or bool(((valid_mask&~known_mask).sum(1)==0).any()):raise ValueError('Both C/free required')
        if self.readout_mode!='none' or len(self.readouts):raise ValueError('No new readout')
        hidden,role=self._embed(x,known_mask,valid_mask)
        cf=self.backbone.t_embedder(t);ck=self.backbone.t_embedder(torch.ones_like(t))
        count=self.backbone.y_embedder(y//self.backbone.face_bin,False)
        cf,ck=cf+count,ck+count
        ft=torch.where(known_mask,torch.ones_like(t[:,None]),t[:,None])
        if observer is not None:observer(dict(site='embedding',hidden=hidden,role=role,known_mask=known_mask,
            valid_mask=valid_mask,face_time=ft,c_free=cf,c_known=ck))
        for i,block in enumerate(self.backbone.layers,1):
            hidden=native_block_forward(block,hidden,cf,ck,known_mask,valid_mask,observer=observer,layer_index=i,face_time=ft)
        return native_final_forward(self.backbone.final_layer,hidden,cf,ck,known_mask,observer=observer,face_time=ft).view(b,n,9)


def load_policy(device='cpu',checkpoint_path=None,init_seed=None):
    """Only the retained EXPOSURE endpoint; fresh adapter unless checkpoint given."""
    path=C.SOURCE_INFERENCE
    if C.sha(path)!=BASE_SHA:raise ValueError('Frozen EXPOSURE export hash changed')
    p=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (p['schema']!='bench_four_routes_exposure_generator_v1' or p['arm']!='M_PLUS3000'
        or p['local_step']!=3000 or p['global_step']!=25500 or p.get('EMA',False)
        or p['model_config']!=MODEL_CONFIG):raise ValueError('EXPOSURE identity mismatch')
    state=p['generator_state'];digest=state_hash(state)
    if digest!=p['generator_state_sha256'] or len(state)!=194:raise ValueError('EXPOSURE state mismatch')
    with torch.device('meta'):model=build_model(copy.deepcopy(MODEL_CONFIG),readout_mode='none',trainable=False)
    model.load_state_dict(state,strict=True,assign=True)
    model.__class__=LocalLoRAPolicy;model.active_adapter='current'
    model.base_file_sha256=BASE_SHA;model.base_state_sha256=digest
    model.init_seed=C.seed('RL','adapter') if init_seed is None else int(init_seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model.init_seed)
        for layer in model.backbone.layers:
            for suffix in TARGET_SUFFIXES:
                parent,child=suffix.rsplit('.',1);owner=layer.get_submodule(parent)
                setattr(owner,child,AdapterLinear(getattr(owner,child)))
    model.to(device).eval();stable.configure_stable_runtime();assert_nft_runtime(model,require_context=False)
    if state_hash(model.base_state_dict())!=digest:raise ValueError('Adapter changed frozen base')
    if checkpoint_path is not None:
        ck=torch.load(C.resolve(checkpoint_path),map_location='cpu',weights_only=True)
        if ck['schema']!=VERSION or ck['base_file_sha256']!=BASE_SHA or ck['update_in_progress']:raise ValueError('RL checkpoint identity')
        if state_hash(ck['adapters'])!=ck['adapters_sha256']:raise ValueError('Adapter hash')
        model.load_adapter_state_dict(ck['adapters'])
    return model


def mixed_mean(state,base_velocity,current_velocity,edit_mask,known_mask):
    """Only selected conditional means depend directly on current adapter."""
    if state.requires_grad:raise ValueError('Observed rollout state must be detached')
    if edit_mask.shape!=state.shape[:2] or edit_mask.dtype!=torch.bool or bool((edit_mask&known_mask).any()):raise ValueError('Edit-mask contract')
    velocity=torch.where(edit_mask[...,None],current_velocity.float(),base_velocity.detach().float())
    return torch.where(known_mask[...,None],state,state+DT*velocity)


def joint_log_prob(action,mean,mask,sigma=SIGMA):
    if action.requires_grad:raise ValueError('Fixed actual action required')
    if action.shape!=mean.shape or mask.shape!=mean.shape[:2] or mask.dtype!=torch.bool:raise ValueError('Density shapes')
    if not bool(torch.isfinite(action).all() and torch.isfinite(mean).all()) or float(sigma)<=0:raise ValueError('Density finite/sigma')
    d=mask.sum(1).double()*9
    squared=torch.where(mask[...,None],(action.double()-mean.double()).square(),0.).sum((1,2))
    return -squared/(2*sigma*sigma)-d*math.log(sigma)-.5*d*math.log(2*math.pi)


def analytic_kl(old,new,mask,sigma=SIGMA):
    if old.requires_grad:raise ValueError('Frozen old conditional mean required')
    return torch.where(mask[...,None],(new.double()-old.double()).square(),0.).sum((1,2))/(2*sigma*sigma)


def rloo(values):
    x=np.asarray(values,np.float64)
    if x.shape!=(GROUP,) or not np.isfinite(x).all() or np.any((x<-1)|(x>1)):raise ValueError('Four finite registered rewards in [-1,1]')
    return x-(x.sum()-x)/(GROUP-1)


def _forward(model,x,step,y,valid,known,branch,event_fn=None):
    t=torch.full((1,),step/STEPS,dtype=torch.float32,device=x.device)
    if event_fn:event_fn(dict(event='FORWARD_START',step=step,branch=branch))
    with model.activate(branch),nft_execution(model,coordinates=(x,t),autocast=x.device.type=='cuda'):
        v=model(x,t,y,valid,known)
    if not bool(torch.isfinite(v).all()):raise FloatingPointError('Nonfinite velocity')
    if event_fn:event_fn(dict(event='FORWARD_RETURN',step=step,branch=branch))
    return v


def rollout(model,layout,noise,policy_noise_seed=None,*,base_raw=None,mode='current',edit_mask=None,policy_noise=None,event_fn=None):
    """Returns RAW, all51 states, audit, saved actual-transition buffer.

    base_raw authenticates the reference identity but is never a model input.
    Select masks with reward.py in a separate CPU process before invoking this.
    ZERO/reference performs the same stochastic kernel without adapter forwards.
    """
    if mode not in ('current','reference','deterministic'):raise ValueError('Unknown rollout mode')
    n,k=int(layout['N']),int(layout['K']);device=next(model.parameters()).device
    noise=np.asarray(noise)
    if noise.dtype!=np.float32 or noise.shape!=(n-k,3,3):raise ValueError('Frozen initial noise shape/dtype')
    mask=np.zeros(n,bool) if edit_mask is None else np.asarray(edit_mask,bool)
    if mask.shape!=(n,) or mask[:k].any() or mask.sum()>8:raise ValueError('Fixed max8 free-face mask')
    if mode=='deterministic' and mask.any():raise ValueError('Deterministic baseline has no edit mask')
    if mode!='deterministic' and base_raw is None:raise ValueError('Explicit same-noise reference RAW required')
    if base_raw is not None and np.asarray(base_raw).shape!=(n,3,3):raise ValueError('Reference RAW shape')
    if policy_noise is None:
        if mode=='deterministic':eps=np.zeros((10,n,3,3),np.float32)
        else:
            if policy_noise_seed is None:raise ValueError('Registered transition seed required')
            eps=np.random.RandomState(int(policy_noise_seed)).standard_normal((10,n,3,3)).astype(np.float32)
            eps[:,~mask]=0
    else:eps=np.asarray(policy_noise)
    if eps.dtype!=np.float32 or eps.shape!=(10,n,3,3) or np.any(eps[:,~mask]) or not np.isfinite(eps).all():raise ValueError('Frozen transition noise contract')
    initial=np.zeros((n,3,3),np.float32);initial[k:]=noise
    initial[:k]=layout['C_vertices'][layout['C_faces_model']]
    x=torch.from_numpy(initial.reshape(1,n,9)).to(device)
    known=torch.arange(n,device=device)[None]<k;valid=torch.ones_like(known);m=torch.from_numpy(mask[None]).to(device)
    y=torch.tensor([n],dtype=torch.int64,device=device);traj=[initial.copy()]
    records={key:[] for key in ('state_before','state_after','base_mean','old_mean','old_logp')}
    forwards=0;tick=time.perf_counter()
    zero_adapter=mode=='current' and all(not bool(module.current_b.detach().any()) for module in model.adapter_modules().values())
    zero_checks=0
    with torch.no_grad():
        for step in range(STEPS):
            vb=_forward(model,x,step,y,valid,known,'reference',event_fn);forwards+=1
            base_mean=torch.where(known[...,None],x,x+DT*vb.float())
            active=step in WINDOW and bool(mask.any()) and mode!='deterministic'
            if active and mode=='current':
                vc=_forward(model,x,step,y,valid,known,'current',event_fn);forwards+=1
                if zero_adapter:
                    if not torch.equal(vc,vb):raise RuntimeError('Fresh zero adapter differs from frozen base velocity')
                    zero_checks+=1
                mean=mixed_mean(x,vb,vc,m,known)
            else:mean=base_mean
            if active:
                ei=torch.from_numpy(eps[step-WINDOW[0]].reshape(1,n,9)).to(device)
                action=torch.where(m[...,None],mean+SIGMA*ei,mean)
                lp=joint_log_prob(action,mean,m)
                for key,value in [('state_before',x),('state_after',action),('base_mean',base_mean),('old_mean',mean),('old_logp',lp)]:
                    records[key].append(value.detach().cpu().numpy().copy())
            else:action=mean
            if not bool(torch.isfinite(action).all()):raise FloatingPointError('Nonfinite policy state')
            x=action.detach();traj.append(x.cpu().numpy().reshape(n,3,3).copy())
    trajectory=np.stack(traj)
    if any(s[:k].tobytes()!=initial[:k].tobytes() for s in trajectory):raise AssertionError('C changed')
    buffer={key:np.stack(v) if v else np.empty((0,),np.float32) for key,v in records.items()}
    buffer.update(edit_mask=mask,N=np.asarray(n),K=np.asarray(k),steps=np.asarray(WINDOW if records['old_logp'] else [],np.int64))
    return trajectory[-1],trajectory,dict(version=VERSION,mode=mode,generator_forwards=forwards,logical_steps=50,
        D0_forwards=0,edit_faces=int(mask.sum()),gaussian_dimension_per_step=int(mask.sum()*9),
        window_steps=len(records['old_logp']),sigma_state=SIGMA,seconds=time.perf_counter()-tick,C_all51_exact=True,
        transition_noise_sha256=hashlib.sha256(eps.tobytes()).hexdigest(),
        zero_adapter_equal_velocity_checks=zero_checks,
        local_contract='outside-mask conditional mean frozen; later cross-face propagation not prohibited'),buffer


def make_optimizer(model):return torch.optim.AdamW(model.current_parameters(),**OPTIMIZER)


def update_once(model,optimizer,buffers,rewards,event_fn=None):
    """One on-policy RLOO step; no likelihood-ratio epochs or masked FM."""
    if len(buffers)!=GROUP:raise ValueError('Exactly4 trajectories per opportunity')
    advantages=rloo(rewards);device=next(model.parameters()).device
    params=model.current_parameters();before=state_hash(model.adapter_state_dict())
    optimizer.zero_grad(set_to_none=True);forward_count=backward_count=0;old_error=0.;rows=[]
    active=any(float(a)!=0 and len(b['steps']) for a,b in zip(advantages,buffers))
    if not active:return dict(optimizer_steps=0,skipped=True,reason='ZERO_ADVANTAGE_OR_EMPTY_MASK',rewards=list(rewards),
        advantages=advantages.tolist(),generator_forwards=0,backwards=0,adapter_before=before,adapter_after=before,old_logp_error=0.)
    for g,b in enumerate(buffers):
        n,k=int(b['N']),int(b['K']);m=torch.from_numpy(b['edit_mask'][None]).to(device)
        known=torch.arange(n,device=device)[None]<k;valid=torch.ones_like(known);y=torch.tensor([n],dtype=torch.int64,device=device)
        for index,step in enumerate(b['steps']):
            x=torch.from_numpy(b['state_before'][index]).to(device);act=torch.from_numpy(b['state_after'][index]).to(device)
            bm=torch.from_numpy(b['base_mean'][index]).to(device);om=torch.from_numpy(b['old_mean'][index]).to(device)
            vc=_forward(model,x,int(step),y,valid,known,'current',event_fn);forward_count+=1
            mu=torch.where(m[...,None],x+DT*vc.float(),bm)
            if not torch.equal(mu.detach(),om):raise RuntimeError('On-policy old mean replay is not exact')
            lp=joint_log_prob(act,mu,m)
            err=abs(float(lp.detach())-float(b['old_logp'][index].reshape(-1)[0]));old_error=max(old_error,err)
            if err!=0:raise RuntimeError('Old Gaussian logp replay differs')
            loss=-(float(advantages[g])/GROUP)*lp.sum()
            if event_fn:event_fn(dict(event='BACKWARD_START',trajectory=g,step=int(step)))
            loss.backward();backward_count+=1
            if event_fn:event_fn(dict(event='BACKWARD_RETURN',trajectory=g,step=int(step)))
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('Nonfinite/missing LoRA gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False))
    if event_fn:event_fn(dict(event='OPTIMIZER_STEP_START'))
    optimizer.step()
    if event_fn:event_fn(dict(event='OPTIMIZER_STEP_RETURN'))
    # Full selected-coordinate conditional KL, summed over10 steps, averaged4 trajectories.
    kl=0.
    with torch.no_grad():
        for g,b in enumerate(buffers):
            n,k=int(b['N']),int(b['K']);m=torch.from_numpy(b['edit_mask'][None]).to(device)
            known=torch.arange(n,device=device)[None]<k;valid=torch.ones_like(known);y=torch.tensor([n],dtype=torch.int64,device=device)
            for index,step in enumerate(b['steps']):
                x=torch.from_numpy(b['state_before'][index]).to(device);bm=torch.from_numpy(b['base_mean'][index]).to(device)
                om=torch.from_numpy(b['old_mean'][index]).to(device)
                vc=_forward(model,x,int(step),y,valid,known,'current',event_fn);forward_count+=1
                mu=torch.where(m[...,None],x+DT*vc.float(),bm);kl+=float(analytic_kl(om,mu,m).sum())/GROUP
    if not math.isfinite(kl):raise FloatingPointError('Nonfinite conditional KL')
    after=state_hash(model.adapter_state_dict())
    return dict(optimizer_steps=1,skipped=False,rewards=list(rewards),advantages=advantages.tolist(),
        generator_forwards=forward_count,backwards=backward_count,gradient_norm_before_clip=norm,
        conditional_KL_sum_window_average_group=kl,KL_is_terminal_mesh_KL=False,old_logp_error=old_error,
        adapter_before=before,adapter_after=after)


def _cpu_reward(mode,base,current,K,dest):
    cmd=[sys.executable,'-m','r2.pilots.bench_three_mechanisms_v1.reward',mode,'--base',str(base),'--K',str(K),'--out',str(dest)]
    if current is not None:cmd+=['--current',str(current)]
    if mode=='evaluate':cmd+=['--reference',str(Path(base).parent/'reference.json')]
    result=subprocess.run(cmd,cwd=str(C.ROOT),capture_output=True,text=True)
    if result.returncode:raise RuntimeError('CPU reward failed: '+result.stderr[-6000:])
    return C.read(dest)


def _save_npz(path,arrays):
    path=Path(path)
    if path.exists():raise RuntimeError('No silent overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)


def train(resource_go_path):
    """Registered16 opportunity controller. A partial attempt requires explicit recovery."""
    reg=C.read(C.REGISTRATION);regsha=C.sha(C.REGISTRATION);go=C.read(resource_go_path)
    if (go.get('status')!='GO' or go.get('arm')!='RL16' or go.get('registration_sha256')!=regsha
        or go.get('rl_registration_sha256')!=C.sha(RL_REGISTRATION)):raise ValueError('Resource/registration GO required')
    expected=json.loads(json.dumps(registration_contract()))
    if C.read(RL_REGISTRATION)!=expected:raise ValueError('Frozen RL source/contract changed')
    if C.read(C.OUT/'rl/cpu_contract.json')['status']!='PASS':raise ValueError('CPU contract missing')
    schedule_path=C.OUT/'data/rl_schedule.json';schedule=C.read(schedule_path)['opportunities']
    if len(schedule)!=16 or [s['opportunity'] for s in schedule]!=list(range(16)):raise ValueError('Fixed opportunity schedule')
    if any(len(s['trajectories'])!=4 for s in schedule):raise ValueError('Exactly4 trajectories')
    work=C.OUT/'rl/training';work.mkdir(parents=True,exist_ok=True)
    if (work/'complete.json').exists():return C.read(work/'complete.json')
    completed=[]
    for i in range(16):
        folder=work/f'opportunity_{i:03d}'
        if (folder/'complete.json').exists():completed.append(C.read(folder/'complete.json'))
        elif folder.exists():raise RuntimeError('Unresolved partial RL opportunity; no silent retry')
        else:break
    model=load_policy('cuda',work/f'opportunity_{len(completed)-1:03d}/checkpoint.pt' if completed else None)
    optimizer=make_optimizer(model);updates=0
    if completed:
        ck=torch.load(work/f'opportunity_{len(completed)-1:03d}/checkpoint.pt',map_location='cpu',weights_only=True)
        if ck['registration_sha256']!=regsha or ck['schedule_sha256']!=C.sha(schedule_path):raise ValueError('Resume registration/schedule')
        optimizer.load_state_dict(ck['optimizer_state']);updates=int(ck['optimizer_updates'])
    initial_base_hash=model.base_state_sha256
    for item in schedule[len(completed):]:
        i=item['opportunity'];folder=work/f'opportunity_{i:03d}';folder.mkdir()
        C.write(folder/'attempt.json',dict(status='STARTED',opportunity=i,registration_sha256=regsha,
            schedule_sha256=C.sha(schedule_path),resource_go=str(resource_go_path),wall_time=time.time()))
        calls=folder/'calls.jsonl';event=lambda value:C.event(calls,dict(opportunity=i,**value))
        row=item['case'];path=C.resolve(row['case_path'])
        if C.sha(path)!=row['case_sha256']:raise ValueError('Frozen training case changed')
        with np.load(path,allow_pickle=False) as z:
            # Deliberately do not read full_target, source_faces, Y or source labels.
            inputs={key:z[key].copy() for key in ('C_vertices','C_faces_original','C_faces_model')}
        layout=inference_layout(**inputs,total_N=int(row['N']));n,k=layout['N'],layout['K']
        tick=time.perf_counter();buffers=[];rewards=[];forward_total=0;sample_rows=[]
        torch.cuda.reset_peak_memory_stats()
        for spec in item['trajectories']:
            j=spec['index'];td=folder/f'trajectory_{j}';td.mkdir()
            noise=np.random.RandomState(spec['noise_seed']).standard_normal((n-k,3,3)).astype(np.float32)
            _save_npz(td/'initial_noise.npz',dict(z=noise,seed=np.asarray(spec['noise_seed'])))
            base,bt,ba,_=rollout(model,layout,noise,mode='deterministic',event_fn=event)
            _save_npz(td/'base_raw.npz',dict(raw=base));_save_npz(td/'base_trajectory.npz',dict(state=bt))
            ref=_cpu_reward('reference',td/'base_raw.npz',None,k,td/'reference.json')
            mask=np.asarray(ref['edit_mask'],bool)
            eps=np.random.RandomState(spec['policy_noise_seed']).standard_normal((10,n,3,3)).astype(np.float32)
            eps[:,~mask]=0
            _save_npz(td/'transition_noise.npz',dict(epsilon=eps,seed=np.asarray(spec['policy_noise_seed']),edit_mask=mask))
            raw,traj,audit,buf=rollout(model,layout,noise,spec['policy_noise_seed'],base_raw=base,edit_mask=mask,policy_noise=eps,event_fn=event)
            _save_npz(td/'policy_raw.npz',dict(raw=raw));_save_npz(td/'policy_trajectory.npz',dict(state=traj))
            _save_npz(td/'buffer.npz',buf)
            reward=_cpu_reward('evaluate',td/'base_raw.npz',td/'policy_raw.npz',k,td/'reward.json')
            buffers.append(buf);rewards.append(float(reward['reward']));forward_total+=ba['generator_forwards']+audit['generator_forwards']
            sr=dict(trajectory=j,noise_seed=spec['noise_seed'],policy_noise_seed=spec['policy_noise_seed'],base_audit=ba,
                policy_audit=audit,reward=reward['reward'],protection_pass=reward['protection_pass'],mask_faces=int(mask.sum()))
            C.write(td/'complete.json',dict(status='COMPLETE',**sr));sample_rows.append(sr)
        update=update_once(model,optimizer,buffers,rewards,event);updates+=update['optimizer_steps']
        adapters=model.adapter_state_dict();ck=dict(schema=VERSION,base_file_sha256=BASE_SHA,base_state_sha256=initial_base_hash,
            registration_sha256=regsha,schedule_sha256=C.sha(schedule_path),completed_opportunities=i+1,
            optimizer_updates=updates,update_in_progress=False,adapters=adapters,adapters_sha256=state_hash(adapters),
            optimizer_state=optimizer.state_dict(),contract=dict(window=list(WINDOW),sigma=SIGMA,group=GROUP,max_edit_faces=8))
        torch.save(ck,folder/'checkpoint.pt')
        result=dict(status='COMPLETE',opportunity=i,parent_id=row['parent_id'],condition_id=row['condition_id'],N=n,K=k,
            samples=sample_rows,update=update,generator_forwards=forward_total+update['generator_forwards'],
            backwards=update['backwards'],seconds=time.perf_counter()-tick,checkpoint_sha256=C.sha(folder/'checkpoint.pt'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            resource=C.resources())
        C.write(folder/'complete.json',result);completed.append(result)
        print(json.dumps(dict(opportunity=i,rewards=rewards,update=update['optimizer_steps'],seconds=result['seconds'])),flush=True)
    if state_hash(model.base_state_dict())!=initial_base_hash:raise RuntimeError('Frozen base changed')
    result=dict(status='COMPLETE',version=VERSION,opportunities=16,policy_trajectories=64,reference_trajectories=64,
        optimizer_updates=updates,skipped_opportunities=16-updates,generator_forwards=sum(r['generator_forwards'] for r in completed),
        backwards=sum(r['backwards'] for r in completed),D0_forwards=0,registration_sha256=regsha,
        final_checkpoint=str(work/'opportunity_015/checkpoint.pt'),final_checkpoint_sha256=C.sha(work/'opportunity_015/checkpoint.pt'),
        base_unchanged=True,no_best_selection=True)
    if result['generator_forwards']>8320 or result['backwards']>640:raise RuntimeError('Registered RL compute budget exceeded')
    C.write(work/'complete.json',result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--resource-go');ap.add_argument('--register-cpu',action='store_true')
    args=ap.parse_args()
    if args.register_cpu:
        if args.train or args.resource_go:ap.error('CPU registration cannot start training')
        print(json.dumps(register_cpu()));return
    if not args.train or not args.resource_go:ap.error('Only explicit --train --resource-go starts a model')
    print(json.dumps(train(args.resource_go)))

if __name__=='__main__':main()
