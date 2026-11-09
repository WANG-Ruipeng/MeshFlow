"""Frozen Native T1 GPU-only guidance sampler. No CPU geometry in its loop."""
from .stable_runtime import configure_stable_runtime
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from .native_inpainting_model import (NativeInpaintingModel,native_execution,assert_native_runtime,
    native_block_forward,native_final_forward,_corners)

class PrevalidatedNative(NativeInpaintingModel):
    """Identical native arithmetic; immutable mask/value checks precede timing."""
    def bind_contract(self,y,valid,known):
        if y.shape!=(1,) or not bool((y==112).all()) or valid.shape!=(1,112) or known.shape!=(1,112):raise ValueError('Wrong fixed contract')
        if not bool(valid.all()) or not bool(known[:,:12].all()) or bool(known[:,12:].any()):raise ValueError('Wrong masks')
        self._contract=(y,valid,known,tuple(a._version for a in (y,valid,known)))
    def forward(self,x,t,y,valid_mask=None,known_mask=None):
        assert_native_runtime(self,coordinates=(x,t))
        a,b,c,versions=self._contract
        if y is not a or valid_mask is not b or known_mask is not c or versions!=tuple(z._version for z in (a,b,c)):raise RuntimeError('Bound mask/count inputs changed')
        if x.shape!=(1,112,9) or t.shape!=(1,):raise ValueError('Wrong native shape')
        batch,faces,_=x.shape
        _,hidden=self.backbone.x_embedder(x);hidden=hidden.view(batch,faces*3,-1)
        role=self.role_embedding[known_mask.long()].to(dtype=hidden.dtype)
        hidden=hidden+_corners(role)
        c_free=self.backbone.t_embedder(t);c_known=self.backbone.t_embedder(torch.ones_like(t))
        count=self.backbone.y_embedder(y//self.backbone.face_bin,False)
        c_free,c_known=c_free+count,c_known+count
        face_time=torch.where(known_mask,torch.ones_like(t[:,None]),t[:,None])
        for index,block in enumerate(self.backbone.layers,1):
            hidden=native_block_forward(block,hidden,c_free,c_known,known_mask,valid_mask,layer_index=index,face_time=face_time)
        result=native_final_forward(self.backbone.final_layer,hidden,c_free,c_known,known_mask,face_time=face_time)
        return result.view(batch,faces,9)


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(value):
    a=np.ascontiguousarray(value);return hashlib.sha256(str((str(a.dtype),tuple(a.shape))).encode()+a.tobytes()).hexdigest()

def masks(device):
    known=torch.zeros((1,112),dtype=torch.bool,device=device);known[:,:12]=True
    return torch.tensor([112],device=device),torch.ones_like(known),known


def integrate(state,velocity,C,delta=None):
    out=state.clone();out[:,12:]=state[:,12:]+velocity[:,12:]*(1/50)
    if delta is not None:out[:,12:]=out[:,12:]+delta.reshape(1,100,9)
    out[:,:12]=C.reshape(1,12,9)
    return out


def clip_displacement(g,L):
    delta=-float(1/50)*float(L)*g
    norms=torch.linalg.vector_norm(delta,dim=-1,keepdim=True)
    rho=float(.001*L)
    scale=torch.clamp(rho/norms.clamp_min(torch.finfo(delta.dtype).tiny),max=1.)
    return delta*scale,norms.squeeze(-1),scale.squeeze(-1)


def _diag(terms,grad,delta,norms,scale,mode):
    g=grad['g_selected'];zero=g.new_zeros(())
    ga=grad.get('g_A',torch.zeros_like(g));gr=grad.get('g_R',torch.zeros_like(g))
    na=torch.linalg.vector_norm(ga);nr=torch.linalg.vector_norm(gr)
    defined=(na>0)&(nr>0);cos=(ga*gr).sum()/(na*nr).clamp_min(torch.finfo(g.dtype).tiny)
    return torch.stack([terms.get('E_A',zero).detach(),terms.get('E_R',zero).detach(),na,nr,
        torch.linalg.vector_norm(g),torch.where(defined,cos,zero),defined.float(),norms.max(),
        torch.linalg.vector_norm(delta,dim=-1).max(),torch.sqrt(delta.square().sum(-1).mean()),
        (scale<1).sum().float(),torch.as_tensor(terms.get('area_floor_count',zero),device=g.device).float(),
        torch.isfinite(g).all().float(),torch.isfinite(delta).all().float(),
        torch.as_tensor(mode in ('GA','GAR'),device=g.device).float(),torch.as_tensor(mode in ('GR','GAR'),device=g.device).float()])

DIAGNOSTIC_NAMES=['E_A','E_R','gradient_A_norm','gradient_R_norm','gradient_combined_norm','gradient_cosine','cosine_defined',
    'unclipped_max_vertex_displacement','clipped_max_vertex_displacement','clipped_vertex_RMS','clipped_vertex_count',
    'area_floor_trigger_count','gradient_all_finite','delta_all_finite','A_evaluated','R_evaluated']


@torch.no_grad()
def sample_gpu(model,noise,C,L,mode,counts,out,label,category,geo_ledger):
    """One 50-forward trajectory; diagnostics and states leave GPU only at end."""
    from .gpu_guidance_energies import energy_terms,gradients_from_terms
    entry_tick=time.perf_counter()
    if mode not in ('G0','GA','GR','GAR'):raise ValueError(mode)
    if noise.shape!=(1,112,9) or noise.dtype!=np.float32 or C.shape!=(12,3,3) or C.dtype!=np.float32:raise ValueError('FP32 archive input contract')
    if counts['generation_attempts']>=59:raise RuntimeError('59 full-generation attempt cap')
    destination=Path(out)/'raw'/f'{label}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    configure_stable_runtime();device=next(model.parameters()).device
    noise_hash=array_hash(noise);C_hash=array_hash(C)
    state=torch.from_numpy(noise.copy()).to(device=device);c=torch.from_numpy(C.copy()).to(device=device)
    state[:,:12]=c.reshape(1,12,9);cn=c/float(L)
    y,valid,known=masks(device);model.bind_contract(y,valid,known)
    counts['generation_attempts']+=1;counts['generation_attempts_by_category'][category]=counts['generation_attempts_by_category'].get(category,0)+1
    counts['active_forward_category']=category
    before=counts['model_forward_attempts'];geo_before=dict(geo_ledger)
    path=[state[0].clone()];diagnostics=[];conditions=[(state[0,:12]==c.reshape(12,9)).all()]
    finite=[torch.isfinite(state).all()];velocity_requires_grad=[]
    mevents=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    fevents=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(10)] if mode!='G0' else []
    bevents=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(10)] if mode!='G0' else []
    total_start=torch.cuda.Event(enable_timing=True);total_end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();total_start.record()
    record=dict(status='RUNNING',mode=mode,category=category,raw_path=str(destination),noise_sha256=noise_hash,C_sha256=C_hash,L=float(L),N=112,K=12)
    try:
        with native_execution(model,coordinates=state) as facts:
            for k in range(50):
                t=torch.full((1,),k/50,device=device,dtype=torch.float32)
                mevents[k][0].record();velocity=model(state,t,y,valid,known).float();mevents[k][1].record()
                velocity_requires_grad.append(velocity.requires_grad)
                delta=None
                if mode!='G0' and k>=40:
                    with torch.enable_grad(),torch.autocast('cuda',enabled=False):
                        Y=(state[0,12:].reshape(100,3,3)/float(L)).detach().requires_grad_(True)
                        fevents[k-40][0].record();terms=energy_terms(Y,cn,mode=mode,ledger=geo_ledger);fevents[k-40][1].record()
                        bevents[k-40][0].record();grad=gradients_from_terms(terms,Y,mode=mode,ledger=geo_ledger);bevents[k-40][1].record()
                    delta,norms,scale=clip_displacement(grad['g_selected'],L)
                    diagnostics.append(_diag(terms,grad,delta,norms,scale,mode).detach())
                state=integrate(state,velocity,c,delta);counts['euler_updates']+=1
                path.append(state[0].clone());conditions.append((state[0,:12]==c.reshape(12,9)).all());finite.append(torch.isfinite(state).all())
        stacked=torch.stack(path);stats=torch.stack(diagnostics) if diagnostics else state.new_empty((0,len(DIAGNOSTIC_NAMES)))
        check_tensor=torch.stack([torch.stack(conditions).all(),torch.stack(finite).all()])
        total_end.record();torch.cuda.synchronize();wall=time.perf_counter()-tick
        costs=dict(model_cuda_seconds=sum(a.elapsed_time(b) for a,b in mevents)/1000,
            geometry_forward_cuda_seconds=sum(a.elapsed_time(b) for a,b in fevents)/1000,
            geometry_backward_cuda_seconds=sum(a.elapsed_time(b) for a,b in bevents)/1000,
            total_cuda_seconds=total_start.elapsed_time(total_end)/1000,wall_seconds=wall,setup_wall_seconds=tick-entry_tick,
            peak_cuda_allocated=torch.cuda.max_memory_allocated(),warmup_included=False,
            synchronization_policy='One explicit pre-timing and post-loop sync; sampler adds no per-step explicit scalar reads/CPU geometry; original masked indexing may synchronize internally',
            total_includes='Euler, GPU diagnostic tensors and51 saved GPU states; no CPU evaluation/W/export')
        transfer=time.perf_counter();actual=stacked.cpu().numpy();stats_cpu=stats.cpu().numpy();checks=check_tensor.cpu().tolist()
        costs['final_D2H_seconds']=time.perf_counter()-transfer
        costs['complete_generation_wall_seconds']=time.perf_counter()-entry_tick
        costs['model_internal_implicit_sync_profile']='NOT_RUN; unchanged original boolean indexing may synchronize'
        np.savez_compressed(destination,output=actual[-1],path=actual,geometry_diagnostics=stats_cpu)
        record.update(raw_sha256=file_hash(destination),output_sha256=array_hash(actual[-1]),costs=costs,runtime=facts,
            geometry_diagnostic_columns=DIAGNOSTIC_NAMES,geometry_diagnostics=stats_cpu.tolist(),active_k=list(range(40,50)) if mode!='G0' else [],
            geometry_ledger_delta={k:v-geo_before.get(k,0) for k,v in geo_ledger.items()},nfe=counts['model_forward_attempts']-before,
            known_all51_bitwise=bool(checks[0]),all_states_finite=bool(checks[1]),model_graph_created=any(velocity_requires_grad),
            C_export_bitwise=bool(np.array_equal(actual[-1,:12],C.reshape(12,9))),
            geometry_input='detach(current free Xt/L), no predictor/GT/backbone graph',geometry_dtype='torch.float32',
            clipping='One combined per-vertex norm clip; FP32 cap diagnostic tolerance1e-6 relative to rho',
            no_CPU_geometry_or_scalar_reads_in_timed_loop=True)
        if record['nfe']!=50 or not all(checks) or any(velocity_requires_grad):raise RuntimeError('Trajectory state/model-graph invariant failed')
        if mode!='G0' and (not np.isfinite(stats_cpu).all() or not (stats_cpu[:,12:14]==1).all()):raise RuntimeError('Nonfinite GPU geometry')
        if mode!='G0' and stats_cpu[:,8].max()>.001*L*(1+1e-6):raise RuntimeError('Norm clipping cap failed')
        if array_hash(noise)!=noise_hash or array_hash(C)!=C_hash:raise RuntimeError('Input mutation')
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):raise RuntimeError('Model gradient detected')
        counts['complete_generations']+=1;counts['completed_by_category'][category]=counts['completed_by_category'].get(category,0)+1
        record['status']='PASS'
        return record
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),nfe_attempts=counts['model_forward_attempts']-before,completed_states=len(path))
        counts.setdefault('failures',[]).append(record)
        if not destination.exists() and path:
            try:
                torch.cuda.synchronize();a=torch.stack(path).cpu().numpy();np.savez_compressed(destination,output=a[-1],path=a)
                record.update(raw_path=str(destination),raw_sha256=file_hash(destination),partial_output=True)
            except BaseException as save_error:record['partial_save_error']=repr(save_error)
        raise


