"""Read-only forward recorder around the unchanged diagnostic Euler loop."""
from native_t1.runtime import configure_stable_runtime
from pathlib import Path
import argparse,time,traceback,subprocess
import numpy as np
import torch
from experiments.mf_h1_local0.jedge5_diagnosis.common import ROOT,ASSETS,load_model,load_cases,noise_bank,read_json,write_json,file_sha256,state_sha256,array_hash
from experiments.mf_h1_local0.jedge5_diagnosis.sampling import sample,new_counts
OLD=ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
OUT=ROOT/'native_t1/runs/jedge5_blackbox_20261002_113600'

def planned_rows(label):
    return [dict(model=label,parent=0,patch=p,seed=s,precision='bf16',nfe=n) for p in (0,4) for s in (10401,10402) for n in (50,100)]

def key(row):
    return f"{row['model']}_P{row['parent']}_patch{row['patch']}_seed{row['seed']}_{row['precision']}_Euler{row['nfe']}"

class VelocityRecorder:
    def __init__(self,model):
        self.v=[];self.t=[];self.copy_seconds=0.;self.handle=model.register_forward_hook(self.hook)
    def hook(self,module,args,output):
        tick=time.perf_counter()
        self.v.append(output.detach().float()[0].cpu().numpy().copy())
        self.t.append(float(args[1][0].detach().cpu()))
        self.copy_seconds+=time.perf_counter()-tick
        # Returning None leaves the actual forward result unchanged.
    def close(self):self.handle.remove()

def check_euler(states,velocities,nfe):
    if states.shape!=(nfe+1,112,9) or velocities.shape!=(nfe,112,9):raise ValueError('Recorder length/shape')
    predicted=states[:-1,12:]+velocities[:,12:]*np.float32(1/nfe)
    if not np.array_equal(predicted.view(np.uint32),states[1:,12:].view(np.uint32)):
        raise RuntimeError('Recorded actual velocities do not reproduce Euler arithmetic')
    return True

def register():
    dest=OUT/'registration/protocol.json'
    if dest.exists():raise FileExistsError(dest)
    files={str(p.relative_to(ROOT)):file_sha256(p) for p in [ROOT/'native_t1/model.py',ROOT/'native_t1/sampling.py',ROOT/'native_t1/runtime.py',ROOT/'native_t1/losses/edge.py',ROOT/'native_t1/objectives.py',ROOT/'experiments/mf_h1_local0/jedge5_diagnosis/sampling.py']}
    protocol=dict(status='REGISTERED',task='Targeted zero-training JEdge5 black-box recorder',previous_run=str(OLD),
        cases=[dict(parent=0,patch=p,seed=s) for p in (0,4) for s in (10401,10402)],models=['B','J'],precisions=['bf16'],nfe=[50,100],
        complete_replays=16,replay_forwards=1200,probe_forwards_max=240,total_model_forwards_max=1440,optimizer_updates=0,backprop_through_model=0,new_training=0,
        reason='Recover every actual velocity and state for existing selected cases; no new endpoints or seed search.',
        parity='All archived states and endpoints bitwise equal to previous A, unchanged Euler loop, observer hook returns None.',
        probes=dict(sigma_std_model_coordinates=1e-4,free_only=True,direction_seeds=[61401,61402],directions='Independent unnormalized Gaussian; same directions across all cases/models/precisions.',
            selection='Three states per model/case on BF16/50: max positive aligned-D increment start; max consecutive velocity-change; k=25. Deduplicate in order, fill by D increment descending with k ascending ties.',
            same_state_precisions=['bf16','fp32'],forwards_per_state=10,states=24,additional_full_rollouts=0),
        caveats=['Observed Euler sensitivity is not proof of ODE divergence or stiffness.','Edge/L5 and training gate are not executed at inference.','D_edge uses fixed source correspondence; posthoc nearest-C target switches are not actual loss branches.','Free slots have no GT vertex/face identity in generation.','Directional response is not a Jacobian spectral norm or condition number.'],
        source_hashes=files,noise_archive_sha256=file_sha256(OLD/'registration/noises.npz'),assets={k:dict(path=str(v['path']),state_sha256=v['state'],file_sha256=file_sha256(v['path'])) for k,v in ASSETS.items() if k in ('B','J')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dispatch='Start one already-budgeted B replay worker, observe resource peak, then launch J if headroom permits; CPU audits overlap with thread limits.')
    write_json(dest,protocol);print('REGISTERED',flush=True)

def run(label):
    protocol=read_json(OUT/'registration/protocol.json');configure_stable_runtime();torch.set_num_threads(1)
    folder=OUT/'trajectories'/label;folder.mkdir(parents=True,exist_ok=True);dest=folder/'results.json'
    if dest.exists():raise FileExistsError(dest)
    counts=new_counts();result=dict(status='RUNNING',model=label,counts=counts,rows=[],seconds=0)
    write_json(dest,result);tick=time.perf_counter()
    oldrows={key(r):r for r in read_json(OLD/'A'/label/'results.json')['rows']}
    cases=load_cases();bank=noise_bank(OLD);model=None
    try:
        model,identity=load_model(label);result['identity']=identity
        for row in planned_rows(label):
            name=key(row);oldrow=oldrows[name];oldpath=OLD/oldrow['path']
            if file_sha256(oldpath)!=oldrow['sha256']:raise ValueError('Old raw changed')
            with np.load(oldpath,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
            C=torch.from_numpy(cases[0,row['patch']]['C'].copy()).cuda();noise=bank[row['seed']].clone().cuda()
            observer=VelocityRecorder(model)
            try:raw,states,audit=sample(model,noise,C,nfe=row['nfe'],precision='bf16',counts=counts)
            finally:observer.close()
            velocities=np.stack(observer.v);times=np.asarray(observer.t,dtype=np.float32)
            check_euler(states,velocities,row['nfe'])
            if not np.array_equal(states[old['state_indices']].view(np.uint32),old['states'].view(np.uint32)):raise RuntimeError('Hook replay changed archived states')
            if not np.array_equal(raw.view(np.uint32),old['raw'].view(np.uint32)):raise RuntimeError('Hook replay changed endpoint')
            path=folder/(name+'.npz')
            with path.open('xb') as stream:np.savez_compressed(stream,raw=raw,states=states,velocities=velocities,times=times,state_indices=np.arange(row['nfe']+1),C=C.cpu().numpy())
            rec=dict(**row,path=str(path.relative_to(OUT)),sha256=file_sha256(path),audit=audit,recorded_velocity_copy_seconds=observer.copy_seconds,
                archived_state_count=len(old['state_indices']),archived_states_bitwise=True,endpoint_bitwise=True,raw_known_velocity_zero=bool(np.all(velocities[:,:12]==0)),known_coordinates_clamped_bitwise=True,euler_reconstruction_bitwise=True,old_path=str(oldpath),old_sha256=oldrow['sha256'])
            result['rows'].append(rec);result['seconds']=time.perf_counter()-tick;write_json(dest,result)
            print('RECORDED',name,'seconds',round(audit['seconds'],3),'peak_reserved',audit['peak_cuda_reserved'],flush=True)
        result['final_state_sha256']=state_sha256(model)
        if result['final_state_sha256']!=ASSETS[label]['state']:raise RuntimeError('Frozen state changed')
        if counts['model_forward_attempts']!=600 or counts['complete_rollouts']!=8:raise RuntimeError('Budget mismatch')
        result['status']='PASS'
    except BaseException as exc:
        result['status']='STOPPED';result['failure']=dict(error=repr(exc),traceback=traceback.format_exc())
        if hasattr(exc,'partial_path'):np.savez_compressed(folder/'failed_partial_states.npz',states=exc.partial_path)
        raise
    finally:
        result['seconds']=time.perf_counter()-tick;write_json(dest,result)
        if model is not None:del model
        torch.cuda.empty_cache()

def test():
    class Mock(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.1),requires_grad=False);self.experiment_trainable=False;self.eval()
        def forward(self,x,t,y,valid,known):return torch.where(known[...,None],torch.zeros_like(x),self.p*x+t[:,None,None])
    model=Mock();g=torch.Generator().manual_seed(4);noise=torch.randn(1,112,9,generator=g);C=torch.randn(12,3,3,generator=g)
    counts=new_counts();a,path,info=sample(model,noise,C,nfe=50,precision='fp32',cpu_mock=True,observe_dtypes=False,counts=counts)
    obs=VelocityRecorder(model)
    try:b,path2,info2=sample(model,noise,C,nfe=50,precision='fp32',cpu_mock=True,observe_dtypes=False,counts=counts)
    finally:obs.close()
    assert np.array_equal(path.view(np.uint32),path2.view(np.uint32));check_euler(path2,np.stack(obs.v),50)
    write_json(OUT/'registration/recorder_cpu_test.json',dict(status='PASS',hook_returns_no_replacement=True,all_states_bitwise=True,euler_reconstruction_bitwise=True,cpu_mock_forward_calls=100,real_model_forwards=0,optimizer_updates=0))
    print('CPU RECORDER PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','test','run']);p.add_argument('--model',choices=['B','J']);a=p.parse_args()
    if a.command=='register':register()
    elif a.command=='test':test()
    else:run(a.model)
