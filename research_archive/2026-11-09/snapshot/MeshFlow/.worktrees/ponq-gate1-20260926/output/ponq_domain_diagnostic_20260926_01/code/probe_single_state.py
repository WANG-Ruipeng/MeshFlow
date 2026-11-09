"""Held-noise conditional-path probes, not free generation or model selection."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import torch
from common import ROOT,dump,sha
from fm_model import DeepSetsVelocity,total_velocity
from single_state_common import arrays,semantic,geometry,target_reference
from sample_single_state import assignment

def main():
    data=arrays(ROOT/'data.npz');target=data['states'][0];p=arrays(ROOT/'probe_noise.npz')
    noise,times=p['noise'],p['times'];out=ROOT/'probes';out.mkdir()
    ck=torch.load(ROOT/'training/direct/final.pt',map_location='cuda',weights_only=False)
    model=DeepSetsVelocity(128).cuda();model.load_state_dict(ck['model_state_dict']);model.eval()
    mean=torch.tensor(data['mean'],dtype=torch.float32,device='cuda');scale=torch.tensor(data['scale'],dtype=torch.float32,device='cuda')
    reference=target_reference();diag=float(np.linalg.norm(np.ptp(target[:,:3],axis=0)))
    rows=[];geometries=[];all_endpoints=[];all_perms=[];all_paths=[]
    for i in range(32):
        perm=assignment(noise[i,:,:3].astype(np.float64)*data['scale'][:3]+data['mean'][:3],target[:,:3])
        assigned=target[perm];zt=torch.tensor(((assigned-data['mean'])/data['scale']).astype(np.float32),device='cuda')[None]
        z0=torch.tensor(noise[i:i+1],device='cuda');u=zt-z0;per_end=[];per_path=[]
        for ti,tvalue in enumerate(times):
            t=float(tvalue);z=(1-t)*z0+t*zt
            with torch.no_grad():velocity,_=total_velocity(model,z,torch.tensor([t],device='cuda'),mean,scale,'direct')
            predicted=(z+(1-t)*velocity)*scale+mean
            physical=predicted[0].cpu().numpy().astype(np.float64);path=(z*scale+mean)[0].cpu().numpy().astype(np.float64)
            sq=(velocity-u).square()[0]
            row=dict(probe_id=i,time=t,generated=False,oracle_assisted=True,
                velocity_mse=float(sq.mean()),velocity_x_mse=float(sq[:,:3].mean()),velocity_n_mse=float(sq[:,3:6].mean()),velocity_F_mse=float(sq[:,6:].mean()),
                endpoint=semantic(physical,assigned,diag),zero_velocity_endpoint=semantic(path,assigned,diag))
            rows.append(row);per_end.append(physical);per_path.append(path)
            if abs(t-.9)<1e-6:
                for domain in ['fixed','enclosing']:
                    gr=geometry(physical,domain,out/f'endpoint09_{i:03d}_{domain}',reference,405000+i,
                        dict(kind='conditional_path_endpoint',probe_id=i,time=t,generated=False,oracle_assisted=True),perm)
                    geometries.append(gr)
                print('PROBE_ENDPOINT09',i,[r['valid'] for r in geometries[-2:]],flush=True)
        all_endpoints.append(np.stack(per_end));all_paths.append(np.stack(per_path));all_perms.append(perm)
    np.savez_compressed(out/'probe_states.npz',endpoint_predictions=np.stack(all_endpoints),path_states=np.stack(all_paths),target_permutations=np.stack(all_perms),times=times)
    dump(out/'PROBE_ROWS.json',rows);dump(out/'GEOMETRY_ROWS.json',geometries)
    dump(out/'SUMMARY.json',dict(probe_count=32,times=times.tolist(),semantic_records=len(rows),geometry_extractions=len(geometries),
        valid_at09_by_domain={d:sum(r['valid'] for r in geometries if r['domain']==d) for d in ['fixed','enclosing']},
        noise_sha256=sha(ROOT/'probe_noise.npz'),checkpoint_sha256=sha(ROOT/'training/direct/final.pt'),
        note='t=.9 input contains 90% target already. A passing endpoint probe alone does not prove solver error. Zero-velocity semantic baselines retained.'))
if __name__=='__main__':main()