"""Paired unconditional final-checkpoint sampling; all 64 outcomes retained."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics,sample
from run_clay import strict_links
from exact_audit import exact_mesh
from safe_extract import extract
from fm_state import decode_state
from fm_model import DeepSetsVelocity,total_velocity

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def site_cd(a,b):
    return float(.5*(np.mean(cKDTree(a).query(b)[0]**2)+np.mean(cKDTree(b).query(a)[0]**2)))

def main():
    torch.set_num_threads(1);data=arrays(ROOT/'data.npz')
    device='cuda';mean=torch.as_tensor(data['mean'],dtype=torch.float32,device=device)
    scale=torch.as_tensor(data['scale'],dtype=torch.float32,device=device)
    rng=np.random.default_rng(260927);noise=rng.standard_normal((32,256,22)).astype(np.float32)
    np.savez_compressed(ROOT/'sample_noise.npz',noise=noise)
    rows=[];timings={};rollouts={}
    for arm in ['direct','transport_residual']:
        checkpoint=ROOT/'training'/arm/'final.pt'
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        model=DeepSetsVelocity(width=128).to(device)
        model.load_state_dict(state['model_state_dict']);model.eval()
        assert state['status']=='COMPLETED' and state['completed_steps']==3000
        final=[];actual_nfe=[];rollout_errors=[];diagnostics=[]
        sync();start=time.perf_counter()
        for i in range(32):
            z=torch.as_tensor(noise[i:i+1],device=device).clone();nfe=0;failure=None
            try:
                with torch.no_grad():
                    for k in range(64):
                        t=torch.full((1,),k/64.,device=device);nfe=k+1
                        velocity,diag=total_velocity(model,z,t,mean,scale,arm)
                        z=z+velocity/64.
                        if not torch.isfinite(z).all():raise FloatingPointError('Nonfinite Euler state')
            except Exception as exc:
                failure=dict(error=repr(exc),traceback=traceback.format_exc(),failed_NFE=nfe)
                failed_dir=ROOT/'samples'/f'{arm}_{i:03d}';failed_dir.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(failed_dir/'failed_rollout_state.npz',standardized_state=z[0].detach().cpu().numpy())
            actual_nfe.append(nfe);rollout_errors.append(failure)
            final.append((z*scale+mean)[0].cpu().numpy() if failure is None else np.full((256,22),np.nan,dtype=np.float32))
        sync();timings[arm]=dict(sampling_32_seconds=time.perf_counter()-start,NFE=64,solver='explicit Euler',batch=1)
        endpoints=np.stack(final);rollouts[arm]=endpoints
        np.savez_compressed(ROOT/(arm+'_endpoints.npz'),states=endpoints)
        for i,physical in enumerate(endpoints):
            rd=ROOT/'samples'/f'{arm}_{i:03d}';rd.mkdir(parents=True,exist_ok=True)
            row=dict(arm=arm,sample_id=i,paired_noise_sha256=sha(ROOT/'sample_noise.npz'),checkpoint_sha256=sha(checkpoint),
                NFE=actual_nfe[i],requested_NFE=64,rollout_failure=rollout_errors[i],input_elements=256,normal_normalization=False,repair_performed=False,extraction_completed=False)
            start=time.perf_counter()
            try:
                if rollout_errors[i] is not None:raise RuntimeError('Rollout failed; see saved rollout_failure and state')
                assert np.isfinite(physical).all(),'Nonfinite final state'
                distances=np.array([site_cd(physical[:,:3],s[:,:3]) for s in data['states']])
                j=int(np.argmin(distances));name=str(data['object_names'][j])
                row.update(nearest_target_by_site_cd=name,site_cd_to_nearest_target=float(distances[j]),
                    all_target_site_cd=distances.tolist(),input_sites_outside_corner_cube=int(np.any(abs(physical[:,:3])>.5,axis=1).sum()))
                p=decode_state(physical,normalize_normals=False);np.savez_compressed(rd/'parameters.npz',**p)
                st=time.perf_counter();v,f,ids,trace,audit=extract(p);row['extraction_completed']=True;row['extraction_seconds']=time.perf_counter()-st
                save_mesh(rd/'mesh.obj',v,f);np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
                np.savez_compressed(rd/'extractor_trace.npz',**trace);dump(rd/'extractor_audit.json',audit);row.update(audit)
                row.update(mesh_audit(v,f));row.update(strict_links(f));row.update(exact_mesh(v,f))
                row['valid']=bool(row['closed_oriented_2manifold'] and row['strict_all_vertex_links_simple_cycles']
                    and [row['beta'+str(k)] for k in range(3)]==[1,0,1] and row['exact_intersection_pairs']==0
                    and row['exact_degenerate_faces']==0 and row['duplicate_faces']==0)
                gv,gf=load_mesh(ROOT/'targets'/name/'mesh.obj');gt,gn,_=sample(gv,gf,8192,29000+j)
                m=metrics(v,f,gv,gf,gt,gn,30000+i,n=8192)
                row['geometry_vs_site_selected_target']=m
                row['mesh_sha256']=sha(rd/'mesh.obj');row['status']='EXTRACTED'
            except Exception as exc:
                row.update(status='DECODE_EXTRACTION_OR_EVALUATION_FAILED',valid=False,error=repr(exc),traceback=traceback.format_exc())
                if hasattr(exc,'audit'):dump(rd/'extractor_audit.json',exc.audit)
                if hasattr(exc,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**exc.trace_arrays)
            row['decode_extract_audit_evaluate_seconds']=time.perf_counter()-start
            rows.append(row);dump(rd/'result.json',row);dump(ROOT/'LEDGER_PARTIAL.json',rows)
            print('SAMPLE',arm,i,row['status'],row.get('valid'),row.get('faces'),flush=True)
    dump(ROOT/'LEDGER.json',rows)
    summary={}
    for arm in ['direct','transport_residual']:
        rr=[r for r in rows if r['arm']==arm]
        near=[r.get('site_cd_to_nearest_target') for r in rr if r.get('site_cd_to_nearest_target') is not None]
        finite=[s for s in rollouts[arm] if np.isfinite(s).all()]
        pairwise=[site_cd(a[:,:3],b[:,:3]) for j,a in enumerate(finite) for b in finite[:j]]
        summary[arm]=dict(requested_samples=32,valid_samples=sum(bool(r['valid']) for r in rr),
            extracted=sum(r['extraction_completed'] for r in rr),pipeline_completed=sum(r['status']=='EXTRACTED' for r in rr),finite_endpoint_count=len(finite),metric_evaluable_count=len(near),timing=timings[arm],
            actual_face_counts=[r.get('faces') for r in rr],nearest_target_site_cd_mean=float(np.mean(near)) if near else None,
            nearest_target_assignments={s:sum(r.get('nearest_target_by_site_cd')==s for r in rr) for s in data['object_names'].tolist()},
            pairwise_sample_site_cd_quantiles=np.quantile(pairwise,[0,.05,.5,.95,1]).tolist() if pairwise else None,
            exact_mesh_hash_unique=len({r['mesh_sha256'] for r in rr if r.get('mesh_sha256')}))
    dump(ROOT/'SUMMARY_FM.json',dict(by_arm=summary,paired_sample_count=32,
        scope='One paired training seed, final checkpoints, 24 related targets from 3 known parents. Closed-set feasibility only, not generalization or superiority evidence.',
        geometry_metric='Distance to target selected by symmetric squared point-set nearest distance across all 24 targets; not a minimization over mesh surface distance.',
        training_complete_required=True))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()