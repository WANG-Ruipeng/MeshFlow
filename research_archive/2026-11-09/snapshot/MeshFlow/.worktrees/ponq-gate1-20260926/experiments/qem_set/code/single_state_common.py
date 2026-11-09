"""Shared semantic and full-geometry diagnostics for one frozen FM target."""
import time,traceback
from pathlib import Path
import numpy as np
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics,sample
from run_clay import strict_links,sorted_simplex,array_hash
from exact_audit import exact_mesh
from safe_extract_domain import extract_domain,corner_domain
from fm_state import decode_state

def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
def semantic(state,target,bbox_diagonal):
    if not np.isfinite(state).all():return dict(finite=False)
    P,n,F=state[:,:3],state[:,3:6],state[:,6:].reshape(-1,4,4)
    Pt,nt,Ft=target[:,:3],target[:,3:6],target[:,6:].reshape(-1,4,4)
    C=F@F.transpose(0,2,1);Ct=Ft@Ft.transpose(0,2,1)
    nl=np.linalg.norm(n,axis=1);ntl=np.linalg.norm(nt,axis=1);good=(nl>0)&(ntl>0)
    angles=np.degrees(np.arccos(np.clip(np.sum(n[good]*nt[good],axis=1)/(nl[good]*ntl[good]),-1,1)))
    row=dict(finite=True,P_rms=float(np.sqrt(np.mean(np.sum((P-Pt)**2,axis=1)))),
        P_rms_pct_bbox=100*float(np.sqrt(np.mean(np.sum((P-Pt)**2,axis=1))))/bbox_diagonal,
        n_mse=float(np.mean((n-nt)**2)),n_magnitude_rms=float(np.sqrt(np.mean((nl-ntl)**2))),
        n_oriented_angle_mean_deg=float(angles.mean()) if len(angles) else None,
        n_oriented_angle_p95_deg=float(np.quantile(angles,.95)) if len(angles) else None,n_zero_count=int((~good).sum()),
        F_mse=float(np.mean((F-Ft)**2)),F_relative=rel(F,Ft),C_relative=rel(C,Ct))
    e=np.linalg.svd(C[:,:3,:3],compute_uv=False)[:,0];et=np.linalg.svd(Ct[:,:3,:3],compute_uv=False)[:,0]
    row['zero_spatial_spectrum_count']=int(((e<=0)|(et<=0)).sum())
    if np.all(e>0) and np.all(et>0):row['C_spectral_relative']=rel(C/e[:,None,None],Ct/et[:,None,None])
    if np.all(good) and np.all(e>0) and np.all(et>0):
        p=decode_state(state,normalize_normals=False);q=decode_state(target,normalize_normals=False)
        row['world_Q_relative']=rel(p['Q'],q['Q'])
        row['world_Q_spectral_relative']=rel(p['Q']/p['eigs'][:,0,None,None],q['Q']/q['eigs'][:,0,None,None])
    return row

def geometry(state,domain,directory,reference,seed,metadata,permutation=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();row=dict(**metadata,domain=domain,status='NOT_RUN',valid=False,extraction_completed=False)
    np.savez_compressed(directory/'state.npz',physical_state=state)
    try:
        params=decode_state(state,normalize_normals=False);np.savez_compressed(directory/'parameters.npz',**params)
        row['parameters_sha256']=sha(directory/'parameters.npz')
        corners,domain_audit=corner_domain(params['v'],domain);row['domain_audit']=domain_audit
        np.savez_compressed(directory/'corners.npz',corners=corners)
        et=time.perf_counter();v,f,ids,trace,audit=extract_domain(params,domain)
        row.update(audit);row['extraction_completed']=True;row['extraction_seconds']=time.perf_counter()-et
        save_mesh(directory/'mesh.obj',v,f);np.savez_compressed(directory/'mesh.npz',vertices=v,faces=f,input_vertex_ids=ids)
        np.savez_compressed(directory/'extractor_trace.npz',**trace);dump(directory/'extractor_audit.json',audit)
        row.update(mesh_audit(v,f));row.update(strict_links(f));row.update(exact_mesh(v,f))
        cap=row.get('graph_capacity_stats',{})
        row['valid']=bool(row['closed_oriented_2manifold'] and row['strict_all_vertex_links_simple_cycles'] and
            [row['beta'+str(k)] for k in range(3)]==[1,0,1] and row['exact_intersection_pairs']==0 and
            row['exact_degenerate_faces']==0 and row['duplicate_faces']==0 and cap.get('nonfinite_count')==0 and cap.get('negative_count')==0)
        row['geometry_valid']=row['valid'];row['mesh_sha256']=sha(directory/'mesh.obj')
        row['surface_input_faces_sha256']=array_hash(sorted_simplex(ids[f]))
        if permutation is not None:
            mapping=np.r_[permutation,np.arange(len(permutation),len(permutation)+8)]
            row['surface_original_target_ids_sha256']=array_hash(sorted_simplex(mapping[ids[f]]))
        gv,gf,gt,gn=reference;mt=time.perf_counter()
        row.update(metrics(v,f,gv,gf,gt,gn,seed,n=8192));row['metric_seconds']=time.perf_counter()-mt
        row['status']='EXTRACTED'
    except Exception as exc:
        row.update(status='DECODE_EXTRACTION_OR_EVALUATION_FAILED',valid=False,error=repr(exc),traceback=traceback.format_exc())
        if hasattr(exc,'audit'):dump(directory/'extractor_audit.json',exc.audit)
        if hasattr(exc,'trace_arrays'):np.savez_compressed(directory/'extractor_trace.npz',**exc.trace_arrays)
    row['wall_seconds']=time.perf_counter()-start;dump(directory/'result.json',row)
    return row

def target_reference():
    v,f=load_mesh(ROOT/'target'/'canonical_mesh.obj');points,normals,_=sample(v,f,8192,304206)
    return v,f,points,normals