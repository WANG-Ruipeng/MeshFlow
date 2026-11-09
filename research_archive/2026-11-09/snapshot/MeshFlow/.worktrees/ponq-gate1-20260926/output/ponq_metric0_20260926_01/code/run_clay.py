"""Frozen 72-extraction controlled sculpting study; no learned updates."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,time,json,datetime,traceback,hashlib
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict,Counter
from common import ROOT,dump,sha,save_mesh,load_mesh,mesh_audit,metrics,sample
from clay_maps import build_map,deform
from clay_reference import prepare_reference,refit_frame,shrink_plane_quadrics
from safe_extract import extract

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ARMS=['freeze','transport','refit']
TIMES=np.linspace(0,1,8)
SHRINK=.7

def translation(x):
    h=np.broadcast_to(np.eye(4),(len(x),4,4)).copy();h[:,:3,3]=x;return h
def centered(Q,v):
    h=translation(v);return h.transpose(0,2,1)@Q@h
def world(C,v):
    h=translation(-v);return h.transpose(0,2,1)@C@h
def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sorted_simplex(a):
    a=np.sort(np.asarray(a,dtype=np.int64),axis=1)
    return a[np.lexsort(a.T[::-1])]
def strict_links(f):
    links=defaultdict(list)
    for tri in f:
        for i in range(3):links[int(tri[i])].append(tuple(sorted((int(tri[(i+1)%3]),int(tri[(i+2)%3])))))
    bad=[]
    for v,edges in links.items():
        counts=Counter(edges);adj=defaultdict(set)
        for a,b in edges:adj[a].add(b);adj[b].add(a)
        seen=set();stack=[next(iter(adj))]
        while stack:
            x=stack.pop()
            if x not in seen:seen.add(x);stack.extend(adj[x])
        if len(adj)<3 or any(a==b or c!=1 for (a,b),c in counts.items()) or any(len(n)!=2 for n in adj.values()) or len(seen)!=len(adj):bad.append(v)
    return dict(strict_noncycle_vertex_count=len(bad),strict_noncycle_vertex_ids=bad,strict_all_vertex_links_simple_cycles=not bad)

def main():
    parent=Path(sys.argv[1]).resolve()
    sources=[('sphere','sphere_squash','synthetic',0,0),('thin_u','thin_u_squeeze','synthetic',2,4),
             ('bottle_local_1','bottle_bend','real',6,1)]
    config=dict(version='PONQ_CONTROLLED_CLAY_V1',network_training_updates=0,budget=256,budgets=[256],
        times=TIMES.tolist(),run_states=ARMS,state_groups={'clean':[],'mild':[]},perturbation_expectations={},
        common_uniform_shrink=SHRINK,grid_scale=32,device='CPU',dtype='float64',numerical_threads=1,
        samples_fit=65536,samples_eval_per_direction=8192,reference='Frozen piecewise-linear deformed subdivided source mesh, no per-frame normalization',
        freeze='Fixed local full C=(A,g,c), normal unchanged; Q follows displaced center',
        transport='Local affine energy pushforward at each original v, n inverse-transpose direction with original mean magnitude; eigenvalues recomputed; no SVD relocation',
        refit='Reference-assisted frozen sample membership, source material sample barycentrics, area-change weights, unit current triangle planes; no relocation',
        shared_initialization='Common reference refit at t=0, checked against shrunk original fit; exactly reused for all arms at t=0',
        reference_levels={'sphere':0,'thin_u':4,'bottle_local_1':1},
        interpretation='Controlled deterministic geometry paths. Not material simulation, generated trajectories, feedback corrections, or all-time topology guarantees.',
        guard='Same safe extractor for every arm. Only 64*eps arccos boundary clamp; actual graph capacities checked before mincut. No Q/mesh repair.',
        thresholds='No new RMS quality threshold. Preserve failed geometry and actual face counts.',
        known_limitations=['three development objects','one trajectory per object','one sample seed','piecewise-linear reference','refit has target access','intermediate mesh validity measured not guaranteed','no process feedback operator tested','both Q and normal update together, not an A-only causal ablation','same 256 input elements, actual output faces can vary','transport is local affine energy pullback, not exact nonlinear Taylor or unit-plane refit','phase timers are diagnostics, not matched-cost or speed benchmarks','each frame maps from the fixed original state, no cumulative integration or process correction comparison'])
    specs={};source_files=[]
    for name,path,study,oi,level in sources:
        src=parent/f'ponq_gate1_{study}_20260926_01'/'inputs'/name
        for file in ['reference.obj','fit_256.npz']:
            p=src/file;source_files.append(dict(path=str(p),sha256=sha(p)))
        for fi,t in enumerate(TIMES):specs[f'{name}_frame{fi:02d}']=dict(source_object=name,path_name=path,frame=fi,t=float(t),expected_betti=[1,0,1])
    config['source_files']=source_files
    dump(ROOT/'FROZEN_PROTOCOL.json',dict(config=config,objects=specs,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
    rows=[];initial_checks=[];path_specs={}
    for name,path,study,oi,levels in sources:
        print('PATH_START',name,path,flush=True)
        src=parent/f'ponq_gate1_{study}_20260926_01'/'inputs'/name
        for file in ['reference.obj','fit_256.npz']:
            p=src/file
            expected=next(item['sha256'] for item in source_files if item['path']==str(p))
            assert sha(p)==expected,('source_changed',str(p))
        gv,gf=load_mesh(src/'reference.obj')
        with np.load(src/'fit_256.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        started=time.perf_counter()
        prepared=prepare_reference(gv,gf,old['p'],oi,levels,shrink=SHRINK)
        rv=prepared['vertices'];rf=prepared['faces'];assoc=prepared['association']
        spec=build_map(path,rv);path_specs[name]=spec
        ref0=refit_frame(rv,rf,assoc)
        originalQ=shrink_plane_quadrics(old['Q'],SHRINK)
        init=dict(object=name,Q_max_abs_difference=float(np.max(abs(ref0['Q']-originalQ))),
            n_max_abs_difference=float(np.max(abs(ref0['n']-old['n']))),preparation=prepared['diagnostics'],
            preparation_seconds=time.perf_counter()-started)
        initial_checks.append(init);dump(ROOT/'audit/INITIALIZATION_CHECKS.json',initial_checks)
        assert init['Q_max_abs_difference']<1e-11 and init['n_max_abs_difference']<1e-11,init
        v0=old['v']*SHRINK;p0=old['p']*SHRINK;Q0=ref0['Q'];n0=ref0['n'];C0=centered(Q0,v0)
        base=dict(v=v0,p=p0,Q=Q0,n=n0,eigs=np.linalg.svd(Q0[:,:3,:3],compute_uv=False))
        binddir=ROOT/'path_inputs'/name;binddir.mkdir(parents=True)
        np.savez_compressed(binddir/'base.npz',**base)
        arrays={k:v for k,v in assoc.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(binddir/'sample_association.npz',**arrays)
        dump(binddir/'path.json',spec);save_mesh(binddir/'source_refined.obj',rv,rf)
        previous={}
        for fi,t in enumerate(TIMES):
            obj=f'{name}_frame{fi:02d}';dest=ROOT/'inputs'/obj;dest.mkdir()
            frame_start=time.perf_counter()
            current_ref,refJ=deform(rv,float(t),spec)
            vv,J=deform(v0,float(t),spec);pp,_=deform(p0,float(t),spec)
            min_det=float(min(np.linalg.det(J).min(),np.linalg.det(refJ).min()))
            margin=float(1-max(abs(current_ref).max(),abs(vv).max()))
            assert min_det>0 and margin>0.02,(obj,min_det,margin)
            save_mesh(dest/'reference.obj',current_ref,rf)
            dump(dest/'reference_audit.json',mesh_audit(current_ref,rf))
            gt,gn,_=sample(current_ref,rf,8192,42000+100*oi+fi)
            np.savez_compressed(dest/'reference_eval_samples.npz',points=gt,normals=gn)
            refit_start=time.perf_counter();refit=refit_frame(current_ref,rf,assoc);refit_seconds=time.perf_counter()-refit_start
            # Reference discretization diagnostic; never used to select an arm or change a path.
            base_sample_key=next((key for key in ['base_samples','samples_base','source_samples_shrunk'] if key in assoc),None)
            approximation=None
            if base_sample_key:
                exact_samples,_=deform(assoc[base_sample_key],float(t),spec)
                err=np.linalg.norm(exact_samples-refit['samples'],axis=1)
                approximation=dict(rms=float(np.sqrt(np.mean(err**2))),p95=float(np.quantile(err,.95)),maximum=float(err.max()))
            dump(dest/'identity.json',dict(path=spec,frame=fi,t=float(t),reference_sha256=sha(dest/'reference.obj'),
                minimum_J_determinant=min_det,corner_margin=margin,reference_approximation=approximation,
                refit_diagnostics=refit['diagnostics'],refit_seconds=refit_seconds))
            frame_prepare_seconds=time.perf_counter()-frame_start
            frame_vertex_hash=array_hash(vv)
            for arm in ARMS:
                print('EXTRACT_START',obj,arm,flush=True)
                rd=ROOT/'runs'/f'{obj}__{arm}';rd.mkdir()
                result=dict(object=obj,source_object=name,path_name=path,frame=fi,t=float(t),state=arm,budget=256,
                    extraction_attempts=1,reference_betti=[1,0,1],input_vertex_sha256=frame_vertex_hash,
                    minimum_J_determinant=min_det,corner_margin=margin,frame_prepare_seconds=frame_prepare_seconds,
                    refit_seconds=refit_seconds if arm=='refit' else 0.,reference_assisted=arm=='refit')
                start=time.perf_counter()
                try:
                    if fi==0:Q=Q0.copy();nn=n0.copy()
                    elif arm=='refit':Q=refit['Q'].copy();nn=refit['n'].copy()
                    else:
                        C=C0.copy();nn=n0.copy()
                        if arm=='transport':
                            inv=np.linalg.inv(J);H=np.broadcast_to(np.eye(4),(len(vv),4,4)).copy();H[:,:3,:3]=inv
                            C=H.transpose(0,2,1)@C0@H
                            direction=np.einsum('nji,nj->ni',inv,n0)
                            nn=direction*(np.linalg.norm(n0,axis=1)/np.linalg.norm(direction,axis=1))[:,None]
                        Q=world(C,vv)
                    eigs=np.linalg.svd(Q[:,:3,:3],compute_uv=False)
                    params=dict(v=vv.copy(),p=pp.copy(),n=nn,Q=Q,eigs=eigs)
                    result['parameter_update_seconds']=time.perf_counter()-start
                    result['Q_min_symmetric_eigenvalue']=float(np.linalg.eigvalsh((Q+Q.transpose(0,2,1))/2).min())
                    result['normal_magnitude_min']=float(np.linalg.norm(nn,axis=1).min())
                    np.savez_compressed(rd/'parameters.npz',**params)
                    et=time.perf_counter();ev,ef,ids,trace,audit=extract(params);result['extraction_seconds']=time.perf_counter()-et
                    np.savez_compressed(rd/'extractor_trace.npz',**trace)
                    dump(rd/'extractor_audit.json',audit)
                    result.update(audit)
                    save_mesh(rd/'mesh.obj',ev,ef);lv,lf=load_mesh(rd/'mesh.obj')
                    assert np.array_equal(lv,ev) and np.array_equal(lf,ef)
                    np.savez_compressed(rd/'mesh.npz',vertices=ev,faces=ef,input_vertex_ids=ids)
                    global_faces=sorted_simplex(ids[ef]);fs=set(map(tuple,global_faces))
                    result['surface_input_faces_sha256']=array_hash(global_faces)
                    result['output_input_vertex_sha256']=array_hash(np.sort(ids))
                    if arm in previous:
                        previous_frame,prev=previous[arm]
                        result['previous_successful_frame']=previous_frame
                        result['previous_comparison_is_adjacent']=previous_frame==fi-1
                        result['surface_faces_added_from_previous']=len(fs-prev)
                        result['surface_faces_removed_from_previous']=len(prev-fs)
                    previous[arm]=(fi,fs)
                    result.update(mesh_audit(ev,ef));result.update(strict_links(ef))
                    mt=time.perf_counter();result.update(metrics(ev,ef,current_ref,rf,gt,gn,52000+100*oi+fi,n=8192));result['metric_seconds']=time.perf_counter()-mt
                    diag=np.linalg.norm(np.ptp(current_ref,axis=0))
                    result['reference_bbox_diagonal']=float(diag);result['rms_absolute']=result['rms_pct_bboxdiag']*diag/100
                    result['p95_absolute']=result['p95_pct_bboxdiag']*diag/100
                    result['betti_match']=[result[f'beta{i}'] for i in range(3)]==[1,0,1]
                    result['mesh_sha256']=sha(rd/'mesh.obj');result['status']='EXTRACTED'
                except Exception as exc:
                    result.update(status='EXTRACTION_OR_EVALUATION_FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_until_failure_seconds=time.perf_counter()-start)
                    if hasattr(exc,'audit'):dump(rd/'extractor_audit.json',exc.audit)
                    if hasattr(exc,'trace_arrays'):np.savez_compressed(rd/'extractor_trace.npz',**exc.trace_arrays)
                result['arm_wall_seconds']=time.perf_counter()-start
                dump(rd/'result.json',result);rows.append(result);dump(ROOT/'LEDGER_PARTIAL.json',rows)
                print('EXTRACT_END',obj,arm,result['status'],result.get('rms_pct_bboxdiag'),flush=True)
    dump(ROOT/'PATHS.json',path_specs);dump(ROOT/'LEDGER_RAW.json',rows)
    assert len(rows)==72
    print('ALL_72_SLOTS_RECORDED',flush=True)

if __name__=='__main__':main()
