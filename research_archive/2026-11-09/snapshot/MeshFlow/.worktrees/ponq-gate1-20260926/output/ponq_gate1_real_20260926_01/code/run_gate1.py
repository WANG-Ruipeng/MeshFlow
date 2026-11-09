"""Frozen 72-slot ledger, 36 executable synthetic cases; no network training."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import sys,time,json,traceback,warnings,hashlib,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from common import *
import torch
from official_vstars import vstars_from_quadrics
from PoNQ_to_mesh import MeshFromPoNQ
from shapely.geometry import Polygon
from shapely.ops import triangulate

torch.set_num_threads(1);torch.set_num_interop_threads(1)
CONFIG=dict(version='PONQ_GATE1_CPU_PARTIAL_V1',seed=20260925,fit_seed=1701,perturb_seed=271828,
            samples_fit=65536,samples_eval_per_direction=8192,chamfer_iterations=80,
            budgets=[128,256],dtype='float64',svd_eps=.05,grid_scale=32,
            add_corners=True,correct_tet_color=True,add_noise=False,
            normal_rotation_degrees=[2,5],position_local_spacing_fraction=[.02,.10],
            quadric_factor_relative_frobenius=[.02,.10],
            spacing='per-element mean distance to four nearest other clean vstars, at K=256',
            position_noise='fixed length alpha*h_i, uniform random direction; translate the full quadric congruently',
            normal_noise='fixed-angle rotation about random perpendicular axis; preserve each normal magnitude',
            quadric_noise='principal PSD square root F of full homogeneous 4x4 Q; Fprime=F+alpha*normF*G/normG; Qprime=Fprime@Fprime.T; recompute vstars with frozen P and eps=.05',
            joint_order='quadric factor (2%), then translate vstar and Q (2% h_clean), then rotate normal (2deg)',
            normalization='bbox centered, bbox diagonal 2; original triangle surface used as reference',
            thresholds='No accuracy threshold invented; report topology/geometry and paired perturbation changes',
            known_limitations=['single seed','only four synthetic objects executable','geometric fitter is not proven global optimum','full4 Q noise is not identical to future 3x3 learned-factor errors'],
            run_states=['clean128','clean256','position_light','position_strong','normal_light','normal_strong','qem_light','qem_strong','joint_light'])
SPECS={
 'sphere':dict(type='trimesh_icosphere',subdivisions=4,radius=1.,expected_betti=[1,0,1]),
 'box':dict(type='box',extents=[1.4,1.,.8],expected_betti=[1,0,1]),
 'thin_u':dict(type='closed_extruded_U',polygon=[[-1,-1],[1,-1],[1,1],[.86,1],[.86,-.86],[-.86,-.86],[-.86,1],[-1,1]],extrusion=.25,expected_betti=[1,0,1]),
 'torus':dict(type='periodic_triangle_torus',major_radius=1.,minor_radius=.32,major_segments=96,minor_segments=32,expected_betti=[1,2,1]),
 'loudspeaker_local_1':dict(status='MISSING_RAW_ASSET'),
 'loudspeaker_local_2':dict(status='MISSING_RAW_ASSET'),
 'bottle_local_1':dict(status='MISSING_RAW_ASSET',priority='old radial failure plus renewed geometry qualification'),
 'bottle_local_2':dict(status='MISSING_RAW_ASSET',priority='old radial failure plus renewed geometry qualification')}

def create_mesh(name):
    spec=SPECS[name]
    if name=='sphere':m=trimesh.creation.icosphere(subdivisions=4,radius=1.)
    elif name=='box':m=trimesh.creation.box(extents=spec['extents'])
    elif name=='thin_u':
        xy=np.array(spec['polygon'],float);poly=Polygon(xy);v=np.vstack([np.c_[xy,np.full(len(xy),-.125)],np.c_[xy,np.full(len(xy),.125)]])
        faces=[]
        for t in triangulate(poly):
            if not poly.covers(t):continue
            coords=np.array(t.exterior.coords)[:3]
            ids=[int(np.argmin(np.sum((xy-q)**2,axis=1))) for q in coords]
            a,b,c=xy[ids]
            if (b-a)[0]*(c-a)[1]-(b-a)[1]*(c-a)[0]<0:ids=ids[::-1]
            faces.append(ids[::-1]);faces.append([i+len(xy) for i in ids])
        for i in range(len(xy)):
            j=(i+1)%len(xy);faces.extend([[i,j,j+len(xy)],[i,j+len(xy),i+len(xy)]])
        m=trimesh.Trimesh(v,np.array(faces),process=False)
    elif name=='torus':
        nu=spec['major_segments'];nv=spec['minor_segments'];u,v=np.meshgrid(np.arange(nu)*2*np.pi/nu,np.arange(nv)*2*np.pi/nv,indexing='ij')
        R=1.;r=.32;vtx=np.c_[((R+r*np.cos(v))*np.cos(u)).ravel(),((R+r*np.cos(v))*np.sin(u)).ravel(),(r*np.sin(v)).ravel()]
        ff=[]
        for i in range(nu):
            for j in range(nv):
                a=i*nv+j;b=((i+1)%nu)*nv+j;c=((i+1)%nu)*nv+(j+1)%nv;d=i*nv+(j+1)%nv
                ff.extend([[a,b,c],[a,c,d]])
        m=trimesh.Trimesh(vtx,np.array(ff),process=False)
    v=np.array(m.vertices);f=np.array(m.faces);center=(v.max(0)+v.min(0))/2;scale=2/np.linalg.norm(v.max(0)-v.min(0));v=(v-center)*scale
    return v,f,dict(center=center,scale=scale)

def fit(v,f,k,obj_index):
    t=time.perf_counter();s,n,_=sample(v,f,CONFIG['samples_fit'],CONFIG['fit_seed']+obj_index)
    picked=np.empty(k,dtype=int);picked[0]=0;dist=np.sum((s-s[0])**2,axis=1)
    for j in range(1,k):
        picked[j]=int(np.argmax(dist));dist=np.minimum(dist,np.sum((s-s[picked[j]])**2,axis=1))
    p=s[picked].copy();tree=cKDTree(s);trace=[];M=len(s)
    for it in range(CONFIG['chamfer_iterations']):
        d,lab=cKDTree(p).query(s,workers=1);dp,near=tree.query(p,workers=1)
        cnt=np.bincount(lab,minlength=k);sums=np.column_stack([np.bincount(lab,weights=s[:,j],minlength=k) for j in range(3)])
        trace.append(dict(iteration=it,cd=float(np.mean(d*d)+np.mean(dp*dp)),empty=int(np.sum(cnt==0))))
        p=(sums+(M/k)*s[near])/(cnt[:,None]+M/k)
    d,lab=cKDTree(p).query(s,workers=1);cnt=np.bincount(lab,minlength=k)
    planes=np.c_[n,-np.einsum('ij,ij->i',n,s)]
    Q=np.zeros((k,4,4));nn=np.zeros((k,3))
    for i in range(4):
        for j in range(4):Q[:,i,j]=np.bincount(lab,weights=planes[:,i]*planes[:,j],minlength=k)/np.maximum(cnt,1)
    for i in range(3):nn[:,i]=np.bincount(lab,weights=n[:,i],minlength=k)/np.maximum(cnt,1)
    assert np.all(cnt>0), 'EMPTY_GEOMETRY_ELEMENT'
    vs,_,eigs=vstars_from_quadrics(torch.from_numpy(Q),torch.from_numpy(p),eps=CONFIG['svd_eps'])
    return dict(p=p,n=nn,Q=Q,v=vs.numpy(),eigs=eigs.numpy(),count=cnt),trace,time.perf_counter()-t

def perturb(base,state,obj_index):
    p=base['p'].copy();Q=base['Q'].copy();n=base['n'].copy();v=base['v'].copy();eigs=base['eigs'].copy()
    h=cKDTree(base['v']).query(base['v'],k=5)[0][:,1:].mean(1)
    rngq=np.random.default_rng(CONFIG['perturb_seed']+100*obj_index+1)
    rngp=np.random.default_rng(CONFIG['perturb_seed']+100*obj_index+2)
    rngn=np.random.default_rng(CONFIG['perturb_seed']+100*obj_index+3)
    fr=None
    if state.startswith('qem') or state=='joint_light':
        a=.10 if state.endswith('strong') else .02
        w,U=np.linalg.eigh(Q);F=(U*np.sqrt(np.maximum(w,0))[:,None,:])@U.transpose(0,2,1)
        G=rngq.normal(size=F.shape);dF=a*np.linalg.norm(F,axis=(1,2))[:,None,None]*G/np.linalg.norm(G,axis=(1,2))[:,None,None]
        fr=np.linalg.norm(dF,axis=(1,2))/np.linalg.norm(F,axis=(1,2))
        F1=F+dF;Q=F1@F1.transpose(0,2,1)
        vs,_,ee=vstars_from_quadrics(torch.from_numpy(Q),torch.from_numpy(p),eps=CONFIG['svd_eps']);v=vs.numpy();eigs=ee.numpy()
    if state.startswith('position') or state=='joint_light':
        a=.10 if state.endswith('strong') else .02
        direction=rngp.normal(size=v.shape);direction/=np.linalg.norm(direction,axis=1)[:,None]
        delta=a*h[:,None]*direction
        T=np.tile(np.eye(4),(len(p),1,1));T[:,:3,3]=-delta
        Q=T.transpose(0,2,1)@Q@T;v=v+delta
    if state.startswith('normal') or state=='joint_light':
        theta=np.deg2rad(5 if state.endswith('strong') else 2)
        norm=np.linalg.norm(n,axis=1);unit=n/np.maximum(norm,1e-300)[:,None]
        r=rngn.normal(size=n.shape);ax=np.cross(unit,r);ax/=np.maximum(np.linalg.norm(ax,axis=1),1e-300)[:,None]
        n=n*np.cos(theta)+np.cross(ax,n)*np.sin(theta)
    moved=np.linalg.norm(v-base['v'],axis=1);nr=np.linalg.norm(n,axis=1);br=np.linalg.norm(base['n'],axis=1)
    rot=np.degrees(np.arccos(np.clip(np.sum(n*base['n'],axis=1)/np.maximum(nr*br,1e-300),-1,1)))
    relq=np.linalg.norm(Q-base['Q'],axis=(1,2))/np.maximum(np.linalg.norm(base['Q'],axis=(1,2)),1e-300)
    A=Q[:,:3,:3];rawA=base['Q'][:,:3,:3]
    audit=dict(position_shift_over_h_mean=float(np.mean(moved/np.maximum(h,1e-300))),position_shift_over_h_max=float(np.max(moved/np.maximum(h,1e-300))),
               normal_rotation_deg_mean=float(rot.mean()),normal_rotation_deg_max=float(rot.max()),normal_magnitude_change_max=float(np.max(abs(nr-br))),
               quadric_relative_frobenius_mean=float(relq.mean()),quadric_relative_frobenius_max=float(relq.max()),
               A_relative_frobenius_mean=float(np.mean(np.linalg.norm(A-rawA,axis=(1,2))/np.linalg.norm(rawA,axis=(1,2)))),
               factor_relative_frobenius_mean=float(np.mean(fr)) if fr is not None else 0.,
               Q_min_eigenvalue=float(np.min(np.linalg.eigvalsh(Q))),
               svd_rank_changes=int(np.sum(np.sum(eigs/eigs[:,0,None]>.05,axis=1)!=np.sum(base['eigs']/base['eigs'][:,0,None]>.05,axis=1))),
               local_h_min=float(h.min()),local_h_median=float(np.median(h)),local_h_max=float(h.max()))
    return dict(p=p,Q=Q,n=n,v=v,eigs=eigs,h=h),audit

def main():
    configpath=ROOT/'FROZEN_PROTOCOL.json'
    if configpath.exists():raise RuntimeError('Protocol already exists. Refusing overwrite or implicit rerun.')
    dump(configpath,dict(config=CONFIG,objects=SPECS,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(configpath)))
    rows=[]
    for oi,(name,spec) in enumerate(SPECS.items()):
        oi=int(spec.get('object_index',oi))
        if 'status' in spec:
            for state in CONFIG['run_states']:rows.append(dict(object=name,state=state,status='NOT_RUN_MISSING_RAW_ASSET',budget=128 if state=='clean128' else 256))
            continue
        v,f,norm=create_mesh(name);objdir=ROOT/'inputs'/name;objdir.mkdir(exist_ok=True)
        save_mesh(objdir/'reference.obj',v,f);baseaudit=mesh_audit(v,f);dump(objdir/'reference_audit.json',baseaudit)
        assert [baseaudit[f'beta{i}'] for i in range(3)]==spec['expected_betti']
        assert baseaudit['closed_oriented_2manifold'];assert baseaudit['signed_volume']>0
        dump(objdir/'identity.json',dict(spec=spec,normalization=norm,sha256=sha(objdir/'reference.obj')))
        gs,gn,_=sample(v,f,CONFIG['samples_eval_per_direction'],9901+oi)
        np.savez_compressed(objdir/'reference_eval_samples.npz',points=gs,normals=gn)
        bases={}
        for k in CONFIG['budgets']:
            print('FIT_START',name,k,flush=True);b,tr,ft=fit(v,f,k,oi);bases[k]=b
            np.savez_compressed(objdir/f'fit_{k}.npz',**b);dump(objdir/f'fit_{k}_trace.json',dict(fitting_seconds=ft,trace=tr,fit_method='deterministic symmetric-Chamfer alternating minimization'))
            print('FIT_END',name,k,round(ft,3),flush=True)
        for state in CONFIG['run_states']:
            k=128 if state=='clean128' else 256;run=ROOT/'runs'/f'{name}__{state}';run.mkdir(exist_ok=True)
            b=bases[k];params,pa=perturb(b,state,oi)
            np.savez_compressed(run/'parameters.npz',**params);dump(run/'perturbation_audit.json',pa)
            result=dict(object=name,state=state,budget=k,extraction_attempts=1,**pa)
            print('EXTRACT_START',name,state,flush=True);t=time.perf_counter();captured=[]
            try:
                with warnings.catch_warnings(record=True) as ww:
                    warnings.simplefilter('always')
                    sc=MeshFromPoNQ(torch.from_numpy(params['v']),torch.from_numpy(params['eigs']),torch.from_numpy(params['Q']),torch.from_numpy(params['n']),grid_scale=CONFIG['grid_scale'],correct_tet_color=True,compute_mincut=True)
                    vv,ff=sc.get_surface()
                    captured=[str(w.message) for w in ww]
                result['extraction_seconds']=time.perf_counter()-t
                result['warnings']=captured
                score=sc.final_scores;finite=bool(np.isfinite(score).all())
                result.update(nonfinite_cut_weights=int(np.sum(~np.isfinite(score))),negative_cut_weights=int(np.sum(score<0)),minimum_cut_weight=float(np.min(score[np.isfinite(score)])) if np.isfinite(score).any() else None,
                              qhull_omitted_input_vertices=int(len(sc.coplanar)),anchor_vertices_in_output=int(np.sum(np.max(abs(vv),axis=1)>=.999999)),tetrahedra=len(sc.simplices))
                np.savez_compressed(run/'tetrahedra.npz',vertices=sc.points,tetrahedra=sc.simplices,labels=sc.tet_colors)
                np.savez_compressed(run/'mesh.npz',vertices=vv,faces=ff)
                save_mesh(run/'mesh.obj',vv,ff)
                av,af=load_mesh(run/'mesh.obj');assert np.array_equal(av,vv) and np.array_equal(af,ff),'OBJ_EXPORT_CHANGED_GEOMETRY'
                result['status']='EXTRACTED' if finite else 'EXTRACTED_INVALID_CUT_WEIGHTS'
                result.update(mesh_audit(av,af));tm=time.perf_counter();result.update(metrics(av,af,v,f,gs,gn,8801+oi,n=CONFIG['samples_eval_per_direction']))
                result['metric_seconds']=time.perf_counter()-tm
                result['reference_betti']=spec['expected_betti'];result['betti_match']=all(result[f'beta{i}']==spec['expected_betti'][i] for i in range(3))
                result['mesh_sha256']=sha(run/'mesh.obj')
                print('EXTRACT_END',name,state,result['status'],result['faces'],'rms',round(result['rms_pct_bboxdiag'],5),'betti',result['beta0'],result['beta1'],result['beta2'],flush=True)
            except Exception as e:
                result.update(status='EXTRACTION_OR_EVALUATION_FAILED',elapsed_until_failure_seconds=time.perf_counter()-t,error=repr(e),traceback=traceback.format_exc(),warnings=captured)
                print('RUN_FAILED',name,state,repr(e),flush=True)
            dump(run/'result.json',result);rows.append(result);dump(ROOT/'LEDGER_PARTIAL.json',rows)
    dump(ROOT/'LEDGER_RAW.json',rows)
    print('ALL_REQUESTED_SLOTS_ACCOUNTED',len(rows),flush=True)
if __name__=='__main__': main()
