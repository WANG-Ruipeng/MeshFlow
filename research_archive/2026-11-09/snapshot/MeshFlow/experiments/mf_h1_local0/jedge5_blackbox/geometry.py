"""CPU-only face attribution, shape, noise and trusted context geometry recorder.

No model loading, GPU, training, OT, geometry repair, or inference correspondence.
"""
from __future__ import annotations
import argparse
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import trimesh
from experiments.mf_h1_local0.jedge5_diagnosis.common import (
    ROOT, load_cases, read_json, write_json, file_sha256)
from experiments.mf_h1_local0.evaluate import sample_mesh_surface, surface_distance_squared

PRIOR = ROOT/'native_t1/runs/jedge5_diagnosis_20261002_102215'
DEFAULT = ROOT/'native_t1/runs/jedge5_blackbox_20261002_113600'


def sample_with_faces(triangles, count, rng):
    tri=np.asarray(triangles,dtype=np.float64)
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=-1)
    ids=rng.choice(len(tri),count,p=area2/area2.sum())
    selected=tri[ids];uv=rng.rand(count,2);root=np.sqrt(uv[:,:1])
    pts=(1-root)*selected[:,0]+root*(1-uv[:,1:])*selected[:,1]+root*uv[:,1:]*selected[:,2]
    return pts,ids


def distance_with_nearest(points, triangles, chunk=32):
    points=np.asarray(points,dtype=np.float64);tri=np.asarray(triangles,dtype=np.float64)
    ds=[];ids=[]
    for start in range(0,len(points),chunk):
        q=points[start:start+chunk];tt=np.tile(tri,(len(q),1,1));qq=np.repeat(q,len(tri),axis=0)
        closest=trimesh.triangles.closest_point(tt,qq)
        d2=np.sum((closest-qq)**2,axis=-1).reshape(len(q),len(tri))
        for qi,fi in np.argwhere(~np.isfinite(d2)):
            t,p=tri[fi],q[qi];v=[]
            for a,b in ((0,1),(1,2),(2,0)):
                edge=t[b]-t[a];den=np.dot(edge,edge)
                u=np.clip(np.dot(p-t[a],edge)/den,0,1) if den>0 else 0.
                v.append(np.sum((p-(t[a]+u*edge))**2))
            d2[qi,fi]=min(v)
        near=np.argmin(d2,axis=1)
        ds.append(d2[np.arange(len(q)),near]);ids.append(near)
    return np.concatenate(ds),np.concatenate(ids)


def shapes(triangles, bbox):
    tri=np.asarray(triangles,dtype=np.float64).reshape(-1,100,3,3)
    e=np.stack([np.linalg.norm(tri[:,:,(i+1)%3]-tri[:,:,i],axis=-1) for i in range(3)],axis=-1)
    normal=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]);area2=np.linalg.norm(normal,axis=-1)
    angles=[]
    for i in range(3):
        u,v=tri[:,:,(i+1)%3]-tri[:,:,i],tri[:,:,(i+2)%3]-tri[:,:,i]
        den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
        cosine=np.divide((u*v).sum(-1),den,out=np.ones_like(den),where=den>0)
        angles.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    longest=e.max(-1);shortest=e.min(-1)
    unit=np.divide(normal,area2[...,None],out=np.zeros_like(normal),where=area2[...,None]>0)
    cosine=np.full_like(area2,np.nan)
    if len(tri)>1:
        cosine[1:]=(unit[1:]*unit[:-1]).sum(-1)
        cosine[1:][(area2[1:]==0)|(area2[:-1]==0)]=np.nan
    return dict(area=area2/2,area_bbox2=area2/2/bbox**2,
        minimum_angle_degrees=np.stack(angles,-1).min(-1),
        longest_edge=longest,longest_edge_bbox=longest/bbox,
        edge_length_ratio=np.divide(longest,shortest,out=np.full_like(longest,np.inf),where=shortest>0),
        altitude_aspect=np.divide(longest**2,area2,out=np.full_like(longest,np.inf),where=area2>0),
        degenerate=(area2<=bbox**2*1e-12),normal_adjacent_cosine=cosine,
        centroids=tri.mean(-2))


def attribution(raw,case):
    tri=np.asarray(raw).reshape(112,3,3)[12:].astype(np.float64)
    target=np.asarray(case['free']).reshape(100,3,3).astype(np.float64)
    rng=np.random.RandomState(9911);a,aid=sample_with_faces(tri,512,rng);b,bid=sample_with_faces(target,512,rng)
    checkrng=np.random.RandomState(9911)
    assert np.array_equal(a,sample_mesh_surface(tri,512,checkrng))
    assert np.array_equal(b,sample_mesh_surface(target,512,checkrng))
    assert np.array_equal(b,case['pairwise_target_points'])
    da,near_gt=distance_with_nearest(a,target);db,near_free=distance_with_nearest(b,tri)
    assert np.array_equal(da,surface_distance_squared(a,target))
    assert np.array_equal(db,surface_distance_squared(b,tri))
    forward=np.bincount(aid,weights=da,minlength=100)/512
    backward=np.bincount(near_free,weights=db,minlength=100)/512
    backward_gt=np.bincount(bid,weights=db,minlength=100)/512
    symmetric=(forward+backward)/2
    assert np.isclose(forward.sum(),da.mean(),rtol=1e-14,atol=1e-18)
    assert np.isclose(backward.sum(),db.mean(),rtol=1e-14,atol=1e-18)
    assert np.isclose(symmetric.sum(),(da.mean()+db.mean())/2,rtol=1e-14,atol=1e-18)
    share=symmetric/symmetric.sum() if symmetric.sum()>0 else symmetric
    order=np.argsort(-symmetric,kind='stable');L=case['bbox']
    return dict(prediction_points=a,prediction_face_ids=aid,target_points=b,target_face_ids=bid,
        prediction_squared_distances=da,target_squared_distances=db,
        nearest_target_face_ids=near_gt,nearest_free_face_ids=near_free,
        forward_mse_by_free_face=forward,backward_mse_by_nearest_free_face=backward,
        backward_mse_by_target_face=backward_gt,symmetric_mse_by_free_face=symmetric,
        symmetric_mse_share_by_free_face=share),dict(
        forward_RMS_bbox_pct=float(100*np.sqrt(da.mean())/L),
        backward_RMS_bbox_pct=float(100*np.sqrt(db.mean())/L),
        symmetric_RMS_bbox_pct=float(100*np.sqrt(symmetric.sum())/L),
        top1_symmetric_mse_share=float(share[order[:1]].sum()),
        top3_symmetric_mse_share=float(share[order[:3]].sum()),
        top5_symmetric_mse_share=float(share[order[:5]].sum()),
        top10_symmetric_mse_share=float(share[order[:10]].sum()),
        faces_for_50pct_mse=int(np.searchsorted(np.cumsum(share[order]),.5)+1),
        faces_for_90pct_mse=int(np.searchsorted(np.cumsum(share[order]),.9)+1),
        top10_free_face_ids=order[:10].tolist(),
        prediction_samples_per_face=np.bincount(aid,minlength=100).tolist(),
        interpretation='Additive MSE; backward nearest free-face attribution is posthoc proximity, not a GT slot correspondence. Unhit faces can have zero sampled forward contribution.')


def components(edges):
    adj=defaultdict(set)
    for a,b in edges:adj[int(a)].add(int(b));adj[int(b)].add(int(a))
    seen=set();out=[]
    for start in sorted(adj):
        if start in seen:continue
        stack=[start];seen.add(start);vs=[]
        while stack:
            x=stack.pop();vs.append(x)
            for y in adj[x]:
                if y not in seen:seen.add(y);stack.append(y)
        out.append((sorted(vs),adj))
    return out


def context_metrics(case):
    v=np.asarray(case['source_vertices'],dtype=np.float64)
    f=np.asarray(case['source_face_vertex_ids'])[case['source_face_ids']]
    incid=defaultdict(list)
    for fi,t in enumerate(f):
        for a,b in ((0,1),(1,2),(2,0)):incid[tuple(sorted((int(t[a]),int(t[b]))))].append(fi)
    edges=[e for e,ls in incid.items() if len(ls)==1]
    lengths=np.asarray([np.linalg.norm(v[a]-v[b]) for a,b in edges]);L=case['bbox']
    records=[]
    for vertices,adj in components(edges):
        es=[(a,b) for a,b in edges if a in vertices];length=sum(np.linalg.norm(v[a]-v[b]) for a,b in es)
        coords=v[vertices];diameter=float(np.linalg.norm(coords[:,None]-coords[None,:],axis=-1).max())
        ends=[x for x in vertices if len(adj[x])==1];branches=[x for x in vertices if len(adj[x])>2]
        closed=all(len(adj[x])==2 for x in vertices)
        chord=float(np.linalg.norm(v[ends[0]]-v[ends[1]])) if len(ends)==2 and not branches else None
        records.append(dict(vertex_ids=vertices,edge_count=len(es),length=float(length),length_bbox=float(length/L),
            closed_simple_cycle=closed,endpoint_count=len(ends),branch_vertex_ids=branches,
            diameter=diameter,open_path_length_over_chord=float(length/chord) if chord and chord>0 else None,
            closed_length_over_twice_diameter=float(length/(2*diameter)) if closed and diameter>0 else None))
    q=np.asarray(case['boundary_points'],dtype=np.float64);w=np.asarray(case['boundary_weights'],dtype=np.float64)
    mean=(q*w[:,None]).sum(0)/w.sum();cov=((q-mean)*w[:,None]).T@(q-mean)/w.sum()
    eigen,axes=np.linalg.eigh(cov);off=(q-mean)@axes[:,0]
    c=np.asarray(case['C'],dtype=np.float64);full=np.asarray(case['full'],dtype=np.float64).reshape(-1,3,3)
    ca=np.linalg.norm(np.cross(c[:,1]-c[:,0],c[:,2]-c[:,0]),axis=-1).sum()/2
    fa=np.linalg.norm(np.cross(full[:,1]-full[:,0],full[:,2]-full[:,0]),axis=-1).sum()/2
    cbox=np.ptp(c.reshape(-1,3),axis=0);fbox=np.ptp(full.reshape(-1,3),axis=0)
    ids=np.unique(f)
    from experiments.mf_h1_local0.t1_native_confirmation_topology import audit_mesh
    topology=audit_mesh(v,f)
    degree=Counter(x for edge in edges for x in edge)
    return dict(parent=case['parent'],patch=case['patch'],K=12,N=112,bbox_diagonal=L,
        boundary_edge_count=len(edges),boundary_total_length=float(lengths.sum()),boundary_length_bbox=float(lengths.sum()/L),
        boundary_components=records,boundary_component_count=len(records),
        boundary_vertex_degree_histogram=dict(Counter(str(n) for n in degree.values())),
        source_C_vertex_links=topology['vertex_links'],
        source_C_edge_incidence=topology['edge_incidence'],
        boundary_nonmanifold_degree_gt2_count=sum(len(x['branch_vertex_ids']) for x in records),
        C_edge_incidence_histogram=dict(Counter(str(len(x)) for x in incid.values())),
        C_nonmanifold_edge_count=sum(len(x)>2 for x in incid.values()),
        boundary_weighted_best_plane_rms_bbox=float(np.sqrt((w*off**2).sum()/w.sum())/L),
        boundary_weighted_best_plane_max_bbox=float(np.abs(off).max()/L),boundary_covariance_eigenvalues=eigen.tolist(),
        C_bbox_extent=cbox.tolist(),C_bbox_volume_over_full_bbox=float(np.prod(cbox)/np.prod(fbox)),
        C_area=ca,full_target_area=fa,C_area_fraction=float(ca/fa),
        unique_C_source_vertices=int(len(ids)),known_face_fraction=12/112,
        caveat='Source IDs used only for offline context topology. Bbox ratio is an extent ratio, not missing material/semantic information. Branches and nonmanifold edges are not repaired.')


def noise_metrics(cases):
    rows=[]
    with np.load(PRIOR/'registration/noises.npz',allow_pickle=False) as z:
        for seed in (10401,10402):
            xyz=z[f'seed_{seed}'].reshape(112,3,3)[12:].reshape(-1,3).astype(np.float64)
            mu=xyz.mean(0);cov=np.cov(xyz,rowvar=False,ddof=1);eig,vec=np.linalg.eigh(cov);order=np.argsort(eig)[::-1]
            eig=eig[order];vec=vec[:,order]
            for j in range(3):
                if vec[np.argmax(np.abs(vec[:,j])),j]<0:vec[:,j]*=-1
            distances=[]
            for patch in (0,4):
                case=cases[0,patch];d2=surface_distance_squared(xyz,case['C']);L=case['bbox']
                distances.append(dict(parent=0,patch=patch,noise_point_to_C_RMS_bbox_pct=float(100*np.sqrt(d2.mean())/L),
                    noise_centroid_to_C_vertex_centroid_bbox=float(np.linalg.norm(mu-case['C'].reshape(-1,3).mean(0))/L)))
            rows.append(dict(seed=seed,point_count=300,mean=mu.tolist(),sample_covariance=cov.tolist(),
                eigenvalues_descending=eig.tolist(),eigenvectors_columns_sign_canonical=vec.tolist(),
                eigenvalue_ratio=float(eig[0]/eig[-1]),coordinate_RMS=float(np.sqrt(np.mean(xyz**2))),
                maximum_point_norm=float(np.linalg.norm(xyz,axis=-1).max()),context_distances=distances,
                caveat='An isotropic Gaussian draw has no canonical chair orientation; eigenvector signs are arbitrary. Two seeds do not establish a noise cause.'))
    return rows


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def endpoint_stage(out):
    folder=out/'geometry';folder.mkdir(parents=True,exist_ok=True);cases=load_cases();tick=time.perf_counter()
    allrows=[];face_rows=[];arrays={};raws={}
    for model in ('B','J'):
        old=read_json(PRIOR/f'scores/A_{model}/results.json')['rows']
        for patch in (0,4):
            for seed in (10401,10402):
                for nfe in (50,100):
                    key=f'{model}_P0_patch{patch}_seed{seed}_bf16_Euler{nfe}'
                    path=PRIOR/f'A/{model}/{key}.npz'
                    with np.load(path,allow_pickle=False) as z:raw=z['raw'].copy()
                    values,summary=attribution(raw,cases[0,patch]);raws[key]=raw.reshape(112,3,3)
                    for name,a in values.items():arrays[key+'__'+name]=a
                    reference=next(x for x in old if x['parent']==0 and x['patch']==patch and x['seed']==seed and x['nfe']==nfe and x['precision']=='bf16')['scalars']
                    checks={}
                    for name,other in (('symmetric_RMS_bbox_pct','remainder_symmetric_RMS_bbox_pct'),('forward_RMS_bbox_pct','free_to_R12_RMS_bbox_pct'),('backward_RMS_bbox_pct','R12_to_free_RMS_bbox_pct')):
                        checks[name]=float(summary[name]-reference[other]);assert np.isclose(summary[name],reference[other],rtol=1e-12,atol=1e-12),(key,name,summary[name],reference[other])
                    shape=shapes(raw.reshape(112,3,3)[12:],cases[0,patch]['bbox'])
                    target_shape=shapes(cases[0,patch]['free'],cases[0,patch]['bbox'])
                    summary.update(key=key,model=model,parent=0,patch=patch,seed=seed,nfe=nfe,
                        archived_raw_sha256=file_sha256(path),original_score_delta=checks,
                        free_total_area=float(shape['area'][0].sum()),GT_free_total_area=float(target_shape['area'][0].sum()),
                        free_area_over_GT_free_area=float(shape['area'][0].sum()/target_shape['area'][0].sum()),max_longest_edge_bbox=float(shape['longest_edge_bbox'][0].max()),
                        minimum_angle_degrees=float(shape['minimum_angle_degrees'][0].min()),
                        min_angle_lt5_count=int((shape['minimum_angle_degrees'][0]<5).sum()),
                        area_degenerate_count=int(shape['degenerate'][0].sum()))
                    allrows.append(summary)
                    for i in range(100):
                        face_rows.append(dict(key=key,free_face=i,total_slot=i+12,
                            forward_mse=float(values['forward_mse_by_free_face'][i]),
                            backward_mse_assigned_to_nearest_face=float(values['backward_mse_by_nearest_free_face'][i]),
                            symmetric_mse_share=float(values['symmetric_mse_share_by_free_face'][i]),
                            area=float(shape['area'][0,i]),minimum_angle_degrees=float(shape['minimum_angle_degrees'][0,i]),
                            longest_edge_bbox=float(shape['longest_edge_bbox'][0,i]),
                            centroid_x=float(shape['centroids'][0,i,0]),centroid_y=float(shape['centroids'][0,i,1]),centroid_z=float(shape['centroids'][0,i,2])))
    np.savez_compressed(folder/'endpoint_face_attribution.npz',**arrays)
    write_csv(folder/'endpoint_faces.csv',face_rows)
    contexts=[context_metrics(cases[p,j]) for p in (0,1) for j in range(5)]
    # Exact boundary-degree histogram, independent of component decomposition.
    for row in contexts:
        case=cases[row['parent'],row['patch']];f=case['source_face_vertex_ids'][case['source_face_ids']]
        counts=Counter(tuple(sorted((int(t[a]),int(t[b])))) for t in f for a,b in ((0,1),(1,2),(2,0)))
        degree=Counter(x for e,n in counts.items() if n==1 for x in e)
        row['boundary_vertex_degree_histogram']=dict(Counter(str(n) for n in degree.values()))
    write_json(folder/'endpoint_results.json',dict(status='PASS',rows=allrows,contexts=contexts,noise=noise_metrics(cases),
        tests=run_tests(),seconds=time.perf_counter()-tick,model_forwards=0,optimizer_updates=0,
        attribution='512 area-uniform samples per direction, original RandomState(9911), exact original scalar reconciliation. Per-face sums are MSE, not RMS.'))
    plot_faces(folder,allrows,arrays,raws,cases)
    print('endpoint PASS',len(allrows),flush=True)


def plot_faces(folder,rows,arrays,raws,cases):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for patch in (0,4):
        keys=[f'{m}_P0_patch{patch}_seed{s}_bf16_Euler{n}' for s in (10401,10402) for m,n in (('B',50),('J',50),('B',100),('J',100))]
        points=np.concatenate([raws[k].reshape(-1,3) for k in keys]+[cases[0,patch]['full'].reshape(-1,3)])
        lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;radius=(hi-lo).max()*.54
        vmax=max(float(arrays[k+'__symmetric_mse_share_by_free_face'].max()*100) for k in keys)
        norm=PowerNorm(.35,vmin=0,vmax=vmax);fig=plt.figure(figsize=(16,9))
        for i,key in enumerate(keys):
            ax=fig.add_subplot(2,4,i+1,projection='3d');tri=raws[key];contrib=arrays[key+'__symmetric_mse_share_by_free_face']*100
            poly=Poly3DCollection(tri[12:],facecolors=plt.cm.viridis(norm(contrib)),edgecolors=(.15,.15,.15,.4),linewidths=.22)
            ax.add_collection3d(poly);ax.add_collection3d(Poly3DCollection(tri[:12],facecolors=(.25,.7,.9,.35),edgecolors='cyan',linewidths=.5))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-55,vertical_axis='y');ax.set_axis_off()
            row=next(r for r in rows if r['key']==key)
            ax.set_title(f"{row['model']} {row['nfe']} steps / seed {row['seed']}\nRMS {row['symmetric_RMS_bbox_pct']:.3f}%; top3 {100*row['top3_symmetric_mse_share']:.1f}% MSE",fontsize=10)
        fig.suptitle(f'P0 patch {patch}: additive symmetric MSE share per free face; C in cyan',fontsize=13)
        fig.subplots_adjust(right=.92,top=.91,bottom=.03,wspace=0,hspace=.08)
        bar=fig.add_axes([.94,.20,.015,.6]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=bar,label='Share of total symmetric MSE (%)')
        fig.savefig(folder/f'face_mse_heatmap_P0_patch{patch}.png',dpi=160);plt.close(fig)



def directional_plots(out):
    """Reuse exact saved score queries; display coverage on GT, not generated faces."""
    folder=out/'geometry';cases=load_cases();results=read_json(folder/'endpoint_results.json');rows=results['rows']
    with np.load(folder/'endpoint_face_attribution.npz',allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    raws={}
    for row in rows:
        path=PRIOR/f"A/{row['model']}/{row['key']}.npz"
        assert file_sha256(path)==row['archived_raw_sha256']
        with np.load(path,allow_pickle=False) as z:raws[row['key']]=z['raw'].reshape(112,3,3).copy()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from experiments.mf_h1_local0.jedge5_diagnosis.common import ASSETS,EVALUATION_SHA,array_hash
    gt=cases[0,0]['full'].reshape(-1,3)
    pts=np.concatenate([x.reshape(-1,3) for x in raws.values()]+[gt])
    lo=pts.min(0);hi=pts.max(0);center=(lo+hi)/2;radius=(hi-lo).max()*.54
    forward_max=max(float(arrays[r['key']+'__forward_mse_by_free_face'].max()/cases[0,r['patch']]['bbox']**2*1e4) for r in rows)
    backward_max=max(float(arrays[r['key']+'__target_squared_distances'].max()/cases[0,r['patch']]['bbox']**2*1e4) for r in rows)
    forward_norm=PowerNorm(.35,vmin=0,vmax=forward_max);backward_norm=PowerNorm(.35,vmin=0,vmax=backward_max)
    manifest=[]
    for patch in (0,4):
        case=cases[0,patch];L=case['bbox'];target=case['free'].reshape(100,3,3)
        for seed in (10401,10402):
            fig=plt.figure(figsize=(16,9))
            selected=[next(r for r in rows if r['patch']==patch and r['seed']==seed and r['model']==m and r['nfe']==n) for m,n in (('B',50),('J',50),('B',100),('J',100))]
            for col,row in enumerate(selected):
                key=row['key'];tri=raws[key];forward=arrays[key+'__forward_mse_by_free_face']/L**2*1e4
                q=arrays[key+'__target_points'];d2=arrays[key+'__target_squared_distances']/L**2*1e4
                assert np.array_equal(q,case['pairwise_target_points'])
                for direction in (0,1):
                    ax=fig.add_subplot(2,4,direction*4+col+1,projection='3d')
                    if direction==0:
                        ax.add_collection3d(Poly3DCollection(tri[12:],facecolors=plt.cm.viridis(forward_norm(forward)),edgecolors=(.12,.12,.12,.3),linewidths=.22))
                        subtitle=f"free to GT RMS {row['forward_RMS_bbox_pct']:.3f}%"
                    else:
                        ax.add_collection3d(Poly3DCollection(target,facecolors=(.75,.75,.75,.05),edgecolors=(.3,.3,.3,.17),linewidths=.35))
                        ax.scatter(q[:,0],q[:,1],q[:,2],c=d2,cmap='viridis',norm=backward_norm,s=11,depthshade=False,edgecolors='none')
                        subtitle=f"GT to free RMS {row['backward_RMS_bbox_pct']:.3f}%"
                    ax.add_collection3d(Poly3DCollection(case['C'],facecolors=(.25,.7,.9,.12),edgecolors='cyan',linewidths=.4))
                    ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
                    ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-55,vertical_axis='y');ax.set_axis_off()
                    ax.set_title(f"{row['model']} / BF16 / Euler {row['nfe']}\n{subtitle}",fontsize=11)
                manifest.append(dict(key=key,parent=0,patch=patch,seed=seed,precision='bf16',nfe=row['nfe'],model=row['model'],
                    model_state_sha256=ASSETS[row['model']]['state'],raw_sha256=row['archived_raw_sha256'],
                    context_array_sha256=array_hash(case['C']),GT_free_array_sha256=array_hash(case['free']),
                    target_points_array_sha256=array_hash(q),target_distances_squared_array_sha256=array_hash(arrays[key+'__target_squared_distances']),
                    original_attribution_npz_sha256=file_sha256(folder/'endpoint_face_attribution.npz')))
            fig.suptitle(f'P0 patch {patch} / seed {seed}: predicted-face error (top), missing GT-region error (bottom)',fontsize=13)
            fig.subplots_adjust(left=.01,right=.91,top=.91,bottom=.035,wspace=0,hspace=.08)
            bar1=fig.add_axes([.935,.57,.013,.29]);bar2=fig.add_axes([.935,.12,.013,.29])
            fig.colorbar(plt.cm.ScalarMappable(norm=forward_norm,cmap='viridis'),cax=bar1,label='Forward face MSE contribution: sum(d^2) / 512 [(% bbox)^2]')
            fig.colorbar(plt.cm.ScalarMappable(norm=backward_norm,cmap='viridis'),cax=bar2,label='GT coverage sample distance^2 [(% bbox)^2]')
            fig.savefig(folder/f'directional_error_P0_patch{patch}_seed{seed}.png',dpi=175);plt.close(fig)
    write_json(folder/'directional_plot_manifest.json',dict(status='PASS',rows=manifest,evaluation_inputs_sha256=EVALUATION_SHA,
        prediction_display='Each generated face: sum of prediction-query squared distances on that face divided by all 512 prediction queries; normalized by L^2 and multiplied by 100^2. Face contributions sum to forward MSE.',
        GT_display='Original 512 area-uniform GT R12 query positions colored by their squared distance to the generated free surface. Missing GT regions remain displayed even when no generated face is nearby. Each point has score weight 1/512, but its plotted color is the unweighted pointwise distance squared.',
        distinction='Two directional colorbars have different aggregation, though both display squared percent-bbox units. Neither is symmetric RMS or a GT face-to-slot correspondence.',
        camera=dict(elevation=20,azimuth=-55,vertical_axis='y',center=center.tolist(),radius=float(radius)),
        color=dict(transform='PowerNorm gamma=0.35; same fixed scale across all four figures within each direction',forward_max=forward_max,backward_max=backward_max),
        no_new_queries=True,model_forwards=0,geometry_distance_calls=0,optimizer_updates=0))
    print('directional plots PASS',len(manifest),flush=True)

def trajectory_stage(out):
    folder=out/'geometry';cases=load_cases();summaries=[];csvrows=[];tick=time.perf_counter()
    saved={}
    for model in ('B','J'):
        for patch in (0,4):
            for seed in (10401,10402):
                for nfe in (50,100):
                    key=f'{model}_P0_patch{patch}_seed{seed}_bf16_Euler{nfe}'
                    path=out/f'trajectories/{model}/{key}.npz'
                    with np.load(path,allow_pickle=False) as z:states=z['states'].copy().reshape(nfe+1,112,3,3);raw=z['raw'].copy().reshape(112,3,3)
                    assert np.array_equal(states[-1],raw)
                    assert all(x[:12].tobytes()==cases[0,patch]['C'].tobytes() for x in states)
                    shape=shapes(states[:,12:],cases[0,patch]['bbox']);shape['t']=np.arange(nfe+1)/nfe
                    np.savez_compressed(folder/(key+'_face_shapes.npz'),**shape)
                    saved[key]=shape
                    for i,t in enumerate(shape['t']):
                        row=dict(key=key,step=i,t=float(t),total_area=float(shape['area'][i].sum()),
                            minimum_angle=float(shape['minimum_angle_degrees'][i].min()),median_minimum_angle=float(np.median(shape['minimum_angle_degrees'][i])),
                            angle_lt5_count=int((shape['minimum_angle_degrees'][i]<5).sum()),
                            area_degenerate_count=int(shape['degenerate'][i].sum()),
                            max_edge_bbox=float(shape['longest_edge_bbox'][i].max()),median_area_bbox2=float(np.median(shape['area_bbox2'][i])),
                            max_altitude_aspect=float(shape['altitude_aspect'][i].max()),normal_cosine_negative_count=int((shape['normal_adjacent_cosine'][i]<0).sum()))
                        csvrows.append(row)
                    summaries.append(dict(key=key,states=nfe+1,source_sha256=file_sha256(path),
                        min_angle_all_states=float(shape['minimum_angle_degrees'].min()),
                        degenerate_face_state_count=int(shape['degenerate'].sum()),
                        degenerate_face_count=int(shape['degenerate'].any(0).sum()),
                        normal_cosine_negative_events=int((shape['normal_adjacent_cosine']<0).sum()),
                        maximum_altitude_aspect=float(shape['altitude_aspect'].max()),
                        endpoint_total_area=float(shape['area'][-1].sum()),
                        caveat='Adjacent-step normal reversal is a coordinate trajectory diagnostic, not proof of topological winding errors. Degenerate threshold uses fixed target bbox.'))
    write_csv(folder/'trajectory_shapes.csv',csvrows)
    write_json(folder/'trajectory_results.json',dict(status='PASS',rows=summaries,seconds=time.perf_counter()-tick,model_forwards=0,
        all_per_face_raw='*_face_shapes.npz',threshold='cross-product norm <= 1e-12 * fixed bbox^2',
        aspect='longest_edge / altitude_to_longest_edge = longest_edge^2 / twice_area'))
    plot_trajectories(folder,saved)
    print('trajectory PASS',len(summaries),flush=True)


def plot_trajectories(folder,saved):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for patch in (0,4):
        fig,axs=plt.subplots(3,2,figsize=(12,10),sharex=True)
        for col,seed in enumerate((10401,10402)):
            for model,nfe,style in (('B',50,'--'),('B',100,'-'),('J',50,'--'),('J',100,'-')):
                key=f'{model}_P0_patch{patch}_seed{seed}_bf16_Euler{nfe}';x=saved[key];color='tab:blue' if model=='B' else 'tab:orange'
                label=f'{model} {nfe}'
                axs[0,col].plot(x['t'],x['area'].sum(1),style,color=color,label=label)
                axs[1,col].plot(x['t'],(x['minimum_angle_degrees']<5).sum(1),style,color=color,label=label)
                axs[2,col].plot(x['t'],x['longest_edge_bbox'].max(1),style,color=color,label=label)
            axs[0,col].set_title(f'P0 patch {patch} / seed {seed}');axs[2,col].set_xlabel('time t');axs[0,col].legend()
        for row,label in enumerate(('Total free area (model units^2)','Free faces with minimum angle < 5 deg','Maximum edge / full GT bbox')):
            for col in range(2):axs[row,col].set_ylabel(label);axs[row,col].grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/f'shape_trajectories_P0_patch{patch}.png',dpi=160);plt.close(fig)


def run_tests():
    tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]],[[0,0,1],[2,0,1],[0,2,1]]])
    a,ids=sample_with_faces(tri,512,np.random.RandomState(9911));b=sample_mesh_surface(tri,512,np.random.RandomState(9911))
    assert np.array_equal(a,b)
    d,n=distance_with_nearest(np.array([[.2,.2,0],[.2,.2,1],[.2,.2,2]]),tri)
    assert np.allclose(d,[0,0,1]) and np.array_equal(n,[0,1,1])
    sh=shapes(np.tile(tri[:1],(100,1,1)),2.)
    assert np.allclose(sh['area'],.5) and np.allclose(sh['minimum_angle_degrees'],45)
    assert np.allclose(sh['altitude_aspect'],2.) and not sh['degenerate'].any()
    gs=components([(0,1),(1,2),(2,0),(3,4)])
    assert len(gs)==2
    return dict(status='PASS',checks=8,description='Sampler identity, nearest triangle distances/IDs, right-triangle area/angle/aspect, degeneracy, graph components; each endpoint independently reconciles all three original RMS scalars.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['endpoints','trajectories','plots','tests'],required=True);a=p.parse_args()
    if a.stage=='endpoints':endpoint_stage(a.out)
    elif a.stage=='trajectories':trajectory_stage(a.out)
    elif a.stage=='plots':directional_plots(a.out)
    else:print(run_tests())

if __name__=='__main__':main()
