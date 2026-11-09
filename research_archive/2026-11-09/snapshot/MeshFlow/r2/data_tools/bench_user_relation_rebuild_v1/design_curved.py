"""Read-only CPU region design on two existing accepted target meshes."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json, csv, hashlib, time, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import psutil

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_user_relation_rebuild_v1/design_curved'
IDS=['9e429b7f77646b4ccc8f8e7f17e9d7b6','943dde2754ddc7822e8ff3556a90169']
BASE=ROOT/'experiments/r2_bench_user_data_v2_expansion_v1/standard_parents'
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(pid):
    p=BASE/pid/'target_mesh.npz'
    d=np.load(p);v=d['vertices'].astype(float);f=d['faces'];t=v[f]
    n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=np.linalg.norm(n,axis=1)/2;n/=np.maximum(2*area[:,None],1e-30)
    return v,f,t,n,area
def draw(ax,t,col,vertices):
    # Display X,Z,Y so vertical remains original Y.
    tt=t[:,:,[0,2,1]]
    ax.add_collection3d(Poly3DCollection(tt,facecolors=col,edgecolors='#555555',linewidths=.25,alpha=.96))
    lo=vertices.min(0)[[0,2,1]];hi=vertices.max(0)[[0,2,1]];mid=(lo+hi)/2;r=(hi-lo).max()/2
    ax.set_xlim(mid[0]-r,mid[0]+r);ax.set_ylim(mid[1]-r,mid[1]+r);ax.set_zlim(mid[2]-r,mid[2]+r)
    ax.set_box_aspect((1,1,1));ax.set_axis_off()
def inspect():
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'resources.json',{'time':time.time(),'cpu_count':psutil.cpu_count(),'ram_available':psutil.virtual_memory().available,'cpu_percent':psutil.cpu_percent(interval=.1),'threads':1,'gpu_model_calls':0})
    for pid in IDS:
        v,f,t,n,a=load(pid)
        with (OUT/(pid[:6]+'_faces.csv')).open('w') as h:
            w=csv.writer(h);w.writerow(['face','v0','v1','v2','cx','cy','cz','nx','ny','nz','area']);w.writerows([[i,*f[i],*t[i].mean(0),*n[i],a[i]]for i in range(len(f))])
        dump(OUT/(pid[:6]+'_vertices.json'),{str(i):x.tolist()for i,x in enumerate(v)})
        fig=plt.figure(figsize=(16,12))
        for i,(az,el) in enumerate([(-60,25),(120,25),(0,0),(90,0)]):
            ax=fig.add_subplot(2,3,i+1,projection='3d');draw(ax,t,np.clip((n+1)/2,.05,.95),v);ax.view_init(el,az);ax.set_title(f'{pid[:6]} normals RGB / az {az}')
        ax=fig.add_subplot(2,3,5)
        for i,tri in enumerate(t):
            ax.plot(tri[[0,1,2,0],2],tri[[0,1,2,0],1],color='gray',lw=.4)
        for i,p in enumerate(v):ax.text(p[2],p[1],str(i),fontsize=5,color='navy')
        ax.set_aspect('equal');ax.set_xlabel('Z');ax.set_ylabel('Y');ax.set_title('Side projection: vertex IDs')
        ax=fig.add_subplot(2,3,6)
        for i,tri in enumerate(t):
            c=tri.mean(0);ax.scatter(c[2],c[1],s=8,color=np.clip((n[i]+1)/2,.05,.95));ax.text(c[2],c[1],str(i),fontsize=5)
        ax.set_aspect('equal');ax.set_xlabel('Z');ax.set_ylabel('Y');ax.set_title('Face centers / face IDs')
        fig.tight_layout();fig.savefig(OUT/(pid[:6]+'_inspect.png'),dpi=160);plt.close(fig)
        print(pid, 'bbox',v.min(0),v.max(0),'normal x histogram',np.histogram(n[:,0],bins=[-1.1,-.9,-.1,.1,.9,1.1])[0])
def region_stats(f,v,ids):
    from collections import Counter,defaultdict
    fs=f[ids];edges=Counter(tuple(sorted((int(a),int(b))))for tri in fs for a,b in zip(tri,np.roll(tri,-1)))
    adj=defaultdict(set)
    for i,tri in zip(ids,fs):
        for a,b in zip(tri,np.roll(tri,-1)):adj[tuple(sorted((int(a),int(b))))].add(int(i))
    neighbors={int(i):set()for i in ids}
    for faces in adj.values():
        for i in faces:neighbors[i].update(faces-{i})
    remaining=set(neighbors);comps=[]
    while remaining:
        seed=min(remaining);stack=[seed];comp=set()
        while stack:
            q=stack.pop()
            if q in comp:continue
            comp.add(q);stack.extend(neighbors[q]-comp)
        remaining-=comp;comps.append(sorted(comp))
    return {'edge_components':len(comps),'component_face_counts':list(map(len,comps)),'boundary_edges':sum(x==1 for x in edges.values()),'nonmanifold_edges':sum(x>2 for x in edges.values()),'unique_vertices':len(np.unique(fs)), 'C_bbox_min':v[np.unique(fs)].min(0).tolist(),'C_bbox_max':v[np.unique(fs)].max(0).tolist()}
def make():
    rows=[]
    for pid in IDS:
        v,f,t,n,a=load(pid)
        if pid==IDS[0]:
            seat=np.flatnonzero((n[:,1]>.6)&(abs(n[:,2])<.05)&(t[:,:,0].min(1)>-2.003)&(t[:,:,0].max(1)<2.002)).tolist()
            # Manually reviewed whole extruded outer wall chain, from bottom foot,
            # around the entire outer arm crown, down the inward riser to seat join.
            wall=[1,3,12,15,36,38,166,168,195,237,240,246,252,253,249,250,241,236,174,176]
            definitions=[('CURVED_THREE_BAY_SEAT_SURFACE',seat,'SEAT_SURFACE','Complete upper surface of all three seating bays, including both natural raised dividers; excludes underside and end-arm risers.','Upper extruded shell selected between existing longitudinal edge pairs (75,77) and (76,78); no added cut vertices. Normal selection identifies the surface, semantic designation verified in views.','Open surface, not a closed seat volume; end risers are explicitly omitted.'),('COMPLETE_LEFT_END_OUTER_SURFACE',wall,'LEG_SUPPORT','Complete curved outer end-support surface from bottom foot over arm crown and down to the seat junction; full depth retained.','Manual face-chain checked against full outer contour and face adjacency; every longitudinal strip retained. No height cutoff.','Open curved wall surface, not the whole solid end block; end caps, inner hollow underside and bottom foot sole are not included.'),('CURVED_SEAT_PLUS_LEFT_END',sorted(set(seat+wall)),'SEAT_SUPPORT','Complete three-bay seat surface plus complete left curved end-support surface.','Union of the two independently inspected regions joined at the original seat-to-riser edge.','One open surface assembly. Remaining right end, undersides and side caps belong to Y; semantic labels are curation only.')]
        else:
            # The middle extruded shell lies between the full length rail endpoints;
            # end supports have |x|>2.45 and must not contaminate the seat/back.
            # QEM retained a 0.000278 x-normal in face203; tolerate it rather
            # than introduce an artificial hole in the verified central shell.
            central=(np.abs(t[:,:,0]).max(1)<2.46)&(abs(n[:,0])<.01)
            seat=np.flatnonzero(central&(n[:,1]>.85)&(t[:,:,1].max(1)<-.15)).tolist()
            back=np.flatnonzero(central&(n[:,2]>.8)&(n[:,1]>.01)&(t[:,:,1].max(1)>-.15)).tolist()
            definitions=[('TRUE_CURVED_SEAT_UPPER_SURFACE',seat,'SEAT_SURFACE','Complete bowed seat upper surface across its full width and depth; no seat-bottom faces.','Exact middle-shell faces with upward normal, bounded by existing front/rear seat rim edges; both end-support surfaces excluded.','Single open seating surface; side caps and underside remain Y.'),('COMPLETE_INNER_CURVED_BACK',back,'BACK_SURFACE','Complete inner curved back surface including the lower transition down to the seat junction.','Exact central front-facing back shell from the bottom joint to the top rim; excludes rear planar back and top cap.','This is the inner surface only, not a closed back slab. Source interface may include acute seam angles; no forced universal normal constraint.'),('CURVED_INNER_SEAT_BACK',sorted(set(seat+back)),'SEAT_BACK','Continuous actual upper seat and inner curved back, joined along their original crease.','Union of complete seat and complete inner back regions, retaining original shared vertex IDs.','Open seating shell with outer back, underside and end supports in Y; no unique-source reconstruction requirement.')]
        for name,ids,part,extent,method,limits in definitions:
            ids=sorted(ids);row={'parent_id':pid,'region_name':name,'part_type':part,'target_mesh_path':str(BASE/pid/'target_mesh.npz'),'target_case_path':str(BASE/pid/'target_case.npz'),'target_mesh_sha256':sha(BASE/pid/'target_mesh.npz'),'target_case_sha256':sha(BASE/pid/'target_case.npz'),'target_face_ids':ids,'source_face_id_semantics':'row indices of frozen target_mesh/target_case source_faces AFTER historical QEM; not original pre-QEM triangle IDs','K':len(ids),'N':len(f),'extent_description':extent,'construction_method':method,'limits':limits,'stats':region_stats(f,v,ids),'review_status':'DESIGN_PENDING_PARENT_FINAL_REVIEW','prior_geometry_receipt':str(BASE/pid/'geometry.json'),'prior_geometry_receipt_sha256':sha(BASE/pid/'geometry.json'),'model_input_policy':'C and N only; no source target/part/owner labels provided to inference'}
            fig=plt.figure(figsize=(16,9));mask=np.ones(len(f),bool);mask[ids]=False
            for j,(what,az,el) in enumerate([('M: blue C / gray Y',-60,25),('C only',-60,25),('Y only',-60,25),('M: opposite view',120,25),('C opposite view',120,25),('C orthographic profile',90 if pid==IDS[0] else 0,0)]):
                ax=fig.add_subplot(2,3,j+1,projection='3d')
                if j in(0,3):
                    colors=np.array(['#b8bec4']*len(f));colors[ids]='#008bc2';draw(ax,t,colors,v)
                elif j==2:draw(ax,t[mask],['#b8bec4']*mask.sum(),v)
                else:draw(ax,t[ids],['#008bc2']*len(ids),v)
                ax.view_init(el,az);ax.set_title(what,fontsize=10)
            fig.suptitle(f'{pid[:6]} {name}\nK={len(ids)}, N={len(f)} | {extent}',fontsize=13)
            fig.tight_layout();img=OUT/(pid[:6]+'_'+name+'.png');fig.savefig(img,dpi=140);plt.close(fig)
            row['image_path']=str(img);row['image_sha256']=sha(img)
            rows.append(row);print(pid[:6],name,len(ids),row['stats'])
    dump(OUT/'candidate_regions.json',rows)
    dump(OUT/'source_hashes.json',{str(p):sha(p)for pid in IDS for p in [BASE/pid/'target_mesh.npz',BASE/pid/'target_case.npz',BASE/pid/'geometry.json',BASE/pid/'lineage.json']})
if __name__=='__main__':
    make()if '--make' in sys.argv else inspect()
