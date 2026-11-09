"""CPU-only source-input audit; no model, repair, or old quality scoring."""
from __future__ import annotations
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_n,'2')
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_regular_patch_control_v1/preparation'
DATA=ROOT/'experiments/r2_user_data_scaling_12h_v1/data'
OLD=ROOT/'experiments/r2_seat_noise_transfer_v1/inputs/manifest.json'
PREFIXES=('73b369','3f7417','85b169','2d08a6','610b16','608b46')
VERSION='regular_source_patch_audit_v1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def components(nodes,links):
    adj={int(i):set() for i in nodes}
    for a,b in links: adj[int(a)].add(int(b));adj[int(b)].add(int(a))
    unseen=set(adj); groups=[]
    while unseen:
        start=min(unseen);seen={start};todo=[start];unseen.remove(start)
        while todo:
            for v in adj[todo.pop()]:
                if v in unseen: unseen.remove(v);seen.add(v);todo.append(v)
        groups.append(sorted(seen))
    return groups,adj

def topology(ids):
    edge=collections.defaultdict(list)
    for fi,face in enumerate(ids):
        for a,b in zip(face,np.roll(face,-1)):
            edge[tuple(sorted((int(a),int(b))))].append((fi,int(a),int(b)))
    links=[]
    for rows in edge.values():
        for i in range(1,len(rows)): links.append((rows[0][0],rows[i][0]))
    groups,_=components(range(len(ids)),links)
    boundary=[list(e) for e,r in edge.items() if len(r)==1]
    verts=sorted(set(np.asarray(boundary).ravel().tolist())) if boundary else []
    bg,adj=components(verts,boundary)
    degree2=all(len(v)==2 for v in adj.values())
    loops=[]
    if degree2:
        for group in bg:
            start=group[0];loop=[start];prev=None;cur=start
            while True:
                nxt=min(v for v in adj[cur] if v!=prev)
                if nxt==start: break
                loop.append(nxt);prev,cur=cur,nxt
                if len(loop)>len(group): raise RuntimeError('Bad boundary cycle')
            assert set(loop)==set(group)
            loops.append(loop)
    winding=sum(len(r)==2 and (r[0][1],r[0][2])!=(r[1][2],r[1][1]) for r in edge.values())
    return dict(source_edge_connected_components=groups,component_count=len(groups),
                boundary_edges=boundary,boundary_vertices=verts,boundary_degree_all_two=degree2,
                boundary_components=bg,boundary_loop_source_vertex_ids=loops,
                boundary_loop_count=len(loops) if degree2 else None,
                nonmanifold_edges=[list(e) for e,r in edge.items() if len(r)>2],
                source_same_direction_internal_edges=winding,
                Euler_characteristic=int(len(np.unique(ids))-len(edge)+len(ids)))

def geometry(tri,ids,top):
    tri=np.asarray(tri,np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); area=np.linalg.norm(cross,axis=1)/2
    assert np.all(area>0)
    center=np.sum(area[:,None]*tri.mean(axis=1),axis=0)/area.sum()
    vv=tri-center;s=vv.sum(axis=1)
    moment=(np.einsum('fvi,fvj->fij',vv,vv)+np.einsum('fi,fj->fij',s,s))/12
    cov=np.sum(moment*area[:,None,None],axis=0)/area.sum();eig,basis=np.linalg.eigh(cov)
    normal=basis[:,0]; mean=cross.sum(axis=0)
    if normal@mean<0: normal=-normal
    e1=basis[:,2]; e2=np.cross(normal,e1)
    uv=np.stack(((tri-center)@e1,(tri-center)@e2),axis=-1)
    pt={int(i):v for face,t in zip(ids,tri) for i,v in zip(face,t)}
    coords=np.array([pt[i] for i in sorted(pt)]);proj=np.stack(((coords-center)@e1,(coords-center)@e2),axis=-1)
    hull=ConvexHull(proj);hp=proj[hull.vertices]; he=np.roll(hp,-1,axis=0)-hp
    best=None
    for e in he:
        u=e/np.linalg.norm(e);v=np.array([-u[1],u[0]]);r=np.stack((hp@u,hp@v),axis=-1);extent=np.ptp(r,axis=0);rect=float(np.prod(extent))
        if best is None or rect<best[0]:best=(rect,extent)
    projected_area=float(np.sum(np.abs(cross@normal))/2)
    perimeter=sum(float(np.linalg.norm(pt[a]-pt[b])) for a,b in top['boundary_edges'])
    diag=float(np.linalg.norm(np.ptp(coords,axis=0)))
    angle=np.degrees(np.arccos(np.clip(cross@normal/(2*area),-1,1)))
    return dict(area=float(area.sum()),patch_bbox_diagonal=diag,area_weighted_center=center.tolist(),
        best_fit_plane_normal=normal.tolist(),plane_surface_RMS=float(np.sqrt(max(0,eig[0]))),
        plane_surface_RMS_over_patch_diagonal=float(np.sqrt(max(0,eig[0]))/diag),
        plane_max_vertex_distance=float(np.abs((coords-center)@normal).max()),
        plane_max_vertex_distance_over_patch_diagonal=float(np.abs((coords-center)@normal).max())/diag,
        source_face_normal_deviation_max_degrees=float(angle.max()),
        source_face_normal_deviation_area_mean_degrees=float(np.sum(angle*area)/area.sum()),
        projected_area=projected_area,projected_convex_hull_area=float(hull.volume),
        projected_area_over_convex_hull_area=projected_area/float(hull.volume),
        minimum_rectangle_area=best[0],projected_area_over_minimum_rectangle_area=projected_area/best[0],
        minimum_rectangle_short_over_long=float(min(best[1])/max(best[1])),boundary_perimeter=perimeter,
        circular_compactness_4piA_over_P2=float(4*np.pi*area.sum()/perimeter**2))

def draw(path,C,whole,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,9)); axs=[fig.add_subplot(2,2,1,projection='3d'),fig.add_subplot(2,2,2,projection='3d'),fig.add_subplot(2,2,3),fig.add_subplot(2,2,4)]
    def three(ax,full,az,el):
        if full is not None: ax.add_collection3d(Poly3DCollection(full[:,:, [0,2,1]],facecolor='#c7ccd0',edgecolor='#60666a',alpha=.26,linewidth=.2))
        ax.add_collection3d(Poly3DCollection(C[:,:,[0,2,1]],facecolor='#1686c0',edgecolor='#00364f',linewidth=.6))
        t=(whole if full is not None else C).reshape(-1,3)[:,[0,2,1]];lo=t.min(0);hi=t.max(0);mid=(lo+hi)/2;r=max(hi-lo)*.55
        ax.set(xlim=(mid[0]-r,mid[0]+r),ylim=(mid[1]-r,mid[1]+r),zlim=(mid[2]-r,mid[2]+r));ax.set_box_aspect((1,1,1));ax.view_init(el,az)
        ax.set_xlabel('X');ax.set_ylabel('Z');ax.set_zlabel('Y')
    three(axs[0],whole,-55,25);three(axs[1],None,35,25)
    axs[0].set_title('Source chair + FULL C (context only)');axs[1].set_title('Actual C, source corner order')
    from matplotlib.collections import PolyCollection
    for ax,dims,title in ((axs[2],(0,2),'C footprint X/Z'),(axs[3],(2,1),'C side profile Z/Y')):
        ax.add_collection(PolyCollection(C[:,:,dims],facecolors='#1686c0',edgecolors='#00364f',linewidth=.7,alpha=.6));ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.grid(alpha=.2)
    fig.suptitle(stats['condition_id']+'\nK='+str(stats['K'])+' / N='+str(stats['N'])+'; source-ID components='+str(stats['topology']['component_count'])+'; boundary loops='+str(stats['topology']['boundary_loop_count'])+'; no edits / no retessellation')
    fig.tight_layout(rect=(0,.04,1,.94));fig.text(.03,.01,'Input eligibility figure. Source style is not a required generation target. Source geometry and C remain unchanged.',fontsize=10)
    fig.savefig(path,dpi=130);plt.close(fig)

def compute():
    started=time.time();target=OUT/'condition_audit.json'
    if target.exists(): raise RuntimeError('Refusing to overwrite existing audit')
    m=read(DATA/'qualified_condition_manifest.json');old=read(OLD);pmap={p['uid']:p for p in m['parents']}
    chosen=[next(c for c in m['conditions'] if c['uid'].startswith(pre) and c['condition_type']=='SEAT_FULL') for pre in PREFIXES]
    protected={str(DATA/'qualified_condition_manifest.json'):sha(DATA/'qualified_condition_manifest.json'),str(OLD):sha(OLD)}
    rows=[];(OUT/'candidate_images').mkdir(parents=True,exist_ok=True)
    for c in chosen:
        p=pmap[c['uid']]
        for path,expected in ((c['npz'],c['sha256']),(p['npz'],p['sha256']),(c['annotation_source'],c['annotation_sha256']),(c['image'],c['image_sha256']),(c['quality_path'],c['quality_sha256'])):
            assert sha(path)==expected,(path,'source hash mismatch');protected[path]=expected
        with np.load(c['npz'],allow_pickle=False) as z,np.load(p['npz'],allow_pickle=False) as par:
            T=z['C_original'].copy();ids=z['C_original_source_vertex_ids'].copy();canonical=z['C'].copy()
            checks=dict(K_matches=int(z['K'])==len(T),source_vertex_coordinates_exact=np.array_equal(par['model_vertices'][ids],T),
                canonical_target_rows_exact=np.array_equal(par['full_target'][z['source_face_ids']].reshape(-1,3,3),canonical),
                entire_declared_region=np.array_equal(np.sort(z['source_face_ids']),np.sort(z['region_source_face_ids'])),
                source_corner_mapping_exact=np.array_equal(np.take_along_axis(T,z['C_source_corner_permutations'][:,:,None],axis=1),canonical),
                retained_faces_K_plus_free_N=len(z['source_face_ids'])+len(z['free_source_ids'])==int(z['N']))
            assert all(checks.values()),checks
            top=topology(ids);geo=geometry(T,ids,top)
            oldq=[q for q in old['queries'] if q['parent_id']==c['uid']]
            r=dict(condition_id=c['condition_id'],parent_id=c['uid'],condition_path=c['npz'],condition_sha256=c['sha256'],parent_npz=p['npz'],
                parent_sha256=p['sha256'],N=int(z['N']),K=int(z['K']),K_over_N=int(z['K'])/int(z['N']),
                source_face_ids=z['source_face_ids'].tolist(),original_source_face_ids=z['C_original_source_face_ids'].tolist(),
                source_vertex_ids=ids.tolist(),source_identity_checks=checks,topology=top,geometry=geo,
                semantic_review={k:c[k] for k in ('review_kind','review_reason','annotation_source','annotation_sha256','image','image_sha256','region_declared')},
                existing_source_quality=c['CC_CF_FF'],source_quality_path=c['quality_path'],source_quality_sha256=c['quality_sha256'],
                source_geometry_new_repairs=0,old_panel_conditions=[q['condition_id'] for q in oldq],
                already_FULL_in_previous_panel=any(q['base_condition_id']==c['condition_id'] for q in oldq),
                whole_region_AI_proxy_status='NOT_REVIEWED_THIS_RUN',input_eligibility='PENDING_INPUT_REVIEW')
            image=OUT/'candidate_images'/f"{c['condition_id']}.png"
            draw(image,T,par['model_vertices'][par['source_faces']],r)
            r['candidate_image']=str(image);r['candidate_image_sha256']=sha(image)
        rows.append(r)
    assert all(sha(p)==h for p,h in protected.items())
    save(target,dict(schema=VERSION,status='CPU_GEOMETRY_COMPLETE_PENDING_AI_PROXY',definitions={
        'topology':'Exact original source vertex IDs and full C source faces; no coordinate welding. Edge-connected components; simple boundary cycles require degree two.',
        'plane':'Area-integrated covariance best fit plane of finite C triangles in existing model units. Normal sign follows mean source normal; nonplanarity is descriptive, not automatic rejection.',
        'outline':'Best-plane projected finite triangle area / convex hull / minimum-area rectangle; ratios and aspect describe regularity, not user-intent validity.',
        'full':'FULL means the complete previously declared upper sitting surface, not the whole volumetric cushion. Curved/beveled rims can be part of that surface.',
        'source_quality':'Inherited pair audit only; CC/CF/FF counts remain distinct. This input audit does not rerun whole mesh intersections or certify clean source geometry.',
        'selection':'Six parent IDs fixed by prior authorized task; input qualifications only, no generated performance selection.',
        'human_review':False},conditions=rows,source_hashes=protected,source_hashes_unchanged=True,code_sha256=sha(__file__),
        actual_calls=dict(training=0,model_forward=0,generation=0,old_quality_scorer=0,source_geometry_mutations=0),elapsed_seconds=time.time()-started))
    print(json.dumps([{'parent':r['parent_id'][:6],'K':r['K'],'N':r['N'],'components':r['topology']['component_count'],'loops':r['topology']['boundary_loop_count'],'rms_over_diag':r['geometry']['plane_surface_RMS_over_patch_diagonal'],'fill':r['geometry']['projected_area_over_minimum_rectangle_area']} for r in rows],indent=2))

def record_review(path):
    p=OUT/'condition_audit.json';audit=read(p);reviews=read(path)
    assert audit['status']=='CPU_GEOMETRY_COMPLETE_PENDING_AI_PROXY'
    for r in audit['conditions']:
        v=reviews[r['parent_id'][:6]];assert v['actually_viewed'] is True
        assert sha(r['candidate_image'])==r['candidate_image_sha256']
        r['whole_region_AI_proxy_status']='REVIEWED_NONBLIND_AI_PROXY';r['input_review']=v;r['input_eligibility']=v['input_eligibility'];r['reviewed_image_sha256']=r['candidate_image_sha256']
    assert all(sha(p)==h for p,h in audit['source_hashes'].items())
    audit['status']='COMPLETE_INPUT_AUDIT';audit['review_record_path']=str(Path(path));audit['review_record_sha256']=sha(path);save(p,audit)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('compute','record-review'));ap.add_argument('--review-json');a=ap.parse_args()
    if a.command=='compute':compute()
    else:record_review(a.review_json)
