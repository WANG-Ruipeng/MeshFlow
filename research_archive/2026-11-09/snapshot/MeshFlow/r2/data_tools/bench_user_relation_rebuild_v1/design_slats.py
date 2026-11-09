"""CPU-only face-selection design; immutable old meshes remain read-only."""
from pathlib import Path
import os, json, hashlib, sys, subprocess
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path('/mnt/e/MeshFlow')
OUT=ROOT/'experiments/r2_bench_user_relation_rebuild_v1/design_slats'
PIDS=['a8e987a6a6486b962d6c8e51365a5a87','7b58a3e5d7f4dddcad1d789f3b2120d0']
def save(name,x):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def mesh(pid):
    plan=json.loads((ROOT/'experiments/r2_bench_relation_coverage_v1/render_plan.json').read_text())
    r=next(x for x in plan['parents'] if x['parent_id']==pid)
    a=np.load(r['target_mesh_path']);return r,a['vertices'],a['faces']
def coplanar(v,f):
    t=v[f].astype(float);cr=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=np.linalg.norm(cr,axis=1)/2;n=cr/np.maximum(2*area[:,None],1e-30)
    edges={}
    for i,face in enumerate(f):
        for a,b in zip(face,np.roll(face,-1)):edges.setdefault(tuple(sorted([int(a),int(b)])),[]).append(i)
    adj=[[] for _ in f]
    for group in edges.values():
        if len(group)==2:
            a,b=group
            if np.dot(n[a],n[b])>.999999 and abs(np.dot(n[a],t[a,0]-t[b,0]))<1e-5:adj[a].append(b);adj[b].append(a)
    seen=set();groups=[]
    for i in range(len(f)):
        if i in seen:continue
        todo=[i];seen.add(i);ids=[]
        while todo:
            j=todo.pop();ids.append(j)
            for k in adj[j]:
                if k not in seen:seen.add(k);todo.append(k)
        ids=sorted(ids);pts=t[ids].reshape(-1,3)
        groups.append(dict(face_ids=ids,K=len(ids),area=float(area[ids].sum()),centroid=np.average(t[ids].mean(1),axis=0,weights=area[ids]).tolist(),normal=n[ids[0]].tolist(),bbox=[pts.min(0).tolist(),pts.max(0).tolist()]))
    return groups
def largest_edge_component(f,ids):
    edge_faces={}
    for i in ids:
        for a,b in zip(f[i],np.roll(f[i],-1)):edge_faces.setdefault(tuple(sorted([int(a),int(b)])),[]).append(i)
    adj={i:set() for i in ids}
    for group in edge_faces.values():
        for i in group:adj[i].update(set(group)-{i})
    seen=set();groups=[]
    for i in ids:
        if i in seen:continue
        todo=[i];seen.add(i);g=[]
        while todo:
            j=todo.pop();g.append(j)
            for k in adj[j]:
                if k not in seen:seen.add(k);todo.append(k)
        groups.append(g)
    return sorted(max(groups,key=lambda g:(len(g),-min(g))))
def render(pid,v,f,entries,name):
    from PIL import Image,ImageDraw
    from r2.data_tools.bench_user_v2.render import cell,basis,font,shared_bounds
    cams=[basis(-60,22),basis(120,22),basis(-90,0)]
    t=v[f];cen,span=shared_bounds([t],cams);s=430
    page=Image.new('RGB',(s*4,90+len(entries)*(s+60)),'white');d=ImageDraw.Draw(page)
    d.text((12,10),pid[:8]+' | exact target face subsets, C blue / Y gray',font=font(19),fill='black')
    for i,e in enumerate(entries):
        ids=e['face_ids'];cy=np.concatenate([t[ids],np.delete(t,ids,axis=0)]);y=90+i*(s+60)
        d.text((10,y),e['name']+' K='+str(len(ids)),font=font(17),fill='black')
        for j,(tri,k,cam) in enumerate([(cy,len(ids),cams[0]),(t[ids],len(ids),cams[0]),(cy,len(ids),cams[1]),(t[ids],len(ids),cams[2])]):
            im,_=cell(tri,k,cam,cen,span,s);page.paste(im,(s*j,y+40))
    path=OUT/(name+'.png');page.save(path);return str(path)
def inspect():
    import psutil
    save('resources.json',dict(cpu_cores=psutil.cpu_count(),available_ram_bytes=psutil.virtual_memory().available,load=os.getloadavg(),workers=1,numeric_threads=1,gpu_work=0,gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout))
    for pid in PIDS:
        r,v,f=mesh(pid);g=coplanar(v,f);save(pid[:6]+'_facets.json',g)
        render(pid,v,f,[dict(name='Whole target',face_ids=list(range(len(f))))],pid[:6]+'_whole')
        print(pid,'bbox',v.min(0),v.max(0))
        for i,x in enumerate(g):
            if x['area']>.05:print(i,'K',x['K'],'area',round(x['area'],3),'center',np.round(x['centroid'],3),'normal',np.round(x['normal'],3),'ids',x['face_ids'])
    assert 'torch' not in sys.modules
def design():
    allrows=[]
    for pid in PIDS:
        r,v,f=mesh(pid);t=v[f]
        if pid.startswith('a8'):
            entries=[dict(name='Full slatted top with end rails',key='SLATTED_TOP_FULL',face_ids=list(range(150,189)),part_type='SEAT_SURFACE',reason='Complete upward planar seat surface includes every slat and both end rails; natural openings retained.'),
                     dict(name='One complete central slat top',key='SINGLE_SLAT_TOP_FULL',face_ids=[177,178],part_type='SEAT_SURFACE',reason='Exact two-triangle rectangular upper face of the full central slat between end rails; no truncation or resampling.'),
                     dict(name='Complete left transverse end frame',key='END_FRAME_FULL',face_ids=np.flatnonzero(t[:,:,0].max(1)<-2.54).tolist(),part_type='LEG_SUPPORT',reason='All original faces contained in the left end-frame component, including two complete legs and crossrail, selected at actual junction rather than mid-leg crop.')]
        else:
            entries=[dict(name='One complete central slat top',key='SINGLE_SLAT_TOP_FULL',face_ids=[237,238],part_type='SEAT_SURFACE',reason='Exact two-triangle rectangular upper face of the full central slat between end walls; natural neighboring gaps retained.'),
                     dict(name='Complete left curved end wall',key='END_WALL_FULL',face_ids=largest_edge_component(f,np.flatnonzero(t[:,:,0].max(1)<-2.23).tolist()),part_type='LEG_SUPPORT',reason='Entire edge-connected outer/inner/rounded surfaces of the left terminal wall; begins at physical top/bottom junctions. Separate slat-end cap faces are excluded. This is an end wall, not two separate leg components.')]
        image=render(pid,v,f,entries,pid[:6]+'_candidates')
        for e in entries:
            e.update(region_name=e['key'],semantic_description=e['reason'])
            e.update(parent_id=pid,target_mesh_path=r['target_mesh_path'],target_case_path=r['target_case_path'],target_mesh_sha256=hashlib.sha256(Path(r['target_mesh_path']).read_bytes()).hexdigest(),target_face_ids=e['face_ids'],K=len(e['face_ids']),N=len(f),image_path=image,reviewer='AI_PROXY',review_status='PENDING_ACTUAL_IMAGE_REVIEW',source_face_id_contract='Indices into unchanged versioned target_mesh/source_faces; upstream original lineage remains in old parent lineage.json',limitations=['Selection is from existing target; not new independent parent','Complete upper surface is not closed solid component' if 'TOP' in e['key'] else 'Whole-component classification requires actual image review and topology validation'])
            allrows.append(e)
    save('selection_proposals.json',allrows)
    print([(r['parent_id'][:6],r['key'],r['K']) for r in allrows])
    assert 'torch' not in sys.modules
def finalize_review():
    rows=json.loads((OUT/'selection_proposals.json').read_text())
    plan=json.loads((ROOT/'experiments/r2_bench_relation_coverage_v1/render_plan.json').read_text())
    decisions=[]
    for row in rows:
        pid=row['parent_id'];_,v,f=mesh(pid);ids=row['target_face_ids'];cf=f[ids];edges={}
        for i,face in enumerate(cf):
            for a,b in zip(face,np.roll(face,-1)):edges.setdefault(tuple(sorted([int(a),int(b)])),[]).append(i)
        adj=[set() for _ in cf]
        for x in edges.values():
            for i in x:adj[i].update(set(x)-{i})
        seen=set();comps=[]
        for i in range(len(cf)):
            if i in seen:continue
            todo=[i];seen.add(i);comp=[]
            while todo:
                j=todo.pop();comp.append(j)
                for k in adj[j]:
                    if k not in seen:todo.append(k);seen.add(k)
            comps.append(len(comp))
        old=[r for r in plan['conditions'].values() if r['parent_id']==pid]
        duplicates=[r['condition_id'] for r in old if sorted(r.get('target_face_ids',[]))==sorted(ids)]
        image=row['image_path'];row.update(region_name=row['key'],semantic_description=row['reason'])
        record=dict(parent_id=pid,region_name=row['key'],reviewer='AI_PROXY',actually_viewed=True,image_path=image,image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest(),decision='ACCEPT_SEMANTIC_PENDING_PRODUCTION_VALIDATION',boundary_edges=sum(len(x)==1 for x in edges.values()),nonmanifold_edges=sum(len(x)>2 for x in edges.values()),edge_connected_components=comps,duplicates_existing=duplicates)
        if row['key']=='SINGLE_SLAT_TOP_FULL':
            vv=v[np.unique(cf)];lens=np.sort(np.ptp(vv,axis=0));record['planar_aspect_ratio']=float(lens[-1]/lens[-2]);record['note']='Complete physical rectangular slat upper surface, K=2; naturally narrow (not random/cropped strip), not a closed solid slat. Do not claim low small-angle rate.'
        elif row['key']=='SLATTED_TOP_FULL':record['note']='Entire upper seating surface includes repeated slats, two edge strips and both end rails; connected by original end rails; all genuine gaps retained.'
        elif row['key']=='END_FRAME_FULL':record['note']='Both complete end legs and their full transverse upper crossrail, including all source side faces; original joints remain open where attached to longitudinal body.'
        else:record['note']='Complete terminal wall from feet to rounded top, inner and outer faces retained; not a planar slice or truncated leg. Junction to long lower board and seat remains open.'
        assert not duplicates
        assert hashlib.sha256(Path(row['target_mesh_path']).read_bytes()).hexdigest()==row['target_mesh_sha256']
        decisions.append(record)
    save('selection_proposals.json',rows)
    save('visual_review_and_topology.json',decisions)
    save('unchanged_and_excluded.json',dict(meshes_unchanged=True,model_calls=0,gpu_work=0,new_parents=0,proposals=5,excluded_duplicate=dict(parent_id=PIDS[1],region='SLATTED_TOP_FULL',target_face_ids=list(range(228,256)),old_condition_id=PIDS[1]+'__SEAT_SURFACE_1__V2E',reason='Exactly the existing accepted full slatted upper-surface condition; not counted as new.'),input_hashes={r['target_mesh_path']:r['target_mesh_sha256'] for r in rows}))
    print(json.dumps(decisions,indent=2))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='design':design()
    elif len(sys.argv)>1 and sys.argv[1]=='finalize':finalize_review()
    else:inspect()
