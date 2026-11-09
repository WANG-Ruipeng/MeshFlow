"""Independent selected-cache graph/lineage review, no detector or model imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/e/MeshFlow/experiments/r2_bench_post_rescue_diagnostic_v1')
SOURCE=Path('/mnt/e/MeshFlow/experiments/r2_bench_parts_convergence_v1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def bfs(seed,adj):
    seen=set(seed);queue=list(seed)
    while queue:
        i=queue.pop()
        for j in adj[i]-seen:seen.add(j);queue.append(j)
    return seen
done=read(ROOT/'point_contact/completion.json')
assert done['code_sha256']==sha('/mnt/e/MeshFlow/r2/pilots/bench_post_rescue_diagnostic_v1/point_contact.py')
for p,h in done['outputs'].items():assert sha(p)==h
rows=csvread(ROOT/'point_contact/per_output.csv');assert len(rows)==96
manifest={q['request_id']:q for q in read(SOURCE/'evaluation_inputs/manifest.json')['requests']}
for row in rows:
    if row['status']=='NA_CLOSED_C':
        assert manifest[row['request_id']]['boundary_edges']==0
        assert all(row[k]=='' for k in ('original_legal_edge_reach_fraction','point_component_bulk_fraction','extra_point_only_bulk_fraction','edge_or_point_bulk_fraction'))
    else:assert row['status']=='PASS' and manifest[row['request_id']]['boundary_edges']>0
selected=[]
for label in ('SEAM_S15000','SEAM_S21000'):
    rr=[r for r in rows if r['label']==label]
    selected.append(max((r for r in rr if r['status']=='PASS'),key=lambda r:float(r['point_component_bulk_fraction'])))
    selected.append(next(r for r in rr if r['status']=='NA_CLOSED_C'))
checked=[]
for row in selected:
    label=row['label'];rid=row['request_id'];d=SOURCE/'evaluation'/label/rid
    sd=SOURCE/'evaluation/scores'/label/rid/'POST'
    z=arr(d/'post/post.npz');raw=arr(d/'raw/raw.npz');K=int(z['known_faces']);f=z['faces'];t=z['output'].astype(float)
    seam=set(np.flatnonzero(z['retained_seam_mask']).tolist());bulk=set(range(K,len(f)))-seam
    assert all(i>=K for i in seam)
    area=.5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    adj=collections.defaultdict(set);badj=collections.defaultdict(set);start=set()
    for e in csvread(sd/'valid_edges.csv'):
        if e['valid_edge']!='True':continue
        i,j=json.loads(e['incident_face_ids'])
        if min(i,j)<K:
            if max(i,j) in seam:start.add(max(i,j))
        else:
            adj[i].add(j);adj[j].add(i)
            if i in bulk and j in bulk:badj[i].add(j);badj[j].add(i)
    reached=bfs(start,adj);Cvid=set(f[:K].ravel().tolist());pairs=read(sd/'pairs.json')
    ff={tuple(sorted((int(r['pair'][0])+K,int(r['pair'][1])+K))):r['classification'] for r in pairs['FF']['contacts']}
    pts=set();npairs=0;tips=0
    for i in seam:
        slot=int(z['source_face_ids'][i]);assert raw['raw_seam_slot_mask'][slot]
        j=np.flatnonzero(z['source_corner_ids'][i]==2).item();vid=int(f[i,j]);tips+=1
        # Actual exported index is the original generated slot corner2 target.
        assert vid==int(z['corner_vertex_ids'][slot,2])
        if vid in Cvid or i not in reached:continue
        for b in bulk:
            shared=set(f[i].tolist())&set(f[b].tolist())
            if shared=={vid} and ff.get(tuple(sorted((i,b))))=='shared_vertex_contact':pts.add(b);npairs+=1
    pointbulk=bfs(pts,badj);basebulk=reached&bulk;extra=pointbulk-basebulk;den=float(area[sorted(bulk)].sum())
    assert len(pointbulk)==int(row['point_component_bulk_faces']) and npairs==int(row['verified_point_pairs'])
    if row['status']=='PASS':
        for key,value in (('point_component_bulk_fraction',area[sorted(pointbulk)].sum()/den),
                          ('extra_point_only_bulk_fraction',area[sorted(extra)].sum()/den),
                          ('edge_or_point_bulk_fraction',area[sorted(extra|basebulk)].sum()/den)):
            assert abs(float(row[key])-value)<1e-12,(rid,key,value,row[key])
    checked.append(dict(label=label,request_id=rid,status=row['status'],tips_checked=tips,verified_point_pairs=npairs,
        point_bulk_faces=len(pointbulk),source_corner2_and_FF_local_to_global='PASS',one_point_then_bulk_only='PASS'))
summary=read(ROOT/'point_contact/summary.json');fields=('original_legal_edge_reach_fraction','point_component_bulk_fraction','extra_point_only_bulk_fraction','edge_or_point_bulk_fraction','any_seam_tip_component_fraction')
for s in summary:
    rr=[r for r in rows if r['label']==s['label'] and (s['split']=='ALL' or r['split']==s['split'])]
    assert len(rr)==s['outputs'] and sum(r['status']=='PASS' for r in rr)==s['applicable']
    assert sum(r['edge_or_point_bulk_fraction']!='' and float(r['edge_or_point_bulk_fraction'])>=.3 for r in rr)==s['edge_or_point_ge30']
    for field in fields:
        pp=[]
        for pid in {r['parent_id'] for r in rr}:
            cr=[r for r in rr if r['parent_id']==pid];cv=[]
            for cid in {r['condition_id'] for r in cr}:
                v=[float(r[field]) for r in cr if r['condition_id']==cid and r[field]!='']
                if v:cv.append(sum(v)/len(v))
            if cv:pp.append(sum(cv)/len(cv))
        assert abs(sum(pp)/len(pp)-s[field])<1e-12
gallery=ROOT/'gallery/RAW_PRE_GEO_POST';definition=read(gallery/'definition.json');gdone=read(gallery/'completion.json')
for p,h in definition['geometry_hashes'].items():assert sha(p)==h
images=[]
for p in gallery.glob('*.json'):
    r=read(p)
    if 'image_sha256' not in r:continue
    assert sha(r['path'])==r['image_sha256'] and sha(gallery/'definition.json')==r['definition_sha256']
    assert r['request_id'] in manifest and len(r['cells'])==3
    for cell,suffix in zip(r['cells'],('raw/raw.npz','post/pre_geo.npz','post/post.npz')):
        expected=SOURCE/'evaluation/SEAM_S21000'/r['request_id']/suffix
        assert Path(cell['path'])==expected and str(expected) in definition['geometry_hashes']
    images.append(dict(request_id=r['request_id'],image=r['path'],image_sha256=r['image_sha256'],stage_identity='PASS',review='IDENTITY_ONLY_NOT_NEW_VISUAL_RATING'))
assert len(images)==3
result=dict(status='PASS',code_review='tip source corner2, FF local+K, exact 1 non-C shared tip, cached shared_vertex_contact, one point jump then legal bulk-only edges',
    selected_cache_checks=checked,all96_NA_and_6_macro_summaries='PASS',gallery_identity=images,
    interpretation='Diagnostic point-touch reach only. It neither certifies legal edge seams nor complete/nonintersecting bulk.',
    denominator_note='40 applicable/48 total per checkpoint; 8 closed requests excluded. Counts9->11 >=30% are point-or-edge diagnostic, not success counts.',
    original_code_sha256=done['code_sha256'],model_calls=0,scorer_calls=0,new_geometry_processing_calls=0,new_visual_ratings=0)
out=ROOT/'audits/point_contact_independent_review.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
