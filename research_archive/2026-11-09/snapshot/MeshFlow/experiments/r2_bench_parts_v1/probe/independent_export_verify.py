"""Independent source/partition/export check; no construction helper imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import hashlib, json, time
from pathlib import Path
import numpy as np

root=Path('/mnt/e/MeshFlow')
out=root/'experiments/r2_bench_parts_v1'
def path(s):
    p=Path(s)
    return p if p.is_absolute() else root/p
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

started=time.time()
oldpath=root/'experiments/r2_bench_control_transfer_v1/data/manifest.json'
old=read(oldpath)
assert sha(oldpath)=='a2d6ed3e3afe608d4e7ef5cb6b596ebcee5be1665946bd3438c1ee14d16f3894'
source={r['parent_id']:r for r in old['train']+old['dev']}
assert len(source)==72
arrays={}
for pid,s in source.items():
    p=path(s['case_path'])
    assert sha(p)==s['case_sha256'],pid
    arrays[pid]=load(p)

mp=out/'data/manifest.json'
manifest=read(mp)
rows=manifest['rows']
seen=set();face_sets=set(); checks=[]
allowed={'C','C_vertices','C_faces_original','C_faces_model','C_source_corner_permutations','N','K'}
for r in rows:
    cid=r['condition_id'];pid=r['parent_id'];s=source[pid];a=arrays[pid]
    assert cid not in seen,cid
    seen.add(cid)
    assert r['group']==s['group'] and r['official_split']==s['official_split'],cid
    assert r['source_case_sha256']==s['case_sha256'],cid
    assert path(r['source_case_path']).resolve()==path(s['case_path']).resolve(),cid
    p=load(path(r['paired_path']));x=load(path(r['input_path']))
    assert sha(path(r['paired_path']))==r['paired_sha256'],cid
    assert sha(path(r['input_path']))==r['input_sha256'],cid
    assert set(x)==allowed,(cid,set(x))
    assert all(same(x[k],p[k]) for k in allowed),cid
    ids=np.asarray(r['source_face_ids'],dtype=np.int64)
    assert len(ids)==len(set(ids.tolist())) and len(ids)>=2,cid
    fk=(pid,tuple(sorted(ids.tolist())))
    assert fk not in face_sets,cid
    face_sets.add(fk)
    free=p['free_ids']
    assert same(ids,p['known_ids']) and same(ids,p['source_face_ids']),cid
    assert same(free,p['free_source_face_ids']),cid
    assert set(ids).isdisjoint(set(free)),cid
    assert sorted(np.r_[ids,free].tolist())==list(range(len(a['full_target']))),cid
    assert int(p['N'])==r['N']==len(a['full_target']),cid
    assert int(p['K'])==r['K']==len(ids) and len(free)>0,cid
    assert same(p['C'],a['full_target'][ids]),cid
    assert same(p['Y'],a['full_target'][free]),cid
    assert same(p['C_pre_ot'],a['pre_ot_full'][ids]),cid
    assert same(p['Y_pre_ot'],a['pre_ot_full'][free]),cid
    reconstructed=np.empty_like(a['full_target'])
    reconstructed[ids]=p['C'];reconstructed[free]=p['Y']
    assert same(reconstructed,a['full_target']),cid
    vids=p['C_source_vertex_ids']
    assert same(p['C_vertices'],a['model_vertices'][vids]),cid
    assert same(vids[p['C_faces_original']],a['source_faces'][ids]),cid
    assert same(vids[p['C_faces_model']],a['source_face_vertex_ids'][ids]),cid
    assert same(p['C_vertices'][p['C_faces_model']],p['C']),cid
    assert same(p['C_source_corner_permutations'],a['source_corner_permutations'][ids]),cid
    assert same(p['C_source_faces_original'],a['source_faces'][ids]),cid
    # Independently parse OBJ and compare float32 roundtrip and ordered faces.
    objpath=path(r['input_obj_path']);assert sha(objpath)==r['input_obj_sha256'],cid
    vv=[];ff=[]
    for line in objpath.read_text('ascii').splitlines():
        token=line.split()
        if token and token[0]=='v': vv.append([float(t) for t in token[1:]])
        if token and token[0]=='f': ff.append([int(t)-1 for t in token[1:]])
    assert same(np.asarray(vv,dtype=x['C_vertices'].dtype),x['C_vertices']),cid
    assert same(np.asarray(ff,dtype=x['C_faces_original'].dtype),x['C_faces_original']),cid
    assert r['review']['decision']=='ACCEPT' and r['review']['human_plausible_input']=='YES',cid
    checks.append(dict(condition_id=cid,source_partition_bitwise=True,original_winding_ids_exact=True,
                       input_keys_only_C_and_N_K=True,obj_roundtrip_bitwise=True,source_split_exact=True))

for group in ['TRAIN','DEV']:
    rr=read(out/('data/'+group.lower()+'.json'))['rows']
    assert {r['condition_id'] for r in rr}=={r['condition_id'] for r in rows if r['group']==group}
train={r['parent_id'] for r in rows if r['group']=='TRAIN'}
dev={r['parent_id'] for r in rows if r['group']=='DEV'}
assert not train&dev
assert train=={r['parent_id'] for r in old['train']}
assert dev=={r['parent_id'] for r in old['dev']}
report=dict(status='PASS',independent_checker='AI_PROXY_bench_parts_constructor',
    manifest_sha256=sha(mp),source_manifest_sha256=sha(oldpath),sources_checked=len(source),
    conditions_checked=len(rows),train_parents=len(train),dev_parents=len(dev),rows=checks,
    wall_seconds=time.time()-started,geometry_rescoring=False,models_imported=False,model_calls=0)
dest=out/'independent_export_verification.json'
with dest.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
