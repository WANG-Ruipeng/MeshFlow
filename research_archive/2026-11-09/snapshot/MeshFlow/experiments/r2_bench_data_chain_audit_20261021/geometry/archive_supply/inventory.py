"""Inventory existing Bench archive members in memory, without extraction or repair.

Bench synset is 02828884, verified against the existing historical source loader.
Only NumPy and the standard library; no learned models, mesh construction or FF.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import numpy as np

ROOT=Path('/mnt/e/MeshFlow')
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'downloaded_data/archives/shapenet-cls.tar.gz'
OLD=ROOT/'experiments/r2_bench_control_transfer_v1/data/manifest.json'
CURRENT=ROOT/'experiments/r2_bench_clean_multipart_seam_v1/data/manifest.json'
EXPECTED='06707caff4ce465f402e6b69f81056204c579921d371eb2c9e4e9a3929f7aef2'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def bin_N(n):
    if n<20:return 'UNDER_20'
    if n<=127:return '20_127'
    if n<=256:return '128_256'
    if n<=400:return '257_400'
    if n<=800:return '401_800'
    return 'OVER_800'

def main():
    started=time.time()
    input_hashes={str(p):sha(p) for p in (ARCHIVE,OLD,CURRENT)}
    if input_hashes[str(ARCHIVE)]!=EXPECTED: raise ValueError('Protected archive differs from existing registered source hash; stopped')
    old=json.loads(OLD.read_text(encoding='utf-8-sig')); cur=json.loads(CURRENT.read_text(encoding='utf-8-sig'))
    membership={r['parent_id']:r['group'] for r in old['train']+old['dev']}
    active={r['parent_id'] for r in cur['train']}
    rows=[]
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        members=[m for m in archive.getmembers() if m.isfile() and m.name.startswith('shapenet-02828884/') and '/objaverse_occ_v5_ids/' in m.name and m.name.endswith('.npz')]
        if len(members)!=560: raise ValueError('Expected560 Bench NPZ members; got'+str(len(members)))
        for m in sorted(members,key=lambda m:m.name):
            payload=archive.extractfile(m).read()
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                if 'vertices' not in z.files or 'faces' not in z.files: raise ValueError('Unexpected Bench NPZ schema: '+m.name)
                v=z['vertices'].copy(); f=z['faces'].copy()
            if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v):
                raise ValueError('Invalid source arrays: '+m.name)
            n=len(f);uid=Path(m.name).stem
            duplicates=n-len(np.unique(np.sort(f,axis=1),axis=0))
            tri=v.astype(np.float64)[f]
            cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            zeros=int(np.count_nonzero(np.all(cross==0,axis=1)))
            rows.append(dict(parent_id=uid,archive_member=m.name,member_sha256=hashlib.sha256(payload).hexdigest(),vertices=len(v),N=n,N_bin=bin_N(n),stored_vertex_dtype=str(v.dtype),
                source_ID_duplicate_excess_faces=duplicates,exact_zero_area_faces_original_float64=zeros,
                passes_two_exact_checks=(duplicates==0 and zeros==0),existing64_8_membership=membership.get(uid,'UNSELECTED'),current_TRAIN58=uid in active,
                has_historical_local_condition=uid in membership,full_FF='NOT_RUN',semantic_region_eligibility='NOT_RUN',intersection_free_certified=False))
    counts=collections.Counter(r['parent_id'] for r in rows)
    bins={}
    for b in ('UNDER_20','20_127','128_256','257_400','401_800','OVER_800'):
        rs=[r for r in rows if r['N_bin']==b]
        bins[b]=dict(mesh_members=len(rs),unique_UIDs=len({r['parent_id'] for r in rs}),
            no_ID_duplicate_members=sum(r['source_ID_duplicate_excess_faces']==0 for r in rs),no_exact_zero_area_members=sum(r['exact_zero_area_faces_original_float64']==0 for r in rs),
            passes_both_exact_checks=sum(r['passes_two_exact_checks'] for r in rs),
            existing_TRAIN64=sum(r['existing64_8_membership']=='TRAIN' for r in rs),existing_DEV8=sum(r['existing64_8_membership']=='DEV' for r in rs),current_TRAIN58=sum(r['current_TRAIN58'] for r in rs),
            unselected_members=sum(r['existing64_8_membership']=='UNSELECTED' for r in rs),unselected_passing_both=sum(r['existing64_8_membership']=='UNSELECTED' and r['passes_two_exact_checks'] for r in rs),
            duplicate_excess_faces=sum(r['source_ID_duplicate_excess_faces'] for r in rs),exact_zero_area_faces=sum(r['exact_zero_area_faces_original_float64'] for r in rs))
    if bins['128_256']['mesh_members']!=88: raise ValueError('Archive N128..256 disagrees with historical88')
    if sum(r['existing64_8_membership']!='UNSELECTED' for r in rows)!=72: raise ValueError('Historical72 parents not all matched')
    with (OUT/'per_parent.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for p,h in input_hashes.items():
        if sha(p)!=h: raise ValueError('Protected input changed: '+p)
    write('input_hashes.json',input_hashes)
    summary=dict(status='COMPLETE',archive=str(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,archive_sha256=EXPECTED,category='Bench',synset='02828884',identifier_correction='02924116 in delegated request corrected using existing source loader; that class was not inventoried as Bench',
        mesh_members=len(rows),unique_UIDs=len(counts),multiple_members_same_UID={k:v for k,v in counts.items() if v>1},bins=bins,
        N_min=min(r['N'] for r in rows),N_max=max(r['N'] for r in rows),
        definitions=dict(bins='Inclusive integer ranges20..127,128..256,257..400,401..800; under20 and over800 separately retained',
            duplicates='Faces with equal unordered original vertex-ID triples; excess beyond first, without welding coincident vertices',
            zero_area='Exact zero cross product calculated in float64 from stored ORIGINAL coordinates; no threshold and no model normalization',
            passing='Only zero same-ID duplicate excess AND zero exact zero-area faces. Not a clean, manifold, intersection-free, complete, diverse or usable-condition certificate',
            membership='Historical64TRAIN/8DEV and current58TRAIN parent membership; no new splits or conditions created'),
        all_input_hashes_unchanged=True,no_extraction=True,source_mutations=0,full_FF='NOT_RUN',GPU_calls=0,model_calls=0,training_updates=0,CPU_workers=1,numeric_threads=1,wall_seconds=time.time()-started,script_sha256=sha(__file__))
    write('summary.json',summary);write('zero_gpu_ledger.json',dict(status='COMPLETE',started_unix=started,finished_unix=time.time(),wall_seconds=time.time()-started,CPU_workers=1,numeric_threads=1,archive_members_read=len(rows),GPU_calls=0,model_imports=0,model_forwards=0,new_generation=0,geometry_construction=0,repairs=0,extractions=0,downloads=0,source_mutations=0))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
