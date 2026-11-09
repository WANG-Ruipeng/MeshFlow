"""Independent CPU-only feasibility audit; does not select or freeze any sample."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import json, hashlib, time, subprocess
from pathlib import Path
from collections import Counter, defaultdict, deque
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'native_t1/runs/control_mvp48_v1'
OUT=RUN/'registration/data_repair_feasibility.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def resources():
    def command(args):
        r=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    return dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        gpu=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        cores=os.cpu_count(),affinity=len(os.sched_getaffinity(0)),
        meminfo=Path('/proc/meminfo').read_text(),loadavg=Path('/proc/loadavg').read_text(),
        processes=command(['ps','-eo','pid,comm,pcpu,pmem,rss,args']),
        workers=1,numerical_threads=1,gpu_workers=0)

def edges(faces):
    result=defaultdict(list)
    for i,f in enumerate(faces):
        for a,b in ((0,1),(1,2),(2,0)):
            result[tuple(sorted((int(f[a]),int(f[b]))))].append(i)
    return result

def patches(v,raw_faces,keep):
    f=raw_faces[keep]; n=len(f); k=int(np.floor(.2*n+.5))
    rawedge=edges(raw_faces); edge=edges(f); adj=[set() for _ in f]
    for members in edge.values():
        for a in members: adj[a].update(x for x in members if x!=a)
    cent=v[f].astype(np.float64).mean(1)
    distances=((cent-cent.mean(0))**2).sum(1)
    starts=set(range(n)); tried=0; sets=set(); coord=set(); good=[]; rejected=Counter()
    while starts and len(good)<4:
        s=min(starts,key=lambda i:(-float(distances[i]),int(keep[i])))
        starts.remove(s); tried+=1
        dist=((cent-cent[s])**2).sum(1)
        distances=dist if tried==1 else np.minimum(distances,dist)
        todo=deque([s]); seen={s}; known=[]
        while todo and len(known)<k:
            i=todo.popleft();known.append(i)
            for j in sorted(adj[i],key=lambda x:int(keep[x])):
                if j not in seen:seen.add(j);todo.append(j)
        reason=None; interface=[]; ks=set(known)
        if len(known)!=k: reason='component_smaller_than_K'
        else:
            for e,mm in sorted(edge.items()):
                left=[i for i in mm if i in ks];right=[i for i in mm if i not in ks]
                if left and right:
                    if len(mm)!=2 or len(left)!=1 or len(right)!=1:
                        reason='cleaned_nonmanifold_crossing';break
                    if len(rawedge[e])!=2 or set(rawedge[e])!={int(keep[left[0]]),int(keep[right[0]])}:
                        reason='original_raw_interface_not_degree2';break
                    interface.append((e,left[0],right[0]))
            if reason is None and not interface:reason='empty_interface'
            if reason is None and tuple(sorted(known)) in sets:reason='duplicate_condition_set'
            if reason is None and tuple(cent[s]) in coord:reason='duplicate_start_centroid'
        if reason:rejected[reason]+=1;continue
        sets.add(tuple(sorted(known)));coord.add(tuple(cent[s]))
        good.append(dict(start_original_source_face_id=int(keep[s]),interface_edge_count=len(interface),
            known_original_source_face_ids=[int(keep[x]) for x in known]))
    return good,dict(rejected)

def main():
    started=time.perf_counter();before=resources()
    old_path=RUN/'data/candidate_audit.json'
    old=json.loads(old_path.read_text(encoding='utf-8-sig'))
    rows=[]
    for row in old['objects']:
        if not 128<=row['N']<=256:continue
        p=ROOT/row['source_path']
        if sha(p)!=row['source_sha256']:raise RuntimeError('Official source changed '+str(p))
        with np.load(p,allow_pickle=False) as z:
            v=z['vertices'].copy();f=z['faces'].copy()
        n=len(f);rec={k:row[k] for k in ('uid','official_split','source_path','source_sha256','group_id','historically_exposed_group')}
        rec['raw_N']=n;rec['rejection_reasons']=[]
        if not np.isfinite(v).all() or np.any(f<0) or np.any(f>=len(v)):
            rec['rejection_reasons'].append('invalid_geometry');rows.append(rec);continue
        ext=float(np.ptp(v,axis=0).max())
        if ext<=0:rec['rejection_reasons'].append('zero_extent');rows.append(rec);continue
        tri=v[f].astype(np.float64);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        area=np.linalg.norm(cross,axis=1)
        zero=np.all(cross==0,axis=1); keep=[];removed_zero=[];removed_dup=[];seen={}
        for i,t in enumerate(tri):
            if zero[i]:removed_zero.append(i);continue
            key=tuple(sorted(tuple(float(x) for x in corner) for corner in t))
            if key in seen:removed_dup.append(dict(removed_face_id=i,retained_face_id=seen[key]));continue
            seen[key]=i;keep.append(i)
        ids=np.array(keep,dtype=np.int64);rec['repaired_N']=len(ids)
        rec['removed_exact_zero_source_face_ids']=removed_zero
        rec['removed_duplicate_source_faces']=removed_dup
        rec['retained_original_source_face_ids']=keep
        rec['tiny_nonzero_retained_count']=int(np.count_nonzero((area[ids]>0)&(area[ids]<=1e-12*ext**2)))
        if not 128<=len(ids)<=256:rec['rejection_reasons'].append('repaired_face_count_outside_128_256')
        if rec['tiny_nonzero_retained_count']:rec['rejection_reasons'].append('tiny_nonzero_area_faces_retained_not_repaired')
        rec['patch_count']=0
        if not rec['rejection_reasons']:
            good,rejections=patches(v,f,ids);rec['patch_count']=len(good)
            rec['accepted_patches']=good;rec['patch_candidate_rejections']=rejections
            if len(good)<2:rec['rejection_reasons'].append('fewer_than_2_applicable_patches')
        if sha(p)!=row['source_sha256']:raise RuntimeError('Official source changed during scan '+str(p))
        rows.append(rec)
    counts={}
    for split in ('train','test'):
        ss=[r for r in rows if r['official_split']==split]
        qualified=[r for r in ss if not r['rejection_reasons']]
        counts[split]=dict(raw_range_count=len(ss),repaired_range_count=sum(128<=r.get('repaired_N',0)<=256 for r in ss),
            applicable_2patch_objects=len(qualified),applicable_4patch_objects=sum(r['patch_count']>=4 for r in qualified),
            unexposed_2patch_existing_groups=len({r['group_id'] for r in qualified if not r['historically_exposed_group']}),
            rejection_reasons=dict(Counter(x for r in ss for x in r['rejection_reasons'])))
    result=dict(schema='control_mvp48_repair_feasibility_v1',experiment_id='control_mvp48_v1',owner='Session A independent audit',
        status='FEASIBILITY_ONLY_NOT_SELECTION_OR_FREEZE',source_audit_sha256=sha(old_path),resources_before=before,resources_after=resources(),
        counts=counts,qualified_test_uids=[r['uid'] for r in rows if r['official_split']=='test' and not r['rejection_reasons'] and not r['historically_exposed_group']],
        repair_rule='Drop exact-zero cross-product faces, then retain lowest original source face ID for exact unordered coordinate triangle duplicates; original vertices and retained face coordinates unchanged. Raw and retained face counts both 128..256. Tiny nonzero area faces remain a rejection.',
        interface_rule='Every repaired crossing edge must have exactly two original edge incidences and same original source face IDs as the retained known/free pair. No labels created by deleting incidents.',
        grouping_limitation='Counts use already audited original-geometry and UID/historical/near groups. Root must additionally group repaired geometry and recheck near duplicates before final selection. This audit does not select a split.',
        models_called=0,ot_called=0,optimizer_updates=0,noise_arrays=0,objects=rows,elapsed_seconds=time.perf_counter()-started)
    content=(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if OUT.exists():
        raise RuntimeError('Refuse overwrite '+str(OUT))
    with OUT.open('xb') as target:target.write(content)
    print(json.dumps({k:result[k] for k in ('status','counts','qualified_test_uids','elapsed_seconds')},sort_keys=True),flush=True)

if __name__=='__main__':main()
