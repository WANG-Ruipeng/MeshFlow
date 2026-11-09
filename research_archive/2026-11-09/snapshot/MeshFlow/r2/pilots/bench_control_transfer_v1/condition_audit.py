"""Read-only finite C-face pair audit, separate from full-source quality."""
import os,sys,json,importlib,csv
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));OUT=ROOT/'experiments/r2_bench_control_transfer_v1/data'
from r2.data_tools.user_conditions_v1.quality import frozen_detector
frozen_detector();D=importlib.import_module('_r2_quality_frozen_geometry_v1.intersections')
manifest=json.loads((OUT/'manifest.json').read_text('utf-8'));allrows=[];pairs=[]
for r in manifest['train']+manifest['dev']:
    with np.load(r['case_path'],allow_pickle=False) as z:v=z['C_vertices'].astype(float);f=z['C_faces_original'];full=z['model_vertices'].astype(float)
    tri=v[f];L=float(np.linalg.norm(np.ptp(full,axis=0)));lo=tri.min(1);hi=tri.max(1);cnt={};bad=0;unresolved=0
    for i in range(len(f)):
        for j in range(i+1,len(f)):
            if np.any(lo[i]>hi[j]+L*1e-9) or np.any(lo[j]>hi[i]+L*1e-9):continue
            q=D.verified_pair(tri[i],tri[j],L);c=q['classification'];cnt[c]=cnt.get(c,0)+1
            if q['illegal'] or not q['verified']:
                bad+=int(q['illegal'] is True);unresolved+=int(not q['verified']);pairs.append(dict(parent_id=r['parent_id'],group=r['group'],pair=[i,j],classification=c,details=q))
    _,_,b=np.linalg.svd(v-v.mean(0),full_matrices=False);res=(v-v.mean(0))@b[-1]
    allrows.append(dict(parent_id=r['parent_id'],group=r['group'],K=len(f),illegal_C_pairs=bad,unresolved_C_pairs=unresolved,max_planar_residual_model=float(np.max(np.abs(res))),classifications=cnt))
out=dict(schema='bench_fixed_C_finite_pair_audit_v1',method='Frozen dual finite-triangle detector; exact AABB rejection; C only, no Y scan',rows=allrows,problem_pairs=pairs,
    counts=dict(parents=len(allrows),illegal_C_parents=sum(r['illegal_C_pairs']>0 for r in allrows),illegal_C_pairs=sum(r['illegal_C_pairs'] for r in allrows),unresolved_C_pairs=sum(r['unresolved_C_pairs'] for r in allrows)))
p=OUT/'condition_geometry_audit.json'
if p.exists():raise FileExistsError(str(p))
p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out['counts']));print([(r['parent_id'][:8],r['illegal_C_pairs'],r['unresolved_C_pairs']) for r in allrows if r['illegal_C_pairs'] or r['unresolved_C_pairs']])
