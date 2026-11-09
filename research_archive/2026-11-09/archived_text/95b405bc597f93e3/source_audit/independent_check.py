from pathlib import Path
import csv,json,hashlib,time
import numpy as np
from trimesh.triangles import closest_point
ROOT=Path('/mnt/e/MeshFlow'); OUT=ROOT/'experiments/r2_seat_failure_diagnosis_v1/source_audit'
m=json.loads((ROOT/'experiments/r2_user_data_scaling_32_67_v1/data/train_manifest.json').read_text())
g=list(csv.DictReader((OUT/'per_duplicate_group.csv').open(encoding='utf-8-sig')))
c={r['condition_id']:r for r in csv.DictReader((OUT/'per_condition.csv').open(encoding='utf-8-sig'))}
L=json.loads((OUT/'rules.json').read_text())['L'];errors=[];maxerr=0.;checked=0
for row in g:
 t=m['tasks'][row['condition_id']]
 with np.load(m['parents'][t['uid']]['npz'],allow_pickle=False) as z:tri=z['model_vertices'][z['source_faces']].astype(np.float64)
 sid=json.loads(row['duplicate_source_face_ids'])[0];point=tri[sid].mean(0)
 for region,ids in [('C',t['source_face_ids']),('R',t['region_source_face_ids'])]:
  reg=tri[ids];near=closest_point(reg,np.repeat(point[None],len(reg),axis=0));d=np.linalg.norm(near-point,axis=1).min()/L
  err=abs(d-float(row[region+'_centroid_distance_over_L']));maxerr=max(maxerr,err);checked+=1
  if err>1e-10:errors.append([row['condition_id'],sid,region,err])
summary=[]
for tau in ['0.005','0.01','0.02']:
 for region in ['C','R']:
  rr=[r for r in g if r[region+'_local_duplicate_'+tau]=='True' and float(r['abs_normal_dot_'+region+'_plane'])>=0.9]
  ids=sorted({r['condition_id'] for r in rr})
  summary.append({'region':region,'tau_L':float(tau),'normal_abs_dot_min':0.9,'conditions':len(ids),'parents':len({r['parent_uid'] for r in rr}),'duplicate_groups_counted_per_condition':len(rr),'exposures':sum(int(c[x]['actual_exposures']) for x in ids),'condition_ids':ids,'interpretation':'Descriptive normal-filtered view of frozen rows; not an illegal-surface or success detector.'})
(OUT/'parallel_local_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
r={'status':'PASS' if not errors else 'FAIL','independent_library':'trimesh.triangles.closest_point','duplicate_region_distances_checked':checked,'max_abs_error_distance_over_L':maxerr,'errors':errors,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_calls':0}
(OUT/'independent_distance_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))