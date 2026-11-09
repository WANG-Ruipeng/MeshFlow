import json,numpy as np
from pathlib import Path
from experiments.a_radial.io_utils import array_hash
run=Path('output/a_taskaligned_20260924_01');old=Path('output/layout_value_20260924_01')
teachers={x['uid']:x for x in json.loads((run/'TEACHERS.json').read_text())['rows']}
rows=[x for x in json.loads((run/'results.json').read_text())['rows'] if x['model']=='teacher_reference']
results=[]
for row in rows:
 t=teachers[row['uid']];srcrow=json.loads((old/'attempts'/row['uid']/t['chosen_arm']/'result.json').read_text())
 with np.load(old/srcrow['mesh_file'],allow_pickle=False) as z: sv=z['vertices'];sf=z['faces'];su=z['u']
 with np.load(run/row['mesh_file'],allow_pickle=False) as z:v=z['vertices'];f=z['faces'];u=z['u']
 results.append({'uid':row['uid'],'chosen_arm':t['chosen_arm'],'teacher_metadata_hash':t['teacher_mesh_hash'],
  'old_source_mesh_actual_hash':array_hash(sv,sf),'new_ledger_hash':row['mesh_hash'],'new_actual_hash':array_hash(v,f),
  'source_faces_dtype':str(sf.dtype),'new_faces_dtype':str(f.dtype),'vertices_exact':bool(np.array_equal(v,sv)),
  'faces_values_exact':bool(np.array_equal(f,sf)),'directions_exact':bool(np.array_equal(u,su)),
  'new_v_old_f_hash_equals_teacher':array_hash(v,sf)==t['teacher_mesh_hash'],
  'metadata_matches_old_actual':t['teacher_mesh_hash']==array_hash(sv,sf),
  'new_row_matches_new_actual':row['mesh_hash']==array_hash(v,f)})
print(json.dumps({'rows':len(results),'hash_diff_count':sum(x['new_ledger_hash']!=x['teacher_metadata_hash'] for x in results),
 'all_vertices_exact':all(x['vertices_exact'] for x in results),'all_face_values_exact':all(x['faces_values_exact'] for x in results),
 'all_directions_exact':all(x['directions_exact'] for x in results),
 'all_source_sha_explained':all(x['new_v_old_f_hash_equals_teacher'] and x['metadata_matches_old_actual'] and x['new_row_matches_new_actual'] for x in results),
 'rows_detail':results},indent=2))
