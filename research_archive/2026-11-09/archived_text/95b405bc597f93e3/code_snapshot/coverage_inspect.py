import json
import numpy as np
from pathlib import Path
b=Path('/mnt/e/MeshFlow/experiments/r2_user_data_scaling_32_67_v1')
m=json.loads((b/'data/train_manifest.json').read_text())
p=json.loads((b/'data/panel_manifest.json').read_text())
print('PANELKEYS',list(p))
print('TRAINPARENT',{k:v for k,v in next(iter(m['parents'].values())).items() if k != 'lineage'})
print('TASK',{k:v for k,v in next(iter(m['tasks'].values())).items() if k not in ['source_face_ids','free_source_ids','original_source_face_ids','original_vertex_ids','region_source_face_ids']})
print('PANEL NONCONDITION',{k:v for k,v in p.items() if k not in ['conditions','parents','requests']})
print('PARENTS',str(p.get('parents'))[:800])
c=Path(p['conditions'][0]['condition_source']); z=np.load(c,allow_pickle=False)
print('CONDITION KEYS',[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
ri=json.loads((b/'evaluation/raw_index.json').read_text())['rows']
print('RAWKEYS',list(ri[-1])); print('RAW ROW',{k:v for k,v in ri[-1].items() if k in ['model','arm','generator','raw_path','path','request_id','parent_id','output_dir']})
