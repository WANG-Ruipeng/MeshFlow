import json
from pathlib import Path
r=Path('native_t1/runs/jedge5_blackbox_20261002_113600')
a=json.loads((r/'geometry/endpoint_results.json').read_text())
for q in a['contexts']:
 if q['parent']==0 and q['patch'] in (0,4): print({k:v for k,v in q.items() if k not in ['source_C_vertex_links','boundary_components']})
print('noise',a['noise'])
for q in a['rows']:
 if q['model']=='J' and q['patch']==4:print({k:q[k] for k in ['seed','nfe','symmetric_RMS_bbox_pct','forward_RMS_bbox_pct','backward_RMS_bbox_pct','free_area_over_GT_free_area','top3_symmetric_mse_share','top10_symmetric_mse_share','faces_for_90pct_mse','top10_free_face_ids']})
