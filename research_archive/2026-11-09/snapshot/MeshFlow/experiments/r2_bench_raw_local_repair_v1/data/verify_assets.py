"""Small independent read-only schema, mapping, and attribution checks."""
import sys, json, csv, hashlib
from pathlib import Path
sys.path.insert(0,'/mnt/e/MeshFlow')
import numpy as np
from r2.pilots.bench_raw_local_repair_v1 import assets as A
from r2.pilots.bench_raw_local_repair_v1 import common as C

m=A.load_manifest(True)
assert len(m['tasks'])==12 and len({t['parent_id'] for t in m['tasks']})==6
assert all(x['status']=='AVAILABLE' for t in m['tasks'] for x in t['assets'].values())
checks=[]
for t in m['tasks']:
    v=A.load_request(t['request_id'],m,False);raw=v['arrays']['RAW']['raw'];n,k=t['N'],t['K']
    assert all(not x.flags.writeable for a in v['arrays'].values() for x in a.values())
    for stage in ('OFFICIAL_BASE','SHAPE_BASE','RELDEF_BASE','GEO_ONLY_BASE'):
        a=v['arrays'][stage];sf=a['source_face_ids'];sc=a['source_corner_ids']
        assert np.array_equal(a['vertices'][a['faces']],a['output'])
        assert np.array_equal(v['maps'][stage+'__output_to_raw_flat_corner'],sf[:,None]*3+sc)
        assert np.array_equal(a['output'][:k],raw[sf[:k,None],sc[:k]])
        iv=a.get('input_corner_vertex_ids',a.get('corner_vertex_ids'))
        assert np.array_equal(iv[sf[:,None],sc],a['faces'])
    checks.append(dict(request_id=t['request_id'],status='PASS',four_base_corner_maps=True,C_exact_under_source_permutation=True,returned_arrays_read_only=True))
with (A.OUT/'data/conflict_attribution.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==587 and len({(r['request_id'],r['kind'],r['raw_slot_i'],r['raw_slot_j']) for r in rows})==587
for r in rows:
    a=r['shape_i_retained']=='True' and r['shape_j_retained']=='True'
    b=r['reldef_i_retained']=='True' and r['reldef_j_retained']=='True'
    expected='COMMON_RETAINED' if a and b else ('NEWLY_RETAINED' if b else ('NO_LONGER_RETAINED' if a else 'UNKNOWN'))
    assert r['retention_attribution']==expected
    if not a:assert r['shape_classification']=='NA_FACE_NOT_RETAINED'
    if not b:assert r['reldef_classification']=='NA_FACE_NOT_RETAINED'
result=dict(status='PASS',requests=12,base_corner_map_checks=48,unique_pair_rows=587,checks=checks,model_calls=0,scoring_calls=0,geometry_calls=0)
C.write(A.OUT/'data/verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
