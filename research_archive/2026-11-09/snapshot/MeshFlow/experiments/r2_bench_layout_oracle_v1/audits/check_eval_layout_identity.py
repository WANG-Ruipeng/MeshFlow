"""Read actual inference C and independently verify source-layout identities."""
from pathlib import Path
import sys, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from r2.pilots.bench_layout_oracle_v1 import evaluation as E

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tri_keys(a):
    # Exact float32 coordinate multiset per unordered triangle; no tolerance weld.
    rows=[]
    for tri in np.asarray(a,np.float32).reshape(-1,3,3):
        pts=sorted(tuple(float(x) for x in v) for v in tri)
        rows.append(tuple(pts))
    return sorted(rows)

def centre(a):
    a=np.asarray(a,np.float64).reshape(-1,3,3)
    area=np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)/2
    assert area.sum()>0
    return np.sum(a.mean(1)*area[:,None],axis=0)/area.sum()

out=ROOT/'experiments/r2_bench_layout_oracle_v1'
records={r['condition_id']:r for r in json.loads((out/'data/layout_records.json').read_text())['records']}
results=[]
for q in E.requests():
    core,source=E.engine().source_assets(q)
    actual=np.asarray(core['C_vertices'])[np.asarray(core['C_faces_model'],np.int64)]
    record=records[q['condition_id']]
    case=np.load(record['case_path'],allow_pickle=False)
    expected=case['full_target'][case['known_ids']]
    layout=np.load(record['layout_npz'],allow_pickle=False)
    assert tri_keys(actual)==tri_keys(expected),q['request_id']
    ac=centre(actual)
    np.testing.assert_allclose(ac,layout['center_C'],rtol=0,atol=1e-6)
    np.testing.assert_allclose(ac,centre(expected),rtol=0,atol=1e-12)
    boxes=np.asarray(layout['absolute_bbox'],np.float64)
    tokens=np.zeros((2,8),np.float32)
    L=float(layout['L'])
    tokens[:,:3]=((boxes[:,0]+boxes[:,1])/2-ac)/L
    tokens[:,3:6]=(boxes[:,1]-boxes[:,0])/L
    tokens[0,6]=1;tokens[1,7]=1
    tokens[~layout['valid']]=0
    np.testing.assert_allclose(tokens,layout['tokens'],rtol=0,atol=1e-7)
    results.append(dict(request_id=q['request_id'],condition_id=q['condition_id'],
        case_sha256=sha(record['case_path']),layout_npz_sha256=sha(record['layout_npz']),
        inference_C_triangle_multiset_exact=True,inference_C_order_exact=bool(np.array_equal(actual,expected)),
        C_face_count=len(actual),C_area_centroid_max_delta=float(np.max(np.abs(ac-layout['center_C']))),
        token_max_delta=float(np.max(np.abs(tokens-layout['tokens']))),
        source_asset_keys=list(source),model_calls=0))
report=dict(status='PASS',requests=len(results),records=results,
    model_forwards=0,model_constructions=0,CUDA_calls=0,OT_calls=0,
    torch_imported='torch' in sys.modules,
    audited_interface='evaluation.engine().source_assets(q)',
    tolerance=dict(centroid_stored_float32_model_units=1e-6,centroid_double=1e-12,tokens=1e-7),
    exact_triangle_multiset_scope='Coordinates float32, vertex order and triangle order ignored; no approximate geometry substitution',
    script_sha256=sha(__file__))
dest=out/'audits/eval_layout_identity.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','requests','model_forwards','torch_imported')}))
