from pathlib import Path
import importlib.util,json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from r2.pilots.unified_rescue_post_v1 import floor,test_geometry,run as R
out=R.ROOT;old=out/'engineering_attempt_01/source/floor.py'
spec=importlib.util.spec_from_file_location('old_floor',old);O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
reg=R.C.read(out/'registration.json');job=next(j for j in reg['jobs'] if j['panel']=='CHAIR_SEAT_S2000' and j['q']['request_id']=='73b369b9_n1')
base=Path(job['base_dir']);raw=R.C.arrays(base/'raw/raw.npz');p=R.C.arrays(base/'post/proposal.npz')['official_unconstrained_proposal']
args=(raw['raw'],p,raw['raw_anchor_ids'],raw['C_vertices'],job['q']['K'],R.guard_geo.RADIUS)
try:O.protect(*args)
except ArithmeticError as e:old_error=repr(e)
else:raise AssertionError('Original failure not reproduced')
guarded,protected,records,pre=floor.protect(*args)
fallback=[r for r in records if r['quantization_event_fallback']]
assert fallback and np.all(floor.lengths(guarded)[[r['raw_slot'] for r in records if r['floor_triggered']]].min(1)>=1.25*R.guard_geo.RADIUS)
old_receipts=R.C.read(out/'engineering_attempt_01/recovery_receipt.json')['completed_receipts'];verified=0
# Only replay inexpensive floor algebra against saved guarded PRE; zero GEO,
# scoring/model calls. All old valid arrays must remain exactly unchanged.
for j in reg['jobs']:
    c=R.cell(j)
    if not (c/'complete.json').exists():continue
    b=Path(j['base_dir']);r=R.C.arrays(b/'raw/raw.npz');d=R.C.arrays(b/'post/proposal.npz')['official_unconstrained_proposal']
    now=floor.protect(r['raw'],d,r['raw_anchor_ids'],r['C_vertices'],j['q']['K'],R.guard_geo.RADIUS)[0]
    prior=R.C.arrays(c/'guarded_pre_geo.npz')['output']
    assert now.tobytes()==prior.tobytes(),(j['panel'],j['q']['request_id'])
    assert R.C.sha(c/'complete.json')==old_receipts[str(c/'complete.json')]
    verified+=1
result=dict(status='PASS',old_failure_reproduced=old_error,request=job['q']['request_id'],panel=job['panel'],
    inputs={str(base/'raw/raw.npz'):R.C.sha(base/'raw/raw.npz'),str(base/'post/proposal.npz'):R.C.sha(base/'post/proposal.npz')},
    R=R.guard_geo.RADIUS,floor=1.25*R.guard_geo.RADIUS,fallback_faces=fallback,old_completed_arrays_byte_identical=verified,
    original_synthetic=test_geometry.run(),model_calls=0,new_geometry_decodes=0,new_scores=0)
R.C.write(out/'engineering_attempt_01/actual_array_regression.json',result)
print(json.dumps(result,indent=2))
