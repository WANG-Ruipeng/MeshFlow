from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from r2.pilots.unified_rescue_post_v1 import run as R
C=R.C;root=R.ROOT;reg=C.read(root/'registration.json');old=root/'engineering_attempt_01'
test=C.read(old/'actual_array_regression.json');assert test['status']=='PASS' and test['old_completed_arrays_byte_identical']==113
current={};prior={};changes=[]
for path,digest in reg['code'].items():
    p=C.resolve(path);now=C.sha(p);current[str(p)]=now
    if now!=digest:
        snapshot=old/'source'/p.name
        assert C.sha(snapshot)==digest,(p,snapshot)
        prior[str(snapshot)]=digest;changes.append(dict(path=str(p),old_sha256=digest,new_sha256=now,snapshot=str(snapshot)))
    else:prior[str(p)]=digest
current.update({str(p):C.sha(p) for p in Path(R.__file__).parent.glob('*.py')})
C.write(root/'execution_revision_02.json',dict(revision='CORRECTED_GEO_FP32_EVENTS_V2',utc=C.utc(),
    original_registration_path=str(root/'registration.json'),original_registration_sha256=C.sha(root/'registration.json'),
    scientific_thresholds_unchanged=True,scope='Only exhausted FP32 alpha retreat fallback; completed113 retained, remaining27',
    changes=changes,code=current,prior_code_snapshots=prior,retained_receipts=C.read(old/'recovery_receipt.json')['completed_receipts'],
    numerical_regression_sha256=C.sha(old/'actual_array_regression.json'),model_calls=0,new_geometry_in_regression=0,new_scores_in_regression=0))
R.verify_code(reg)
print(str(root/'execution_revision_02.json'));print(C.sha(root/'execution_revision_02.json'))
