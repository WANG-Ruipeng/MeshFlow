from pathlib import Path
import json,shutil,hashlib
R=Path('/mnt/e/MeshFlow');out=R/'experiments/r2_unified_rescue_post_v1';dest=out/'engineering_attempt_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if dest.exists():raise FileExistsError(dest)
dest.mkdir()
code=R/'r2/pilots/unified_rescue_post_v1'
shutil.copytree(code,dest/'source')
for name in ('registration.json','dispatch.json','synthetic_unit_tests.json'):
    shutil.copy2(out/name,dest/name)
cells=list((out/'cells').glob('*/*/complete.json'))
failed=out/'cells/CHAIR_SEAT_S2000/73b369b9_n1'
assert len(cells)==113 and (failed/'failure.json').exists()
assert {p.name for p in failed.iterdir()}=={'started.json','failure.json'}
before={str(p):sha(p) for p in cells}
target=dest/'failed_cell'
assert failed.resolve().is_relative_to(out.resolve()) and target.resolve().is_relative_to(out.resolve())
shutil.move(str(failed),str(target))
receipt=dict(status='ARCHIVED_NUMERICAL_WRITEBACK_FAILURE',completed_valid_cells=113,failed_before_GEO=1,unstarted=26,
    failed_panel='CHAIR_SEAT_S2000',failed_request_id='73b369b9_n1',old_registration_sha256=sha(out/'registration.json'),
    completed_receipts=before,source_snapshot={str(p):sha(p) for p in (dest/'source').glob('*.py')},
    model_calls=0,D0_calls=0,new_noise=0,new_scores_in_failed_cell=0,
    unchanged_scientific_definition='Exact quadratic safe prefix and same R/floors/selection; improve FP32 endpoint representability only')
(dest/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('completed_receipts','source_snapshot')},indent=2))
