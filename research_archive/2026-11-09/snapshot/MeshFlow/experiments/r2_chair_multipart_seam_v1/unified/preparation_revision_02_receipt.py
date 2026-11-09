"""One-time metadata receipt for approved Step1 execution-amendment binding."""
from pathlib import Path
import hashlib,json,difflib

root=Path('/mnt/e/MeshFlow')
out=root/'experiments/r2_chair_multipart_seam_v1/unified'
old=out/'preparation_revision_01'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a,b=read(old/'registration.json'),read(out/'registration.json')
assert a['parameters']==b['parameters'] and a['stages']==b['stages']
assert len(a['jobs'])==len(b['jobs'])==44
assert all(x['q']==y['q'] and x['base_dir']==y['base_dir'] for x,y in zip(a['jobs'],b['jobs']))
assert not (out/'cells').exists() and not (out/'input_registration.json').exists()
source=root/'r2/pilots/chair_multipart_seam_v1/unified_post.py'
diff=''.join(difflib.unified_diff((old/'unified_post.py').read_text().splitlines(True),source.read_text().splitlines(True),fromfile='preparation_revision_01/unified_post.py',tofile='current/unified_post.py'))
(out/'preparation_revision_02_code.diff').write_text(diff)
result=dict(status='PASS',reason='Approved upstream FP32 retreat event fix; explicit identity migration before any Step2 geometry',
  original_preparation=str(old),old_registration_sha256=sha(old/'registration.json'),new_registration_sha256=sha(out/'registration.json'),
  original_source_registration_sha256=b['source_pipeline_registration_sha256'],
  source_execution_revision=b['source_execution_revision'],source_execution_registration_sha256=b['source_execution_registration_sha256'],
  same_scientific_parameters=True,same44_input_identities=True,geometry_already_executed=0,
  old_adapter_sha256=sha(old/'unified_post.py'),new_adapter_sha256=sha(source),diff_sha256=sha(out/'preparation_revision_02_code.diff'),
  cpu_interface_check_sha256=sha(out/'cpu_interface_check.json'),model_calls=0,geometry_calls=0,scoring_calls=0)
with (out/'preparation_revision_02_receipt.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
