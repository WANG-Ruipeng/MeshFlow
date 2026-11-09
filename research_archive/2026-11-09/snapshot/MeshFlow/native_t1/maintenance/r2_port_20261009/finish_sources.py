from pathlib import Path
import json, hashlib, shutil
root=Path('/mnt/e/MeshFlow/r2')
p=root/'src/meshflow_control/r2_checkpoints.py'
s=p.read_text()
s=s.replace('    import torch\n    value, stage, kind, sha = read(path, expected_sha256)', '    from . import runtime\n    import torch\n    value, stage, kind, sha = read(path, expected_sha256)')
p.write_text(s)
p=root/'tests/test_training_data.py'
if p.exists(): p.unlink()
p=root/'docs/source_manifest.json'
data=json.loads(p.read_text())
data['omitted_tests']=[dict(path='tests/test_training_data.py',reason='Generic ABC trainer recovery tests depend on the retired trainer, which is outside the R2 dependency closure. R2 data/train/confirmation tests are retained.')]
data['source_files_copied_byte_exact']=[r for r in data['source_files_copied_byte_exact'] if r['path']!='tests/test_training_data.py']
for name in ['model_source_map.json','data_training_source_map.json']:
    source=Path('/mnt/e/MeshFlow-Control/docs')/name
    target=root/'docs'/name
    shutil.copyfile(source,target)
    data['source_files_copied_byte_exact'].append(dict(path=str(target.relative_to(root)),source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),bytes=source.stat().st_size))
p.write_text(json.dumps(data,indent=2)+'\n')
print('Source provenance and test boundary updated.')
