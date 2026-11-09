from pathlib import Path
import ast,hashlib,json,re,shutil,zipfile,xml.etree.ElementTree as ET
root=Path('/mnt/e/MeshFlow')
pkg=root/'r2'
audit=root/'native_t1/maintenance/r2_port_20261009'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
validation=json.loads((audit/'real_asset_validation.json').read_text())
assert validation['status']=='PASS'
shutil.copyfile(audit/'real_asset_validation.json',pkg/'docs/validation_receipt.json')
suite=ET.parse(audit/'pytest_final.xml').getroot()[0]
assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0
files=set((audit/'git_visible_files.txt').read_text().splitlines())
files.add('r2/docs/validation_receipt.json')
source=json.loads((pkg/'docs/source_manifest.json').read_text())
for row in source['source_files_copied_byte_exact']:
    p=pkg/row['path']
    assert sha(p)==row['sha256']==sha(Path(row['source']))
    assert str(p.relative_to(root)) in files, 'Source ignored by Git: '+row['path']
checked=[]
for rel in sorted(files):
    p=root/rel
    if p.suffix=='.py': ast.parse(p.read_text(encoding="utf-8-sig"),filename=rel)
    assert p.suffix.lower() not in ('.pt','.pth','.ckpt','.npz','.npy','.zip','.whl')
    checked.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p)))
for p in pkg.rglob('*.md'):
    if 'build' in p.parts: continue
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (p.parent/link.split('#')[0]).exists(),(p,link)
wheel=next((audit/'wheelcheck').glob('*.whl'))
with zipfile.ZipFile(wheel) as z:
    package_files=[n for n in z.namelist() if n.startswith('meshflow_control/') and n.endswith('.py')]
    for name in package_files:
        assert z.read(name)==(pkg/'src'/name).read_bytes(),name
    assert len(package_files)==len(list((pkg/'src/meshflow_control').rglob('*.py')))
result=dict(status='PASS',verified_remote_master='8961b7a41fc153b5b45f0c8903370c6bfd1cf04f',
    tests=int(suite.attrib['tests']),test_failures=0,byte_exact_sources=len(source['source_files_copied_byte_exact']),
    source_package_files=len(package_files),wheel_sha256=sha(wheel),review_file_count=len(checked),
    review_bytes=sum(r['bytes'] for r in checked),files=checked,real_assets='real_asset_validation.json',
    real_model_forwards=0,real_training_updates=0,new_samples=0,new_conditions=0,new_OT_calls=0,
    git_commit=False,git_push=False,checkpoint_selection_changed=False)
(audit/'final_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
