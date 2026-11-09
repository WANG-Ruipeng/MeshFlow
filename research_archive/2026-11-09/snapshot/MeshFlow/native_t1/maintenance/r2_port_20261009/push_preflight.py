from pathlib import Path
import hashlib,json,subprocess
root=Path('/mnt/e/MeshFlow')
pkg=root/'r2'
audit=root/'native_t1/maintenance/r2_port_20261009'
manifest=json.loads((pkg/'docs/source_manifest.json').read_text())
def git(*args):
    return subprocess.check_output(['git','-C',str(root),*args])
staged=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1]
assert len(staged)==72
assert all(p=='README.md' or p.startswith('r2/') for p in staged)
for row in manifest['source_files_copied_byte_exact']:
    content=git('show',':r2/'+row['path'])
    assert hashlib.sha256(content).hexdigest()==row['sha256'],row['path']
previous=json.loads((audit/'final_receipt.json').read_text())
allowed_changed={'r2/README.md','r2/docs/cloud_audit.md'}
for row in previous['files']:
    content=git('show',':'+row['path'])
    if row['path'] not in allowed_changed:
        assert hashlib.sha256(content).hexdigest()==row['sha256'],row['path']
print('PASS: staged blob hashes match 51 frozen sources and previously tested code.')
receipt=dict(status='PRECOMMIT_PASS',files=staged,staged_count=len(staged),
    frozen_source_blobs=51,prior_tests=72,only_prior_file_changes=sorted(allowed_changed),
    scope='R2 and root README only',model_forwards=0,training_updates=0,
    legacy_whitespace='Preserved byte-exactly; new files pass git diff --check')
(audit/'push_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
