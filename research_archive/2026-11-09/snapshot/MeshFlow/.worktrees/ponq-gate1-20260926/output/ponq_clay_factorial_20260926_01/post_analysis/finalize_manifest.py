import sys,json,hashlib,subprocess,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((R/'post_analysis/PRE_POSTANALYSIS_MANIFEST.json').read_text())
bad=[e['path'] for e in old['files'] if not (R/e['path']).is_file() or sha(R/e['path'])!=e['sha256']]
assert not bad,bad
baseline=[]
for p in [Path('/mnt/e/MeshFlow'),R.parents[1]]:
    head=subprocess.check_output(['git','-c','core.autocrlf=true','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','-c','core.autocrlf=true','-C',str(p),'status','--short','--untracked-files=no'],text=True).strip()
    assert head=='8cc74da013d3d2feec1853cccbb1f9ac37814b95' and not status,(str(p),head,status)
    baseline.append(dict(path=str(p),head=head,tracked_status=status))
proof=dict(passed=True,checked_original_artifacts=len(old['files']),original_artifacts_changed=bad,baseline=baseline,
           note='Post-analysis only added files. Frozen run sources, inputs, meshes, scores and all original artifacts retain their hashes.',
           created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/'audit/FINAL_INTEGRITY.json').write_text(json.dumps(proof,indent=2)+'\n')
files=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(R.rglob('*')) if p.is_file() and p!=R/'MANIFEST.json']
(R/'MANIFEST.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
for e in files:assert sha(R/e['path'])==e['sha256']
print(json.dumps(dict(passed=True,original_artifacts_verified=len(old['files']),manifest_files=len(files),total_bytes=sum(e['bytes'] for e in files),baseline_clean=True),indent=2))