"""Verify archived review-package bytes, without modifying originals."""
from pathlib import Path
import hashlib, json, zipfile, time
R=Path('/mnt/e/MeshFlow');O=R/'maintenance/storage_cleanup_20261025'
E=R/'experiments/r2_single_step_denoiser_distillation_v1'
P=E/'review_package';Z=E/'R2_SINGLE_STEP_DENOISER_DISTILLATION_REVIEW.zip'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def w(p):return 'E:/'+str(p).split('/mnt/e/',1)[1]
done=json.loads((E/'completion.json').read_text());actual=sha(Z)
assert actual==done['archive']['sha256'], 'Archive SHA differs from original completion'
members=[]
with zipfile.ZipFile(Z) as z:
 names=set(z.namelist())
 for p in sorted(P.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(P).as_posix();h=sha(p)
  assert rel in names, f'Not archived: {rel}'
  assert hashlib.sha256(z.read(rel)).hexdigest()==h, f'Archive mismatch: {rel}'
  members.append({'path':w(p),'bytes':p.stat().st_size,'sha256':h,'retained_archive':w(Z),'archive_member':rel})
 assert set(x['archive_member'] for x in members)==names
keep=[m for m in members if m['archive_member']=='PACKAGE_MANIFEST.csv']
candidates=[m for m in members if m not in keep]
result={'status':'READ_ONLY_VERIFIED_EXACT_ARCHIVED_COPY','archive':w(Z),'archive_sha256':actual,'archive_bytes':Z.stat().st_size,'retained_original_root':w(E),'verified_archive_members':len(members),'keep_expanded_metadata':keep,'files':candidates,'count':len(candidates),'bytes':sum(x['bytes'] for x in candidates),'safe_scope':'Only individual listed expanded review_package files. Keep PACKAGE_MANIFEST.csv because historical finalizer verify_finished reads it. Original experiment tables, RAW/POST, source snapshots, code, completion and ZIP remain. No deletion performed. Local expanded-package browsing would retire; ZIP stays fully reproducible byte-for-byte.'}
(O/'dataset_archived_package_whitelist.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
