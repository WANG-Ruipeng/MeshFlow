"""Archive actual source and compact evidence; never package large weights."""
from pathlib import Path
import csv,json,hashlib,shutil,difflib,zipfile,datetime
REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'experiments/r2_seam_slot_c200_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def resolve(s):
 p=Path(s)
 if p.exists():return p
 if len(s)>2 and s[1]==':':return Path('/mnt')/s[0].lower()/s[3:].replace('\\','/')
 raise FileNotFoundError(s)
def main():
 assert read(ROOT/'evaluation/closeout_summary.json')['actual_reviewed_cells']==64
 audit=sorted((ROOT/'audits').glob('execution_final_*.json'))[-1]
 assert read(audit)['status']=='PASS'
 assert (ROOT/'REPORT.md').exists()
 weights=[]
 for arm in ('FULL4_C200','SEAM_SLOT_C200'):
  st=read(ROOT/'training'/arm/'status.json')
  for k in ('checkpoint','inference_export','occupancy_export'):
   # Training receipt names are enumerated below as well.
   v=st.get(k)
   if isinstance(v,dict) and 'path' in v:weights.append({'kind':k,'arm':arm,**v})
  # Full checkpoints have a separate receipt, locate it without loading tensors.
  cp=ROOT/'checkpoints'/(arm+'_step200.pt')
  if not any(str(cp)==w.get('path') for w in weights):
   weights.append(dict(kind='full_recovery_checkpoint',arm=arm,path=str(cp),bytes=cp.stat().st_size,sha256=sha(cp)))
  for p in sorted((ROOT/'inference').glob(arm+'*.json')):
   v=read(p)
   if not any(v.get('path')==w.get('path') for w in weights):weights.append(dict(kind='inference_export',**v))
 write(ROOT/'WEIGHTS.json',dict(status='COMPLETE',weights=weights,large_weights_in_review_zip=False))
 freeze=read(ROOT/'execution_freeze.json');code=dict(freeze['code_hashes'])
 for p in sorted((REPO/'r2/pilots/seam_slot_c200_v1').iterdir()):
  if p.is_file() and p.suffix in ('.py','.ps1','.md'):code.setdefault(str(p),sha(p))
 snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False);records=[]
 for source,h in sorted(code.items()):
  p=resolve(source);assert sha(p)==h,(p,'source changed')
  parts=p.parts
  relative=Path(*parts[3:]) if str(p).startswith('/mnt/e/') else Path(*parts[1:])
  dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  records.append(dict(original_path=str(p),snapshot_path=str(dest.relative_to(ROOT)),sha256=h,bytes=p.stat().st_size,
   scope='FROZEN_EXECUTED_SOURCE' if source in freeze['code_hashes'] else 'DERIVED_REPORT_CODE'))
 with (ROOT/'SOURCE_HASHES.csv').open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 diff=['# Additive source snapshot relative to empty files, not a Git worktree diff.\n',
       '# Public source/scorers remain unchanged; their copies are in source_snapshot.\n']
 for folder in ('r2/prototypes/seam_slot_c200_v1','r2/pilots/seam_slot_c200_v1'):
  for p in sorted((REPO/folder).iterdir()):
   if p.is_file() and p.suffix in ('.py','.ps1'):
    diff.extend(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(True),fromfile='/dev/null',tofile=str(p.relative_to(REPO)),n=3))
 (ROOT/'code.diff').write_text(''.join(diff),encoding='utf-8')
 request=Path('/mnt/c/Users/warpwang/.codex/attachments/0ad09f2a-9cc0-4988-8b28-8ee2315d208a/Pasted text.txt')
 shutil.copyfile(request,ROOT/'USER_REQUEST.txt')
 (ROOT/'REVIEW_README.md').write_text('''# Compact review evidence

Read REPORT.md first. All two-arm training, 32 RAW + 32 POST, 64 scores and 64 AI_PROXY image reviews were actually completed.

Included: frozen protocol/manifests, all aggregate and per-output/per-parent/per-noise tables, all saved scorer JSON/CSV records, 16 complete paired gallery pages, training curves and logs, independent audits, actual executed source copies and SHA256, additive code.diff, final weight identities.

Excluded to keep this archive below 10 MB: multi-GB weights, training OT cache, raw/POST mesh arrays, trajectories and per-query vector NPZ. They remain locally preserved at the manifest paths. Thus this zip permits recalculating table aggregates and inspecting the evidence/source; it is not a standalone full geometry rerun package. WEIGHTS.json names the actual locally retained weights. No mesh or weight was deleted during packaging.

Gallery reviewer labels are AI_PROXY, nonblind. A visible contact is not a legal indexed edge certificate. Candidate-only seam-to-bulk fields are NA for FULL.

Recovery entrance: r2/pilots/seam_slot_c200_v1/resume.ps1 (also in source_snapshot). Default invocation only reports status. It does not start additional experiments.
''',encoding='utf-8')
 files=[]
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ('.md','.json','.txt','.csv','.diff','.jsonl') and p.name not in ('completion.json','PACKAGE_MANIFEST.json','package_receipt.json'):files.append(p)
 for directory in ('audits','statistics','source_snapshot'):
  files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file())
 for p in (ROOT/'evaluation').rglob('*'):
  if not p.is_file():continue
  relative=p.relative_to(ROOT/'evaluation')
  if len(relative.parts)==1 and p.suffix in ('.json','.csv'):files.append(p)
  elif relative.parts[0]=='gallery' and p.suffix in ('.png','.html','.json','.csv'):files.append(p)
  elif relative.parts[0]=='scores' and p.suffix in ('.json','.csv'):files.append(p)
 for p in (ROOT/'training').rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl'):files.append(p)
 files=sorted(set(files))
 manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 write(ROOT/'PACKAGE_MANIFEST.json',dict(files=manifest,source_count=len(records),large_models_excluded=True))
 files.append(ROOT/'PACKAGE_MANIFEST.json')
 dest=ROOT/'SEAM_SLOT_C200_REVIEW.zip'
 with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in files:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  assert len(z.namelist())==len(files)
  for r in manifest:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
 assert dest.stat().st_size<10_000_000,('Archive exceeds 10 MB',dest.stat().st_size)
 receipt=dict(status='PASS',archive=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),files=len(files),
  executed_source_copies=len(records),all_zip_member_hashes_verified=True)
 write(ROOT/'package_receipt.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
