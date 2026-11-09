"""CPU-only complete evidence archive. No imports of model or scientific runners."""
from __future__ import annotations
import argparse,csv,difflib,hashlib,html,json,os,posixpath,re,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/r2_source_winding_export_v1'
ZIP=OUT/'R2_SOURCE_WINDING_EXPORT_REVIEW.zip'
FINAL=OUT/'packaging'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def checked(p):
 p=Path(p).absolute()
 if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('Symlink output')
 p=p.resolve()
 if p==OUT.resolve() or not p.is_relative_to(OUT.resolve()):raise ValueError('Write outside new output')
 return p
def atomic(p,fn,binary=False):
 p=checked(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=checked(p.with_name(p.name+'.tmp'))
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
  fn(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def write(p,obj):atomic(p,lambda f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False))
def textfile(p,s):atomic(p,lambda f:f.write(s))
def immutable_copy(src,dst):
 src=Path(src);dst=checked(dst)
 if dst.exists():
  if sha(src)!=sha(dst):raise RuntimeError('Existing snapshot differs: '+str(dst))
 else:atomic(dst,lambda f:f.write(src.read_bytes()),True)
def assert_hash(p,h):
 if sha(p)!=h:raise ValueError('Hash mismatch: '+str(p))


def prerequisite_check():
 required=['REPORT.md','README_RESUME.md','USER_REQUEST.md','completion.json','registration.json','execution_completion.json',
  'aggregate_completion.json','restoration_checks/completion.json','independent_audit/completion.json',
  'paired_per_parent_noise.csv','AVMR_PMR_per_output.csv','quality_inherited_per_output.csv',
  'topology_per_output.csv','topology_summary.csv','source_C_face_mapping.csv',
  'review/source_restoration_summary.csv','review/source_restoration_contract_audit.json']
 for name in required:
  if not (OUT/name).is_file():raise RuntimeError('BLOCKED_MISSING '+name)
 done=read(OUT/'execution_completion.json')
 assert done['status']=='PASS' and done['outputs']==done['source_C_restorations']==done['orientation_solves']==36
 assert done['scientific_stage_stopped'] and done['frozen_sources_unchanged']
 for key in ['model_loads','model_forwards','training_updates','new_GEO_C','new_full_quality_scores','new_images']:assert done[key]==0
 assert read(OUT/'completion.json')['status'] in ('PASS','COMPLETE')
 for name in ['independent_audit/completion.json','restoration_checks/completion.json','review/source_restoration_contract_audit.json']:
  assert read(OUT/name)['status'] in ('PASS','COMPLETE')
 audit=read(OUT/'independent_audit/completion.json')
 assert audit['audit_source_sha256']==sha(CODE/'independent_audit.py')
 assert audit['outputs']==36 and audit['actual_PLY_roundtrips']==72 and audit['stage_topology_rows']==108
 assert len(read(OUT/'restoration_checks/completion.json')['cases'])==8
 reg=read(OUT/'registration.json');assert len(reg['records'])==36 and len(reg['conditions'])==6
 assert len(rows(OUT/'AVMR_PMR_per_output.csv'))==864 and len(rows(OUT/'quality_inherited_per_output.csv'))==108
 for p,h in reg['runner_hashes'].items():assert_hash(p,h)
 assert_hash(reg['orientation_solver_path'],reg['orientation_solver_sha256'])
 for r in reg['records']:
  folder=OUT/'outputs'/r['method']/r['request_id'];q=read(folder/'receipt.json');assert q['status']=='PASS'
  for path,h in q['files'].items():assert_hash(path,h)
  for stage in ['restored_C','oriented']:
   for ext in ['npz','ply']:assert (folder/(stage+'.'+ext)).is_file()
 return {str(OUT/n):sha(OUT/n) for n in required}

def source_snapshot():
 reg=read(OUT/'registration.json');protected=dict(reg['protected']);selected={}
 review=OUT/'review/source_restoration_contract_audit.json'
 if review.exists():
  rd=read(review);assert rd['status']=='PASS';protected.update(rd.get('protected_before',{}))
 for p,h in {**protected,**reg['runner_hashes']}.items():
  if p.endswith('.py') and not any(s in p for s in ['/source_snapshot/','/scoring_code/']):
   assert_hash(p,h);selected[p]=h
 new={str(p):sha(p) for p in CODE.glob('*.py')};selected.update(new)
 for rel in ['LICENSE','r2/LICENSE','r2/THIRD_PARTY_NOTICES.md','requirements.txt']:
  p=ROOT/rel
  if p.exists():selected[str(p)]=sha(p)
 for p in [Path('/mnt/e/MeshFlow-Control/LICENSE'),Path('/mnt/e/MeshFlow-Control/requirements.txt')]:
  if p.exists():selected[str(p)]=sha(p)
 records=[]
 for name,h in sorted(selected.items()):
  p=Path(name);rel=('repository/'+p.relative_to(ROOT).as_posix()) if p.is_relative_to(ROOT) else ('external/'+p.relative_to('/mnt/e').as_posix())
  target=OUT/'source_snapshot'/rel;immutable_copy(p,target);assert_hash(target,h)
  records.append(dict(source_path=name,snapshot_path=target.relative_to(OUT).as_posix(),sha256=h,bytes=p.stat().st_size,new_this_experiment=name in new,registered_before_execution=name in protected or name in reg['runner_hashes']))
 textfile(OUT/'code.diff',''.join(line for name in sorted(new) for line in difflib.unified_diff([],Path(name).read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+Path(name).relative_to(ROOT).as_posix())))
 textfile(OUT/'old_source.diff','')
 write(OUT/'source_hashes.json',dict(schema='actual_sources_v1',sources=records,
  old_source_diff='empty; all copied old sources match frozen protected hashes',new_diff='All new .py files including audit and assembler are additions from /dev/null.'))
 return records

def baseline_copies():
 reg=read(OUT/'registration.json');copied=[];mesh_rows=[];seen=set()
 def copy(p,target,expected=None):
  p=Path(p);target=checked(target)
  if expected:assert_hash(p,expected)
  immutable_copy(p,target)
  if str(target) not in seen:
   copied.append(dict(source_path=str(p),archive_path=target.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size));seen.add(str(target))
  return target.relative_to(OUT).as_posix()
 for r in reg['records']:
  folder=OUT/'baseline_MODEL_POST'/r['method']/r['request_id']
  npz=copy(r['post_path'],folder/'model_post.npz',r['post_sha256']);ply=copy(r['post_ply'],folder/'model_post.ply')
  copy(r['diagnostic_path'],folder/'diagnostic.json')
  cond=copy(r['condition_source'],OUT/'source_conditions'/(r['condition_id']+'.npz'),r['condition_source_sha256'])
  parent=copy(r['parent_prepared_path'],OUT/'source_parents'/(r['parent_id']+'.npz'),r['parent_prepared_sha256'])
  for stage in ['MODEL_POST','RESTORED_C_ONLY','ORIENTED']:
   name={'RESTORED_C_ONLY':'restored_C','ORIENTED':'oriented'}.get(stage)
   row=dict(method=r['method'],request_id=r['request_id'],parent_id=r['parent_id'],condition_id=r['condition_id'],candidate=r['candidate'],stage=stage,
     NPZ_path=npz if stage=='MODEL_POST' else f"outputs/{r['method']}/{r['request_id']}/{name}.npz",
     PLY_path=ply if stage=='MODEL_POST' else f"outputs/{r['method']}/{r['request_id']}/{name}.ply",condition_path=cond,prepared_parent_path=parent)
   for ext in ['NPZ','PLY']:row[ext+'_sha256']=sha(OUT/row[ext+'_path'])
   mesh_rows.append(row)
 old=ROOT/'experiments/r2_c_restore_bounded_v1'
 for p in sorted((old/'review').glob('source_C_orientation*')):
  if p.is_file():copy(p,OUT/'prior_source_C_evidence'/p.name)
 copy(old/'registration.json',OUT/'prior_source_C_evidence/parent_registration.json',reg['parent_registration_sha256'])
 cols=list(mesh_rows[0]);atomic(OUT/'mesh_export_index.csv',lambda f:(lambda w:(w.writeheader(),w.writerows(mesh_rows)))(csv.DictWriter(f,fieldnames=cols)))
 write(OUT/'baseline_MODEL_POST/source_index.json',dict(status='PASS',outputs=36,conditions=6,prepared_parents=6,files=copied,
  policy='Original MODEL_POST NPZ/PLY and diagnostics copied byte-for-byte. Absolute provenance paths remain original. Current package paths are in mesh_export_index.csv. Prior source-C evidence is historical source lineage, not a new experiment result.'))
 return copied

def omit_reason(p):
 if p.suffix.lower() in {'.pt','.pth','.ckpt','.npy','.png'}:return 'External model/large raw evidence or image; this winding-only task makes no model call or new image.'
 if p.name=='pairs.json':return 'Original full pair detector JSON retained locally; original classifications are not re-scored, all new CSVs and code are included.'
 if p.name.endswith('.tmp'):return 'Retained incomplete engineering artifact, not a completed result.'
 return None

def select_files():
 selected=[];omitted=[]
 for p in sorted(OUT.rglob('*')):
  if not p.is_file() or p.is_relative_to(FINAL) or p.name in {ZIP.name} or p.name.endswith('.zip.tmp'):continue
  reason=omit_reason(p)
  if reason:omitted.append(dict(path=str(p),relative_path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),hash_basis='Current archive CPU hash',reason=reason))
  else:selected.append(p)
 protected=dict(read(OUT/'registration.json')['protected'])
 review=OUT/'review/source_restoration_contract_audit.json'
 if review.exists():protected.update(read(review).get('protected_before',{}))
 for name,h in protected.items():
  p=Path(name)
  if not p.is_relative_to(OUT):
   omitted.append(dict(path=name,bytes=p.stat().st_size,sha256=h,hash_basis='Frozen registration; root complete before/after protection verification and independent nonweight audit',reason='External retained read-only evidence. Relevant actual source and diagnostic copies are separately indexed.'))
 write(FINAL/'omitted_local_assets.json',dict(files=omitted,deleted_files=0,policy='Omission is only from portable ZIP. Local evidence remains at original paths.'))
 return selected

def verify():
 complete=read(FINAL/'archive_completion.json');assert complete['status']=='COMPLETE';assert_hash(ZIP,complete['zip_sha256'])
 manifest=read(FINAL/'package_manifest.json');names={r['path'] for r in manifest['files']}
 with zipfile.ZipFile(ZIP) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  assert set(z.namelist())==names|{'packaging/package_manifest.json'}
  for row in manifest['files']:
   assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'];assert_hash(OUT/row.get('source_relative_path',row['path']),row['sha256'])
  assert hashlib.sha256(z.read('packaging/package_manifest.json')).hexdigest()==sha(FINAL/'package_manifest.json')
  source=read(OUT/'source_hashes.json')
  for row in source['sources']:assert row['snapshot_path'] in names and sha(row['source_path'])==row['sha256']
  assert sum(n.endswith('.ply') for n in names)==108
  assert sum(n.endswith('.npz') for n in names)==120
  assert not any(Path(n).suffix in {'.pt','.pth','.ckpt','.png'} for n in names)
  index=rows(OUT/'mesh_export_index.csv');assert len(index)==108
  for row in index:
   for ext in ['NPZ','PLY']:
    assert row[ext+'_path'] in names and hashlib.sha256(z.read(row[ext+'_path'])).hexdigest()==row[ext+'_sha256']
  csvs={p.relative_to(OUT).as_posix() for p in OUT.rglob('*.csv') if not p.is_relative_to(FINAL)}
  assert csvs<=names
  checked_links=0
  for name in names:
   if Path(name).suffix.lower() not in {'.html','.md'}:continue
   body=z.read(name).decode('utf-8-sig')
   links=re.findall(r'(?:href|src)=["\']([^"\']+)',body) if name.endswith('.html') else re.findall(r'\[[^\]]*\]\(([^)]+)\)',body)
   for link in links:
    link=html.unescape(link).strip().split(' "')[0].strip('<>');link=link.split('#')[0]
    if not link or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',link) or link.startswith('/'):continue
    target=posixpath.normpath(posixpath.join(posixpath.dirname(name),link))
    assert target in names or any(n.startswith(target.rstrip('/')+'/') for n in names),(name,target,'missing relative document link')
    checked_links+=1
  methods=sorted({i.compress_type for i in z.infolist()});assert methods==[zipfile.ZIP_DEFLATED]
  assert ZIP.stat().st_size<10000000,'Archive exceeds 10 MB; retain package, report size before changing contents'
  return dict(status='PASS',zip_path=str(ZIP),bytes=ZIP.stat().st_size,sha256=sha(ZIP),members=len(z.namelist()),CSV_count=len(csvs),
   NPZ_count=120,PLY_count=108,mesh_stages=108,source_files=len(source['sources']),CRC='PASS',all_member_hashes='PASS',all_CSV_present=True,
   checked_relative_links=checked_links,compression='DEFLATE',under_10MB=True,model_forwards=0,training_updates=0,new_scoring=0,new_orientation_solves=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
 for path in [ROOT/'REPORT.md',ROOT/'experiments/r2_c_restore_bounded_v1/REPORT.md',OUT.parent/'outside']:
  try:checked(path)
  except ValueError:pass
  else:raise AssertionError('Negative output guard failed')
 if a.verify or (FINAL/'archive_completion.json').exists():print(json.dumps(verify(),ensure_ascii=False));return
 prereqs=prerequisite_check()
 if a.check_only:print(json.dumps(dict(status='PASS',prerequisites=len(prereqs),negative_output_guards=3,scientific_calls=0)));return
 before={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(FINAL) and not any(x in p.parts for x in ['source_snapshot','baseline_MODEL_POST','source_conditions','source_parents','prior_source_C_evidence']) and p.name not in {'code.diff','old_source.diff','source_hashes.json','mesh_export_index.csv',ZIP.name}}
 sources=source_snapshot();baselines=baseline_copies();selected=select_files()
 write(FINAL/'prerequisites.json',dict(status='PASS',hashes=prereqs,preexisting_evidence_hashes=before))
 textfile(FINAL/'README_PACKAGE.md','# Source-winding export review package\n\nOpen ../REPORT.md. All new per-output, per-source-face, per-corner witness, edge-constraint, component, pairing and full-threshold CSVs are included. See ../mesh_export_index.csv for all 108 mesh stages: 36 original MODEL_POST, 36 restored C, and 36 final free-oriented outputs, each in NPZ and PLY. The six original condition NPZ and six prepared parent NPZ are included for source lineage. No coordinates or connectivity were changed, and inherited geometric quality is not a new scoring result.\n\nOriginal MODEL_POST diagnostics and prior source-C orientation evidence are copied unchanged and separately indexed. All actual new and original solver/scorer source snapshots, diffs, hashes, registrations, receipts and independent audits are retained. Weights and large unrelated earlier assets remain local, listed with source path/hash in omitted_local_assets.json. No new images were requested or made.\n\nStandard ZIP DEFLATE. Source code.diff lists new files as additions; old_source.diff is empty after frozen hash verification. Archive hash/verification receipts are stored locally after assembly to avoid a self-hash cycle. Absolute references inside historical receipts are provenance, while mesh_export_index.csv supplies portable mesh paths. The historical WORKSPACE.before.md backup is included byte-for-byte as WORKSPACE.before.md.txt; the manifest records its original path so old local links are preserved as text, not presented as portable navigation.\n')
 selected.extend([FINAL/'omitted_local_assets.json',FINAL/'prerequisites.json',FINAL/'README_PACKAGE.md'])
 members=[]
 for p in sorted(set(selected)):
  rel=p.relative_to(OUT).as_posix();archived=rel+'.txt' if p.name=='WORKSPACE.before.md' else rel
  members.append(dict(path=archived,source_relative_path=rel,bytes=p.stat().st_size,sha256=sha(p),adaptation='Historical Markdown backup presented as plain text; bytes unchanged' if archived!=rel else 'none'))
 write(FINAL/'package_manifest.json',dict(schema='source_winding_export_review_v1',files=members,source_files=len(sources),baseline_cache_files=len(baselines),scientific_calls=0))
 temp=checked(ZIP.with_name(ZIP.name+'.tmp'))
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for r in members:z.write(OUT/r['source_relative_path'],r['path'])
  z.write(FINAL/'package_manifest.json','packaging/package_manifest.json');assert z.testzip() is None
 temp.replace(checked(ZIP))
 assert before=={p:sha(p) for p in before},'Preexisting scientific/documentation evidence changed during archive'
 write(FINAL/'archive_completion.json',dict(status='COMPLETE',task='R2_SOURCE_WINDING_EXPORT_V1',zip_path=str(ZIP),zip_bytes=ZIP.stat().st_size,zip_sha256=sha(ZIP),
  source_restorations=36,orientation_outputs=36,NPZ_exports=108,PLY_exports=108,source_condition_and_parent_NPZ=12,
  new_images=0,no_default_change=True,no_git=True,packaging_model_forwards=0,packaging_training_updates=0,packaging_new_scores=0,packaging_orientation_solves=0))
 result=verify();write(FINAL/'verification.json',dict(**result,all_preexisting_evidence_hashes_unchanged=True,compared_files=len(before)))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
