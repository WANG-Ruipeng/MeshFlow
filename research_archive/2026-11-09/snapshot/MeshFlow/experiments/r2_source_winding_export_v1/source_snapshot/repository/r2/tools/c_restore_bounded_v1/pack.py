"""CPU-only complete evidence archive. No imports of model or scientific runners."""
from __future__ import annotations
import argparse,csv,difflib,hashlib,html,json,os,posixpath,re,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/r2_c_restore_bounded_v1'
ZIP=OUT/'R2_C_RESTORE_BOUNDED_REVIEW.zip'
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
 required=['REPORT.md','README_RESUME.md','completion.json','registration.json','execution_completion.json',
  'geometry_completion.json','score_completion.json','post_completion.json','aggregate_completion.json',
  'blend_checks/completion.json','orientation_checks/completion.json','independent_audit/completion.json',
  'gallery/manifest.json','gallery/index.html','gallery/visual_review.csv','gallery/visual_review_completion.json',
  'decision_per_output.csv','decision_summary.csv','paired_per_parent_noise.csv','AVMR_PMR_per_output.csv']
 for name in required:
  if not (OUT/name).is_file():raise RuntimeError('BLOCKED_MISSING '+name)
 done=read(OUT/'execution_completion.json')
 assert done['status']=='PASS' and done['new_blend_outputs']==done['new_GEO_C']==done['new_POST_quality']==12
 assert done['orientation_outputs']==36 and done['new_model_forwards']==done['training_updates']==done['new_noise']==done['new_generator_outputs']==0
 for name in ['independent_audit/completion.json','blend_checks/completion.json','orientation_checks/completion.json','gallery/visual_review_completion.json']:
  assert read(OUT/name)['status'] in ('PASS','COMPLETE')
 assert read(OUT/'independent_audit/completion.json')['audit_source_sha256']==sha(CODE/'independent_audit.py')
 assert read(OUT/'blend_checks/completion.json')['checks']==8 and len(read(OUT/'orientation_checks/completion.json')['cases'])==7
 g=read(OUT/'gallery/manifest.json');assert g['status']=='PASS' and g['images']==24 and g['requests']==12
 lookup={}
 for q in g['receipts']:
  for typ in ['png','jpg']:
   assert_hash(q[typ],q[typ+'_sha256']);lookup[q[typ]]=q[typ+'_sha256']
  for p,h in q['identity']['source_hashes'].items():assert_hash(p,h)
 records=[]
 for p in sorted((OUT/'gallery').glob('*images.json')):
  q=read(p)
  for item in q.get('images',q.get('records',[])):
   if not item.get('actually_viewed'):raise ValueError('Review image not actually viewed')
   path=item.get('viewed_path',item.get('path'))
   if path is None:path=str(OUT/'gallery'/item['file'])
   h=item.get('viewed_sha256',item.get('sha256'))
   assert lookup[path]==h;assert_hash(path,h)
   records.append(dict(path=path,sha256=h,actually_viewed=True,format=Path(path).suffix,receipt=str(p.relative_to(OUT))))
 assert len(records)==24 and len({r['path'] for r in records})==24
 # A source view may be observed in PNG or JPEG but never counted twice.
 image_ids={str(Path(r['path']).with_suffix('')) for r in records}
 assert len(image_ids)==24 and image_ids=={str(Path(q['png']).with_suffix('')) for q in g['receipts']}
 review=rows(OUT/'gallery/visual_review.csv');expected={(r['request_id'],m) for r in read(OUT/'registration.json')['records'] for m in ['D0_BASE','RBF','BLEND']}
 assert len(review)==36 and {(r['request_id'],r['method']) for r in review}==expected
 assert all(r['review_kind']=='AI_PROXY' and r['review_status'] in ('COMPLETE','REVIEWED') for r in review)
 assert len(read(OUT/'post_index.json'))==36 and len(rows(OUT/'AVMR_PMR_per_output.csv'))==576
 return {str(OUT/n):sha(OUT/n) for n in required},records

def source_snapshot():
 reg=read(OUT/'registration.json');protected=dict(reg['protected']);selected={}
 review=OUT/'review/source_C_orientation_receipt.json'
 if review.exists():
  rd=read(review);assert rd['status']=='PASS';protected.update(rd['protected_before'])
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
 reg=read(OUT/'registration.json');copied=[]
 for r in reg['baselines']:
  folder=OUT/'baseline_diagnostics'/r['method']/r['request_id'];d=read(r['diagnostic_path'])
  files={Path(r['diagnostic_path'])}
  for p in d.get('files',{}):
   if Path(p).suffix=='.csv':files.add(Path(p))
  files.add(Path(d['quality']['shape_details_path']))
  for p in sorted(files):
   target=folder/p.name;immutable_copy(p,target)
   copied.append(dict(source_path=str(p),archive_path=target.relative_to(OUT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
 write(OUT/'baseline_diagnostics/source_index.json',dict(status='PASS',methods=['D0_BASE','RBF'],outputs=24,files=copied,
  policy='Unchanged original diagnostic/cache bytes; absolute references remain original provenance paths. Full36 aggregate and per-output CSVs are in the package root. Large pair JSON and geometry remain local.'))
 return copied

def omit_reason(p):
 if p.suffix.lower() in {'.npz','.npy','.pt','.pth','.ckpt','.ply','.png'}:return 'Retained original arrays, mesh, weight or PNG; portable review uses unchanged JPEG and complete CSVs.'
 if p.name=='pairs.json':return 'Large complete pair detector JSON retained locally; all scalar metrics, edge constraints, original scorer code and source identities are included.'
 if p.name.endswith('.tmp'):return 'Retained temporary engineering artifact, not a completed result.'
 return None

def select_files():
 selected=[];omitted=[]
 for p in sorted(OUT.rglob('*')):
  if not p.is_file() or p.is_relative_to(FINAL) or p.name in {ZIP.name} or p.name.endswith('.zip.tmp'):continue
  reason=omit_reason(p)
  if reason:omitted.append(dict(path=str(p),relative_path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),hash_basis='Current archive CPU hash',reason=reason))
  else:selected.append(p)
 protected=dict(read(OUT/'registration.json')['protected'])
 review=OUT/'review/source_C_orientation_receipt.json'
 if review.exists():protected.update(read(review)['protected_before'])
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
   assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'];assert_hash(OUT/row['path'],row['sha256'])
  assert hashlib.sha256(z.read('packaging/package_manifest.json')).hexdigest()==sha(FINAL/'package_manifest.json')
  source=read(OUT/'source_hashes.json')
  for row in source['sources']:assert row['snapshot_path'] in names and sha(row['source_path'])==row['sha256']
  assert sum(n.endswith('.jpg') for n in names)==24
  assert not any(Path(n).suffix in {'.pt','.pth','.npz','.npy','.ply','.png'} for n in names)
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
  return dict(status='PASS',zip_path=str(ZIP),bytes=ZIP.stat().st_size,sha256=sha(ZIP),members=len(z.namelist()),CSV_count=len(csvs),JPEG_count=24,
   source_files=len(source['sources']),CRC='PASS',all_member_hashes='PASS',all_CSV_present=True,checked_relative_links=checked_links,
   compression='DEFLATE',under_10MB=ZIP.stat().st_size<10000000,model_forwards=0,training_updates=0,new_scoring=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');ap.add_argument('--verify',action='store_true');a=ap.parse_args()
 for path in [ROOT/'REPORT.md',ROOT/'experiments/r2_c_restore_coupling_v1/REPORT.md',OUT.parent/'outside']:
  try:checked(path)
  except ValueError:pass
  else:raise AssertionError('Negative output guard failed')
 if a.verify or (FINAL/'archive_completion.json').exists():print(json.dumps(verify(),ensure_ascii=False));return
 prereqs,viewed=prerequisite_check()
 if a.check_only:print(json.dumps(dict(status='PASS',prerequisites=len(prereqs),actually_viewed_images=len(viewed),scientific_calls=0)));return
 before={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(FINAL) and 'source_snapshot' not in p.parts and 'baseline_diagnostics' not in p.parts and p.name not in {'code.diff','old_source.diff','source_hashes.json',ZIP.name}}
 sources=source_snapshot();baselines=baseline_copies();selected=select_files()
 write(FINAL/'prerequisites.json',dict(status='PASS',hashes=prereqs,preexisting_evidence_hashes=before))
 write(FINAL/'visual_review_coverage.json',dict(status='PASS',images=viewed,actually_viewed_images=24,review_kind='AI_PROXY',blind=False,
  viewed_format_counts={ext:sum(r['format']==ext for r in viewed) for ext in sorted({r['format'] for r in viewed})},not_topology_certificate=True))
 textfile(FINAL/'README_PACKAGE.md','# Portable review package\n\nOpen ../REPORT.md and ../gallery/index.html. All new per-output/per-corner/edge-constraint/pairing/full-threshold CSVs, all24 original JPEGs, registrations, ledgers, independent audit, actual scoring/code snapshots and diffs are included. No adverse result is removed. Unchanged D0/RBF diagnostic and small shape/edge cache copies are under baseline_diagnostics, with source paths and hashes. NPZ/PLY/PNG/weights and large full pair JSON remain local and are indexed in omitted_local_assets.json. Source face/corner mappings and geometry-equivalence certificates document flip-only reuse without claiming a new quality scorer call.\n\nStandard ZIP DEFLATE compression. Old source diff is empty after hash verification; new code.diff represents this experiment as additions from /dev/null. Archive completion/hash verification receipts are local packaging outputs created after ZIP assembly to avoid self-hash circularity.\n')
 selected.extend([FINAL/'omitted_local_assets.json',FINAL/'prerequisites.json',FINAL/'visual_review_coverage.json',FINAL/'README_PACKAGE.md'])
 members=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(selected))]
 write(FINAL/'package_manifest.json',dict(schema='bounded_C_restore_review_v1',files=members,source_files=len(sources),baseline_cache_files=len(baselines),scientific_calls=0))
 temp=checked(ZIP.with_name(ZIP.name+'.tmp'))
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for r in members:z.write(OUT/r['path'],r['path'])
  z.write(FINAL/'package_manifest.json','packaging/package_manifest.json');assert z.testzip() is None
 temp.replace(checked(ZIP))
 assert before=={p:sha(p) for p in before},'Preexisting scientific/documentation evidence changed during archive'
 write(FINAL/'archive_completion.json',dict(status='COMPLETE',task='R2_C_RESTORE_BOUNDED_AND_ORIENTATION_V1',
  zip_path=str(ZIP),zip_bytes=ZIP.stat().st_size,zip_sha256=sha(ZIP),new_blend_outputs=12,new_GEO=12,new_POST_quality=12,
  orientation_outputs=36,comparison_images=24,actually_viewed_images=24,no_default_change=True,no_git=True,
  packaging_model_forwards=0,packaging_training_updates=0,packaging_new_scores=0))
 result=verify();write(FINAL/'verification.json',dict(**result,all_preexisting_evidence_hashes_unchanged=True,compared_files=len(before)))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
