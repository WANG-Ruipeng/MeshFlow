"""CPU-only complete evidence archive. No imports of model or scientific runners."""
from __future__ import annotations
import argparse,csv,difflib,hashlib,html,json,os,posixpath,re,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
OUT=ROOT/'experiments/r2_c_restore_coupling_v1'
ZIP=OUT/'R2_C_RESTORE_COUPLING_REVIEW.zip'
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
 required=['REPORT.md','README_RESUME.md','USER_PROPOSAL.md','completion.json','registration.json','score_registration.json','denoise_completion.json','geometry_completion.json',
  'geometry_checks/completion.json','feedback_checks.json','score_completion.json','score_aggregate_completion.json',
  'post_completion.json','independent_audit/chain_audit.json','independent_audit/score_audit.json','independent_audit/completion.json',
  'gallery/gallery_manifest.json','gallery/visual_review_completion.json','gallery/visual_review.csv','gallery/index.html','gallery/review_534b_f596.csv','gallery/review_534b_f596_images.json',
  'gallery/review_86ec_bdd.csv','gallery/review_86ec_bdd_receipt.json','gallery/review_seats.csv','gallery/review_seats_images.json']
 for name in required:
  if not (OUT/name).is_file():raise RuntimeError('BLOCKED_MISSING '+name)
 d=read(OUT/'denoise_completion.json');s=read(OUT/'score_completion.json');p=read(OUT/'post_completion.json')
 assert d['status']=='PASS' and d['completed_queries']==d['total_forward_attempts']==24
 assert d['new_generator_forwards']==d['training_updates']==d['new_noise']==d['synthetic_D0_forwards']==0
 assert d['FP32_parameters_unchanged'] and d['all_F_RAW_inputs_exact'] and d['all_export_C_exact']
 assert s['status']=='PASS' and s['new_POST_quality']==s['new_GEO']==24 and s['reused_POST_quality']==24 and s['new_model_calls']==0
 assert p['status']=='PASS' and p['outputs']==48
 for name in ['independent_audit/chain_audit.json','independent_audit/score_audit.json','independent_audit/completion.json','geometry_checks/completion.json','feedback_checks.json']:
  assert read(OUT/name)['status']=='PASS'
 assert len(read(OUT/'geometry_checks/completion.json')['cases'])==4 and len(read(OUT/'feedback_checks.json')['cases'])==3
 gallery=read(OUT/'gallery/gallery_manifest.json')
 assert gallery['status']=='COMPLETE' and gallery['PNG_count']==gallery['JPEG_count']==24 and gallery['model_calls']==0
 expected={r['png']:r['sha256'] for r in gallery['receipts']};assert len(expected)==24
 for receipt in gallery['receipts']:
  assert_hash(receipt['png'],receipt['sha256']);assert_hash(receipt['jpg'],receipt['jpg_sha256'])
  for source,h in receipt['identity']['source_hashes'].items():assert_hash(source,h)
 for source,h in gallery['definition_hashes'].items():assert_hash(source,h)
 viewed=[];reviewrows=[]
 for csvname,name in [('review_534b_f596.csv','review_534b_f596_images.json'),('review_86ec_bdd.csv','review_86ec_bdd_receipt.json'),('review_seats.csv','review_seats_images.json')]:
  data=read(OUT/'gallery'/name);items=data.get('images',data.get('records',[]));assert data['status'] in ['COMPLETE','PASS'] and len(items)==8
  for item in items:
   png=item.get('source_PNG_path',item.get('path'));png_hash=item.get('source_PNG_sha256',item.get('sha256'))
   actual=item.get('viewed_path',item.get('path'));actual_hash=item.get('viewed_sha256',item.get('sha256'))
   assert item['actually_viewed'] is True and expected[png]==png_hash;assert_hash(actual,actual_hash)
   viewed.append(dict(path=actual,sha256=actual_hash,actually_viewed=True,source_PNG_path=png,source_PNG_sha256=png_hash,viewed_format=Path(actual).suffix,receipt='gallery/'+name))
  rr=rows(OUT/'gallery'/csvname);assert len(rr)==16 and all(r['review_status'] in ['COMPLETE','REVIEWED'] and r['review_kind']=='AI_PROXY' for r in rr)
  reviewrows.extend(rr)
 assert len(viewed)==24 and {r['source_PNG_path'] for r in viewed}==set(expected)
 expected_review={(r['request_id'],m) for r in read(OUT/'registration.json')['records'] for m in ['D0_BASE','RBF','FEEDBACK','D3']}
 assert len(reviewrows)==48 and {(r['request_id'],r['method']) for r in reviewrows}==expected_review
 assert len(rows(OUT/'post_index.csv'))==48 and len(rows(OUT/'paired_per_output.csv'))==48
 return {str(OUT/n):sha(OUT/n) for n in required},viewed

def portable_gallery():
 g=read(OUT/'gallery/gallery_manifest.json');parts=['<!doctype html><meta charset="utf-8"><title>C restoration coupling review</title><style>body{font-family:Arial;margin:20px}img{width:min(100%,1500px);height:auto}figure{margin:20px 0}</style><h1>C restoration coupling: all 12 fixed requests</h1><p>Four panels: D0_BASE, RBF, FEEDBACK, D3. Original blue C, gray free. Same views/bounds; no GT or hidden-face removal. AI_PROXY non-blind review; images do not certify attachment.</p>']
 for item in g['receipts']:
  jpg=Path(item['jpg']);relative=jpg.relative_to(OUT/'gallery').as_posix()
  ident=item['identity'];parts.append('<figure><figcaption>'+html.escape(ident['request_id']+' view '+str(ident['view_id']))+'</figcaption><a href="'+relative+'"><img loading="lazy" src="'+relative+'"></a></figure>')
 textfile(OUT/'gallery/index_portable.html','\n'.join(parts))

def source_snapshot():
 reg=read(OUT/'registration.json');score=read(OUT/'score_registration.json');registered={}
 for table in [reg.get('runner_hashes',{}),reg.get('protected',{}),score.get('quality_source_hashes',{}),score.get('protected',{})]:
  for p,h in table.items():
   if p.endswith('.py'):
    if p in registered and registered[p]!=h:raise ValueError('Conflicting source registration')
    registered[p]=h
 for p,h in registered.items():assert_hash(p,h)
 new={str(p):sha(p) for p in CODE.glob('*.py')}
 selected={**registered,**new}
 # Actual renderer ancestor and legal/dependency declarations accompany copied code.
 for p in [ROOT/'r2/tools/multistep_denoise_v1/gallery.py',ROOT/'LICENSE',ROOT/'r2/LICENSE',ROOT/'r2/THIRD_PARTY_NOTICES.md',ROOT/'requirements.txt']:
  if p.exists():selected[str(p)]=sha(p)
 manifest=[]
 for name,h in sorted(selected.items()):
  p=Path(name)
  rel=('repository/'+p.relative_to(ROOT).as_posix()) if p.is_relative_to(ROOT) else ('external/'+p.relative_to('/mnt/e').as_posix())
  dest=OUT/'source_snapshot'/rel;immutable_copy(p,dest);assert_hash(dest,h)
  manifest.append(dict(source_path=name,snapshot_path=dest.relative_to(OUT).as_posix(),sha256=h,bytes=p.stat().st_size,new_this_experiment=name in new,registered_before_execution=name in registered))
 diffs=[]
 for name in sorted(new):
  p=Path(name);lines=p.read_text(encoding='utf-8').splitlines(keepends=True)
  diffs.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix()))
 textfile(OUT/'code.diff',''.join(diffs));textfile(OUT/'old_source.diff','')
 write(OUT/'source_hashes.json',{'schema':'actual_sources_v1','sources':manifest,'old_source_diff':'empty; registered old source hashes verified unchanged','new_diff':'All new experiment .py files, including this assembler, represented as additions from /dev/null.'})
 return manifest

def omit_reason(p):
 if p.suffix.lower() in {'.npz','.npy','.pt','.pth','.ckpt','.ply','.png'}:return 'Original geometry, weights or full-resolution PNG retained locally; portable JPEG and full CSV evidence included.'
 if p.name=='pairs.json':return 'Large raw pair JSON retained locally; complete FF fate/source-face CSV and scoring source are included.'
 if p.name.endswith('.tmp'):return 'Temporary incomplete engineering artifact; retained locally.'
 return None

def select_files():
 selected=[];omitted=[]
 for p in sorted(OUT.rglob('*')):
  if not p.is_file() or p.is_relative_to(FINAL) or p.name in {ZIP.name} or p.name.endswith('.zip.tmp'):continue
  reason=omit_reason(p)
  if reason:omitted.append(dict(path=str(p),relative_path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),hash_basis='computed during current CPU archive',reason=reason))
  else:selected.append(p)
 protected={}
 for name in ['registration.json','score_registration.json']:
  for p,h in read(OUT/name).get('protected',{}).items():
   if p in protected and protected[p]['sha256']!=h:raise ValueError('Conflicting protected file hashes')
   protected[p]=dict(path=p,bytes=Path(p).stat().st_size,sha256=h,hash_basis='Frozen '+name+' plus completed independent audit; not re-reading large weights for archive.',reason='External retained protected asset; original path is read-only and preserved locally.')
 omitted.extend(v for k,v in protected.items() if not Path(k).is_relative_to(OUT))
 write(FINAL/'omitted_local_assets.json',{'files':omitted,'deleted_files':0,'policy':'Omission applies only to the portable ZIP; every original stays at its recorded local path.'})
 return selected

def verify():
 complete=read(FINAL/'archive_completion.json');assert complete['status']=='COMPLETE';assert_hash(ZIP,complete['zip_sha256'])
 manifest=read(FINAL/'package_manifest.json');names={r['path'] for r in manifest['files']}
 with zipfile.ZipFile(ZIP) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  assert set(z.namelist())==names|{'packaging/package_manifest.json'}
  for row in manifest['files']:
   assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256'];assert_hash(row.get('local_source_path',OUT/row['path']),row['sha256'])
  assert hashlib.sha256(z.read('packaging/package_manifest.json')).hexdigest()==sha(FINAL/'package_manifest.json')
  source=read(OUT/'source_hashes.json')
  for row in source['sources']:assert row['snapshot_path'] in names and sha(row['source_path'])==row['sha256']
  assert sum(n.endswith('.jpg') for n in names)==24
  assert not any(Path(n).suffix in {'.pt','.pth','.npz','.npy','.ply','.png'} for n in names)
  actual_csv={p.relative_to(OUT).as_posix() for p in OUT.rglob('*.csv') if not p.is_relative_to(FINAL)}
  assert actual_csv<=names
  body=z.read('gallery/index_portable.html').decode('utf-8');links=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',body)
  checked_links=[]
  for value in links:
   value=html.unescape(value)
   if value.startswith(('http:','https:','#','data:')):continue
   target=posixpath.normpath(posixpath.join('gallery',value.split('#')[0]));assert target in names,(target,'missing portable link');checked_links.append(target)
  checked_document_links=0
  for name in names:
   if Path(name).suffix.lower() not in {'.html','.md'}:continue
   body=z.read(name).decode('utf-8-sig')
   matches=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',body) if name.endswith('.html') else re.findall(r'\[[^\]]*\]\(([^)]+)\)',body)
   for link in matches:
    link=html.unescape(link).strip().split(' \"')[0].strip('<>');link=link.split('#')[0]
    if not link or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',link) or link.startswith('/'):continue
    target=posixpath.normpath(posixpath.join(posixpath.dirname(name),link))
    assert target in names or any(n.startswith(target.rstrip('/')+'/') for n in names),(name,target,'missing document link')
    checked_document_links+=1
  result=dict(status='PASS',checked_document_links=checked_document_links,zip_path=str(ZIP),bytes=ZIP.stat().st_size,sha256=sha(ZIP),members=len(z.namelist()),CSV_count=len(actual_csv),JPEG_count=24,source_files=len(source['sources']),CRC='PASS',all_member_hashes='PASS',portable_links=len(checked_links),all_CSV_present=True,compression_methods=sorted(set(i.compress_type for i in z.infolist())),model_forwards=0,training_updates=0,new_scoring=0)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');ap.add_argument('--verify',action='store_true');args=ap.parse_args()
 for forbidden in [ROOT/'REPORT.md',ROOT/'experiments/r2_multistep_denoise_v1/not_allowed',OUT.parent/'outside']:
  try:checked(forbidden)
  except ValueError:pass
  else:raise AssertionError('Output guard failed')
 if args.verify or (FINAL/'archive_completion.json').exists():print(json.dumps(verify(),ensure_ascii=False));return
 prereqs,viewed=prerequisite_check()
 if args.check_only:print(json.dumps({'status':'PASS','prerequisites':len(prereqs),'actually_viewed_images':len(viewed),'viewed_PNG':sum(r['viewed_format']=='.png' for r in viewed),'viewed_JPEG':sum(r['viewed_format']=='.jpg' for r in viewed),'scientific_calls':0}));return
 scientific_completion=read(OUT/'completion.json');scientific_completion['review_ZIP_status']='READY_FOR_PACKAGE';write(OUT/'completion.json',scientific_completion)
 before={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and not p.is_relative_to(FINAL) and 'source_snapshot' not in p.parts and p.name not in {'code.diff','old_source.diff','source_hashes.json',ZIP.name}}
 portable_gallery();sources=source_snapshot();selected=select_files()
 write(FINAL/'visual_review_coverage.json',{'status':'COMPLETE','review_kind':'AI_PROXY','blind':False,'actually_viewed_images':24,'actually_viewed_PNG_count':16,'actually_viewed_JPEG_count':8,'images':viewed,'limitation':'Non-blind actual image observations; not human ratings or topology certification.'})
 write(FINAL/'prerequisites.json',{'status':'PASS','hashes':prereqs,'scientific_before_hashes':before})
 textfile(FINAL/'README_PACKAGE.md','# Review package\n\nOpen ../REPORT.md and ../gallery/index_portable.html. All experiment CSV bytes, 24 JPEG views, actual old/new source code, diffs, registrations, receipts and ledgers are included. No adverse metric or request is removed. Large raw pair JSON, geometry arrays, weights and original PNGs remain local and are individually indexed with SHA256 in omitted_local_assets.json. The archive gallery/index.html is adapted to the same portable JPEG view as index_portable.html; the original local full-PNG index is unchanged. portable_adaptations.json records both hashes. REPORT.md is copied byte-for-byte.\n\nCompression is standard ZIP DEFLATE. The source snapshot preserves existing source bytes; code.diff consists only of the new experiment files. old_source.diff is empty after registered original source hashes pass. packaging/archive_completion.json and archive verification are produced outside the ZIP after packaging to avoid a circular ZIP self-hash.\n')
 selected.extend([FINAL/'omitted_local_assets.json',FINAL/'visual_review_coverage.json',FINAL/'prerequisites.json',FINAL/'README_PACKAGE.md'])
 adapted_index=FINAL/'portable_gallery_index.html';immutable_copy(OUT/'gallery/index_portable.html',adapted_index)
 write(FINAL/'portable_adaptations.json',{'REPORT_bytes_unchanged':True,'adaptations':[{'archive_path':'gallery/index.html','local_original':str(OUT/'gallery/index.html'),'local_original_sha256':sha(OUT/'gallery/index.html'),'archive_source':str(adapted_index),'archive_sha256':sha(adapted_index),'reason':'Portable-only gallery points at all24 original JPEGs; original local full-PNG index remains unchanged.'}]})
 selected.append(FINAL/'portable_adaptations.json')
 file_records=[]
 for p in sorted(set(selected)):
  name=p.relative_to(OUT).as_posix();source=adapted_index if name=='gallery/index.html' else p
  file_records.append(dict(path=name,bytes=source.stat().st_size,sha256=sha(source),local_source_path=str(source)))
 write(FINAL/'package_manifest.json',{'schema':'r2_c_restore_coupling_review_package_v1','files':file_records,'source_files':len(sources),'no_scientific_calls_during_packaging':True})
 temp=checked(ZIP.with_name(ZIP.name+'.tmp'))
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for row in file_records:z.write(row['local_source_path'],row['path'])
  z.write(FINAL/'package_manifest.json','packaging/package_manifest.json')
  assert z.testzip() is None
 temp.replace(checked(ZIP))
 assert before=={p:sha(p) for p in before},'Scientific evidence changed during CPU packaging'
 write(FINAL/'archive_completion.json',dict(status='COMPLETE',task='R2_C_RESTORE_COUPLING_V1',conclusion='See unchanged REPORT.md; no automatic adoption',student_forwards=0,new_GEO=24,new_quality=24,teacher_forwards=24,generator_forwards=0,training_updates=0,CPU_geometry_controls=4,CPU_analytic_feedback_controls=3,actually_viewed_images=24,actually_viewed_PNGs=16,actually_viewed_JPEGs=8,chain_audit='PASS',score_audit='PASS',no_default_change=True,no_git=True,zip_path=str(ZIP),zip_bytes=ZIP.stat().st_size,zip_sha256=sha(ZIP),packaging_scientific_calls=0))
 result=verify();write(FINAL/'verification.json',{**result,'all_preexisting_evidence_hashes_unchanged':True,'compared_files':len(before)});print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
