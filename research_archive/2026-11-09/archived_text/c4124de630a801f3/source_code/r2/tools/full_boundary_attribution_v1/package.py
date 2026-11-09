"""Archive real code and bounded review evidence without touching originals."""
from . import common as C
import io,json,hashlib,difflib,zipfile
from pathlib import Path
from PIL import Image
ZIPNAME='R2_FULL_BOUNDARY_ATTRIBUTION_V1_REVIEW_SMALL.zip'

def main():
 done=C.OUT/'package_completion.json'
 if done.exists():
  d=C.read(done);assert C.sha(d['zip_path'])==d['zip_sha256'];print('Cached archive verified',d['zip_bytes']);return d
 reg=C.read(C.OUT/'registration.json')
 for p,h in reg['protected_hashes'].items():assert C.sha(p)==h,p
 assert C.read(C.OUT/'independent_audit/completion.json')['status']=='PASS'
 assert C.read(C.OUT/'independent_audit/aggregation_completion.json')['status']=='PASS'
 assert len(C.csvread(C.OUT/'AI_PROXY_boundary_review.csv'))==44
 code={};source_rows=[]
 def archive_source(p,expected=None):
  p=C.resolve(p);h=C.sha(p)
  if expected:assert h==expected,p
  if p.is_relative_to(C.ROOT):rel=p.relative_to(C.ROOT)
  else:rel=Path('external_MeshFlow_Control')/p.relative_to(Path('/mnt/e/MeshFlow-Control'))
  target=C.OUT/'source_code'/rel;C._write_bytes(target,p.read_bytes());code[str(rel)]=h
  source_rows.append(dict(original_path=str(p),archive_path='source_code/'+str(rel),sha256=h,executed_this_task=p.is_relative_to(C.CODE) or p==C.ROOT/'r2/tools/cref_layer_audit_v1/geometry.py'))
 for p in sorted(C.CODE.iterdir()):
  if p.is_file() and p.suffix in ['.py','.ps1']:archive_source(p)
 for p in [C.ROOT/'r2/tools/cref_layer_audit_v1/geometry.py',C.ROOT/'r2/tools/cref_layer_audit_v1/render.py',C.ROOT/'r2/pilots/regular_patch_control_v1/depth_gallery.py']:
  archive_source(p)
 for p,h in C.read(C.BASE/'evaluation/score_completion.json')['definition_hashes'].items():archive_source(p,h)
 C.csvwrite(C.OUT/'source_hashes.csv',source_rows)
 diff=''.join(''.join(difflib.unified_diff([],p.read_text(encoding='utf-8-sig').splitlines(keepends=True),fromfile='/dev/null',tofile=str(p.relative_to(C.ROOT)),lineterm='\n')) for p in sorted(C.CODE.iterdir()) if p.is_file() and p.suffix in ['.py','.ps1'])
 C._write_bytes(C.OUT/'code.diff',diff.encode('utf-8'))
 entries={}
 def add(p,name=None):entries[name or str(p.relative_to(C.OUT))]=p.read_bytes()
 for fn in ['REPORT.md','registration.json','analysis_completion.json','aggregation_completion.json','ledger.jsonl','resources_start.json','resources_after_first_cell.json','resources_after_analysis.json','AI_PROXY_boundary_review.csv','root_visual_notes.csv','source_hashes.csv','code.diff']:
  add(C.OUT/fn)
 for folder in ['tables','independent_audit','interpretation','source_code']:
  for p in sorted((C.OUT/folder).rglob('*')):
   if p.is_file():add(p)
 for p in sorted(C.OUT.glob('*review*.json')):add(p)
 for c in reg['cells']:
  cid=c['cell_id'];add(C.resolve(c['geometry_path']))
  inherit=C.read(c['inherited_path']);pairs={}
  for scope,scan in inherit['pairs'].items():
   if not isinstance(scan,dict):continue
   pairs[scope]={k:v for k,v in scan.items() if k!='contacts'}
   pairs[scope]['contacts']=[dict(pair=x['pair'],classification=x.get('classification'),detector={k:x.get('detector',{}).get(k) for k in ['illegal','resolved']}) for x in scan.get('contacts',[])]
  e=dict(source_inherited_sha256=c['inherited_sha256'],source_score_path=c['score_path'],source_score_sha256=c['score_sha256'],pairs=pairs,
   numeric=inherit['numeric'],review=inherit['review'],old_boundary_vertices=inherit['old_boundary_vertices'],old_boundary_distances=inherit['old_boundary_distances'],
   valid_CF_edges=[x for x in inherit['valid_edges'] if x['is_C_F']=='True'],
   compression_note='Full original face-pair detector witness geometry omitted; pair IDs/classification/illegal/resolved and scan coverage metadata retained. Original files protected locally.')
  entries['evidence/'+cid+'.json']=(json.dumps(e,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
  for name in ['samples.csv','summary.json','boundary_edges.csv','boundary_vertices.csv','edge_summary.csv','free_faces.csv']:
   add(C.OUT/'results'/cid/name)
 # All 11 images retained as smaller display previews, no selection or geometry edits.
 gallery=C.read(C.OUT/'gallery_v2/manifest.json');images=[]
 for im in gallery['images']:
  p=C.resolve(im['path']);assert C.sha(p)==im['sha256']
  picture=Image.open(p);picture.thumbnail((960,840));buf=io.BytesIO();picture.save(buf,format='JPEG',quality=82,optimize=True)
  name='gallery/'+p.name;entries[name]=buf.getvalue();images.append(dict(request_id=im['request_id'],path=name,sha256=hashlib.sha256(buf.getvalue()).hexdigest(),source_full_resolution_path=str(p),source_sha256=im['sha256'],pixels=list(picture.size)))
 entries['gallery/manifest.json']=json.dumps(dict(images=images,note='All11; reduced display resolution only. Original reviewed full resolution images remain local gallery_v2.')).encode()
 entries['gallery/index.html']=('<meta charset="utf-8">'+''.join('<h2>'+x['request_id']+'</h2><img width="960" src="'+Path(x['path']).name+'">' for x in images)).encode()
 entries['RECOMPUTE_REVIEW.py']=(C.CODE/'RECOMPUTE_REVIEW.py').read_bytes()
 entries['README.txt']=('FULL_BOUNDARY_ATTRIBUTION small review archive\n'
 'Contains all11requests/all44cells, byte-identical compact mesh NPZ, all27112sample CSV rows with full original columns, boundary/face tables, inherited metrics, compact face-pair classifications, actual source and original scorer code, hashes and reviews.\n'
 'Omitted redundant vectors, full per-F edge/vertex tables, all normal-hit/interval NPZ and layers, unabridged detector witnesses, V1 images and original full-res JPG; all are retained in the local experiment folder. Geometry+actual code reproduce these quantities. No weights needed.\n'
 'Original absolute provenance paths in registration/receipts refer to the protected local originals, not paths expected to exist in this small archive. Portable check uses relative compact entries instead.\n'
 'Extract, install numpy, run: python RECOMPUTE_REVIEW.py --cell 3f7417_n0__CREF_C200__RAW\n'
 'Use --cell all to recompute all44cells. This uses CPU NumPy only, never imports model code or reruns original quality scorers.\n'
 'Gallery previews are all11reviewed pages, downscaled only. AI_PROXY review hashes identify original full-resolution images; gallery manifest records the preview transformation.\n'
 'CSV UTF-8 with BOM; report UTF-8. No S success metric; all-boundary distances descriptive. Selection is observed-case diagnosis, not a held-out success rate.\n').encode('utf-8')
 manifest={name:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for name,b in sorted(entries.items())}
 entries['ARCHIVE_MANIFEST.json']=json.dumps(manifest,indent=2,sort_keys=True).encode()
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name,b in sorted(entries.items()):z.writestr(name,b)
 payload=buffer.getvalue();assert len(payload)<10000000,('Review archive exceeds10MB',len(payload))
 dest=C.OUT/ZIPNAME;C._write_bytes(dest,payload)
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
 result=dict(status='COMPLETE',zip_path=str(dest),zip_bytes=len(payload),zip_sha256=C.sha(dest),entries=len(entries),cells=44,per_point_csv_rows=27112,all11images=True,archive_manifest_verified=True,
  original_files_unchanged=len(reg['protected_hashes']),model_calls=0,new_quality_scans=0)
 C.write(done,result);C.event('REVIEW_ARCHIVE_COMPLETE',**result);print(json.dumps(result));return result
if __name__=='__main__':main()