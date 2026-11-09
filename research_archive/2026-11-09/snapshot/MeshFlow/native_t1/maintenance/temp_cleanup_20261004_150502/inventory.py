"""Read-only storage inventory and conservative temporary-cache candidates."""
from pathlib import Path
import os,json,csv,hashlib,collections,subprocess,time,stat
ROOT=Path('E:/MeshFlow').resolve()
OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def csvwrite(name,rows,fields):
 with (OUT/name).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def walk():
 files=[];skipped=[];errors=[]
 for cur,dirs,names in os.walk(ROOT,followlinks=False):
  kept=[]
  for name in dirs:
   p=Path(cur)/name
   if p.is_symlink() or p.is_junction():skipped.append(str(p.relative_to(ROOT)))
   else:kept.append(name)
  dirs[:]=kept
  for name in names:
   p=Path(cur)/name
   try:
    st=p.stat(follow_symlinks=False)
    if stat.S_ISLNK(st.st_mode):skipped.append(str(p.relative_to(ROOT)));continue
    files.append(dict(path=str(p.resolve()),relative=str(p.relative_to(ROOT)).replace('\\','/'),bytes=st.st_size,mtime_ns=st.st_mtime_ns,suffix=p.suffix.lower()))
   except OSError as e:errors.append(dict(path=str(p),error=str(e)))
 return files,skipped,errors

if __name__=='__main__':
 start=time.time();files,skipped,errors=walk()
 tiers={d:collections.defaultdict(lambda:dict(files=0,bytes=0)) for d in (1,2,3,4)};ext=collections.defaultdict(lambda:dict(files=0,bytes=0))
 for r in files:
  parts=r['relative'].split('/')
  for d,bins in tiers.items():
   k='/'.join(parts[:d]) if len(parts)>d else '/'.join(parts[:-1]) or '[root files]'
   bins[k]['files']+=1;bins[k]['bytes']+=r['bytes']
  ext[r['suffix']]['files']+=1;ext[r['suffix']]['bytes']+=r['bytes']
 csvwrite('file_inventory_before.csv',files,['path','relative','bytes','mtime_ns','suffix'])
 for d,bins in tiers.items():csvwrite(f'directory_sizes_depth{d}.csv',[dict(directory=k,**v) for k,v in sorted(bins.items(),key=lambda x:x[1]['bytes'],reverse=True)],['directory','files','bytes'])
 latest=ROOT/'native_t1/runs/chair_control_attribution_v1/source_manifest.json';protected=set()
 if latest.exists():
  for r in json.loads(latest.read_text(encoding='utf8'))['sources']:
   x=r.get('relative_path')
   if x:protected.add(x.replace('\\','/').lower())
 tracked=set(subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True,check=True).stdout.decode('utf8').split('\x00'))
 candidates=[];review=[]
 for r in files:
  rel=r['relative'];parts=rel.split('/');suffix=r['suffix'];name=parts[-1]
  excluded=parts[0] in ('.git','.worktrees','.codex','.agents') or rel.lower() in protected or rel in tracked or rel.startswith(str(OUT.relative_to(ROOT)).replace('\\','/'))
  cache=any(p in ('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache') for p in parts)
  if cache and not excluded and (suffix in ('.pyc','.pyo','.nbc','.nbi') or any(p in ('.pytest_cache','.mypy_cache','.ruff_cache') for p in parts)):
   candidates.append(dict(**r,reason='Rebuildable interpreter/test/lint cache; untracked; outside protected source manifest',sha256=sha(r['path'])))
  elif (name.endswith(('.tmp','.temp','.bak','.swp','.swo')) or '.tmp.' in name or '.tmp_' in name):review.append(dict(**r,protected=excluded))
 dump('cache_deletion_candidates.json',candidates);dump('temporary_named_files_review_only.json',review)
 rawroot=ROOT/'experiments/mf_h1_local0/assets/shapenet-03001627'
 raw=[r for r in files if r['relative'].startswith(str(rawroot.relative_to(ROOT)).replace('\\','/'))]
 dataset_dirs=[dict(directory=k,**v) for k,v in tiers[4].items() if any(s in k.lower() for s in ('shapenet','downloaded_data','datasets','assets'))]
 extra={}
 for p in (rawroot/'split').glob('*'):
  if p.is_file():
   try:
    o=json.loads(p.read_text(encoding='utf8'));extra[p.name]=dict(type=type(o).__name__,length=len(o),keys=list(o)[:5] if isinstance(o,dict) else None)
   except Exception:extra[p.name]=dict(bytes=p.stat().st_size)
 summary=dict(root=str(ROOT),files=len(files),bytes=sum(r['bytes'] for r in files),logical_size_note='Sum of file lengths; hardlink copies may count more than once; junctions/symlinks not followed',seconds=time.time()-start,top_directories=[dict(directory=k,**v) for k,v in sorted(tiers[1].items(),key=lambda x:x[1]['bytes'],reverse=True)],extensions=dict(sorted(ext.items(),key=lambda x:x[1]['bytes'],reverse=True)),largest_files=sorted(files,key=lambda r:r['bytes'],reverse=True)[:35],cache_candidates=dict(files=len(candidates),bytes=sum(r['bytes'] for r in candidates)),other_temp_named=dict(files=len(review),bytes=sum(r['bytes'] for r in review)),dataset_dirs=dataset_dirs,chair_source_files=len(raw),chair_source_bytes=sum(r['bytes'] for r in raw),split_files=extra,skipped_links=skipped,errors=errors)
 dump('inventory_summary.json',summary)
 print(json.dumps({k:summary[k] for k in ('files','bytes','seconds','top_directories','cache_candidates','other_temp_named','chair_source_files','chair_source_bytes','split_files','skipped_links','errors')},ensure_ascii=False))
 print('LARGEST_FILES',json.dumps([dict(path=r['relative'],bytes=r['bytes']) for r in summary['largest_files'][:12]]))
 print('OTHER_TEMP',json.dumps(review[:25]))
