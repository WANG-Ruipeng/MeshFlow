"""Read existing frozen manifests only; no array generation or model calls."""
from pathlib import Path
import json, hashlib, collections, csv
R=Path('/mnt/e/MeshFlow'); O=R/'maintenance/storage_cleanup_20261025'
sources=[
'experiments/r2_bench_full_seat_curriculum_v1/data/manifest.json',
'experiments/r2_bench_seat_coverage_v1/data/manifest.json',
'experiments/r2_bench_geo_only_acceptance_v1/manifest.json',
'experiments/r2_bench_backend_shape_preservation_v1/data/manifest.json',
'experiments/r2_bench_backend_relative_deformation_v1/data/relative_cache_manifest.json',
'experiments/r2_bench_raw_local_repair_v1/data/manifest.json',
'r2/maintenance/bench_reference_20261023/retained_assets.json']
refs=collections.defaultdict(set); manifests=[]
def walk(x,src):
 if isinstance(x,dict):
  for k,v in x.items():walk(k,src);walk(v,src)
 elif isinstance(x,list):
  for v in x:walk(v,src)
 elif isinstance(x,str) and len(x)<1200:
  s=x.replace('\\','/')
  if s.startswith('E:/'):s='/mnt/e/'+s[3:]
  elif s.startswith(('experiments/','r2/','native_t1/','downloaded_data/')):s=str(R/s)
  if s.startswith('/mnt/e/') and '\n' not in s:refs[s].add(src)
for rel in sources:
 p=R/rel
 if not p.exists():manifests.append({'path':rel,'status':'MISSING'});continue
 b=p.read_bytes();d=json.loads(b.decode('utf-8-sig'));walk(d,rel)
 manifests.append({'path':rel,'sha256':hashlib.sha256(b).hexdigest(),'status':'READ'})
rows=[]
for s,srcs in sorted(refs.items()):
 p=Path(s)
 if p.suffix.lower() in {'.pt','.pth','.safetensors'}:continue
 try:exists=p.exists();is_file=p.is_file();size=p.stat().st_size if is_file else None
 except OSError:continue
 rows.append({'path':s,'exists':exists,'kind':'file' if is_file else ('directory' if exists else 'missing'),'bytes':size,'referenced_by':sorted(srcs)})
counts=collections.Counter()
for r in rows:
 s=r['path'].split('/experiments/',1)
 group=s[1].split('/',1)[0] if len(s)==2 else 'other'
 counts[group]+=1
result={'status':'READ_ONLY','scope':'Direct file references recursively extracted from selected current frozen manifests; not a claim of exhaustive project dependency closure','manifests':manifests,'nonweight_references':rows,'references_by_experiment':dict(counts),'action':'PROTECT references, source originals, masks, C/Y, fixed noise/query, RAW/POST, original evidence. No deletion performed.'}
(O/'dataset_active_dependencies.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'manifests':len(manifests),'nonweight_refs':len(rows),'missing':sum(not r['exists'] for r in rows),'groups':dict(counts)},indent=2))
