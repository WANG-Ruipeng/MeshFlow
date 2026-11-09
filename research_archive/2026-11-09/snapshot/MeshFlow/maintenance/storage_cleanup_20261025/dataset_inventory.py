"""Read-only dataset inventory. Does not move/delete datasets or import models."""
from pathlib import Path
import os, csv, json, hashlib, time

ROOT=Path('/mnt/e/MeshFlow')
ART=Path('/mnt/e/MeshFlow-Control-artifacts')
OUT=ROOT/'maintenance/storage_cleanup_20261025'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def win(p): return 'E:/'+str(p).split('/mnt/e/',1)[1]
def scan(p):
    n=b=0; ex={}; large=[]
    for base,ds,fs in os.walk(p,followlinks=False):
        ds[:]=[d for d in ds if d not in ('.git','__pycache__')]
        for fn in fs:
            q=Path(base)/fn
            if q.is_symlink():continue
            s=q.stat().st_size;n+=1;b+=s
            ext=q.suffix or '(no extension)';ex[ext]=ex.get(ext,0)+s
            if s>=20*1024**2:large.append({'path':win(q),'bytes':s})
    return {'path':win(p),'files':n,'bytes':b,'extension_bytes':ex,'large':large}
roots=[ROOT/'downloaded_data',ART/'data',ART/'training_stream']
for outer in [ROOT/'experiments',ART/'experiments']:
    for exp in outer.iterdir():
        if not exp.is_dir():continue
        for p in exp.iterdir():
            if p.is_dir() and (p.name in {'data','dataset','datasets','assets','input','inputs','cache','ot_cache','training_stream','targets'} or p.name.endswith('_cache')):
                roots.append(p)
rows=[scan(p) for p in roots if p.exists()]
rows.sort(key=lambda r:-r['bytes'])
(OUT/'dataset_root_sizes.json').write_text(json.dumps({'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'selected dataset/source/generated input/cache roots; not whole disk; all readonly','rows':rows},indent=2),encoding='utf8')
with (OUT/'dataset_root_sizes.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=['path','files','bytes']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
print(json.dumps({'roots':len(rows),'bytes':sum(r['bytes'] for r in rows),'top':[{k:r[k] for k in ['path','files','bytes']} for r in rows[:20]]},indent=2))

package=ROOT/'experiments/r2_single_step_denoiser_distillation_v1/review_package'
retained=package.parent
dups=[]; unique=[]
for p in package.rglob('*'):
    if not p.is_file():continue
    q=retained/p.relative_to(package)
    if q.is_file() and p.stat().st_size==q.stat().st_size:
        a=sha(p);b=sha(q)
        if a==b:
            dups.append({'path':win(p),'retained_path':win(q),'bytes':p.stat().st_size,'sha256':a});continue
    unique.append({'path':win(p),'bytes':p.stat().st_size})
(OUT/'dataset_duplicate_package_review.json').write_text(json.dumps({'status':'READ_ONLY_CANDIDATES','duplicate_files':dups,'unique_files_keep':unique,'duplicate_bytes':sum(x['bytes'] for x in dups),'important':'Only exact matching files are candidates, never delete whole review_package; parent must check provenance and approve whitelist.'},indent=2),encoding='utf8')
print(json.dumps({'package_exact_duplicates':len(dups),'bytes':sum(x['bytes'] for x in dups),'unique_keep':len(unique)}))
