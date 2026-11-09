"""Lossless CPU PNG->WebP cache; preserve every pixel of every existing page."""
import hashlib,json,os,time
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
from PIL import Image
import numpy as np
OUT=Path('/mnt/e/MeshFlow/native_t1/runs/chair_capped500_half_v1')
tick=time.perf_counter();rows=[]
for panel in ('OLD','FRESH'):
    latest=json.loads((OUT/'visuals'/('latest_'+panel+'.json')).read_text())
    m=json.loads(Path(latest['manifest']).read_text());assert m['status']=='COMPLETE'
    for page in m['parent_pages']:
        p=Path(page['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==page['sha256']
        im=Image.open(p).convert('RGB');q=OUT/'delivery/review/gallery'/panel/(p.stem+'.webp');q.parent.mkdir(parents=True,exist_ok=True)
        if not q.exists():im.save(q,format='WEBP',lossless=True,method=6)
        assert np.array_equal(np.asarray(im),np.asarray(Image.open(q).convert('RGB')))
        rows.append(dict(source=str(p),path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest(),decoded_RGB_bitwise_equal=True))
r=dict(status='PASS',pages=len(rows),bytes=sum(x['bytes'] for x in rows),seconds=time.perf_counter()-tick,records=rows,model_calls=0)
(OUT/'delivery/gallery_preparation.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='records'}))
