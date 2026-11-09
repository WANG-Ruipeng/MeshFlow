"""Package the completed review, checking frozen inputs and archive CRCs."""
import hashlib
import json
import zipfile
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
for name,digest in protocol['source_cache_hashes'].items():
    assert sha(root/name)==digest,name
required=['REPORT.md','index.html','per_output.csv','summary_all_metrics.csv','paired_per_noise.csv','paired_per_parent.csv','metric_catalog.csv','metric_definitions.json','independent_validation.json']
for name in required:
    assert (out/name).is_file(),name
files=sorted(p for p in out.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='package_manifest.json')
manifest={'status':'COMPLETE_STOPPED','original_inputs_unchanged':True,'model_calls':0,'new_geometry_scoring':0,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
(out/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=out/'HINGE_BASELINE_METRIC_REVIEW.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in files+[out/'package_manifest.json']:
        z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1}))
