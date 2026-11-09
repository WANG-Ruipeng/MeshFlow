from pathlib import Path
import json,csv,statistics,difflib,hashlib
r=Path('/mnt/e/MeshFlow-Control-artifacts/followups/readout_precision_path_v1');repo=Path('/mnt/e/MeshFlow-Control')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
old=read(r/'source_before_manifest.json')['files'];prior={x['path']:x['sha256'] for x in old}
current={str(p.relative_to(repo)):p for p in repo.rglob('*') if p.is_file() and not any(x in ('.git','__pycache__','.pytest_cache') for x in p.relative_to(repo).parts) and p.suffix not in ('.pyc','.pyo')}
changes=[];diff=[];hashes=[]
for rel,p in sorted(current.items()):
    h=hashlib.sha256(p.read_bytes()).hexdigest();status='ADDED' if rel not in prior else 'UNCHANGED' if h==prior[rel] else 'MODIFIED'
    hashes.append(dict(path=rel,sha256=h,before_sha256=prior.get(rel,''),status=status))
    if status!='UNCHANGED':
        a=(r/'source_before'/rel).read_text(encoding='utf-8-sig').splitlines(True) if rel in prior else []
        b=p.read_text(encoding='utf-8-sig').splitlines(True)
        diff.extend(difflib.unified_diff(a,b,fromfile='a/'+rel if a else '/dev/null',tofile='b/'+rel));changes.append(dict(path=rel,status=status))
assert {x['path'] for x in changes if x['status']=='MODIFIED'}=={'src/meshflow_control/models/native.py','src/meshflow_control/models/readout.py'}
(r/'code.diff').write_text(''.join(diff),encoding='utf-8')
with (r/'source_hashes.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(hashes[0]));w.writeheader();w.writerows(hashes)
for name in ['fixed_state_pairs.csv','layer_precision_metrics.csv','prediction_precision_interactions.csv']:(r/name).write_bytes((r/'analysis'/name).read_bytes())
rows=read(r/'outputs/manifest.json')['rows'];cost=[]
for x in rows:
    if x['generation_status']=='VERIFIED_CACHE':continue
    d=read(Path(x['result_path']));cost.append(dict(label=x['model_label'],request=x['request_id'],precision=x['precision'],readout=x['readout'],seconds=d['seconds'],forward_seconds=d['model_forward_seconds'],allocated=d['cuda_peak_allocated_bytes'],reserved=d['cuda_peak_reserved_bytes']))
agg=[]
for key in [('LEGACY_BF16','OFF'),('FP32_REFERENCE','ON'),('FP32_REFERENCE','OFF')]:
    rr=[x for x in cost if (x['precision'],x['readout'])==key]
    agg.append(dict(precision=key[0],readout=key[1],n=len(rr),wall_mean=statistics.mean(x['seconds'] for x in rr),forward_mean=statistics.mean(x['forward_seconds'] for x in rr),allocated_max=max(x['allocated'] for x in rr),reserved_max=max(x['reserved'] for x in rr)))
(r/'cost_summary.json').write_text(json.dumps(dict(per_output=cost,by_path=agg,scope='Actual instrumented generation with per-step journaling. No extra performance run; stage-end peak reset caveat in independent verification.'),indent=2)+'\n')
print(json.dumps(dict(source_count=len(hashes),changes=changes,costs=agg)))