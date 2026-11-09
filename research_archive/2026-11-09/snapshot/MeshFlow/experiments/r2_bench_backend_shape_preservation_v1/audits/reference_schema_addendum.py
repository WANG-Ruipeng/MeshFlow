"""Correct audit-only original-score schema assumption; preserve failed audit."""
import csv,hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(v):
    if v is None or isinstance(v,bool) or v in ('','True','False'):return None
    try:x=float(v)
    except (TypeError,ValueError):return None
    return x if math.isfinite(x) else None
oldpath=OUT/'audits/independent_artifact_audit.json';old=read(oldpath)
remaining=[e for e in old['errors'] if not e.startswith('missing original field ')]
refs=list(csv.DictReader((OUT/'scores/original_reference_scores.csv').open(encoding='utf-8-sig',newline='')))
complete=read(OUT/'scores/COMPLETE.json');fields=complete['metric_fields'];cells=0;source_paths={}
for r in refs:
    p=Path(r['source_score_path']).parent/'row.json' if r.get('source_score_path') else OUT.parent/'r2_bench_geo_only_acceptance_v1'/'scores'/r['request_id']/'row.json'
    source=read(p);source_paths[str(p)]=sha(p)
    for field in fields:
        if field not in source:remaining.append('missing original row field '+str(p)+' '+field);continue
        a,b=num(r.get(field)),num(source[field]);cells+=1
        if not(a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)):
            remaining.append('original numeric mismatch '+r['request_id']+' '+field)
result=dict(status='PASS' if not remaining and cells==210*177 else 'FAIL',
    previous_audit_path=str(oldpath),previous_audit_sha256=sha(oldpath),
    correction='Previous audit treated nested original score.json as flat. This addendum checks all177 values against the original independently stored row.json; these files were included in the completed2753-file hash verification.',
    discarded_schema_only_errors=len(old['errors'])-len([e for e in old['errors'] if not e.startswith('missing original field ')]),
    source_rows=210,reference_numeric_cells_checked=cells,original_row_hashes=source_paths,
    preserved_checks={k:old[k] for k in ('training_event_counts','eval_counts','loss_updates','training_instances','sample_kinds','aggregate_numeric_and_denominator_cells_checked','pair_cells_checked','hash_files_checked','hash_receipts_checked','checks')},
    scientific_engineering_failures=1,zero_replay_resume=True,audit_harness_total_attempts=4,
    audit_harness_failure_scope='CSV BOM, missing GEO_ONLY source_score_path, nested original score schema; no scientific model/geometry/score reruns.',
    errors=remaining,model_calls=0,scoring_calls=0,large_weight_hash_repeats=0)
dest=OUT/'audits/independent_audit_final.json'
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(status=result['status'],reference_numeric_cells=cells,hash_files=old['hash_files_checked'],remaining_errors=remaining[:10],error_count=len(remaining))))
if result['status']!='PASS':raise SystemExit(1)
