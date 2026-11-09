"""Transcribe the ten actually displayed six-column pages; no automatic rating."""
import datetime,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
observations=read(OUT/'audits/first10_actual_observations.json')
tasks=read(ROOT/'experiments/r2_bench_geo_only_acceptance_v1/manifest.json')['tasks']
axes=['overall_shape','C_use','support_retention','visible_fragment_or_gap','layering_or_fold']
now=datetime.datetime.now(datetime.timezone.utc).isoformat();rows=[]
for page in observations['pages']:
 t=tasks[page['page']];rid=t['request_id'];receipt=read(OUT/'gallery'/rid/'complete.json')
 assert receipt['image_sha256']==sha(OUT/'gallery'/rid/'comparison.png')
 assert len(receipt['cells'])==len(page['ratings'])==len(page['notes'])==6
 for cell,rating,note in zip(receipt['cells'],page['ratings'],page['notes']):
  r={k:receipt[k] for k in ('request_id','parent_id','condition_id','noise_index','eval_group')}
  r.update({k:cell[k] for k in ('cell_id','column','label','stage')})
  r.update(image_path=receipt['path'],image_sha256=receipt['image_sha256'],reviewer='AI_PROXY_backend_scope_audit',
   reviewer_kind='AI_PROXY',blind=False,actually_viewed=True,actual_view_source='functions.view_image',viewed_at_utc=now,
   page_index=page['page'],metric_values_consulted_for_labels=False,
   review_time_semantics='Time of transcription after all ten pages were actually displayed, not a claimed per-tool timestamp.',
   image_display='4494x1932 source displayed by view_image at 2048x880; two fixed oblique views and two original central sections reviewed.',
   note=note,**dict(zip(axes,rating)))
  rows.append(r)
assert len(rows)==60 and len({r['cell_id'] for r in rows})==60
dest=OUT/'gallery/review_shards/backend_scope_audit.json';dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x',encoding='utf-8') as f:json.dump(dict(status='ACTUALLY_REVIEWED_AI_PROXY',pages=10,cells=60,
 reviewer='AI_PROXY_backend_scope_audit',disclosure='Nonblind AI proxy review; not human certification. Historical labels were not copied; small unresolved slits/folds remain uncertain.',
 observations_sha256=sha(OUT/'audits/first10_actual_observations.json'),rows=rows),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(path=str(dest),pages=10,cells=60,sha256=sha(dest))))
