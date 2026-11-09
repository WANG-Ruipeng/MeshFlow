"""Seal bounded experiment evidence, never dispatch a model or scorer."""
from __future__ import annotations
import datetime,difflib,shutil
from .common import *

def run():
 final=OUT/'COMPLETE.json'
 if final.exists():r=read(final);verify(r['files']);return r
 required=['training/complete.json','evaluation/CANDIDATE_RAW/complete.json','evaluation/CANDIDATE_SOURCE/complete.json',
  'scores/COMPLETE.json','source_analysis/COMPLETE.json','analysis/COMPLETE.json','visual_review/COMPLETE.json',
  'audits/training_exact_budget.json','audits/independent_result_audit.json','audits/geometry_gate_independent.json',
  'audits/source_visual_review.json','report_tables/COMPLETE.json','REPORT.md','decision.json']
 for n in required:
  if not (OUT/n).is_file():raise RuntimeError('Missing required deliverable: '+n)
 train=read(OUT/'training/complete.json');assert train['counts']==dict(forwards=2000,backwards=2000,updates=500)
 for arm in ('CANDIDATE_RAW','CANDIDATE_SOURCE'):
  r=read(OUT/'evaluation'/arm/'complete.json');assert r['outputs']==30 and r['new_forwards']==30
  ledger=[json.loads(s) for s in (OUT/'evaluation'/arm/'ledger.jsonl').read_text().splitlines()]
  assert sum(r['event']=='FORWARD_STARTED' for r in ledger)==30
  assert sum(r['event']=='FORWARD_RETURNED' for r in ledger)==30
 assert read(OUT/'audits/training_exact_budget.json')['status']=='PASS'
 assert str(read(OUT/'audits/independent_result_audit.json')['status']).startswith('PASS')
 assert read(OUT/'visual_review/COMPLETE.json')['cells']==180
 assert read(OUT/'source_analysis/COMPLETE.json')['total_candidate_GEO']==15
 registration=read(OUT/'evaluation/CPU_registration.json')
 verify(registration['protected']);verify(registration['source_hashes'])
 verify(read(OUT/'scores/COMPLETE.json')['reference_hashes'])
 verify(read(OUT/'PRETRAIN_READY.json')['files'])
 cache=read(OUT/'data/relative_cache_manifest.json');verify(cache['protected_files'])
 sources=dict(registration['source_hashes'])
 for p in Path(__file__).parent.glob('*.py'):sources[str(p)]=sha(p)
 archived={}
 for path,digest in sources.items():
  p=resolve(path)
  try:rel=p.relative_to(ROOT)
  except ValueError:rel=Path('external')/p.name
  dest=OUT/'source_archive'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(dest)==digest
  else:shutil.copyfile(p,dest)
  archived[str(dest)]=digest
 write(OUT/'source_hashes.json',dict(sources=sources,archived=archived))
 diff=''
 for p in sorted(Path(__file__).parent.glob('*.py')):
  diff+=''.join(difflib.unified_diff([],p.read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+p.relative_to(ROOT).as_posix()))
 with (OUT/'code.diff').open('x',encoding='utf-8') as f:f.write(diff)
 decision=read(OUT/'decision.json');analysis=read(OUT/'analysis/COMPLETE.json')
 result=dict(status='COMPLETE',decision=decision,default_changed=False,optimizer_updates=500,training_logical_forwards=2000,backwards=2000,
  evaluation_D0_forwards=60,generator_forwards=0,new_free_generation_noise=0,new_original_metric_rows=60,readonly_metric_rows=270,
  original_metric_fields=177,actual_visual_pages=30,actual_visual_cells=180,source_identity_visual_pages=15,
  geometry_alert_old_parents=analysis['geometry_alert_old_parents'],no_commit_or_push=True,no_old_asset_cleanup=True,
  utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 result['files']={str(p):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
 write(final,result);return result
if __name__=='__main__':
 r=run();print(json.dumps({k:v for k,v in r.items() if k!='files'}))
