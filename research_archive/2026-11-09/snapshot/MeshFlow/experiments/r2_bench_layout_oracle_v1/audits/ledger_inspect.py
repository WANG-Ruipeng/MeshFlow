from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
def outline(o):
 if isinstance(o,dict):
  return {k:(f'dict[{len(v)}]' if isinstance(v,dict) and len(v)>15 else f'list[{len(v)}]' if isinstance(v,list) and len(v)>5 else v) for k,v in o.items() if k not in ['source_hashes','files','references','protected_sources','per_request']}
 return o
paths=[R/'registration.json',R/'audits/training_completion.json',R/'results/completion.json',R/'dispatch/PLAIN_go.json']
paths += list((R/'evaluation').glob('*/*complete_full.json'))
paths += list((R/'evaluation').glob('*/*resource_observer*/resource_observation.json'))
paths += list((R/'evaluation').glob('*/*attempt*/dispatch.json'))
paths += list((R/'evaluation').glob('*/*attempt*/resources.json'))
paths += list((R/'analysis').glob('*/summary.json'))
for p in paths:
 print('\nFILE',p.relative_to(R)); print(json.dumps(outline(json.loads(p.read_text(encoding='utf-8-sig'))),ensure_ascii=False))
for pattern in ['training/*/*attempt*/resources*.json','evaluation/*/sampling_attempt*/resources_before.json','evaluation/*/postprocess_attempt*/resources_before.json']:
 for p in R.glob(pattern):
  print('\nRESOURCE',p.relative_to(R)); print(json.dumps(outline(json.loads(p.read_text())),ensure_ascii=False))
