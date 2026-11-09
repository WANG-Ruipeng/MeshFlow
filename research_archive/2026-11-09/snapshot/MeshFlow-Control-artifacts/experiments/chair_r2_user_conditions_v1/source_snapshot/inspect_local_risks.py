from pathlib import Path
import json
import numpy as np
from prepare import DEFAULT_OUT,read,geometry
from quality import partition_condition
out=DEFAULT_OUT
parents={r['uid']:r for r in read(out/'parent_inventory.json')['rows']}
for uid in parents:
 if uid.startswith('3ede'):
  c=read(out/'candidates'/f'{uid}.json'); ids=next(r['source_face_ids'] for r in c['candidates'] if r['candidate_id']=='p01')
  q=partition_condition(uid,ids,out/'quality'); m=geometry(parents[uid])['full_target'].reshape(-1,3,3)
  print(json.dumps({'uid':uid,'seat_ids':ids,'seat_bounds':[m[ids].reshape(-1,3).min(0).tolist(),m[ids].reshape(-1,3).max(0).tolist()]}))
  for pair in q['partitions']['CF']['problem_pairs']: print(json.dumps(pair))
