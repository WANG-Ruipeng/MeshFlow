"""All-candidate free-only pair distances; fixed Q_F2048, no alignment."""
from collections import defaultdict
from itertools import combinations
import numpy as np
from .io import *
from .surface import surface_distance_squared
def diversity(rows,out):
    groups=defaultdict(list);pairs=[];summary=[]
    for r in rows:
        groups[(r['label'],r['panel'],r['condition_id'])].append(r)
    for (label,panel,cid),rr in sorted(groups.items()):
        lookup={r['candidate']:r for r in rr if r.get('status')=='PASS'};expected=(0,1,4,5) if panel=='P1' else (0,1,2,3)
        distances=[]
        for a,b in combinations(expected,2):
            x,y=lookup.get(a),lookup.get(b);value=None
            if x and y:
                qa=arrays(x['quality_vectors_path'])['Q_F2048'];qb=arrays(y['quality_vectors_path'])['Q_F2048']
                xa=arrays(x['raw_path']);yb=arrays(y['raw_path'])
                fa=xa.get('raw',xa.get('output')).reshape(-1,3,3)[x['K']:]
                fb=yb.get('raw',yb.get('output')).reshape(-1,3,3)[y['K']:]
                if x['L0']!=y['L0']:raise ValueError('Condition scale differs')
                value=float((surface_distance_squared(qa,fb).mean()+surface_distance_squared(qb,fa).mean())/(2*x['L0']**2));distances.append(value)
            pairs.append(dict(label=label,panel=panel,condition_id=cid,candidate_a=a,candidate_b=b,D_free_pair=value,
                status='PASS' if value is not None else 'MISSING_CANDIDATE',distance='symmetric mean squared point-to-finite-triangle distance/L^2'))
        summary.append(dict(label=label,panel=panel,condition_id=cid,expected_pairs=6,available_pairs=len(distances),
            D_free_mean=float(np.mean(distances)) if len(distances)==6 else None,status='PASS' if len(distances)==6 else 'INCOMPLETE',
            direction='variation only, not larger-is-better',not_original_TMD=True))
    write_csv(resolve(out)/'diversity_pairs.csv',pairs);write_csv(resolve(out)/'diversity_per_condition.csv',summary)
    return summary
