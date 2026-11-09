"""Same inherited exact narrow phase. Vectorize only min/max AABB rejection.
No intersection tolerance, no adjacency skip beyond exact inherited reasoning.
"""
import numpy as np
from functools import lru_cache
from .exact_triangles import point,normal,zero,dot,sub,exceeds_shared_simplex
from .audit_fast import intersect_cached

@lru_cache(maxsize=8)
def pair_indices(n):return np.triu_indices(n,1)

def audit(v,f):
    v=np.asarray(v,dtype=np.float64);f=np.asarray(f,dtype=np.int64)
    if not np.isfinite(v).all():raise ValueError('Nonfinite input')
    vv=[point(p) for p in v.tolist()];tt=[[vv[k] for k in a] for a in f];nn=[normal(t) for t in tt];ss=[set(a) for a in f]
    deg=[i for i,n in enumerate(nn) if zero(n) or len(ss[i])!=3];bad=set(deg)
    lo=v[f].min(1);hi=v[f].max(1);ii,jj=pair_indices(len(f))
    keep=np.all(hi[ii]>=lo[jj],axis=1)&np.all(hi[jj]>=lo[ii],axis=1)
    pairs=[];witnesses=[];broad=0;narrow=0;edge_skips=0
    for i,j in zip(ii[keep],jj[keep]):
        if i in bad or j in bad:continue
        broad+=1;shared=sorted(ss[i]&ss[j]);a,b=tt[i],tt[j];na,nb=nn[i],nn[j]
        if len(shared)==2:
            extra=next(k for k in ss[j] if k not in ss[i])
            if dot(na,sub(vv[extra],a[0]))!=0:edge_skips+=1;continue
        if len(shared)==1:
            ds=[dot(na,sub(vv[k],a[0])) for k in ss[j] if k!=shared[0]]
            if ds[0]*ds[1]>0:continue
        narrow+=1;pts=intersect_cached(a,b,na,nb)
        if exceeds_shared_simplex(pts,[vv[k] for k in shared]):
            pairs.append([int(i),int(j)])
            if len(witnesses)<3:witnesses.append({'faces':[int(i),int(j)],'points':[[str(c) for c in p] for p in pts]})
    return {'pair_count':len(pairs),'degenerate_faces':deg,'improper_intersection_pairs':pairs,'first_witnesses':witnesses,'pair_count_total':len(f)*(len(f)-1)//2,'aabb_candidate_pairs':broad,'exact_narrow_phase_pairs':narrow,'exact_non_coplanar_shared_edge_skips':edge_skips,'embedded_endpoint_pass':not deg and not pairs,'arithmetic':'inherited exact Fraction narrow phase; vectorized exact-comparison AABB broad phase; no geometric tolerance'}
