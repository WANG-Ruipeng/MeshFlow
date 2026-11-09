"""Exact static endpoint audit with safe broad phase and cached Fraction geometry.

All narrow-phase coordinates are exact rationals of the stored float64 values.
No geometric tolerance decides intersections. Shared vertices/edges are legal
only to the extent of the shared indexed simplex. Degeneracy is separate.
Not a formally verified kernel; independently cross-checked in verify.py.
"""
import numpy as np
from itertools import combinations
from .exact_triangles import point,normal,zero,cross,dot,sub,section,coplanar_clip,exceeds_shared_simplex,unique,add,mul

def intersect_cached(a,b,na,nb):
    line=cross(na,nb)
    if zero(line):
        if dot(na,sub(b[0],a[0]))!=0:return []
        return coplanar_clip(a,b,nb)
    sa=section(a,b[0],nb);sb=section(b,a[0],na)
    if not sa or not sb:return []
    k=next(i for i,x in enumerate(line) if x!=0)
    amin,amax=min(sa,key=lambda x:x[k]),max(sa,key=lambda x:x[k]);bmin,bmax=min(sb,key=lambda x:x[k]),max(sb,key=lambda x:x[k])
    lo=max(amin[k],bmin[k]);hi=min(amax[k],bmax[k])
    if lo>hi:return []
    if amin[k]!=amax[k]:p,q=amin,amax
    elif bmin[k]!=bmax[k]:p,q=bmin,bmax
    else:return [amin] if amin==bmin else []
    def interp(t):return add(p,mul(sub(q,p),(t-p[k])/(q[k]-p[k])))
    return unique([interp(lo),interp(hi)])

def audit(v,f):
    v=np.asarray(v,dtype=np.float64);f=np.asarray(f,dtype=np.int64)
    if not np.isfinite(v).all():raise ValueError('Nonfinite input')
    vv=[point(p) for p in v.tolist()];tt=[[vv[k] for k in a] for a in f];nn=[normal(t) for t in tt];ss=[set(a) for a in f]
    deg=[i for i,n in enumerate(nn) if zero(n) or len(ss[i])!=3];bad=set(deg)
    lo=v[f].min(1);hi=v[f].max(1)
    pairs=[];witnesses=[];broad=0;narrow=0;edge_skips=0
    for i,j in combinations(range(len(f)),2):
        if i in bad or j in bad:continue
        # min/max and comparisons on floats are EXACT for their represented values.
        if np.any(hi[i]<lo[j]) or np.any(hi[j]<lo[i]):continue
        broad+=1;shared=sorted(ss[i]&ss[j]);a,b=tt[i],tt[j];na,nb=nn[i],nn[j]
        if len(shared)==2:
            # Distinct noncoplanar triangle planes share exactly their indexed edge.
            extra=next(k for k in ss[j] if k not in ss[i])
            if dot(na,sub(vv[extra],a[0]))!=0:
                edge_skips+=1;continue
        if len(shared)==1:
            # If B touches the plane of A only at their shared vertex, no excess intersection.
            ds=[dot(na,sub(vv[k],a[0])) for k in ss[j] if k!=shared[0]]
            if ds[0]*ds[1]>0:continue
        narrow+=1
        pts=intersect_cached(a,b,na,nb)
        if exceeds_shared_simplex(pts,[vv[k] for k in shared]):
            pairs.append([i,j])
            if len(witnesses)<3:witnesses.append({'faces':[i,j],'points':[[str(c) for c in p] for p in pts]})
    return {'pair_count':len(pairs),'degenerate_faces':deg,'improper_intersection_pairs':pairs,'first_witnesses':witnesses,'pair_count_total':len(f)*(len(f)-1)//2,'aabb_candidate_pairs':broad,'exact_narrow_phase_pairs':narrow,'exact_non_coplanar_shared_edge_skips':edge_skips,'embedded_endpoint_pass':not deg and not pairs,'arithmetic':'Fraction exact for stored float64; safe min/max AABB only; no intersection tolerance'}
