"""Small exact-rational static triangle intersection auditor.

For each unordered face pair, tests whether its geometric intersection exceeds
its shared indexed simplex. Adjacent faces are NOT blindly skipped. All input
coordinates are converted to Fraction (binary floats via exact as_integer_ratio).
Degenerate triangles are reported separately, not assigned zero intersections.
This is an independently implemented research reference, not CGAL/CCD.
"""
from fractions import Fraction as Q
from itertools import combinations


def point(x):
    return tuple(Q(v) if not isinstance(v, Q) else v for v in x)

def add(a,b): return tuple(x+y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def mul(a,t): return tuple(x*t for x in a)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def normal(t): return cross(sub(t[1],t[0]),sub(t[2],t[0]))
def zero(v): return all(x==0 for x in v)
def unique(v): return list(dict.fromkeys(v))

def section(t, p, n):
    s=[dot(n,sub(a,p)) for a in t]
    out=[t[i] for i in range(3) if s[i]==0]
    for i,j in ((0,1),(1,2),(2,0)):
        if s[i]*s[j]<0:
            out.append(add(t[i],mul(sub(t[j],t[i]),s[i]/(s[i]-s[j]))))
    return unique(out)

def coplanar_clip(a,b,n):
    # Polygon clipping on B's oriented edge halfplanes, all in exact 3D.
    poly=list(a)
    for i in range(3):
        origin=b[i]; e=sub(b[(i+1)%3],origin)
        def side(p): return dot(cross(e,sub(p,origin)),n)
        if not poly: break
        new=[]
        for k in range(len(poly)):
            p,q=poly[k-1],poly[k]
            sp,sq=side(p),side(q)
            if (sp<0)!=(sq<0):
                new.append(add(p,mul(sub(q,p),sp/(sp-sq))))
            if sq>=0: new.append(q)
        poly=unique(new)
    return poly

def intersection(a,b):
    na,nb=normal(a),normal(b)
    if zero(na) or zero(nb):
        raise ValueError('Degenerate triangles need a separate validity result')
    # Exact axis-aligned interval rejection, no tolerance.
    for k in range(3):
        if max(p[k] for p in a)<min(p[k] for p in b) or max(p[k] for p in b)<min(p[k] for p in a):
            return []
    line=cross(na,nb)
    if zero(line):
        if dot(na,sub(b[0],a[0]))!=0: return []
        return coplanar_clip(a,b,nb)
    sa=section(a,b[0],nb); sb=section(b,a[0],na)
    if not sa or not sb: return []
    k=next(i for i,x in enumerate(line) if x!=0)
    amin,amax=min(sa,key=lambda x:x[k]),max(sa,key=lambda x:x[k])
    bmin,bmax=min(sb,key=lambda x:x[k]),max(sb,key=lambda x:x[k])
    lo=max(amin[k],bmin[k]); hi=min(amax[k],bmax[k])
    if lo>hi: return []
    if amin[k]!=amax[k]: p,q=amin,amax
    elif bmin[k]!=bmax[k]: p,q=bmin,bmax
    else: return [amin] if amin==bmin else []
    def interp(t): return add(p,mul(sub(q,p),(t-p[k])/(q[k]-p[k])))
    return unique([interp(lo),interp(hi)])

def on_segment(p,a,b):
    if a==b: return p==a
    if not zero(cross(sub(p,a),sub(b,a))): return False
    return dot(sub(p,a),sub(p,b))<=0

def exceeds_shared_simplex(points, shared):
    if not points: return False
    if not shared: return True
    if len(shared)==1: return any(p!=shared[0] for p in points)
    if len(shared)==2: return any(not on_segment(p,*shared) for p in points)
    # Repeated faces are invalid, even if they share all three indices.
    return True

def audit(vertices, faces):
    v=[point(p) for p in vertices]; fs=[tuple(map(int,f)) for f in faces]
    if any(len(f)!=3 for f in fs): raise ValueError('Only triangles supported')
    if any(i<0 or i>=len(v) for f in fs for i in f): raise ValueError('Face index out of bounds')
    deg=[i for i,f in enumerate(fs) if len(set(f))!=3 or zero(normal([v[j] for j in f]))]
    pairs=[]; witnesses=[]; checked=0
    for i,j in combinations(range(len(fs)),2):
        if i in deg or j in deg: continue
        checked+=1
        a,b=fs[i],fs[j]
        pts=intersection([v[k] for k in a],[v[k] for k in b])
        if exceeds_shared_simplex(pts,[v[k] for k in sorted(set(a)&set(b))]):
            pairs.append([i,j]); witnesses.append([[str(c) for c in p] for p in pts])
    return {'arithmetic':'fractions.Fraction, exact for represented inputs',
            'nondegenerate_pairs_checked':checked, 'degenerate_faces':deg,
            'improper_intersection_pairs':pairs, 'pair_count':len(pairs),
            'intersection_vertices':witnesses,
            'embedded_endpoint_pass':not deg and not pairs}
