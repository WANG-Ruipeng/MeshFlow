"""Exact dyadic/rational triangle intersection audit on the EXPORTED OBJ.

AABB candidate rejection uses only ordering of original binary64 coordinates.
All subsequent predicates use arbitrary-size integers (common dyadic scale),
with Fraction for actual intersection vertices. Legal indexed common vertices
and edges are excluded; additional intersections and unindexed contacts count.
This is an independently written auditor, not CGAL and not a global theorem.
"""
from common import *
from fractions import Fraction as F
import time,sys

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def sign(x):return (x>0)-(x<0)
def to_exact(v):
    ratios=[[float(x).as_integer_ratio() for x in p] for p in v]
    e=max(d.bit_length()-1 for p in ratios for _,d in p)
    return [tuple(n*((1<<e)//d) for n,d in p) for p in ratios],e

def proj(p,ax):return tuple(p[j] for j in range(3) if j!=ax)
def inside2(p,t):
    s=[orient(t[j],t[(j+1)%3],p) for j in range(3)]
    return all(x>=0 for x in s) or all(x<=0 for x in s)
def onseg(p,a,b):
    if any(cross(sub(p,a),sub(b,a))):return False
    return dot(sub(p,a),sub(p,b))<=0

def coplanar_poly(A,B,axis):
    a=[tuple(F(x) for x in proj(p,axis)) for p in A]
    b=[proj(p,axis) for p in B];s=sign(orient(*b));out=a
    for j in range(3):
        c,d=b[j],b[(j+1)%3];inp=out;out=[]
        if not inp:break
        prev=inp[-1];pv=s*orient(c,d,prev)
        for cur in inp:
            cv=s*orient(c,d,cur)
            if (cv>=0)!=(pv>=0):
                t=pv/(pv-cv);out.append(tuple(x+t*(y-x) for x,y in zip(prev,cur)))
            if cv>=0:out.append(cur)
            prev=cur;pv=cv
    return out

def illegal_intersection(A,B,ia,ib):
    shared=set(map(int,ia)) & set(map(int,ib))
    if len(shared)==3:return True,'duplicate_face'
    nA=cross(sub(A[1],A[0]),sub(A[2],A[0]));nB=cross(sub(B[1],B[0]),sub(B[2],B[0]))
    if not any(nA) or not any(nB):return False,'degenerate_skipped'
    da=[dot(nB,sub(p,B[0])) for p in A];db=[dot(nA,sub(p,A[0])) for p in B]
    if all(x>0 for x in da) or all(x<0 for x in da) or all(x>0 for x in db) or all(x<0 for x in db):return False,'separated'
    common=[A[list(map(int,ia)).index(x)] for x in sorted(shared)]
    if all(x==0 for x in da):
        axis=max(range(3),key=lambda j:abs(nA[j]));poly=coplanar_poly(A,B,axis)
        if not poly:return False,'coplanar_separated'
        c=[proj(p,axis) for p in common]
        if not c:return True,'coplanar_unindexed_contact_or_overlap'
        if len(c)==1:return (any(p!=c[0] for p in poly),'coplanar_beyond_shared_vertex')
        def allowed(p):return orient(c[0],c[1],p)==0 and dot(sub(p,c[0]),sub(p,c[1]))<=0
        return (any(not allowed(p) for p in poly),'coplanar_beyond_shared_edge')
    if len(shared)==2:return False,'legal_nonplanar_shared_edge'
    pts=[]
    for T,U,d,n in [(A,B,da,nB),(B,A,db,nA)]:
        axis=max(range(3),key=lambda j:abs(n[j]));u=[proj(p,axis) for p in U]
        for j in range(3):
            x,y=T[j],T[(j+1)%3];sx,sy=d[j],d[(j+1)%3]
            if sx==0 and inside2(proj(x,axis),u):pts.append(x)
            if sx*sy<0:
                den=sy-sx;p=tuple(F(sy*a-sx*b,den) for a,b in zip(x,y))
                if inside2(proj(p,axis),u):pts.append(p)
    if not pts:return False,'no_overlap'
    if not common:return True,'nonplanar_unindexed_contact_or_crossing'
    return any(p!=common[0] for p in pts),'nonplanar_beyond_shared_vertex'

def exact_mesh(v,f):
    start=time.perf_counter();iv,e=to_exact(v);t=v[f];lo=t.min(1);hi=t.max(1)
    exactdeg=[i for i,tri in enumerate(f) if not any(cross(sub(iv[int(tri[1])],iv[int(tri[0])]),sub(iv[int(tri[2])],iv[int(tri[0])])))]
    pairs=[];categories=Counter();tested=0
    for i in range(len(f)):
        js=np.flatnonzero(np.all(lo[i]<=hi[i+1:],axis=1)&np.all(lo[i+1:]<=hi[i],axis=1))+i+1
        a=[iv[int(x)] for x in f[i]]
        for j in js:
            tested+=1;b=[iv[int(x)] for x in f[j]];bad,why=illegal_intersection(a,b,f[i],f[j])
            if bad:pairs.append([i,int(j),why]);categories[why]+=1
    fac=sorted(set(x for pair in pairs for x in pair[:2]))
    return dict(exact_intersection_pairs=len(pairs),exact_intersection_faces=len(fac),exact_intersection_face_ratio=len(fac)/len(f),
                intersection_pairs=pairs,intersection_face_ids=fac,intersection_categories=dict(categories),
                exact_degenerate_faces=len(exactdeg),exact_degenerate_face_ids=exactdeg,
                aabb_candidate_pairs=tested,dyadic_common_exponent=e,exact_audit_seconds=time.perf_counter()-start)

def selftest():
    tests=[]
    def check(name,v,f,expected):
        r=exact_mesh(np.array(v,float),np.array(f,int));got=r['exact_intersection_pairs']
        tests.append(dict(name=name,expected=expected,got=got,passed=got==expected));assert got==expected,(name,r)
    A=[[0,0,0],[2,0,0],[0,2,0]]
    check('coplanar_disjoint',A+[[3,0,0],[4,0,0],[3,1,0]],[[0,1,2],[3,4,5]],0)
    check('coplanar_overlap',A+[[.25,.25,0],[1.25,.25,0],[.25,1.25,0]],[[0,1,2],[3,4,5]],1)
    check('nonplanar_crossing',A+[[.5,.5,-1],[.5,.5,1],[1.5,.5,0]],[[0,1,2],[3,4,5]],1)
    check('legal_shared_edge_coplanar',A+[[0,-2,0]],[[0,1,2],[1,0,3]],0)
    check('illegal_shared_edge_coplanar',A+[[.5,.5,0]],[[0,1,2],[0,1,3]],1)
    check('legal_shared_edge_fold',A+[[0,0,2]],[[0,1,2],[1,0,3]],0)
    check('legal_shared_vertex',A+[[-2,0,0],[0,-2,0]],[[0,1,2],[0,3,4]],0)
    check('illegal_shared_vertex_overlap',A+[[1,1,-1],[1,1,1]],[[0,1,2],[0,3,4]],1)
    check('unindexed_vertex_contact',A+[[0,0,0],[-2,0,0],[0,-2,0]],[[0,1,2],[3,4,5]],1)
    check('dyadic_tiny_gap',A+[[0,0,2**-80],[2,0,2**-80],[0,2,2**-80]],[[0,1,2],[3,4,5]],0)
    check('duplicate_face',A,[[0,1,2],[2,1,0]],1)
    # Independent 2D polygon intersection oracle on random coplanar triangles.
    from shapely.geometry import Polygon
    rng=np.random.default_rng(178)
    for i in range(100):
        v2=rng.integers(-100,101,(6,2))
        if orient(*v2[:3])==0 or orient(*v2[3:])==0:continue
        expected=int(Polygon(v2[:3]).intersects(Polygon(v2[3:])))
        check('random_coplanar_'+str(i),np.c_[v2,np.zeros(6)],[[0,1,2],[3,4,5]],expected)
    dump(ROOT/'audit/EXACT_AUDITOR_SELFTEST.json',dict(tests=tests,passed=all(x['passed'] for x in tests),count=len(tests)))
    print('SELFTEST_PASS',len(tests),flush=True)

def main():
    selftest()
    paths=list(sorted((ROOT/'runs').glob('*/mesh.obj')))+list(sorted((ROOT/'inputs').glob('*/reference.obj')))
    for p in paths:
        print('EXACT_START',p.parent.name,p.name,flush=True);v,f=load_mesh(p);r=exact_mesh(v,f)
        r.update(source_sha256=sha(p),method='integer dyadic orientation + exact rational intersection; no epsilon predicates; legal shared indexed simplices excluded')
        target=p.parent/('exact_audit.json' if p.name=='mesh.obj' else 'reference_exact_audit.json');dump(target,r)
        print('EXACT_END',p.parent.name,r['exact_intersection_pairs'],r['exact_degenerate_faces'],round(r['exact_audit_seconds'],3),flush=True)
if __name__=='__main__':main()
