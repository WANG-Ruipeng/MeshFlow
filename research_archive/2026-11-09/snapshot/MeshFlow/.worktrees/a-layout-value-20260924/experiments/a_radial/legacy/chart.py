"""A sphere chart constructed from topology, and guarded tangent retraction.
No target geometry is used. Safe paths have positive oriented face determinants.
This is floating-point control, NOT formally certified continuous collision detection.
"""
import numpy as np
from scipy.optimize import linprog

def dets(u,f):
    a,b,c=u[f[:,0]],u[f[:,1]],u[f[:,2]]
    return np.einsum('ij,ij->i',a,np.cross(b,c))

def solid_angles(u,f):
    a,b,c=u[f[:,0]],u[f[:,1]],u[f[:,2]]
    d=dets(u,f);den=1+(a*b).sum(1)+(b*c).sum(1)+(c*a).sum(1)
    return 2*np.arctan2(d,den)

def coeff(u,d,f):
    a,b,c=u[f[:,0]],u[f[:,1]],u[f[:,2]];da,db,dc=d[f[:,0]],d[f[:,1]],d[f[:,2]]
    dt=lambda x,y,z:np.einsum('ij,ij->i',x,np.cross(y,z))
    return (dt(a,b,c),dt(da,b,c)+dt(a,db,c)+dt(a,b,dc),dt(da,db,c)+dt(da,b,dc)+dt(a,db,dc),dt(da,db,dc))

def retract(u,delta,f,margin=0.05):
    """Whole straight pre-normalization path, using Bernstein convex-hull lower bound."""
    d=delta-(delta*u).sum(1,keepdims=True)*u
    c0,c1,c2,c3=coeff(u,d,f)
    if c0.min()<=0:raise ValueError('invalid chart entering guard')
    alpha=1.
    for _ in range(64):
        # On [0,alpha], cubic Bernstein coefficients bound polynomial from below.
        p1=alpha*c1;p2=alpha*alpha*c2;p3=alpha**3*c3
        b1=c0+p1/3;b2=c0+2*p1/3+p2/3;b3=c0+p1+p2+p3
        if np.all(np.minimum(np.minimum(b1,b2),b3)>margin*c0):break
        alpha*=.5
    if alpha<2**-60:alpha=0.
    v=u+alpha*d;v/=np.linalg.norm(v,axis=1,keepdims=True)
    assert dets(v,f).min()>0
    return v,alpha

def initial_chart(f):
    """Tutte disk + piecewise linear bowl, center in strict visibility kernel.
    For rare numerical non-kernel cases, return a documented exception; no resampling.
    """
    n=int(f.max())+1;boundary=f[0].copy();interior=np.array([i for i in range(n) if i not in boundary]);A=np.zeros((n,n))
    for tri in f:
        for a,b in zip(tri,np.roll(tri,-1)):A[a,b]=A[b,a]=1
    L=np.diag(A.sum(1))-A
    xy=np.zeros((n,2));th=2*np.pi*np.arange(3)/3;xy[boundary]=np.c_[np.cos(th),np.sin(th)]
    xy[interior]=np.linalg.solve(L[np.ix_(interior,interior)],-L[np.ix_(interior,boundary)]@xy[boundary])
    # Maxwell-Cremona lifting: positive unit stress on internal edges.
    # Adjacent face gradients differ by the right normal of their shared edge.
    disk=[list(t) for t in f if set(t)!=set(boundary)]
    for t in disk:
        aa,bb,cc=xy[t];area=np.linalg.det(np.stack((bb-aa,cc-aa)))
        if area<0:t[1],t[2]=t[2],t[1]
    edgeface={}
    for j,t in enumerate(disk):
        for a,b in zip(t,np.roll(t,-1)):edgeface[(int(a),int(b))]=j
    planes=np.full((len(disk),3),np.nan);planes[0]=0;todo=[0]
    residual=0.
    for j in todo:
        t=disk[j]
        for a,b in zip(t,np.roll(t,-1)):
            other=edgeface.get((int(b),int(a)))
            if other is None:continue
            e=xy[b]-xy[a];dg=np.array([e[1],-e[0]])
            pp=planes[j]+np.r_[dg,-dg@xy[a]]
            if np.isnan(planes[other,0]):planes[other]=pp;todo.append(other)
            else:residual=max(residual,float(np.max(np.abs(pp-planes[other]))))
    assert len(todo)==len(disk)
    z=np.zeros(n);ct=np.zeros(n)
    for t,p in zip(disk,planes):
        for v in t:z[v]+=xy[v]@p[:2]+p[2];ct[v]+=1
    z/=ct
    affine=np.linalg.solve(np.c_[xy[boundary],np.ones(3)],z[boundary])
    z-=np.c_[xy,np.ones(n)]@affine;z[boundary]=0
    if z.min()>=-1e-14:raise ValueError('lifting is not lower convex cap')
    z/=abs(z.min());x=np.c_[xy,z]
    # Orient faces coherently as seen from a point on the inside; sphere may be globally reversed.
    ff=f.copy();vol=np.einsum('ij,ij->i',x[ff[:,0]],np.cross(x[ff[:,1]],x[ff[:,2]])).sum()/6
    if vol<0:ff=ff[:,[0,2,1]]
    a,b,c=x[ff[:,0]],x[ff[:,1]],x[ff[:,2]];norm=np.cross(b-a,c-a);norm/=np.linalg.norm(norm,axis=1,keepdims=True)
    res=linprog([0,0,0,-1],A_ub=np.c_[norm,np.ones(len(ff))],b_ub=(norm*a).sum(1),bounds=[(None,None)]*3+[(0,None)],method='highs')
    if not res.success or res.x[3]<1e-12:raise ValueError('Tutte cap visibility kernel empty or unresolved')
    center=res.x[:3];u=x-center;u/=np.linalg.norm(u,axis=1,keepdims=True)
    if dets(u,ff).min()<=0:raise ValueError('bad spherical fan')
    if abs(solid_angles(u,ff).sum()-4*np.pi)>1e-6:raise ValueError('not degree-one chart')
    return u,ff,dict(kernel_margin=float(res.x[3]),initial_det_min=float(dets(u,ff).min()))

def regularize(f,steps=60):
    u,f,info=initial_chart(f);n=len(u);incident=[]
    for v in range(n):
        nb=[]
        for t in f:
            if v in t:
                k=list(t).index(v);nb.append([t[(k+1)%3],t[(k+2)%3]])
        incident.append(np.array(nb))
    hist=[]
    for it in range(steps):
        for v in range(n):
            nb=incident[v];norm=np.cross(u[nb[:,0]],u[nb[:,1]])
            d=norm@u[v]
            if np.min(d)<=0:raise ValueError('coordinate chart lost sign')
            anchor=np.eye(3)[np.argmin(abs(u[v]))];b1=np.cross(u[v],anchor);b1/=np.linalg.norm(b1);b2=np.cross(u[v],b1);B=np.stack([b1,b2],axis=1)
            a=(norm@B)/d[:,None];H=a.T@a+len(d)*np.eye(2);g=a.sum(0)
            t=np.linalg.solve(H,g);t*=min(1.,1/max(np.linalg.norm(t),1e-20));delta=B@t
            old=np.log(d).sum();alpha=1.
            for j in range(30):
                uv=u[v]+alpha*delta;uv/=np.linalg.norm(uv);newd=norm@uv
                if np.min(newd)>0 and np.log(newd).sum()>old+1e-13:break
                alpha*=.5
            else:continue
            u[v]=uv
        if it in [0,4,9,19,39,59]:hist.append([it,float(-np.log(dets(u,f)).mean()),float(dets(u,f).min())])
    sa=solid_angles(u,f);info.update(steps=it+1,det_min=float(dets(u,f).min()),area_min=float(sa.min()),area_max=float(sa.max()),area_cv=float(sa.std()/sa.mean()),solid_angle_sum=float(sa.sum()),trace=hist)
    return u,f,info
