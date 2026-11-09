"""Exact safe-prefix quadratic edge floor; no monotonicity assumption."""
from __future__ import annotations
import numpy as np

EDGES=((0,1),(1,2),(2,0))

def lengths(tri):
    x=np.asarray(tri,np.float64)
    return np.stack([np.linalg.norm(x[...,a,:]-x[...,b,:],axis=-1) for a,b in EDGES],axis=-1)

def first_entry(e0,de,threshold):
    """First alpha where norm(e0+alpha*de) enters the open forbidden ball.

    A double root merely touches the sphere and imposes no restriction.
    The returned cap defines the whole safe prefix [0,cap], not just its end.
    """
    e0,de=np.asarray(e0,np.float64),np.asarray(de,np.float64)
    a=float(de@de);b=float(2*(e0@de));c=float(e0@e0-threshold*threshold)
    if c < 0:raise ValueError('Path begins below the protected edge floor')
    if a==0:return 1.0,dict(a=a,b=b,c=c,kind='constant',roots=[])
    disc=b*b-4*a*c
    if disc<=0:return 1.0,dict(a=a,b=b,c=c,kind='no_open_negative_interval',roots=[])
    s=float(np.sqrt(disc));q=-.5*(b+np.copysign(s,b))
    roots=sorted((q/a,c/q)) if q else [(-b-s)/(2*a),(-b+s)/(2*a)]
    lo,hi=roots
    cap=max(0.,lo) if hi>0 and lo<1 and hi>max(0.,lo) else 1.
    return min(1.,cap),dict(a=a,b=b,c=c,kind='quadratic',roots=roots)

def protect(raw,proposal,anchors,C_vertices,K,radius):
    if raw.dtype!=np.float32 or proposal.dtype!=np.float32 or raw.shape!=proposal.shape:
        raise ValueError('Existing same-domain FP32 RAW and D0 proposal required')
    fixed=anchors>=0;pre=proposal.copy();pre[fixed]=C_vertices[anchors[fixed]]
    if pre[:K].tobytes()!=raw[:K].tobytes():raise ValueError('Hard restored C differs from RAW')
    rawlen=lengths(raw);proplen=lengths(pre);floor=1.25*radius
    trigger=(rawlen.min(1)>=1.5*radius)&(proplen.min(1)<floor);trigger[:K]=False
    protected=(rawlen.max(1)>3*radius)&(rawlen.min(1)>radius);protected[:K]=False
    guarded=pre.copy();records=[]
    for i in range(len(raw)):
        alpha=1.;cap=1.;roots=[];roundoff=0
        if trigger[i]:
            x=raw[i].astype(float);d=pre[i].astype(float)-x
            for a,b in EDGES:
                edgecap,rr=first_entry(x[a]-x[b],d[a]-d[b],floor);roots.append(rr);cap=min(cap,edgecap)
            # Quantization can place the rounded FP32 endpoint just inside the
            # sphere. Move at most a few FP32 alpha ULPs toward RAW, never seek
            # a later feasible interval. This is not a monotonic binary search.
            aa=np.float32(cap)
            if float(aa)>cap:aa=np.nextafter(aa,np.float32(0))
            for roundoff in range(257):
                trial=(x+float(aa)*d).astype(np.float32);trial[fixed[i]]=C_vertices[anchors[i][fixed[i]]]
                if float(lengths(trial).min())>=floor:break
                aa=np.nextafter(aa,np.float32(0))
            else:raise ArithmeticError('Cannot quantize analytic safe-prefix endpoint safely')
            alpha=float(aa);guarded[i]=trial
        records.append(dict(raw_slot=i,known=i<K,floor_triggered=bool(trigger[i]),anti_collapse_protected=bool(protected[i]),
            raw_min_edge=float(rawlen[i].min()),raw_max_edge=float(rawlen[i].max()),proposal_min_edge=float(proplen[i].min()),
            guarded_min_edge=float(lengths(guarded[i]).min()),alpha_exact_prefix=cap,alpha_written=alpha,
            quantization_inward_steps=roundoff,quadratics=roots,
            max_guard_minus_raw=float(np.linalg.norm(guarded[i].astype(float)-raw[i],axis=1).max()),
            max_guard_minus_proposal=float(np.linalg.norm(guarded[i].astype(float)-pre[i],axis=1).max())))
    if guarded[fixed].tobytes()!=pre[fixed].tobytes():raise AssertionError('Hard anchors moved')
    return guarded,protected,records,pre
