"""Exact safe-prefix quadratic edge floor; no monotonicity assumption."""
from __future__ import annotations
import numpy as np

EDGES=((0,1),(1,2),(2,0))

def quantized_prefix(x,d,cap,floor,fixed,anchor_values):
    """Largest visited FP32 safe endpoint, with exact rounding-event fallback.

    The historical 257 endpoint checks stay byte-identical. If they are
    exhausted, skip constant-coordinate intervals using FP32 bin midpoints.
    This walks backward within the analytic safe prefix, never bisects a
    presumed monotone predicate and never changes the edge threshold.
    """
    aa=np.float32(cap)
    if float(aa)>cap:aa=np.nextafter(aa,np.float32(0))
    initial=aa
    def trial_at(a):
        t=(x+float(a)*d).astype(np.float32);t[fixed]=anchor_values
        return t
    for roundoff in range(257):
        trial=trial_at(aa)
        if float(lengths(trial).min())>=floor:return aa,trial,roundoff,None
        aa=np.nextafter(aa,np.float32(0))
    failed_alpha=float(aa);failed_min=float(lengths(trial_at(aa)).min());events=0
    movable=np.broadcast_to(~fixed[:,None],d.shape)&(d!=0)
    while events<100000:
        trial=trial_at(aa)
        if float(lengths(trial).min())>=floor:
            total=int(initial.view(np.uint32))-int(aa.view(np.uint32))
            return aa,trial,total,dict(method='FP32_coordinate_rounding_events',
                initial_alpha=float(initial),alpha_after_257_ulp_retreat=failed_alpha,
                min_edge_after_257_ulp_retreat=failed_min,floor=floor,
                below_floor_difference=failed_min-floor,coordinate_events=events,
                total_alpha_ULPs=total,final_alpha=float(aa),final_min_edge=float(lengths(trial).min()))
        # Moving alpha toward zero crosses the next lower/upper FP32 bin
        # boundary according to the sign of d. All intervening alpha values
        # have the identical written triangle and can be skipped exactly.
        direction=np.where(d>0,-np.inf,np.inf).astype(np.float32)
        neighbor=np.nextafter(trial,direction)
        midpoint=(trial.astype(np.float64)+neighbor.astype(np.float64))*.5
        critical=np.full(d.shape,-np.inf,np.float64)
        np.divide(midpoint-x,d,out=critical,where=movable)
        valid=movable&(critical>=0)&(critical<=float(aa)+8*np.spacing(float(aa)))
        if valid.any():
            boundary=min(float(aa),float(critical[valid].max()))
            next_a=np.float32(boundary)
            if float(next_a)>boundary:next_a=np.nextafter(next_a,np.float32(0))
            if next_a>=aa:next_a=np.nextafter(aa,np.float32(0))
        else:next_a=np.float32(0)
        # A tie-to-even midpoint or float64 cancellation can retain the old
        # coordinate at the event itself. The next alpha float leaves it.
        aa=next_a;events+=1
        if aa==0:
            trial=trial_at(aa)
            if float(lengths(trial).min())<floor:raise ArithmeticError('RAW endpoint violates protected floor')
    raise ArithmeticError('FP32 coordinate-event walk exceeded explicit safety bound')

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
        alpha=1.;cap=1.;roots=[];roundoff=0;fallback=None
        if trigger[i]:
            x=raw[i].astype(float);d=pre[i].astype(float)-x
            for a,b in EDGES:
                edgecap,rr=first_entry(x[a]-x[b],d[a]-d[b],floor);roots.append(rr);cap=min(cap,edgecap)
            aa,trial,roundoff,fallback=quantized_prefix(x,d,cap,floor,fixed[i],C_vertices[anchors[i][fixed[i]]])
            alpha=float(aa);guarded[i]=trial
        records.append(dict(raw_slot=i,known=i<K,floor_triggered=bool(trigger[i]),anti_collapse_protected=bool(protected[i]),
            raw_min_edge=float(rawlen[i].min()),raw_max_edge=float(rawlen[i].max()),proposal_min_edge=float(proplen[i].min()),
            guarded_min_edge=float(lengths(guarded[i]).min()),alpha_exact_prefix=cap,alpha_written=alpha,
            quantization_inward_steps=roundoff,quantization_event_fallback=fallback,quadratics=roots,
            max_guard_minus_raw=float(np.linalg.norm(guarded[i].astype(float)-raw[i],axis=1).max()),
            max_guard_minus_proposal=float(np.linalg.norm(guarded[i].astype(float)-pre[i],axis=1).max())))
    if guarded[fixed].tobytes()!=pre[fixed].tobytes():raise AssertionError('Hard anchors moved')
    return guarded,protected,records,pre
