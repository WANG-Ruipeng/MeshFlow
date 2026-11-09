"""Versioned CPU RAW local-quality reward; no model, GT, repair or face deletion.

Run this module in a process that has not imported torch. The original exact
CF/FF detector is reused unchanged. The reference is a frozen same-noise model
output, never the unknown source completion. Retention is a geometric proxy,
not certification of semantics, support, manifoldness or usability.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from . import common as C

VERSION='bench_local_quality_reward_v1'
MAX_EDIT_FACES=8
SURFACE_TOLERANCE=0.01*C.L
MIN_PROTECTED_COVERAGE=0.98
MAX_AREA_RELATIVE_CHANGE=0.05
AREA_EPS=1e-12*C.L*C.L
ILLEGAL=frozenset(('proper_transverse_penetration','coplanar_positive_area_overlap'))
BARYCENTRIC=np.asarray([(i/3,j/3,(3-i-j)/3) for i in range(4) for j in range(4-i)],np.float64)


def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def checked(raw,K):
    a=np.asarray(raw)
    if a.dtype!=np.float32 or a.ndim!=3 or a.shape[1:]!=(3,3) or not np.isfinite(a).all():
        raise ValueError('Finite float32[N,3,3] RAW required')
    if type(K) is not int or not 0<K<len(a):raise ValueError('Valid C/free partition required')
    return a


def areas(tri):
    a=np.asarray(tri,np.float64)
    return np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)*.5


def scan(raw,K):
    if 'torch' in sys.modules:raise RuntimeError('Reward must execute in an isolated CPU process')
    from r2.tools.hinge_opposite_v1 import engine as H
    if abs(float(H.C.L)-C.L)>1e-12:raise ValueError('Original geometry scale changed')
    return H.scan(raw,K)


def bad_pairs(scans,kind):
    return {tuple(map(int,r['pair'])) for r in scans[kind]['contacts'] if r['classification'] in ILLEGAL}


def mask_from_pairs(N,K,pairs,max_faces=MAX_EDIT_FACES):
    """Lexicographic pair order; both free endpoints always enter together."""
    chosen=set();accepted=[]
    for i,j in sorted(pairs):
        if not 0<=i<j<N-K:raise ValueError('FF pair must use local free-face IDs')
        candidate=chosen|{i,j}
        if len(candidate)<=max_faces:
            chosen=candidate;accepted.append([i,j])
    mask=np.zeros(N,bool)
    if chosen:mask[np.asarray(sorted(chosen),np.int64)+K]=True
    return mask,accepted


def reference(raw,K,scans=None):
    raw=checked(raw,K); scans=scan(raw,K) if scans is None else scans
    ff,cf=bad_pairs(scans,'FF'),bad_pairs(scans,'CF')
    mask,selected=mask_from_pairs(len(raw),K,ff)
    m=set(np.flatnonzero(mask[K:]).tolist())
    target={p for p in ff if set(p)&m};outside=ff-target
    a=areas(raw[K:])
    return dict(version=VERSION,reference_RAW_sha256=array_hash(raw),N=len(raw),K=K,
        edit_mask=mask.tolist(),edit_free_face_ids=sorted(m),selected_pair_endpoints=selected,
        local_bad_pairs=[list(p) for p in sorted(target)],outside_bad_pairs=[list(p) for p in sorted(outside)],
        CF_bad_pairs=[list(p) for p in sorted(cf)],reference_free_areas=a.tolist(),
        reference_free_area=float(a.sum()),reference_degenerate_free_ids=np.flatnonzero(a<=AREA_EPS).tolist(),
        input_policy='C/N + same-noise frozen EXPOSURE RAW only; source Y not read',
        mask_policy='fixed before stochastic action; lexicographic illegal FF pairs; complete endpoints; max8',
        surface_query='ten fixed barycentric samples per non-edited reference face, weighted by reference face area',
        surface_tolerance_model=SURFACE_TOLERANCE,minimum_protected_coverage=MIN_PROTECTED_COVERAGE,
        maximum_area_relative_change=MAX_AREA_RELATIVE_CHANGE,area_epsilon=AREA_EPS)


def nearest(points,tri):
    if 'torch' in sys.modules:raise RuntimeError('Distance scorer must remain CPU/model-free')
    from native_t1.metric_geometry import surface_distance_squared
    return np.sqrt(surface_distance_squared(points,tri,chunk=32))


def evaluate(base,current,K,ref=None,scans=None,distance_fn=None):
    base=checked(base,K);current=checked(current,K)
    if current.shape!=base.shape:raise ValueError('Fixed N/slot identities; no deletions permitted')
    ref=reference(base,K) if ref is None else ref
    if ref['version']!=VERSION or ref['reference_RAW_sha256']!=array_hash(base):raise ValueError('Reference identity mismatch')
    mask=np.asarray(ref['edit_mask'],bool)
    if mask.shape!=(len(base),) or mask[:K].any() or mask.sum()>MAX_EDIT_FACES:raise ValueError('Edit mask contract')
    scans=scan(current,K) if scans is None else scans
    ff,cf=bad_pairs(scans,'FF'),bad_pairs(scans,'CF')
    selected=set(np.flatnonzero(mask[K:]).tolist())
    local={p for p in ff if set(p)&selected};outside=ff-local
    base_local={tuple(p) for p in ref['local_bad_pairs']}
    base_outside={tuple(p) for p in ref['outside_bad_pairs']}
    base_cf={tuple(p) for p in ref['CF_bad_pairs']}
    a=areas(current[K:]);a0=np.asarray(ref['reference_free_areas'])
    area_change=abs(float(a.sum())-float(a0.sum()))/max(float(a0.sum()),AREA_EPS)
    new_degenerate=set(np.flatnonzero(a<=AREA_EPS).tolist())-set(ref['reference_degenerate_free_ids'])
    protected=~mask[K:]
    points=np.einsum('pc,fcd->fpd',BARYCENTRIC,base[K:][protected].astype(np.float64)).reshape(-1,3)
    weights=np.repeat(a0[protected]/len(BARYCENTRIC),len(BARYCENTRIC))
    distances=(nearest if distance_fn is None else distance_fn)(points,current[K:])
    if np.asarray(distances).shape!=(len(points),) or not np.isfinite(distances).all():raise ValueError('Finite surface distance contract')
    retained=float(np.sum(weights*(distances<=SURFACE_TOLERANCE))/max(float(weights.sum()),AREA_EPS))
    reasons=[]
    if current[:K].tobytes()!=base[:K].tobytes():reasons.append('C_CHANGED')
    if new_degenerate:reasons.append('NEW_DEGENERATE_FACE')
    if len(cf)>len(base_cf):reasons.append('CF_PAIR_COUNT_INCREASE')
    if len(outside)>len(base_outside):reasons.append('OUTSIDE_FF_PAIR_COUNT_INCREASE')
    if area_change>MAX_AREA_RELATIVE_CHANGE:reasons.append('FREE_AREA_CHANGE_GT5PCT')
    if retained<MIN_PROTECTED_COVERAGE:reasons.append('PROTECTED_SURFACE_COVERAGE_LT98PCT')
    local_reward=float(np.clip((len(base_local)-len(local))/max(1,len(base_local)),-1,1))
    value=-1.0 if reasons else local_reward
    return dict(version=VERSION,reward=value,local_reward=local_reward,protection_pass=not reasons,
        rejection_reasons=reasons,mask_empty=not bool(mask.any()),reference_RAW_sha256=array_hash(base),
        current_RAW_sha256=array_hash(current),N=len(base),K=K,deleted_faces=0,edit_faces=int(mask.sum()),
        baseline_local_FF_pairs=len(base_local),current_local_FF_pairs=len(local),
        eliminated_local_FF_pairs=[list(p) for p in sorted(base_local-local)],
        new_local_FF_pairs=[list(p) for p in sorted(local-base_local)],
        baseline_outside_FF_pairs=len(base_outside),current_outside_FF_pairs=len(outside),
        new_outside_FF_pairs=[list(p) for p in sorted(outside-base_outside)],
        baseline_CF_pairs=len(base_cf),current_CF_pairs=len(cf),new_CF_pairs=[list(p) for p in sorted(cf-base_cf)],
        new_degenerate_free_ids=sorted(new_degenerate),free_area=float(a.sum()),baseline_free_area=float(a0.sum()),
        free_area_relative_change=area_change,protected_surface_retention=retained,
        protected_query_count=len(points),protected_query_max_distance=float(np.max(distances)) if len(points) else None,
        zero_D0=True,zero_GT=True,zero_repair=True,zero_face_deletion=True,
        interpretation='local quality reward with reference-surface protection; not semantic completeness certification')


def worker_reference(raw,K):return reference(raw,K)

def worker_evaluate(base,current,K,ref):return evaluate(base,current,K,ref)

select_reference=reference


def draw_policy_noise(seed,N,edit_mask):
    mask=np.asarray(edit_mask,bool)
    if mask.shape!=(N,):raise ValueError('Fixed face mask shape differs')
    epsilon=np.random.RandomState(int(seed)).standard_normal((10,N,3,3)).astype(np.float32)
    epsilon[:,~mask]=0
    return epsilon


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['reference','evaluate']);ap.add_argument('--base',required=True)
    ap.add_argument('--current');ap.add_argument('--reference');ap.add_argument('--K',type=int,required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();base=C.arrays(args.base)['raw']
    result=reference(base,args.K) if args.mode=='reference' else evaluate(base,C.arrays(args.current)['raw'],args.K,
        ref=C.read(args.reference) if args.reference else None)
    C.write(args.out,result);print(json.dumps({k:v for k,v in result.items() if k in ('reward','protection_pass','edit_free_face_ids','version')}))

if __name__=='__main__':main()
