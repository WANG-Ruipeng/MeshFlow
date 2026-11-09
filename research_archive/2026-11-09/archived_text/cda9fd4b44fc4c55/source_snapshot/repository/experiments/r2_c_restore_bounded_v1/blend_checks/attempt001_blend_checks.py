"""Meaningful CPU fixtures only; never loads the real panel or any model."""
from __future__ import annotations
import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[2]
OUT=ROOT/'experiments/r2_c_restore_bounded_v1/blend_checks'
spec=importlib.util.spec_from_file_location('_bounded_blend_checks_subject',CODE/'blend.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def one_anchor(center,target,free):
    center=np.asarray(center,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
    C=np.repeat(target[None,:],3,axis=0).reshape(1,3,3)
    proposal=np.concatenate([np.repeat(center[None,:],3,axis=0),np.asarray(free,dtype=np.float32).reshape(-1,3)]).reshape(-1,3,3)
    return C,proposal,np.zeros((1,3),dtype=np.int64)

def run_subject(C,P,ids,L=1.):
    before=[a.tobytes() for a in (C,P,ids)]
    for a in (C,P,ids):a.flags.writeable=False
    result=B.coordinate_restore(C,P,ids,L=L)
    assert [a.tobytes() for a in (C,P,ids)]==before,'Caller arrays mutated'
    O,d,a=result;K=len(C)
    assert O.dtype==np.float32 and O.shape==P.shape
    assert O[:K].tobytes()==C.tobytes(),'C bits changed'
    actual=O[K:].astype(np.float64).reshape(-1,3)-P[K:].astype(np.float64).reshape(-1,3)
    assert np.array_equal(actual,a['free_actual_displacement_FP32'])
    assert np.all(np.linalg.norm(actual,axis=1)<=a['free_actual_bound'])
    assert np.all(np.linalg.norm(actual,axis=1)<=.02*L)
    assert np.isfinite(O).all()
    return result

def kernel_compact_symmetric():
    x=np.array([[0,0,0],[.05,0,0],[.1,0,0],[.2,0,0],[-.05,0,0]],dtype=float)
    observed=B.kernel(x,[[0,0,0]],.1).reshape(-1)
    assert np.array_equal(observed,np.array([1.,.1875,0.,0.,.1875]))
    return dict(observed=observed.tolist())

def zero_b_and_no_support():
    C,P,ids=one_anchor([0,0,0],[-0.,0,0],[[.02,0,0],[-.02,0,0],[0,.02,0]])
    O,d,a=run_subject(C,P,ids)
    assert O[1:].tobytes()==P[1:].tobytes() and not a['free_displacement_unclipped'].any()
    C,P,ids=one_anchor([0,0,0],[0,.01,0],[[-1,0,0],[1,0,0],[0,1,0]])
    O,d,a=run_subject(C,P,ids)
    assert O[1:].tobytes()==P[1:].tobytes() and d['no_support_free_corners']==3
    assert np.array_equal(a['free_actual_bound'],np.zeros(3))
    return dict(zero_displacement_control=True,no_support_corners=3,signed_zero_C_preserved=True)

def single_anchor_support_fade():
    C,P,ids=one_anchor([0,0,0],[0,.005,0],[[.05,0,0],[.099,0,0],[-.05,0,0]])
    O,d,a=run_subject(C,P,ids)
    # At half radius a single source has scalar Wendland value exactly 3/16
    # for exact 0.05; account only for the fixture's actual FP32 x value.
    x=float(P[1,0,0]);q=x/.1;expected=(1-q)**4*(4*q+1)*float(C[0,0,1])
    assert O[1,0,1]==np.float32(expected)
    assert np.isclose(a['free_alpha_sum'][0],3/16,rtol=1e-7)
    assert 0<a['free_alpha_sum'][1]<a['free_alpha_sum'][0]<1
    assert 0<O[1,1,1]<O[1,0,1]<C[0,0,1]
    assert d['clipped_free_corners']==0
    return dict(half_radius_alpha_sum=float(a['free_alpha_sum'][0]),near_boundary_alpha_sum=float(a['free_alpha_sum'][1]),unwanted_full_normalization_rejected=True)

def multiplicity_invariance():
    centers=np.array([[0,0,0],[2,0,0],[4,0,0]],dtype=np.float32)
    targets=centers.copy();targets[:,1]=[.002,.004,.006]
    free=np.array([[.05,0,0],[2.05,0,0],[4.05,0,0]],dtype=np.float32)
    C=targets.reshape(1,3,3);P=np.concatenate([centers,free]).reshape(2,3,3);ids=np.array([[0,1,2]],dtype=np.int64)
    a,_,ad=run_subject(C,P,ids)
    # Uneven replication: source 0 is represented four times, sources 1/2 once.
    order=np.array([0,0,0,0,1,2])
    C2=targets[order].reshape(2,3,3);P2=np.concatenate([centers[order],free]).reshape(3,3,3);ids2=np.array([[0,0,0],[0,1,2]],dtype=np.int64)
    b,bd,bdet=run_subject(C2,P2,ids2)
    assert a[1:].tobytes()==b[2:].tobytes()
    assert np.allclose(ad['free_alpha_sum'],bdet['free_alpha_sum'],rtol=0,atol=0)
    assert all(x['total_weight']==1. for x in bd['duplicates'])
    return dict(unique_sources=3,corner_counts=[3,6],uneven_duplicate_count=4,identical_free_output=True)

def conflicting_copies_retained():
    C=np.zeros((1,3,3),dtype=np.float32)
    centers=np.array([[-.01,0,0],[.01,0,0],[0,0,0]],dtype=np.float32)
    free=np.array([[0,0,0],[.02,0,0],[1,0,0]],dtype=np.float32)
    P=np.concatenate([centers,free]).reshape(2,3,3);ids=np.zeros((1,3),dtype=np.int64)
    O,d,a=run_subject(C,P,ids)
    assert np.array_equal(a['anchor_proposal_model'],centers.astype(float))
    assert a['free_displacement_unclipped'][0,0]==0. and O[1,0].tobytes()==P[1,0].tobytes()
    assert a['free_displacement_unclipped'][1,0]<0
    assert d['split_source_vertex_groups']==1 and d['anchor_corners']==3
    assert np.all(a['anchor_weight']==1/3)
    return dict(conflicting_observations=3,symmetric_cancellation=True,off_center_blend_not_zero=True)

def dense_support_subconvex_bound():
    centers=np.zeros((6,3),dtype=np.float32)
    target=np.array([[.01,0,0],[0,.01,0],[0,0,.01],[-.005,0,0],[0,-.003,0],[0,0,-.004]],dtype=np.float32)
    free=np.array([[0,0,0],[.02,0,0],[.08,0,0]],dtype=np.float32)
    C=target.reshape(2,3,3);P=np.concatenate([centers,free]).reshape(3,3,3);ids=np.arange(6,dtype=np.int64).reshape(2,3)
    O,d,a=run_subject(C,P,ids)
    assert a['free_support_sum'][0]==6 and a['free_alpha_sum'][0]==1
    assert np.allclose(a['free_displacement_unclipped'][0],target.astype(float).mean(0),rtol=0,atol=2e-18)
    assert 0<a['free_alpha_sum'][2]<1
    assert np.all(np.linalg.norm(a['free_displacement_unclipped'],axis=1)<=a['free_participating_b_max_norm'])
    assert np.all(a['free_actual_displacement_norm']<=a['free_participating_b_max_norm'])
    return dict(dense_support_sum=6,alpha_sum=float(a['free_alpha_sum'][0]),sparse_alpha_sum=float(a['free_alpha_sum'][2]),convex_norm_bound=True)

def cap_after_FP32_rounding():
    C,P,ids=one_anchor([.25,0,0],[.45,0,0],[[.25,0,0],[.251,0,0],[.26,0,0]])
    O,d,a=run_subject(C,P,ids)
    assert d['safety_cap_active_free_corners']==3
    assert d['FP32_cap_rounding_adjusted_corners']>=1
    naive=float(np.float32(.25+.02))-float(np.float32(.25))
    assert naive>.02 and float(a['free_actual_displacement_norm'][0])<=.02
    assert O[1,0,0]==np.nextafter(np.float32(.27),np.float32(.25))
    assert np.array_equal(a['free_actual_bound'],np.full(3,.02))
    return dict(naive_FP32_step=naive,actual_FP32_step=float(a['free_actual_displacement_norm'][0]),cap_active=3,rounding_adjusted=d['FP32_cap_rounding_adjusted_corners'])

def invalid_input_contracts():
    C,P,ids=one_anchor([0,0,0],[0,.01,0],[[0,0,0],[.01,0,0],[.02,0,0]])
    cases=[(C.astype(float),P,ids,1.,TypeError),(C,P,ids.astype(float),1.,ValueError),(C,P,ids,0.,ValueError)]
    bad=C.copy();bad[0,1,0]=.1;cases.append((bad,P,ids,1.,ValueError))
    badp=P.copy();badp[1,0,0]=np.nan;cases.append((C,badp,ids,1.,ValueError))
    for c,p,i,L,expected in cases:
        try:B.coordinate_restore(c,p,i,L=L)
        except expected:pass
        else:raise AssertionError('Invalid input not rejected')
    return dict(invalid_cases_rejected=len(cases))

TESTS=[
 ('kernel_compact_symmetric',kernel_compact_symmetric,'Check hand-computable Wendland values, strict compact support, and negative-coordinate symmetry.'),
 ('zero_b_and_no_support',zero_b_and_no_support,'Negative controls must not move free corners; signed-zero C bytes are retained.'),
 ('single_anchor_support_fade',single_anchor_support_fade,'Prevent accidental normalization to one outside dense support; near-radius response must fade.'),
 ('multiplicity_invariance',multiplicity_invariance,'Uneven repeated source IDs must not increase their total influence.'),
 ('conflicting_copies_retained',conflicting_copies_retained,'Keep split proposal observations and allow opposing contributions to cancel instead of last-copy selection.'),
 ('dense_support_subconvex_bound',dense_support_subconvex_bound,'Dense support becomes a convex mean, sparse support stays subconvex, and the participating displacement norm bound holds.'),
 ('cap_after_FP32_rounding',cap_after_FP32_rounding,'Catch a real FP32 rounding overshoot of .02L and enforce the actual stored displacement bound.'),
 ('invalid_input_contracts',invalid_input_contracts,'Reject dtype/scale/source-ID inconsistency and nonfinite geometry rather than silently changing definitions.')]

def main():
    start=time.perf_counter();rng_before=np.random.get_state();oldsolve=np.linalg.solve
    def forbidden(*a,**kw):raise AssertionError('Matrix solve is prohibited for local blend')
    np.linalg.solve=forbidden;results=[];failure=None
    try:
        for name,fn,reason in TESTS:
            try:results.append(dict(name=name,status='PASS',reason=reason,evidence=fn()))
            except Exception as exc:
                results.append(dict(name=name,status='FAIL',reason=reason,error=repr(exc)));failure=exc;break
    finally:np.linalg.solve=oldsolve
    rng_after=np.random.get_state()
    rng_unchanged=all(np.array_equal(a,b) for a,b in zip(rng_before,rng_after))
    assert rng_unchanged
    report=dict(status='PASS' if failure is None else 'FAIL',checks=len(results),tests=results,
        input_immutability='Read-only NumPy inputs and byte checks enforced in every successful fixture',
        RNG_unchanged=rng_unchanged,matrix_solve_interceptor=True,model_calls=0,real_panel_outputs=0,
        numpy_version=np.__version__,seconds=time.perf_counter()-start,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),CODE/'blend.py')})
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'completion.json';temp=path.with_suffix('.json.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    print(json.dumps(dict(status=report['status'],checks=len(results),seconds=report['seconds'],output=str(path))))
    if failure is not None:raise failure

if __name__=='__main__':main()
