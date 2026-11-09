"""Frozen CPU-only relation tests and preregistered real-fixture quadrature gate.

No model is instantiated, called or updated. A real-data gate reads only the
explicitly frozen fixture manifest and tensor files; it never picks objects,
queries, seeds, bandwidths or quadrature counts from observed numerical results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .relations import (MODES, FEATURE_NAMES, quadrature_integer_nodes,
    barycentric_quadrature, triangle_statistics, relation_features, build_relations)

SYNTHETIC_SPEC = dict(seed=2026100507,known_counts=[1,2,4],times=[0.,.5,.9,1.],
    quadrature_n=[8,64,128],production_points=192,reference_points=[12288,49152],
    double_atol=2e-12,double_rtol=2e-11,finite_difference_h=1e-6,
    finite_difference_atol=2e-6,finite_difference_rtol=2e-4,
    model_forwards=0,model_backwards=0,optimizer_updates=0)
REAL_GATE_SPEC = dict(production_n=8,reference_n=64,convergence_n=128,
    ordinary_mu_max_L2_lt=.01,ordinary_covariance_max_Frobenius_lt=.01,
    ordinary_compressed_mass_max_abs_lt=.01,reference_convergence_each_metric_lt=.001,
    times=[0.,.5,.9,1.],ordinary_query_count=7,extreme_query_indices=[7],
    reference_query_indices=[0,6],reference_known_rule='[0,K//2,K-1] with duplicates removed',
    extreme_probes_are_not_accuracy_gate=True,model_forwards=0,model_backwards=0,optimizer_updates=0)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _source_identity():
    return {p.name:file_hash(p) for p in (Path(__file__),Path(__file__).with_name('relations.py'))}


def _begin(out,name,spec,identity=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=out/(name+'.json')
    if summary.exists():
        prior=read_json(summary)
        if prior.get('status')=='PASS' and prior.get('source_sha256')==_source_identity() and prior.get('input_identity')==identity:
            return prior,None
        raise RuntimeError('Existing test evidence is non-PASS or has different source/input identity; preserve and review it before any explicit rerun')
    attempt=out/(name+'.attempt.json')
    if attempt.exists():raise RuntimeError('Unresolved CPU test attempt; no silent retry')
    state=dict(status='RUNNING',spec=spec,source_sha256=_source_identity(),input_identity=identity,
        model_forwards=0,model_backwards=0,optimizer_updates=0,relation_evaluations=0,
        relation_autograd_calls=0,checks={},started_unix=time.time())
    atomic_json(attempt,state)
    return state,summary


def _close(state,path,status,started,exc=None):
    state.update(status=status,wall_seconds=time.perf_counter()-started,CUDA_initialized=torch.cuda.is_initialized())
    if exc is not None:state.update(error=repr(exc),traceback=traceback.format_exc())
    atomic_json(path,state)
    return state


def _assert_close(a,b,atol=2e-12,rtol=2e-11):
    if not torch.allclose(a,b,atol=atol,rtol=rtol):
        raise AssertionError('Tensor comparison failed; max_abs='+str(float((a-b).abs().max())))


def _synthetic_triangles(k,dtype=torch.float64):
    template=torch.tensor([[.05,-.1,.02],[.85,.03,.11],[.17,.74,-.04]],dtype=dtype)
    return torch.stack([template+torch.tensor([.17*i,-.11*i,.09*i],dtype=dtype) for i in range(k)])


def run_synthetic(out_dir):
    state,path=_begin(out_dir,'relations_synthetic',SYNTHETIC_SPEC)
    if path is None:return state
    if torch.cuda.is_initialized():raise RuntimeError('Synthetic relation checks are CPU-only')
    torch.set_num_threads(2);started=time.perf_counter()
    def check(name,**extra):state['checks'][name]=dict(status='PASS',**extra)
    def stats(*args,**kwargs):
        state['relation_evaluations']+=1
        return triangle_statistics(*args,**kwargs)
    def build(*args,**kwargs):
        state['relation_evaluations']+=len(args[0])
        return build_relations(*args,**kwargs)
    try:
        perms=list(itertools.permutations(range(3)))
        for n in (8,64,128):
            integer=quadrature_integer_nodes(n)
            original=sorted(integer)
            assert len(integer)==3*n*n
            for perm in perms:
                assert sorted(tuple(row[i] for i in perm) for row in integer)==original
            bary,weights=barycentric_quadrature(n,dtype=torch.float64)
            assert bool((weights>0).all()) and abs(float(weights.sum())-1)<2e-14
            C=_synthetic_triangles(2)
            nodes=torch.einsum('ra,kad->krd',bary,C)
            mu=torch.einsum('r,krd->kd',weights,nodes)
            residual=nodes-mu[:,None]
            covariance=torch.einsum('r,kri,krj->kij',weights,residual,residual)
            centroid=C.mean(1);centered=C-centroid[:,None]
            expected=torch.einsum('kai,kaj->kij',centered,centered)/12
            _assert_close(mu,centroid);_assert_close(covariance,expected)
        check('quadrature_exact_S3_positive_weights_degree2',nodes=[192,12288,49152],six_corner_permutations=True)

        q=torch.tensor([[.13,.21,.07],[-.3,.6,.2],[1.1,-.2,.15]],dtype=torch.float64)
        C=_synthetic_triangles(4)
        for mode in MODES:
            s=stats(q,C,.5,mode,reference=True)
            R=relation_features(q,s)
            assert R.shape==(3,4,11) and R.dtype==torch.float64
            _assert_close(R[...,:3],s['mu']-q[:,None])
            for index,(a,b) in enumerate(((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)),3):
                _assert_close(R[...,index],s['Sigma'][...,a,b])
            _assert_close(R[...,9],torch.log1p((-s['z']).clamp_min(0)))
            _assert_close(R[...,10],torch.full((3,4),.5,dtype=torch.float64))
            chunked=stats(q,C,.5,mode,reference=True,query_chunk=1,known_chunk=1)
            for key in ('mu','Sigma','z'):_assert_close(s[key],chunked[key])
            scaled=stats(q*2,C*2,.5,mode,sigma_z=2,reference=True)
            for key in ('mu','Sigma','z'):_assert_close(s[key],scaled[key])
            if mode=='MOMENT_XY':
                _assert_close(s['mu'],C.mean(1)[None].expand(3,-1,-1))
                _assert_close(s['z'],-(q[:,None]-C.mean(1)[None]).square().sum(-1)/(2*(.5**2+.1**2)))
            else:
                bary,_=barycentric_quadrature(8,dtype=torch.float64)
                nodes=torch.einsum('ra,kad->krd',bary,C)
                logw=-(q[:,None,None]-nodes[None]).square().sum(-1)/(2*(.5**2+.1**2))-np.log(192)
                p=torch.softmax(logw,-1);mu=torch.einsum('qkr,krd->qkd',p,nodes)
                centered=nodes[None]-mu[:,:,None]
                covariance=torch.einsum('qkr,qkri,qkrj->qkij',p,centered,centered)
                _assert_close(s['mu'],mu);_assert_close(s['Sigma'],covariance);_assert_close(s['z'],torch.logsumexp(logw,-1))
        check('literal_11_dimensions_and_chunking',feature_names=FEATURE_NAMES,constant_sigma_only=True,space_uses_x_minus_y=True)

        permutation_checks=0
        for k in (1,2,4):
            n=k+3;coordinates=torch.zeros((1,n,9),dtype=torch.float64)
            known=torch.zeros((1,n),dtype=torch.bool);known[:,1:k+1]=True
            valid=torch.ones((1,n),dtype=torch.bool);valid[:,-1]=False
            coordinates[0,1:k+1]=_synthetic_triangles(k).reshape(k,9)
            coordinates[0,0]=torch.tensor([-.2,.3,.4,.4,.3,.2,.7,.1,-.1],dtype=torch.float64)
            coordinates[0,k+1]=torch.tensor([.1,.5,.2,.6,.7,.1,-.2,.1,.3],dtype=torch.float64)
            coordinates[0,-1]=float('nan')  # Invalid padding cannot enter R.
            for mode in MODES:
                original=build(coordinates,torch.tensor([.7]),known,valid,mode,reference=True)[0]
                free_positions={int(f):i for i,f in enumerate(original['free_face_indices'])}
                known_positions={int(f):i for i,f in enumerate(original['known_indices'])}
                for offset in range(6):
                    cp=torch.tensor([perms[(face+offset)%6] for face in range(n)],dtype=torch.long)
                    changed=coordinates.reshape(1,n,3,3).clone()
                    changed[0]=changed[0,torch.arange(n)[:,None],cp]
                    order=torch.tensor(list(reversed(range(n))),dtype=torch.long)
                    changed=changed[:,order].reshape(1,n,9)
                    measured=build(changed,torch.tensor([.7]),known[:,order],valid[:,order],mode,reference=True)[0]
                    korder=[known_positions[int(order[j])] for j in measured['known_indices']]
                    expected=[]
                    for j in measured['free_face_indices']:
                        old=int(order[j]);r=original['R'][free_positions[old],cp[old]]
                        expected.append(r[:,korder])
                    _assert_close(measured['R'],torch.stack(expected))
                    permutation_checks+=1
                no_known=build(coordinates,.7,torch.zeros_like(known),valid,mode,reference=True)[0]
                assert no_known['R'].shape==(n-1,3,0,11) and bool(torch.isfinite(no_known['R']).all())
                no_free=build(coordinates,.7,valid,valid,mode,reference=True)[0]
                assert no_free['R'].shape==(0,3,n-1,11)
        check('nonzero_relations_face_corner_permutations_masks',comparisons=permutation_checks,K=[1,2,4],
              invalid_padding_NaN_ignored=True,no_known_no_softmax=True,no_free_supported=True)

        # Input-only autograd; no nn.Module, model backward, or optimizer exists.
        gradient_rows=[]
        for mode in MODES:
            q=q.clone().requires_grad_();C=_synthetic_triangles(2).requires_grad_();t=torch.tensor(.63,dtype=torch.float64,requires_grad=True)
            coefficients=torch.linspace(-.3,.5,11,dtype=torch.float64)
            def objective(a,b,c):
                ss=stats(a,b,c,mode,reference=True)
                return (relation_features(a,ss)*coefficients).square().mean()
            loss=objective(q,C,t);grad=torch.autograd.grad(loss,(q,C,t));state['relation_autograd_calls']+=1
            for which,index in ((0,(0,0)),(0,(2,2)),(1,(1,2,1)),(2,())):
                plus=[q.detach().clone(),C.detach().clone(),t.detach().clone()]
                minus=[a.clone() for a in plus]
                h=SYNTHETIC_SPEC['finite_difference_h'];plus[which][index]+=h;minus[which][index]-=h
                numeric=float((objective(*plus)-objective(*minus))/(2*h));automatic=float(grad[which][index])
                tolerance=SYNTHETIC_SPEC['finite_difference_atol']+SYNTHETIC_SPEC['finite_difference_rtol']*abs(numeric)
                assert abs(numeric-automatic)<=tolerance
                gradient_rows.append(dict(mode=mode,input=('query','known','t')[which],index=list(index),
                    autograd=automatic,central_difference=numeric,absolute_error=abs(numeric-automatic),tolerance=tolerance))
        check('relation_autograd_central_difference',comparisons=gradient_rows,h=1e-6)

        cases=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],
            [[0.,0.,0.],[1.,0.,0.],[.2,1e-7,0.]],
            [[0.,0.,0.],[1.,0.,0.],[.2,1e-10,0.]],
            [[0.,0.,0.],[.5,0.,0.],[1.,0.,0.]],
            [[.1,.1,.1],[.1,.1,.1],[.1,.1,.1]]],dtype=torch.float64)
        probes=torch.tensor([[1/3,1/3,0],[.5,0,0],[0,0,0],[20,-20,20],[10000,-10000,10000]],dtype=torch.float64)
        finite_rows=[]
        for reference in (False,True):
            for mode in MODES:
                for value in (0.,.5,.9,1.):
                    s=stats(probes,cases,value,mode,reference=reference)
                    R=relation_features(probes,s)
                    assert all(bool(torch.isfinite(s[key]).all()) for key in ('mu','Sigma','z')) and bool(torch.isfinite(R).all())
                    assert s['mu'].dtype==(torch.float64 if reference else torch.float32)
                    finite_rows.append(dict(mode=mode,t=value,dtype=str(s['mu'].dtype),max_abs_feature=float(R.abs().max())))
        check('far_edge_interior_slender_degenerate_finite',cases=['ordinary','slender','near_degenerate','line','collapsed'],
              coordinates_never_clamped=True,degenerate_scope='Finite barycentric push-forward only; zero-area triangles are not legal area surfaces',rows=finite_rows)

        x=torch.arange(5*9,dtype=torch.float32).reshape(1,5,9)/37
        known=torch.tensor([[False,True,False,True,False]]);valid=torch.tensor([[True,True,True,True,False]])
        for mode in MODES:
            cache=[]
            first=build(x,.4,known,valid,mode,static_cache=cache)
            cache_id=id(cache[0]);repeated=build(x,.9,known,valid,mode,static_cache=cache)
            assert id(cache[0])==cache_id
            assert not torch.equal(first[0]['R'],repeated[0]['R'])
            changed=x.clone();changed.reshape(1,5,3,3)[:,known[0],:,0]*=1.1
            old=cache[0];edited=build(changed,.9,known,valid,mode,static_cache=cache)
            assert cache[0] is not old
            uncached=build(changed,.9,known,valid,mode)
            assert torch.equal(edited[0]['R'],uncached[0]['R'])
        check('static_cache_identity_current_query_time_recomputed',changed_C_refreshes=True,unchanged_C_reuses_static_only=True)
        with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
            autocast_result=build(x,.4,known,valid,'SURFACE_XY')[0]['R']
        plain_result=build(x,.4,known,valid,'SURFACE_XY')[0]['R']
        assert autocast_result.dtype==torch.float32 and torch.equal(autocast_result,plain_result)
        check('production_geometry_autocast_disabled',dtype='torch.float32',CPU_BF16_autocast_exact=True)
        assert not torch.cuda.is_initialized()
        return _close(state,path,'PASS',started)
    except BaseException as exc:
        _close(state,path,'FAIL',started,exc);raise


def _errors(production,reference):
    mu=torch.linalg.vector_norm(production['mu'].double()-reference['mu'].double(),dim=-1)
    covariance=torch.linalg.matrix_norm(production['Sigma'].double()-reference['Sigma'].double(),ord='fro',dim=(-2,-1))
    mass=(torch.log1p((-production['z'].double()).clamp_min(0))-torch.log1p((-reference['z'].double()).clamp_min(0))).abs()
    return dict(mu_L2=mu,covariance_Frobenius=covariance,compressed_mass_abs=mass)


def run_real_fixtures(fixture_manifest_path,out_dir):
    """Read frozen manifest: fixtures[{fixture_id,path,sha256,...identity}].

    NPZ fields: C[K,3,3], queries[4,8,3], times[4]. The first seven queries are
    ordinary, final query is the separately reported extreme probe. Reference
    convergence uses queries[0,6] and known indices sorted(set([0,K//2,K-1])).
    Relative paths resolve from manifest data_root, or its own directory when
    data_root is absent. The manifest is never generated or edited here.
    """
    fixture_manifest_path=Path(fixture_manifest_path).resolve()
    manifest=read_json(fixture_manifest_path)
    identity=dict(manifest_path=str(fixture_manifest_path),manifest_sha256=file_hash(fixture_manifest_path))
    state,path=_begin(out_dir,'relations_real_gate',REAL_GATE_SPEC,identity)
    if path is None:return state
    if torch.cuda.is_initialized():raise RuntimeError('Real relation numerical gate must run on CPU')
    torch.set_num_threads(2);started=time.perf_counter();out_dir=Path(out_dir)
    rows=[];convergence_rows=[];fixture_results=[]
    try:
        fixtures=manifest['fixtures']
        if len(fixtures)!=4:raise ValueError('Require the four preregistered median/max × 20%/40% fixtures')
        base=Path(manifest.get('data_root',fixture_manifest_path.parent))
        for index,record in enumerate(fixtures):
            stored=record.get('path',record.get('npz_path'))
            if stored is None:raise ValueError('Frozen fixture path absent')
            source=Path(stored);source=source if source.is_absolute() else base/source
            expected_hash=record.get('sha256',record.get('npz_sha256'))
            if expected_hash is None or file_hash(source)!=expected_hash:raise ValueError('Frozen geometry fixture bytes changed')
            with np.load(source,allow_pickle=False) as data:
                C=torch.from_numpy(data['C'].copy())
                queries=torch.from_numpy(data['queries'].copy())
                times=torch.from_numpy(data['times'].copy())
            if C.ndim!=3 or C.shape[1:]!=(3,3) or queries.shape!=(4,8,3) or times.shape!=(4,):
                raise ValueError('Frozen query/known fixture dimensions differ')
            if not torch.allclose(times.double(),torch.tensor([0.,.5,.9,1.],dtype=torch.float64),atol=3e-8,rtol=0):
                raise ValueError('Frozen geometry time grid differs')
            k=len(C);known_indices=sorted(set((0,k//2,k-1)));query_indices=[0,6]
            if k<1:raise ValueError('Real multiple-known fixture has no known triangles')
            fixture_id=record.get('fixture_id',str(index));ordinary={k:0. for k in ('mu_L2','covariance_Frobenius','compressed_mass_abs')}
            extreme=dict(ordinary);converged=dict(ordinary)
            for ti,value in enumerate(times):
                production=triangle_statistics(queries[ti],C,value,'SURFACE_XY',n=8,reference=False)
                reference=triangle_statistics(queries[ti],C,value,'SURFACE_XY',n=64,reference=True)
                dense=triangle_statistics(queries[ti,query_indices],C[known_indices],value,'SURFACE_XY',n=128,reference=True)
                state['relation_evaluations']+=3
                error=_errors(production,reference)
                for field,array in error.items():
                    if not bool(torch.isfinite(array).all()):raise ValueError('Nonfinite real quadrature comparison')
                    ordinary[field]=max(ordinary[field],float(array[:7].max()))
                    extreme[field]=max(extreme[field],float(array[7].max()))
                for qi in range(8):
                    for ki in range(k):
                        rows.append(dict(fixture_id=fixture_id,t=float(value),query_index=qi,known_index=ki,
                            scope='ordinary_gate' if qi<7 else 'extreme_report_only',
                            mu_L2=float(error['mu_L2'][qi,ki]),covariance_Frobenius=float(error['covariance_Frobenius'][qi,ki]),
                            compressed_mass_abs=float(error['compressed_mass_abs'][qi,ki]),
                            production_dtype=str(production['mu'].dtype),reference_dtype=str(reference['mu'].dtype)))
                subset={key:reference[key][query_indices][:,known_indices] for key in ('mu','Sigma','z')}
                convergence=_errors(subset,dense)
                for field,array in convergence.items():converged[field]=max(converged[field],float(array.max()))
                for qi,qid in enumerate(query_indices):
                    for ki,kid in enumerate(known_indices):
                        convergence_rows.append(dict(fixture_id=fixture_id,t=float(value),query_index=qid,known_index=kid,
                            mu_L2=float(convergence['mu_L2'][qi,ki]),covariance_Frobenius=float(convergence['covariance_Frobenius'][qi,ki]),
                            compressed_mass_abs=float(convergence['compressed_mass_abs'][qi,ki])))
            accuracy_pass=all(v<.01 for v in ordinary.values());reference_pass=all(v<.001 for v in converged.values())
            fixture_results.append(dict(fixture_id=fixture_id,identity=record,actual_npz_sha256=expected_hash,K=k,
                ordinary_max_errors=ordinary,extreme_max_errors=extreme,reference_max_errors=converged,
                accuracy_pass=accuracy_pass,reference_convergence_pass=reference_pass,
                status='PASS' if accuracy_pass and reference_pass else 'IMPLEMENTATION_BLOCKED'))
        for name,values in (('relations_real_pair_errors.csv',rows),('relations_reference_convergence.csv',convergence_rows)):
            with (out_dir/name).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        state.update(fixtures=fixture_results,actual_comparison_pairs=len(rows),actual_reference_pairs=len(convergence_rows),
            geometry_only=True,query_selection='Read unchanged frozen tensors; no task or probe selection performed',
            reference_convergence_scope='Fixed ordinary query/known subset only; not a uniform full-space error proof')
        passed=all(r['status']=='PASS' for r in fixture_results)
        return _close(state,path,'PASS' if passed else 'IMPLEMENTATION_BLOCKED',started)
    except BaseException as exc:
        _close(state,path,'IMPLEMENTATION_BLOCKED',started,exc);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path)
    p.add_argument('--synthetic',action='store_true');p.add_argument('--fixtures',type=Path)
    a=p.parse_args()
    if not a.synthetic and a.fixtures is None:raise ValueError('Select explicit synthetic or frozen real-fixture checks')
    results=[]
    if a.synthetic:results.append(run_synthetic(a.out))
    if a.fixtures is not None:results.append(run_real_fixtures(a.fixtures,a.out))
    print(json.dumps([dict(status=r['status'],relation_evaluations=r['relation_evaluations'],
        model_forwards=r['model_forwards'],model_backwards=r['model_backwards'],optimizer_updates=r['optimizer_updates']) for r in results]))
    if any(r['status']!='PASS' for r in results):raise SystemExit(2)


if __name__=='__main__':main()
