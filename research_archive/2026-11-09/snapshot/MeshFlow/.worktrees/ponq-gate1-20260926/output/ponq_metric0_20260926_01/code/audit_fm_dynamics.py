"""Frozen zero-training dynamics/gauge audit. No learned sampling or extraction.

Replay uses the true endpoint solely as an explicitly labeled velocity oracle.
Only a fresh output directory is written; all prior code/data/weights are read.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')


def module_from(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def errors(actual,expected):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    delta=a-b;den=float(np.linalg.norm(b))
    return dict(max_absolute=float(np.max(abs(delta))),root_mean_square=float(np.sqrt(np.mean(delta**2))),
                relative_frobenius=float(np.linalg.norm(delta)/den) if den else None,
                all_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                byteidentical=a.shape==b.shape and a.tobytes()==b.tobytes())


def ast_contract(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)
             and node.module=='fm_model' and any(n.name=='total_velocity' for n in node.names)]
    calls=[ast.unparse(node) for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=='total_velocity']
    assignments=[ast.unparse(node) for node in ast.walk(tree) if isinstance(node,(ast.Assign,ast.AnnAssign))]
    return dict(imports_shared_total_velocity=bool(imports),calls=calls,assignments=assignments)


def run(args):
    start=time.perf_counter();out=Path(args.out).resolve();source=Path(args.source_run).resolve()
    target=Path(args.target).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'code').mkdir();(out/'inputs').mkdir();(out/'audit').mkdir()
    source_names=['fm_model.py','fm_state.py','train_fm_pilot.py','sample_fm_pilot.py','prepare_fm_pilot.py']
    bindings=[]
    for name in source_names:
        original=source/'code'/name;destination=out/'code'/name
        shutil.copy2(original,destination)
        bindings.append(dict(source=str(original),snapshot=str(destination.relative_to(out)),sha256=sha(original)))
    shutil.copy2(__file__,out/'code'/'audit_fm_dynamics.py')
    bindings.append(dict(source=str(Path(__file__).resolve()),snapshot='code/audit_fm_dynamics.py',sha256=sha(__file__)))
    for original,name in [(target,'target_parameters.npz'),(source/'data.npz','previous_data.npz')]:
        shutil.copy2(original,out/'inputs'/name)
        bindings.append(dict(source=str(original),snapshot='inputs/'+name,sha256=sha(original)))
    for arm in ['direct','transport_residual']:
        original=source/'training'/arm/'PROTOCOL.json'
        shutil.copy2(original,out/'inputs'/f'{arm}_training_protocol.json')
        bindings.append(dict(source=str(original),snapshot=f'inputs/{arm}_training_protocol.json',sha256=sha(original)))
    # Freeze every tolerance, random seed, chart and probe before encoding/testing.
    tolerances={
        'float64':dict(chart_atol=5e-12,chart_rtol=5e-12,chain_atol=1e-12,chain_rtol=1e-11),
        'float32':dict(chart_atol=5e-5,chart_rtol=5e-5,chain_atol=1e-5,chain_rtol=2e-5)}
    protocol=dict(version='FM_DYNAMICS_GAUGE_AUDIT_V1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(source),target=str(target),source_bindings=bindings,
        target_identity='thin_u_frame06__q_only, valid analytic-Q control; not freeze',
        scope='Zero training. True endpoint velocity oracle is an interface test, not generation. No extraction or weight changes.',
        prior_seed=260930,permutation_seed=260931,noise_count=1,steps=64,solver='explicit Euler at k/64',
        chart_names=['previous_closed24_chart','single_target_chart'],
        single_target_statistics='component mean over 256 elements; one RMS-about-component-mean scalar per x3/n3/F16 block',
        target_factor='principal PSD square root of complete centered C; no Cholesky substitution',
        float64_gauge_atol=5e-12,float64_gauge_rtol=5e-12,tolerances=tolerances,
        chain_probe_times=[0.,.37,.83,1.],
        oracle_direct='raw velocity u=z1-z0; call the unchanged shared total_velocity(...,direct)',
        oracle_residual='raw velocity u-T(u_x,current state), then call unchanged shared total_velocity(...,transport_residual); x/n part unchanged',
        training_sampling_contract='Read AST of archived actual train/sample scripts; both import/call fm_model.total_velocity; total-velocity loss and chart Euler update checked explicitly',
        tolerance_interpretation='Declared numerical engineering tolerances only; not a mesh safety radius or a statistical criterion',
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,device='CPU'))
    dump(out/'RUN_PROTOCOL.json',protocol)
    dump(out/'RUN_PROTOCOL_SHA256.json',dict(sha256=sha(out/'RUN_PROTOCOL.json')))
    dump(out/'SOURCE_SNAPSHOT.json',dict(files=[dict(path=b['snapshot'],sha256=sha(out/b['snapshot'])) for b in bindings]))
    torch.set_num_threads(1)
    fm=module_from(out/'code/fm_model.py','frozen_fm_dynamics_model')
    state_api=module_from(out/'code/fm_state.py','frozen_fm_dynamics_state')
    checks=[];ledger=[];saved={}
    def check(name,passed,category='shared_direct',**detail):
        checks.append(dict(name=name,passed=bool(passed),category=category,**detail))
    train=ast_contract(out/'code/train_fm_pilot.py');sample=ast_contract(out/'code/sample_fm_pilot.py')
    check('same_total_velocity_import',train['imports_shared_total_velocity'] and sample['imports_shared_total_velocity'])
    check('training_calls_shared_total',train['calls']==['total_velocity(model, z, times, mean, scale, args.arm)'])
    check('sampling_calls_shared_total',sample['calls']==['total_velocity(model, z, t, mean, scale, arm)'])
    check('training_supervises_total_velocity','squared = (predicted - true_velocity).square()' in train['assignments']
          and 'loss = squared.mean()' in train['assignments'])
    check('sampling_chart_euler_update','z = z + velocity / 64.0' in sample['assignments'])
    dump(out/'audit/CALL_SITE_CONTRACT.json',dict(training=train,sampling=sample))
    for arm in ['direct','transport_residual']:
        prior=json.loads((out/'inputs'/f'{arm}_training_protocol.json').read_text())
        prior_sources={Path(b['path']).name:b['sha256'] for b in prior['source_files']}
        check(arm+':trained_source_hash_matches',all(prior_sources.get(name)==sha(out/'code'/name)
              for name in ['train_fm_pilot.py','fm_model.py']))
    params=load_arrays(out/'inputs/target_parameters.npz')
    encoded,encoding=state_api.encode_state(params)
    repeated,repeated_info=state_api.encode_state(params)
    permutation=np.random.default_rng(protocol['permutation_seed']).permutation(len(encoded))
    permuted_params={key:value[permutation] for key,value in params.items()}
    permuted,permuted_info=state_api.encode_state(permuted_params)
    restored=permuted[np.argsort(permutation)]
    F=encoded[:,6:].reshape(-1,4,4)
    gauge=dict(repeated_full_chart=errors(repeated,encoded),permutation_restored_full_chart=errors(restored,encoded),
               repeated_factor=errors(repeated[:,6:],encoded[:,6:]),
               permutation_restored_factor=errors(restored[:,6:],encoded[:,6:]),
               factor_symmetry_max_absolute=float(np.max(abs(F-F.transpose(0,2,1)))),
               factor_min_symmetric_eigenvalue=float(np.linalg.eigvalsh((F+F.transpose(0,2,1))/2).min()),
               encoding=encoding,repeated_encoding=repeated_info,permuted_encoding=permuted_info)
    check('repeated_encoding_consistent',np.allclose(repeated,encoded,rtol=5e-12,atol=5e-12))
    check('element_permutation_encoding_equivariant',np.allclose(restored,encoded,rtol=5e-12,atol=5e-12))
    check('factor_is_symmetric_principal_sqrt',gauge['factor_symmetry_max_absolute']<=5e-12
          and gauge['factor_min_symmetric_eigenvalue']>=-5e-12)
    dump(out/'audit/GAUGE.json',gauge)
    old=load_arrays(out/'inputs/previous_data.npz')
    one_mean=encoded.mean(axis=0);one_scale=np.empty(22)
    for a,b in [(0,3),(3,6),(6,22)]:
        one_scale[a:b]=np.sqrt(np.mean((encoded[:,a:b]-one_mean[a:b])**2))
    charts={'previous_closed24_chart':(old['mean'],old['scale']),
            'single_target_chart':(one_mean,one_scale)}
    np.savez_compressed(out/'inputs/derived_target_chart.npz',state=encoded,mean=one_mean,scale=one_scale,permutation=permutation)
    raw_noise=np.random.default_rng(protocol['prior_seed']).standard_normal((1,len(encoded),22))
    for chart_name,(mean_np,scale_np) in charts.items():
        for dtype_name,dtype in [('float64',torch.float64),('float32',torch.float32)]:
            tol=tolerances[dtype_name];mean=torch.tensor(mean_np,dtype=dtype);scale=torch.tensor(scale_np,dtype=dtype)
            fm.validate_chart(mean,scale)
            z0=torch.tensor(raw_noise,dtype=dtype)
            physical_x=(z0[0,:,:3].numpy().astype(np.float64)*scale_np[:3]+mean_np[:3])
            cost=((physical_x[:,None,:]-encoded[None,:,:3])**2).sum(axis=-1)
            ids,assignment=linear_sum_assignment(cost)
            check(chart_name+':'+dtype_name+':coupling_all_rows',np.array_equal(ids,np.arange(len(encoded))))
            z1=torch.tensor((encoded[assignment]-mean_np)/scale_np,dtype=dtype)[None]
            u=z1-z0
            tag=chart_name+'__'+dtype_name
            saved[tag+'__z0']=z0.numpy();saved[tag+'__z1']=z1.numpy();saved[tag+'__assignment']=assignment
            chain=[]
            with torch.no_grad():
                for time_value in protocol['chain_probe_times']:
                    z=(1-time_value)*z0+time_value*z1
                    _,diag=fm.compose_velocity(u,z,mean,scale,'transport_residual')
                    factor=(z*scale+mean)[...,6:].reshape(1,len(encoded),4,4)
                    dotF=(diag['transport'][...,6:]*scale[6:]).reshape_as(factor)
                    C=factor@factor.transpose(-1,-2)
                    K=torch.zeros_like(C);K[...,:3,:3]=diag['L']
                    lhs=dotF@factor.transpose(-1,-2)+factor@dotF.transpose(-1,-2)
                    rhs=-K.transpose(-1,-2)@C-C@K
                    passed=torch.allclose(lhs,rhs,atol=tol['chain_atol'],rtol=tol['chain_rtol'])
                    detail=errors(lhs.numpy(),rhs.numpy())
                    check(tag+':factor_chain_rule_at_'+str(time_value),passed,category='transport',**detail)
                    chain.append(dict(t=time_value,error=detail,L_max_absolute=float(diag['L'].abs().max()),
                        factor_transport_max_absolute=float(diag['transport'][...,6:].abs().max())))
            class DirectOracle(torch.nn.Module):
                def forward(self,z,t):return u
            class ResidualOracle(torch.nn.Module):
                def forward(self,z,t):
                    _,diag=fm.compose_velocity(u,z,mean,scale,'transport_residual')
                    return u-diag['transport']
            for arm,oracle in [('direct',DirectOracle()),('transport_residual',ResidualOracle())]:
                z=z0.clone();maximum_velocity_error=0.;maximum_chain_error=0.
                with torch.no_grad():
                    for step in range(64):
                        t=torch.full((1,),step/64,dtype=dtype)
                        velocity,diag=fm.total_velocity(oracle,z,t,mean,scale,arm)
                        maximum_velocity_error=max(maximum_velocity_error,float((velocity-u).abs().max()))
                        z=z+velocity/64.
                        if not bool(torch.isfinite(z).all()):raise FloatingPointError('Nonfinite oracle integration')
                detail=errors(z.numpy(),z1.numpy())
                passed=torch.allclose(z,z1,atol=tol['chart_atol'],rtol=tol['chart_rtol'])
                check(tag+':'+arm+':oracle_target_replay',passed,
                      category='shared_direct' if arm=='direct' else 'transport',**detail)
                physical=(z*scale+mean)[0].numpy()
                target_physical=encoded[assignment]
                decoded=state_api.decode_state(physical,normalize_normals=False)
                target_decoded=state_api.decode_state(target_physical,normalize_normals=False)
                semantic={key:errors(decoded[key],target_decoded[key]) for key in ['v','n','Q','eigs']}
                ledger.append(dict(chart=chart_name,dtype=dtype_name,arm=arm,passed=bool(passed),oracle_uses_true_endpoint=True,
                    interpretation='Interface replay only, not generation',steps=64,chart_error=detail,
                    maximum_total_velocity_absolute_error=maximum_velocity_error,
                    physical_chart_error=errors(physical,target_physical),decoded_semantic_errors=semantic,
                    factor_chain_rule_probes=chain))
                saved[tag+'__'+arm+'__endpoint']=z.numpy()
    np.savez_compressed(out/'ORACLE_ENDPOINTS.npz',**saved)
    dump(out/'LEDGER.json',ledger)
    passed=all(c['passed'] for c in checks)
    result=dict(passed=passed,check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,
        oracle_replay_count=len(ledger),oracle_pass_count=sum(r['passed'] for r in ledger),
        blocking_direct_interface_error=any(not c['passed'] for c in checks if c['category']=='shared_direct'),
        training_performed=False,extractions_performed=0,learned_weights_loaded=False,
        seconds=time.perf_counter()-start,
        conclusion_scope='No claim of learned generation, rollout robustness, mesh validity or unique factors for arbitrary generated F. Repeated target labels use canonical principal PSD square roots.')
    dump(out/'RESULT.json',result)
    dump(out/'MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    base=Path(__file__).resolve().parents[3]
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,default=base/'output/ponq_fm_pilot_20260926_01')
    parser.add_argument('--target',type=Path,default=base/'output/ponq_clay_factorial_20260926_01/runs/thin_u_frame06__q_only/parameters.npz')
    parser.add_argument('--out',type=Path,default=base/'output/ponq_fm_dynamics_20260926_01')
    result=run(parser.parse_args())
    raise SystemExit(0 if result['passed'] else 1)
