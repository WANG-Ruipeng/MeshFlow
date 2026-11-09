"""Twenty frozen parameter controls on the locally reconstructed synthetics."""
import sys,time,json,datetime,traceback,warnings,shutil
from pathlib import Path
import run_gate1 as gate
from common import *
from parameter_isolation import make_parameters,STATES
from PoNQ_to_mesh import MeshFromPoNQ
import torch

baseline=Path(sys.argv[1]).resolve()
names=['sphere','box','thin_u','torus']
old=json.loads((baseline/'FROZEN_PROTOCOL.json').read_text())
objects={name:old['objects'][name] for name in names}
inputs={}
for name in names:
    files=[baseline/'inputs'/name/f for f in ['reference.obj','reference_eval_samples.npz','fit_256.npz']]
    files += [baseline/'runs'/f'{name}__qem_{level}'/'parameters.npz' for level in ['light','strong']]
    inputs[name]=[dict(path=str(p),sha256=sha(p)) for p in files]
config=dict(version='PONQ_PARAMETER_ISOLATION_V1',run_states=list(STATES),budgets=[256],dtype='float64',
    seed=271828,svd_eps=.05,grid_scale=32,network_training_updates=0,
    state_groups={'clean':['roundtrip'],'mild':['fixed_v_metric_light','position_matched_q_light']},
    perturbation_expectations={},baseline=str(baseline),source_files=inputs,
    roundtrip='Full centered A,g,c plus explicit floating residual. No implicit regularization or re-solving vertex.',
    fixed_v_metric='Spatial S=I+alpha H centered at frozen v; full Q congruence transforms A,g, preserves c. v,n fixed.',
    fixed_v_metric_doses=[.02,.1],dose_warning='Spatial-transform dose is not old full4 factor-noise dose.',
    position_matched='Exactly copy locally saved original qem-arm v; translate original Q coherently, A/n/eigs unchanged.',
    samples_eval_per_direction=8192,thresholds='Report all outcomes; no accuracy success threshold invented',
    known_limitations=['four synthetic objects','one fixed seed','interface controls, not learned error distribution',
                      'fixed_v_metric also changes centered g','does not establish generative model performance'])
dump(ROOT/'FROZEN_PROTOCOL.json',dict(config=config,objects=objects,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
dump(ROOT/'FROZEN_PROTOCOL_SHA256.json',dict(sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
rows=[]
for oi,name in enumerate(names):
    for identity in inputs[name]:
        assert sha(identity['path'])==identity['sha256']
    dest=ROOT/'inputs'/name;dest.mkdir()
    for filename in ['reference.obj','reference_eval_samples.npz','fit_256.npz']:
        shutil.copy2(baseline/'inputs'/name/filename,dest/filename)
    gv,gf=load_mesh(dest/'reference.obj')
    with np.load(dest/'fit_256.npz',allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
    with np.load(dest/'reference_eval_samples.npz',allow_pickle=False) as z:gs=z['points'].copy();gn=z['normals'].copy()
    dump(dest/'reference_audit.json',mesh_audit(gv,gf))
    for state in STATES:
        rd=ROOT/'runs'/f'{name}__{state}';rd.mkdir()
        result=dict(object=name,state=state,budget=256,extraction_attempts=1,reference_betti=objects[name]['expected_betti'])
        started=time.perf_counter()
        print('EXTRACT_START',name,state,flush=True)
        try:
            source=None
            if state.startswith('position_matched'):
                level='strong' if state.endswith('strong') else 'light'
                with np.load(baseline/'runs'/f'{name}__qem_{level}'/'parameters.npz',allow_pickle=False) as z:
                    source={k:z[k].copy() for k in z.files}
            params,pa=make_parameters(base,state,obj_index=oi,source_params=source)
            np.savez_compressed(rd/'parameters.npz',**params);dump(rd/'perturbation_audit.json',pa)
            result.update(pa)
            h=cKDTree(base['v']).query(base['v'],k=5)[0][:,1:].mean(1)
            shifts=np.linalg.norm(params['v']-base['v'],axis=1)/h
            result.update(position_shift_over_h_mean=float(shifts.mean()),position_shift_over_h_max=float(shifts.max()))
            t=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                sc=MeshFromPoNQ(torch.from_numpy(params['v']),torch.from_numpy(params['eigs']),
                     torch.from_numpy(params['Q']),torch.from_numpy(params['n']),grid_scale=32,
                     correct_tet_color=True,compute_mincut=True)
                v,f=sc.get_surface()
            result.update(extraction_seconds=time.perf_counter()-t,warnings=[str(w.message) for w in caught])
            score=sc.final_scores
            result.update(nonfinite_cut_weights=int(np.sum(~np.isfinite(score))),negative_cut_weights=int(np.sum(score<0)),
                minimum_cut_weight=float(np.min(score[np.isfinite(score)])) if np.isfinite(score).any() else None,
                tetrahedra=len(sc.simplices),qhull_omitted_input_vertices=int(len(sc.coplanar)),
                anchor_vertices_in_output=int(np.sum(np.max(abs(v),axis=1)>=.999999)))
            np.savez_compressed(rd/'tetrahedra.npz',vertices=sc.points,tetrahedra=sc.simplices,labels=sc.tet_colors)
            np.savez_compressed(rd/'mesh.npz',vertices=v,faces=f)
            save_mesh(rd/'mesh.obj',v,f);av,af=load_mesh(rd/'mesh.obj')
            assert np.array_equal(av,v) and np.array_equal(af,f)
            result['mesh_sha256']=sha(rd/'mesh.obj')
            result['status']='EXTRACTED' if np.isfinite(score).all() and np.all(score>=0) else 'EXTRACTED_INVALID_CUT_WEIGHTS'
            result.update(mesh_audit(av,af));t=time.perf_counter()
            result.update(metrics(av,af,gv,gf,gs,gn,8801+oi,n=8192))
            result['metric_seconds']=time.perf_counter()-t
            result['betti_match']=[result[f'beta{i}'] for i in range(3)]==result['reference_betti']
            result['reference_clean_mesh_byte_equal']=sha(rd/'mesh.obj')==sha(baseline/'runs'/f'{name}__clean256'/'mesh.obj')
        except Exception as exc:
            result.update(status='EXTRACTION_OR_EVALUATION_FAILED',error=repr(exc),traceback=traceback.format_exc(),
                          elapsed_until_failure_seconds=time.perf_counter()-started)
        dump(rd/'result.json',result);rows.append(result);dump(ROOT/'LEDGER_PARTIAL.json',rows)
        print('EXTRACT_END',name,state,result['status'],result.get('rms_pct_bboxdiag'),flush=True)
dump(ROOT/'LEDGER_RAW.json',rows)