"""Light provenance/failure-accounting tests; no production samples or CUDA."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from . import learned_outlets as m


def original_rows():
    rows=[]
    for seed in m.SEEDS:
        for sid in m.expected_ids():
            index=m.SEEDS.index(seed)*64+int(sid)
            for arm in ('fixed','a','qem'):
                rows.append({'seed':seed,'sample_id':sid,'arm':arm,'metric_status':'ok',
                    'source_nodes_sha256':f'nodes-{seed}-{sid}','reference_surface_hash':'surface',
                    'mesh_npz_sha256':'mesh','surface_metrics':[{'seed':b+index,'points_per_direction':8192,
                        'rms_bbox_percent':1.} for b in m.SEED_BASES]})
    return rows


def test_fixed_slots_and_baseline_require_all_seeds_samples():
    assert len(m.expected_slots())==256
    rows=original_rows();assert len(m.baseline_map(rows))==384
    for bad in (rows[:-1],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError,match='slots'):m.baseline_map(bad)
    swapped=deepcopy(rows);swapped[0]['seed']=12
    with pytest.raises(ValueError,match='slots'):m.baseline_map(swapped)


def test_baseline_rejects_other_nodes_mesh_sampling_or_reference():
    row=original_rows()[3*64+7*3]
    args=(11,'00007','nodes-11-00007','surface','mesh')
    m.validate_baseline_row(row,*args)
    variants=[]
    for key in ('source_nodes_sha256','reference_surface_hash','mesh_npz_sha256'):
        changed=deepcopy(row);changed[key]='other';variants.append(changed)
    for key,value in [('seed',7),('points_per_direction',2048),('rms_bbox_percent',float('nan'))]:
        changed=deepcopy(row);changed['surface_metrics'][0][key]=value;variants.append(changed)
    for changed in variants:
        with pytest.raises(ValueError):m.validate_baseline_row(changed,*args)


def declarations(tmp_path):
    return {f'{arm}_{seed}':{'source_run':str(tmp_path/arm),'kind':arm,'seed':seed,
        'path':str(tmp_path/arm/'training'/f'joint_{seed}'/'final.pt'),
        'expected_arm':'c0' if arm=='longfit' else 'c0_parent_weighted','job':f'joint_{seed}'}
        for arm in m.ARMS for seed in m.SEEDS}


def test_registration_is_all_four_fixed_finals_and_single_write(tmp_path,monkeypatch):
    (tmp_path/'RUN_PROTOCOL.json').write_text('{}')
    p={'model_declarations':declarations(tmp_path),'source_protocols':{a:{'sha256':a} for a in m.ARMS}}
    monkeypatch.setattr(m,'verify_contract',lambda run:p)
    seen=[]
    def validate(source,kind,seed):
        seen.append((kind,seed))
        if kind=='balanced' and seed==11:raise FileNotFoundError('await fixed final')
        d=p['model_declarations'][f'{kind}_{seed}']
        return {},{'path':d['path'],'arm':d['expected_arm'],'protocol_sha256':kind}
    monkeypatch.setattr(m,'validate_final',validate)
    with pytest.raises(FileNotFoundError):m.register_models(tmp_path)
    assert len(seen)==4 and not (tmp_path/'MODEL_REGISTRY.json').exists()
    monkeypatch.setattr(m,'validate_final',lambda source,kind,seed:({},
        {'path':p['model_declarations'][f'{kind}_{seed}']['path'],
         'arm':p['model_declarations'][f'{kind}_{seed}']['expected_arm'],'protocol_sha256':kind}))
    reg=m.register_models(tmp_path)
    assert len(reg['models'])==4 and reg['fixed_final_240k_only'] and reg['no_quality_or_RMS_selection']
    with pytest.raises(ValueError,match='immutable'):m.register_models(tmp_path)


def test_model_loader_rejects_registered_wrong_seed_and_stale_tensors(tmp_path):
    (tmp_path/'RUN_PROTOCOL.json').write_text('{}')
    d=declarations(tmp_path);p={'model_declarations':d,'source_protocols':{a:{'sha256':a} for a in m.ARMS}}
    model=m.DomainVelocity();weights=model.state_dict();mh=m.model_sha256(weights);entries={}
    for name,spec in d.items():
        path=Path(spec['path']);path.parent.mkdir(parents=True)
        state={'model':weights,'model_sha256':mh,'update':240000,'is_predeclared_final':True,
            'protocol_sha256':spec['kind'],'config':{'seed':spec['seed'],'job_id':spec['job'],'arm':spec['expected_arm']}}
        if name=='longfit_11':state['config']['seed']=10
        if name=='balanced_11':state['model_sha256']='stale'
        torch.save(state,path)
        entries[name]={'path':str(path),'file_sha256':m.file_hash(path),'model_sha256':mh,'protocol_sha256':spec['kind']}
    m.write_json(tmp_path/'MODEL_REGISTRY.json',{'protocol_sha256':m.file_hash(tmp_path/'RUN_PROTOCOL.json'),'models':entries})
    models,errors,_=m.load_models(tmp_path,p)
    assert set(models)=={'longfit_10','balanced_10'} and set(errors)=={'longfit_11','balanced_11'}
    assert all(not x.training and not any(t.requires_grad for t in x.parameters()) for x in models.values())


def test_raw_geometry_rejects_stale_radius_and_changed_connectivity():
    u=np.eye(3);faces=np.array([[0,1,2]],np.int64);nodes=np.array([1.,2.,3.])
    support=SimpleNamespace(query=lambda nodes,u:np.sum(u*nodes,axis=1))
    r=support.query(nodes,u);mesh={'u':u,'r':r,'vertices':u*r[:,None],'faces':faces}
    m.verify_raw_geometry(mesh,nodes,support,faces)
    for key in ('r','vertices','faces'):
        bad=deepcopy(mesh);bad[key].flat[0]+=1
        with pytest.raises(ValueError):m.verify_raw_geometry(bad,nodes,support,faces)


def new_rows():
    rows=[]
    for seed,sid,arm in sorted(m.expected_slots()):
        rows.append({'seed':seed,'sample_id':sid,'arm':arm,'metric_status':'failed','qualified':None,
            'paired_available':False,'gate_available':False})
    return rows


def test_summary_keeps_failures_and_computes_per_sample_gain_distribution():
    baseline=m.baseline_map(original_rows());rows=new_rows()
    selected=[r for r in rows if r['seed']==10 and r['arm']=='longfit'][:3]
    # Unequal Fixed scales: median of gains differs from ratio of RMS medians.
    for row,base,actual in zip(selected,(1.,10.,100.),(.5,9.,200.)):
        fixed=baseline[(row['seed'],row['sample_id'],'fixed')]
        for metric in fixed['surface_metrics']:metric['rms_bbox_percent']=base
        row.update(metric_status='ok',source_nodes_sha256=fixed['source_nodes_sha256'],
            surface_metrics=[{**x,'rms_bbox_percent':actual} for x in fixed['surface_metrics']],
            qualified=True,gate_available=True,audit={'valid':True},quality={'face_count':512},
            min_angle_margin_deg=.1,p05_angle_margin_deg=.2)
        row.update(m.paired_metrics(row,fixed))
    value=m.summarize(rows,baseline)
    group=next(g for g in value['groups'] if g['seed']==10 and g['arm']=='longfit')
    assert group['attempts']==group['planned_samples']==64
    assert group['metric_success_count']==3 and group['metric_failed_count']==61
    assert group['qualified_ge2pct_both_count']==2 and group['worse_either_measurement_count']==1
    gains=group['paired_gain_percent_distribution_two_measurements'][0]
    assert gains['median']==pytest.approx(10.) and gains['mean']==pytest.approx(-40/3)
    assert group['worst5_by_first_measurement'][0]['sample_id']==selected[-1]['sample_id']
    assert group['actual_face_count_histogram']=={'512':3}
    assert len(value['failed_slots'])==253 and value['raw_only'] and value['no_Fixed_fallback']
    with pytest.raises(ValueError,match='attempts'):m.summarize(rows[:-1],baseline)
    selected[0]['source_nodes_sha256']='foreign'
    with pytest.raises(ValueError,match='mainfield'):m.summarize(rows,baseline)


def test_unavailable_fixed_quality_gate_does_not_drop_successful_geometry():
    rows=new_rows();baseline=m.baseline_map(original_rows());r=rows[0]
    fixed=baseline[(r['seed'],r['sample_id'],'fixed')]
    r.update(metric_status='ok',source_nodes_sha256=fixed['source_nodes_sha256'],surface_metrics=fixed['surface_metrics'],
        quality={'face_count':511},audit={'valid':False})
    r.update(m.paired_metrics(r,fixed))
    value=m.summarize(rows,baseline)
    group=next(g for g in value['groups'] if g['seed']==r['seed'] and g['arm']==r['arm'])
    assert group['metric_success_count']==1 and group['gate_unavailable_count']==64
    assert group['quality_min_angle_margin_distribution'] is None
    assert group['actual_face_count_histogram']=={'511':1} and group['qualified_count']==0


def test_bound_contract_rejects_thread_policy_or_input_changes(tmp_path):
    reference=tmp_path/'REFERENCE_METRICS.json';reference.write_text('{}')
    bound=tmp_path/'input.npz';bound.write_bytes(b'fixed source')
    p={'seeds':list(m.SEEDS),'sample_ids':m.expected_ids(),'arms':list(m.ARMS),
        'primary_CPU_threads':1,'primary_interop_threads':1,'steps':32,'proxy_points':256,
        'MC_seed_bases':list(m.SEED_BASES),'points_each_direction':8192,'raw_only':True,'Fixed_fallback':False,
        'reference_metrics_copy_sha256':m.file_hash(reference),'original_metrics_sha256':m.file_hash(reference),
        'bound_files_sha256':{str(bound):m.file_hash(bound)}}
    def put(value):(tmp_path/'RUN_PROTOCOL.json').write_text(json.dumps(value))
    put(p);assert m.verify_contract(tmp_path)['primary_CPU_threads']==1
    for key,value in [('primary_CPU_threads',4),('Fixed_fallback',True),('MC_seed_bases',[1,2])]:
        bad=deepcopy(p);bad[key]=value;put(bad)
        with pytest.raises(ValueError):m.verify_contract(tmp_path)
    put(p);bound.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Bound original'):m.verify_contract(tmp_path)

