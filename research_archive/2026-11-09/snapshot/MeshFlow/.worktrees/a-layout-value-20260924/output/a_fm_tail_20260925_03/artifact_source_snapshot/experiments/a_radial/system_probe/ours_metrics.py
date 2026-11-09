"""Offline same-mainfield outlet metrics with immutable sample pairing.

This does not generate, repair, retry or select a mesh. Surface integrals are
Monte Carlo estimates using nearest points on full triangle surfaces. Exactness
of the separate audit concerns stored endpoints, not continuous trajectories.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
import torch
from ..exporter import ASSETS,RadialSupport
from ..io_utils import array_hash,file_hash,new_dir,write_json
from ..layout_value.geometry import _arrays,mesh_quality,evaluate_mesh,audit_mesh
from ..legacy.geometry_task import fibonacci,hull_faces

SEEDS=(10,11)
ARMS=('fixed','a','qem')
SAMPLES_PER_SEED=64
POINTS=8192
SEED_BASES=(993501,1093501)
QUALITY_EPS_DEG=1e-6


def require(ok,message):
    if not ok:raise ValueError(message)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def inside(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    require(path.is_relative_to(root),'Artifact path escapes registered directory')
    return path


def expected_ids():return [f'{i:05d}' for i in range(SAMPLES_PER_SEED)]


def registered_records(manifest,count_key):
    require(manifest[count_key]==SAMPLES_PER_SEED,'Requested sample count changed')
    records=manifest['records'];ids=[r['sample_id'] for r in records]
    require(len(ids)==len(set(ids)),'Duplicate sample IDs')
    require(set(ids)<=set(expected_ids()),'Unregistered sample IDs')
    require(all(r['status'] in ('ok','failed') for r in records),'Unrecognized source status')
    return {r['sample_id']:r for r in records}


def verify_protocol(run,protocol):
    require(tuple(protocol['seeds'])==SEEDS,'Training seeds changed')
    require(protocol['ae']['steps']==3200 and protocol['fm']['steps']==8000,'Fixed AE/FM budget changed')
    require(protocol['sampling']['samples_per_seed']==64 and protocol['sampling']['euler_steps']==32,'Sampling contract changed')
    require(protocol['sampling']['noise_seed_by_training_seed']=={'10':73510,'11':73511},'Noise seeds changed')
    require(protocol['target_faces']==512 and protocol['shared_mainfield_per_seed_sample'] is True,'Common field/face budget changed')
    data=Path(protocol['data']).resolve()
    for rel,digest in protocol['data_hashes'].items():
        require(file_hash(inside(data,rel))==digest,'Dataset input changed: '+rel)
    module_root=Path(__file__).resolve().parents[1]
    for rel,digest in protocol['training_source_sha256'].items():
        require(file_hash(inside(module_root,rel))==digest,'Frozen generation/export source changed: '+rel)
    return data


def seed_inputs(run,seed,protocol):
    """Checkpoint and manifest bindings, without any model forward or sampling."""
    run=Path(run).resolve();folder=run/f'seed{seed}'
    paths={stage:folder/stage/f'step{protocol[stage]["steps"]:08d}.pt' for stage in ('ae','fm')}
    states={stage:torch.load(path,map_location='cpu',weights_only=True) for stage,path in paths.items()}
    hashes={stage:file_hash(path) for stage,path in paths.items()}
    complete=read_json(run/'TRAINING_COMPLETE.json')
    require(complete['status']=='COMPLETE' and complete['fixed_final_selection'] is True,'Fixed training completion missing')
    for stage,state in states.items():
        cfg=state['config'];settings=protocol[stage]
        require(state['step']==cfg['steps']==settings['steps'] and cfg['stage']==stage and cfg['seed']==seed,
                'Checkpoint stage/step/seed differs')
        require(cfg['dataset_sha256']==protocol['data_hashes']['radial_train.npz']
                and cfg['dataset_manifest_sha256']==protocol['data_hashes']['DATASET.json'],'Checkpoint dataset binding differs')
        require(cfg['batch']==settings['batch'] and cfg['lr']==settings['lr'],'Checkpoint fixed hyperparameters differ')
        matches=[r for r in complete['stages'] if r['seed']==seed and r['stage']==stage]
        require(len(matches)==1 and matches[0]['fixed_final'] is True and matches[0]['final_sha256']==hashes[stage],
                'Checkpoint differs from completed fixed final')
        require(inside(run,matches[0]['final_checkpoint'])==paths[stage],'Fixed final path differs')
        require(file_hash(paths[stage])==hashes[stage],'Checkpoint changed during read')
    require(states['fm']['config']['ae_sha256']==hashes['ae'],'FM is bound to another AE')
    sample_dir=folder/'samples';sample_path=sample_dir/'SAMPLES.json';samples=read_json(sample_path)
    require(samples['ae_sha256']==hashes['ae'] and samples['fm_sha256']==hashes['fm'],'Sample checkpoint binding differs')
    require(samples['dataset_sha256']==protocol['data_hashes']['radial_train.npz']
            and samples['dataset_manifest_sha256']==protocol['data_hashes']['DATASET.json'],'Sample dataset binding differs')
    require(samples['noise_seed']==protocol['sampling']['noise_seed_by_training_seed'][str(seed)]
            and samples['fm_steps']==samples['fm_nfe']==32,'Sample noise/Euler contract differs')
    sample_records=registered_records(samples,'requested')
    export_dir=folder/'exports';manifests={};maps={};manifest_hashes={}
    for arm in ARMS:
        path=export_dir/f'{arm}_MANIFEST.json';manifest=read_json(path)
        require(manifest['row']=='ours_'+arm,'Outlet manifest name differs')
        manifests[arm]=manifest;maps[arm]=registered_records(manifest,'requested_samples');manifest_hashes[arm]=file_hash(path)
    # CPU RNG only verifies the already registered noise; it does not regenerate a shape.
    generator=torch.Generator(device='cpu').manual_seed(samples['noise_seed'])
    expected_noise=torch.randn((64,64),generator=generator).numpy()
    return {'seed':seed,'sample_dir':sample_dir,'sample_manifest':samples,'sample_records':sample_records,
        'sample_manifest_sha256':file_hash(sample_path),'export_dir':export_dir,'outlet_records':maps,
        'outlet_manifest_sha256':manifest_hashes,'expected_noise':expected_noise,
        'checkpoint_sha256':hashes,'checkpoint_paths':{k:str(v.relative_to(run)) for k,v in paths.items()}}


def load_sample(inputs,sample_id):
    record=inputs['sample_records'].get(sample_id)
    require(record is not None,'Missing registered sample record')
    if record['status']!='ok':raise RuntimeError('Recorded sample generation failure: '+str(record.get('error')))
    path=inside(inputs['sample_dir'],record['path']);before=file_hash(path)
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)=={'noise','z_standardized','z','nodes'},'Sample array schema differs')
        arrays={k:np.array(archive[k]) for k in ('noise','z_standardized','z','nodes')}
    require(all(np.isfinite(x).all() for x in arrays.values()),'Nonfinite registered sample array')
    require(all(arrays[k].shape==(64,) for k in ('noise','z_standardized','z')) and arrays['nodes'].shape==(2050,),
            'Sample latent/node dimensions differ')
    require(array_hash(*arrays.values())==record['array_hash'],'Sample arrays changed')
    require(array_hash(arrays['nodes'])==record['nodes_hash'],'Sample mainfield nodes changed')
    require(np.array_equal(arrays['noise'],inputs['expected_noise'][int(sample_id)]),'Saved noise differs from fixed noise index')
    require(file_hash(path)==before,'Sample file changed during read')
    return arrays,{'sample_npz_sha256':before,'sample_array_hash':record['array_hash'],
                  'source_nodes_sha256':record['nodes_hash'],'sample_file':str(path.relative_to(inputs['sample_dir']))}


def verify_outlet_binding(inputs,arm,sample_id,sample_record):
    row=inputs['outlet_records'][arm].get(sample_id)
    require(row is not None,'Missing registered outlet attempt')
    manifest=inputs['sample_manifest']
    require(row['row']=='ours_'+arm and row['sample_id']==sample_id,'Outlet identity differs')
    for key,expected in [('source_bundle_sha256',inputs['sample_manifest_sha256']),
        ('source_nodes_sha256',sample_record.get('nodes_hash')),('ae_sha256',manifest['ae_sha256']),
        ('fm_sha256',manifest['fm_sha256']),('dataset_sha256',manifest['dataset_sha256']),
        ('dataset_manifest_sha256',manifest['dataset_manifest_sha256']),('requested_faces',512)]:
        require(row.get(key)==expected,'Outlet provenance mismatch: '+key)
    require(row.get('a_weight_seed')==(inputs['seed'] if arm=='a' else None),'Original A seed does not match training seed')
    return row


def load_mesh(inputs,row,arm,nodes,support):
    if row['status']!='ok':raise RuntimeError('Recorded outlet failure: '+str(row.get('error')))
    path=inside(inputs['export_dir'],row['mesh_path']);before=file_hash(path)
    require(before==row['mesh_sha256'],'Exported mesh file changed')
    with np.load(path,allow_pickle=False) as archive:
        mesh={k:np.array(archive[k]) for k in archive.files}
    vertices,faces=_arrays(mesh['vertices'],mesh['faces'])
    if arm in ('fixed','a'):
        base=fibonacci(258);template=hull_faces(base)
        require(np.array_equal(mesh['faces'],template),'Radial outlet connectivity differs from original template')
        require(mesh['u'].shape==(258,3) and np.isfinite(mesh['u']).all()
                and np.allclose(np.linalg.norm(mesh['u'],axis=1),1,rtol=0,atol=1e-12),'Invalid accepted directions')
        if arm=='fixed':require(np.array_equal(mesh['u'],base),'Fixed directions changed')
        queried=support.query(nodes,mesh['u'])
        require(np.array_equal(mesh['r'],queried) and np.array_equal(mesh['vertices'],mesh['u']*queried[:,None]),
                'Radial outlet radius is stale or differs from saved common field')
    # QEM intentionally has its own connectivity and need not store radial u/r.
    require(file_hash(path)==before,'Mesh changed during read')
    return vertices,faces,{'mesh_file':str(path.relative_to(inputs['export_dir'])),
        'mesh_npz_sha256':before,'stored_array_hash':array_hash(mesh['vertices'],mesh['faces']),
        'stored_array_hash_schema':'array_hash(vertices_as_saved,faces_as_saved); dtype and shape included',
        'canonical_geometry_hash':array_hash(vertices,faces),
        'canonical_geometry_hash_schema':'array_hash(vertices_float64,faces_int64); QEM connectivity independent',
        'stored_vertices_dtype':str(mesh['vertices'].dtype),'stored_faces_dtype':str(mesh['faces'].dtype),
        'accepted_radius_requery_exact':True if arm in ('fixed','a') else None,
        'connectivity_compared_to_fixed':arm in ('fixed','a')}


def audit_inputs(run,protocol):
    """Audit all registered files before any surface-distance or quality metric."""
    run=Path(run).resolve();data=verify_protocol(run,protocol);support=RadialSupport()
    records=[];manifests=[];errors=[]
    for seed in SEEDS:
        try:
            inputs=seed_inputs(run,seed,protocol)
            manifests.append({'seed':seed,'sample_manifest_sha256':inputs['sample_manifest_sha256'],
                'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],
                'checkpoint_sha256':inputs['checkpoint_sha256']})
        except Exception as exc:
            errors.append({'seed':seed,'stage':'seed_provenance','error':repr(exc)})
            for sid in expected_ids():
                for arm in ARMS:records.append({'seed':seed,'sample_id':sid,'arm':arm,'status':'failed','error':repr(exc)})
            continue
        for sid in expected_ids():
            sample_record=inputs['sample_records'].get(sid)
            arrays=None;sample_identity={};sample_error=None
            try:arrays,sample_identity=load_sample(inputs,sid)
            except Exception as exc:sample_error=repr(exc)
            for arm in ARMS:
                record={'seed':seed,'sample_id':sid,'arm':arm,**sample_identity}
                try:
                    require(sample_record is not None,'Missing source sample attempt')
                    source=verify_outlet_binding(inputs,arm,sid,sample_record)
                    record['source_export_status']=source['status']
                    if sample_error:raise RuntimeError(sample_error)
                    _,_,identity=load_mesh(inputs,source,arm,arrays['nodes'],support)
                    record.update(identity,status='ok')
                except Exception as exc:record.update(status='failed',error=repr(exc))
                records.append(record)
    require(len(records)==384,'Input audit lost fixed slots')
    return {'status':'PASS' if all(r['status']=='ok' for r in records) else 'FAILURES_RETAINED',
        'expected_registered_samples':128,'expected_outlet_attempts':384,'outlet_attempts':len(records),
        'successful_outlet_bindings':sum(r['status']=='ok' for r in records),
        'successful_sample_bindings':len({(r['seed'],r['sample_id']) for r in records if 'sample_npz_sha256' in r}),
        'seed_errors':errors,'manifests':manifests,'records':records,'shape_generation_executed':False,
        'surface_metrics_executed':False,'source_artifacts_modified':False,
        'generation_protocol_sha256':file_hash(run/'RUN_PROTOCOL.json')}


def extended_quantile(values,quantile):
    """Linear quantile in the extended reals, preserving degenerate +inf faces."""
    values=np.sort(np.asarray(values,np.float64));require(len(values)>0 and not np.isnan(values).any(),'Invalid quantile input')
    position=(len(values)-1)*quantile;lower=int(np.floor(position));upper=int(np.ceil(position));weight=position-lower
    if np.isinf(values[lower]) or (weight>0 and np.isinf(values[upper])):return None,True
    return float(values[lower]*(1-weight)+values[upper]*weight),False


def quality_tails(vertices,faces):
    vertices,faces=_arrays(vertices,faces);quality=mesh_quality(vertices,faces);tri=vertices[faces]
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    degenerate=area2<=quality['area2_degeneracy_tolerance']
    angles=[]
    for i in range(3):
        a,b=tri[:,(i+1)%3]-tri[:,i],tri[:,(i+2)%3]-tri[:,i]
        angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))))
    minimum=np.min(angles,axis=0);minimum[degenerate]=0
    squared_edges=np.sum((np.roll(tri,-1,axis=1)-tri)**2,axis=2)
    shape=np.divide(2*np.sqrt(3)*area2,squared_edges.sum(axis=1),out=np.zeros(len(faces)),where=squared_edges.sum(axis=1)>0)
    shape[degenerate]=0
    aspect=np.divide(squared_edges.max(axis=1),area2,out=np.full(len(faces),np.inf),where=~degenerate)
    p95,inf95=extended_quantile(aspect,.95);p99,inf99=extended_quantile(aspect,.99)
    return {**quality,'min_triangle_angle_deg':quality['min_angle_deg'],
        'triangle_min_angle_p01_deg':float(np.quantile(minimum,.01)),
        'triangle_min_angle_p05_deg':float(np.quantile(minimum,.05)),
        'shape_quality_p01':float(np.quantile(shape,.01)),'shape_quality_p05':float(np.quantile(shape,.05)),
        'aspect_ratio_p95':p95,'aspect_ratio_p99':p99,'aspect_ratio_p95_is_infinite':inf95,
        'aspect_ratio_p99_is_infinite':inf99,'aspect_ratio_infinite_face_count':int(np.isinf(aspect).sum()),
        'aspect_ratio_definition':'longest edge / shortest altitude = longest_edge_squared / (2*area); equilateral=2/sqrt(3)',
        'quantile_definition':'linear quantile over every actual face; degenerate min-angle/q=0 and aspect=+infinity; infinity stored null plus explicit flag'}


def common_gate(quality,audit,fixed_quality,fixed_audit):
    if fixed_quality is None or fixed_audit is None or fixed_audit.get('valid') is not True:
        return {'qualified':None,'gate_available':False,'failure_reasons':['same_field_Fixed_reference_missing_or_invalid']}
    reasons=[]
    if quality['face_count']!=512:reasons.append('actual_face_budget')
    if audit.get('valid') is not True:reasons.append('geometry_validity')
    if quality['degenerate_count']:reasons.append('degenerate_faces')
    for key in ('min_triangle_angle_deg','triangle_min_angle_p05_deg'):
        if quality[key]+QUALITY_EPS_DEG<fixed_quality[key]:reasons.append(key)
    return {'qualified':not reasons,'gate_available':True,'failure_reasons':reasons,
        'min_angle_margin_deg':quality['min_triangle_angle_deg']-fixed_quality['min_triangle_angle_deg'],
        'p05_angle_margin_deg':quality['triangle_min_angle_p05_deg']-fixed_quality['triangle_min_angle_p05_deg']}


def paired_metrics(row,fixed):
    if fixed is None or row.get('metric_status')!='ok' or fixed.get('metric_status')!='ok':
        return {'paired_available':False,'rms_gain_percent_two_measurements':None,'qualified_ge2pct_both':False}
    require(row['seed']==fixed['seed'] and row['sample_id']==fixed['sample_id']
            and row['source_nodes_sha256']==fixed['source_nodes_sha256'],'Pair uses different saved mainfields')
    require([x['seed'] for x in row['surface_metrics']]==[x['seed'] for x in fixed['surface_metrics']],'Pair uses different MC seeds')
    base=np.array([x['rms_bbox_percent'] for x in fixed['surface_metrics']]);actual=np.array([x['rms_bbox_percent'] for x in row['surface_metrics']])
    require(len(base)==len(actual)==2 and np.isfinite(base).all() and np.isfinite(actual).all(),'Invalid paired metric vector')
    if (base<=0).any():return {'paired_available':False,'rms_gain_percent_two_measurements':None,'qualified_ge2pct_both':False,'zero_fixed_rms':True}
    gains=100*(1-actual/base)
    return {'paired_available':True,'rms_gain_percent_two_measurements':gains.tolist(),
        'rms_absolute_difference_bbox_percent':(actual-base).tolist(),
        'improved_both':bool((gains>0).all()),
        'qualified_ge2pct_both':bool(row.get('qualified') is True and (gains>=2).all())}


def summary_groups(rows):
    groups=[]
    for seed in ('pooled',10,11):
        denominator=128 if seed=='pooled' else 64
        for arm in ARMS:
            selected=[r for r in rows if r['arm']==arm and (seed=='pooled' or r['seed']==seed)]
            good=[r for r in selected if r.get('metric_status')=='ok'];paired=[r for r in selected if r.get('paired_available')]
            groups.append({'seed':seed,'arm':arm,'planned_samples':denominator,'recorded_attempts':len(selected),
                'metric_success_count':len(good),'metric_failed_or_missing_count':denominator-len(good),
                'actual_face_count_histogram':{str(n):sum(r.get('quality',{}).get('face_count')==n for r in selected)
                    for n in sorted({r['quality']['face_count'] for r in selected if 'quality' in r})},
                'exact_valid_count':sum(r.get('audit',{}).get('valid') is True for r in selected),
                'qualified_count':sum(r.get('qualified') is True for r in selected),
                'gate_unavailable_count':sum(not r.get('gate_available',False) for r in selected),
                'paired_count':len(paired),'improved_both_count':sum(r.get('improved_both',False) for r in paired),
                'qualified_ge2pct_both_count':sum(r.get('qualified_ge2pct_both',False) for r in paired),
                'paired_gain_percent_median_two_measurements':[float(np.median([r['rms_gain_percent_two_measurements'][k] for r in paired])) if paired else None for k in (0,1)],
                'paired_gain_percent_mean_two_measurements':[float(np.mean([r['rms_gain_percent_two_measurements'][k] for r in paired])) if paired else None for k in (0,1)],
                'surface_rms_bbox_percent_median_two_measurements':[float(np.median([r['surface_metrics'][k]['rms_bbox_percent'] for r in good])) if good else None for k in (0,1)]})
    return groups


def run_metrics(run,out=None,distribution=False):
    run=Path(run).resolve();protocol=read_json(run/'RUN_PROTOCOL.json');out=new_dir(out or run/'metrics')
    ph=file_hash(run/'RUN_PROTOCOL.json');global_error=None;data=None
    try:
        data=verify_protocol(run,protocol)
        preflight=audit_inputs(run,protocol)
    except Exception as exc:
        global_error={'error':repr(exc),'traceback':traceback.format_exc()}
        preflight={'status':'GLOBAL_PROVENANCE_ERROR','expected_registered_samples':128,'expected_outlet_attempts':384,**global_error}
    write_json(out/'INPUT_AUDIT.json',preflight)
    print(json.dumps({'event':'input_audit_complete','status':preflight['status'],
                      'successful_sample_bindings':preflight.get('successful_sample_bindings',0),
                      'successful_outlet_bindings':preflight.get('successful_outlet_bindings',0)}),flush=True)
    support=RadialSupport();support_sha=file_hash(ASSETS/'support.npz');rows=[];input_records=[];errors=[]
    write_json(out/'METRIC_PROTOCOL.json',{'scope':'CLOSED16_AE_FM_OUTLET_ONLY_NO_EXTERNAL_RANKING',
        'generation_protocol_sha256':ph,'seeds':list(SEEDS),'samples_per_seed':64,'expected_samples':128,'expected_outlet_attempts':384,
        'arms':list(ARMS),'surface_reference':'same saved nodes mapped by original RadialSupport.mesh; no new AE/FM pass',
        'points_each_direction':POINTS,'MC_seed_bases':list(SEED_BASES),'seed_rule':'base + training_seed_index*64 + integer(sample_id)',
        'distance':'nearest point on full triangle surfaces, not sampled-point or nearest-vertex distance',
        'symmetric_RMS':'sqrt((mean(mesh_to_field_distance_squared)+mean(field_to_mesh_distance_squared))/2)',
        'normalization':'same-field support mesh bbox diagonal only; no per-outlet rescaling',
        'quality_gate':'exact endpoint valid; actual512; no degenerate; minimum angle and per-face-min p05 no worse than same-field Fixed, tolerance1e-6 degrees',
        'QEM_same_connectivity_required':False,'quality_tail_scope':'all actual faces, including degenerate faces',
        'formal_end_to_end_timing_computed':False,'parallel_export_seconds_are_not_formal_timing':True,
        'PCA_already_interpolates_train16':True,'AE_low_training_loss_is_not_new_learning_capacity_evidence':True,
        'support_file_sha256':support_sha,'a_assets_current_sha256':{str(seed):file_hash(ASSETS/f'original_A_{seed}.pt') for seed in SEEDS},
        'a_asset_note':'original export ledger binds archived A by seed; these asset file hashes are captured at offline evaluation',
        'metric_source_sha256':file_hash(__file__),'geometry_source_sha256':file_hash(Path(__file__).parents[1]/'layout_value/geometry.py')})
    for seed_index,seed in enumerate(SEEDS):
        seed_error=global_error;inputs=None
        if seed_error is None:
            try:
                inputs=seed_inputs(run,seed,protocol)
                input_records.append({'seed':seed,'sample_manifest_sha256':inputs['sample_manifest_sha256'],
                    'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],'checkpoint_sha256':inputs['checkpoint_sha256'],
                    'missing_sample_ids':sorted(set(expected_ids())-set(inputs['sample_records'])),
                    'missing_outlet_ids':{a:sorted(set(expected_ids())-set(inputs['outlet_records'][a])) for a in ARMS}})
            except Exception as exc:seed_error={'error':repr(exc),'traceback':traceback.format_exc()}
        if seed_error:errors.append({'stage':'seed_provenance','seed':seed,**seed_error})
        for sid in expected_ids():
            index=seed_index*64+int(sid);sample_error=seed_error;arrays=None;sample_identity={}
            if sample_error is None:
                try:
                    arrays,sample_identity=load_sample(inputs,sid)
                    reference_v,reference_f=support.mesh(arrays['nodes'])
                    require(np.isfinite(reference_v).all(),'Nonfinite support reference')
                except Exception as exc:sample_error={'error':repr(exc),'traceback':traceback.format_exc()}
            local=[]
            for arm in ARMS:
                row={'seed':seed,'sample_id':sid,'global_sample_index':index,'arm':arm,'requested_faces':512,
                    'metric_status':'failed','qualified':None,'gate_available':False,**sample_identity,
                    'generation_protocol_sha256':ph,'no_retry_or_output_modification':True}
                try:
                    if sample_error:raise RuntimeError(sample_error['error'])
                    source=verify_outlet_binding(inputs,arm,sid,inputs['sample_records'][sid])
                    row['source_export_status']=source['status'];row['source_export_error']=source.get('error')
                    row['source_bundle_sha256']=inputs['sample_manifest_sha256'];row['source_nodes_sha256']=source['source_nodes_sha256']
                    v,f,identity=load_mesh(inputs,source,arm,arrays['nodes'],support);row.update(identity)
                    row['quality']=quality_tails(v,f);row['audit']=audit_mesh(v,f,exact=True)
                    row['reference_surface_hash']=array_hash(reference_v,reference_f)
                    row['surface_metrics']=[evaluate_mesh(v,f,reference_v,reference_f,points=POINTS,seed=base+index) for base in SEED_BASES]
                    row['metric_status']='ok'
                except Exception as exc:
                    row['metric_error']=repr(exc);row['metric_traceback']=traceback.format_exc()
                local.append(row)
            fixed=local[0]
            for row in local:
                if 'quality' in row and 'audit' in row:
                    row.update(common_gate(row['quality'],row['audit'],fixed.get('quality'),fixed.get('audit')))
                row.update(paired_metrics(row,fixed))
                write_json(out/'attempts'/f'seed{seed}'/sid/(row['arm']+'.json'),row)
                rows.append(row)
            if (int(sid)+1)%16==0:
                print(json.dumps({'event':'surface_metrics_progress','seed':seed,'samples_completed_in_seed':int(sid)+1,
                    'outlet_rows_completed':len(rows),'metric_failures':sum(r['metric_status']!='ok' for r in rows)}),flush=True)
    require(len(rows)==384 and len({(r['seed'],r['sample_id'],r['arm']) for r in rows})==384,'Predeclared metric slots disappeared')
    require(file_hash(run/'RUN_PROTOCOL.json')==ph and file_hash(ASSETS/'support.npz')==support_sha,'Input changed during metric evaluation')
    result={'rows':rows,'expected_samples':128,'expected_attempts':384,'attempt_count':384,
        'metric_success_count':sum(r['metric_status']=='ok' for r in rows),'provenance_errors':errors,
        'failed_slots':[[r['seed'],r['sample_id'],r['arm']] for r in rows if r['metric_status']!='ok'],
        'registered_input_manifests':input_records,'generation_protocol_sha256':ph,
        'external_ranking_qualified':False,'formal_timing_available':False}
    write_json(out/'RESULTS.json',result)
    groups=summary_groups(rows);write_json(out/'SUMMARY.json',{'groups':groups,'results_sha256':file_hash(out/'RESULTS.json'),
        'closed_set_only':True,'same_mainfield_paired_comparison':True,'QEM_connectivity_independent':True,
        'PCA_interpolation_is_initial_reference_not_new_learning_result':True,'external_baseline_victory_claim':False,
        'formal_end_to_end_timing_available':False,'two_MC_measurements_not_independent_training_runs':True})
    distribution_records=[]
    if distribution:
        require(global_error is None and not errors,'Distribution metrics blocked by provenance errors')
        for seed in SEEDS:
            target=run/f'seed{seed}'/'distribution'
            command=[sys.executable,'-B','-m','experiments.a_radial.evaluate','--data',str(data),'--ours',str(run/f'seed{seed}'/'exports'),
                     '--out',str(target),'--exact','--points','2048']
            completed=subprocess.run(command,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
            write_json(out/'distribution'/f'seed{seed}_INVOCATION.json',{'command':command,'returncode':completed.returncode,
                'stdout':completed.stdout,'stderr':completed.stderr,'original_evaluate_sha256':file_hash(Path(__file__).parents[1]/'evaluate.py')})
            distribution_records.append({'seed':seed,'returncode':completed.returncode,'path':str(target.relative_to(run))})
        write_json(out/'DISTRIBUTION_STATUS.json',distribution_records)
    lines=['AE/FM闭集三出口：同主场离线评测','',
        f"登记128个生成样本、384个出口尝试；指标成功 {result['metric_success_count']}，失败 {len(result['failed_slots'])}。全部失败保留，没有重采样、修复或替换输出。",'',
        '|训练种子|出口|精确有效/计划|共同质量合格/计划|配对数|相对Fixed的RMS改善中位1/2 (%)|合格且双次≥2%|',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        gain=g['paired_gain_percent_median_two_measurements'];text=' / '.join('-' if x is None else f'{x:.3f}' for x in gain)
        lines.append(f"|{g['seed']}|{g['arm']}|{g['exact_valid_count']}/{g['planned_samples']}|{g['qualified_count']}/{g['planned_samples']}|{g['paired_count']}|{text}|{g['qualified_ge2pct_both_count']}|")
    lines += ['', 'RMS以各自共享的2050节点主场三角面为参考，双向面积采样各8192点，并计算到完整三角面的最近点距离；两次固定MC测量均保留。这是表面积分的MC估计，不是解析积分或精确Hausdorff距离。','',
        '质量统计保留实际面数、精确端点相交/退化、最小角及每面最小角p01/p05、形状质量q01/q05和长宽比p95/p99。QEM不要求与Fixed相同连接；共同门槛相对同一主场的实际Fixed输出，缺失或无效Fixed时门槛不可用。','',
        '16个训练对象已由PCA初始化插值，AE低训练损失不能独立证明新增学习能力。该表不支持未知类别泛化或外部MeshFlow基线胜出结论。并发导出耗时不作为正式冷/热CPU-GPU完整计时。','',
        '文件：[完整指标](RESULTS.json)、[汇总](SUMMARY.json)、[指标协议](METRIC_PROTOCOL.json)。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'metrics_dir':str(out),'attempts':384,'metric_success':result['metric_success_count'],'failures':len(result['failed_slots'])}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--out');parser.add_argument('--distribution',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);run_metrics(args.run,args.out,args.distribution)


if __name__=='__main__':main()
