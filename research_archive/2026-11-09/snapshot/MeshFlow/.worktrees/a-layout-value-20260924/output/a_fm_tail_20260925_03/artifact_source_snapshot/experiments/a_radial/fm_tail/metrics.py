"""Offline fixed-final64k primary metrics and separate same-noise8k pairing.

Frozen common geometry functions are reused. The original8k evaluator and every
source artifact remain unchanged; auxiliary128 fields never enter these rows.
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
from ..exporter import RadialSupport
from ..io_utils import array_hash,file_hash,new_dir,write_json
from ..system_probe import ours_metrics as common
from . import artifacts


def primary_only(inputs):
    common.require(inputs.get('primary') is True,'Auxiliary or unregistered fields cannot enter primary metrics')
    identity=inputs['identity'];manifest=inputs['sample_manifest']
    common.require(identity['scope']==artifacts.ARTIFACT_SCOPE and identity['FM_step']==64000 and identity['AE_step']==3200,
                   'Wrong final generation identity')
    common.require(manifest['fm_steps']==manifest['fm_nfe']==32,'Primary evaluation requires registered Euler32')
    return inputs


def seed_inputs(run,seed):
    inputs=primary_only(artifacts.view_samples(run,seed));folder=artifacts.seed_folder(run,seed)
    identity=inputs['identity'];record=common.read_json(folder/'EXPORT_IDENTITY.json')
    expected={'schema':artifacts.SCHEMA,'scope':artifacts.ARTIFACT_SCOPE,
        'training_protocol_sha256':identity['training_protocol_sha256'],
        'artifact_protocol_sha256':identity['artifact_protocol_sha256'],
        'original_A_sha256':identity['original_A_sha256'],'support_sha256':identity['support_sha256'],
        'source_bundle_sha256':inputs['sample_manifest_sha256'],'one_primary_mainfield_three_outlets':True}
    for key,value in expected.items():common.require(record.get(key)==value,'Export identity differs: '+key)
    folder=folder/'exports';maps={};hashes={}
    for arm in common.ARMS:
        path=folder/f'{arm}_MANIFEST.json';manifest=common.read_json(path);hashes[arm]=file_hash(path)
        common.require(manifest['row']=='ours_'+arm,'Wrong outlet name')
        common.require(hashes[arm]==record['outlet_manifest_sha256'][arm],'Outlet manifest changed since export')
        maps[arm]=common.registered_records(manifest,'requested_samples')
    return {**inputs,'export_dir':folder,'outlet_records':maps,'outlet_manifest_sha256':hashes,
        'export_identity_sha256':file_hash(folder.parent/'EXPORT_IDENTITY.json')}


def manifest_record(inputs):
    return {'seed':inputs['seed'],'sample_manifest_sha256':inputs['sample_manifest_sha256'],
        'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],
        'export_identity_sha256':inputs['export_identity_sha256'],'identity':inputs['identity'],
        'missing_sample_ids':sorted(set(common.expected_ids())-set(inputs['sample_records'])),
        'missing_outlet_ids':{a:sorted(set(common.expected_ids())-set(inputs['outlet_records'][a])) for a in common.ARMS}}


def audit_inputs(run):
    p,ap,sha=artifacts.artifact_contract(run);support=RadialSupport();rows=[];errors=[];manifests=[]
    for seed in common.SEEDS:
        try:inputs=seed_inputs(run,seed);manifests.append(manifest_record(inputs))
        except Exception as exc:
            errors.append({'seed':seed,'error':repr(exc)})
            for sid in common.expected_ids():
                for arm in common.ARMS:rows.append({'seed':seed,'sample_id':sid,'arm':arm,'status':'failed','error':repr(exc)})
            continue
        for sid in common.expected_ids():
            arrays=None;sample_identity={};error=None
            try:arrays,sample_identity=common.load_sample(inputs,sid)
            except Exception as exc:error=repr(exc)
            for arm in common.ARMS:
                row={'seed':seed,'sample_id':sid,'arm':arm,**sample_identity}
                try:
                    sample_record=inputs['sample_records'].get(sid)
                    common.require(sample_record is not None,'Missing registered sample attempt')
                    source=common.verify_outlet_binding(inputs,arm,sid,sample_record)
                    row['source_export_status']=source['status']
                    if error:raise ValueError(error)
                    _,_,mesh_identity=common.load_mesh(inputs,source,arm,arrays['nodes'],support)
                    row.update(mesh_identity,status='ok')
                except Exception as exc:row.update(status='failed',error=repr(exc))
                rows.append(row)
    common.require(len(rows)==384,'Fixed input audit slots disappeared')
    return {'status':'PASS' if all(r['status']=='ok' for r in rows) else 'FAILURES_RETAINED',
        'expected_registered_samples':128,'expected_outlet_attempts':384,'outlet_attempts':len(rows),
        'successful_outlet_bindings':sum(r['status']=='ok' for r in rows),
        'successful_sample_bindings':len({(r['seed'],r['sample_id']) for r in rows if 'sample_npz_sha256' in r}),
        'artifact_protocol_sha256':sha,'training_protocol_sha256':ap['training_protocol_sha256'],
        'seed_errors':errors,'manifests':manifests,'records':rows,'shape_generation_executed':False,
        'auxiliary_excluded':True,'source_artifacts_modified':False}


def collect_rows(run,out,training_sha,artifact_sha,global_error=None):
    """Keep all384 predeclared slots, including provenance/missing/geometry errors."""
    support=RadialSupport();rows=[];manifests=[];errors=[]
    for seed_index,seed in enumerate(common.SEEDS):
        inputs=None;seed_error=global_error
        if seed_error is None:
            try:inputs=seed_inputs(run,seed);manifests.append(manifest_record(inputs))
            except Exception as exc:seed_error=repr(exc)
        if seed_error:errors.append({'seed':seed,'stage':'seed_provenance','error':seed_error})
        for sid in common.expected_ids():
            index=seed_index*64+int(sid);sample_error=seed_error;arrays=None;sample_identity={}
            if sample_error is None:
                try:
                    arrays,sample_identity=common.load_sample(inputs,sid)
                    reference_v,reference_f=support.mesh(arrays['nodes'])
                    common.require(np.isfinite(reference_v).all(),'Nonfinite saved-field surface')
                except Exception as exc:sample_error=repr(exc)
            local=[]
            for arm in common.ARMS:
                row={'seed':seed,'sample_id':sid,'global_sample_index':index,'arm':arm,'requested_faces':512,
                    'metric_status':'failed','qualified':None,'gate_available':False,**sample_identity,
                    'generation_protocol_sha256':training_sha,'artifact_protocol_sha256':artifact_sha,
                    'AE_step':3200,'FM_step':64000,'euler_steps':32,'auxiliary_excluded':True,
                    'no_retry_or_output_modification':True}
                try:
                    if sample_error:raise ValueError(sample_error)
                    row['generation_identity']=inputs['identity']
                    source=common.verify_outlet_binding(inputs,arm,sid,inputs['sample_records'][sid])
                    row.update(source_export_status=source['status'],source_export_error=source.get('error'),
                        source_bundle_sha256=inputs['sample_manifest_sha256'],source_nodes_sha256=source['source_nodes_sha256'])
                    vertices,faces,mesh_identity=common.load_mesh(inputs,source,arm,arrays['nodes'],support)
                    row.update(mesh_identity);row['quality']=common.quality_tails(vertices,faces)
                    row['audit']=common.audit_mesh(vertices,faces,exact=True)
                    row['reference_surface_hash']=array_hash(reference_v,reference_f)
                    row['surface_metrics']=[common.evaluate_mesh(vertices,faces,reference_v,reference_f,
                        points=common.POINTS,seed=base+index) for base in common.SEED_BASES]
                    row['metric_status']='ok'
                except Exception as exc:row.update(metric_error=repr(exc),metric_traceback=traceback.format_exc())
                local.append(row)
            fixed=local[0]
            for row in local:
                if 'quality' in row and 'audit' in row:
                    row.update(common.common_gate(row['quality'],row['audit'],fixed.get('quality'),fixed.get('audit')))
                row.update(common.paired_metrics(row,fixed))
                write_json(Path(out)/'attempts'/f'seed{seed}'/sid/(row['arm']+'.json'),row);rows.append(row)
            if (int(sid)+1)%16==0:
                print(json.dumps({'event':'tail_primary_metrics','seed':seed,'completed_samples_in_seed':int(sid)+1,
                    'rows':len(rows),'metric_failures':sum(r['metric_status']!='ok' for r in rows)}),flush=True)
    common.require(len(rows)==384 and len({(r['seed'],r['sample_id'],r['arm']) for r in rows})==384,'Lost or duplicate primary slots')
    return rows,manifests,errors


def distribution(run,out,artifact_protocol):
    records=[]
    for seed in common.SEEDS:
        inputs=seed_inputs(run,seed);primary_only(inputs)
        target=Path(out)/'distribution'/f'seed{seed}'
        command=[sys.executable,'-B','-m','experiments.a_radial.evaluate','--data',artifact_protocol['evaluation_data'],
            '--ours',str(inputs['export_dir']),'--out',str(target),'--exact','--points','2048']
        result=subprocess.run(command,cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True)
        record={'seed':seed,'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'generation_identity':inputs['identity'],'sample_manifest_sha256':inputs['sample_manifest_sha256'],
            'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],'euler_steps':32,'auxiliary_excluded':True,
            'original_evaluate_sha256':file_hash(Path(__file__).parents[1]/'evaluate.py')}
        write_json(Path(out)/'distribution'/f'seed{seed}_INVOCATION.json',record);records.append(record)
    write_json(Path(out)/'DISTRIBUTION_STATUS.json',records)
    return records


def run_metrics(run,out=None,with_distribution=False):
    run=Path(run).resolve();out=new_dir(out or run/'artifacts'/'metrics');global_error=None;ap=None
    training_sha=file_hash(run/'RUN_PROTOCOL.json');artifact_sha=file_hash(run/'ARTIFACT_PROTOCOL.json')
    try:
        _,ap,actual_sha=artifacts.artifact_contract(run);common.require(actual_sha==artifact_sha,'Artifact protocol changed')
        preflight=audit_inputs(run)
    except Exception as exc:
        global_error=repr(exc);preflight={'status':'GLOBAL_PROVENANCE_ERROR','error':global_error,
            'traceback':traceback.format_exc(),'expected_registered_samples':128,'expected_outlet_attempts':384}
    write_json(out/'INPUT_AUDIT.json',preflight)
    write_json(out/'METRIC_PROTOCOL.json',{'scope':artifacts.ARTIFACT_SCOPE,'generation_protocol_sha256':training_sha,
        'artifact_protocol_sha256':artifact_sha,'AE_step':3200,'FM_step':64000,'euler_steps':32,
        'seeds':[10,11],'samples_per_seed':64,'expected_samples':128,'expected_outlet_attempts':384,'arms':list(common.ARMS),
        'surface_reference':'same saved2050 logradius nodes, original RadialSupport.mesh, no AE/FM forward',
        'points_each_direction':common.POINTS,'MC_seed_bases':list(common.SEED_BASES),
        'seed_rule':'base + seed_index*64 + integer(sample_id)',
        'distance':'nearest full triangle surface; symmetric RMS=sqrt((mean forward squared+mean reverse squared)/2)',
        'normalization':'same-field support mesh bbox diagonal, no per-outlet rescaling',
        'quality_gate':'exact endpoint valid, actual512, nondegenerate, min/p05 per-face-min angle no worse than same-field Fixed within1e-6 degree',
        'QEM_connectivity_independent':True,'auxiliary_excluded':True,'failures_keep_fixed_denominators':True,
        'metric_source_sha256':file_hash(__file__),'shared_metric_source_sha256':file_hash(common.__file__),
        'geometry_source_sha256':file_hash(Path(__file__).parents[1]/'layout_value/geometry.py'),
        'not_formal_end_to_end_timing':True,'closed_set_only':True,'no_external_baseline_victory_claim':True})
    rows,manifests,errors=collect_rows(run,out,training_sha,artifact_sha,global_error)
    common.require(file_hash(run/'RUN_PROTOCOL.json')==training_sha and file_hash(run/'ARTIFACT_PROTOCOL.json')==artifact_sha,
                   'Protocols changed during metrics')
    if global_error is None:artifacts.artifact_contract(run)
    result={'rows':rows,'expected_samples':128,'expected_attempts':384,'attempt_count':384,
        'metric_success_count':sum(r['metric_status']=='ok' for r in rows),'provenance_errors':errors,
        'failed_slots':[[r['seed'],r['sample_id'],r['arm']] for r in rows if r['metric_status']!='ok'],
        'registered_input_manifests':manifests,'generation_protocol_sha256':training_sha,'artifact_protocol_sha256':artifact_sha,
        'AE_step':3200,'FM_step':64000,'euler_steps':32,'auxiliary_excluded':True,'external_ranking_qualified':False,
        'formal_timing_available':False}
    write_json(out/'RESULTS.json',result)
    write_json(out/'SUMMARY.json',{'groups':common.summary_groups(rows),'results_sha256':file_hash(out/'RESULTS.json'),
        'closed_set_only':True,'same_mainfield_paired_comparison':True,'auxiliary_excluded':True,
        'QEM_connectivity_independent':True,'two_MC_measurements_not_independent_training_runs':True,
        'external_baseline_victory_claim':False,'formal_end_to_end_timing_available':False})
    if with_distribution:
        common.require(global_error is None and not errors,'Provenance errors block distribution metrics')
        distribution(run,out,ap)
    return result


def paired_runs(run,out=None):
    """Compare8k and64k same epsilon; each has its own mainfield, not a shared reference."""
    run=Path(run).resolve();p,ap,sha=artifacts.artifact_contract(run);source=Path(ap['source_run'])
    old_protocol=common.read_json(source/'RUN_PROTOCOL.json')
    common.require(file_hash(source/'RUN_PROTOCOL.json')==ap['source_training_protocol_sha256'],'Source8k protocol changed')
    common.verify_protocol(source,old_protocol)
    out=new_dir(out or run/'artifacts'/'paired_8k_64k');rows=[];manifests=[];errors=[]
    with np.load(run/'data/radial_train.npz',allow_pickle=False) as z:training=z['logr'];uids=z['uids'].tolist()
    for seed in common.SEEDS:
        error=None;before=after=None
        try:
            before=common.seed_inputs(source,seed,old_protocol);after=primary_only(artifacts.view_samples(run,seed))
            for key in ('ae_sha256','dataset_sha256','dataset_manifest_sha256'):
                common.require(before['sample_manifest'][key]==after['sample_manifest'][key],'Cross-run binding differs: '+key)
            common.require(before['checkpoint_sha256']['fm']==p['jobs'][str(seed)]['parent_sha256'],'Old FM is not registered parent8000')
            manifests.append({'seed':seed,'first_FM_step':8000,'second_FM_step':64000,
                'first_sample_manifest_sha256':before['sample_manifest_sha256'],
                'second_sample_manifest_sha256':after['sample_manifest_sha256'],'tail_identity':after['identity']})
        except Exception as exc:error=repr(exc);errors.append({'seed':seed,'error':error})
        for sid in common.expected_ids():
            row={'seed':seed,'sample_id':sid,'status':'failed','first_FM_step':8000,'second_FM_step':64000,
                'euler_steps':32,'same_mainfield_reference':False,'not_same_field_outlet_improvement':True}
            try:
                if error:raise ValueError(error)
                a,ai=common.load_sample(before,sid);b,bi=common.load_sample(after,sid)
                row.update(artifacts.compare_arrays(a,b,training),status='ok',
                    first_sample_sha256=ai['sample_npz_sha256'],second_sample_sha256=bi['sample_npz_sha256'])
            except Exception as exc:row['error']=repr(exc)
            write_json(out/'attempts'/f'seed{seed}'/(sid+'.json'),row);rows.append(row)
    common.require(len(rows)==128,'Lost cross-run comparison slots')
    groups=[]
    for seed in ['pooled',10,11]:
        selected=[r for r in rows if seed=='pooled' or r['seed']==seed];good=[r for r in selected if r['status']=='ok']
        groups.append({'seed':seed,'planned':128 if seed=='pooled' else 64,'successful':len(good),'failed':len(selected)-len(good),
            'paired_node_RMS_mean':float(np.mean([r['paired_logradius_node_RMS'] for r in good])) if good else None,
            'nearest_parent_counts':{side:{uid:sum(r[side+'_nearest_parent_index']==i for r in good) for i,uid in enumerate(uids)} for side in ('first','second')},
            'nearest_training_logradius_RMS_mean':{side:float(np.mean([r[side+'_nearest_training_logradius_RMS'] for r in good])) if good else None for side in ('first','second')}})
    result={'scope':'CLOSED16_SAME_NOISE_8K_64K_NODE_DIAGNOSTIC','artifact_protocol_sha256':sha,
        'source_training_protocol_sha256':ap['source_training_protocol_sha256'],'rows':rows,'planned':128,
        'successful':sum(r['status']=='ok' for r in rows),'groups':groups,'input_manifests':manifests,'errors':errors,
        'first_step':8000,'second_step':64000,'euler_steps':32,'auxiliary_excluded':True,
        'own_mainfield_each_run_not_same_field_outlet_gain':True,
        'nearest_training_support_is_not_a_distribution_or_generalization_score':True,
        'successful_subset_means_and_failed_counts_both_reported':True,'no_checkpoint_or_sample_selection':True}
    write_json(out/'RESULTS.json',result);artifacts.artifact_contract(run);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['audit','metrics','pair'])
    parser.add_argument('--run',required=True);parser.add_argument('--out');parser.add_argument('--distribution',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.command=='audit':
        result=audit_inputs(args.run);out=new_dir(args.out or Path(args.run)/'artifacts'/'input_audit');write_json(out/'INPUT_AUDIT.json',result)
    elif args.command=='metrics':run_metrics(args.run,args.out,args.distribution)
    else:paired_runs(args.run,args.out)


if __name__=='__main__':main()
