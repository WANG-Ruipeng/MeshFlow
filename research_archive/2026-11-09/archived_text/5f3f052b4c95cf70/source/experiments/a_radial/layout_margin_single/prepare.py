"""Prepare eight single-parent CPU fits from one shared v1 model and two targets."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,new_dir,save_npz,write_json
from ..legacy.networks import point_features
from ..layout_student.train import model_sha256
from ..layout_student.deployment import named_quality,qualification
from ..exporter import RadialSupport

UIDS=('7f1be3e7b6adeeae9bb399e812556da2','2379915c82c2b2d02530f02db7e9157e',
      '8c919e992db7cb081e5b68ee10358ea3','736820a85063a4371e5b68ee10358ea3')
KINDS=('old','margin')
ARCHIVE_SHA='1713e7fc7bad1ecdf4e6073372c3c1da071ceff10651147f0816c05d80a2c826'
MODEL_SHA='7c02c7926d11c76f22d520a024fa7fac5d4c08c69050cb3745c226a4e6c1b760'
FROZEN_DEPENDENCIES=('layout_fitdiag/train.py','layout_endpoint/train.py','layout_endpoint/rollout.py','layout_student/train.py',
    'layout_value/optimizer.py','legacy/networks.py','legacy/chart.py','io_utils.py')


def require(ok,message):
    if not ok:raise ValueError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def config_hash(training,loss,jobs):
    return hashlib.sha256(json.dumps({'training':training,'loss':loss,'jobs':jobs},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def job_table(initial_file_sha,initial_model_sha):
    return {f'{kind}_{uid[:8]}':{'uid':uid,'index':index,'seed':10,'target_kind':kind,
        'target_key':'directions1_'+kind,'active_training_uids':[uid],'initial_checkpoint':'initial/common_v1_model.pt',
        'initial_checkpoint_sha256':initial_file_sha,'initial_model_sha256':initial_model_sha,
        'start_update':0,'total_updates':20000,'initialization_mode':'shared_model_fresh_cpu_AdamW',
        'pair_id':uid,'diagnostic_uids':[uid]} for index,uid in enumerate(UIDS) for kind in KINDS}


def assert_contract(p):
    require(p['schema']=='A-MARGIN-SINGLE-v1' and p['train_uids']==list(UIDS),'Unexpected four-parent study')
    require(p['jobs']==job_table(p['hashes']['initial_checkpoint_sha256'],p['hashes']['initial_model_sha256']),
        'Registered eight jobs changed')
    t=p['training'];require(t['batch_size']==4 and t['total_updates']==20000 and t['learning_rate']==1e-4
        and t['weight_decay']==1e-4 and t['device']=='cpu' and t['torch_threads']==t['interop_threads']==1
        and t['checkpoint_updates']==[0,5000,20000],'Training budget/device changed')
    require(p['hashes']['config_sha256']==config_hash(t,p['loss'],p['jobs']),'Config digest changed')
    require(p['hashes']['initial_model_sha256']==MODEL_SHA and p['parent_joint_archive_sha256']==ARCHIVE_SHA,
        'Shared initialization or target lineage changed')
    require(p['loss']['name']=='c0_unscaled_discrete_bridge_velocity_MSE' and p['loss']['bridge_steps']==32
        and p['loss']['endpoint_scale_unused_by_c0'] is True,'C0 objective changed')
    e=p['evaluation'];require(e['planned_endpoints']==20 and e['points_each_direction']==8192
        and e['seed_bases']==[1593501,1693501] and e['deployment_threads']==4 and e['deployment_steps']==32
        and e['fallback'] is False,'Final deployment/evaluation changed')
    require(p['diagnostics']['updates']==[0,5000,20000] and p['diagnostics']['planned_job_observations']==24
        and p['diagnostics']['selection']=='final20000_only','Observation/selection changed')
    require(p['scope']['training_seeds']==[10] and p['scope']['pure_margin_scalar_causal_claim'] is False
        and p['scope']['generalization_claim'] is False,'Research scope changed')
    require(e['reference_arrays_exactly_equal_RadialSupport_mesh'] is True
        and e['original_raw_asset_GT_reference_claim'] is False
        and e['evaluation_samples_independent_of_optimizer_internal_samples'] is True,
        'Frozen full-field reference semantics changed')


def check_common_models(states):
    require(len(states)==2,'Two original initial states required')
    hashes=[model_sha256(s['model']) for s in states]
    require(hashes[0]==hashes[1]==MODEL_SHA,'The two single source initial model parameters differ')
    require(all(s['model_sha256']==hashes[i] for i,s in enumerate(states)),'Source model digest invalid')
    require(sum(v.numel() for v in states[0]['model'].values())==29907,'Original architecture differs')
    return hashes[0]


def verify(run):
    run=Path(run).resolve();p=read(run/'RUN_PROTOCOL.json');assert_contract(p)
    require(file_hash(run/'DATA_MANIFEST.json')==p['hashes']['data_manifest_sha256'],'Data manifest changed')
    for rel,digest in read(run/'DATA_MANIFEST.json')['files'].items():
        target=(run/rel).resolve();require(target.is_relative_to(run),'Input path escapes run')
        require(file_hash(target)==digest,'Prepared input changed: '+rel)
    root=Path(__file__).resolve().parents[1]
    for rel,digest in p['source_sha256'].items():require(file_hash(root/rel)==digest,'Frozen C0 dependency changed')
    for path,digest in p['bound_source_sha256'].items():require(file_hash(path)==digest,'Frozen parent evidence changed: '+path)
    return p


def prepare(joint,fitdiag,out):
    require(not torch.cuda.is_initialized(),'CPU-only preparation; never restore source CUDA RNG')
    joint=Path(joint).resolve();fitdiag=Path(fitdiag).resolve();out=Path(out).resolve()
    require(not out.exists(),'Do not overwrite a prepared study')
    seal_path=joint/'SEALED.json';seal=read(seal_path);archive=Path(seal['archive'])
    require(seal['status']=='SEALED' and seal['sha256']==ARCHIVE_SHA and file_hash(archive)==ARCHIVE_SHA,
        'Wrong or changed joint target archive')
    require(seal['CRC_and_all_member_SHA256_pass'] and seal['manifest_itself_payload_zip_fresh_equal']
        and seal['fresh_CPU_readback']['status']=='PASS','Joint target archive lacks successful readback')
    payload=joint/'payload';manifest_path=payload/'BUNDLE_MANIFEST.json';manifest=read(manifest_path)['files']
    for rel,entry in manifest.items():
        path=(payload/rel).resolve();require(path.is_relative_to(payload),'Bundle path escapes')
        require(path.stat().st_size==entry['bytes'] and file_hash(path)==entry['sha256'],'Joint sealed payload changed')
    strict=payload/'runs/strict';parent=read(strict/'RUN_PROTOCOL.json');require(parent['uids']==list(UIDS),'Fixed parent list differs')
    fit_manifest_path=fitdiag/'MANIFEST_SHA256.json';fit_manifest=read(fit_manifest_path)['files']
    fit_protocol_path=fitdiag/'RUN_PROTOCOL.json';fp=read(fit_protocol_path)
    fit_archive=fitdiag.with_name(fitdiag.name+'_results.zip');fit_receipt_path=fit_archive.with_suffix('.zip.json');fit_receipt=read(fit_receipt_path)
    require(fit_receipt['fresh_original_cpu_replay']=='PASS' and file_hash(fit_archive)==fit_receipt['sha256'],'Fitdiag source seal changed')
    initials=[fitdiag/'training'/job/'initial.pt' for job in ('single_energy_10','single_prior_10')]
    for path in initials+[fit_protocol_path]:require(file_hash(path)==fit_manifest[path.relative_to(fitdiag).as_posix()],'V1 source initial not in sealed manifest')
    states=[torch.load(path,map_location='cpu',weights_only=False) for path in initials]
    common_model_sha=check_common_models(states)
    require(all(state['update']==3000 for state in states),'Expected old v1 metadata update3000; not a freshCPU step0')
    copied={'source_evidence/JOINT_SEALED.json':seal_path,'source_evidence/JOINT_BUNDLE_MANIFEST.json':manifest_path,
        'source_evidence/strict_RUN_PROTOCOL.json':strict/'RUN_PROTOCOL.json','source_evidence/strict_REVIEW.json':strict/'REVIEW.json',
        'source_evidence/fitdiag_RUN_PROTOCOL.json':fit_protocol_path,'source_evidence/fitdiag_MANIFEST_SHA256.json':fit_manifest_path,
        'source_evidence/fitdiag_ARCHIVE_REPORT.json':fit_receipt_path,
        'source_evidence/single_energy_10_initial.pt':initials[0],'source_evidence/single_prior_10_initial.pt':initials[1]}
    lists={key:[] for key in ('directions0','directions1_old','directions1_margin','proxy_points','point_features')}
    energy={};targets={};support=RadialSupport();faces=None
    for uid in UIDS:
        data_path=strict/'data'/f'{uid}.npz'
        with np.load(data_path,allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        reference_v,reference_f=support.mesh(data['logr'])
        require(np.array_equal(reference_v,data['ref_vertices']) and np.array_equal(reference_f,data['ref_faces']),
            'Reference is not the same full frozen radial-field surface')
        require(data['base_u'].shape==(258,3) and data['faces'].shape==(512,3) and data['faces'].dtype==np.int32,'Unexpected frozen layout')
        require(data['proxy_points'].shape==(256,3) and data['point_features'].shape==(256,4)
            and np.array_equal(point_features(data['proxy_points']),data['point_features']),'Original proxy features differ')
        row_path=strict/'endpoints'/uid/'opt_teacher/RESULT.json';row=read(row_path);mesh_path=strict/row['mesh_file']
        require(row['status']=='success' and row['qualified_strict']['qualified'] and row['audit']['valid']
            and file_hash(mesh_path)==row['mesh_npz_sha256'],'Predeclared strict teacher endpoint not verified')
        with np.load(mesh_path,allow_pickle=False) as z:
            margin_u=np.array(z['u']);r=support.query(data['logr'],margin_u);v=margin_u*r[:,None]
            require(np.array_equal(z['r'],r) and np.array_equal(z['vertices'],v) and np.array_equal(z['faces'],data['faces']),
                'New target stale radius or altered faces')
        fq=named_quality(data['base_u']*support.query(data['logr'],data['base_u'])[:,None],data['faces'])
        q=named_quality(v,data['faces']);strict_floor={k:fq[k]+.1 for k in ('min_triangle_angle_deg','triangle_min_angle_p05_deg')}
        require(qualification(q,row['audit'],strict_floor)['qualified'],'New target does not meet declared floor')
        old_path=strict/'endpoints'/uid/'teacher/mesh.npz'
        with np.load(old_path,allow_pickle=False) as z:require(np.array_equal(z['u'],data['teacher_u']),'Old teacher changed')
        require(faces is None or np.array_equal(faces,data['faces']),'Mixed template connectivity');faces=data['faces'].copy()
        for key,value in [('directions0',data['base_u']),('directions1_old',data['teacher_u']),('directions1_margin',margin_u),
                          ('proxy_points',data['proxy_points']),('point_features',data['point_features'])]:lists[key].append(value)
        energy[uid]={'old':float(np.square(data['teacher_u']-data['base_u']).mean()),'margin':float(np.square(margin_u-data['base_u']).mean())}
        targets[uid]={'old_u_sha256':array_hash(data['teacher_u']),'margin_u_sha256':array_hash(margin_u),
            'mainfield_sha256':array_hash(data['logr']),'margin_source_mesh_sha256':file_hash(mesh_path),
            'margin_source_row_sha256':file_hash(row_path),'fixed_quality':fq,'margin_quality':q,
            'new_target_chosen_by':'all four predeclared strict-floor teacher-start endpoints; no per-UID GT selection'}
        copied[f'data/{uid}.npz']=data_path
        copied[f'target_references/{uid}/margin_mesh.npz']=mesh_path;copied[f'target_references/{uid}/margin_RESULT.json']=row_path
        copied[f'target_references/{uid}/old_teacher_mesh.npz']=old_path
    bound={str(path):file_hash(path) for path in copied.values()}
    out=new_dir(out);files={}
    for rel,path in copied.items():
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        files[rel]=file_hash(dest);require(files[rel]==bound[str(path)],'Source copy changed')
    label_path=out/'train_labels.npz'
    save_npz(label_path,train_uids=np.asarray(UIDS),faces=faces,**{key:np.stack(value) for key,value in lists.items()})
    files['train_labels.npz']=file_hash(label_path)
    initial=out/'initial/common_v1_model.pt';initial.parent.mkdir()
    model={k:v.detach().cpu().clone() for k,v in states[0]['model'].items()}
    with initial.open('xb') as stream:
        torch.save({'model':model,'model_sha256':common_model_sha,'source_model_only':True,
            'source_checkpoint_sha256':[file_hash(path) for path in initials],
            'source_stored_update':3000,'new_training_initial_update':0,
            'optimizer_rng_scheduler_sampler_from_source_omitted':True},stream)
    files['initial/common_v1_model.pt']=file_hash(initial)
    write_json(out/'DATA_MANIFEST.json',{'files':files})
    training={'batch_size':4,'total_updates':20000,'learning_rate':1e-4,'weight_decay':1e-4,
        'checkpoint_updates':[0,5000,20000],'device':'cpu','torch_threads':1,'interop_threads':1,
        'optimizer':'AdamW','scheduler':'constant','CUDA_VISIBLE_DEVICES':'','cpu_rng_only':True,'fresh_optimizer':True}
    scale=float(fp['endpoint_scale_squared']);loss={'name':'c0_unscaled_discrete_bridge_velocity_MSE','bridge_steps':32,
        'endpoint_scale_squared':scale,'endpoint_scale_unused_by_c0':True,'no_quality_loss':True,'no_new_network_inputs':True}
    jobs=job_table(file_hash(initial),common_model_sha);root=Path(__file__).resolve().parents[1]
    protocol={'schema':'A-MARGIN-SINGLE-v1','train_uids':list(UIDS),'uids':list(UIDS),'all16_sorted_uids':parent['all16_sorted_uids'],
        'training':training,'loss':loss,'jobs':jobs,'endpoint_scale_squared':scale,
        'paths':{'training_labels':'train_labels.npz','data_manifest':'DATA_MANIFEST.json','initial_checkpoint':'initial/common_v1_model.pt'},
        'hashes':{'training_labels_sha256':file_hash(label_path),'data_manifest_sha256':file_hash(out/'DATA_MANIFEST.json'),
            'initial_checkpoint_sha256':file_hash(initial),'initial_model_sha256':common_model_sha,
            'config_sha256':config_hash(training,loss,jobs)},
        'diagnostics':{'updates':[0,5000,20000],'planned_job_observations':24,'selection':'final20000_only',
            'teacher_forced_all32_time_grid':True,'direction_endpoint_error':True,'zero_target_denominator':'null; absolute error retained'},
        'evaluation':{'planned_endpoints':20,'endpoint_names':['fixed','old_teacher','margin_teacher','old_final','margin_final'],
            'uids':list(UIDS),'seed_bases':[1593501,1693501],'points_each_direction':8192,'deployment_threads':4,
            'deployment_steps':32,'network_device':'cpu','fallback':False,'final_radius_requery':True,
            'exact_geometry_and_F512_audit':True,'quality_gates':['original_fixed','fixed_plus_0.1_degree'],
            'same_mainfield_reference_MC_as_parent_teacher_study':True,
            'reference_surface':'complete2050node4096face triangulated surface of the same frozen radial mainfield',
            'reference_arrays_exactly_equal_RadialSupport_mesh':True,
            'evaluation_samples_independent_of_optimizer_internal_samples':True,
            'original_raw_asset_GT_reference_claim':False},
        'scope':{'training_seeds':[10],'single_parent_jobs':8,'previously_seen_development_parents':4,
            'generalization_claim':False,'pure_margin_scalar_causal_claim':False,
            'new_target_contains_optimizer_trajectory_and_quality_margin_changes':True,
            'historical_parent_reference_term_correction':'saved ref arrays are the same full radial-field surface, not the original raw asset mesh; sealed parent text unchanged',
            'network_conditioning':'original proxy256 only; bridge positions and velocity supervision derive from target directions; no extra full-mesh or teacher condition',
            'no_old_GPU_model_as_paired_CPU_target_causal_control':True,'no_evaluation_error_target_or_checkpoint_selection':True,
            'no_old_label_updates':True,'per_job_parent_exposure':80000},
        'time_draw_pairing':'same UID old/margin jobs share seed10 and CPU random draw sequence; full k-sequence hash required',
        'targets':targets,'target_displacement_energy':energy,'parent_joint_archive_sha256':ARCHIVE_SHA,
        'parent_joint_SEALED_sha256':file_hash(seal_path),'parent_joint_bundle_manifest_sha256':file_hash(manifest_path),
        'common_initial_source_sha256':[file_hash(path) for path in initials],
        'source_sha256':{rel:file_hash(root/rel) for rel in FROZEN_DEPENDENCIES},'bound_source_sha256':bound,
        'prepare_source_sha256':file_hash(__file__),
        'required_CPU_gate':'all8jobs continuous8 vs resume3+5 semantic state bitwise; independent original C0 next-step oracle; CUDA never initialized',
        'new_trainer_source_binding':'CPU gate records final train/smoke source hashes before formal training',
        'projected_8x20000_core_minutes_from_50update_CPU_smoke':37.307267614717905,
        'cost_scope':'core projection only; excludes checkpoint IO/diagnostics and is not an isolated benchmark',
        'formal_training_authorized_by_prepare':False}
    assert_contract(protocol);write_json(out/'RUN_PROTOCOL.json',protocol);verify(out)
    require(not torch.cuda.is_initialized(),'Preparation initialized CUDA')
    return protocol


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--joint-package',required=True);a.add_argument('--fitdiag-run',required=True);a.add_argument('--out',required=True);args=a.parse_args()
    p=prepare(args.joint_package,args.fitdiag_run,args.out)
    print(json.dumps({'run':str(Path(args.out).resolve()),'protocol_sha256':file_hash(Path(args.out)/'RUN_PROTOCOL.json'),
        'jobs':list(p['jobs']),'data_files':len(read(Path(args.out)/'DATA_MANIFEST.json')['files']),
        'initial_model_sha256':p['hashes']['initial_model_sha256'],'cuda_initialized':torch.cuda.is_initialized()},indent=2),flush=True)


if __name__=='__main__':main()
