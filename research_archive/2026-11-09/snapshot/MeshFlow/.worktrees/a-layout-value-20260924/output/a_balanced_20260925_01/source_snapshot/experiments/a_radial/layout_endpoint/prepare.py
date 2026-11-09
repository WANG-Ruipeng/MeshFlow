"""Freeze the endpoint experiment from the sealed A-TASKALIGNED-v1 run."""
from __future__ import annotations
import argparse,datetime,json,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from ..exporter import ASSETS,RadialSupport
from ..io_utils import array_hash,file_hash,save_npz,write_json
from ..verify_repo import verify
from ..layout_student.deployment import named_quality
from ..layout_student.train import model_sha256
from ..layout_student.prepare import split_uids,config_hash


def require(ok,message):
    if not ok:raise ValueError(message)


def prepare(source_run,run):
    source=Path(source_run).resolve();run=Path(run).resolve()
    require(not run.exists(),'New run directory must not exist')
    manifest_path=source/'MANIFEST_SHA256.json'
    manifest=json.loads(manifest_path.read_text())['files']
    for rel,expected in manifest.items():
        p=(source/rel).resolve()
        require(p.is_relative_to(source) and p.is_file() and file_hash(p)==expected,'Source manifest mismatch: '+rel)
    old=json.loads((source/'RUN_PROTOCOL.json').read_text())
    require(old['experiment']=='A-TASKALIGNED-v1','Wrong initialization experiment')
    uids=sorted(old['uids']);train,check=split_uids(uids)
    require(train==old['train_uids'] and check==old['internal_check_uids'],'Parent split changed')
    repo=Path(__file__).resolve().parents[3]
    identity=verify(repo)
    require(identity['pass'],'Protected official code identity mismatch')
    for rel,h in old['fixed_source_hashes'].items():
        require(file_hash(ASSETS.parent/rel)==h,'Previously frozen source changed: '+rel)
    require(torch.cuda.is_available(),'Actual CUDA required')
    support=RadialSupport();dataset={};floors=[]
    for uid in uids:
        p=source/'data'/f'{uid}.npz'
        with np.load(p,allow_pickle=False) as z:dataset[uid]={k:np.array(z[k]) for k in z.files}
        d=dataset[uid];f=d['faces'];base=d['base_u']
        require(f.dtype==np.int32 and f.shape==(512,3),'Original connectivity contract changed')
        require(d['teacher_u'].shape==base.shape==(258,3),'Direction shape mismatch')
        require(np.array_equal(f,dataset[uids[0]]['faces']),'Different connectivity')
    with np.load(source/'train_labels.npz',allow_pickle=False) as z:
        original_labels={k:np.array(z[k]) for k in z.files}
    require(original_labels['train_uids'].tolist()==train,'Training-only archive UID mismatch')
    for name,key in [('directions0','base_u'),('directions1','teacher_u'),('proxy_points','proxy_points'),('point_features','point_features')]:
        require(np.array_equal(original_labels[name],np.stack([dataset[uid][key] for uid in train])),'Training archive changed')
    for uid in train:
        d=dataset[uid];v=d['base_u']*support.query(d['logr'],d['base_u'])[:,None]
        q=named_quality(v,d['faces'])
        floors.append([q['min_triangle_angle_deg'],q['triangle_min_angle_p05_deg']])
    displacement_squared=float(np.mean((original_labels['directions1']-original_labels['directions0'])**2))
    require(np.isfinite(displacement_squared) and displacement_squared>0,'No moving training teachers')
    moving=[u for u in train if not np.array_equal(dataset[u]['base_u'],dataset[u]['teacher_u'])]
    diagnostic_uid=min(moving)
    checkpoint_hashes={};model_hashes={}
    for seed in [10,11]:
        p=source/'training'/f'seed_{seed}'/'final.pt'
        state=torch.load(p,map_location='cpu',weights_only=False)
        require(state['update']==3000 and state['is_predeclared_final'],'Initialization is not predeclared v1 final')
        require(state['model_sha256']==model_sha256(state['model']),'Initialization model hash mismatch')
        checkpoint_hashes[str(seed)]=file_hash(p);model_hashes[str(seed)]=state['model_sha256']
    run.mkdir(parents=True,exist_ok=False);(run/'data').mkdir();(run/'initial').mkdir()
    files={}
    for uid in uids:
        rel=f'data/{uid}.npz';shutil.copy2(source/rel,run/rel);files[rel]=file_hash(run/rel)
        require(files[rel]==manifest[rel],'Copied data changed')
    labels={**original_labels,'logr':np.stack([dataset[u]['logr'] for u in train]),
            'faces':dataset[uids[0]]['faces'],'quality_floors_deg':np.asarray(floors,dtype=np.float64)}
    save_npz(run/'train_labels.npz',**labels);files['train_labels.npz']=file_hash(run/'train_labels.npz')
    for seed in [10,11]:
        rel=f'initial/seed_{seed}.pt';shutil.copy2(source/'training'/f'seed_{seed}'/'final.pt',run/rel)
        files[rel]=file_hash(run/rel);require(files[rel]==checkpoint_hashes[str(seed)],'Copied checkpoint changed')
    shutil.copy2(source/'TEACHERS.json',run/'TEACHERS.json');files['TEACHERS.json']=file_hash(run/'TEACHERS.json')
    write_json(run/'DATA_MANIFEST.json',{'files':files})
    training={'arms':['c0','c1','c2'],'seeds':[10,11],'updates':1000,'batch_size':4,
        'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':1e-4,'schedule':'constant_learning_rate',
        'checkpoint_every':250,'checkpoint_selection':'final_update_1000_only_no_metric_selection',
        'optimizer_reset_for_every_arm':True,'same_balanced_parent_sequence_across_arms':True,
        'precision':'network_float32_geometry_float64','tf32':False,
        'trainable':'unchanged_29907_parameter_DomainVelocity_only','teacher_forcing_in_rollout':False,
        'rollout_steps':32,'detached_discrete_choices':['guard_alpha','radial_chart_owner'],
        'gradient_through_all_continuous_32_steps':True,'equal_updates_not_equal_compute':True}
    losses={'endpoint_scale_squared':displacement_squared,
        'endpoint_scale_definition':'mean squared coordinate displacement Fixed-to-teacher over all 12 training parents, vertices and xyz',
        'quality_scale_deg':0.1,'weights':{'endpoint':1.,'min_angle':1.,'p05_angle':1.},
        'c0':'original discrete_bridge tangent velocity mean square error',
        'c1':'mean((free_rollout_u32-teacher_u)^2)/endpoint_scale_squared',
        'c2':'c1 + mean_over_parents(relu((floor_min-actual_min)/0.1)^2 + relu((floor_p05-actual_p05)/0.1)^2)',
        'quality_floor_tolerance_in_loss_deg':0.0,'quality_evaluation_tolerance_deg':1e-6,
        'scale_does_not_relax_qualification':True,'scale_uses_only_training_labels':True,
        'field_and_teacher_access':'Full field and endpoint only for loss supervision; network input remains proxy256.'}
    protocol={'experiment':'A-ENDPOINT-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization':'User approved executing fixed C0/C1/C2 endpoint experiment on existing data.',
        'source_run':str(source),'source_run_manifest_sha256':file_hash(manifest_path),
        'uids':uids,'train_uids':train,'internal_check_uids':check,'split':old['split'],
        'paths':{'training_labels':'train_labels.npz','data_dir':'data','teacher_manifest':'TEACHERS.json',
            'initial_checkpoints':{str(seed):f'initial/seed_{seed}.pt' for seed in [10,11]}},
        'training':training,'loss':losses,'endpoint_scale_squared':displacement_squared,
        'diagnostic':{'uid':diagnostic_uid,'arm':'c1','seed':10,'updates':200,
            'selection_rule':'lexicographically smallest moving training UID, before any new results',
            'weight_used_for_formal_training':False,'endpoint_scale_from_all_12_training_parents':True},
        'evaluation':{'models':['fixed','teacher_reference','v1_10','v1_11','c0_10','c0_11','c1_10','c1_11','c2_10','c2_11'],
            'attempt_count':160,'expected_rows':288,'seed_base':392401,'second_seed_base':492401,
            'seed_rule':'base + index in sorted UID list','points_each_direction':8192,'hot_repeats':3,
            'raw_and_quality_fallback_separate':True,'all_attempts_retained':True,
            'final_cpu_original_deployment_only':True,'paired_improvement_not_ratio_of_medians':True},
        'continue_gate':{'exploratory_not_significance':True,'arm':'c2','both_training_seeds_required':True,
            'all16_qualified_and_ge2pct_both_min':8,'internal4_qualified_and_ge2pct_both_min':2},
        'quality':old['quality'],'deployment':old['deployment'],'input':old['input'],
        'frozen':old['frozen']+['v1_final_checkpoints','v1_teachers','parent_split'],
        'heldout_generalization':False,'full_MeshFlow_comparison':'independent_pending_track_not_run_by_this_experiment',
        'fixed_source_hashes':{rel:file_hash(ASSETS.parent/rel) for rel in [
            'exporter.py','legacy/networks.py','legacy/deform.py','legacy/chart.py','legacy/common.py',
            'layout_value/optimizer.py','layout_value/geometry.py','layout_student/deployment.py','assets/support.npz']},
        'hashes':{'data_manifest_sha256':file_hash(run/'DATA_MANIFEST.json'),
            'training_labels_sha256':files['train_labels.npz'],'initial_checkpoint_sha256':checkpoint_hashes,
            'initial_model_sha256':model_hashes,'config_sha256':config_hash({'training':training,'loss':losses}),
            'source_run_manifest_sha256':file_hash(manifest_path)}}
    write_json(run/'RUN_PROTOCOL.json',protocol)
    write_json(run/'SOURCE_PROVENANCE.json',{'source_run':str(source),'source_manifest_sha256':file_hash(manifest_path),
        'all_source_manifest_files_verified':len(manifest),'verified_manifest_files':manifest,
        'copied_data_arrays_and_initial_checkpoints_byte_identical':True,'old_outputs_written':False,
        'train_only_geometry_archive':True,'train_count':12,'internal_check_count':4,'prepare_source_sha256':file_hash(__file__)})
    write_json(run/'REPO_IDENTITY.json',identity)
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'compute_capability':list(torch.cuda.get_device_capability()),
        'device_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'repo_status':subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        'free_output_disk_bytes':shutil.disk_usage(run).free,'training_device':'cuda','deployment_device':'cpu'}
    write_json(run/'ENVIRONMENT.json',env)
    print(json.dumps({'run':str(run),'protocol_sha256':file_hash(run/'RUN_PROTOCOL.json'),
        'endpoint_scale_squared':displacement_squared,'diagnostic_uid':diagnostic_uid,
        'source_files_verified':len(manifest),'data_files':len(files),'environment':env}),flush=True)
    return protocol


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-run',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();prepare(a.source_run,a.run)