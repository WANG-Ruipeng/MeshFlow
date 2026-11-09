"""Fixed64k FM final artifacts; E32 primary and E128 diagnostics stay separate."""
from __future__ import annotations
import argparse
from argparse import Namespace
import json
import sys
from importlib.metadata import version
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from .. import sample_ours,export_ours
from ..exporter import ASSETS
from ..io_utils import new_dir,write_json,file_hash,array_hash
from ..system_probe.official import configure,require,exact_same
from ..system_probe import ours_metrics as common
from .train import contract,FINAL,START,SCOPE,LR

ARTIFACT_SCOPE='FM_TAIL_FINAL64000_PRIMARY32_SEPARATE_AUX128'
SCHEMA=1


def runtime_identity():
    return {'python':sys.version,'torch':str(torch.__version__),'torch_cuda_build':torch.version.cuda,
        'numpy':np.__version__,'scipy':version('scipy'),'vtk':version('vtk')}


def dependencies(repo):
    names=['experiments/a_radial/fm_tail/'+n for n in ['artifacts.py','metrics.py','test_artifacts.py']]
    names += ['experiments/a_radial/'+n for n in ['sample_ours.py','export_ours.py','evaluate.py','registry.py',
        'mesh_ops.py','exporter.py','models.py','io_utils.py','layout_value/geometry.py','system_probe/ours_metrics.py']]
    names += [str(x.relative_to(repo)) for x in sorted((repo/'experiments/a_radial/legacy').glob('*.py'))]
    return {name:file_hash(repo/name) for name in names}


def prepare(run):
    run=Path(run).resolve();p,sha=contract(run);repo=Path(p['repository'])
    old=common.read_json(run/'initial/source_RUN_PROTOCOL.json');data=Path(old['data'])
    require(file_hash(data/'DATASET.json')==p['source_data_hashes']['DATASET.json'],'Original evaluation data changed')
    raw={uid:file_hash(data/'raw'/f'{uid}.npz') for uid in p['parent_uids']}
    protocol={'schema':SCHEMA,'scope':ARTIFACT_SCOPE,'training_protocol_sha256':sha,'seeds':[10,11],
        'AE_step':3200,'FM_final_step':FINAL,'checkpoint_selection':'registered final64000 only',
        'expected_AE_sha256':{seed:job['ae_sha256'] for seed,job in p['jobs'].items()},
        'source_parent_FM8000_sha256':{seed:job['parent_sha256'] for seed,job in p['jobs'].items()},
        'primary':{'samples_per_seed':64,'noise_seeds':{'10':73510,'11':73511},'euler_steps':32,'sampling_device':'cuda',
            'directory':'artifacts/seed{seed}/samples','export_directory':'artifacts/seed{seed}/exports',
            'arms':['fixed','a','qem'],'target_faces':512,'one_saved_mainfield_for_three_outlets':True},
        'auxiliary':{'samples_per_seed':64,'same_noise_as_primary':True,'euler_steps':128,'sampling_device':'cuda',
            'directory':'artifacts/seed{seed}/auxiliary_e128/samples','export_meshes':False,
            'allowed_metrics':['paired_logradius_node_RMS','nearest_training_logradius_RMS','nearest_parent_counts'],
            'excluded_from_primary_table':True,'never_selects_or_replaces_primary':True},
        'surface_metric':{'points_each_direction':8192,'MC_seed_bases':[993501,1093501],
            'global_sample_index':'seed_index*64+int(sample_id)','quality_tolerance_deg':1e-6},
        'cross_run_pairing':'original8k versus final64k by seed/sample_id/exact epsilon; mainfields differ, so cross-run is not same-field outlet improvement',
        'source_run':p['source_run'],'source_training_protocol_sha256':p['source_protocol_sha256'],
        'evaluation_data':str(data),'source_data_hashes':p['source_data_hashes'],'raw_mesh_sha256':raw,
        'source_sha256':dependencies(repo),'support_sha256':file_hash(ASSETS/'support.npz'),
        'original_A_sha256':{str(seed):file_hash(ASSETS/f'original_A_{seed}.pt') for seed in (10,11)},
        'runtime':runtime_identity(),'TF32':False,'deterministic_algorithms':True,'source_artifacts_immutable':True,
        'distribution_groups':'four fixed16-generated blocks per seed; original evaluate.py unchanged',
        'no_external_ranking_or_generalization_qualification':True,'failed_slots_retained':True}
    write_json(run/'ARTIFACT_PROTOCOL.json',protocol)
    snapshot=run/'artifact_source_snapshot'
    for name in protocol['source_sha256']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    return protocol


def artifact_contract(run):
    run=Path(run).resolve();p,sha=contract(run);ap=common.read_json(run/'ARTIFACT_PROTOCOL.json')
    exact_same(ap['runtime'],runtime_identity(),'artifact_runtime')
    require(ap['schema']==SCHEMA and ap['scope']==ARTIFACT_SCOPE and ap['training_protocol_sha256']==sha,'Artifact scope/protocol changed')
    require(ap['AE_step']==3200 and ap['FM_final_step']==64000 and ap['seeds']==[10,11],'Fixed final identity changed')
    require(ap['primary']['samples_per_seed']==64 and ap['primary']['euler_steps']==32 and ap['primary']['arms']==list(common.ARMS),'Primary sampling contract changed')
    require(ap['primary']['noise_seeds']=={'10':73510,'11':73511} and ap['primary']['target_faces']==512,'Primary noise/budget changed')
    require(ap['auxiliary']['euler_steps']==128 and ap['auxiliary']['export_meshes'] is False and ap['auxiliary']['excluded_from_primary_table'] is True,'Auxiliary mixed with primary')
    require(ap['source_run']==p['source_run'] and ap['source_training_protocol_sha256']==p['source_protocol_sha256'],'Source8k run binding changed')
    require(ap['surface_metric']=={'points_each_direction':8192,'MC_seed_bases':[993501,1093501],
        'global_sample_index':'seed_index*64+int(sample_id)','quality_tolerance_deg':1e-6},'Metric registration changed')
    require(ap['source_data_hashes']==p['source_data_hashes'] and set(ap['raw_mesh_sha256'])==set(p['parent_uids']),'Reference data binding changed')
    require(ap['expected_AE_sha256']=={seed:job['ae_sha256'] for seed,job in p['jobs'].items()},'AE registration changed')
    require(ap['source_parent_FM8000_sha256']=={seed:job['parent_sha256'] for seed,job in p['jobs'].items()},'ParentFM registration changed')
    require(ap['auxiliary']['samples_per_seed']==64 and ap['auxiliary']['same_noise_as_primary'] is True
        and ap['auxiliary']['never_selects_or_replaces_primary'] is True,'Auxiliary registration changed')
    exact_same(ap['source_sha256'],dependencies(Path(p['repository'])),'artifact_source_hashes')
    for name,digest in ap['source_sha256'].items():require(file_hash(run/'artifact_source_snapshot'/name)==digest,'Artifact source snapshot changed')
    require(file_hash(ASSETS/'support.npz')==ap['support_sha256'],'Support asset changed')
    for seed,digest in ap['original_A_sha256'].items():require(file_hash(ASSETS/f'original_A_{seed}.pt')==digest,'Original A weights changed')
    data=Path(ap['evaluation_data'])
    for name,digest in ap['source_data_hashes'].items():require(file_hash(data/name)==digest,'Evaluation data changed: '+name)
    for uid,digest in ap['raw_mesh_sha256'].items():require(file_hash(data/'raw'/f'{uid}.npz')==digest,'Raw reference changed: '+uid)
    return p,ap,file_hash(run/'ARTIFACT_PROTOCOL.json')


def validate_final_state(state,p,seed,training_sha):
    job=p['jobs'][str(seed)];cfg=state['config']
    require(state['scope']==SCOPE and state['step']==FINAL and state['is_predeclared_final'] is True,'Only final64000 continuation weights allowed')
    for key,value in {'seed':seed,'protocol_sha256':training_sha,'ae_sha256':job['ae_sha256'],
        'source_fm_sha256':job['parent_sha256'],'stage':'fm','steps':FINAL,'initial_step':START,'final_step':FINAL,
        'dataset_sha256':p['source_data_hashes']['radial_train.npz'],'dataset_manifest_sha256':p['source_data_hashes']['DATASET.json'],
        'batch':128,'lr':LR,'scope':SCOPE,'device':'cuda',
        'data_manifest_sha256':p['data_manifest_sha256'],'source_manifest_sha256':p['source_manifest_sha256']}.items():require(cfg.get(key)==value,'Final checkpoint binding differs: '+key)
    require(state['scheduler']['last_epoch']==FINAL and state['scheduler']['base_lrs']==[LR],'Final scheduler changed')
    require(state['optimizer']['state'] and all(int(v['step'])==FINAL for v in state['optimizer']['state'].values()),'Final Adam counts differ')
    require(all(g['lr']==g['initial_lr']==LR for g in state['optimizer']['param_groups']),'Final optimizer LR differs')
    require(state['parent_exposures_each']==512000 and not state['draw_queue'],'Final exposure/sampler changed')
    require(len(state['trace'])==FINAL-START and all(r['step']==START+i+1 for i,r in enumerate(state['trace'])),'Missing or reordered continuation trace')
    require(state['rng']['torch_cuda'] and state['batch_rng'].dtype==torch.uint8,'Incomplete CUDA/RNG continuation state')
    require(state['boundary']=={'complete_update':True,'pending_data':False},'Partial update cannot be sampled')


def final_inputs(run,seed):
    run=Path(run).resolve();p,ap,artifact_sha=artifact_contract(run);training_sha=file_hash(run/'RUN_PROTOCOL.json')
    gate=common.read_json(run/'GPU_GATE.json')
    require(gate['status']=='PASS' and gate['protocol_sha256']==training_sha and gate['actual_new_optimizer_updates']==36
        and gate['source_sha256']==file_hash(Path(__file__).with_name('smoke.py')),'Tail CUDA gate missing or stale')
    require(seed in ap['seeds'],'Unregistered seed');job=p['jobs'][str(seed)];folder=run/'training'/f'seed_{seed}'
    done=common.read_json(folder/'TRAINING_COMPLETE.json')
    require(done['status']=='FIXED_BUDGET_COMPLETE_UNQUALIFIED' and done['seed']==seed and done['final_step']==FINAL,'Fixed final training incomplete')
    require(done['protocol_sha256']==training_sha and done['selection']=='fixed_final64000_only','Completion binding changed')
    fp=Path(done['checkpoint']).resolve();require(fp.is_relative_to(folder.resolve()),'Final checkpoint escapes this seed')
    require(file_hash(fp)==done['checkpoint_sha256'],'Final checkpoint file changed')
    record=common.read_json(fp.with_suffix('.json'));require(record['sha256']==file_hash(fp) and record['step']==FINAL,'Final completion record differs')
    state=torch.load(fp,map_location='cpu',weights_only=True);validate_final_state(state,p,seed,training_sha)
    ae=run/job['ae_checkpoint'];require(file_hash(ae)==job['ae_sha256']==ap['expected_AE_sha256'][str(seed)],'Frozen AE changed')
    parent=torch.load(run/job['parent_checkpoint'],map_location='cpu',weights_only=True)
    require(parent['model'].keys()==state['model'].keys(),'FM architecture keys changed')
    for key,value in state['model'].items():
        require(value.shape==parent['model'][key].shape and value.dtype==parent['model'][key].dtype
                and bool(torch.isfinite(value).all()),'FM architecture/dtype/nonfinite tensor changed: '+key)
    exact_same(state['source_scheduler'],parent['scheduler'],'archived_parent_scheduler')
    exact_same(state['source_config'],parent['config'],'archived_parent_config')
    for key in ('mu','sd'):require(torch.equal(parent['model'][key],state['model'][key]),'FM normalization drift')
    require(file_hash(fp)==done['checkpoint_sha256'],'Final checkpoint changed during read')
    return {'schema':SCHEMA,'scope':ARTIFACT_SCOPE,'seed':seed,'training_protocol_sha256':training_sha,
        'artifact_protocol_sha256':artifact_sha,'AE_step':3200,'FM_step':FINAL,
        'ae_path':str(ae.resolve()),'fm_path':str(fp),'ae_sha256':file_hash(ae),'fm_sha256':file_hash(fp),
        'training_complete_sha256':file_hash(folder/'TRAINING_COMPLETE.json'),'gpu_gate_sha256':file_hash(run/'GPU_GATE.json'),
        'dataset_sha256':p['source_data_hashes']['radial_train.npz'],'dataset_manifest_sha256':p['source_data_hashes']['DATASET.json'],
        'original_A_sha256':ap['original_A_sha256'][str(seed)],'support_sha256':ap['support_sha256']}


def seed_folder(run,seed):return Path(run)/'artifacts'/f'seed{seed}'


def view_samples(run,seed,auxiliary=False):
    folder=seed_folder(run,seed);identity=common.read_json(folder/'IDENTITY.json');actual=final_inputs(run,seed)
    exact_same(identity,actual,'fixed_final_artifact_identity')
    path=folder/'auxiliary_e128'/'samples' if auxiliary else folder/'samples'
    manifest=common.read_json(path/'SAMPLES.json');steps=128 if auxiliary else 32
    for name in ('ae_sha256','fm_sha256','dataset_sha256','dataset_manifest_sha256'):
        require(manifest[name]==identity[name],'Sample identity differs: '+name)
    require(manifest['schema']==1 and manifest['sampling_device']=='cuda','Unexpected sample schema/device')
    require(manifest['noise_seed']==73500+seed and manifest['fm_steps']==manifest['fm_nfe']==steps,'Wrong registered noise/Euler steps')
    maps=common.registered_records(manifest,'requested');gen=torch.Generator(device='cpu').manual_seed(73500+seed)
    noise=torch.randn((64,64),generator=gen).numpy()
    return {'seed':seed,'sample_dir':path,'sample_manifest':manifest,'sample_records':maps,
        'sample_manifest_sha256':file_hash(path/'SAMPLES.json'),'expected_noise':noise,'identity':identity,
        'primary':not auxiliary,'checkpoint_sha256':{'ae':identity['ae_sha256'],'fm':identity['fm_sha256']}}


def sample_final(run,seed):
    run=Path(run).resolve();identity=final_inputs(run,seed);folder=new_dir(seed_folder(run,seed))
    write_json(folder/'IDENTITY.json',identity);configure(seed)
    require(torch.cuda.is_available(),'Production sampling is the predeclared CUDA batch1 path')
    try:
        for auxiliary,steps in [(False,32),(True,128)]:
            out=folder/'auxiliary_e128'/'samples' if auxiliary else folder/'samples'
            sample_ours.run(Namespace(device='cuda',ae=identity['ae_path'],fm=identity['fm_path'],out=out,
                noise_seed=73500+seed,samples=64,steps=steps))
            view=view_samples(run,seed,auxiliary)
            require(len(view['sample_records'])==64,'Interrupted or dropped sample attempts')
            write_json(folder/('AUXILIARY_SAMPLING_COMPLETE.json' if auxiliary else 'PRIMARY_SAMPLING_COMPLETE.json'),
                {'steps':steps,'requested':64,'sample_manifest_sha256':view['sample_manifest_sha256'],
                 'failed_ids':[sid for sid,row in view['sample_records'].items() if row['status']!='ok'],'no_retry':True,
                 'excluded_from_primary_table':auxiliary})
    except BaseException as exc:
        write_json(folder/'SAMPLING_FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry_or_replacement':True});raise
    return folder


def export_final(run,seed):
    run=Path(run).resolve();inputs=view_samples(run,seed);folder=seed_folder(run,seed)
    require(inputs['primary'],'Never export auxiliary fields into primary table')
    require(len(inputs['sample_records'])==64,'Primary sample ledger incomplete')
    # Validate successful full bundles before the unchanged exporter reads nodes.
    for sid,row in inputs['sample_records'].items():
        if row['status']=='ok':common.load_sample(inputs,sid)
    before=inputs['sample_manifest_sha256']
    export_ours.run(Namespace(samples=inputs['sample_dir'],out=folder/'exports',a_seed=seed,faces=512,threads=1,exact=True))
    require(file_hash(inputs['sample_dir']/'SAMPLES.json')==before,'Sample bundle changed during export')
    _,ap,_=artifact_contract(run)
    write_json(folder/'EXPORT_IDENTITY.json',{'schema':SCHEMA,'scope':ARTIFACT_SCOPE,'training_protocol_sha256':inputs['identity']['training_protocol_sha256'],
        'artifact_protocol_sha256':inputs['identity']['artifact_protocol_sha256'],'original_A_sha256':ap['original_A_sha256'][str(seed)],
        'source_bundle_sha256':before,'support_sha256':ap['support_sha256'],'one_primary_mainfield_three_outlets':True,
        'outlet_manifest_sha256':{arm:file_hash(folder/'exports'/f'{arm}_MANIFEST.json') for arm in common.ARMS}})
    return folder/'exports'


def compare_arrays(primary,other,training_nodes):
    require(primary['noise'].shape==other['noise'].shape==(64,) and primary['noise'].dtype==other['noise'].dtype==np.float32
        and array_hash(primary['noise'])==array_hash(other['noise']),'Comparison does not use identical saved epsilon')
    require(primary['nodes'].shape==other['nodes'].shape==(2050,) and np.shape(training_nodes)==(16,2050),'Wrong node/support dimensions')
    require(all(np.isfinite(x).all() for x in [primary['nodes'],other['nodes'],training_nodes]),'Nonfinite node comparison')
    a=np.asarray(primary['nodes'],np.float64);b=np.asarray(other['nodes'],np.float64);y=np.asarray(training_nodes,np.float64)
    first=np.sqrt(np.square(y-a).mean(-1));second=np.sqrt(np.square(y-b).mean(-1))
    return {'same_epsilon':True,'noise_sha256':array_hash(primary['noise']),
        'paired_logradius_node_RMS':float(np.sqrt(np.square(a-b).mean())),
        'first_nearest_training_logradius_RMS':float(first.min()),'second_nearest_training_logradius_RMS':float(second.min()),
        'first_nearest_parent_index':int(first.argmin()),'second_nearest_parent_index':int(second.argmin()),
        'first_nodes_sha256':array_hash(primary['nodes']),'second_nodes_sha256':array_hash(other['nodes']),
        'not_a_surface_distance_or_generalization_metric':True}


def auxiliary_diagnostic(run,seed):
    run=Path(run).resolve();primary=view_samples(run,seed);aux=view_samples(run,seed,True)
    out=new_dir(seed_folder(run,seed)/'auxiliary_e128'/'diagnostics');rows=[]
    with np.load(run/'data/radial_train.npz',allow_pickle=False) as z:y=z['logr'];uids=z['uids'].tolist()
    for sid in common.expected_ids():
        row={'seed':seed,'sample_id':sid,'status':'failed','primary_steps':32,'auxiliary_steps':128,'excluded_from_primary_table':True}
        try:
            a,ai=common.load_sample(primary,sid);b,bi=common.load_sample(aux,sid)
            row.update(compare_arrays(a,b,y),status='ok',primary_sample_sha256=ai['sample_npz_sha256'],aux_sample_sha256=bi['sample_npz_sha256'])
        except Exception as exc:row['error']=repr(exc)
        rows.append(row)
    good=[r for r in rows if r['status']=='ok']
    result={'scope':'PREDECLARED_E128_AUXILIARY_ONLY','rows':rows,'planned':64,'success':len(good),
        'failed':64-len(good),'artifact_protocol_sha256':primary['identity']['artifact_protocol_sha256'],
        'primary_unchanged_and_not_selected':True,'no_auxiliary_meshes_or_primary_metric_rows':True,
        'mean_paired_node_RMS':float(np.mean([r['paired_logradius_node_RMS'] for r in good])) if good else None,
        'nearest_parent_counts':{name:{uid:sum(r[name+'_nearest_parent_index']==i for r in good) for i,uid in enumerate(uids)} for name in ('first','second')},
        'successful_subset_mean_not_failure_free_distribution_qualification':True}
    write_json(out/'RESULTS.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','sample','export','auxiliary','all'])
    parser.add_argument('--run',required=True);parser.add_argument('--seed',type=int,choices=[10,11]);args=parser.parse_args()
    if args.command=='prepare':prepare(args.run);return
    if args.seed is None:parser.error('--seed is required')
    if args.command in ('sample','all'):sample_final(args.run,args.seed)
    if args.command in ('export','all'):export_final(args.run,args.seed)
    if args.command in ('auxiliary','all'):auxiliary_diagnostic(args.run,args.seed)


if __name__=='__main__':main()
