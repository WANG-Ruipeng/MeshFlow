"""Fixed50000 EMA-only official sampling with immutable64-slot accounting."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np
import torch
from .. import official_sample
from ..io_utils import new_dir,write_json,file_hash,array_hash
from ..system_probe.official import require,exact_same,configure
from ..system_probe import ours_metrics as ours
from .train import load_contract,validate_checkpoint,FORMAL_SCOPE

SCOPE='CLOSED16_FIXED50000_EMA_RAW_PUBLIC_CODE_NOT_PAPER_FULL'
SAMPLES=64
ARMS={'raw':'meshflow_raw','post':'meshflow_public_post'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def slots():return [f'{i:05d}' for i in range(SAMPLES)]
def inside(root,relative):return ours.inside(root,relative)


def source_hashes(repo):
    names=['experiments/a_radial/system_formal/'+n for n in ['sample.py','finish.py','test_sample.py','test_train.py']]
    names+=['experiments/a_radial/'+n for n in ['official_sample.py','evaluate.py','registry.py','mesh_ops.py',
        'exporter.py','io_utils.py','system_probe/ours_metrics.py','layout_value/geometry.py']]
    names += [str(p.relative_to(repo)) for p in sorted((repo/'experiments/a_radial/legacy').glob('*.py'))]
    return {name:file_hash(repo/name) for name in names}


def register_ours(source,data_sha):
    source=Path(source).resolve();p=read(source/'RUN_PROTOCOL.json');ours.verify_protocol(source,p)
    require(p['data_hashes']['DATASET.json']==data_sha,'Ours and official use different16 data')
    records={}
    for seed in (10,11):
        inputs=ours.seed_inputs(source,seed,p)
        require(len(inputs['sample_records'])==64 and all(len(rows)==64 for rows in inputs['outlet_records'].values()),'Original8k ledgers must retain64 slots')
        for sid in slots():
            record=inputs['sample_records'][sid]
            if record['status']=='ok':ours.load_sample(inputs,sid)
            for arm in ours.ARMS:
                row=ours.verify_outlet_binding(inputs,arm,sid,record)
                if row['status']=='ok':require(file_hash(inside(inputs['export_dir'],row['mesh_path']))==row['mesh_sha256'],'Ours mesh changed')
        records[str(seed)]={'sample_manifest_sha256':inputs['sample_manifest_sha256'],
            'outlet_manifest_sha256':inputs['outlet_manifest_sha256'],'checkpoint_sha256':inputs['checkpoint_sha256']}
    return {'run':str(source),'protocol_sha256':file_hash(source/'RUN_PROTOCOL.json'),'AE_step':3200,'FM_step':8000,
        'euler_steps':32,'samples_per_seed':64,'seeds':records,'main_table_ours_reference':True}


def prepare(run,ours_run,tail_run=None):
    run=Path(run).resolve();p,cfg,identity=load_contract(run);repo=Path(p['repository'])
    require(not torch.cuda.is_initialized(),'Preparation must not initialize CUDA')
    require(cfg['sample']['sampling_method']=='euler' and cfg['sample']['num_sampling_steps']==50
        and cfg['sample']['cfg_scale']==2. and cfg['sample']['requested_faces']==512,'Original sample config changed')
    original=register_ours(ours_run,p['dataset_manifest_sha256'])
    optional=None
    if tail_run is not None:
        from ..fm_tail.artifacts import artifact_contract
        tail_run=Path(tail_run).resolve();tp,ta,tsha=artifact_contract(tail_run)
        require(tp['source_protocol_sha256']==original['protocol_sha256'],'Tail is not continuation of registered primary8k')
        require(tp['source_data_hashes']['DATASET.json']==p['dataset_manifest_sha256'],'Tail data differ')
        optional={'run':str(tail_run),'training_protocol_sha256':file_hash(tail_run/'RUN_PROTOCOL.json'),
            'artifact_protocol_sha256':tsha,'AE_step':3200,'FM_step':64000,'euler_steps':32,
            'separate_table_never_overwrites_main':True,'required_for_main_completion':False}
    sources=source_hashes(repo);protected=read(run/'SOURCE_MANIFEST.json')['protected_sha256']
    protocol={'schema':1,'scope':SCOPE,'training_protocol_sha256':identity['protocol_sha256'],
        'training_scope':FORMAL_SCOPE,'fixed_checkpoint_step':50000,'fixed_weight':'ema','EMA_decay':.9999,
        'seeds':[10,11],'requested_samples_per_seed':64,'noise_seeds':{'10':93610,'11':93611},
        'noise_shape_per_seed':[64,800,9],'noise_source':'local CPU torch.Generator, original sampler, float32',
        'sample_config':cfg['sample'],'Euler_time_grid_points':50,'CFG_scale':2.,
        'observed_ODE_RHS_calls':'record actual calls; grid points are not relabeled as measured calls',
        'public_post':{'level':'PUBLIC_CODE_NOT_PAPER_FULL','paper_full_qualified':False,
            'merge_duplicates':True,'merge_tolerance':0.,'fill_holes':False,'max_hole_edges':200},
        'raw':'original unquantized triangle soup tokens times0.3762/2, no welding/quantization/repair',
        'primary_ours':original,'optional_tail_ours':optional,
        'evaluation':{'rows':['meshflow_raw','meshflow_public_post','ours_fixed','ours_a','ours_qem'],
            'points':2048,'exact':True,'four_fixed16_groups_per_seed':True,'main_comparison':'original8k_ours_vs_official50000_EMA'},
        'data':p['data'],'dataset_manifest_sha256':p['dataset_manifest_sha256'],
        'raw_reference_sha256':{uid:file_hash(Path(p['data'])/'raw'/f'{uid}.npz') for uid in p['parent_uids']},
        'source_sha256':sources,'upstream_protected11_sha256':protected,
        'CUDA_build':torch.version.cuda,'torch':str(torch.__version__),'numpy':np.__version__,
        'TF32':False,'deterministic_algorithms':True,'source_outputs_immutable':True,
        'sampling_retries_or_replacements':False,'incomplete_or_failed_source_slots':'explicit failed rows in separate evaluation view',
        'unavailable_post_dependency_is_not_an_algorithm_failure':True,
        'fixed_budget_completion_is_not_convergence_or_publication_qualification':True,
        'closed_training16_only_not_class_generalization':True}
    write_json(run/'SAMPLING_PROTOCOL.json',protocol)
    for name in sources:
        target=run/'sampling_source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    return protocol


def contract(run,check_ours=False):
    run=Path(run).resolve();p,cfg,identity=load_contract(run);sp=read(run/'SAMPLING_PROTOCOL.json')
    require(sp['schema']==1 and sp['scope']==SCOPE and sp['training_protocol_sha256']==identity['protocol_sha256'],'Sampling/training protocol binding changed')
    require(sp['fixed_checkpoint_step']==50000 and sp['fixed_weight']=='ema' and sp['seeds']==[10,11],'Only final50000 EMA for both seeds')
    require(sp['requested_samples_per_seed']==64 and sp['noise_seeds']=={'10':93610,'11':93611}
        and sp['noise_shape_per_seed']==[64,800,9],'Fixed sample/noise registration changed')
    exact_same(sp['sample_config'],cfg['sample'],'sampling_config')
    exact_same(sp['public_post'],{'level':'PUBLIC_CODE_NOT_PAPER_FULL','paper_full_qualified':False,
        'merge_duplicates':True,'merge_tolerance':0.,'fill_holes':False,'max_hole_edges':200},'public_post_scope')
    require(sp['dataset_manifest_sha256']==p['dataset_manifest_sha256'] and sp['data']==p['data'],'Sampling dataset changed')
    require(set(sp['raw_reference_sha256'])==set(p['parent_uids']),'Reference parent IDs changed')
    exact_same(sp['source_sha256'],source_hashes(Path(p['repository'])),'sampling_source')
    exact_same(sp['upstream_protected11_sha256'],read(run/'SOURCE_MANIFEST.json')['protected_sha256'],'official11')
    for name,digest in sp['source_sha256'].items():require(file_hash(run/'sampling_source_snapshot'/name)==digest,'Sampling source snapshot changed')
    for uid,digest in sp['raw_reference_sha256'].items():require(file_hash(Path(sp['data'])/'raw'/f'{uid}.npz')==digest,'Raw evaluation reference changed')
    require(sp['CUDA_build']==torch.version.cuda and sp['torch']==str(torch.__version__) and sp['numpy']==np.__version__,'Sampling runtime changed')
    if check_ours:exact_same(sp['primary_ours'],register_ours(sp['primary_ours']['run'],p['dataset_manifest_sha256']),'registered_primary8k')
    return p,cfg,identity,sp,file_hash(run/'SAMPLING_PROTOCOL.json')


def validate_final(state,done,identity,cfg,seed):
    expected={**identity,'seed':seed}
    require(done['status']=='FIXED_BUDGET_COMPLETE_UNQUALIFIED' and done['scope']==FORMAL_SCOPE
        and done['step']==50000 and done['seed']==seed,'Training completion is not this final50000 seed')
    require(done['primary_weight']=='ema' and done['protocol_sha256']==identity['protocol_sha256'],'Only registered EMA completion')
    require(state['step']==50000 and state['is_predeclared_final'] is True,'Probe/intermediate checkpoint cannot be sampled')
    validate_checkpoint(state,expected,cfg,FORMAL_SCOPE)
    require(done['parent_exposures']==[50000]*16,'Completion parent exposure mismatch')
    require('rng' in state and 'sampler_rng' in state and state['rng']['torch_cuda'],'Incomplete CUDA/RNG final checkpoint')
    require(state['model'].keys()==state['ema'].keys(),'EMA architecture differs from current model')
    for name,value in state['ema'].items():
        require(value.shape==state['model'][name].shape and value.dtype==state['model'][name].dtype,'EMA shape/dtype mismatch')


def final_identity(run,seed):
    run=Path(run).resolve();p,cfg,identity,sp,sha=contract(run)
    require(seed in sp['seeds'],'Unregistered training seed')
    root=run/'training'/f'seed_{seed}';done_path=root/'TRAINING_COMPLETE.json';done=read(done_path)
    path=Path(done['primary_checkpoint']).resolve();require(path.is_relative_to((root/'invocations').resolve()),'Checkpoint escapes formal seed invocations')
    require(path.name=='step00050000.pt','Only fixed final50000 path')
    record_path=path.parent/'checkpoints'/(path.stem+'.json');record=read(record_path)
    require(Path(record['path']).resolve()==path and record['step']==50000,'Checkpoint completion record path/step differs')
    digest=file_hash(path);require(digest==record['sha256']==done['primary_checkpoint_sha256']
        and path.stat().st_size==record['bytes'],'Final checkpoint SHA/size differs')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);validate_final(state,done,identity,cfg,seed)
    probe_path=path.parent/'probes'/'step00050000.json';probe=read(probe_path)
    require(probe['step']==50000 and probe['not_a_checkpoint_selection_criterion'] is True,'Final training-support probe missing')
    require(file_hash(path)==digest,'Final checkpoint changed during validation')
    result={'scope':SCOPE,'seed':seed,'step':50000,'weight':'ema','training_protocol_sha256':identity['protocol_sha256'],
        'sampling_protocol_sha256':sha,'checkpoint':str(path),'checkpoint_sha256':digest,
        'checkpoint_record_sha256':file_hash(record_path),'training_complete_sha256':file_hash(done_path),
        'ema_sha256':state['ema_sha256'],'model_sha256':state['model_sha256'],
        'dataset_manifest_sha256':identity['dataset_sha256'],'source_manifest_sha256':identity['source_manifest_sha256'],
        'final_fit_probe':str(probe_path),'final_fit_probe_sha256':file_hash(probe_path),
        'sample_config':cfg['sample'],'noise_seed':93600+seed,'requested_samples':64,
        'fixed_budget_not_convergence_qualification':True,'paper_full_post_qualified':False}
    del state;gc.collect();return result


def inventory(root):
    root=Path(root)
    return {str(path.relative_to(root)):file_hash(path) for path in sorted(root.rglob('*')) if path.is_file()} if root.exists() else {}


def base_row(identity,arm,sid,noise):
    row={'row':ARMS[arm],'sample_id':sid,'checkpoint_sha256':identity['checkpoint_sha256'],
        'dataset_manifest_sha256':identity['dataset_manifest_sha256'],'requested_faces':512,
        'noise_sha256':array_hash(noise),'config':identity['sample_config'],'reference_scope':'closed_set16',
        'fit_qualification':'FIXED_BUDGET_NOT_CONVERGENCE_PROOF','status':'failed'}
    if arm=='post':row.update(post_level='PUBLIC_CODE_NOT_PAPER_FULL',paper_full_qualified=False,
        post_config={'merge_duplicates':True,'merge_tolerance':0.,'fill_holes':False,'max_hole_edges':200})
    return row


def parse_source_manifest(path,arm):
    manifest=read(path);require(manifest['row']==ARMS[arm] and manifest['requested_samples']==64,'Source row/count changed')
    ids=[r['sample_id'] for r in manifest['records']]
    require(len(ids)==len(set(ids)) and set(ids)<=set(slots()),'Duplicate/unregistered source IDs')
    require(all(r.get('status') in ('ok','failed') for r in manifest['records']),'Unknown source status')
    return {r['sample_id']:r for r in manifest['records']}


def verify_source_row(row,identity,arm,sid,noise,generated,raw_record=None):
    expected=base_row(identity,arm,sid,noise)
    for key in ['row','sample_id','checkpoint_sha256','dataset_manifest_sha256','requested_faces','noise_sha256','config','reference_scope']:
        exact_same(row.get(key),expected[key],'source_record.'+key)
    if arm=='post':
        for key in ['post_level','paper_full_qualified','post_config']:exact_same(row.get(key),expected[key],'source_record.'+key)
    if row['status']!='ok':raise RuntimeError('Preserved source failure: '+str(row.get('error')))
    path=inside(generated,row['mesh_path']);require(file_hash(path)==row['mesh_sha256'],'Source mesh file SHA changed')
    with np.load(path,allow_pickle=False) as z:mesh={k:z[k].copy() for k in z.files}
    v,f=mesh['vertices'],mesh['faces']
    require(v.ndim==2 and v.shape[1]==3 and np.isfinite(v).all() and f.ndim==2 and f.shape[1]==3
        and np.issubdtype(f.dtype,np.integer),'Malformed stored mesh arrays')
    require(row['actual_faces']==len(f),'Source actual face count differs from file')
    if arm=='raw':
        require(set(mesh)=={'vertices','faces','tokens','noise'},'Raw archive schema differs')
        require(mesh['tokens'].shape==(512,9) and mesh['tokens'].dtype==np.float32 and np.isfinite(mesh['tokens']).all(),'Raw tokens changed')
        require(array_hash(mesh['noise'])==array_hash(noise),'Raw saved epsilon differs from fixed slot')
        require(array_hash(mesh['tokens'])==row['raw_tensor_sha256'],'Raw token hash changed')
        require(np.array_equal(v,mesh['tokens'].reshape(-1,3)*(.3762*.5))
            and np.array_equal(f,np.arange(1536,dtype=np.int64).reshape(512,3)),'Raw coordinates/connectivity changed')
        require(row['time_grid_points']==50 and row['CFG_batch_factor']==2,'Raw ODE/CFG record differs')
    else:
        require(raw_record is not None and raw_record['status']=='ok','Public post has no valid paired Raw')
        require(row['source_raw_tensor_sha256']==raw_record['raw_tensor_sha256'],'Post is from a different raw tensor')
        quantized=inside(generated,'public_quantized/'+sid+'.obj')
        require(file_hash(quantized)==row['quantized_export_sha256'],'Public quantized intermediate changed')
    return path


def finalize_view(folder,identity,returncode):
    """Never resample: preserve original artifacts, copy validated meshes, fill missing slots."""
    folder=Path(folder);generated=folder/'generated';before=inventory(generated)
    view=new_dir(folder/'evaluation_view');generator=torch.Generator(device='cpu').manual_seed(identity['noise_seed'])
    noise=torch.randn(64,800,9,generator=generator).numpy();maps={};errors={};valid_raw={};ledger=[]
    for arm in ARMS:
        try:maps[arm]=parse_source_manifest(generated/f'{arm}_MANIFEST.json',arm)
        except Exception as exc:maps[arm]={};errors[arm]=repr(exc)
    for arm in ARMS:
        rows=[]
        for i,sid in enumerate(slots()):
            source=maps[arm].get(sid);row=base_row(identity,arm,sid,noise[i]);row['no_retry_or_replacement']=True
            try:
                require(source is not None,'Missing source slot: '+errors.get(arm,'sampling stopped before this attempt'))
                path=verify_source_row(source,identity,arm,sid,noise[i],generated,valid_raw.get(sid))
                target=view/arm/(sid+'.npz');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                require(file_hash(target)==source['mesh_sha256'],'Evaluation copy changed geometry bytes')
                row=deepcopy(source);row.update(mesh_path=str(target.relative_to(view)),no_retry_or_replacement=True,
                    original_mesh_path=source['mesh_path'],copied_bytes_sha256=file_hash(target))
                if arm=='raw':valid_raw[sid]=source
            except Exception as exc:
                row.update(status='failed',error=repr(exc),source_record=source,
                    source_actual_faces=None if source is None else source.get('actual_faces'))
            rows.append(row);ledger.append({'arm':arm,'sample_id':sid,'status':row['status'],'error':row.get('error')})
        write_json(view/f'{arm}_MANIFEST.json',{'row':ARMS[arm],'requested_samples':64,'records':rows,
            'ledger_scope':'original attempts copied to immutable evaluation view; missing slots explicitly failed',
            'sampling_protocol_sha256':identity['sampling_protocol_sha256'],
            'paper_full_post':'NOT_IMPLEMENTED_WITHOUT_VERIFIED_DENOISER_AND_RECIPE' if arm=='post' else None})
    exact_same(before,inventory(generated),'original_sampler_outputs_unchanged')
    result={'scope':SCOPE,'identity':identity,'sampler_returncode':returncode,'expected_per_row':64,'actual_per_row':64,
        'successful':{arm:sum(r['arm']==arm and r['status']=='ok' for r in ledger) for arm in ARMS},
        'failed':{arm:sum(r['arm']==arm and r['status']!='ok' for r in ledger) for arm in ARMS},
        'manifest_errors':errors,'all_slots':ledger,'original_inventory_sha256':before,'evaluation_inventory_sha256':inventory(view),
        'no_resampling':True,'paper_full_qualified':False,'fill_holes':False,'fixed_budget_not_convergence_qualification':True}
    write_json(folder/'SAMPLING_COMPLETE.json',result);return result


def audit_completed(folder,identity):
    folder=Path(folder);done=read(folder/'SAMPLING_COMPLETE.json');exact_same(done['identity'],identity,'saved_sampling_identity')
    exact_same(done['original_inventory_sha256'],inventory(folder/'generated'),'source_outputs')
    exact_same(done['evaluation_inventory_sha256'],inventory(folder/'evaluation_view'),'evaluation_outputs')
    generator=torch.Generator(device='cpu').manual_seed(identity['noise_seed'])
    noise=torch.randn(64,800,9,generator=generator).numpy()
    for arm in ARMS:
        manifest=read(folder/'evaluation_view'/f'{arm}_MANIFEST.json')
        require(manifest['requested_samples']==64 and [r['sample_id'] for r in manifest['records']]==slots(),'Evaluation lost/reordered64 slots')
        require(all(r['row']==ARMS[arm] for r in manifest['records']),'Evaluation row label changed')
        require(done['successful'][arm]==sum(r['status']=='ok' for r in manifest['records'])
            and done['failed'][arm]==sum(r['status']=='failed' for r in manifest['records'])==64-done['successful'][arm],
            'Evaluation success/failure accounting changed')
        for index,row in enumerate(manifest['records']):
            expected=base_row(identity,arm,slots()[index],noise[index])
            for key in ['row','sample_id','checkpoint_sha256','dataset_manifest_sha256','requested_faces','noise_sha256','config','reference_scope']:
                exact_same(row.get(key),expected[key],'evaluation_record.'+key)
            if arm=='post':
                for key in ['post_level','paper_full_qualified','post_config']:exact_same(row.get(key),expected[key],'evaluation_post.'+key)
            if row['status']=='ok':
                require(file_hash(inside(folder/'evaluation_view',row['mesh_path']))==row['mesh_sha256']==row['copied_bytes_sha256'],
                    'Evaluation mesh differs from copied source bytes')
    require(done['paper_full_qualified'] is False and done['fill_holes'] is False,'False OfficialFull claim')
    return done


def child(run,seed,out):
    run=Path(run).resolve();identity=final_identity(run,seed)
    expected=run/'posttraining'/f'seed_{seed}'/'generated';require(Path(out).resolve()==expected,'Unregistered sampler output path')
    invocation=read(expected.parent/'INVOCATION.json');exact_same(invocation['identity'],identity,'child_final_identity')
    configure(seed);require(torch.cuda.is_available(),'Original official CFG requires CUDA')
    previous=sys.argv
    try:
        sys.argv=['official_sample','--repo',str(read(run/'RUN_PROTOCOL.json')['repository']),
            '--checkpoint',identity['checkpoint'],'--out',str(expected),'--samples','64',
            '--noise-seed',str(identity['noise_seed']),'--public-post']
        # No --fill-holes. Original file and original EMA/transport/model implementation remain unchanged.
        official_sample.main()
    finally:sys.argv=previous


def run_seed(run,seed):
    run=Path(run).resolve();identity=final_identity(run,seed);folder=run/'posttraining'/f'seed_{seed}'
    if folder.exists():return audit_completed(folder,identity) # Never rerun even a failed sampling attempt.
    new_dir(folder);command=[sys.executable,'-B','-m','experiments.a_radial.system_formal.sample','_child',
        '--run',str(run),'--seed',str(seed),'--out',str(folder/'generated')]
    write_json(folder/'INVOCATION.json',{'identity':identity,'command':command,'automatic_retries':False,
        'wrapper_source_sha256':file_hash(__file__),'original_sampler_sha256':file_hash(official_sample.__file__)})
    process=None
    try:
        with (folder/'sampler.log').open('x',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[3],stdout=stream,stderr=subprocess.STDOUT)
            code=process.wait()
        result=finalize_view(folder,identity,code);contract(run);return result
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        ledger_error=None
        if not (folder/'SAMPLING_COMPLETE.json').exists() and not (folder/'evaluation_view').exists():
            try:finalize_view(folder,identity,None if process is None else process.returncode)
            except Exception as ledger_exc:ledger_error=repr(ledger_exc)
        write_json(folder/'WRAPPER_FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'no_automatic_retry':True,'all_existing_source_outputs_retained':True,'failure_ledger_error':ledger_error});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run','audit','_child'])
    parser.add_argument('--run',required=True);parser.add_argument('--ours-run');parser.add_argument('--tail-run')
    parser.add_argument('--seed',type=int,choices=[10,11]);parser.add_argument('--out');args=parser.parse_args()
    if args.command=='prepare':
        if not args.ours_run:parser.error('--ours-run required')
        prepare(args.run,args.ours_run,args.tail_run);return
    if args.seed is None:parser.error('--seed required')
    if args.command=='run':run_seed(args.run,args.seed)
    elif args.command=='audit':audit_completed(Path(args.run)/'posttraining'/f'seed_{args.seed}',final_identity(args.run,args.seed))
    else:
        if args.out is None:parser.error('--out required for internal child')
        child(args.run,args.seed,args.out)


if __name__=='__main__':main()
