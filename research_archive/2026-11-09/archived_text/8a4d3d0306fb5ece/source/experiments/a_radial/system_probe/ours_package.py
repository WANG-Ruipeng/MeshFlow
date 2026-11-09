"""CPU-only provenance and archive audit of the immutable original AE3200/FM8000 run.

No optimizer, generation, new geometric evaluation, or CUDA forward is executed.
Original files and all failures are retained. Absolute provenance paths in original
JSON are preserved; PORTABLE_PATHS maps archived copies without rewriting history.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
import torch
from ..io_utils import array_hash,file_hash,model_hash,new_dir,write_json
from ..models import ShapeAE,ShapeFM
from ..exporter import RadialSupport
from . import ours_metrics as metrics
from . import ours_timing as timing

# Explicit whitelist, including import-only dependencies of official.configure.
SOURCE_NAMES=(
    '__init__.py','models.py','train_ours.py','sample_ours.py','export_ours.py','exporter.py',
    'evaluate.py','registry.py','io_utils.py','mesh_ops.py','prepare_data.py',
    'official_contract.py','official_train.py','verify_repo.py',
    'layout_student/__init__.py','layout_student/train.py','layout_value/__init__.py','layout_value/geometry.py',
    'system_probe/__init__.py','system_probe/official.py','system_probe/ours.py','system_probe/ours_run.py',
    'system_probe/ours_metrics.py','system_probe/ours_timing.py','system_probe/ours_diagnose.py',
    'system_probe/bayes_review.py','system_probe/ours_gallery.py','system_probe/ours_figures.py',
    'system_probe/test_ours_metrics.py','system_probe/test_ours_timing.py',
    'system_probe/ours_package.py','system_probe/test_ours_package.py',
    'legacy/__init__.py','legacy/audit_fast.py','legacy/chart.py','legacy/common.py','legacy/deform.py',
    'legacy/exact_accel.py','legacy/exact_triangles.py','legacy/geometry_task.py','legacy/networks.py',
    'assets/support.npz','assets/original_A_10.pt','assets/original_A_11.pt','assets/parents.json','assets/UPSTREAM_LOCK.json')


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def compare(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        require(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
            and torch.equal(a.detach().cpu(),b.detach().cpu()),path)
    elif isinstance(a,np.ndarray):
        require(isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),path)
    elif isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),path+'.keys')
        for k in a:compare(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        require(type(a)==type(b) and len(a)==len(b),path+'.length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    else:require(a==b,path)


def files_manifest(root):
    root=Path(root);files={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'Symlink not allowed in immutable package: '+str(path))
        if path.is_file():files[str(path.relative_to(root)).replace('\\','/')]={'sha256':file_hash(path),'bytes':path.stat().st_size}
    return files


def verify_files(root,expected,exclude=()):
    actual=files_manifest(root)
    for name in exclude:actual.pop(name,None)
    compare(actual,expected,'file_manifest')
    return len(actual)


def copied_sources(repo):
    root=Path(repo)/'experiments/a_radial'
    return {str(Path('experiments/a_radial')/name):root/name for name in SOURCE_NAMES}


def training_audit(run,protocol,gate):
    root=Path(__file__).parents[1];metrics.verify_protocol(run,protocol)
    complete=read(run/'TRAINING_COMPLETE.json')
    require(complete['actual_optimizer_updates']==22400 and complete['fixed_final_selection'] is True
        and complete['scientific_quality_pass'] is False,'Wrong original training budget/claim')
    require(file_hash(gate/'GATE.json')==protocol['gate_sha256'],'Training gate identity differs')
    g=read(gate/'GATE.json');require(g['status']=='PASS' and g['actual_optimizer_updates']==16
        and g['source_and_data_unchanged'],'Original CUDA gate incomplete')
    compare(g['data_hashes'],protocol['data_hashes'],'gate data')
    require(g['original_trainer_sha256']==file_hash(root/'train_ours.py'),'Gate trainer changed')
    gate_rows=[]
    for stage in ('ae','fm'):
        a=gate/stage/'whole/step00000004.pt';b=gate/stage/'resumed/step00000004.pt';e=g['stages'][stage]
        require(file_hash(a)==e['continuous_checkpoint_sha256'] and file_hash(b)==e['resume_checkpoint_sha256'],'Gate checkpoint bytes changed')
        compare(torch.load(a,map_location='cpu',weights_only=True),torch.load(b,map_location='cpu',weights_only=True),stage+' full-state gate')
        gate_rows.append({'stage':stage,'continuous4_vs_resume2plus2_saved_state_exact':True,'not_formal_weights':True})
    stages=[]
    for seed in (10,11):
        for stage,cls in [('ae',ShapeAE),('fm',ShapeFM)]:
            folder=run/f'seed{seed}'/stage;steps=protocol[stage]['steps'];path=folder/f'step{steps:08d}.pt'
            ck=torch.load(path,map_location='cpu',weights_only=True);cfg=read(folder/'CONFIG.json');done=read(folder/'COMPLETE.json')
            compare(ck['config'],cfg,'checkpoint config')
            require(ck['step']==cfg['steps']==done['step']==steps and cfg['seed']==seed and cfg['stage']==stage,'Final identity')
            require(cfg['device']=='cuda' and cfg['batch']==protocol[stage]['batch'] and cfg['lr']==protocol[stage]['lr'],'Final settings')
            require(cfg['dataset_sha256']==protocol['data_hashes']['radial_train.npz']
                and cfg['dataset_manifest_sha256']==protocol['data_hashes']['DATASET.json'],'Final data')
            expected_ae=file_hash(run/f'seed{seed}/ae/step00003200.pt') if stage=='fm' else None
            require(cfg['ae_sha256']==expected_ae,'FM/AE pairing')
            entries=[x for x in complete['stages'] if x['seed']==seed and x['stage']==stage]
            require(len(entries)==1 and entries[0]['fixed_final'] and entries[0]['final_checkpoint']==str(path.relative_to(run)), 'Final stage ledger')
            require(file_hash(path)==done['checkpoint_sha256']==entries[0]['final_sha256'],'Final file hash')
            sched=ck['scheduler'];require(sched['last_epoch']==steps and sched['T_max']==steps and sched['eta_min']==cfg['lr']*.1,'Final cosine state')
            require(ck['optimizer']['state'] and all(int(x['step'])==steps for x in ck['optimizer']['state'].values()),'Final Adam updates')
            require(all(x['weight_decay']==.0001 and np.isclose(x['lr'],cfg['lr']*.1,rtol=0,atol=1e-15) for x in ck['optimizer']['param_groups']),'Final optimizer hyperparameters')
            require(ck['draw_queue']==[] and ck['batch_rng'].dtype==torch.uint8 and ck['rng']['torch_cuda'],'Complete sampler/RNG state')
            model=cls();model.load_state_dict(ck['model'],strict=True)
            require(sum(p.numel() for p in model.parameters())==protocol[stage]['parameters'] and model_hash(model)==done['state_sha256'],'Final architecture/model hash')
            require(all(bool(torch.isfinite(t).all()) for t in ck['model'].values()),'Nonfinite learned tensors')
            trace=read(folder/'TRACE.json');require([x['step'] for x in trace]==list(range(100,steps+1,100)),'Logged trace omitted/reordered')
            require(all(np.isfinite(x['train_batch_loss']) and np.isfinite(x['grad_norm']) for x in trace),'Nonfinite trace')
            exposures=steps*cfg['batch']//16;require(exposures==entries[0]['parent_exposures_each'],'Exposure accounting')
            stages.append({'seed':seed,'stage':stage,'steps':steps,'file_sha256':file_hash(path),'model_sha256':done['state_sha256'],
                'parent_exposures_each':exposures,'optimizer_scheduler_sampler_verified':True,'trace_is_every100_not_every_update':True})
            del ck,model
    return {'stages':stages,'total_actual_updates':22400,'gate':gate_rows,'CUDA_executed_in_this_audit':False}


def metric_audit(run,p):
    result=read(run/'metrics/RESULTS.json');rows=result['rows'];expected={(s,i,a) for s in metrics.SEEDS for i in metrics.expected_ids() for a in metrics.ARMS}
    require(len(rows)==384 and len({(r['seed'],r['sample_id'],r['arm']) for r in rows})==384
        and {(r['seed'],r['sample_id'],r['arm']) for r in rows}==expected,'Metric slots')
    require(result['metric_success_count']==384 and not result['provenance_errors'] and not result['failed_slots'],'Original metric errors')
    mp=read(run/'metrics/METRIC_PROTOCOL.json');root=Path(__file__).parents[1]
    require(mp['metric_source_sha256']==file_hash(root/'system_probe/ours_metrics.py')
        and mp['geometry_source_sha256']==file_hash(root/'layout_value/geometry.py'),'Metric source changed')
    require(mp['generation_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json') and mp['points_each_direction']==8192
        and mp['MC_seed_bases']==list(metrics.SEED_BASES),'Metric protocol')
    for seed in (10,11):require(mp['a_assets_current_sha256'][str(seed)]==file_hash(root/f'assets/original_A_{seed}.pt'),'Original A changed')
    require(mp['support_file_sha256']==file_hash(root/'assets/support.npz'),'Support changed')
    support=RadialSupport();mapping={(r['seed'],r['sample_id'],r['arm']):r for r in rows};failures=[]
    for seed in (10,11):
        inputs=metrics.seed_inputs(run,seed,p)
        require(len(inputs['sample_records'])==64 and all(len(v)==64 for v in inputs['outlet_records'].values()),'Missing samples or outputs')
        for sid in metrics.expected_ids():
            arrays,identity=metrics.load_sample(inputs,sid);rv,rf=support.mesh(arrays['nodes'])
            for arm in metrics.ARMS:
                row=mapping[(seed,sid,arm)];source=metrics.verify_outlet_binding(inputs,arm,sid,inputs['sample_records'][sid])
                v,f,mesh_identity=metrics.load_mesh(inputs,source,arm,arrays['nodes'],support)
                compare(read(run/'metrics/attempts'/f'seed{seed}'/sid/(arm+'.json')),row,'Metric row copy')
                for k,val in {**identity,**mesh_identity}.items():compare(val,row[k],'Metric stored identity '+k)
                require(row['reference_surface_hash']==array_hash(rv,rf),'RMS reference mainfield')
                compare(source['topology'],row['audit']['topology'],'Export vs metric topology')
                compare(source['exact'],row['audit']['exact'],'Export vs metric exact evidence')
                require(row['audit']['exact_executed'] and row['quality']['face_count']==len(f),'Actual faces/exact evidence')
                compare(metrics.quality_tails(v,f),row['quality'],'Readback quality tails')
                fixed=mapping[(seed,sid,'fixed')]
                for k,val in metrics.common_gate(row['quality'],row['audit'],fixed['quality'],fixed['audit']).items():compare(val,row[k],'Quality gate')
                require(len(row['surface_metrics'])==2,'MC count')
                for k,m in enumerate(row['surface_metrics']):
                    require(m['seed']==metrics.SEED_BASES[k]+(seed-10)*64+int(sid) and m['points_per_direction']==8192,'MC sampling')
                    value=np.sqrt(.5*(m['mesh_to_reference_rms_bbox_percent']**2+m['reference_to_mesh_rms_bbox_percent']**2))
                    require(np.isclose(value,m['rms_bbox_percent'],atol=1e-12,rtol=1e-13),'RMS formula')
                    require(np.isclose(m['rms_world_units'],m['rms_bbox_percent']*.01*m['reference_bbox_diagonal'],atol=1e-14,rtol=1e-13),'RMS units')
                for k,val in metrics.paired_metrics(row,fixed).items():compare(val,row[k],'Paired metric')
                if not row['audit']['valid'] or len(f)!=512:failures.append({'seed':seed,'sample_id':sid,'arm':arm,'faces':len(f),'intersection_pairs':row['audit']['selfintersection_count'],'topology_pass':row['audit']['topology_pass']})
        print(json.dumps({'audit':'metric_readback','seed':seed,'samples':64,'outputs':192}),flush=True)
    compare(metrics.summary_groups(rows),read(run/'metrics/SUMMARY.json')['groups'],'Metric summary')
    require(len(failures)==5 and all(x['arm']=='qem' for x in failures),'Failure ledger changed')
    return {'samples':128,'outputs':384,'all_saved_mesh_and_mainfield_hashes_verified':True,'readback_quality_tails_recomputed':True,
        'prior_exact_ledger_crosscheck':True,'RMS_seed_reference_and_algebra_checked':True,'new_384_exact_or_MC_recomputation':False,
        'new_generation_or_GPU_forward':False,'geometric_failures_retained':failures,'groups':metrics.summary_groups(rows)}


def distribution_audit(run):
    output=[]
    compare(read(run/'metrics/DISTRIBUTION_STATUS.json'),[{'seed':s,'returncode':0,'path':f'seed{s}/distribution'} for s in (10,11)],'Distribution completion')
    for seed in (10,11):
        root=run/f'seed{seed}/distribution';groups=read(root/'DISTRIBUTION_GROUPS.json');per=read(root/'PER_SAMPLE.json');matrix=read(root/'MATRIX.json')
        require(len(groups)==12 and len(per)==192 and len(matrix)==3,'Distribution denominators')
        require(len({(x['row'],x['sample_id']) for x in per})==192,'Distribution duplicate samples')
        for arm in metrics.ARMS:
            name='ours_'+arm;gg=[g for g in groups if g['row']==name];rr=[r for r in per if r['row']==name]
            require(len(rr)==64 and {r['sample_id'] for r in rr}==set(metrics.expected_ids()),'Distribution sample slots')
            require(len(gg)==4 and {g['offset'] for g in gg}=={0,16,32,48} and all(g['status']=='ok' and g['n_each']==16 for g in gg),'Distribution blocks')
            for g in gg:
                with np.load(root/f'{name}_group{g["offset"]:04d}.npz',allow_pickle=False) as z:C=z['cross_distance'];D=z['all_distance']
                require(C.shape==(16,16) and D.shape==(32,32) and np.isfinite(C).all() and (C>=0).all(),'Distribution matrix shape')
                require(np.all(np.isinf(np.diag(D))) and np.array_equal(D,D.T),'Distribution self exclusions')
                labels=np.r_[np.zeros(16,int),np.ones(16,int)]
                values={'MMD_CD':float(C.min(0).mean()),'coverage':len(np.unique(C.argmin(1)))/16,
                    'one_NNA':float((labels[D.argmin(1)]==labels).mean()),'nearest_train_distance_mean':float(C.min(1).mean())}
                for k,v in values.items():compare(v,g[k],'Distribution matrix statistic '+k)
            row=next(x for x in matrix if x['row']==name)
            require(row['attempts']==row['metrics_readable']==64 and row['valid_metric_groups']==4,'Distribution summary denominator')
            require(row['budget512_count']==sum(r['actual_faces']==512 for r in rr)
                and row['joint_valid_exact_count']==sum(r['joint_valid_exact'] for r in rr),'Distribution failure accounting')
            for key in ('MMD_CD','coverage','one_NNA'):compare(float(np.mean([g[key] for g in gg])),row[key],'Distribution group mean')
            output.append({'seed':seed,**row})
    return {'rows':output,'blocks_per_seed':12,'each_block_generated_and_reference':16,'new_surface_point_sampling':False,
        'statistics_recomputed_from_all_saved_distance_matrices':True,'invalid_geometry_not_filtered_from_readable_distribution':True}


def timing_audit(run,p):
    root=run/'timing';plan=read(root/'TIMING_PROTOCOL.json');result=read(root/'RESULTS.json');rows=result['rows']
    require(file_hash(root/'TIMING_PROTOCOL.json')==result['timing_protocol_sha256'] and plan['generation_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Timing identity')
    timing.verify_bound_inputs(plan)
    compare(timing.summarize(rows,result['cold_parent_processes']),read(root/'SUMMARY.json'),'Timing summary')
    require(len(rows)==120 and len({(r['seed'],r['sample_id']) for r in rows})==8,'Timing population')
    expected={};meshes={}
    for seed in (10,11):
        inputs=metrics.seed_inputs(run,seed,p)
        for sid in timing.IDS:
            expected[(seed,sid)]=metrics.load_sample(inputs,sid)[0]
            for arm in metrics.ARMS:
                src=inputs['outlet_records'][arm][sid]
                with np.load(inputs['export_dir']/src['mesh_path'],allow_pickle=False) as z:meshes[(seed,sid,arm)]={k:z[k].copy() for k in z.files}
    for row in rows:
        folder=root/'cold'/row['case_id'] if row['mode']=='cold' else root/'warm'/row['phase']/f'repeat{row["repeat"]}'/row['case_id']
        compare(read(folder/'RESULT.json'),row,'Timing row copy')
        require(row['runtime_status']=='ok','Timing runtime failed')
        for label in ('generated','mesh'):require(file_hash(folder/row[label+'_file'])==row[label+'_file_sha256'],'Timing saved file changed')
        with np.load(folder/row['generated_file'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        with np.load(folder/row['mesh_file'],allow_pickle=False) as z:mesh={k:z[k].copy() for k in z.files}
        replay=timing.replay_identity(expected[(row['seed'],row['sample_id'])],arrays,meshes[(row['seed'],row['sample_id'],row['arm'])],mesh)
        compare(replay,row['replay'],'Timing saved output exact replay evidence')
        require(replay['exact_production_replay'],'Timing output differs from production')
    return {'records':120,'cold_processes':24,'warmups':24,'warm_measured':72,'unique_samples':8,
        'all120_saved_generation_and_mesh_arrays_exact_against_production':True,'new_GPU_replay_executed':False,
        'historical_GPU_replay_evidence_preserved':True,'summary':read(root/'SUMMARY.json')}


def diagnostic_audit(run):
    root=run/'representation_diagnostics';r=read(root/'RESULT.json');b=read(root/'BAYES_ORACLE_REVIEW.json')
    require(r['protocol_sha256']==b['source_protocol_sha256']==file_hash(run/'RUN_PROTOCOL.json'),'Diagnostic protocol')
    require(r['source_sha256']==file_hash(Path(__file__).with_name('ours_diagnose.py')),'Representation source')
    require(b['original_representation_RESULT_sha256']==file_hash(root/'RESULT.json')
        and b['review_source_sha256']==file_hash(root/'BAYES_ORACLE_REVIEW.py')==file_hash(Path(__file__).with_name('bayes_review.py')),'Bayes source/results')
    require(b['training_updates']==0 and not b['generated_samples_changed'] and not b['model_selection_performed'],'Bayes scope')
    for seed in (10,11):
        row=r['seeds'][str(seed)];br=b['seeds'][str(seed)]
        require(len(row['records'])==64 and all(x['status']=='ok' for x in row['records']),'Diagnostic sample denominators')
        require(br['ae_sha256']==file_hash(run/f'seed{seed}/ae/step00003200.pt') and br['fm_sha256']==file_hash(run/f'seed{seed}/fm/step00008000.pt'),'Diagnostic final identity')
        require(br['probe_count']==4096 and sum(x['count'] for x in br['t_bins'])==4096,'Bayes probe counts')
        compare(br['network_pair_MSE'],row['fixed_velocity_probe_final_MSE'],'Bayes same fixed probe result')
    return {'CPU32_is_not_bitwise_GPU32_replay':True,'new_CPU_or_GPU_generation':False,
        'original_CPU32_vs_production_GPU32_max_abs_nodes':{s:r['seeds'][s]['CPU32_vs_production_GPU32_max_abs_nodes'] for s in ('10','11')},
        'auxiliary128_not_primary_or_checkpoint_selection':True,'fixed_probe_per_seed':4096,'PCA_initial_interpolates_train16':True}


def verify_archive_payload(root):
    root=Path(root);manifest=read(root/'FILE_MANIFEST.json');verify_files(root,manifest['files'],exclude=('FILE_MANIFEST.json',))
    # Path-independent readback after actual extraction, with no original paths.
    run=root/'run';p=read(run/'RUN_PROTOCOL.json');sample_count=mesh_count=0
    for seed in (10,11):
        folder=run/f'seed{seed}';samples=read(folder/'samples/SAMPLES.json');sm={r['sample_id']:r for r in samples['records']}
        require(len(sm)==64,'Portable sample count')
        for sid,row in sm.items():
            with np.load(metrics.inside(folder/'samples',row['path']),allow_pickle=False) as z:
                require(array_hash(z['noise'],z['z_standardized'],z['z'],z['nodes'])==row['array_hash']
                    and array_hash(z['nodes'])==row['nodes_hash'],'Portable mainfield array hash')
            sample_count+=1
        for arm in metrics.ARMS:
            records=read(folder/'exports'/f'{arm}_MANIFEST.json')['records'];require(len(records)==64,'Portable export count')
            for row in records:
                require(row['status']=='ok' and file_hash(metrics.inside(folder/'exports',row['mesh_path']))==row['mesh_sha256'],'Portable export identity')
                require(row['source_nodes_sha256']==sm[row['sample_id']]['nodes_hash'],'Portable same-field binding');mesh_count+=1
    return {'files':len(manifest['files']),'samples':sample_count,'outputs':mesh_count,'all_file_SHA256_and_size_match':True,
        'portable_saved_array_readback':True,'new_GPU_replay':False}


def extract_and_verify(archive,target):
    target=new_dir(target)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate ZIP members')
        for name in names:
            path=(target/name).resolve();require(path.is_relative_to(target.resolve()),'ZIP path traversal')
        require(z.testzip() is None,'ZIP CRC failure');z.extractall(target)
    return verify_archive_payload(target)


def package(run,gate,out):
    require(not torch.cuda.is_initialized(),'CPU-only archive');torch.set_num_threads(1)
    run=Path(run).resolve();gate=Path(gate).resolve();out=new_dir(out);started=time.perf_counter()
    try:
        originals=files_manifest(run);p=read(run/'RUN_PROTOCOL.json');data=Path(p['data']);repo=Path(__file__).resolve().parents[3]
        write_json(out/'ORIGINAL_RUN_BEFORE.json',originals)
        audit={'scope':'PROVENANCE_SAVED_ARRAY_AND_ARCHIVE_VERIFICATION_NOT_NEW_384_NUMERICAL_REPRODUCTION',
            'training':training_audit(run,p,gate),'metrics':metric_audit(run,p),'distribution':distribution_audit(run),
            'timing':timing_audit(run,p),'representation_diagnostics':diagnostic_audit(run)}
        metadata=read(data/'DATASET.json')
        with np.load(data/'radial_train.npz',allow_pickle=False) as z:
            require(z['logr'].shape==(16,2050) and z['uids'].tolist()==[r['uid'] for r in metadata['rows']], 'Training parent data')
        for row in metadata['rows']:
            require(file_hash(data/'raw'/f'{row["uid"]}.npz')==row['raw_sha256'],'Raw parent hash')
            require(file_hash(data/'meshflow512/objaverse_occ_v5_ids'/f'{row["uid"]}.npz')==row['refined_sha256'],'Refined parent hash')
        audit['data']={'parents':16,'raw_and_refined_hashes_verified':True,'files':files_manifest(data)}
        audit['tests_reused']={'metric_tests':{'passed':8,'seconds':2.27},'timing_tests':{'passed':8,'seconds':2.54},
            'evidence_type':'prior session tool execution receipts; no original machine XML exists',
            'not_rerun_for_packaging':True,'test_sources_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in ('test_ours_metrics.py','test_ours_timing.py')}}
        audit['new_GPU_execution']=False;audit['source_run_modified']=False
        write_json(out/'AUDIT.json',audit)
        payload=out/'payload';payload.mkdir()
        shutil.copytree(run,payload/'run');shutil.copytree(data,payload/'data');shutil.copytree(gate,payload/'evidence/original_AE_FM_GPU_gate')
        sources=copied_sources(repo);source_records={}
        for name,path in sources.items():
            destination=payload/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            source_records[name]={'sha256':file_hash(path),'bytes':path.stat().st_size}
        shutil.copyfile(out/'AUDIT.json',payload/'AUDIT.json')
        write_json(payload/'SOURCE_WHITELIST.json',{'files':source_records,'snapshot_policy':'explicit names only; no parallel fm_tail/system_formal/learned_outlets/longfit/balanced modules',
            'official_helpers_import_only_not_official_scientific_weights':True})
        write_json(payload/'PORTABLE_PATHS.json',{'original_run':str(run),'run':'run','original_data':str(data),'data':'data',
            'original_gate':str(gate),'gate':'evidence/original_AE_FM_GPU_gate','original_repository':str(repo),'source':'source',
            'original_JSON_absolute_paths_preserved':True,'do_not_rewrite_original_protocols_to_claim_same_hash':True})
        versions={}
        for name in ('torch','numpy','scipy','trimesh','numba','vtk','rtree','matplotlib','pytest'):
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]=None
        write_json(payload/'AUDIT_ENVIRONMENT.json',{'python':sys.version,'libraries':versions,'audit_device':'cpu','cuda_initialized':torch.cuda.is_initialized(),
            'original_cuda_environment_evidence':'run/timing/cold/*/*/*/SETUP.json'})
        (payload/'README_zh.md').write_text('原 AE3200/FM8000 双种子封闭16对象实验封存。\n\nrun 保留全部原始文件、全部中间权重和失败输出；data 保存实际输入；source 为显式依赖白名单。evidence 内的4步 GPU 门禁权重只证明恢复正确性，绝不可当作正式训练结果。\n\n本次只做身份、保存数组、既有 exact 账本、RMS 参数与代数、完整分母、分布矩阵统计及归档验证；没有重新运行384份完整几何误差，也没有新 GPU 推理。此前120次 GPU 计时重放及本次保存数组逐位读回证据保留。CPU32 与 GPU32 本来存在舍入差异，未声称 CPU 重新生成逐位一致。\n\n所有历史 JSON 绝对路径不改，重定位对照见 PORTABLE_PATHS.json。FILE_MANIFEST.json 绑定所有其余文件。可从source运行 ours_package verify --payload <解压目录> 做全量 SHA256/尺寸与保存主场核验；完整GPU重放须另行授权与环境门禁。\n\n128主场、384输出中，5份QEM几何失败完整保留；冷/热计时仅8个固定样本、24案例，120记录含24预热；不能将重复计时当独立样本或声称外部基线胜出。\n',encoding='utf-8')
        require(not torch.cuda.is_initialized(),'Unexpected CUDA context')
        verify_files(run,originals)
        manifest=files_manifest(payload);write_json(payload/'FILE_MANIFEST.json',{'schema':'AEFM8K-ARCHIVE-v1','files':manifest,'self_excluded':True})
        archive=out/'AEFM8K_CLOSED16_FULL.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for path in sorted(payload.rglob('*')):
                if path.is_file():z.write(path,str(path.relative_to(payload)).replace('\\','/'))
        fresh=extract_and_verify(archive,out/'fresh_extract')
        # Fresh-path source import proves no hidden parallel-research module is needed.
        command=[sys.executable,'-B','-c','import experiments.a_radial.system_probe.ours_package; import experiments.a_radial.system_probe.ours_timing; print("FRESH_SOURCE_IMPORT_PASS")']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(out/'fresh_extract/source'))
        completed=subprocess.run(command,cwd=out/'fresh_extract',env=env,capture_output=True,text=True)
        require(completed.returncode==0,'Fresh source dependency import failed: '+completed.stderr)
        # Imports must not mutate the sealed/extracted file sets.
        verify_archive_payload(out/'fresh_extract');verify_files(run,originals)
        final={'status':'SEALED','archive':archive.name,'archive_sha256':file_hash(archive),'archive_bytes':archive.stat().st_size,
            'archive_CRC_all_members_pass':True,'fresh_extraction':fresh,'fresh_source_import':completed.stdout.strip(),
            'source_run_files':len(originals),'source_run_unchanged':True,'all_original_files_preserved':True,
            'prior_gpu_exact_replay_count':120,'new_GPU_replay':False,'new_384_numerical_reproduction':False,
            'elapsed_seconds':time.perf_counter()-started,'file_manifest_sha256':file_hash(payload/'FILE_MANIFEST.json')}
        write_json(out/'SEALED.json',final);print(json.dumps(final),flush=True);return final
    except BaseException as exc:
        write_json(out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),'source_run_never_modified':True,
            'partial_files_retained':True,'elapsed_seconds':time.perf_counter()-started});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('package');a.add_argument('--run',required=True);a.add_argument('--gate',required=True);a.add_argument('--out',required=True)
    v=sub.add_parser('verify');v.add_argument('--payload',required=True)
    args=p.parse_args()
    if args.command=='package':package(args.run,args.gate,args.out)
    else:print(json.dumps(verify_archive_payload(args.payload)),flush=True)


if __name__=='__main__':main()
